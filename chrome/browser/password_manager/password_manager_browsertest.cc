// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <string>
#include <utility>

#include "base/command_line.h"
#include "base/macros.h"
#include "base/memory/ptr_util.h"
#include "base/metrics/histogram_samples.h"
#include "base/metrics/statistics_recorder.h"
#include "base/path_service.h"
#include "base/run_loop.h"
#include "base/stl_util.h"
#include "base/strings/stringprintf.h"
#include "base/strings/utf_string_conversions.h"
#include "base/test/metrics/histogram_tester.h"
#include "base/test/scoped_feature_list.h"
#include "base/threading/thread_restrictions.h"
#include "build/build_config.h"
#include "chrome/browser/chrome_notification_types.h"
#include "chrome/browser/metrics/subprocess_metrics_provider.h"
#include "chrome/browser/password_manager/chrome_password_manager_client.h"
#include "chrome/browser/password_manager/password_manager_test_base.h"
#include "chrome/browser/password_manager/password_store_factory.h"
#include "chrome/browser/profiles/profile.h"
#include "chrome/browser/ui/browser.h"
#include "chrome/browser/ui/browser_navigator_params.h"
#include "chrome/browser/ui/login/login_handler.h"
#include "chrome/browser/ui/login/login_handler_test_utils.h"
#include "chrome/browser/ui/passwords/manage_passwords_ui_controller.h"
#include "chrome/browser/ui/tabs/tab_strip_model.h"
#include "chrome/browser/ui/test/test_browser_dialog.h"
#include "chrome/common/chrome_paths.h"
#include "chrome/common/chrome_switches.h"
#include "chrome/test/base/ui_test_utils.h"
#include "components/autofill/core/browser/autofill_experiments.h"
#include "components/autofill/core/browser/autofill_test_utils.h"
#include "components/autofill/core/browser/test_autofill_client.h"
#include "components/autofill/core/common/password_form.h"
#include "components/password_manager/content/browser/content_password_manager_driver.h"
#include "components/password_manager/content/browser/content_password_manager_driver_factory.h"
#include "components/password_manager/core/browser/login_model.h"
#include "components/password_manager/core/browser/test_password_store.h"
#include "components/password_manager/core/common/password_manager_features.h"
#include "components/version_info/version_info.h"
#include "content/public/browser/browser_thread.h"
#include "content/public/browser/navigation_controller.h"
#include "content/public/browser/notification_service.h"
#include "content/public/browser/render_frame_host.h"
#include "content/public/browser/render_process_host.h"
#include "content/public/browser/render_view_host.h"
#include "content/public/browser/render_widget_host_view.h"
#include "content/public/browser/web_contents.h"
#include "content/public/browser/web_contents_observer.h"
#include "content/public/common/content_switches.h"
#include "content/public/test/browser_test_utils.h"
#include "net/base/filename_util.h"
#include "net/test/embedded_test_server/http_request.h"
#include "net/test/embedded_test_server/http_response.h"
#include "net/url_request/test_url_fetcher_factory.h"
#include "testing/gmock/include/gmock/gmock.h"
#include "third_party/blink/public/platform/web_input_event.h"
#include "ui/base/ui_base_switches.h"
#include "ui/events/keycodes/keyboard_codes.h"
#include "ui/gfx/geometry/point.h"

using testing::_;
using testing::ElementsAre;

namespace {

// Test params:
//  - bool popup_views_enabled: whether feature AutofillExpandedPopupViews
//        is enabled for testing.
class PasswordManagerBrowserTestWithViewsFeature
    : public PasswordManagerBrowserTestBase,
      public ::testing::WithParamInterface<bool> {
 public:
  PasswordManagerBrowserTestWithViewsFeature() = default;
  ~PasswordManagerBrowserTestWithViewsFeature() override = default;

  void SetUpCommandLine(base::CommandLine* command_line) override {
    PasswordManagerBrowserTestBase::SetUpCommandLine(command_line);

    const bool popup_views_enabled = GetParam();
    scoped_feature_list_.InitWithFeatureState(
        autofill::kAutofillExpandedPopupViews, popup_views_enabled);
  }

 private:
  base::test::ScopedFeatureList scoped_feature_list_;
};

class MockLoginModelObserver : public password_manager::LoginModelObserver {
 public:
  MOCK_METHOD2(OnAutofillDataAvailableInternal,
               void(const base::string16&, const base::string16&));

 private:
  void OnLoginModelDestroying() override {}
};

GURL GetFileURL(const char* filename) {
  base::ScopedAllowBlockingForTesting allow_blocking;
  base::FilePath path;
  base::PathService::Get(chrome::DIR_TEST_DATA, &path);
  path = path.AppendASCII("password").AppendASCII(filename);
  CHECK(base::PathExists(path));
  return net::FilePathToFileURL(path);
}

// Handles |request| to "/basic_auth". If "Authorization" header is present,
// responds with a non-empty HTTP 200 page (regardless of its value). Otherwise
// serves a Basic Auth challenge.
std::unique_ptr<net::test_server::HttpResponse> HandleTestAuthRequest(
    const net::test_server::HttpRequest& request) {
  if (!base::StartsWith(request.relative_url, "/basic_auth",
                        base::CompareCase::SENSITIVE))
    return std::unique_ptr<net::test_server::HttpResponse>();

  if (base::ContainsKey(request.headers, "Authorization")) {
    std::unique_ptr<net::test_server::BasicHttpResponse> http_response(
        new net::test_server::BasicHttpResponse);
    http_response->set_code(net::HTTP_OK);
    http_response->set_content("Success!");
    return std::move(http_response);
  } else {
    std::unique_ptr<net::test_server::BasicHttpResponse> http_response(
        new net::test_server::BasicHttpResponse);
    http_response->set_code(net::HTTP_UNAUTHORIZED);
    http_response->AddCustomHeader("WWW-Authenticate",
                                   "Basic realm=\"test realm\"");
    return std::move(http_response);
  }
}

class ObservingAutofillClient
    : public autofill::TestAutofillClient,
      public content::WebContentsUserData<ObservingAutofillClient> {
 public:
  ~ObservingAutofillClient() override {}

  // Wait until the autofill popup is shown.
  void WaitForAutofillPopup() {
    base::RunLoop run_loop;
    run_loop_ = &run_loop;
    run_loop.Run();
    DCHECK(!run_loop_);
  }

  bool popup_shown() const { return popup_shown_; }

  void ShowAutofillPopup(
      const gfx::RectF& element_bounds,
      base::i18n::TextDirection text_direction,
      const std::vector<autofill::Suggestion>& suggestions,
      bool autoselect_first_suggestion,
      base::WeakPtr<autofill::AutofillPopupDelegate> delegate) override {
    if (run_loop_)
      run_loop_->Quit();
    run_loop_ = nullptr;
    popup_shown_ = true;
  }

 private:
  explicit ObservingAutofillClient(content::WebContents* web_contents)
      : run_loop_(nullptr), popup_shown_(false) {}
  friend class content::WebContentsUserData<ObservingAutofillClient>;

  base::RunLoop* run_loop_;
  bool popup_shown_;

  DISALLOW_COPY_AND_ASSIGN(ObservingAutofillClient);
};

void TestPromptNotShown(const char* failure_message,
                        content::WebContents* web_contents) {
  SCOPED_TRACE(testing::Message(failure_message));

  NavigationObserver observer(web_contents);
  std::string fill_and_submit =
      "document.getElementById('username_failed').value = 'temp';"
      "document.getElementById('password_failed').value = 'random';"
      "document.getElementById('failed_form').submit()";

  ASSERT_TRUE(content::ExecuteScript(web_contents, fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(BubbleObserver(web_contents).IsSavePromptShownAutomatically());
}

// Generate HTML for a simple password form with the specified action URL.
std::string GeneratePasswordFormForAction(const GURL& action_url) {
  return "<form method='POST' action='" + action_url.spec() + "'"
         "      onsubmit='return true;' id='testform'>"
         "  <input type='password' id='password_field'>"
         "</form>";
}

// Inject an about:blank frame with a password form that uses the specified
// action URL into |web_contents|.
void InjectBlankFrameWithPasswordForm(content::WebContents* web_contents,
                                      const GURL& action_url) {
  std::string form_html = GeneratePasswordFormForAction(action_url);
  std::string inject_blank_frame_with_password_form =
      "var frame = document.createElement('iframe');"
      "frame.id = 'iframe';"
      "document.body.appendChild(frame);"
      "frame.contentDocument.body.innerHTML = \"" + form_html + "\"";
  ASSERT_TRUE(content::ExecuteScript(web_contents,
                                     inject_blank_frame_with_password_form));
}

// Fills in a fake password and submits the form in |frame|, waiting for the
// submit navigation to finish.  |action_url| is the form action URL to wait
// for.
void SubmitInjectedPasswordForm(content::WebContents* web_contents,
                                content::RenderFrameHost* frame,
                                const GURL& action_url) {
  std::string submit_form =
      "document.getElementById('password_field').value = 'pa55w0rd';"
      "document.getElementById('testform').submit();";
  NavigationObserver observer(web_contents);
  observer.SetPathToWaitFor(action_url.path());
  ASSERT_TRUE(content::ExecuteScript(frame, submit_form));
  observer.Wait();
}

}  // namespace

DEFINE_WEB_CONTENTS_USER_DATA_KEY(ObservingAutofillClient);

namespace password_manager {

// Actual tests ---------------------------------------------------------------

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForNormalSubmit) {
  NavigateToFile("/password/password_form.html");

  // Fill a form and submit through a <input type="submit"> button. Nothing
  // special.
  NavigationObserver observer(WebContents());
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();

  // Save the password and check the store.
  BubbleObserver bubble_observer(WebContents());
  EXPECT_TRUE(bubble_observer.IsSavePromptShownAutomatically());
  bubble_observer.AcceptSavePrompt();
  WaitForPasswordStore();

  CheckThatCredentialsStored("temp", "random");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptIfFormReappeared) {
  NavigateToFile("/password/failed.html");
  TestPromptNotShown("normal form", WebContents());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptIfFormReappearedWithPartsHidden) {
  NavigateToFile("/password/failed_partly_visible.html");
  TestPromptNotShown("partly visible form", WebContents());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptIfFormReappearedInputOutsideFor) {
  NavigateToFile("/password/failed_input_outside.html");
  TestPromptNotShown("form with input outside", WebContents());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptIfPasswordFormManagerDestroyed) {
  NavigateToFile("/password/password_form.html");
  // Simulate the Credential Manager API essentially destroying all the
  // PasswordFormManager instances.
  ChromePasswordManagerClient::FromWebContents(WebContents())
      ->NotifyStorePasswordCalled();

  // Fill a form and submit through a <input type="submit"> button. The renderer
  // should not send "PasswordFormsParsed" messages after the page was loaded.
  NavigationObserver observer(WebContents());
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForSubmitWithSameDocumentNavigation) {
  NavigateToFile("/password/password_navigate_before_submit.html");

  // Fill a form and submit through a <input type="submit"> button. Nothing
  // special. The form does an in-page navigation before submitting.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       LoginSuccessWithUnrelatedForm) {
  // Log in, see a form on the landing page. That form is not related to the
  // login form (=has different input fields), so we should offer saving the
  // password.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_unrelated').value = 'temp';"
      "document.getElementById('password_unrelated').value = 'random';"
      "document.getElementById('submit_unrelated').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       LoginFailed) {
  // Log in, see a form on the landing page. That form is not related to the
  // login form (=has a different action), so we should offer saving the
  // password.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_failed').value = 'temp';"
      "document.getElementById('password_failed').value = 'random';"
      "document.getElementById('submit_failed').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature, Redirects) {
  NavigateToFile("/password/password_form.html");

  // Fill a form and submit through a <input type="submit"> button. The form
  // points to a redirection page.
  NavigationObserver observer1(WebContents());
  std::string fill_and_submit =
      "document.getElementById('username_redirect').value = 'temp';"
      "document.getElementById('password_redirect').value = 'random';"
      "document.getElementById('submit_redirect').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer1.Wait();
  BubbleObserver bubble_observer(WebContents());
  EXPECT_TRUE(bubble_observer.IsSavePromptShownAutomatically());

  // The redirection page now redirects via Javascript. We check that the
  // bubble stays.
  NavigationObserver observer2(WebContents());
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(), "window.location.href = 'done.html';"));
  observer2.Wait();
  EXPECT_TRUE(bubble_observer.IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForSubmitUsingJavaScript) {
  NavigateToFile("/password/password_form.html");

  // Fill a form and submit using <button> that calls submit() on the form.
  // This should work regardless of the type of element, as long as submit() is
  // called.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForDynamicForm) {
  // Adding a PSL matching form is a workaround explained later.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  GURL psl_orogin = embedded_test_server()->GetURL("psl.example.com", "/");
  signin_form.signon_realm = psl_orogin.spec();
  signin_form.origin = psl_orogin;
  signin_form.username_value = base::ASCIIToUTF16("unused_username");
  signin_form.password_value = base::ASCIIToUTF16("unused_password");
  password_store->AddLogin(signin_form);

  // Show the dynamic form.
  ui_test_utils::NavigateToURL(
      browser(), embedded_test_server()->GetURL(
                     "example.com", "/password/dynamic_password_form.html"));
  ASSERT_TRUE(content::ExecuteScript(
      WebContents(), "document.getElementById('create_form_button').click();"));

  // Blink has a timer for 0.3 seconds before it updates the browser with the
  // new dynamic form. We wait for the form being detected by observing the UI
  // state. The state changes due to the matching credential saved above. Later
  // the form submission is definitely noticed by the browser.
  BubbleObserver(WebContents()).WaitForManagementState();

  // Fill the dynamic password form and submit.
  NavigationObserver observer(WebContents());
  std::string fill_and_submit =
      "document.dynamic_form.username.value = 'tempro';"
      "document.dynamic_form.password.value = 'random';"
      "document.dynamic_form.submit()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();

  EXPECT_TRUE(BubbleObserver(WebContents()).IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForNavigation) {
  NavigateToFile("/password/password_form.html");

  // Don't fill the password form, just navigate away. Shouldn't prompt.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(), "window.location.href = 'done.html';"));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForSubFrameNavigation) {
  NavigateToFile("/password/multi_frames.html");

  // If you are filling out a password form in one frame and a different frame
  // navigates, this should not trigger the infobar.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  observer.SetPathToWaitFor("/password/done.html");
  std::string fill =
      "var first_frame = document.getElementById('first_frame');"
      "var frame_doc = first_frame.contentDocument;"
      "frame_doc.getElementById('username_field').value = 'temp';"
      "frame_doc.getElementById('password_field').value = 'random';";
  std::string navigate_frame =
      "var second_iframe = document.getElementById('second_frame');"
      "second_iframe.contentWindow.location.href = 'done.html';";

  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill));
  ASSERT_TRUE(content::ExecuteScript(WebContents(), navigate_frame));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForSameFormWithDifferentAction) {
  // Log in, see a form on the landing page. That form is related to the login
  // form (has a different action but has same input fields), so we should not
  // offer saving the password.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  auto prompt_observer = std::make_unique<BubbleObserver>(WebContents());
  std::string fill_and_submit =
      "document.getElementById('username_different_action').value = 'temp';"
      "document.getElementById('password_different_action').value = 'random';"
      "document.getElementById('submit_different_action').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForActionMutation) {
  NavigateToFile("/password/password_form_action_mutation.html");

  // Need to pay attention for a message that XHR has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;
  BubbleObserver prompt_observer(WebContents());
  std::string fill_and_submit =
      "document.getElementById('username_action_mutation').value = 'temp';"
      "document.getElementById('password_action_mutation').value = 'random';"
      "document.getElementById('submit_action_mutation').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"XHR_FINISHED\"")
      break;
  }
  EXPECT_FALSE(prompt_observer.IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForFormWithEnteredUsername) {
  // Log in, see a form on the landing page. That form is not related to the
  // login form but has the same username as was entered previously, so we
  // should not offer saving the password.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  auto prompt_observer = std::make_unique<BubbleObserver>(WebContents());
  std::string fill_and_submit =
      "document.getElementById('username_contains_username').value = 'temp';"
      "document.getElementById('password_contains_username').value = 'random';"
      "document.getElementById('submit_contains_username').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForDifferentFormWithEmptyAction) {
  // Log in, see a form on the landing page. That form is not related to the
  // signin form. The signin and the form on the landing page have empty
  // actions, so we should offer saving the password.
  NavigateToFile("/password/navigate_to_same_url_empty_actions.html");

  NavigationObserver observer(WebContents());
  auto prompt_observer = std::make_unique<BubbleObserver>(WebContents());
  std::string fill_and_submit =
      "document.getElementById('username').value = 'temp';"
      "document.getElementById('password').value = 'random';"
      "document.getElementById('submit-button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptAfterSubmitWithSubFrameNavigation) {
  NavigateToFile("/password/multi_frames.html");

  // Make sure that we prompt to save password even if a sub-frame navigation
  // happens first.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  observer.SetPathToWaitFor("/password/done.html");
  std::string navigate_frame =
      "var second_iframe = document.getElementById('second_frame');"
      "second_iframe.contentWindow.location.href = 'other.html';";
  std::string fill_and_submit =
      "var first_frame = document.getElementById('first_frame');"
      "var frame_doc = first_frame.contentDocument;"
      "frame_doc.getElementById('username_field').value = 'temp';"
      "frame_doc.getElementById('password_field').value = 'random';"
      "frame_doc.getElementById('input_submit_button').click();";

  ASSERT_TRUE(content::ExecuteScript(WebContents(), navigate_frame));
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptAfterSubmitFromSubFrameToMainFrame) {
  NavigateToFile("/password/password_form_in_crosssite_iframe.html");
  // Create an iframe and navigate cross-site.
  NavigationObserver iframe_observer(WebContents());
  iframe_observer.SetPathToWaitFor("/password/password_form_with_hash.html");
  GURL iframe_url =
      embedded_test_server()->GetURL("/password/password_form_with_hash.html");
  std::string create_iframe =
      base::StringPrintf("create_iframe('%s');", iframe_url.spec().c_str());
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       create_iframe));
  iframe_observer.Wait();

  // Make sure that we prompt to save the password even if the sub-frame submits
  // the form to the main frame.
  NavigationObserver observer(WebContents());
  std::string fill_and_submit =
      "var first_frame = document.getElementById('iframe');"
      "var frame_doc = first_frame.contentDocument;"
      "frame_doc.getElementById('username_field').value = 'Admin';"
      "frame_doc.getElementById('password_field').value = '12345';"
      "frame_doc.getElementById('input_submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();

  BubbleObserver prompt_observer(WebContents());
  EXPECT_TRUE(prompt_observer.IsSavePromptShownAutomatically());
  prompt_observer.AcceptSavePrompt();
  WaitForPasswordStore();

  // The typed credentials are saved despite the password is hashed before
  // submission.
  CheckThatCredentialsStored("Admin", "12345");
}

IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForFailedLoginFromMainFrameWithMultiFramesSameDocument) {
  NavigateToFile("/password/multi_frames.html");

  // Make sure that we don't prompt to save the password for a failed login
  // from the main frame with multiple frames in the same page.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_failed').value = 'temp';"
      "document.getElementById('password_failed').value = 'random';"
      "document.getElementById('submit_failed').click();";

  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForFailedLoginFromSubFrameWithMultiFramesSameDocument) {
  NavigateToFile("/password/multi_frames.html");

  // Make sure that we don't prompt to save the password for a failed login
  // from a sub-frame with multiple frames in the same page.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "var first_frame = document.getElementById('first_frame');"
      "var frame_doc = first_frame.contentDocument;"
      "frame_doc.getElementById('username_failed').value = 'temp';"
      "frame_doc.getElementById('password_failed').value = 'random';"
      "frame_doc.getElementById('submit_failed').click();";

  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.SetPathToWaitFor("/password/failed.html");
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForXHRSubmit) {
  NavigateToFile("/password/password_xhr_submit.html");

  // Verify that we show the save password prompt if a form returns false
  // in its onsubmit handler but instead logs in/navigates via XHR.
  // Note that calling 'submit()' on a form with javascript doesn't call
  // the onsubmit handler, so we click the submit button instead.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForXHRSubmitWithoutNavigation) {
  NavigateToFile("/password/password_xhr_submit.html");

  // Need to pay attention for a message that XHR has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if XHR without navigation occurs and the form has been filled
  // out we try and save the password. Note that in general the submission
  // doesn't need to be via form.submit(), but for testing purposes it's
  // necessary since we otherwise ignore changes made to the value of these
  // fields by script.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"XHR_FINISHED\"")
      break;
  }

  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForXHRSubmitWithoutNavigation_SignupForm) {
  NavigateToFile("/password/password_xhr_submit.html");

  // Need to pay attention for a message that XHR has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if XHR without navigation occurs and the form has been filled
  // out we try and save the password. Note that in general the submission
  // doesn't need to be via form.submit(), but for testing purposes it's
  // necessary since we otherwise ignore changes made to the value of these
  // fields by script.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('signup_username_field').value = 'temp';"
      "document.getElementById('signup_password_field').value = 'random';"
      "document.getElementById('confirmation_password_field').value = 'random';"
      "document.getElementById('signup_submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"XHR_FINISHED\"")
      break;
  }

  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForXHRSubmitWithoutNavigationWithUnfilledForm) {
  NavigateToFile("/password/password_xhr_submit.html");

  // Need to pay attention for a message that XHR has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if XHR without navigation occurs and the form has NOT been
  // filled out we don't prompt.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"XHR_FINISHED\"")
      break;
  }

  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForXHRSubmitWithoutNavigationWithUnfilledForm_SignupForm) {
  NavigateToFile("/password/password_xhr_submit.html");

  // Need to pay attention for a message that XHR has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if XHR without navigation occurs and the form has NOT been
  // filled out we don't prompt.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('signup_username_field').value = 'temp';"
      "document.getElementById('signup_submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"XHR_FINISHED\"")
      break;
  }

  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForFetchSubmit) {
  NavigateToFile("/password/password_fetch_submit.html");

  // Verify that we show the save password prompt if a form returns false
  // in its onsubmit handler but instead logs in/navigates via Fetch.
  // Note that calling 'submit()' on a form with javascript doesn't call
  // the onsubmit handler, so we click the submit button instead.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForFetchSubmitWithoutNavigation) {
  NavigateToFile("/password/password_fetch_submit.html");

  // Need to pay attention for a message that XHR has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if XHR without navigation occurs and the form has been filled
  // out we try and save the password. Note that in general the submission
  // doesn't need to be via form.submit(), but for testing purposes it's
  // necessary since we otherwise ignore changes made to the value of these
  // fields by script.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"FETCH_FINISHED\"")
      break;
  }

  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForFetchSubmitWithoutNavigation_SignupForm) {
  NavigateToFile("/password/password_fetch_submit.html");

  // Need to pay attention for a message that Fetch has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if Fetch without navigation occurs and the form has been filled
  // out we try and save the password. Note that in general the submission
  // doesn't need to be via form.submit(), but for testing purposes it's
  // necessary since we otherwise ignore changes made to the value of these
  // fields by script.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('signup_username_field').value = 'temp';"
      "document.getElementById('signup_password_field').value = 'random';"
      "document.getElementById('confirmation_password_field').value = 'random';"
      "document.getElementById('signup_submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"FETCH_FINISHED\"")
      break;
  }

  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForFetchSubmitWithoutNavigationWithUnfilledForm) {
  NavigateToFile("/password/password_fetch_submit.html");

  // Need to pay attention for a message that Fetch has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if Fetch without navigation occurs and the form has NOT been
  // filled out we don't prompt.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"FETCH_FINISHED\"")
      break;
  }

  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForFetchSubmitWithoutNavigationWithUnfilledForm_SignupForm) {
  NavigateToFile("/password/password_fetch_submit.html");

  // Need to pay attention for a message that Fetch has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  // Verify that if Fetch without navigation occurs and the form has NOT been
  // filled out we don't prompt.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "navigate = false;"
      "document.getElementById('signup_username_field').value = 'temp';"
      "document.getElementById('signup_submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"FETCH_FINISHED\"")
      break;
  }

  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptIfLinkClicked) {
  NavigateToFile("/password/password_form.html");

  // Verify that if the user takes a direct action to leave the page, we don't
  // prompt to save the password even if the form is already filled out.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_click_link =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('link').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_click_link));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       VerifyPasswordGenerationUpload) {
  // Prevent Autofill requests from actually going over the wire.
  net::TestURLFetcherFactory factory;
  // Disable Autofill requesting access to AddressBook data. This causes
  // the test to hang on Mac.
  autofill::test::DisableSystemServices(browser()->profile()->GetPrefs());

  // Visit a signup form.
  NavigateToFile("/password/signup_form.html");

  // Enter a password and save it.
  NavigationObserver first_observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('other_info').value = 'stuff';"
      "document.getElementById('username_field').value = 'my_username';"
      "document.getElementById('password_field').value = 'password';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));

  first_observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
  prompt_observer->AcceptSavePrompt();

  // Now navigate to a login form that has similar HTML markup.
  NavigateToFile("/password/password_form.html");

  // Simulate a user click to force an autofill of the form's DOM value, not
  // just the suggested value.
  content::SimulateMouseClick(WebContents(), 0,
                              blink::WebMouseEvent::Button::kLeft);
  WaitForElementValue("username_field", "my_username");
  WaitForElementValue("password_field", "password");

  // Submit the form and verify that there is no infobar (as the password
  // has already been saved).
  NavigationObserver second_observer(WebContents());
  std::unique_ptr<BubbleObserver> second_prompt_observer(
      new BubbleObserver(WebContents()));
  std::string submit_form =
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), submit_form));
  second_observer.Wait();
  EXPECT_FALSE(second_prompt_observer->IsSavePromptShownAutomatically());

  // Verify that we sent two pings to Autofill. One vote for of PASSWORD for
  // the current form, and one vote for ACCOUNT_CREATION_PASSWORD on the
  // original form since it has more than 2 text input fields and was used for
  // the first time on a different form.
  base::HistogramBase* upload_histogram =
      base::StatisticsRecorder::FindHistogram(
          "PasswordGeneration.UploadStarted");
  ASSERT_TRUE(upload_histogram);
  std::unique_ptr<base::HistogramSamples> snapshot =
      upload_histogram->SnapshotSamples();
  EXPECT_EQ(0, snapshot->GetCount(0 /* failure */));
  EXPECT_EQ(2, snapshot->GetCount(1 /* success */));

  autofill::test::ReenableSystemServices();
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForSubmitFromIframe) {
  NavigateToFile("/password/password_submit_from_iframe.html");

  // Submit a form in an iframe, then cause the whole page to navigate without a
  // user gesture. We expect the save password prompt to be shown here, because
  // some pages use such iframes for login forms.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "var iframe = document.getElementById('test_iframe');"
      "var iframe_doc = iframe.contentDocument;"
      "iframe_doc.getElementById('username_field').value = 'temp';"
      "iframe_doc.getElementById('password_field').value = 'random';"
      "iframe_doc.getElementById('submit_button').click()";

  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForInputElementWithoutName) {
  // Check that the prompt is shown for forms where input elements lack the
  // "name" attribute but the "id" is present.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field_no_name').value = 'temp';"
      "document.getElementById('password_field_no_name').value = 'random';"
      "document.getElementById('input_submit_button_no_name').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForInputElementWithoutId) {
  // Check that the prompt is shown for forms where input elements lack the
  // "id" attribute but the "name" attribute is present.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementsByName('username_field_no_id')[0].value = 'temp';"
      "document.getElementsByName('password_field_no_id')[0].value = 'random';"
      "document.getElementsByName('input_submit_button_no_id')[0].click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForInputElementWithoutIdAndName) {
  // Check that prompt is shown for forms where the input fields lack both
  // the "id" and the "name" attributes.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "var form = document.getElementById('testform_elements_no_id_no_name');"
      "var username = form.children[0];"
      "username.value = 'temp';"
      "var password = form.children[1];"
      "password.value = 'random';"
      "form.children[2].click()";  // form.children[2] is the submit button.
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
  prompt_observer->AcceptSavePrompt();

  // Check that credentials are stored.
  WaitForPasswordStore();
  CheckThatCredentialsStored("temp", "random");
}

// Test for checking that no prompt is shown for URLs with file: scheme.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForFileSchemeURLs) {
  GURL url = GetFileURL("password_form.html");
  ui_test_utils::NavigateToURL(browser(), url);

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForLandingPageWithHTTPErrorStatusCode) {
  // Check that no prompt is shown for forms where the landing page has
  // HTTP status 404.
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field_http_error').value = 'temp';"
      "document.getElementById('password_field_http_error').value = 'random';"
      "document.getElementById('input_submit_button_http_error').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       DeleteFrameBeforeSubmit) {
  NavigateToFile("/password/multi_frames.html");

  NavigationObserver observer(WebContents());
  // Make sure we save some password info from an iframe and then destroy it.
  std::string save_and_remove =
      "var first_frame = document.getElementById('first_frame');"
      "var frame_doc = first_frame.contentDocument;"
      "frame_doc.getElementById('username_field').value = 'temp';"
      "frame_doc.getElementById('password_field').value = 'random';"
      "frame_doc.getElementById('input_submit_button').click();"
      "first_frame.parentNode.removeChild(first_frame);";
  // Submit from the main frame, but without navigating through the onsubmit
  // handler.
  std::string navigate_frame =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click();"
      "window.location.href = 'done.html';";

  ASSERT_TRUE(content::ExecuteScript(WebContents(), save_and_remove));
  ASSERT_TRUE(content::ExecuteScript(WebContents(), navigate_frame));
  observer.Wait();
  // The only thing we check here is that there is no use-after-free reported.
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       UsernameAndPasswordValueAccessible) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("admin");
  signin_form.password_value = base::ASCIIToUTF16("12345");
  password_store->AddLogin(signin_form);

  // Steps from https://crbug.com/337429#c37.
  // Navigate to the page, click a link that opens a second tab, reload the
  // first tab and observe that the password is accessible.
  NavigateToFile("/password/form_and_link.html");

  // Click on a link to open a new tab, then switch back to the first one.
  EXPECT_EQ(1, browser()->tab_strip_model()->count());
  std::string click =
      "document.getElementById('testlink').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), click));
  EXPECT_EQ(2, browser()->tab_strip_model()->count());
  browser()->tab_strip_model()->ActivateTabAt(0, false);

  // Reload the original page to have the saved credentials autofilled.
  NavigationObserver reload_observer(WebContents());
  NavigateToFile("/password/form_and_link.html");
  reload_observer.Wait();

  // Now check that the username and the password are not accessible yet.
  CheckElementValue("username_field", "");
  CheckElementValue("password_field", "");
  // Let the user interact with the page.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));
  // Wait until that interaction causes the username and the password value to
  // be revealed.
  WaitForElementValue("username_field", "admin");
  WaitForElementValue("password_field", "12345");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PasswordValueAccessibleOnSubmit) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("admin");
  signin_form.password_value = base::ASCIIToUTF16("random_secret");
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/form_and_link.html");

  // Get the position of the 'submit' button.
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(),
      "var submitRect = document.getElementById('input_submit_button')"
      ".getBoundingClientRect();"));

  int top;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(), "window.domAutomationController.send(submitRect.top);",
      &top));
  int left;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send(submitRect.left);", &left));

  NavigationObserver submit_observer(WebContents());
  // Submit the form via a tap on the submit button.
  content::SimulateTapDownAt(WebContents(), gfx::Point(left + 1, top + 1));
  content::SimulateTapAt(WebContents(), gfx::Point(left + 1, top + 1));
  submit_observer.Wait();
  std::string query = WebContents()->GetURL().query();
  EXPECT_THAT(query, testing::HasSubstr("random_secret"));
}

// Test fix for crbug.com/338650.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       DontPromptForPasswordFormWithDefaultValue) {
  NavigateToFile("/password/password_form_with_default_value.html");

  // Don't prompt if we navigate away even if there is a password value since
  // it's not coming from the user.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  NavigateToFile("/password/done.html");
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       DontPromptForPasswordFormWithReadonlyPasswordField) {
  NavigateToFile("/password/password_form_with_password_readonly.html");

  // Fill a form and submit through a <input type="submit"> button. Nothing
  // special.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptWhenEnableAutomaticPasswordSavingSwitchIsNotSet) {
  NavigateToFile("/password/password_form.html");

  // Fill a form and submit through a <input type="submit"> button.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

// Test fix for crbug.com/368690.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptWhenReloading) {
  NavigateToFile("/password/password_form.html");

  std::string fill =
      "document.getElementById('username_redirect').value = 'temp';"
      "document.getElementById('password_redirect').value = 'random';";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill));

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  GURL url = embedded_test_server()->GetURL("/password/password_form.html");
  NavigateParams params(browser(), url, ::ui::PAGE_TRANSITION_RELOAD);
  ui_test_utils::NavigateToURL(&params);
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

// Test that if a form gets dynamically added between the form parsing and
// rendering, and while the main frame still loads, it still is registered, and
// thus saving passwords from it works.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       FormsAddedBetweenParsingAndRendering) {
  NavigateToFile("/password/between_parsing_and_rendering.html");

  NavigationObserver observer(WebContents());
  std::string submit =
      "document.getElementById('username').value = 'temp';"
      "document.getElementById('password').value = 'random';"
      "document.getElementById('submit-button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), submit));
  observer.Wait();

  EXPECT_TRUE(BubbleObserver(WebContents()).IsSavePromptShownAutomatically());
}

// Test that if a hidden form gets dynamically added between the form parsing
// and rendering, it still is registered, and autofilling works.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       HiddenFormAddedBetweenParsingAndRendering) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("admin");
  signin_form.password_value = base::ASCIIToUTF16("12345");
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/between_parsing_and_rendering.html?hidden");

  std::string show_form =
      "document.getElementsByTagName('form')[0].style.display = 'block'";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), show_form));

  // Wait until the username is filled, to make sure autofill kicked in.
  WaitForElementValue("username", "admin");
  WaitForElementValue("password", "12345");
}

// https://crbug.com/713645
// Navigate to a page that can't load some of the subresources. Create a hidden
// form when the body is loaded. Make the form visible. Chrome should autofill
// the form.
// The fact that the form is hidden isn't super important but reproduces the
// actual bug.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       SlowPageFill) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("admin");
  signin_form.password_value = base::ASCIIToUTF16("12345");
  password_store->AddLogin(signin_form);

  GURL url =
      embedded_test_server()->GetURL("/password/infinite_password_form.html");
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), url, WindowOpenDisposition::CURRENT_TAB,
      ui_test_utils::BROWSER_TEST_NONE);

  // Wait for autofill.
  BubbleObserver bubble_observer(WebContents());
  bubble_observer.WaitForManagementState();

  // Show the form and make sure that the password was autofilled.
  std::string show_form =
      "document.getElementsByTagName('form')[0].style.display = 'block'";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), show_form));

  CheckElementValue("username", "admin");
  CheckElementValue("password", "12345");
}

// Test that if there was no previous page load then the PasswordManagerDriver
// does not think that there were SSL errors on the current page. The test opens
// a new tab with a URL for which the embedded test server issues a basic auth
// challenge.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoLastLoadGoodLastLoad) {
  // We must use a new test server here because embedded_test_server() is
  // already started at this point and adding the request handler to it would
  // not be thread safe.
  net::EmbeddedTestServer http_test_server;

  // Teach the embedded server to handle requests by issuing the basic auth
  // challenge.
  http_test_server.RegisterRequestHandler(base::Bind(&HandleTestAuthRequest));
  ASSERT_TRUE(http_test_server.Start());

  LoginPromptBrowserTestObserver login_observer;
  // We need to register to all sources, because the navigation observer we are
  // interested in is for a new tab to be opened, and thus does not exist yet.
  login_observer.Register(content::NotificationService::AllSources());

  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS).get());
  ASSERT_TRUE(password_store->IsEmpty());

  // Navigate to a page requiring HTTP auth. Wait for the tab to get the correct
  // WebContents, but don't wait for navigation, which only finishes after
  // authentication.
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), http_test_server.GetURL("/basic_auth"),
      WindowOpenDisposition::CURRENT_TAB, ui_test_utils::BROWSER_TEST_NONE);

  content::NavigationController* nav_controller =
      &WebContents()->GetController();
  NavigationObserver nav_observer(WebContents());
  WindowedAuthNeededObserver auth_needed_observer(nav_controller);
  auth_needed_observer.Wait();

  WindowedAuthSuppliedObserver auth_supplied_observer(nav_controller);
  // Offer valid credentials on the auth challenge.
  ASSERT_EQ(1u, login_observer.handlers().size());
  LoginHandler* handler = *login_observer.handlers().begin();
  ASSERT_TRUE(handler);
  // Any username/password will work.
  handler->SetAuth(base::UTF8ToUTF16("user"), base::UTF8ToUTF16("pwd"));
  auth_supplied_observer.Wait();

  // The password manager should be working correctly.
  nav_observer.Wait();
  WaitForPasswordStore();
  BubbleObserver bubble_observer(WebContents());
  EXPECT_TRUE(bubble_observer.IsSavePromptShownAutomatically());
  bubble_observer.AcceptSavePrompt();

  // Spin the message loop to make sure the password store had a chance to save
  // the password.
  WaitForPasswordStore();
  EXPECT_FALSE(password_store->IsEmpty());
}

// Fill out a form and click a button. The Javascript removes the form, creates
// a similar one with another action, fills it out and submits. Chrome can
// manage to detect the new one and create a complete matching
// PasswordFormManager. Otherwise, the all-but-action matching PFM should be
// used. Regardless of the internals the user sees the bubble in 100% cases.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PreferPasswordFormManagerWhichFinishedMatching) {
  NavigateToFile("/password/create_form_copy_on_submit.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string submit =
      "document.getElementById('username').value = 'overwrite_me';"
      "document.getElementById('password').value = 'random';"
      "document.getElementById('non-form-button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), submit));
  observer.Wait();

  WaitForPasswordStore();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

// Test that if login fails and content server pushes a different login form
// with action URL having different schemes. Heuristic shall be able
// identify such cases and *shall not* prompt to save incorrect password.
IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForLoginFailedAndServerPushSeperateLoginForm_HttpToHttps) {
  std::string path =
      "/password/separate_login_form_with_onload_submit_script.html";
  GURL http_url(embedded_test_server()->GetURL(path));
  ASSERT_TRUE(http_url.SchemeIs(url::kHttpScheme));

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  ui_test_utils::NavigateToURL(browser(), http_url);

  observer.SetPathToWaitFor("/password/done_and_separate_login_form.html");
  observer.Wait();

  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForLoginFailedAndServerPushSeperateLoginForm_HttpsToHttp) {
  // This test case cannot inject the scripts via content::ExecuteScript() in
  // files served through HTTPS. Therefore the scripts are made part of the HTML
  // site and executed on load.
  std::string path =
      "/password/separate_login_form_with_onload_submit_script.html";
  GURL https_url(https_test_server().GetURL(path));
  ASSERT_TRUE(https_url.SchemeIs(url::kHttpsScheme));

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  ui_test_utils::NavigateToURL(browser(), https_url);

  observer.SetPathToWaitFor("/password/done_and_separate_login_form.html");
  observer.Wait();

  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

// Tests whether a attempted submission of a malicious credentials gets blocked.
// This simulates a case which is described in http://crbug.com/571580.
IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    NoPromptForSeperateLoginFormWhenSwitchingFromHttpsToHttp) {
  std::string path = "/password/password_form.html";
  GURL https_url(https_test_server().GetURL(path));
  ASSERT_TRUE(https_url.SchemeIs(url::kHttpsScheme));

  NavigationObserver form_observer(WebContents());
  ui_test_utils::NavigateToURL(browser(), https_url);
  form_observer.Wait();

  std::string fill_and_submit_redirect =
      "document.getElementById('username_redirect').value = 'user';"
      "document.getElementById('password_redirect').value = 'password';"
      "document.getElementById('submit_redirect').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit_redirect));

  NavigationObserver redirect_observer(WebContents());
  redirect_observer.SetPathToWaitFor("/password/redirect.html");
  redirect_observer.Wait();

  WaitForPasswordStore();
  BubbleObserver prompt_observer(WebContents());
  EXPECT_TRUE(prompt_observer.IsSavePromptShownAutomatically());

  // Normally the redirect happens to done.html. Here an attack is simulated
  // that hijacks the redirect to a attacker controlled page.
  GURL http_url(
      embedded_test_server()->GetURL("/password/simple_password.html"));
  std::string attacker_redirect =
      "window.location.href = '" + http_url.spec() + "';";
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       attacker_redirect));

  NavigationObserver attacker_observer(WebContents());
  attacker_observer.SetPathToWaitFor("/password/simple_password.html");
  attacker_observer.Wait();

  EXPECT_TRUE(prompt_observer.IsSavePromptShownAutomatically());

  std::string fill_and_submit_attacker_form =
      "document.getElementById('username_field').value = 'attacker_username';"
      "document.getElementById('password_field').value = 'attacker_password';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(
      content::ExecuteScript(WebContents(), fill_and_submit_attacker_form));

  NavigationObserver done_observer(WebContents());
  done_observer.SetPathToWaitFor("/password/done.html");
  done_observer.Wait();

  EXPECT_TRUE(prompt_observer.IsSavePromptShownAutomatically());
  prompt_observer.AcceptSavePrompt();

  // Wait for password store and check that credentials are stored.
  WaitForPasswordStore();
  CheckThatCredentialsStored("user", "password");
}

// Tests that after HTTP -> HTTPS migration the credential is autofilled.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       HttpMigratedCredentialAutofilled) {
  // Add an http credential to the password store.
  GURL https_origin = https_test_server().base_url();
  ASSERT_TRUE(https_origin.SchemeIs(url::kHttpsScheme));
  GURL::Replacements rep;
  rep.SetSchemeStr(url::kHttpScheme);
  GURL http_origin = https_origin.ReplaceComponents(rep);
  autofill::PasswordForm http_form;
  http_form.signon_realm = http_origin.spec();
  http_form.origin = http_origin;
  // Assume that the previous action was already HTTPS one matching the current
  // page.
  http_form.action = https_origin;
  http_form.username_value = base::ASCIIToUTF16("user");
  http_form.password_value = base::ASCIIToUTF16("12345");
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS).get());
  password_store->AddLogin(http_form);

  NavigationObserver form_observer(WebContents());
  ui_test_utils::NavigateToURL(
      browser(), https_test_server().GetURL("/password/password_form.html"));
  form_observer.Wait();
  WaitForPasswordStore();

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling fields.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));
  WaitForElementValue("username_field", "user");
  WaitForElementValue("password_field", "12345");
}

// Tests that obsolete HTTP credentials are moved when a site migrated to HTTPS
// and has HSTS enabled.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ObsoleteHttpCredentialMovedOnMigrationToHstsSite) {
  // Add an http credential to the password store.
  GURL https_origin = https_test_server().base_url();
  ASSERT_TRUE(https_origin.SchemeIs(url::kHttpsScheme));
  GURL::Replacements rep;
  rep.SetSchemeStr(url::kHttpScheme);
  GURL http_origin = https_origin.ReplaceComponents(rep);
  autofill::PasswordForm http_form;
  http_form.signon_realm = http_origin.spec();
  http_form.origin = http_origin;
  http_form.username_value = base::ASCIIToUTF16("user");
  http_form.password_value = base::ASCIIToUTF16("12345");
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  password_store->AddLogin(http_form);

  // Treat the host of the HTTPS test server as HSTS.
  AddHSTSHost(https_test_server().host_port_pair().host());

  // Navigate to HTTPS page and trigger the migration.
  NavigationObserver form_observer(WebContents());
  ui_test_utils::NavigateToURL(
      browser(), https_test_server().GetURL("/password/password_form.html"));
  form_observer.Wait();

  // Issue the query for HTTPS credentials.
  WaitForPasswordStore();

  // Realize there are no HTTPS credentials and issue the query for HTTP
  // credentials instead.
  WaitForPasswordStore();

  // Sync with IO thread before continuing. This is necessary, because the
  // credential migration triggers a query for the HSTS state which gets
  // executed on the IO thread. The actual task is empty, because only the reply
  // is relevant. By the time the reply is executed it is guaranteed that the
  // migration is completed.
  base::RunLoop run_loop;
  content::BrowserThread::PostTaskAndReply(content::BrowserThread::IO,
                                           FROM_HERE, base::BindOnce([]() {}),
                                           run_loop.QuitClosure());
  run_loop.Run();

  // Migration updates should touch the password store.
  WaitForPasswordStore();
  // Only HTTPS passwords should be present.
  EXPECT_TRUE(
      password_store->stored_passwords().at(http_origin.spec()).empty());
  EXPECT_FALSE(
      password_store->stored_passwords().at(https_origin.spec()).empty());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptWhenPasswordFormWithoutUsernameFieldSubmitted) {
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS).get());

  EXPECT_TRUE(password_store->IsEmpty());

  NavigateToFile("/password/form_with_only_password_field.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string submit =
      "document.getElementById('password').value = 'password';"
      "document.getElementById('submit-button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), submit));
  observer.Wait();

  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
  prompt_observer->AcceptSavePrompt();

  WaitForPasswordStore();
  EXPECT_FALSE(password_store->IsEmpty());
}

// Test that if a form gets autofilled, then it gets autofilled on re-creation
// as well.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ReCreatedFormsGetFilled) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("random");
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/dynamic_password_form.html");
  const std::string create_form =
      "document.getElementById('create_form_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), create_form));
  // Wait until the username is filled, to make sure autofill kicked in.
  WaitForElementValue("username_id", "temp");

  // Now the form gets deleted and created again. It should get autofilled
  // again.
  const std::string delete_form =
      "var form = document.getElementById('dynamic_form_id');"
      "form.parentNode.removeChild(form);";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), delete_form));
  ASSERT_TRUE(content::ExecuteScript(WebContents(), create_form));
  WaitForElementValue("username_id", "temp");
}

// Test that if the same dynamic form is created multiple times then all of them
// are autofilled and no unnecessary PasswordStore requests are fired.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       DuplicateFormsGetFilled) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("random");
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/recurring_dynamic_form.html");
  ASSERT_TRUE(content::ExecuteScript(WebContents(), "addForm();"));
  // Wait until the username is filled, to make sure autofill kicked in.
  WaitForJsElementValue("document.body.children[0].children[0]", "temp");
  WaitForJsElementValue("document.body.children[0].children[1]", "random");

  // It's a trick. There should be no second request to the password store since
  // the existing PasswordFormManager will manage the new form.
  password_store->Clear();
  // Add one more form.
  ASSERT_TRUE(content::ExecuteScript(WebContents(), "addForm();"));
  // Wait until the username is filled, to make sure autofill kicked in.
  WaitForJsElementValue("document.body.children[1].children[0]", "temp");
  WaitForJsElementValue("document.body.children[1].children[1]", "random");
}

// Test that an autofilled credential is deleted then the password manager
// doesn't try to resurrect it on navigation.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       DeletedPasswordIsNotRevived) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("admin");
  signin_form.password_value = base::ASCIIToUTF16("1234");
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/password_form.html");
  // Let the user interact with the page.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));
  // Wait until that interaction causes the username and the password value to
  // be revealed.
  WaitForElementValue("username_field", "admin");

  // Now the credential is removed via the settings or the bubble.
  password_store->RemoveLogin(signin_form);
  WaitForPasswordStore();

  // Submit the form. It shouldn't revive the credential in the store.
  NavigationObserver observer(WebContents());
  ASSERT_TRUE(content::ExecuteScript(
      WebContents(), "document.getElementById('input_submit_button').click()"));
  observer.Wait();

  WaitForPasswordStore();
  EXPECT_TRUE(password_store->IsEmpty());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForPushStateWhenFormDisappears) {
  NavigateToFile("/password/password_push_state.html");

  // Verify that we show the save password prompt if 'history.pushState()'
  // is called after form submission is suppressed by, for example, calling
  // preventDefault() in a form's submit event handler.
  // Note that calling 'submit()' on a form with javascript doesn't call
  // the onsubmit handler, so we click the submit button instead.
  // Also note that the prompt will only show up if the form disappers
  // after submission
  NavigationObserver observer(WebContents());
  observer.set_quit_on_entry_committed(true);
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

// Similar to the case above, but this time the form persists after
// 'history.pushState()'. And save password prompt should not show up
// in this case.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptForPushStateWhenFormPersists) {
  NavigateToFile("/password/password_push_state.html");

  // Set |should_delete_testform| to false to keep submitted form visible after
  // history.pushsTate();
  NavigationObserver observer(WebContents());
  observer.set_quit_on_entry_committed(true);
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "should_delete_testform = false;"
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

// The password manager should distinguish forms with empty actions. After
// successful login, the login form disappears, but the another one shouldn't be
// recognized as the login form. The save prompt should appear.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForPushStateWhenFormWithEmptyActionDisappears) {
  NavigateToFile("/password/password_push_state.html");

  NavigationObserver observer(WebContents());
  observer.set_quit_on_entry_committed(true);
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('ea_username_field').value = 'temp';"
      "document.getElementById('ea_password_field').value = 'random';"
      "document.getElementById('ea_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

// Similar to the case above, but this time the form persists after
// 'history.pushState()'. The password manager should find the login form even
// if the action of the form is empty. Save password prompt should not show up.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForPushStateWhenFormWithEmptyActionPersists) {
  NavigateToFile("/password/password_push_state.html");

  NavigationObserver observer(WebContents());
  observer.set_quit_on_entry_committed(true);
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "should_delete_testform = false;"
      "document.getElementById('ea_username_field').value = 'temp';"
      "document.getElementById('ea_password_field').value = 'random';"
      "document.getElementById('ea_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

// Current and target URLs contain different parameters and references. This
// test checks that parameters and references in origins are ignored for
// form origin comparison.
IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    PromptForPushStateWhenFormDisappears_ParametersInOrigins) {
  NavigateToFile("/password/password_push_state.html?login#r");

  NavigationObserver observer(WebContents());
  observer.set_quit_on_entry_committed(true);
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "add_parameters_to_target_url = true;"
      "document.getElementById('pa_username_field').value = 'temp';"
      "document.getElementById('pa_password_field').value = 'random';"
      "document.getElementById('pa_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

// Similar to the case above, but this time the form persists after
// 'history.pushState()'. The password manager should find the login form even
// if target and current URLs contain different parameters or references.
// Save password prompt should not show up.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForPushStateWhenFormPersists_ParametersInOrigins) {
  NavigateToFile("/password/password_push_state.html?login#r");

  NavigationObserver observer(WebContents());
  observer.set_quit_on_entry_committed(true);
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "should_delete_testform = false;"
      "add_parameters_to_target_url = true;"
      "document.getElementById('pa_username_field').value = 'temp';"
      "document.getElementById('pa_password_field').value = 'random';"
      "document.getElementById('pa_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       InFrameNavigationDoesNotClearPopupState) {
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("random123");
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/password_form.html");

  // Mock out the AutofillClient so we know how long to wait. Unfortunately
  // there isn't otherwise a good event to wait on to verify that the popup
  // would have been shown.
  password_manager::ContentPasswordManagerDriverFactory* driver_factory =
      password_manager::ContentPasswordManagerDriverFactory::FromWebContents(
          WebContents());
  ObservingAutofillClient::CreateForWebContents(WebContents());
  ObservingAutofillClient* observing_autofill_client =
      ObservingAutofillClient::FromWebContents(WebContents());
  password_manager::ContentPasswordManagerDriver* driver =
      driver_factory->GetDriverForFrame(WebContents()->GetMainFrame());
  DCHECK(driver);
  driver->GetPasswordAutofillManager()->set_autofill_client(
      observing_autofill_client);

  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(),
      "var usernameRect = document.getElementById('username_field')"
      ".getBoundingClientRect();"));

  // Trigger in page navigation.
  std::string in_page_navigate = "location.hash = '#blah';";
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       in_page_navigate));

  // Click on the username field to display the popup.
  int top;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send(usernameRect.top);", &top));
  int left;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send(usernameRect.left);", &left));

  content::SimulateMouseClickAt(WebContents(), 0,
                                blink::WebMouseEvent::Button::kLeft,
                                gfx::Point(left + 1, top + 1));
  // Make sure the popup would be shown.
  observing_autofill_client->WaitForAutofillPopup();
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ChangePwdFormBubbleShown) {
  NavigateToFile("/password/password_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('chg_username_field').value = 'temp';"
      "document.getElementById('chg_password_field').value = 'random';"
      "document.getElementById('chg_new_password_1').value = 'random1';"
      "document.getElementById('chg_new_password_2').value = 'random1';"
      "document.getElementById('chg_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ChangePwdFormPushStateBubbleShown) {
  NavigateToFile("/password/password_push_state.html");

  NavigationObserver observer(WebContents());
  observer.set_quit_on_entry_committed(true);
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('chg_username_field').value = 'temp';"
      "document.getElementById('chg_password_field').value = 'random';"
      "document.getElementById('chg_new_password_1').value = 'random1';"
      "document.getElementById('chg_new_password_2').value = 'random1';"
      "document.getElementById('chg_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoPromptOnBack) {
  // Go to a successful landing page through submitting first, so that it is
  // reachable through going back, and the remembered page transition is form
  // submit. There is no need to submit non-empty strings.
  NavigateToFile("/password/password_form.html");

  NavigationObserver dummy_submit_observer(WebContents());
  std::string just_submit =
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), just_submit));
  dummy_submit_observer.Wait();

  // Now go to a page with a form again, fill the form, and go back instead of
  // submitting it.
  NavigateToFile("/password/dummy_submit.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  // The (dummy) submit is necessary to provisionally save the typed password. A
  // user typing in the password field would not need to submit to provisionally
  // save it, but the script cannot trigger that just by assigning to the
  // field's value.
  std::string fill_and_back =
      "document.getElementById('password_field').value = 'random';"
      "document.getElementById('input_submit_button').click();"
      "window.history.back();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_back));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
}

// Regression test for http://crbug.com/452306
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ChangingTextToPasswordFieldOnSignupForm) {
  NavigateToFile("/password/signup_form.html");

  // In this case, pretend that username_field is actually a password field
  // that starts as a text field to simulate placeholder.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string change_and_submit =
      "document.getElementById('other_info').value = 'username';"
      "document.getElementById('username_field').type = 'password';"
      "document.getElementById('username_field').value = 'mypass';"
      "document.getElementById('password_field').value = 'mypass';"
      "document.getElementById('testform').submit();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), change_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

// Regression test for http://crbug.com/451631
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       SavingOnManyPasswordFieldsTest) {
  // Simulate Macy's registration page, which contains the normal 2 password
  // fields for confirming the new password plus 2 more fields for security
  // questions and credit card. Make sure that saving works correctly for such
  // sites.
  NavigateToFile("/password/many_password_signup_form.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'username';"
      "document.getElementById('password_field').value = 'mypass';"
      "document.getElementById('confirm_field').value = 'mypass';"
      "document.getElementById('security_answer').value = 'hometown';"
      "document.getElementById('SSN').value = '1234';"
      "document.getElementById('testform').submit();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       SaveWhenIFrameDestroyedOnFormSubmit) {
  NavigateToFile("/password/frame_detached_on_submit.html");

  // Need to pay attention for a message that XHR has finished since there
  // is no navigation to wait for.
  content::DOMMessageQueue message_queue;

  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "var iframe = document.getElementById('login_iframe');"
      "var frame_doc = iframe.contentDocument;"
      "frame_doc.getElementById('username_field').value = 'temp';"
      "frame_doc.getElementById('password_field').value = 'random';"
      "frame_doc.getElementById('submit_button').click();";

  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  std::string message;
  while (message_queue.WaitForMessage(&message)) {
    if (message == "\"SUBMISSION_FINISHED\"")
      break;
  }

  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

// Check that a username and password are filled into forms in iframes
// that don't share the security origin with the main frame, but have PSL
// matched origins.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PSLMatchedCrossSiteFillTest) {
  GURL main_frame_url = embedded_test_server()->GetURL(
      "www.foo.com", "/password/password_form_in_crosssite_iframe.html");
  NavigationObserver observer(WebContents());
  ui_test_utils::NavigateToURL(browser(), main_frame_url);
  observer.Wait();

  // Create an iframe and navigate cross-site.
  NavigationObserver iframe_observer(WebContents());
  iframe_observer.SetPathToWaitFor("/password/crossite_iframe_content.html");
  GURL iframe_url = embedded_test_server()->GetURL(
      "abc.foo.com", "/password/crossite_iframe_content.html");
  std::string create_iframe =
      base::StringPrintf("create_iframe('%s');", iframe_url.spec().c_str());
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       create_iframe));
  iframe_observer.Wait();

  // Store a password for autofill later.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = iframe_url.GetOrigin().spec();
  signin_form.origin = iframe_url;
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("pa55w0rd");
  password_store->AddLogin(signin_form);
  WaitForPasswordStore();

  // Visit the form again.
  NavigationObserver reload_observer(WebContents());
  ui_test_utils::NavigateToURL(browser(), main_frame_url);
  reload_observer.Wait();

  NavigationObserver iframe_observer_2(WebContents());
  iframe_observer_2.SetPathToWaitFor("/password/crossite_iframe_content.html");
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       create_iframe));
  iframe_observer_2.Wait();

  // Simulate the user interaction in the iframe which should trigger autofill.
  // Click in the middle of the frame to avoid the border.
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(),
      "var iframeRect = document.getElementById("
      "'iframe').getBoundingClientRect();"));
  int y;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send((iframeRect.top +"
      "iframeRect.bottom) / 2);",
      &y));
  int x;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send((iframeRect.left + iframeRect.right)"
      "/ 2);",
      &x));

  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(x, y));

  std::string username_field;
  std::string password_field;

  // Verify username has been autofilled
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), "sendMessage('get_username');", &username_field));
  EXPECT_EQ("temp", username_field);

  // Verify password has been autofilled
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), "sendMessage('get_password');", &password_field));
  EXPECT_EQ("pa55w0rd", password_field);
}

// Check that a username and password are not filled in forms in iframes
// that don't have PSL matched origins.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PSLUnMatchedCrossSiteFillTest) {
  GURL main_frame_url = embedded_test_server()->GetURL(
      "www.foo.com", "/password/password_form_in_crosssite_iframe.html");
  NavigationObserver observer(WebContents());
  ui_test_utils::NavigateToURL(browser(), main_frame_url);
  observer.Wait();

  // Create an iframe and navigate cross-site.
  NavigationObserver iframe_observer(WebContents());
  iframe_observer.SetPathToWaitFor("/password/crossite_iframe_content.html");
  GURL iframe_url = embedded_test_server()->GetURL(
      "www.bar.com", "/password/crossite_iframe_content.html");
  std::string create_iframe =
      base::StringPrintf("create_iframe('%s');", iframe_url.spec().c_str());
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       create_iframe));
  iframe_observer.Wait();

  // Store a password for autofill later.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = iframe_url.GetOrigin().spec();
  signin_form.origin = iframe_url;
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("pa55w0rd");
  password_store->AddLogin(signin_form);
  WaitForPasswordStore();

  // Visit the form again.
  NavigationObserver reload_observer(WebContents());
  ui_test_utils::NavigateToURL(browser(), main_frame_url);
  reload_observer.Wait();

  NavigationObserver iframe_observer_2(WebContents());
  iframe_observer_2.SetPathToWaitFor("/password/crossite_iframe_content.html");
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       create_iframe));
  iframe_observer_2.Wait();

  // Simulate the user interaction in the iframe which should trigger autofill.
  // Click in the middle of the frame to avoid the border.
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(),
      "var iframeRect = document.getElementById("
      "'iframe').getBoundingClientRect();"));
  int y;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send((iframeRect.top +"
      "iframeRect.bottom) / 2);",
      &y));
  int x;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send((iframeRect.left + iframeRect.right)"
      "/ 2);",
      &x));

  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(x, y));

  // Verify username is not autofilled
  std::string empty_username;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), "sendMessage('get_username');", &empty_username));
  EXPECT_EQ("", empty_username);
  // Verify password is not autofilled
  std::string empty_password;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), "sendMessage('get_password');", &empty_password));
  EXPECT_EQ("", empty_password);
}

// Check that a password form in an iframe of same origin will not be
// filled in until user interact with the iframe.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       SameOriginIframeAutoFillTest) {
  // Visit the sign-up form to store a password for autofill later
  NavigateToFile("/password/password_form_in_same_origin_iframe.html");
  NavigationObserver observer(WebContents());
  observer.SetPathToWaitFor("/password/done.html");
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));

  std::string submit =
      "var ifrmDoc = document.getElementById('iframe').contentDocument;"
      "ifrmDoc.getElementById('username_field').value = 'temp';"
      "ifrmDoc.getElementById('password_field').value = 'pa55w0rd';"
      "ifrmDoc.getElementById('input_submit_button').click();";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
  prompt_observer->AcceptSavePrompt();

  // Visit the form again
  NavigationObserver reload_observer(WebContents());
  NavigateToFile("/password/password_form_in_same_origin_iframe.html");
  reload_observer.Wait();

  // Verify password and username are not accessible yet.
  CheckElementValue("iframe", "username_field", "");
  CheckElementValue("iframe", "password_field", "");

  // Simulate the user interaction in the iframe which should trigger autofill.
  // Click in the middle of the frame to avoid the border.
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(),
      "var iframeRect = document.getElementById("
      "'iframe').getBoundingClientRect();"));
  int y;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send((iframeRect.top +"
      "iframeRect.bottom) / 2);",
      &y));
  int x;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send((iframeRect.left + iframeRect.right)"
      "/ 2);",
      &x));

  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(x, y));
  // Verify username and password have been autofilled
  WaitForElementValue("iframe", "username_field", "temp");
  WaitForElementValue("iframe", "password_field", "pa55w0rd");
}

// The password manager driver will kill processes when they try to access
// passwords of sites other than the site the process is dedicated to, under
// site isolation.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       CrossSitePasswordEnforcement) {
  // The code under test is only active under site isolation.
  if (!content::AreAllSitesIsolatedForTesting()) {
    return;
  }

  // Navigate the main frame.
  GURL main_frame_url = embedded_test_server()->GetURL(
      "/password/password_form_in_crosssite_iframe.html");
  NavigationObserver observer(WebContents());
  ui_test_utils::NavigateToURL(browser(), main_frame_url);
  observer.Wait();

  // Create an iframe and navigate cross-site.
  NavigationObserver iframe_observer(WebContents());
  iframe_observer.SetPathToWaitFor("/password/crossite_iframe_content.html");
  GURL iframe_url = embedded_test_server()->GetURL(
      "foo.com", "/password/crossite_iframe_content.html");
  std::string create_iframe =
      base::StringPrintf("create_iframe('%s');", iframe_url.spec().c_str());
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(RenderFrameHost(),
                                                       create_iframe));
  iframe_observer.Wait();

  // The iframe should get its own process.
  content::RenderFrameHost* main_frame = WebContents()->GetMainFrame();
  content::RenderFrameHost* iframe = iframe_observer.render_frame_host();
  content::SiteInstance* main_site_instance = main_frame->GetSiteInstance();
  content::SiteInstance* iframe_site_instance = iframe->GetSiteInstance();
  EXPECT_NE(main_site_instance, iframe_site_instance);
  EXPECT_NE(main_frame->GetProcess(), iframe->GetProcess());

  content::RenderProcessHostWatcher iframe_killed(
      iframe->GetProcess(),
      content::RenderProcessHostWatcher::WATCH_FOR_PROCESS_EXIT);

  // Try to get cross-site passwords from the subframe's process and wait for it
  // to be killed.
  std::vector<autofill::PasswordForm> password_forms;
  password_forms.push_back(autofill::PasswordForm());
  password_forms.back().origin = main_frame_url;
  autofill::mojom::PasswordManagerDriver* driver =
      ChromePasswordManagerClient::FromWebContents(WebContents());
  EXPECT_TRUE(driver);
  driver->PasswordFormsParsed(password_forms);

  iframe_killed.Wait();
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ChangePwdNoAccountStored) {
  NavigateToFile("/password/password_form.html");

  // Fill a form and submit through a <input type="submit"> button.
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));

  std::string fill_and_submit =
      "document.getElementById('chg_password_wo_username_field').value = "
      "'old_pw';"
      "document.getElementById('chg_new_password_wo_username_1').value = "
      "'new_pw';"
      "document.getElementById('chg_new_password_wo_username_2').value = "
      "'new_pw';"
      "document.getElementById('chg_submit_wo_username_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  // No credentials stored before, so save bubble is shown.
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
  prompt_observer->AcceptSavePrompt();

  // Check that credentials are stored.
  WaitForPasswordStore();
  CheckThatCredentialsStored("", "new_pw");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ChangePwd1AccountStored) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.password_value = base::ASCIIToUTF16("pw");
  signin_form.username_value = base::ASCIIToUTF16("temp");
  password_store->AddLogin(signin_form);

  // Check that password update bubble is shown.
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit_change_password =
      "document.getElementById('chg_password_wo_username_field').value = "
      "'random';"
      "document.getElementById('chg_new_password_wo_username_1').value = "
      "'new_pw';"
      "document.getElementById('chg_new_password_wo_username_2').value = "
      "'new_pw';"
      "document.getElementById('chg_submit_wo_username_button').click()";
  ASSERT_TRUE(
      content::ExecuteScript(WebContents(), fill_and_submit_change_password));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsUpdatePromptShownAutomatically());

  // We emulate that the user clicks "Update" button.
  const autofill::PasswordForm& pending_credentials =
      ManagePasswordsUIController::FromWebContents(WebContents())
          ->GetPendingPassword();
  prompt_observer->AcceptUpdatePrompt(pending_credentials);

  WaitForPasswordStore();
  CheckThatCredentialsStored("temp", "new_pw");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PasswordOverridenUpdateBubbleShown) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("pw");
  password_store->AddLogin(signin_form);

  // Disable autofill. If a password is autofilled then all the Javacript
  // changes are discarded. The test would not be able to feed the new password
  // below.
  base::test::ScopedFeatureList scoped_feature_list;
  scoped_feature_list.InitAndEnableFeature(features::kFillOnAccountSelect);

  // Check that password update bubble is shown.
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'new_pw';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  // The stored password "pw" was overriden with "new_pw", so update prompt is
  // expected.
  EXPECT_TRUE(prompt_observer->IsUpdatePromptShownAutomatically());

  const autofill::PasswordForm stored_form =
      password_store->stored_passwords().begin()->second[0];
  prompt_observer->AcceptUpdatePrompt(stored_form);
  WaitForPasswordStore();
  CheckThatCredentialsStored("temp", "new_pw");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PasswordNotOverridenUpdateBubbleNotShown) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("pw");
  password_store->AddLogin(signin_form);

  // Check that password update bubble is shown.
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username_field').value = 'temp';"
      "document.getElementById('password_field').value = 'pw';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  // The stored password "pw" was not overriden, so update prompt is not
  // expected.
  EXPECT_FALSE(prompt_observer->IsUpdatePromptShownAutomatically());
  CheckThatCredentialsStored("temp", "pw");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       MultiplePasswordsWithPasswordSelectionEnabled) {
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  // It is important that these 3 passwords are different. Because if two of
  // them are the same, it is going to be treated as a password update and the
  // dropdown will not be shown.
  std::string fill_and_submit =
      "document.getElementById('chg_password_wo_username_field').value = "
      "'pass1';"
      "document.getElementById('chg_new_password_wo_username_1').value = "
      "'pass2';"
      "document.getElementById('chg_new_password_wo_username_2').value = "
      "'pass3';"
      "document.getElementById('chg_submit_wo_username_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  // 3 possible passwords are going to be shown in a dropdown when the password
  // selection feature is enabled. The first one will be selected as the main
  // password by default. All three will be in the all_possible_passwords
  // list. The save password prompt is expected.
  BubbleObserver bubble_observer(WebContents());
  EXPECT_TRUE(bubble_observer.IsSavePromptShownAutomatically());
  EXPECT_EQ(base::ASCIIToUTF16("pass1"),
            ManagePasswordsUIController::FromWebContents(WebContents())
                ->GetPendingPassword()
                .password_value);
  EXPECT_THAT(
      ManagePasswordsUIController::FromWebContents(WebContents())
          ->GetPendingPassword()
          .all_possible_passwords,
      ElementsAre(autofill::ValueElementPair(
                      base::ASCIIToUTF16("pass1"),
                      base::ASCIIToUTF16("chg_password_wo_username_field")),
                  autofill::ValueElementPair(
                      base::ASCIIToUTF16("pass2"),
                      base::ASCIIToUTF16("chg_new_password_wo_username_1")),
                  autofill::ValueElementPair(
                      base::ASCIIToUTF16("pass3"),
                      base::ASCIIToUTF16("chg_new_password_wo_username_2"))));
  bubble_observer.AcceptSavePrompt();
  WaitForPasswordStore();
  CheckThatCredentialsStored("", "pass1");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ChangePwdWhenTheFormContainNotUsernameTextfield) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.password_value = base::ASCIIToUTF16("pw");
  signin_form.username_value = base::ASCIIToUTF16("temp");
  password_store->AddLogin(signin_form);

  // Check that password update bubble is shown.
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit_change_password =
      "document.getElementById('chg_text_field').value = '3';"
      "document.getElementById('chg_password_withtext_field').value"
      " = 'random';"
      "document.getElementById('chg_new_password_withtext_username_1').value"
      " = 'new_pw';"
      "document.getElementById('chg_new_password_withtext_username_2').value"
      " = 'new_pw';"
      "document.getElementById('chg_submit_withtext_button').click()";
  ASSERT_TRUE(
      content::ExecuteScript(WebContents(), fill_and_submit_change_password));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsUpdatePromptShownAutomatically());

  const autofill::PasswordForm stored_form =
      password_store->stored_passwords().begin()->second[0];
  prompt_observer->AcceptUpdatePrompt(stored_form);
  WaitForPasswordStore();
  CheckThatCredentialsStored("temp", "new_pw");
}

// Test whether the password form with the username and password fields having
// ambiguity in id attribute gets autofilled correctly.
IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    AutofillSuggestionsForPasswordFormWithAmbiguousIdAttribute) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the password form having ambiguous Ids for username and
  // password fields and verify whether username and password is autofilled.
  NavigateToFile("/password/ambiguous_password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling fields.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  WaitForElementValue("ambiguous_form", 0 /* elements_index */, "myusername");
  WaitForElementValue("ambiguous_form", 1 /* elements_index */, "mypassword");
}

// Test whether the password form having username and password fields without
// name and id attribute gets autofilled correctly.
IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    AutofillSuggestionsForPasswordFormWithoutNameOrIdAttribute) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the password form having no Ids for username and password
  // fields and verify whether username and password is autofilled.
  NavigateToFile("/password/ambiguous_password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling fields.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  WaitForElementValue("no_name_id_form", 0 /* elements_index */, "myusername");
  WaitForElementValue("no_name_id_form", 1 /* elements_index */, "mypassword");
}

// Test whether the change password form having username and password fields
// without name and id attribute gets autofilled correctly.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       AutofillSuggestionsForChangePwdWithEmptyNames) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the password form having no Ids for username and password
  // fields and verify whether username and password is autofilled.
  NavigateToFile("/password/ambiguous_password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling fields.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  WaitForElementValue("change_pwd_but_no_autocomplete", 0 /* elements_index */,
                      "myusername");
  WaitForElementValue("change_pwd_but_no_autocomplete", 1 /* elements_index */,
                      "mypassword");

  std::string get_new_password =
      "window.domAutomationController.send("
      "  document.getElementById("
      "    'change_pwd_but_no_autocomplete').elements[2].value);";
  std::string new_password;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), get_new_password, &new_password));
  EXPECT_EQ("", new_password);
}

// Test whether the change password form having username and password fields
// with empty names but having |autocomplete='current-password'| gets autofilled
// correctly.
IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    AutofillSuggestionsForChangePwdWithEmptyNamesAndAutocomplete) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the password form having no Ids for username and password
  // fields and verify whether username and password is autofilled.
  NavigateToFile("/password/ambiguous_password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling fields.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  WaitForElementValue("change_pwd", 0 /* elements_index */, "myusername");
  WaitForElementValue("change_pwd", 1 /* elements_index */, "mypassword");

  std::string get_new_password =
      "window.domAutomationController.send("
      "  document.getElementById('change_pwd').elements[2].value);";
  std::string new_password;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), get_new_password, &new_password));
  EXPECT_EQ("", new_password);
}

// Test whether the change password form having username and password fields
// with empty names but having only new password fields having
// |autocomplete='new-password'| atrribute do not get autofilled.
IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    AutofillSuggestionsForChangePwdWithEmptyNamesButOnlyNewPwdField) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the password form having no Ids for username and password
  // fields and verify whether username and password is autofilled.
  NavigateToFile("/password/ambiguous_password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling fields.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  std::string get_username =
      "window.domAutomationController.send("
      "  document.getElementById("
      "    'change_pwd_but_no_old_pwd').elements[0].value);";
  std::string actual_username;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), get_username, &actual_username));
  EXPECT_EQ("", actual_username);

  std::string get_new_password =
      "window.domAutomationController.send("
      "  document.getElementById("
      "    'change_pwd_but_no_old_pwd').elements[1].value);";
  std::string new_password;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), get_new_password, &new_password));
  EXPECT_EQ("", new_password);

  std::string get_retype_password =
      "window.domAutomationController.send("
      "  document.getElementById("
      "    'change_pwd_but_no_old_pwd').elements[2].value);";
  std::string retyped_password;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      RenderFrameHost(), get_retype_password, &retyped_password));
  EXPECT_EQ("", retyped_password);
}

// When there are multiple LoginModelObservers (e.g., multiple HTTP auth dialogs
// as in http://crbug.com/537823), ensure that credentials from PasswordStore
// distributed to them are filtered by the realm.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       BasicAuthSeparateRealms) {
  // We must use a new test server here because embedded_test_server() is
  // already started at this point and adding the request handler to it would
  // not be thread safe.
  net::EmbeddedTestServer http_test_server;
  http_test_server.RegisterRequestHandler(base::Bind(&HandleTestAuthRequest));
  ASSERT_TRUE(http_test_server.Start());

  // Save credentials for "test realm" in the store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm creds;
  creds.scheme = autofill::PasswordForm::SCHEME_BASIC;
  creds.signon_realm = http_test_server.base_url().spec() + "test realm";
  creds.password_value = base::ASCIIToUTF16("pw");
  creds.username_value = base::ASCIIToUTF16("temp");
  password_store->AddLogin(creds);
  WaitForPasswordStore();
  ASSERT_FALSE(password_store->IsEmpty());

  // In addition to the LoginModelObserver created automatically for the HTTP
  // auth dialog, also create a mock observer, for a different realm.
  MockLoginModelObserver mock_login_model_observer;
  PasswordManager* password_manager =
      static_cast<password_manager::PasswordManagerClient*>(
          ChromePasswordManagerClient::FromWebContents(WebContents()))
          ->GetPasswordManager();
  autofill::PasswordForm other_form(creds);
  other_form.signon_realm = "https://example.com/other realm";
  password_manager->AddObserverAndDeliverCredentials(&mock_login_model_observer,
                                                     other_form);
  // The mock observer should not receive the stored credentials.
  EXPECT_CALL(mock_login_model_observer, OnAutofillDataAvailableInternal(_, _))
      .Times(0);

  // Now wait until the navigation to the test server causes a HTTP auth dialog
  // to appear.
  content::NavigationController* nav_controller =
      &WebContents()->GetController();
  WindowedAuthNeededObserver auth_needed_observer(nav_controller);
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), http_test_server.GetURL("/basic_auth"),
      WindowOpenDisposition::CURRENT_TAB, ui_test_utils::BROWSER_TEST_NONE);
  auth_needed_observer.Wait();

  // The auth dialog caused a query to PasswordStore, make sure it was
  // processed.
  WaitForPasswordStore();

  password_manager->RemoveObserver(&mock_login_model_observer);
}

// Test whether the password form which is loaded as hidden is autofilled
// correctly. This happens very often in situations when in order to sign-in the
// user clicks a sign-in button and a hidden passsword form becomes visible.
// This test differs from AutofillSuggestionsForProblematicPasswordForm in that
// the form is hidden and in that test only some fields are hidden.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       AutofillSuggestionsHiddenPasswordForm) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the hidden password form and verify whether username and
  // password is autofilled.
  NavigateToFile("/password/password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling the password.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  WaitForElementValue("hidden_password_form_username", "myusername");
  WaitForElementValue("hidden_password_form_password", "mypassword");
}

// Test whether the password form with the problematic invisible password field
// gets autofilled correctly.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       AutofillSuggestionsForProblematicPasswordForm) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the password form with a hidden password field and verify
  // whether username and password is autofilled.
  NavigateToFile("/password/password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling the password.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  WaitForElementValue("form_with_hidden_password_username", "myusername");
  WaitForElementValue("form_with_hidden_password_password", "mypassword");
}

// Test whether the password form with the problematic invisible password field
// in ambiguous password form gets autofilled correctly.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       AutofillSuggestionsForProblematicAmbiguousPasswordForm) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::PasswordStore> password_store =
      PasswordStoreFactory::GetForProfile(browser()->profile(),
                                          ServiceAccessType::IMPLICIT_ACCESS);
  autofill::PasswordForm login_form;
  login_form.signon_realm = embedded_test_server()->base_url().spec();
  login_form.action = embedded_test_server()->GetURL("/password/done.html");
  login_form.username_value = base::ASCIIToUTF16("myusername");
  login_form.password_value = base::ASCIIToUTF16("mypassword");
  password_store->AddLogin(login_form);

  // Now, navigate to the password form having ambiguous Ids for username and
  // password fields and verify whether username and password is autofilled.
  NavigateToFile("/password/ambiguous_password_form.html");

  // Let the user interact with the page, so that DOM gets modification events,
  // needed for autofilling fields.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  WaitForElementValue("hidden_password_form", 0 /* elements_index */,
                      "myusername");
  WaitForElementValue("hidden_password_form", 2 /* elements_index */,
                      "mypassword");
}

// Check that the internals page contains logs from the renderer.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       InternalsPage_Renderer) {
  // Open the internals page.
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), GURL("chrome://password-manager-internals"),
      WindowOpenDisposition::CURRENT_TAB,
      ui_test_utils::BROWSER_TEST_WAIT_FOR_NAVIGATION);
  content::WebContents* internals_web_contents = WebContents();

  // The renderer is supposed to ask whether logging is available. To avoid
  // race conditions between the answer "Logging is available" arriving from
  // the browser and actual logging callsites reached in the renderer, open
  // first an arbitrary page to ensure that the renderer queries the
  // availability of logging and has enough time to receive the answer.
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), embedded_test_server()->GetURL("/password/done.html"),
      WindowOpenDisposition::NEW_FOREGROUND_TAB,
      ui_test_utils::BROWSER_TEST_WAIT_FOR_NAVIGATION);
  content::WebContents* forms_web_contents =
      browser()->tab_strip_model()->GetActiveWebContents();

  // Now navigate to another page, containing some forms, so that the renderer
  // attempts to log. It should be a different page than the current one,
  // because just reloading the current one sometimes confused the Wait() call
  // and lead to timeouts (https://crbug.com/804398).
  NavigationObserver observer(forms_web_contents);
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), embedded_test_server()->GetURL("/password/password_form.html"),
      WindowOpenDisposition::CURRENT_TAB,
      ui_test_utils::BROWSER_TEST_WAIT_FOR_NAVIGATION);
  observer.Wait();

  std::string find_logs =
      "var text = document.getElementById('log-entries').innerText;"
      "var logs_found = /PasswordAutofillAgent::/.test(text);"
      "window.domAutomationController.send(logs_found);";
  bool logs_found = false;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractBool(
      internals_web_contents->GetMainFrame(), find_logs, &logs_found));
  EXPECT_TRUE(logs_found);
}

// Check that the internals page contains logs from the browser.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       InternalsPage_Browser) {
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), GURL("chrome://password-manager-internals"),
      WindowOpenDisposition::CURRENT_TAB,
      ui_test_utils::BROWSER_TEST_WAIT_FOR_NAVIGATION);
  content::WebContents* internals_web_contents = WebContents();

  ui_test_utils::NavigateToURLWithDisposition(
      browser(), embedded_test_server()->GetURL("/password/password_form.html"),
      WindowOpenDisposition::NEW_FOREGROUND_TAB,
      ui_test_utils::BROWSER_TEST_WAIT_FOR_NAVIGATION);

  std::string find_logs =
      "var text = document.getElementById('log-entries').innerText;"
      "var logs_found = /PasswordManager::/.test(text);"
      "window.domAutomationController.send(logs_found);";
  bool logs_found = false;
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractBool(
      internals_web_contents->GetMainFrame(), find_logs, &logs_found));
  EXPECT_TRUE(logs_found);
}

// Tests that submitted credentials are saved on a password form without
// username element when there are no stored credentials.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PasswordRetryFormSaveNoUsernameCredentials) {
  // Check that password save bubble is shown.
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('retry_password_field').value = 'pw';"
      "document.getElementById('retry_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
  prompt_observer->AcceptSavePrompt();

  WaitForPasswordStore();
  CheckThatCredentialsStored("", "pw");
}

// Tests that no bubble shown when a password form without username submitted
// and there is stored credentials with the same password.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PasswordRetryFormNoBubbleWhenPasswordTheSame) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("pw");
  password_store->AddLogin(signin_form);
  signin_form.username_value = base::ASCIIToUTF16("temp1");
  signin_form.password_value = base::ASCIIToUTF16("pw1");
  password_store->AddLogin(signin_form);

  // Check that no password bubble is shown when the submitted password is the
  // same in one of the stored credentials.
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('retry_password_field').value = 'pw';"
      "document.getElementById('retry_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());
  EXPECT_FALSE(prompt_observer->IsUpdatePromptShownAutomatically());
}

// Tests that the update bubble shown when a password form without username is
// submitted and there are stored credentials but with different password.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PasswordRetryFormUpdateBubbleShown) {
  // At first let us save credentials to the PasswordManager.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.username_value = base::ASCIIToUTF16("temp");
  signin_form.password_value = base::ASCIIToUTF16("pw");
  password_store->AddLogin(signin_form);

  // Check that password update bubble is shown.
  NavigateToFile("/password/password_form.html");
  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('retry_password_field').value = 'new_pw';"
      "document.getElementById('retry_submit_button').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  // The new password "new_pw" is used, so update prompt is expected.
  EXPECT_TRUE(prompt_observer->IsUpdatePromptShownAutomatically());

  const autofill::PasswordForm stored_form =
      password_store->stored_passwords().begin()->second[0];
  prompt_observer->AcceptUpdatePrompt(stored_form);

  WaitForPasswordStore();
  CheckThatCredentialsStored("temp", "new_pw");
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoCrashWhenNavigatingWithOpenAccountPicker) {
  // Save credentials with 'skip_zero_click'.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.password_value = base::ASCIIToUTF16("password");
  signin_form.username_value = base::ASCIIToUTF16("user");
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.skip_zero_click = true;
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/password_form.html");

  // Call the API to trigger the notification to the client, which raises the
  // account picker dialog.
  ASSERT_TRUE(content::ExecuteScript(
      WebContents(), "navigator.credentials.get({password: true})"));

  // Navigate while the picker is open.
  NavigateToFile("/password/password_form.html");

  // No crash!
}

// Tests that the prompt to save the password is still shown if the fields have
// the "autocomplete" attribute set off.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       PromptForSubmitWithAutocompleteOff) {
  NavigateToFile("/password/password_autocomplete_off_test.html");

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit =
      "document.getElementById('username').value = 'temp';"
      "document.getElementById('password').value = 'random';"
      "document.getElementById('submit').click()";
  ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
  observer.Wait();
  EXPECT_TRUE(prompt_observer->IsSavePromptShownAutomatically());
}

IN_PROC_BROWSER_TEST_P(
    PasswordManagerBrowserTestWithViewsFeature,
    SkipZeroClickNotToggledAfterSuccessfulSubmissionWithAPI) {
  // Save credentials with 'skip_zero_click'
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.password_value = base::ASCIIToUTF16("password");
  signin_form.username_value = base::ASCIIToUTF16("user");
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.skip_zero_click = true;
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/password_form.html");

  // Call the API to trigger the notification to the client.
  ASSERT_TRUE(content::ExecuteScript(
      WebContents(),
      "navigator.credentials.get({password: true, unmediated: true })"));

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit_change_password =
      "document.getElementById('username_field').value = 'user';"
      "document.getElementById('password_field').value = 'password';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(
      content::ExecuteScript(WebContents(), fill_and_submit_change_password));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());

  // Verify that the form's 'skip_zero_click' is not updated.
  auto& passwords_map = password_store->stored_passwords();
  ASSERT_EQ(1u, passwords_map.size());
  auto& passwords_vector = passwords_map.begin()->second;
  ASSERT_EQ(1u, passwords_vector.size());
  const autofill::PasswordForm& form = passwords_vector[0];
  EXPECT_EQ(base::ASCIIToUTF16("user"), form.username_value);
  EXPECT_EQ(base::ASCIIToUTF16("password"), form.password_value);
  EXPECT_TRUE(form.skip_zero_click);
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       SkipZeroClickNotToggledAfterSuccessfulAutofill) {
  // Save credentials with 'skip_zero_click'
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.password_value = base::ASCIIToUTF16("password");
  signin_form.username_value = base::ASCIIToUTF16("user");
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.skip_zero_click = true;
  password_store->AddLogin(signin_form);

  NavigateToFile("/password/password_form.html");

  // No API call.

  NavigationObserver observer(WebContents());
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  std::string fill_and_submit_change_password =
      "document.getElementById('username_field').value = 'user';"
      "document.getElementById('password_field').value = 'password';"
      "document.getElementById('input_submit_button').click()";
  ASSERT_TRUE(
      content::ExecuteScript(WebContents(), fill_and_submit_change_password));
  observer.Wait();
  EXPECT_FALSE(prompt_observer->IsSavePromptShownAutomatically());

  // Verify that the form's 'skip_zero_click' is not updated.
  auto& passwords_map = password_store->stored_passwords();
  ASSERT_EQ(1u, passwords_map.size());
  auto& passwords_vector = passwords_map.begin()->second;
  ASSERT_EQ(1u, passwords_vector.size());
  const autofill::PasswordForm& form = passwords_vector[0];
  EXPECT_EQ(base::ASCIIToUTF16("user"), form.username_value);
  EXPECT_EQ(base::ASCIIToUTF16("password"), form.password_value);
  EXPECT_TRUE(form.skip_zero_click);
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ReattachWebContents) {
  auto detached_web_contents = content::WebContents::Create(
      content::WebContents::CreateParams(WebContents()->GetBrowserContext()));
  NavigationObserver observer(detached_web_contents.get());
  detached_web_contents->GetController().LoadURL(
      embedded_test_server()->GetURL("/password/multi_frames.html"),
      content::Referrer(), ::ui::PAGE_TRANSITION_AUTO_TOPLEVEL, std::string());
  observer.Wait();
  // Ensure that there is at least one more frame created than just the main
  // frame.
  EXPECT_LT(1u, detached_web_contents->GetAllFrames().size());

  auto* tab_strip_model = browser()->tab_strip_model();
  // Check that the autofill and password manager driver factories are notified
  // about all frames, not just the main one. The factories should receive
  // messages for non-main frames, in particular
  // AutofillHostMsg_PasswordFormsParsed. If that were the first time the
  // factories hear about such frames, this would crash.
  tab_strip_model->AddWebContents(std::move(detached_web_contents), -1,
                                  ::ui::PAGE_TRANSITION_AUTO_TOPLEVEL,
                                  TabStripModel::ADD_ACTIVE);
}

IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       FillWhenFormWithHiddenUsername) {
  // At first let us save a credential to the password store.
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = embedded_test_server()->base_url().spec();
  signin_form.origin = embedded_test_server()->base_url();
  signin_form.action = embedded_test_server()->base_url();
  signin_form.username_value = base::ASCIIToUTF16("current_username");
  signin_form.password_value = base::ASCIIToUTF16("current_username_password");
  password_store->AddLogin(signin_form);
  signin_form.username_value = base::ASCIIToUTF16("last_used_username");
  signin_form.password_value = base::ASCIIToUTF16("last_used_password");
  signin_form.preferred = true;
  password_store->AddLogin(signin_form);
  WaitForPasswordStore();

  NavigateToFile("/password/hidden_username.html");

  // Let the user interact with the page.
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(1, 1));

  // current_username is hardcoded in the invisible text on the page so
  // current_username_password should be filled rather than last_used_password.
  WaitForElementValue("password", "current_username_password");
}

// Harness for showing dialogs as part of the DialogBrowserTest suite.
// Test params:
//  - bool popup_views_enabled: whether feature AutofillExpandedPopupViews
//        is enabled for testing.
class PasswordManagerDialogBrowserTest
    : public SupportsTestDialog<PasswordManagerBrowserTestBase>,
      public ::testing::WithParamInterface<bool> {
 public:
  PasswordManagerDialogBrowserTest() = default;

  // SupportsTestUi:
  void SetUp() override {
    // Secondary UI needs to be enabled before ShowUi for the test to work.
    UseMdOnly();
    SupportsTestUi::SetUp();
  }

  void SetUpCommandLine(base::CommandLine* command_line) override {
    SupportsTestDialog<PasswordManagerBrowserTestBase>::SetUpCommandLine(
        command_line);

    const bool popup_views_enabled = GetParam();
    scoped_feature_list_.InitWithFeatureState(
        autofill::kAutofillExpandedPopupViews, popup_views_enabled);
  }

  void ShowUi(const std::string& name) override {
    // Note regarding flakiness: LocationBarBubbleDelegateView::ShowForReason()
    // uses ShowInactive() unless the bubble is invoked with reason ==
    // USER_GESTURE. This means that, so long as these dialogs are not triggered
    // by gesture, the dialog does not attempt to take focus, and so should
    // never _lose_ focus in the test, which could cause flakes when tests are
    // run in parallel. LocationBarBubbles also dismiss on other events, but
    // only events in the WebContents. E.g. Rogue mouse clicks should not cause
    // the dialog to dismiss since they won't be sent via WebContents.
    // A user gesture is determined in browser_commands.cc by checking
    // ManagePasswordsUIController::IsAutomaticallyOpeningBubble(), but that's
    // set and cleared immediately while showing the bubble, so it can't be
    // checked here.
    NavigateToFile("/password/password_form.html");
    NavigationObserver observer(WebContents());
    std::string fill_and_submit =
        "document.getElementById('username_field').value = 'temp';"
        "document.getElementById('password_field').value = 'random';"
        "document.getElementById('input_submit_button').click()";
    ASSERT_TRUE(content::ExecuteScript(WebContents(), fill_and_submit));
    observer.Wait();
  }

 private:
  base::test::ScopedFeatureList scoped_feature_list_;

  DISALLOW_COPY_AND_ASSIGN(PasswordManagerDialogBrowserTest);
};

IN_PROC_BROWSER_TEST_P(PasswordManagerDialogBrowserTest, InvokeUi_normal) {
  ShowAndVerifyUi();
}

// Verify that password manager ignores passwords on forms injected into
// about:blank frames.  See https://crbug.com/756587.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       AboutBlankFramesAreIgnored) {
  // Start from a page without a password form.
  NavigateToFile("/password/other.html");

  // Add a blank iframe and then inject a password form into it.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  GURL submit_url(embedded_test_server()->GetURL("/password/done.html"));
  InjectBlankFrameWithPasswordForm(WebContents(), submit_url);
  content::RenderFrameHost* frame =
      ChildFrameAt(WebContents()->GetMainFrame(), 0);
  EXPECT_EQ(GURL(url::kAboutBlankURL), frame->GetLastCommittedURL());
  EXPECT_TRUE(frame->IsRenderFrameLive());
  EXPECT_FALSE(prompt_observer->IsSavePromptAvailable());

  // Fill in the password and submit the form.  This shouldn't bring up a save
  // password prompt and shouldn't result in a renderer kill.
  base::HistogramTester histogram_tester;
  SubmitInjectedPasswordForm(WebContents(), frame, submit_url);
  EXPECT_TRUE(frame->IsRenderFrameLive());
  EXPECT_EQ(submit_url, frame->GetLastCommittedURL());
  EXPECT_FALSE(prompt_observer->IsSavePromptAvailable());

  SubprocessMetricsProvider::MergeHistogramDeltasForTesting();
  histogram_tester.ExpectUniqueSample(
      "PasswordManager.AboutBlankPasswordSubmission",
      false /* is_main_frame */, 1);
}

// Verify that password manager ignores passwords on forms injected into
// about:blank popups.  See https://crbug.com/756587.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       AboutBlankPopupsAreIgnored) {
  // Start from a page without a password form.
  NavigateToFile("/password/other.html");

  // Open an about:blank popup and inject the password form into it.
  content::WindowedNotificationObserver popup_observer(
      chrome::NOTIFICATION_TAB_ADDED,
      content::NotificationService::AllSources());
  GURL submit_url(embedded_test_server()->GetURL("/password/done.html"));
  std::string form_html = GeneratePasswordFormForAction(submit_url);
  std::string open_blank_popup_with_password_form =
      "var w = window.open('about:blank');"
      "w.document.body.innerHTML = \"" + form_html + "\";";
  ASSERT_TRUE(content::ExecuteScript(WebContents(),
                                     open_blank_popup_with_password_form));
  popup_observer.Wait();
  ASSERT_EQ(2, browser()->tab_strip_model()->count());
  content::WebContents* newtab =
      browser()->tab_strip_model()->GetActiveWebContents();

  // Submit the password form and check that there was no renderer kill and no
  // prompt to save passwords.
  base::HistogramTester histogram_tester;
  std::unique_ptr<BubbleObserver> prompt_observer(new BubbleObserver(newtab));
  SubmitInjectedPasswordForm(newtab, newtab->GetMainFrame(), submit_url);
  EXPECT_FALSE(prompt_observer->IsSavePromptAvailable());
  EXPECT_TRUE(newtab->GetMainFrame()->IsRenderFrameLive());
  EXPECT_EQ(submit_url, newtab->GetMainFrame()->GetLastCommittedURL());

  SubprocessMetricsProvider::MergeHistogramDeltasForTesting();

  histogram_tester.ExpectUniqueSample(
      "PasswordManager.AboutBlankPasswordSubmission",
      true /* is_main_frame */, 1);
}

// Verify that previously saved passwords for about:blank frames are not used
// for autofill.  See https://crbug.com/756587.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       ExistingAboutBlankPasswordsAreNotUsed) {
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS).get());
  autofill::PasswordForm signin_form;
  signin_form.origin = GURL(url::kAboutBlankURL);
  signin_form.signon_realm = "about:";
  GURL submit_url(embedded_test_server()->GetURL("/password/done.html"));
  signin_form.action = submit_url;
  signin_form.password_value = base::ASCIIToUTF16("pa55w0rd");
  password_store->AddLogin(signin_form);

  // Start from a page without a password form.
  NavigateToFile("/password/other.html");

  // Inject an about:blank frame with password form.
  InjectBlankFrameWithPasswordForm(WebContents(), submit_url);
  content::RenderFrameHost* frame =
      ChildFrameAt(WebContents()->GetMainFrame(), 0);
  EXPECT_EQ(GURL(url::kAboutBlankURL), frame->GetLastCommittedURL());

  // Simulate user interaction in the iframe which normally triggers
  // autofill. Click in the middle of the frame to avoid the border.
  EXPECT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(),
      "var iframeRect = "
      "    document.getElementById('iframe').getBoundingClientRect();"));
  int x;
  EXPECT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send("
      "    parseInt((iframeRect.left + iframeRect.right) / 2));",
      &x));
  int y;
  EXPECT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractInt(
      RenderFrameHost(),
      "window.domAutomationController.send("
      "    parseInt((iframeRect.top + iframeRect.bottom) / 2));",
      &y));
  content::SimulateMouseClickAt(
      WebContents(), 0, blink::WebMouseEvent::Button::kLeft, gfx::Point(x, y));

  // Verify password is not autofilled.  Blink has a timer for 0.3 seconds
  // before it updates the browser with the new dynamic form, so wait long
  // enough for this timer to fire before checking the password.  Note that we
  // can't wait for any other events here, because when the test passes, there
  // should be no password manager IPCs sent from the renderer to browser.
  std::string empty_password;
  EXPECT_TRUE(content::ExecuteScriptWithoutUserGestureAndExtractString(
      frame,
      "setTimeout(function() {"
      "    domAutomationController.send("
      "        document.getElementById('password_field').value);"
      "}, 1000);",
      &empty_password));
  EXPECT_EQ("", empty_password);

  EXPECT_TRUE(frame->IsRenderFrameLive());
}

// Test params:
//  - bool popup_views_enabled: whether feature AutofillExpandedPopupViews
//        is enabled for testing.
class SitePerProcessPasswordManagerBrowserTest
    : public PasswordManagerBrowserTestBase,
      public ::testing::WithParamInterface<bool> {
 public:
  SitePerProcessPasswordManagerBrowserTest() {}

  void SetUpCommandLine(base::CommandLine* command_line) override {
    PasswordManagerBrowserTestBase::SetUpCommandLine(command_line);
    content::IsolateAllSitesForTesting(command_line);

    const bool popup_views_enabled = GetParam();
    scoped_feature_list_.InitWithFeatureState(
        autofill::kAutofillExpandedPopupViews, popup_views_enabled);
  }

 private:
  base::test::ScopedFeatureList scoped_feature_list_;

  DISALLOW_COPY_AND_ASSIGN(SitePerProcessPasswordManagerBrowserTest);
};

// Verify that there is no renderer kill when filling out a password on a
// subframe with a data: URL.
IN_PROC_BROWSER_TEST_P(SitePerProcessPasswordManagerBrowserTest,
                       NoRendererKillWithDataURLFrames) {
  // Start from a page without a password form.
  NavigateToFile("/password/other.html");

  // Add a iframe with a data URL that has a password form.
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  GURL submit_url(embedded_test_server()->GetURL("/password/done.html"));
  std::string form_html = GeneratePasswordFormForAction(submit_url);
  std::string inject_data_frame_with_password_form =
      "var frame = document.createElement('iframe');\n"
      "frame.src = \"data:text/html," + form_html + "\";\n"
      "document.body.appendChild(frame);\n";
  ASSERT_TRUE(content::ExecuteScript(WebContents(),
                                     inject_data_frame_with_password_form));
  content::WaitForLoadStop(WebContents());
  content::RenderFrameHost* frame =
      ChildFrameAt(WebContents()->GetMainFrame(), 0);
  EXPECT_TRUE(frame->GetLastCommittedURL().SchemeIs(url::kDataScheme));
  EXPECT_TRUE(frame->IsRenderFrameLive());
  EXPECT_FALSE(prompt_observer->IsSavePromptAvailable());

  // Fill in the password and submit the form.  This shouldn't bring up a save
  // password prompt and shouldn't result in a renderer kill.
  SubmitInjectedPasswordForm(WebContents(), frame, submit_url);
  EXPECT_TRUE(frame->IsRenderFrameLive());
  EXPECT_EQ(submit_url, frame->GetLastCommittedURL());
  EXPECT_FALSE(prompt_observer->IsSavePromptAvailable());
}

// Test that for HTTP auth (i.e., credentials not put through web forms) the
// password manager works even though it should be disabled on the previous
// page.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       CorrectEntryForHttpAuth) {
  // The embedded_test_server() is already started at this point and adding the
  // request handler to it would not be thread safe. Therefore, use a new
  // server.
  net::EmbeddedTestServer http_test_server;

  // Teach the embedded server to handle requests by issuing the basic auth
  // challenge.
  http_test_server.RegisterRequestHandler(base::Bind(&HandleTestAuthRequest));
  ASSERT_TRUE(http_test_server.Start());

  LoginPromptBrowserTestObserver login_observer;
  login_observer.Register(content::Source<content::NavigationController>(
      &WebContents()->GetController()));

  // Navigate to about:blank first. This is a page where password manager should
  // not work.
  ui_test_utils::NavigateToURL(browser(), GURL("about:blank"));

  // Navigate to a page requiring HTTP auth. Wait for the tab to get the correct
  // WebContents, but don't wait for navigation, which only finishes after
  // authentication.
  ui_test_utils::NavigateToURLWithDisposition(
      browser(), http_test_server.GetURL("/basic_auth"),
      WindowOpenDisposition::CURRENT_TAB, ui_test_utils::BROWSER_TEST_NONE);

  content::NavigationController* nav_controller =
      &WebContents()->GetController();
  NavigationObserver nav_observer(WebContents());
  WindowedAuthNeededObserver auth_needed_observer(nav_controller);
  auth_needed_observer.Wait();

  WindowedAuthSuppliedObserver auth_supplied_observer(nav_controller);
  // Offer valid credentials on the auth challenge.
  ASSERT_EQ(1u, login_observer.handlers().size());
  LoginHandler* handler = *login_observer.handlers().begin();
  ASSERT_TRUE(handler);
  // Any username/password will work.
  handler->SetAuth(base::UTF8ToUTF16("user"), base::UTF8ToUTF16("pwd"));
  auth_supplied_observer.Wait();

  // The password manager should be working correctly.
  nav_observer.Wait();
  WaitForPasswordStore();
  BubbleObserver bubble_observer(WebContents());
  EXPECT_TRUE(bubble_observer.IsSavePromptShownAutomatically());
}

// This test emulates what was observed in https://crbug.com/856543: Imagine the
// user stores a single username/password pair on origin A, and later submits a
// username-less password-reset form on origin B. In the bug, A and B were
// PSL-matches (different, but with the same eTLD+1), and Chrome ended up
// overwriting the old password with the new one. This test checks that update
// bubble is shown instead of silent update.
IN_PROC_BROWSER_TEST_P(PasswordManagerBrowserTestWithViewsFeature,
                       NoSilentOverwriteOnPSLMatch) {
  // Store a password at origin A.
  const GURL url_A = embedded_test_server()->GetURL("abc.foo.com", "/");
  scoped_refptr<password_manager::TestPasswordStore> password_store =
      static_cast<password_manager::TestPasswordStore*>(
          PasswordStoreFactory::GetForProfile(
              browser()->profile(), ServiceAccessType::IMPLICIT_ACCESS)
              .get());
  autofill::PasswordForm signin_form;
  signin_form.signon_realm = url_A.GetOrigin().spec();
  signin_form.origin = url_A;
  signin_form.username_value = base::ASCIIToUTF16("user");
  signin_form.password_value = base::ASCIIToUTF16("oldpassword");
  password_store->AddLogin(signin_form);
  WaitForPasswordStore();

  // Visit origin B with a form only containing new- and confirmation-password
  // fields.
  GURL url_B = embedded_test_server()->GetURL(
      "www.foo.com", "/password/new_password_form.html");
  NavigationObserver observer_B(WebContents());
  ui_test_utils::NavigateToURL(browser(), url_B);
  observer_B.Wait();

  // Fill in the new password and submit.
  GURL url_done =
      embedded_test_server()->GetURL("www.foo.com", "/password/done.html");
  NavigationObserver observer_done(WebContents());
  observer_done.SetPathToWaitFor("/password/done.html");
  ASSERT_TRUE(content::ExecuteScriptWithoutUserGesture(
      RenderFrameHost(),
      "document.getElementById('new_p').value = 'new password';"
      "document.getElementById('conf_p').value = 'new password';"
      "document.getElementById('testform').submit();"));
  observer_done.Wait();

  // Check that the password for origin A was not updated automatically and the
  // update bubble is shown instead.
  WaitForPasswordStore();  // Let the navigation take its effect on storing.
  CheckThatCredentialsStored("user", "oldpassword");
  std::unique_ptr<BubbleObserver> prompt_observer(
      new BubbleObserver(WebContents()));
  EXPECT_TRUE(prompt_observer->IsUpdatePromptShownAutomatically());

  // Check that the password is updated correctly if the user clicks Update.
  const autofill::PasswordForm& stored_form =
      password_store->stored_passwords().begin()->second[0];
  prompt_observer->AcceptUpdatePrompt(stored_form);

  WaitForPasswordStore();
  CheckThatCredentialsStored("user", "new password");
}

INSTANTIATE_TEST_CASE_P(All,
                        PasswordManagerBrowserTestWithViewsFeature,
                        /*popup_views_enabled=*/::testing::Bool());

INSTANTIATE_TEST_CASE_P(All,
                        PasswordManagerDialogBrowserTest,
                        /*popup_views_enabled=*/::testing::Bool());

INSTANTIATE_TEST_CASE_P(All,
                        SitePerProcessPasswordManagerBrowserTest,
                        /*popup_views_enabled=*/::testing::Bool());

}  // namespace password_manager
