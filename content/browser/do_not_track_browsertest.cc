// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "content/public/browser/content_browser_client.h"
#include "content/public/browser/web_contents.h"
#include "content/public/common/renderer_preferences.h"
#include "content/public/test/browser_test_utils.h"
#include "content/public/test/content_browser_test.h"
#include "content/public/test/content_browser_test_utils.h"
#include "content/shell/browser/shell.h"
#include "net/test/embedded_test_server/embedded_test_server.h"
#include "net/test/embedded_test_server/http_request.h"
#include "net/test/embedded_test_server/http_response.h"

#if defined(OS_ANDROID)
#include "base/sys_info.h"
#endif

namespace content {

namespace {

class MockContentBrowserClient final : public ContentBrowserClient {
 public:
  void UpdateRendererPreferencesForWorker(BrowserContext*,
                                          RendererPreferences* prefs) override {
    prefs->enable_do_not_track = true;
    prefs->enable_referrers = true;
  }
};

class DoNotTrackTest : public ContentBrowserTest {
 protected:
  void TearDownOnMainThread() override {
    if (original_client_)
      SetBrowserClientForTesting(original_client_);
  }

  // Returns false if we cannot enable do not track. It happens only when
  // Android Kitkat or older systems.
  // TODO(crbug.com/864403): It seems that we call unsupported Android APIs on
  // KitKat when we set a ContentBrowserClient. Don't call such APIs and make
  // this test available on KitKat.
  bool EnableDoNotTrack() {
#if defined(OS_ANDROID)
    int32_t major_version = 0, minor_version = 0, bugfix_version = 0;
    base::SysInfo::OperatingSystemVersionNumbers(&major_version, &minor_version,
                                                 &bugfix_version);
    if (major_version < 5)
      return false;
#endif
    original_client_ = SetBrowserClientForTesting(&client_);
    RendererPreferences* prefs =
        shell()->web_contents()->GetMutableRendererPrefs();
    EXPECT_FALSE(prefs->enable_do_not_track);
    prefs->enable_do_not_track = true;
    return true;
  }

  void ExpectPageTextEq(const std::string& expected_content) {
    std::string text;
    ASSERT_TRUE(ExecuteScriptAndExtractString(
        shell(),
        "window.domAutomationController.send(document.body.innerText);",
        &text));
    EXPECT_EQ(expected_content, text);
  }

  std::string GetDOMDoNotTrackProperty() {
    std::string value;
    EXPECT_TRUE(ExecuteScriptAndExtractString(
        shell(),
        "window.domAutomationController.send("
        "    navigator.doNotTrack === null ? '' : navigator.doNotTrack)",
        &value));
    return value;
  }

 private:
  ContentBrowserClient* original_client_ = nullptr;
  MockContentBrowserClient client_;
};

std::unique_ptr<net::test_server::HttpResponse> CaptureHeaderHandler(
    const std::string& path,
    net::test_server::HttpRequest::HeaderMap* header_map,
    base::OnceClosure done_callback,
    const net::test_server::HttpRequest& request) {
  GURL request_url = request.GetURL();
  if (request_url.path() != path)
    return nullptr;

  *header_map = request.headers;
  std::move(done_callback).Run();
  return std::make_unique<net::test_server::BasicHttpResponse>();
}

// Checks that the DNT header is not sent by default.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, NotEnabled) {
  ASSERT_TRUE(embedded_test_server()->Start());
  GURL url = embedded_test_server()->GetURL("/echoheader?DNT");
  EXPECT_TRUE(NavigateToURL(shell(), url));
  ExpectPageTextEq("None");
  // And the DOM property is not set.
  EXPECT_EQ("", GetDOMDoNotTrackProperty());
}

// Checks that the DNT header is sent when the corresponding preference is set.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, Simple) {
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;
  GURL url = embedded_test_server()->GetURL("/echoheader?DNT");
  EXPECT_TRUE(NavigateToURL(shell(), url));
  ExpectPageTextEq("1");
}

// Checks that the DNT header is preserved during redirects.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, Redirect) {
  ASSERT_TRUE(embedded_test_server()->Start());
  GURL final_url = embedded_test_server()->GetURL("/echoheader?DNT");
  GURL url = embedded_test_server()->GetURL(std::string("/server-redirect?") +
                                            final_url.spec());
  if (!EnableDoNotTrack())
    return;
  // We don't check the result NavigateToURL as it returns true only if the
  // final URL is equal to the passed URL.
  NavigateToURL(shell(), url);
  ExpectPageTextEq("1");
}

// Checks that the DOM property is set when the corresponding preference is set.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, DOMProperty) {
  ASSERT_TRUE(embedded_test_server()->Start());
  GURL url = embedded_test_server()->GetURL("/echo");
  if (!EnableDoNotTrack())
    return;
  EXPECT_TRUE(NavigateToURL(shell(), url));
  EXPECT_EQ("1", GetDOMDoNotTrackProperty());
}

// Checks that the DNT header is sent in a request for a dedicated worker
// script.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, Worker) {
  net::test_server::HttpRequest::HeaderMap header_map;
  base::RunLoop loop;
  embedded_test_server()->RegisterRequestHandler(base::BindRepeating(
      &CaptureHeaderHandler, "/capture", &header_map, loop.QuitClosure()));
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;
  const GURL url = embedded_test_server()->GetURL(
      std::string("/workers/create_worker.html?worker_url=/capture"));
  NavigateToURL(shell(), url);
  loop.Run();

  EXPECT_TRUE(header_map.find("DNT") != header_map.end());
  EXPECT_EQ("1", header_map["DNT"]);
}

// Checks that the DNT header is sent in a request for a shared worker
// script.
// Disabled due to crbug.com/853085.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, DISABLED_SharedWorker) {
  net::test_server::HttpRequest::HeaderMap header_map;
  base::RunLoop loop;
  embedded_test_server()->RegisterRequestHandler(base::BindRepeating(
      &CaptureHeaderHandler, "/capture", &header_map, loop.QuitClosure()));
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;
  const GURL url = embedded_test_server()->GetURL(
      std::string("/workers/create_shared_worker.html?worker_url=/capture"));
  NavigateToURL(shell(), url);
  loop.Run();

  EXPECT_TRUE(header_map.find("DNT") != header_map.end());
  EXPECT_EQ("1", header_map["DNT"]);
}

// Checks that the DNT header is sent in a request for a service worker
// script.
// Disabled due to crbug.com/853085.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, DISABLED_ServiceWorker) {
  net::test_server::HttpRequest::HeaderMap header_map;
  base::RunLoop loop;
  embedded_test_server()->RegisterRequestHandler(base::BindRepeating(
      &CaptureHeaderHandler, "/capture", &header_map, loop.QuitClosure()));
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;
  const GURL url = embedded_test_server()->GetURL(std::string(
      "/service_worker/create_service_worker.html?worker_url=/capture"));
  NavigateToURL(shell(), url);
  loop.Run();

  EXPECT_TRUE(header_map.find("DNT") != header_map.end());
  EXPECT_EQ("1", header_map["DNT"]);
}

// Checks that the DNT header is preserved when fetching from a dedicated
// worker.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, FetchFromWorker) {
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;
  const GURL fetch_url = embedded_test_server()->GetURL("/echoheader?DNT");
  const GURL url = embedded_test_server()->GetURL(
      std::string("/workers/fetch_from_worker.html?url=") + fetch_url.spec());
  NavigateToURL(shell(), url);

  const base::string16 title = base::ASCIIToUTF16("DONE");
  TitleWatcher watcher(shell()->web_contents(), title);
  EXPECT_EQ(title, watcher.WaitAndGetTitle());

  ExpectPageTextEq("1");
}

// Checks that the DNT header is preserved when fetching from a shared worker.
//
// Disabled on Android since a shared worker is not available on Android.
#if defined(OS_ANDROID)
#define MAYBE_FetchFromSharedWorker DISABLED_FetchFromSharedWorker
#else
#define MAYBE_FetchFromSharedWorker FetchFromSharedWorker
#endif
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, MAYBE_FetchFromSharedWorker) {
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;
  const GURL fetch_url = embedded_test_server()->GetURL("/echoheader?DNT");
  const GURL url = embedded_test_server()->GetURL(
      std::string("/workers/fetch_from_shared_worker.html?url=") +
      fetch_url.spec());
  NavigateToURL(shell(), url);

  const base::string16 title = base::ASCIIToUTF16("DONE");
  TitleWatcher watcher(shell()->web_contents(), title);
  EXPECT_EQ(title, watcher.WaitAndGetTitle());

  ExpectPageTextEq("1");
}

// Checks that the DNT header is preserved when fetching from a service worker.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest, FetchFromServiceWorker) {
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;
  const GURL fetch_url = embedded_test_server()->GetURL("/echoheader?DNT");
  const GURL url = embedded_test_server()->GetURL(
      std::string("/service_worker/fetch_from_service_worker.html?url=") +
      fetch_url.spec());
  NavigateToURL(shell(), url);

  const base::string16 title = base::ASCIIToUTF16("DONE");
  TitleWatcher watcher(shell()->web_contents(), title);
  EXPECT_EQ(title, watcher.WaitAndGetTitle());

  ExpectPageTextEq("1");
}

// Checks that the DNT header is preserved when fetching from a page controlled
// by a service worker which doesn't have a fetch handler and falls back to the
// network.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest,
                       FetchFromServiceWorkerControlledPage_NoFetchHandler) {
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;

  {
    // Register a service worker which controls /service_worker.
    const GURL url = embedded_test_server()->GetURL(
        "/service_worker/create_service_worker.html?"
        "worker_url=/service_worker/empty.js");
    EXPECT_TRUE(NavigateToURL(shell(), url));
    const base::string16 title = base::ASCIIToUTF16("DONE");
    TitleWatcher watcher(shell()->web_contents(), title);
    EXPECT_EQ(title, watcher.WaitAndGetTitle());
  }

  {
    // Issue a request from a controlled page.
    const GURL url = embedded_test_server()->GetURL(
        "/service_worker/fetch_from_page.html?url=/echoheader?DNT");
    EXPECT_TRUE(NavigateToURL(shell(), url));
    const base::string16 title = base::ASCIIToUTF16("DONE");
    TitleWatcher watcher(shell()->web_contents(), title);
    EXPECT_EQ(title, watcher.WaitAndGetTitle());
  }

  ExpectPageTextEq("1");
}

// Checks that the DNT header is preserved when fetching from a page controlled
// by a service worker which has a fetch handler but falls back to the network.
IN_PROC_BROWSER_TEST_F(DoNotTrackTest,
                       FetchFromServiceWorkerControlledPage_PassThrough) {
  ASSERT_TRUE(embedded_test_server()->Start());
  if (!EnableDoNotTrack())
    return;

  {
    // Register a service worker which controls /service_worker.
    const GURL url = embedded_test_server()->GetURL(
        "/service_worker/create_service_worker.html?"
        "worker_url=/service_worker/fetch_event_pass_through.js");
    EXPECT_TRUE(NavigateToURL(shell(), url));
    const base::string16 title = base::ASCIIToUTF16("DONE");
    TitleWatcher watcher(shell()->web_contents(), title);
    EXPECT_EQ(title, watcher.WaitAndGetTitle());
  }

  {
    // Issue a request from a controlled page.
    const GURL url = embedded_test_server()->GetURL(
        "/service_worker/fetch_from_page.html?url=/echoheader?DNT");
    EXPECT_TRUE(NavigateToURL(shell(), url));
    const base::string16 title = base::ASCIIToUTF16("DONE");
    TitleWatcher watcher(shell()->web_contents(), title);
    EXPECT_EQ(title, watcher.WaitAndGetTitle());
  }

  ExpectPageTextEq("1");
}

}  // namespace

}  // namespace content
