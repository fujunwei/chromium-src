// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/web_applications/extensions/bookmark_app_data_retriever.h"

#include <utility>

#include "base/strings/utf_string_conversions.h"
#include "chrome/test/base/chrome_render_view_host_test_harness.h"
#include "chrome/test/base/testing_profile.h"
#include "content/public/browser/navigation_entry.h"
#include "content/public/browser/site_instance.h"
#include "content/public/test/test_browser_thread_bundle.h"
#include "content/public/test/web_contents_tester.h"
#include "mojo/public/cpp/bindings/associated_binding.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/blink/public/common/associated_interfaces/associated_interface_provider.h"

namespace extensions {

namespace {

const char kFooUrl[] = "https://foo.example";
const char kFooUrl2[] = "https://foo.example/bar";
const char kFooTitle[] = "Foo Title";
const char kBarUrl[] = "https://bar.example";

}  // namespace

class FakeChromeRenderFrame
    : public chrome::mojom::ChromeRenderFrameInterceptorForTesting {
 public:
  explicit FakeChromeRenderFrame(const WebApplicationInfo& web_app_info)
      : web_app_info_(web_app_info) {}
  ~FakeChromeRenderFrame() override = default;

  ChromeRenderFrame* GetForwardingInterface() override {
    NOTREACHED();
    return nullptr;
  }

  void Bind(mojo::ScopedInterfaceEndpointHandle handle) {
    binding_.Bind(
        mojo::AssociatedInterfaceRequest<ChromeRenderFrame>(std::move(handle)));
  }

  void GetWebApplicationInfo(
      const GetWebApplicationInfoCallback& callback) override {
    callback.Run(web_app_info_);
  }

 private:
  WebApplicationInfo web_app_info_;

  mojo::AssociatedBinding<chrome::mojom::ChromeRenderFrame> binding_{this};
};

class BookmarkAppDataRetrieverTest : public ChromeRenderViewHostTestHarness {
 public:
  BookmarkAppDataRetrieverTest() = default;
  ~BookmarkAppDataRetrieverTest() override = default;

  void SetFakeChromeRenderFrame(
      FakeChromeRenderFrame* fake_chrome_render_frame) {
    web_contents()
        ->GetMainFrame()
        ->GetRemoteAssociatedInterfaces()
        ->OverrideBinderForTesting(
            chrome::mojom::ChromeRenderFrame::Name_,
            base::BindRepeating(&FakeChromeRenderFrame::Bind,
                                base::Unretained(fake_chrome_render_frame)));
  }

  void GetWebApplicationInfoCallback(
      base::OnceClosure quit_closure,
      base::Optional<WebApplicationInfo> web_app_info) {
    web_app_info_ = web_app_info;
    std::move(quit_closure).Run();
  }

 protected:
  content::WebContentsTester* web_contents_tester() {
    return content::WebContentsTester::For(web_contents());
  }

  const base::Optional<WebApplicationInfo>& web_app_info() {
    return web_app_info_.value();
  }

 private:
  base::Optional<base::Optional<WebApplicationInfo>> web_app_info_;

  DISALLOW_COPY_AND_ASSIGN(BookmarkAppDataRetrieverTest);
};

TEST_F(BookmarkAppDataRetrieverTest, GetWebApplicationInfo_NoEntry) {
  base::RunLoop run_loop;
  BookmarkAppDataRetriever retriever;
  retriever.GetWebApplicationInfo(
      web_contents(),
      base::BindOnce(
          &BookmarkAppDataRetrieverTest::GetWebApplicationInfoCallback,
          base::Unretained(this), run_loop.QuitClosure()));
  run_loop.Run();

  EXPECT_EQ(base::nullopt, web_app_info());
}

TEST_F(BookmarkAppDataRetrieverTest, GetWebApplicationInfo_AppUrlAbsent) {
  web_contents_tester()->NavigateAndCommit(GURL(kFooUrl));

  WebApplicationInfo original_web_app_info;
  original_web_app_info.app_url = GURL();

  FakeChromeRenderFrame fake_chrome_render_frame(original_web_app_info);
  SetFakeChromeRenderFrame(&fake_chrome_render_frame);

  base::RunLoop run_loop;
  BookmarkAppDataRetriever retriever;
  retriever.GetWebApplicationInfo(
      web_contents(),
      base::BindOnce(
          &BookmarkAppDataRetrieverTest::GetWebApplicationInfoCallback,
          base::Unretained(this), run_loop.QuitClosure()));
  run_loop.Run();

  // If the WebApplicationInfo has no URL, we fallback to the last committed
  // URL.
  EXPECT_EQ(GURL(kFooUrl), web_app_info()->app_url);
}

TEST_F(BookmarkAppDataRetrieverTest, GetWebApplicationInfo_AppUrlPresent) {
  web_contents_tester()->NavigateAndCommit(GURL(kFooUrl));

  WebApplicationInfo original_web_app_info;
  original_web_app_info.app_url = GURL(kBarUrl);

  FakeChromeRenderFrame fake_chrome_render_frame(original_web_app_info);
  SetFakeChromeRenderFrame(&fake_chrome_render_frame);

  base::RunLoop run_loop;
  BookmarkAppDataRetriever retriever;
  retriever.GetWebApplicationInfo(
      web_contents(),
      base::BindOnce(
          &BookmarkAppDataRetrieverTest::GetWebApplicationInfoCallback,
          base::Unretained(this), run_loop.QuitClosure()));
  run_loop.Run();

  EXPECT_EQ(original_web_app_info.app_url, web_app_info()->app_url);
}

TEST_F(BookmarkAppDataRetrieverTest,
       GetWebApplicationInfo_TitleAbsentFromRenderer) {
  web_contents_tester()->NavigateAndCommit(GURL(kFooUrl));

  const auto web_contents_title = base::UTF8ToUTF16(kFooTitle);
  web_contents_tester()->SetTitle(web_contents_title);

  WebApplicationInfo original_web_app_info;
  original_web_app_info.title = base::UTF8ToUTF16("");

  FakeChromeRenderFrame fake_chrome_render_frame(original_web_app_info);
  SetFakeChromeRenderFrame(&fake_chrome_render_frame);

  base::RunLoop run_loop;
  BookmarkAppDataRetriever retriever;
  retriever.GetWebApplicationInfo(
      web_contents(),
      base::BindOnce(
          &BookmarkAppDataRetrieverTest::GetWebApplicationInfoCallback,
          base::Unretained(this), run_loop.QuitClosure()));
  run_loop.Run();

  // If the WebApplicationInfo has no title, we fallback to the WebContents
  // title.
  EXPECT_EQ(web_contents_title, web_app_info()->title);
}

TEST_F(BookmarkAppDataRetrieverTest,
       GetWebApplicationInfo_TitleAbsentFromWebContents) {
  web_contents_tester()->NavigateAndCommit(GURL(kFooUrl));

  web_contents_tester()->SetTitle(base::UTF8ToUTF16(""));

  WebApplicationInfo original_web_app_info;
  original_web_app_info.title = base::UTF8ToUTF16("");

  FakeChromeRenderFrame fake_chrome_render_frame(original_web_app_info);
  SetFakeChromeRenderFrame(&fake_chrome_render_frame);

  base::RunLoop run_loop;
  BookmarkAppDataRetriever retriever;
  retriever.GetWebApplicationInfo(
      web_contents(),
      base::BindOnce(
          &BookmarkAppDataRetrieverTest::GetWebApplicationInfoCallback,
          base::Unretained(this), run_loop.QuitClosure()));
  run_loop.Run();

  // If the WebApplicationInfo has no title and the WebContents has no title,
  // we fallback to app_url.
  EXPECT_EQ(base::UTF8ToUTF16(web_app_info()->app_url.spec()),
            web_app_info()->title);
}

TEST_F(BookmarkAppDataRetrieverTest,
       GetWebApplicationInfo_WebContentsDestroyed) {
  web_contents_tester()->NavigateAndCommit(GURL(kFooUrl));

  FakeChromeRenderFrame fake_chrome_render_frame{WebApplicationInfo()};
  SetFakeChromeRenderFrame(&fake_chrome_render_frame);

  base::RunLoop run_loop;
  BookmarkAppDataRetriever retriever;
  retriever.GetWebApplicationInfo(
      web_contents(),
      base::BindOnce(
          &BookmarkAppDataRetrieverTest::GetWebApplicationInfoCallback,
          base::Unretained(this), run_loop.QuitClosure()));
  DeleteContents();
  run_loop.Run();

  EXPECT_EQ(base::nullopt, web_app_info());
}

TEST_F(BookmarkAppDataRetrieverTest, GetWebApplicationInfo_FrameNavigated) {
  web_contents_tester()->NavigateAndCommit(GURL(kFooUrl));

  FakeChromeRenderFrame fake_chrome_render_frame{WebApplicationInfo()};
  SetFakeChromeRenderFrame(&fake_chrome_render_frame);

  base::RunLoop run_loop;
  BookmarkAppDataRetriever retriever;
  retriever.GetWebApplicationInfo(
      web_contents(),
      base::BindOnce(
          &BookmarkAppDataRetrieverTest::GetWebApplicationInfoCallback,
          base::Unretained(this), run_loop.QuitClosure()));
  web_contents_tester()->NavigateAndCommit(GURL(kFooUrl2));
  run_loop.Run();

  EXPECT_EQ(base::nullopt, web_app_info());
}

}  // namespace extensions
