// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/web_applications/extensions/bookmark_app_data_retriever.h"

#include <utility>

#include "base/bind.h"
#include "base/callback.h"
#include "base/strings/utf_string_conversions.h"
#include "chrome/common/chrome_render_frame.mojom.h"
#include "chrome/common/web_application_info.h"
#include "content/public/browser/navigation_entry.h"
#include "content/public/browser/render_frame_host.h"
#include "content/public/browser/web_contents.h"
#include "third_party/blink/public/common/associated_interfaces/associated_interface_provider.h"

namespace extensions {

BookmarkAppDataRetriever::BookmarkAppDataRetriever() = default;

BookmarkAppDataRetriever::~BookmarkAppDataRetriever() = default;

void BookmarkAppDataRetriever::GetWebApplicationInfo(
    content::WebContents* web_contents,
    GetWebApplicationInfoCallback callback) {
  // Concurrent calls are not allowed.
  CHECK(!get_web_app_info_callback_);
  get_web_app_info_callback_ = std::move(callback);

  content::NavigationEntry* entry =
      web_contents->GetController().GetLastCommittedEntry();
  if (!entry) {
    base::ThreadTaskRunnerHandle::Get()->PostTask(
        FROM_HERE,
        base::BindOnce(std::move(get_web_app_info_callback_), base::nullopt));
    return;
  }

  chrome::mojom::ChromeRenderFrameAssociatedPtr chrome_render_frame;
  web_contents->GetMainFrame()->GetRemoteAssociatedInterfaces()->GetInterface(
      &chrome_render_frame);

  // Set the error handler so that we can run |get_web_app_info_callback_| if
  // the WebContents or the RenderFrameHost are destroyed and the connection
  // to ChromeRenderFrame is lost.
  chrome_render_frame.set_connection_error_handler(
      base::BindOnce(&BookmarkAppDataRetriever::OnGetWebApplicationInfoFailed,
                     weak_ptr_factory_.GetWeakPtr()));
  // Bind the InterfacePtr into the callback so that it's kept alive
  // until there's either a connection error or a response.
  auto* web_app_info_proxy = chrome_render_frame.get();
  web_app_info_proxy->GetWebApplicationInfo(base::Bind(
      &BookmarkAppDataRetriever::OnGetWebApplicationInfo,
      weak_ptr_factory_.GetWeakPtr(), base::Passed(&chrome_render_frame),
      web_contents, entry->GetUniqueID()));
}

void BookmarkAppDataRetriever::OnGetWebApplicationInfo(
    chrome::mojom::ChromeRenderFrameAssociatedPtr chrome_render_frame,
    content::WebContents* web_contents,
    int last_committed_nav_entry_unique_id,
    const WebApplicationInfo& web_app_info) {
  content::NavigationEntry* entry =
      web_contents->GetController().GetLastCommittedEntry();
  if (!entry || last_committed_nav_entry_unique_id != entry->GetUniqueID()) {
    std::move(get_web_app_info_callback_).Run(base::nullopt);
    return;
  }

  base::Optional<WebApplicationInfo> info(web_app_info);
  if (info->app_url.is_empty())
    info->app_url = web_contents->GetLastCommittedURL();

  if (info->title.empty())
    info->title = web_contents->GetTitle();
  if (info->title.empty())
    info->title = base::UTF8ToUTF16(info->app_url.spec());

  std::move(get_web_app_info_callback_).Run(std::move(info));
}

void BookmarkAppDataRetriever::OnGetWebApplicationInfoFailed() {
  std::move(get_web_app_info_callback_).Run(base::nullopt);
}

}  // namespace extensions
