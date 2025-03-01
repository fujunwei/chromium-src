// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_UI_SEARCH_SEARCH_IPC_ROUTER_POLICY_IMPL_H_
#define CHROME_BROWSER_UI_SEARCH_SEARCH_IPC_ROUTER_POLICY_IMPL_H_

#include "base/macros.h"
#include "build/build_config.h"
#include "chrome/browser/ui/search/search_ipc_router.h"

#if defined(OS_ANDROID)
#error "Instant is only used on desktop";
#endif

namespace content {
class WebContents;
}

// The SearchIPCRouter::Policy implementation.
class SearchIPCRouterPolicyImpl : public SearchIPCRouter::Policy {
 public:
  explicit SearchIPCRouterPolicyImpl(const content::WebContents* web_contents);
  ~SearchIPCRouterPolicyImpl() override;

 private:
  friend class SearchIPCRouterPolicyTest;

  // Overridden from SearchIPCRouter::Policy:
  bool ShouldProcessFocusOmnibox(bool is_active_tab) override;
  bool ShouldProcessDeleteMostVisitedItem() override;
  bool ShouldProcessUndoMostVisitedDeletion() override;
  bool ShouldProcessUndoAllMostVisitedDeletions() override;
  bool ShouldProcessAddCustomLink() override;
  bool ShouldProcessDeleteCustomLink() override;
  bool ShouldProcessUndoDeleteCustomLink() override;
  bool ShouldProcessResetCustomLinks() override;
  bool ShouldProcessLogEvent() override;
  bool ShouldProcessPasteIntoOmnibox(bool is_active_tab) override;
  bool ShouldProcessChromeIdentityCheck() override;
  bool ShouldProcessHistorySyncCheck() override;
  bool ShouldSendSetInputInProgress(bool is_active_tab) override;
  bool ShouldSendOmniboxFocusChanged() override;
  bool ShouldSendMostVisitedItems() override;
  bool ShouldSendThemeBackgroundInfo() override;
  bool ShouldProcessSetCustomBackgroundURL() override;
  bool ShouldProcessSetCustomBackgroundURLWithAttributions() override;
  bool ShouldProcessSelectLocalBackgroundImage() override;

  // Used by unit tests.
  void set_is_incognito(bool is_incognito) {
    is_incognito_ = is_incognito;
  }

  const content::WebContents* web_contents_;
  bool is_incognito_;

  DISALLOW_COPY_AND_ASSIGN(SearchIPCRouterPolicyImpl);
};

#endif  // CHROME_BROWSER_UI_SEARCH_SEARCH_IPC_ROUTER_POLICY_IMPL_H_
