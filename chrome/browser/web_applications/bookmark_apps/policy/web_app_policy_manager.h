// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_WEB_APPLICATIONS_BOOKMARK_APPS_POLICY_WEB_APP_POLICY_MANAGER_H_
#define CHROME_BROWSER_WEB_APPLICATIONS_BOOKMARK_APPS_POLICY_WEB_APP_POLICY_MANAGER_H_

#include <memory>
#include <vector>

#include "base/macros.h"
#include "chrome/browser/web_applications/components/pending_app_manager.h"
#include "url/gurl.h"

class PrefService;

namespace web_app {

// Tracks the policy that affects Web Apps and also tracks which Web Apps are
// currently installed based on this policy. Based on these, it decides which
// apps to install, uninstall, and update, via a PendingAppManager.
class WebAppPolicyManager {
 public:
  // Constructs a WebAppPolicyManager instance that uses
  // extensions::PendingBookmarkAppManager to manage apps.
  explicit WebAppPolicyManager(PrefService* pref_service);

  // Constructs a WebAppPolicyManager instance that uses |pending_app_manager|
  // to manage apps.
  explicit WebAppPolicyManager(
      PrefService* pref_service,
      std::unique_ptr<PendingAppManager> pending_app_manager);

  ~WebAppPolicyManager();

  const PendingAppManager& pending_app_manager() {
    return *pending_app_manager_;
  }

 private:
  std::vector<PendingAppManager::AppInfo> GetAppsToInstall();

  PrefService* pref_service_;
  std::unique_ptr<PendingAppManager> pending_app_manager_;

  DISALLOW_COPY_AND_ASSIGN(WebAppPolicyManager);
};

}  // namespace web_app

#endif  // CHROME_BROWSER_WEB_APPLICATIONS_BOOKMARK_APPS_POLICY_WEB_APP_POLICY_MANAGER_H_
