// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/ui/app_list/internal_app/internal_app_metadata.h"

#include "ash/public/cpp/app_list/internal_app_id_constants.h"
#include "ash/public/cpp/resources/grit/ash_public_unscaled_resources.h"
#include "base/logging.h"
#include "base/no_destructor.h"
#include "base/strings/string16.h"
#include "base/strings/string_util.h"
#include "base/time/time.h"
#include "chrome/browser/chromeos/arc/arc_util.h"
#include "chrome/browser/profiles/profile.h"
#include "chrome/browser/sync/profile_sync_service_factory.h"
#include "chrome/browser/ui/app_list/app_list_client_impl.h"
#include "chrome/browser/ui/app_list/arc/arc_app_utils.h"
#include "chrome/browser/ui/ash/ksv/keyboard_shortcut_viewer_util.h"
#include "chrome/browser/ui/chrome_pages.h"
#include "chrome/browser/ui/extensions/app_launch_params.h"
#include "chrome/browser/ui/extensions/application_launch.h"
#include "chrome/common/pref_names.h"
#include "chrome/grit/chrome_unscaled_resources.h"
#include "chrome/grit/generated_resources.h"
#include "components/crx_file/id_util.h"
#include "components/prefs/pref_service.h"
#include "components/sessions/core/serialized_navigation_entry.h"
#include "components/sync/driver/sync_service.h"
#include "components/sync/protocol/sync_enums.pb.h"
#include "components/sync_sessions/open_tabs_ui_delegate.h"
#include "components/sync_sessions/synced_session.h"
#include "extensions/browser/extension_registry.h"
#include "extensions/common/extension.h"
#include "extensions/common/extension_urls.h"
#include "net/base/url_util.h"
#include "ui/base/l10n/l10n_util.h"
#include "ui/base/page_transition_types.h"
#include "ui/base/resource/resource_bundle.h"
#include "ui/gfx/image/image_skia_operations.h"
#include "url/gurl.h"

namespace app_list {

namespace {
constexpr char kChromeCameraAppId[] = "hfhhnacclhffhdffklopdkcgdhifgngh";
constexpr char kAndroidCameraAppId[] = "goamfaniemdfcajgcmmflhchgkmbngka";
}  // namespace

const std::vector<InternalApp>& GetInternalAppList() {
  static const base::NoDestructor<std::vector<InternalApp>> internal_app_list(
      {{kInternalAppIdKeyboardShortcutViewer,
        IDS_INTERNAL_APP_KEYBOARD_SHORTCUT_VIEWER, IDR_SHORTCUT_VIEWER_LOGO_192,
        /*recommendable=*/false,
        /*searchable=*/true,
        /*show_in_launcher=*/false,
        IDS_LAUNCHER_SEARCHABLE_KEYBOARD_SHORTCUT_VIEWER},

       {kInternalAppIdSettings, IDS_INTERNAL_APP_SETTINGS,
        IDR_SETTINGS_LOGO_192,
        /*recommendable=*/true,
        /*searchable=*/true,
        /*show_in_launcher=*/true,
        /*searchable_string_resource_id=*/0},

       {kInternalAppIdContinueReading, IDS_INTERNAL_APP_CONTINUOUS_READING,
        IDR_PRODUCT_LOGO_256,
        /*recommendable=*/true,
        /*searchable=*/false,
        /*show_in_launcher=*/false,
        /*searchable_string_resource_id=*/0},

       {kInternalAppIdCamera, IDS_INTERNAL_APP_CAMERA, IDR_CAMERA_LOGO_192,
        /*recommendable=*/true,
        /*show_in_launcher=*/true, IDS_INTERNAL_APP_CAMERA}});
  return *internal_app_list;
}

const InternalApp* FindInternalApp(const std::string& app_id) {
  for (const auto& app : GetInternalAppList()) {
    if (app_id == app.app_id)
      return &app;
  }
  return nullptr;
}

bool IsInternalApp(const std::string& app_id) {
  return !!FindInternalApp(app_id);
}

base::string16 GetInternalAppNameById(const std::string& app_id) {
  const auto* app = FindInternalApp(app_id);
  return app ? l10n_util::GetStringUTF16(app->name_string_resource_id)
             : base::string16();
}

int GetIconResourceIdByAppId(const std::string& app_id) {
  const auto* app = FindInternalApp(app_id);
  return app ? app->icon_resource_id : 0;
}

void ShowCameraApp(const std::string& app_id,
                   Profile* profile,
                   int event_flags) {
  std::string chrome_app_id(kChromeCameraAppId);
  const extensions::ExtensionRegistry* registry =
      extensions::ExtensionRegistry::Get(profile);
  const extensions::Extension* extension =
      registry->GetInstalledExtension(chrome_app_id);

  bool arc_enabled = arc::IsArcPlayStoreEnabledForProfile(profile);
  bool media_consolidated =
      profile->GetPrefs()->GetBoolean(prefs::kCameraMediaConsolidated);
  AppListClientImpl* controller = AppListClientImpl::GetInstance();
  if (arc_enabled && (!extension || media_consolidated)) {
    // Open ARC++ camera app.
    arc::LaunchApp(profile, kAndroidCameraAppId, event_flags,
                   arc::UserInteractionType::APP_STARTED_FROM_LAUNCHER,
                   controller->GetAppListDisplayId());
  } else if (extension) {
    // Open Chrome camera app.
    AppLaunchParams params = CreateAppLaunchParamsWithEventFlags(
        profile, extension, event_flags, extensions::SOURCE_APP_LAUNCHER,
        controller->GetAppListDisplayId());
    params.launch_id = ash::ShelfID(extension->id()).launch_id;
    OpenApplication(params);
  } else {
    // Open Chrome camera app detail page in CWS.
    const GURL store_url = net::AppendQueryParameter(
        GURL(extension_urls::GetWebstoreItemDetailURLPrefix() + chrome_app_id),
        extension_urls::kWebstoreSourceField,
        extension_urls::kLaunchSourceAppListSearch);
    controller->OpenURL(profile, store_url, ui::PAGE_TRANSITION_LINK,
                        ui::DispositionFromEventFlags(event_flags));
  }
}

void OpenInternalApp(const std::string& app_id,
                     Profile* profile,
                     int event_flags) {
  if (app_id == kInternalAppIdKeyboardShortcutViewer) {
    keyboard_shortcut_viewer_util::ShowKeyboardShortcutViewer();
  } else if (app_id == kInternalAppIdSettings) {
    chrome::ShowSettingsSubPageForProfile(profile, std::string());
  } else if (app_id == kInternalAppIdCamera) {
    ShowCameraApp(app_id, profile, event_flags);
  }
}

gfx::ImageSkia GetIconForResourceId(int resource_id, int resource_size_in_dip) {
  if (resource_id == 0)
    return gfx::ImageSkia();

  gfx::ImageSkia* source =
      ui::ResourceBundle::GetSharedInstance().GetImageSkiaNamed(resource_id);
  return gfx::ImageSkiaOperations::CreateResizedImage(
      *source, skia::ImageOperations::RESIZE_BEST,
      gfx::Size(resource_size_in_dip, resource_size_in_dip));
}

bool HasRecommendableForeignTab(Profile* profile,
                                base::string16* title,
                                GURL* url) {
  syncer::SyncService* sync_service =
      ProfileSyncServiceFactory::GetSyncServiceForBrowserContext(profile);
  std::vector<const sync_sessions::SyncedSession*> foreign_sessions;
  sync_sessions::OpenTabsUIDelegate* delegate =
      sync_service->GetOpenTabsUIDelegate();
  if (delegate != nullptr)
    delegate->GetAllForeignSessions(&foreign_sessions);

  constexpr int kMaxForeignTabAgeInMinutes = 120;
  base::Time latest_timestamp;
  bool has_recommendation = false;
  for (const sync_sessions::SyncedSession* session : foreign_sessions) {
    if (latest_timestamp > session->modified_time)
      continue;

    auto device_type = session->device_type;
    if (device_type != sync_pb::SyncEnums::TYPE_PHONE &&
        device_type != sync_pb::SyncEnums::TYPE_TABLET) {
      continue;
    }

    for (const auto& key_value : session->windows) {
      for (const std::unique_ptr<sessions::SessionTab>& tab :
           key_value.second->wrapped_window.tabs) {
        if (tab->navigations.empty())
          continue;

        const sessions::SerializedNavigationEntry& navigation =
            tab->navigations.back();
        const GURL& virtual_url = navigation.virtual_url();

        // Only show pages with http or https.
        if (!virtual_url.SchemeIsHTTPOrHTTPS())
          continue;

        // Only show pages recently opened.
        const base::TimeDelta tab_age = base::Time::Now() - tab->timestamp;
        if (tab_age > base::TimeDelta::FromMinutes(kMaxForeignTabAgeInMinutes))
          continue;

        if (latest_timestamp < tab->timestamp) {
          has_recommendation = true;
          latest_timestamp = tab->timestamp;
          if (title)
            *title = navigation.title();

          if (url)
            *url = virtual_url;
        }
      }
    }
  }
  return has_recommendation;
}

size_t GetNumberOfInternalAppsShowInLauncherForTest(std::string* apps_name) {
  size_t num_of_internal_apps_show_in_launcher = 0u;
  std::vector<std::string> internal_apps_name;
  for (const auto& app : GetInternalAppList()) {
    if (app.show_in_launcher) {
      ++num_of_internal_apps_show_in_launcher;
      if (apps_name) {
        internal_apps_name.emplace_back(
            l10n_util::GetStringUTF8(app.name_string_resource_id));
      }
    }
  }
  if (apps_name)
    *apps_name = base::JoinString(internal_apps_name, ",");
  return num_of_internal_apps_show_in_launcher;
}

}  // namespace app_list
