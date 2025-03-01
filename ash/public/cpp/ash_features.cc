// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ash/public/cpp/ash_features.h"

#include "ash/public/cpp/ash_switches.h"
#include "base/command_line.h"

namespace ash {
namespace features {

const base::Feature kDockedMagnifier{"DockedMagnifier",
                                     base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kDragAppsInTabletMode{"DragAppsInTabletMode",
                                          base::FEATURE_DISABLED_BY_DEFAULT};

const base::Feature kDragTabsInTabletMode{"DragTabsInTabletMode",
                                          base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kKeyboardShortcutViewer{"KeyboardShortcutViewer",
                                            base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kKeyboardShortcutViewerApp{
    "KeyboardShortcutViewerApp", base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kLockScreenNotifications{"LockScreenNotifications",
                                             base::FEATURE_DISABLED_BY_DEFAULT};

const base::Feature kNewWallpaperPicker{"NewWallpaperPicker",
                                        base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kNightLight{"NightLight", base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kNotificationScrollBar{"NotificationScrollBar",
                                           base::FEATURE_DISABLED_BY_DEFAULT};

const base::Feature kOverviewSwipeToClose{"OverviewSwipeToClose",
                                          base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kSystemTrayUnified{"SystemTrayUnified",
                                       base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kTapVisualizerApp{"TapVisualizerApp",
                                      base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kTrilinearFiltering{"TrilinearFiltering",
                                        base::FEATURE_ENABLED_BY_DEFAULT};

const base::Feature kViewsLogin{"ViewsLogin", base::FEATURE_ENABLED_BY_DEFAULT};

bool IsDockedMagnifierEnabled() {
  return base::FeatureList::IsEnabled(kDockedMagnifier);
}

bool IsKeyboardShortcutViewerEnabled() {
  return base::FeatureList::IsEnabled(kKeyboardShortcutViewer);
}

bool IsKeyboardShortcutViewerAppEnabled() {
  return base::FeatureList::IsEnabled(kKeyboardShortcutViewerApp);
}

bool IsLockScreenNotificationsEnabled() {
  return base::FeatureList::IsEnabled(kLockScreenNotifications);
}

bool IsNewWallpaperPickerEnabled() {
  static bool use_new_wallpaper_picker =
      base::FeatureList::IsEnabled(kNewWallpaperPicker);
  return use_new_wallpaper_picker;
}

bool IsNightLightEnabled() {
  return base::FeatureList::IsEnabled(kNightLight);
}

bool IsNotificationScrollBarEnabled() {
  return base::FeatureList::IsEnabled(kNotificationScrollBar);
}

bool IsSystemTrayUnifiedEnabled() {
  return base::FeatureList::IsEnabled(kSystemTrayUnified);
}

bool IsTrilinearFilteringEnabled() {
  static bool use_trilinear_filtering =
      base::FeatureList::IsEnabled(kTrilinearFiltering);
  return use_trilinear_filtering;
}

bool IsViewsLoginEnabled() {
  // Always show webui login if --show-webui-login is present, which is passed
  // by session manager for automatic recovery. Otherwise, only show views login
  // if the feature is enabled.
  return !base::CommandLine::ForCurrentProcess()->HasSwitch(
             ash::switches::kShowWebUiLogin) &&
         base::FeatureList::IsEnabled(kViewsLogin);
}

}  // namespace features
}  // namespace ash
