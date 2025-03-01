// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/ui/ash/launcher/settings_window_observer.h"

#include "ash/public/cpp/app_list/internal_app_id_constants.h"
#include "ash/public/cpp/resources/grit/ash_public_unscaled_resources.h"
#include "ash/public/cpp/shelf_item.h"
#include "ash/public/cpp/window_properties.h"
#include "chrome/browser/ui/browser.h"
#include "chrome/browser/ui/browser_window.h"
#include "chrome/browser/ui/settings_window_manager_chromeos.h"
#include "components/strings/grit/components_strings.h"
#include "services/ui/public/interfaces/window_manager.mojom.h"
#include "ui/aura/client/aura_constants.h"
#include "ui/aura/window.h"
#include "ui/base/class_property.h"
#include "ui/base/l10n/l10n_util.h"
#include "ui/base/resource/resource_bundle.h"
#include "ui/gfx/image/image_skia.h"

namespace {

// A helper class that updates the title of Chrome OS Settings browser windows.
class AuraWindowSettingsTitleTracker : public aura::WindowTracker {
 public:
  AuraWindowSettingsTitleTracker() {}
  ~AuraWindowSettingsTitleTracker() override {}

  // aura::WindowTracker:
  void OnWindowTitleChanged(aura::Window* window) override {
    // Name the window "Settings" instead of "Google Chrome - Settings".
    window->SetTitle(l10n_util::GetStringUTF16(IDS_SETTINGS_TITLE));
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(AuraWindowSettingsTitleTracker);
};

}  // namespace

SettingsWindowObserver::SettingsWindowObserver() {
  aura_window_tracker_ = std::make_unique<AuraWindowSettingsTitleTracker>();
  chrome::SettingsWindowManager::GetInstance()->AddObserver(this);
}

SettingsWindowObserver::~SettingsWindowObserver() {
  chrome::SettingsWindowManager::GetInstance()->RemoveObserver(this);
}

void SettingsWindowObserver::OnNewSettingsWindow(Browser* settings_browser) {
  aura::Window* window = settings_browser->window()->GetNativeWindow();
  window->SetTitle(l10n_util::GetStringUTF16(IDS_SETTINGS_TITLE));
  const ash::ShelfID shelf_id(app_list::kInternalAppIdSettings);
  window->SetProperty(ash::kShelfIDKey, new std::string(shelf_id.Serialize()));
  window->SetProperty<int>(ash::kShelfItemTypeKey, ash::TYPE_APP);
  ui::ResourceBundle& rb = ui::ResourceBundle::GetSharedInstance();
  // The new gfx::ImageSkia instance is owned by the window itself.
  window->SetProperty(
      aura::client::kWindowIconKey,
      new gfx::ImageSkia(*rb.GetImageSkiaNamed(IDR_SETTINGS_LOGO_192)));
  aura_window_tracker_->Add(window);
}
