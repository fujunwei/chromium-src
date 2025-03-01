// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_CHROMEOS_LOGIN_UI_LOGIN_DISPLAY_HOST_WEBUI_H_
#define CHROME_BROWSER_CHROMEOS_LOGIN_UI_LOGIN_DISPLAY_HOST_WEBUI_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "base/memory/weak_ptr.h"
#include "chrome/browser/chromeos/login/existing_user_controller.h"
#include "chrome/browser/chromeos/login/signin_screen_controller.h"
#include "chrome/browser/chromeos/login/ui/login_display.h"
#include "chrome/browser/chromeos/login/ui/login_display_host_common.h"
#include "chrome/browser/chromeos/login/wizard_controller.h"
#include "chrome/browser/chromeos/settings/device_settings_service.h"
#include "chrome/browser/ui/ash/multi_user/multi_user_window_manager.h"
#include "chromeos/audio/cras_audio_handler.h"
#include "chromeos/dbus/session_manager_client.h"
#include "content/public/browser/notification_observer.h"
#include "content/public/browser/notification_registrar.h"
#include "content/public/browser/web_contents_observer.h"
#include "ui/display/display_observer.h"
#include "ui/events/devices/input_device_event_observer.h"
#include "ui/gfx/geometry/rect.h"
#include "ui/views/widget/widget_removals_observer.h"

namespace ash {
class FocusRingController;
}

namespace chromeos {

class LoginDisplayWebUI;
class WebUILoginView;

// An implementation class for OOBE/login WebUI screen host.
// It encapsulates controllers, wallpaper integration and flow.
class LoginDisplayHostWebUI : public LoginDisplayHostCommon,
                              public content::WebContentsObserver,
                              public chromeos::SessionManagerClient::Observer,
                              public chromeos::CrasAudioHandler::AudioObserver,
                              public display::DisplayObserver,
                              public ui::InputDeviceEventObserver,
                              public views::WidgetRemovalsObserver,
                              public MultiUserWindowManager::Observer {
 public:
  LoginDisplayHostWebUI();
  ~LoginDisplayHostWebUI() override;

  // LoginDisplayHost:
  LoginDisplay* GetLoginDisplay() override;
  ExistingUserController* GetExistingUserController() override;
  gfx::NativeWindow GetNativeWindow() const override;
  OobeUI* GetOobeUI() const override;
  content::WebContents* GetOobeWebContents() const override;
  WebUILoginView* GetWebUILoginView() const override;
  void OnFinalize() override;
  void SetStatusAreaVisible(bool visible) override;
  void StartWizard(OobeScreen first_screen) override;
  WizardController* GetWizardController() override;
  void OnStartUserAdding() override;
  void CancelUserAdding() override;
  void OnStartSignInScreen(const LoginScreenContext& context) override;
  void OnPreferencesChanged() override;
  void OnStartAppLaunch() override;
  void OnStartArcKiosk() override;
  bool IsVoiceInteractionOobe() override;
  void StartVoiceInteractionOobe() override;
  void OnBrowserCreated() override;
  void ShowGaiaDialog(
      bool can_close,
      const base::Optional<AccountId>& prefilled_account) override;
  void HideOobeDialog() override;
  void UpdateOobeDialogSize(int width, int height) override;
  const user_manager::UserList GetUsers() override;
  void ShowFeedback() override;
  void ShowDialogForCaptivePortal() override;
  void HideDialogForCaptivePortal() override;
  void UpdateAddUserButtonStatus() override;

  void OnCancelPasswordChangedFlow() override;

  // Trace id for ShowLoginWebUI event (since there exists at most one login
  // WebUI at a time).
  static const int kShowLoginWebUIid;

  views::Widget* login_window_for_test() { return login_window_; }

  // Disable GaiaScreenHandler restrictive proxy check.
  static void DisableRestrictiveProxyCheckForTest();

 protected:
  class KeyboardDrivenOobeKeyHandler;

  // LoginDisplayHost:
  void Observe(int type,
               const content::NotificationSource& source,
               const content::NotificationDetails& details) override;

  // content::WebContentsObserver:
  void RenderProcessGone(base::TerminationStatus status) override;

  // chromeos::SessionManagerClient::Observer:
  void EmitLoginPromptVisibleCalled() override;

  // chromeos::CrasAudioHandler::AudioObserver:
  void OnActiveOutputNodeChanged() override;

  // display::DisplayObserver:
  void OnDisplayAdded(const display::Display& new_display) override;
  void OnDisplayMetricsChanged(const display::Display& display,
                               uint32_t changed_metrics) override;

  // ui::InputDeviceEventObserver
  void OnTouchscreenDeviceConfigurationChanged() override;

  // views::WidgetRemovalsObserver:
  void OnWillRemoveView(views::Widget* widget, views::View* view) override;

  // chrome::MultiUserWindowManager::Observer:
  void OnUserSwitchAnimationFinished() override;

 private:
  class LoginWidgetDelegate;

  // Way to restore if renderer have crashed.
  enum RestorePath {
    RESTORE_UNKNOWN,
    RESTORE_WIZARD,
    RESTORE_SIGN_IN,
    RESTORE_ADD_USER_INTO_SESSION,
  };

  // Type of animations to run after the login screen.
  enum FinalizeAnimationType {
    ANIMATION_NONE,       // No animation.
    ANIMATION_WORKSPACE,  // Use initial workspace animation (drop and
                          // and fade in workspace). Used for user login.
    ANIMATION_FADE_OUT,   // Fade out login screen. Used for app launch.
    ANIMATION_ADD_USER,   // Use UserSwitchAnimatorChromeOS animation when
                          // adding a user into multi-profile session.
  };

  // Schedules workspace transition animation.
  void ScheduleWorkspaceAnimation();

  // Schedules fade out animation.
  void ScheduleFadeOutAnimation(int animation_speed_ms);

  // Loads given URL. Creates WebUILoginView if needed.
  void LoadURL(const GURL& url);

  // Shows OOBE/sign in WebUI that was previously initialized in hidden state.
  void ShowWebUI();

  // Starts postponed WebUI (OOBE/sign in) if it was waiting for
  // wallpaper animation end.
  void StartPostponedWebUI();

  // Initializes |login_window_| and |login_view_| fields if needed.
  void InitLoginWindowAndView();

  // Closes |login_window_| and resets |login_window_| and |login_view_| fields.
  void ResetLoginWindowAndView();

  // Toggles OOBE progress bar visibility, the bar is hidden by default.
  void SetOobeProgressBarVisible(bool visible);

  // Tries to play startup sound. If sound can't be played right now,
  // for instance, because cras server is not initialized, playback
  // will be delayed.
  void TryToPlayOobeStartupSound();

  // Called when login-prompt-visible signal is caught.
  void OnLoginPromptVisible();

  // Creates or recreates |existing_user_controller_|.
  void CreateExistingUserController();

  // Sign in screen controller.
  std::unique_ptr<ExistingUserController> existing_user_controller_;

  // OOBE and some screens (camera, recovery) controller.
  std::unique_ptr<WizardController> wizard_controller_;

  std::unique_ptr<SignInScreenController> signin_screen_controller_;

  // Whether progress bar is shown on the OOBE page.
  bool oobe_progress_bar_visible_ = false;

  // Container of the screen we are displaying.
  views::Widget* login_window_ = nullptr;

  // The delegate of |login_window_|; owned by |login_window_|.
  LoginWidgetDelegate* login_window_delegate_ = nullptr;

  // Container of the view we are displaying.
  WebUILoginView* login_view_ = nullptr;

  // Login display we are using.
  std::unique_ptr<LoginDisplayWebUI> login_display_;

  // True if the login display is the current screen.
  bool is_showing_login_ = false;

  // True if NOTIFICATION_WALLPAPER_ANIMATION_FINISHED notification has been
  // received.
  bool is_wallpaper_loaded_ = false;

  // Stores status area current visibility to be applied once login WebUI
  // is shown.
  bool status_area_saved_visibility_ = false;

  // If true, WebUI is initialized in a hidden state and shown after the
  // wallpaper animation is finished (when it is enabled) or the user pods have
  // been loaded (otherwise).
  // By default is true. Could be used to tune performance if needed.
  bool initialize_webui_hidden_;

  // True if WebUI is initialized in hidden state and we're waiting for
  // wallpaper load animation to finish.
  bool waiting_for_wallpaper_load_;

  // How many times renderer has crashed.
  int crash_count_ = 0;

  // Way to restore if renderer have crashed.
  RestorePath restore_path_ = RESTORE_UNKNOWN;

  // Stored parameters for StartWizard, required to restore in case of crash.
  OobeScreen first_screen_;

  // A focus ring controller to draw focus ring around view for keyboard
  // driven oobe.
  std::unique_ptr<ash::FocusRingController> focus_ring_controller_;

  // Handles special keys for keyboard driven oobe.
  std::unique_ptr<KeyboardDrivenOobeKeyHandler>
      keyboard_driven_oobe_key_handler_;

  FinalizeAnimationType finalize_animation_type_ = ANIMATION_WORKSPACE;

  // Time when login prompt visible signal is received. Used for
  // calculations of delay before startup sound.
  base::TimeTicks login_prompt_visible_time_;

  // True when request to play startup sound was sent to
  // SoundsManager.
  // After OOBE is completed, this is always initialized with true.
  bool oobe_startup_sound_played_ = false;

  bool is_voice_interaction_oobe_ = false;

  base::WeakPtrFactory<LoginDisplayHostWebUI> weak_factory_;

  DISALLOW_COPY_AND_ASSIGN(LoginDisplayHostWebUI);
};

}  // namespace chromeos

#endif  // CHROME_BROWSER_CHROMEOS_LOGIN_UI_LOGIN_DISPLAY_HOST_WEBUI_H_
