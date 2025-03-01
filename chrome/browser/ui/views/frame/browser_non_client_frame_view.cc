// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/ui/views/frame/browser_non_client_frame_view.h"

#include "base/metrics/histogram_macros.h"
#include "build/build_config.h"
#include "chrome/app/vector_icons/vector_icons.h"
#include "chrome/browser/browser_process.h"
#include "chrome/browser/profiles/avatar_menu.h"
#include "chrome/browser/profiles/profile.h"
#include "chrome/browser/profiles/profile_attributes_entry.h"
#include "chrome/browser/profiles/profile_manager.h"
#include "chrome/browser/themes/theme_properties.h"
#include "chrome/browser/ui/extensions/hosted_app_browser_controller.h"
#include "chrome/browser/ui/layout_constants.h"
#include "chrome/browser/ui/view_ids.h"
#include "chrome/browser/ui/views/frame/browser_view.h"
#include "chrome/browser/ui/views/tabs/tab_strip.h"
#include "chrome/common/chrome_features.h"
#include "chrome/grit/theme_resources.h"
#include "components/signin/core/browser/profile_management_switches.h"
#include "third_party/skia/include/core/SkColor.h"
#include "ui/base/material_design/material_design_controller.h"
#include "ui/base/theme_provider.h"
#include "ui/gfx/canvas.h"
#include "ui/gfx/color_palette.h"
#include "ui/gfx/image/image.h"
#include "ui/gfx/paint_vector_icon.h"
#include "ui/gfx/scoped_canvas.h"
#include "ui/views/background.h"

#if defined(OS_CHROMEOS)
#include "chrome/browser/ui/ash/multi_user/multi_user_window_manager.h"
#include "chrome/browser/ui/ash/session_util.h"
#endif  // defined(OS_CHROMEOS)

#if defined(OS_WIN)
#include "chrome/browser/ui/views/frame/taskbar_decorator_win.h"
#endif

using MD = ui::MaterialDesignController;

BrowserNonClientFrameView::BrowserNonClientFrameView(BrowserFrame* frame,
                                                     BrowserView* browser_view)
    : frame_(frame),
      browser_view_(browser_view),
      profile_switcher_(this),
      profile_indicator_icon_(nullptr),
      tab_strip_observer_(this) {
  // The profile manager may by null in tests.
  if (g_browser_process->profile_manager()) {
    g_browser_process->profile_manager()->
        GetProfileAttributesStorage().AddObserver(this);
  }
  MaybeObserveTabstrip();
}

BrowserNonClientFrameView::~BrowserNonClientFrameView() {
  // The profile manager may by null in tests.
  if (g_browser_process->profile_manager()) {
    g_browser_process->profile_manager()->
        GetProfileAttributesStorage().RemoveObserver(this);
  }
}

// static
int BrowserNonClientFrameView::GetAvatarIconPadding() {
  return MD::IsNewerMaterialUi() ? 8 : 4;
}

// static
int BrowserNonClientFrameView::GetTabstripPadding() {
  // In Refresh, the apparent padding around the tabstrip is contained within
  // the tabs and/or new tab button.
  return MD::IsRefreshUi() ? 0 : 4;
}

void BrowserNonClientFrameView::OnBrowserViewInitViewsComplete() {
  MaybeObserveTabstrip();
  OnSingleTabModeChanged();
  UpdateMinimumSize();
}

void BrowserNonClientFrameView::OnMaximizedStateChanged() {}

void BrowserNonClientFrameView::OnFullscreenStateChanged() {}

bool BrowserNonClientFrameView::CaptionButtonsOnLeadingEdge() const {
  return false;
}

void BrowserNonClientFrameView::UpdateFullscreenTopUI(
    bool is_exiting_fullscreen) {}

bool BrowserNonClientFrameView::ShouldHideTopUIForFullscreen() const {
  return frame()->IsFullscreen();
}

bool BrowserNonClientFrameView::HasClientEdge() const {
  return !MD::IsRefreshUi();
}

gfx::ImageSkia BrowserNonClientFrameView::GetIncognitoAvatarIcon() const {
  const SkColor icon_color = color_utils::PickContrastingColor(
      SK_ColorWHITE, gfx::kChromeIconGrey, GetFrameColor());
  return gfx::CreateVectorIcon(kIncognitoIcon, icon_color);
}

SkColor BrowserNonClientFrameView::GetToolbarTopSeparatorColor() const {
  const auto color_id =
      ShouldPaintAsActive()
          ? ThemeProperties::COLOR_TOOLBAR_TOP_SEPARATOR
          : ThemeProperties::COLOR_TOOLBAR_TOP_SEPARATOR_INACTIVE;
  return GetThemeOrDefaultColor(color_id);
}

SkColor BrowserNonClientFrameView::GetTabSeparatorColor() const {
  DCHECK(MD::IsRefreshUi());
  constexpr SkAlpha kTabSeparatorAlpha = 0x4D;  // 30%
  const SkColor frame_color = GetFrameColor();
  const SkColor base_color =
      color_utils::BlendTowardOppositeLuma(frame_color, SK_AlphaOPAQUE);
  return color_utils::AlphaBlend(base_color, frame_color, kTabSeparatorAlpha);
}

SkColor BrowserNonClientFrameView::GetTabBackgroundColor(TabState state) const {
  if (state == TAB_INACTIVE && MD::IsRefreshUi())
    return GetFrameColor();
  const auto color_id = state == TAB_ACTIVE
                            ? ThemeProperties::COLOR_TOOLBAR
                            : ThemeProperties::COLOR_BACKGROUND_TAB;
  return GetThemeOrDefaultColor(color_id);
}

SkColor BrowserNonClientFrameView::GetTabForegroundColor(TabState state) const {
  if (MD::IsRefreshUi() && state == TAB_INACTIVE &&
      !GetThemeProvider()->HasCustomColor(
          ThemeProperties::COLOR_BACKGROUND_TAB_TEXT)) {
    const SkColor background_color = GetTabBackgroundColor(TAB_INACTIVE);
    const SkColor default_color = color_utils::IsDark(background_color)
                                      ? gfx::kGoogleGrey500
                                      : gfx::kGoogleGrey700;
    return color_utils::GetColorWithMinimumContrast(default_color,
                                                    background_color);
  }

  const auto color_id = state == TAB_ACTIVE
                            ? ThemeProperties::COLOR_TAB_TEXT
                            : ThemeProperties::COLOR_BACKGROUND_TAB_TEXT;
  return GetThemeOrDefaultColor(color_id);
}

views::Button* BrowserNonClientFrameView::GetProfileSwitcherButton() const {
  return profile_switcher_.avatar_button();
}

void BrowserNonClientFrameView::UpdateClientArea() {}

void BrowserNonClientFrameView::UpdateMinimumSize() {}

int BrowserNonClientFrameView::GetTabStripLeftInset() const {
  int left_inset = GetTabstripPadding();
  if (profile_indicator_icon())
    left_inset += GetAvatarIconPadding() + GetIncognitoAvatarIcon().width();
  return left_inset;
}

void BrowserNonClientFrameView::ChildPreferredSizeChanged(views::View* child) {
  if (child == GetProfileSwitcherButton()) {
    // Perform a re-layout if the avatar button has changed, since that can
    // affect the size of the tabs.
    frame()->GetRootView()->Layout();
  }
}

void BrowserNonClientFrameView::VisibilityChanged(views::View* starting_from,
                                                  bool is_visible) {
  // UpdateTaskbarDecoration() calls DrawTaskbarDecoration(), but that does
  // nothing if the window is not visible.  So even if we've already gotten the
  // up-to-date decoration, we need to run the update procedure again here when
  // the window becomes visible.
  if (is_visible)
    OnProfileAvatarChanged(base::FilePath());
}

void BrowserNonClientFrameView::OnSingleTabModeChanged() {
  SchedulePaint();
}

bool BrowserNonClientFrameView::ShouldPaintAsThemed() const {
  return browser_view_->IsBrowserTypeNormal();
}

bool BrowserNonClientFrameView::IsSingleTabModeAvailable() const {
  // Single-tab mode is only available in Refresh and when the window is active.
  // The special color we use won't be visible if there's a frame image, but
  // since it's used to determine constrast of other UI elements, the theme
  // color should be used instead.
  return base::FeatureList::IsEnabled(features::kSingleTabMode) &&
         MD::IsRefreshUi() && ShouldPaintAsActive() && GetFrameImage().isNull();
}

bool BrowserNonClientFrameView::ShouldPaintAsSingleTabMode() const {
  return browser_view()->IsTabStripVisible() &&
         browser_view()->tabstrip()->SingleTabMode();
}

SkColor BrowserNonClientFrameView::GetFrameColor(bool active) const {
  extensions::HostedAppBrowserController* hosted_app_controller =
      browser_view()->browser()->hosted_app_controller();
  if (hosted_app_controller && hosted_app_controller->GetThemeColor())
    return *hosted_app_controller->GetThemeColor();

  ThemeProperties::OverwritableByUserThemeProperty color_id;
  if (ShouldPaintAsSingleTabMode()) {
    color_id = ThemeProperties::COLOR_TOOLBAR;
  } else {
    color_id = active ? ThemeProperties::COLOR_FRAME
                      : ThemeProperties::COLOR_FRAME_INACTIVE;
  }
  return ShouldPaintAsThemed()
             ? GetThemeProviderForProfile()->GetColor(color_id)
             : ThemeProperties::GetDefaultColor(color_id,
                                                browser_view_->IsIncognito());
}

gfx::ImageSkia BrowserNonClientFrameView::GetFrameImage(bool active) const {
  const ui::ThemeProvider* tp = GetThemeProviderForProfile();
  int frame_image_id = active ? IDR_THEME_FRAME : IDR_THEME_FRAME_INACTIVE;
  return ShouldPaintAsThemed() && (tp->HasCustomImage(frame_image_id) ||
                                   tp->HasCustomImage(IDR_THEME_FRAME))
             ? *tp->GetImageSkiaNamed(frame_image_id)
             : gfx::ImageSkia();
}

gfx::ImageSkia BrowserNonClientFrameView::GetFrameOverlayImage(
    bool active) const {
  if (browser_view_->IsIncognito() || !browser_view_->IsBrowserTypeNormal())
    return gfx::ImageSkia();

  const ui::ThemeProvider* tp = GetThemeProviderForProfile();
  int frame_overlay_image_id =
      active ? IDR_THEME_FRAME_OVERLAY : IDR_THEME_FRAME_OVERLAY_INACTIVE;
  return tp->HasCustomImage(frame_overlay_image_id)
             ? *tp->GetImageSkiaNamed(frame_overlay_image_id)
             : gfx::ImageSkia();
}

SkColor BrowserNonClientFrameView::GetFrameColor() const {
  return GetFrameColor(ShouldPaintAsActive());
}

gfx::ImageSkia BrowserNonClientFrameView::GetFrameImage() const {
  return GetFrameImage(ShouldPaintAsActive());
}

gfx::ImageSkia BrowserNonClientFrameView::GetFrameOverlayImage() const {
  return GetFrameOverlayImage(ShouldPaintAsActive());
}

void BrowserNonClientFrameView::UpdateProfileIcons() {
  const AvatarButtonStyle avatar_button_style = GetAvatarButtonStyle();
  if (avatar_button_style != AvatarButtonStyle::NONE &&
      browser_view()->IsRegularOrGuestSession()) {
    // Platform supports a profile switcher that will be shown. Skip the rest.
    profile_switcher_.Update(avatar_button_style);
    return;
  }

  if (!ShouldShowProfileIndicatorIcon()) {
    if (profile_indicator_icon_) {
      delete profile_indicator_icon_;
      profile_indicator_icon_ = nullptr;
      frame_->GetRootView()->Layout();
    }
    return;
  }

  if (!profile_indicator_icon_) {
    profile_indicator_icon_ = new ProfileIndicatorIcon();
    profile_indicator_icon_->set_id(VIEW_ID_PROFILE_INDICATOR_ICON);
    AddChildView(profile_indicator_icon_);
    // Invalidate here because adding a child does not invalidate the layout.
    InvalidateLayout();
    frame_->GetRootView()->Layout();
  }

  gfx::Image icon;
  Profile* profile = browser_view()->browser()->profile();
  const bool is_incognito =
      profile->GetProfileType() == Profile::INCOGNITO_PROFILE;
  if (is_incognito) {
    icon = gfx::Image(GetIncognitoAvatarIcon());
    profile_indicator_icon_->set_stroke_color(SK_ColorTRANSPARENT);
  } else {
#if defined(OS_CHROMEOS)
    icon = gfx::Image(GetAvatarImageForContext(profile));
    // Draw a stroke around the profile icon only for the avatar.
    profile_indicator_icon_->set_stroke_color(GetToolbarTopSeparatorColor());
#else
    NOTREACHED();
#endif
  }

  profile_indicator_icon_->SetIcon(icon);
}

void BrowserNonClientFrameView::LayoutIncognitoButton() {
  DCHECK(profile_indicator_icon());
#if !defined(OS_CHROMEOS)
  // ChromeOS shows avatar on V1 app.
  DCHECK(browser_view()->IsTabStripVisible());
#endif
  gfx::ImageSkia incognito_icon = GetIncognitoAvatarIcon();
  int avatar_bottom = GetTopInset(false) + browser_view()->GetTabStripHeight() -
                      GetAvatarIconPadding();
  int avatar_y = avatar_bottom - incognito_icon.height();
  int avatar_height = incognito_icon.height();
  gfx::Rect avatar_bounds(GetAvatarIconPadding(), avatar_y,
                          incognito_icon.width(), avatar_height);

  profile_indicator_icon()->SetBoundsRect(avatar_bounds);
  profile_indicator_icon()->SetVisible(true);
}

void BrowserNonClientFrameView::PaintToolbarTopStroke(
    gfx::Canvas* canvas) const {
  if (TabStrip::ShouldDrawStrokes()) {
    gfx::Rect toolbar_bounds(browser_view()->GetToolbarBounds());
    gfx::Rect tabstrip_bounds =
        GetMirroredRect(GetBoundsForTabStrip(browser_view()->tabstrip()));

    gfx::ScopedCanvas scoped_canvas(canvas);
    canvas->ClipRect(tabstrip_bounds, SkClipOp::kDifference);

    const gfx::Rect separator_rect(toolbar_bounds.x(), tabstrip_bounds.bottom(),
                                   toolbar_bounds.width(), 0);
    BrowserView::Paint1pxHorizontalLine(canvas, GetToolbarTopSeparatorColor(),
                                        separator_rect, true);
  }
}

void BrowserNonClientFrameView::ViewHierarchyChanged(
    const ViewHierarchyChangedDetails& details) {
  if (details.is_add && details.child == this)
    UpdateProfileIcons();
}

void BrowserNonClientFrameView::ActivationChanged(bool active) {
  // On Windows, while deactivating the widget, this is called before the
  // active HWND has actually been changed.  Since we want the avatar state to
  // reflect that the window is inactive, we force NonClientFrameView to see the
  // "correct" state as an override.
  set_active_state_override(&active);
  UpdateProfileIcons();

  if (MD::IsRefreshUi()) {
    // Single-tab mode's availability depends on activation, but even if it's
    // unavailable for other reasons the inactive tabs' text color still needs
    // to be recalculated if the frame color changes. SingleTabModeChanged will
    // handle both cases.
    browser_view_->tabstrip()->SingleTabModeChanged();
  } else {
    // The toolbar top separator color (used as the stroke around the tabs and
    // the new tab button) needs to be recalculated.
    browser_view_->tabstrip()->FrameColorsChanged();
  }

  set_active_state_override(nullptr);

  // Changing the activation state may change the visible frame color.
  SchedulePaint();
}

bool BrowserNonClientFrameView::DoesIntersectRect(const views::View* target,
                                                  const gfx::Rect& rect) const {
  DCHECK_EQ(target, this);
  if (!views::ViewTargeterDelegate::DoesIntersectRect(this, rect)) {
    // |rect| is outside the frame's bounds.
    return false;
  }

  bool should_leave_to_top_container = false;
#if defined(OS_CHROMEOS)
  // In immersive mode, the caption buttons container is reparented to the
  // TopContainerView and hence |rect| should not be claimed here.  See
  // BrowserNonClientFrameViewAsh::OnImmersiveRevealStarted().
  should_leave_to_top_container =
      browser_view()->immersive_mode_controller()->IsRevealed();
#endif  // defined(OS_CHROMEOS)

  if (!browser_view()->IsTabStripVisible()) {
    // Claim |rect| if it is above the top of the topmost client area view.
    return !should_leave_to_top_container && (rect.y() < GetTopInset(false));
  }

  // If the rect is outside the bounds of the client area, claim it.
  gfx::RectF rect_in_client_view_coords_f(rect);
  View::ConvertRectToTarget(this, frame()->client_view(),
                            &rect_in_client_view_coords_f);
  gfx::Rect rect_in_client_view_coords =
      gfx::ToEnclosingRect(rect_in_client_view_coords_f);
  if (!frame()->client_view()->HitTestRect(rect_in_client_view_coords))
    return true;

  // Otherwise, claim |rect| only if it is above the bottom of the tabstrip in
  // a non-tab portion.
  TabStrip* tabstrip = browser_view()->tabstrip();
  gfx::RectF rect_in_tabstrip_coords_f(rect);
  View::ConvertRectToTarget(this, tabstrip, &rect_in_tabstrip_coords_f);
  gfx::Rect rect_in_tabstrip_coords =
      gfx::ToEnclosingRect(rect_in_tabstrip_coords_f);
  if (rect_in_tabstrip_coords.y() >= tabstrip->GetLocalBounds().bottom()) {
    // |rect| is below the tabstrip.
    return false;
  }

  if (tabstrip->HitTestRect(rect_in_tabstrip_coords)) {
    // Claim |rect| if it is in a non-tab portion of the tabstrip.
    return tabstrip->IsRectInWindowCaption(rect_in_tabstrip_coords);
  }

  // We claim |rect| because it is above the bottom of the tabstrip, but
  // not in the tabstrip itself. In particular, the avatar label/button is left
  // of the tabstrip and the window controls are right of the tabstrip.
  return !should_leave_to_top_container;
}

void BrowserNonClientFrameView::OnProfileAdded(
    const base::FilePath& profile_path) {
  OnProfileAvatarChanged(profile_path);
}

void BrowserNonClientFrameView::OnProfileWasRemoved(
    const base::FilePath& profile_path,
    const base::string16& profile_name) {
  OnProfileAvatarChanged(profile_path);
}

void BrowserNonClientFrameView::OnProfileAvatarChanged(
    const base::FilePath& profile_path) {
  UpdateTaskbarDecoration();
  UpdateProfileIcons();
}

void BrowserNonClientFrameView::OnProfileHighResAvatarLoaded(
    const base::FilePath& profile_path) {
  UpdateTaskbarDecoration();
}

void BrowserNonClientFrameView::MaybeObserveTabstrip() {
  if (browser_view()->tabstrip()) {
    DCHECK(!tab_strip_observer_.IsObserving(browser_view()->tabstrip()));
    tab_strip_observer_.Add(browser_view()->tabstrip());
  }
}

const ui::ThemeProvider*
BrowserNonClientFrameView::GetThemeProviderForProfile() const {
  // Because the frame's accessor reads the ThemeProvider from the profile and
  // not the widget, it can be called even before we're in a view hierarchy.
  return frame_->GetThemeProvider();
}

void BrowserNonClientFrameView::UpdateTaskbarDecoration() {
#if defined(OS_WIN)
  if (browser_view()->browser()->profile()->IsGuestSession() ||
      // Browser process and profile manager may be null in tests.
      (g_browser_process && g_browser_process->profile_manager() &&
       g_browser_process->profile_manager()
               ->GetProfileAttributesStorage()
               .GetNumberOfProfiles() <= 1)) {
    chrome::DrawTaskbarDecoration(frame_->GetNativeWindow(), nullptr);
    return;
  }

  // For popups and panels which don't have the avatar button, we still
  // need to draw the taskbar decoration. Even though we have an icon on the
  // window's relaunch details, we draw over it because the user may have
  // pinned the badge-less Chrome shortcut which will cause Windows to ignore
  // the relaunch details.
  // TODO(calamity): ideally this should not be necessary but due to issues
  // with the default shortcut being pinned, we add the runtime badge for
  // safety. See crbug.com/313800.
  gfx::Image decoration;
  AvatarMenu::ImageLoadStatus status = AvatarMenu::GetImageForMenuButton(
      browser_view()->browser()->profile()->GetPath(), &decoration);

  UMA_HISTOGRAM_ENUMERATION(
      "Profile.AvatarLoadStatus", status,
      static_cast<int>(AvatarMenu::ImageLoadStatus::MAX) + 1);

  // If the user is using a Gaia picture and the picture is still being loaded,
  // wait until the load finishes. This taskbar decoration will be triggered
  // again upon the finish of the picture load.
  if (status == AvatarMenu::ImageLoadStatus::LOADING ||
      status == AvatarMenu::ImageLoadStatus::PROFILE_DELETED) {
    return;
  }

  chrome::DrawTaskbarDecoration(frame_->GetNativeWindow(), &decoration);
#endif
}

bool BrowserNonClientFrameView::ShouldShowProfileIndicatorIcon() const {
#if !defined(OS_CHROMEOS)
  // Outside ChromeOS, in Material Refresh, we use a toolbar button for all
  // profile/incognito-related purposes. ChromeOS uses it for teleportation (see
  // below).
  if (MD::IsRefreshUi())
    return false;
#endif  // !defined(OS_CHROMEOS)

  Browser* browser = browser_view()->browser();
  Profile* profile = browser->profile();
  const bool is_incognito =
      profile->GetProfileType() == Profile::INCOGNITO_PROFILE;

  // In newer material UIs we only show the avatar icon for the teleported
  // browser windows between multi-user sessions (Chrome OS only). Note that you
  // can't teleport an incognito window.
  if (is_incognito && MD::IsNewerMaterialUi())
    return false;

#if defined(OS_CHROMEOS)
  if (!browser->is_type_tabbed() && !browser->is_app())
    return false;

  if (!is_incognito && !MultiUserWindowManager::ShouldShowAvatar(
                           browser_view()->GetNativeWindow())) {
    return false;
  }
#endif  // defined(OS_CHROMEOS)
  return true;
}

SkColor BrowserNonClientFrameView::GetThemeOrDefaultColor(int color_id) const {
  return ShouldPaintAsThemed() ? GetThemeProvider()->GetColor(color_id)
                               : ThemeProperties::GetDefaultColor(
                                     color_id, browser_view_->IsIncognito());
}
