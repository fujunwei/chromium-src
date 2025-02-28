// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/ui/views/tabs/tab_close_button.h"

#include <map>
#include <memory>
#include <vector>

#include "base/hash.h"
#include "base/no_destructor.h"
#include "base/stl_util.h"
#include "chrome/app/vector_icons/vector_icons.h"
#include "chrome/browser/ui/layout_constants.h"
#include "chrome/browser/ui/views/tabs/tab.h"
#include "chrome/browser/ui/views/tabs/tab_controller.h"
#include "chrome/common/chrome_features.h"
#include "components/strings/grit/components_strings.h"
#include "ui/base/l10n/l10n_util.h"
#include "ui/base/material_design/material_design_controller.h"
#include "ui/gfx/animation/tween.h"
#include "ui/gfx/canvas.h"
#include "ui/gfx/color_palette.h"
#include "ui/gfx/image/image_skia_operations.h"
#include "ui/gfx/paint_vector_icon.h"
#include "ui/views/rect_based_targeting_utils.h"
#include "ui/views/style/platform_style.h"

#if defined(USE_AURA)
#include "ui/aura/env.h"
#endif

using MD = ui::MaterialDesignController;

TabCloseButton::TabCloseButton(views::ButtonListener* listener,
                               MouseEventCallback mouse_event_callback)
    : views::ImageButton(listener),
      mouse_event_callback_(std::move(mouse_event_callback)) {
  SetEventTargeter(std::make_unique<views::ViewTargeter>(this));
  SetAccessibleName(l10n_util::GetStringUTF16(IDS_ACCNAME_CLOSE));
  // Disable animation so that the red danger sign shows up immediately
  // to help avoid mis-clicks.
  SetAnimationDuration(0);
  SetInstallFocusRingOnFocus(views::PlatformStyle::kPreferFocusRings);

  if (focus_ring())
    SetFocusPainter(nullptr);
}

TabCloseButton::~TabCloseButton() {}

// static
int TabCloseButton::GetWidth() {
  const gfx::VectorIcon& close_icon = MD::IsTouchOptimizedUiEnabled()
                                          ? kTabCloseButtonTouchIcon
                                          : kTabCloseNormalIcon;
  return gfx::GetDefaultSizeOfVectorIcon(close_icon);
}

void TabCloseButton::SetIconColors(SkColor color) {
  GenerateImages(color, MD::IsNewerMaterialUi() ? color : SK_ColorWHITE);
}

views::View* TabCloseButton::GetTooltipHandlerForPoint(
    const gfx::Point& point) {
  // Tab close button has no children, so tooltip handler should be the same
  // as the event handler. In addition, a hit test has to be performed for the
  // point (as GetTooltipHandlerForPoint() is responsible for it).
  if (!HitTestPoint(point))
    return nullptr;
  return GetEventHandlerForPoint(point);
}

bool TabCloseButton::OnMousePressed(const ui::MouseEvent& event) {
  mouse_event_callback_.Run(this, event);

  bool handled = ImageButton::OnMousePressed(event);
  // Explicitly mark midle-mouse clicks as non-handled to ensure the tab
  // sees them.
  return !event.IsMiddleMouseButton() && handled;
}

void TabCloseButton::OnMouseMoved(const ui::MouseEvent& event) {
  mouse_event_callback_.Run(this, event);
  Button::OnMouseMoved(event);
}

void TabCloseButton::OnMouseReleased(const ui::MouseEvent& event) {
  mouse_event_callback_.Run(this, event);
  Button::OnMouseReleased(event);
}

void TabCloseButton::OnGestureEvent(ui::GestureEvent* event) {
  // Consume all gesture events here so that the parent (Tab) does not
  // start consuming gestures.
  ImageButton::OnGestureEvent(event);
  event->SetHandled();
}

const char* TabCloseButton::GetClassName() const {
  return "TabCloseButton";
}

void TabCloseButton::Layout() {
  ImageButton::Layout();
  if (focus_ring()) {
    SkPath path;
    path.addOval(gfx::RectToSkRect(GetMirroredRect(GetContentsBounds())));
    focus_ring()->SetPath(path);
  }
}

void TabCloseButton::PaintButtonContents(gfx::Canvas* canvas) {
  canvas->SaveLayerAlpha(GetOpacity());
  ButtonState button_state = state();
  if (button_state != views::Button::STATE_NORMAL) {
    // Draw the background circle highlight.
    gfx::Path path;
    SkColor background_color =
        static_cast<Tab*>(parent())->GetCloseTabButtonColor(button_state);
    gfx::Point center = GetContentsBounds().CenterPoint();
    path.setFillType(SkPath::kEvenOdd_FillType);
    path.addCircle(center.x(), center.y(), GetWidth() / 2);
    cc::PaintFlags flags;
    flags.setAntiAlias(true);
    flags.setColor(background_color);
    canvas->DrawPath(path, flags);
  }
  views::ImageButton::PaintButtonContents(canvas);
  canvas->Restore();
}

views::View* TabCloseButton::TargetForRect(views::View* root,
                                           const gfx::Rect& rect) {
  CHECK_EQ(root, this);

  if (!views::UsePointBasedTargeting(rect))
    return ViewTargeterDelegate::TargetForRect(root, rect);

  // Ignore the padding set on the button.
  gfx::Rect contents_bounds = GetMirroredRect(GetContentsBounds());

#if defined(USE_AURA)
  // Include the padding in hit-test for touch events.
  // TODO(pkasting): It seems like touch events would generate rects rather
  // than points and thus use the TargetForRect() call above.  If this is
  // reached, it may be from someone calling GetEventHandlerForPoint() while a
  // touch happens to be occurring.  In such a case, maybe we don't want this
  // code to run?  It's possible this block should be removed, or maybe this
  // whole function deleted.  Note that in these cases, we should probably
  // also remove the padding on the close button bounds (see Tab::Layout()),
  // as it will be pointless.
  if (aura::Env::GetInstance()->is_touch_down())
    contents_bounds = GetLocalBounds();
#endif

  return contents_bounds.Intersects(rect) ? this : parent();
}

bool TabCloseButton::GetHitTestMask(gfx::Path* mask) const {
  // We need to define this so hit-testing won't include the border region.
  mask->addRect(gfx::RectToSkRect(GetMirroredRect(GetContentsBounds())));
  return true;
}

SkAlpha TabCloseButton::GetOpacity() {
  Tab* tab = static_cast<Tab*>(parent());
  if (base::FeatureList::IsEnabled(features::kCloseButtonsInactiveTabs) ||
      IsMouseHovered() || tab->IsActive())
    return SK_AlphaOPAQUE;
  const double animation_value = tab->hover_controller()->GetAnimationValue();
  return gfx::Tween::IntValueBetween(animation_value, SK_AlphaTRANSPARENT,
                                     SK_AlphaOPAQUE);
}

void TabCloseButton::GenerateImages(SkColor normal_icon_color,
                                    SkColor hover_pressed_icon_color) {
  const gfx::VectorIcon& button_icon = MD::IsTouchOptimizedUiEnabled()
                                           ? kTabCloseButtonTouchIcon
                                           : kTabCloseNormalIcon;
  const gfx::ImageSkia normal =
      gfx::CreateVectorIcon(button_icon, normal_icon_color);
  const gfx::ImageSkia hover_pressed =
      normal_icon_color != hover_pressed_icon_color
          ? gfx::CreateVectorIcon(button_icon, hover_pressed_icon_color)
          : normal;
  SetImage(views::Button::STATE_NORMAL, normal);
  SetImage(views::Button::STATE_HOVERED, hover_pressed);
  SetImage(views::Button::STATE_PRESSED, hover_pressed);
}
