// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ash/frame/header_view.h"

#include <memory>

#include "ash/client_image_registry.h"
#include "ash/frame/caption_buttons/caption_button_model.h"
#include "ash/frame/caption_buttons/frame_back_button.h"
#include "ash/frame/caption_buttons/frame_caption_button_container_view.h"
#include "ash/frame/custom_frame_view_ash.h"
#include "ash/frame/default_frame_header.h"
#include "ash/public/cpp/config.h"
#include "ash/public/cpp/window_properties.h"
#include "ash/shell.h"
#include "ash/wm/tablet_mode/tablet_mode_controller.h"
#include "ash/wm/window_state.h"
#include "ui/aura/client/aura_constants.h"
#include "ui/aura/window.h"
#include "ui/base/ui_base_features.h"
#include "ui/views/controls/image_view.h"
#include "ui/views/widget/widget.h"

namespace ash {

namespace {

// An appearance provider that relies on window properties which have been set
// by the client. Only used in Mash.
class WindowPropertyAppearanceProvider
    : public CustomFrameHeader::AppearanceProvider {
 public:
  explicit WindowPropertyAppearanceProvider(aura::Window* window)
      : window_(window) {}
  ~WindowPropertyAppearanceProvider() override = default;

  SkColor GetTitleColor() override {
    return window_->GetProperty(kFrameTextColorKey);
  }

  SkColor GetFrameHeaderColor(bool active) override {
    return window_->GetProperty(active ? kFrameActiveColorKey
                                       : kFrameInactiveColorKey);
  }

  gfx::ImageSkia GetFrameHeaderImage(bool active) override {
    return LookUpImageForProperty(active ? kFrameImageActiveKey
                                         : kFrameImageInactiveKey);
  }

  gfx::ImageSkia GetFrameHeaderOverlayImage(bool active) override {
    return LookUpImageForProperty(active ? kFrameImageOverlayActiveKey
                                         : kFrameImageOverlayInactiveKey);
  }

  bool IsTabletMode() const override {
    return Shell::Get()
        ->tablet_mode_controller()
        ->IsTabletModeWindowManagerEnabled();
  }

 private:
  gfx::ImageSkia LookUpImageForProperty(
      const aura::WindowProperty<base::UnguessableToken*>* property_key) {
    const base::UnguessableToken* token = window_->GetProperty(property_key);
    const gfx::ImageSkia* image =
        token ? Shell::Get()->client_image_registry()->GetImage(*token)
              : nullptr;

    return image ? *image : gfx::ImageSkia();
  }

  aura::Window* window_;

  DISALLOW_COPY_AND_ASSIGN(WindowPropertyAppearanceProvider);
};

}  // namespace

// The view used to draw the content (background and title string)
// of the header. This is a separate view so that it can use
// different scaling strategy than the rest of the frame such
// as caption buttons.
class HeaderView::HeaderContentView : public views::View {
 public:
  HeaderContentView(HeaderView* header_view) : header_view_(header_view) {}
  ~HeaderContentView() override = default;

  // views::View:
  views::PaintInfo::ScaleType GetPaintScaleType() const override {
    return scale_type_;
  }
  void OnPaint(gfx::Canvas* canvas) override {
    header_view_->PaintHeaderContent(canvas);
  }

  void SetScaleType(views::PaintInfo::ScaleType scale_type) {
    scale_type_ = scale_type;
  }

 private:
  HeaderView* header_view_;
  views::PaintInfo::ScaleType scale_type_ =
      views::PaintInfo::ScaleType::kScaleWithEdgeSnapping;
  DISALLOW_COPY_AND_ASSIGN(HeaderContentView);
};

HeaderView::HeaderView(views::Widget* target_widget,
                       mojom::WindowStyle window_style,
                       std::unique_ptr<CaptionButtonModel> model)
    : target_widget_(target_widget),
      avatar_icon_(nullptr),
      header_content_view_(new HeaderContentView(this)),
      caption_button_container_(nullptr),
      fullscreen_visible_fraction_(0),
      should_paint_(true) {
  AddChildView(header_content_view_);

  caption_button_container_ =
      new FrameCaptionButtonContainerView(target_widget_, std::move(model));
  caption_button_container_->UpdateCaptionButtonState(false /*=animate*/);
  AddChildView(caption_button_container_);

  if (window_style == mojom::WindowStyle::DEFAULT) {
    frame_header_ = std::make_unique<DefaultFrameHeader>(
        target_widget, this, caption_button_container_);
  } else {
    DCHECK_EQ(mojom::WindowStyle::BROWSER, window_style);
    DCHECK(!::features::IsAshInBrowserProcess());
    appearance_provider_ = std::make_unique<WindowPropertyAppearanceProvider>(
        target_widget_->GetNativeWindow());
    auto frame_header = std::make_unique<CustomFrameHeader>(
        target_widget, this, appearance_provider_.get(),
        caption_button_container_);
    frame_header_ = std::move(frame_header);
  }

  UpdateBackButton();

  aura::Window* window = target_widget->GetNativeWindow();
  frame_header_->SetFrameColors(window->GetProperty(kFrameActiveColorKey),
                                window->GetProperty(kFrameInactiveColorKey));
  window_observer_.Add(target_widget_->GetNativeWindow());
  Shell::Get()->tablet_mode_controller()->AddObserver(this);
}

HeaderView::~HeaderView() {
  if (Shell::Get()->tablet_mode_controller())
    Shell::Get()->tablet_mode_controller()->RemoveObserver(this);
}

void HeaderView::SchedulePaintForTitle() {
  frame_header_->SchedulePaintForTitle();
}

void HeaderView::ResetWindowControls() {
  caption_button_container_->ResetWindowControls();
}

int HeaderView::GetPreferredOnScreenHeight() {
  if (is_immersive_delegate_ && in_immersive_mode_) {
    return static_cast<int>(GetPreferredHeight() *
                            fullscreen_visible_fraction_);
  }
  return GetPreferredHeight();
}

int HeaderView::GetPreferredHeight() {
  // Calculating the preferred height requires at least one Layout().
  if (!did_layout_)
    Layout();
  return frame_header_->GetHeaderHeightForPainting();
}

int HeaderView::GetMinimumWidth() const {
  return frame_header_->GetMinimumHeaderWidth();
}

void HeaderView::SetAvatarIcon(const gfx::ImageSkia& avatar) {
  const bool show = !avatar.isNull();
  if (!show) {
    if (!avatar_icon_)
      return;
    delete avatar_icon_;
    avatar_icon_ = nullptr;
  } else {
    DCHECK_EQ(avatar.width(), avatar.height());
    if (!avatar_icon_) {
      avatar_icon_ = new views::ImageView();
      AddChildView(avatar_icon_);
    }
    avatar_icon_->SetImage(avatar);
  }
  frame_header_->SetLeftHeaderView(avatar_icon_);
  Layout();
}

void HeaderView::UpdateCaptionButtons() {
  caption_button_container_->ResetWindowControls();
  caption_button_container_->UpdateCaptionButtonState(true /*=animate*/);

  UpdateBackButton();

  Layout();
}

void HeaderView::SetWidthInPixels(int width_in_pixels) {
  frame_header_->SetWidthInPixels(width_in_pixels);
  // If the width is given in pixels, use uniform scaling
  // so that UndoDeviceScaleFactor can correctly undo the scaling.
  header_content_view_->SetScaleType(
      width_in_pixels > 0
          ? views::PaintInfo::ScaleType::kUniformScaling
          : views::PaintInfo::ScaleType::kScaleWithEdgeSnapping);
}

///////////////////////////////////////////////////////////////////////////////
// HeaderView, views::View overrides:

void HeaderView::Layout() {
  did_layout_ = true;
  header_content_view_->SetBoundsRect(GetLocalBounds());
  frame_header_->LayoutHeader();
}

void HeaderView::ChildPreferredSizeChanged(views::View* child) {
  if (child != caption_button_container_)
    return;

  // May be null during view initialization.
  if (parent())
    parent()->Layout();
}

void HeaderView::OnTabletModeStarted() {
  caption_button_container_->UpdateCaptionButtonState(true /*=animate*/);
  parent()->Layout();
  if (Shell::Get()->tablet_mode_controller()->ShouldAutoHideTitlebars(
          target_widget_)) {
    target_widget_->non_client_view()->Layout();
  }
}

void HeaderView::OnTabletModeEnded() {
  caption_button_container_->UpdateCaptionButtonState(true /*=animate*/);
  parent()->Layout();
  target_widget_->non_client_view()->Layout();
}

void HeaderView::OnWindowPropertyChanged(aura::Window* window,
                                         const void* key,
                                         intptr_t old) {
  DCHECK_EQ(target_widget_->GetNativeWindow(), window);
  if (key == kFrameImageActiveKey || key == kFrameImageInactiveKey ||
      key == kFrameImageOverlayActiveKey ||
      key == kFrameImageOverlayInactiveKey) {
    SchedulePaint();
  } else if (key == aura::client::kAvatarIconKey) {
    gfx::ImageSkia* const avatar_icon =
        window->GetProperty(aura::client::kAvatarIconKey);
    SetAvatarIcon(avatar_icon ? *avatar_icon : gfx::ImageSkia());
  } else if (key == kFrameActiveColorKey || key == kFrameInactiveColorKey) {
    frame_header_->SetFrameColors(window->GetProperty(kFrameActiveColorKey),
                                  window->GetProperty(kFrameInactiveColorKey));
  } else if (key == kFrameBackButtonStateKey) {
    UpdateCaptionButtons();
  } else if (key == aura::client::kShowStateKey) {
    frame_header_->OnShowStateChanged(
        window->GetProperty(aura::client::kShowStateKey));
  }
}

void HeaderView::OnWindowDestroying(aura::Window* window) {
  window_observer_.Remove(window);
}

views::View* HeaderView::avatar_icon() const {
  return avatar_icon_;
}

void HeaderView::SetShouldPaintHeader(bool paint) {
  if (should_paint_ == paint)
    return;

  should_paint_ = paint;
  caption_button_container_->SetVisible(should_paint_);
  SchedulePaint();
}

FrameCaptionButton* HeaderView::GetBackButton() {
  return frame_header_->GetBackButton();
}

///////////////////////////////////////////////////////////////////////////////
// HeaderView,
//   ImmersiveFullscreenControllerDelegate overrides:

void HeaderView::OnImmersiveRevealStarted() {
  fullscreen_visible_fraction_ = 0;
  SetPaintToLayer();
  // AppWindow may call this before being added to the widget.
  // https://crbug.com/825260.
  if (layer()->parent()) {
    // The immersive layer should always be top.
    layer()->parent()->StackAtTop(layer());
  }
  parent()->Layout();
}

void HeaderView::OnImmersiveRevealEnded() {
  fullscreen_visible_fraction_ = 0;
  DestroyLayer();
  parent()->Layout();
}

void HeaderView::OnImmersiveFullscreenEntered() {
  in_immersive_mode_ = true;
}

void HeaderView::OnImmersiveFullscreenExited() {
  in_immersive_mode_ = false;
  fullscreen_visible_fraction_ = 0;
  DestroyLayer();
  parent()->Layout();
}

void HeaderView::SetVisibleFraction(double visible_fraction) {
  if (fullscreen_visible_fraction_ != visible_fraction) {
    fullscreen_visible_fraction_ = visible_fraction;
    parent()->Layout();
  }
}

std::vector<gfx::Rect> HeaderView::GetVisibleBoundsInScreen() const {
  // TODO(pkotwicz): Implement views::View::ConvertRectToScreen().
  gfx::Rect visible_bounds(GetVisibleBounds());
  gfx::Point visible_origin_in_screen(visible_bounds.origin());
  views::View::ConvertPointToScreen(this, &visible_origin_in_screen);
  std::vector<gfx::Rect> bounds_in_screen;
  bounds_in_screen.push_back(
      gfx::Rect(visible_origin_in_screen, visible_bounds.size()));
  return bounds_in_screen;
}

void HeaderView::PaintHeaderContent(gfx::Canvas* canvas) {
  if (!should_paint_)
    return;

  bool paint_as_active =
      target_widget_->non_client_view()->frame_view()->ShouldPaintAsActive();
  frame_header_->SetPaintAsActive(paint_as_active);

  FrameHeader::Mode header_mode =
      paint_as_active ? FrameHeader::MODE_ACTIVE : FrameHeader::MODE_INACTIVE;
  frame_header_->PaintHeader(canvas, header_mode);
}

void HeaderView::UpdateBackButton() {
  bool has_back_button =
      caption_button_container_->model()->IsVisible(CAPTION_BUTTON_ICON_BACK);
  FrameCaptionButton* back_button = frame_header_->GetBackButton();
  if (has_back_button) {
    if (!back_button) {
      back_button = new FrameBackButton();
      AddChildView(back_button);
      frame_header_->SetBackButton(back_button);
    }
    back_button->SetEnabled(caption_button_container_->model()->IsEnabled(
        CAPTION_BUTTON_ICON_BACK));
  } else {
    delete back_button;
    frame_header_->SetBackButton(nullptr);
  }
}

}  // namespace ash
