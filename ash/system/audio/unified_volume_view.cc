// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ash/system/audio/unified_volume_view.h"

#include "ash/resources/vector_icons/vector_icons.h"
#include "ash/strings/grit/ash_strings.h"
#include "ash/system/audio/unified_volume_slider_controller.h"
#include "ash/system/tray/tray_constants.h"
#include "ash/system/tray/tray_popup_utils.h"
#include "components/vector_icons/vector_icons.h"
#include "ui/base/l10n/l10n_util.h"
#include "ui/gfx/canvas.h"
#include "ui/gfx/image/image_skia_operations.h"
#include "ui/gfx/paint_vector_icon.h"
#include "ui/views/animation/flood_fill_ink_drop_ripple.h"
#include "ui/views/animation/ink_drop_highlight.h"
#include "ui/views/animation/ink_drop_impl.h"
#include "ui/views/animation/ink_drop_mask.h"
#include "ui/views/controls/image_view.h"
#include "ui/views/layout/box_layout.h"

using chromeos::CrasAudioHandler;

namespace ash {

namespace {

// Threshold to ignore update on the slider value.
const float kSliderIgnoreUpdateThreshold = 0.01;

// References to the icons that correspond to different volume levels.
const gfx::VectorIcon* const kVolumeLevelIcons[] = {
    &kUnifiedMenuVolumeMuteIcon,    // Muted.
    &kUnifiedMenuVolumeLowIcon,     // Low volume.
    &kUnifiedMenuVolumeMediumIcon,  // Medium volume.
    &kUnifiedMenuVolumeHighIcon,    // High volume.
    &kUnifiedMenuVolumeHighIcon,    // Full volume.
};

// The maximum index of kVolumeLevelIcons.
constexpr int kVolumeLevels = arraysize(kVolumeLevelIcons) - 1;

// Get vector icon reference that corresponds to the given volume level. |level|
// is between 0.0 to 1.0.
const gfx::VectorIcon& GetVolumeIconForLevel(float level) {
  int index = static_cast<int>(std::ceil(level * kVolumeLevels));
  if (index < 0)
    index = 0;
  else if (index > kVolumeLevels)
    index = kVolumeLevels;
  return *kVolumeLevelIcons[index];
}

class MoreButton : public views::Button {
 public:
  explicit MoreButton(views::ButtonListener* listener)
      : views::Button(listener) {
    SetLayoutManager(std::make_unique<views::BoxLayout>(
        views::BoxLayout::kHorizontal,
        gfx::Insets((kTrayItemSize -
                     GetDefaultSizeOfVectorIcon(vector_icons::kHeadsetIcon)) /
                    2),
        2));

    auto* headset = new views::ImageView();
    headset->set_can_process_events_within_subtree(false);
    headset->SetImage(
        CreateVectorIcon(vector_icons::kHeadsetIcon, kUnifiedMenuIconColor));
    AddChildView(headset);

    auto* more = new views::ImageView();
    more->set_can_process_events_within_subtree(false);
    more->SetImage(gfx::ImageSkiaOperations::CreateRotatedImage(
        CreateVectorIcon(kUnifiedMenuExpandIcon, kUnifiedMenuIconColor),
        SkBitmapOperations::ROTATION_90_CW));
    AddChildView(more);

    SetTooltipText(l10n_util::GetStringUTF16(IDS_ASH_STATUS_TRAY_AUDIO));
    TrayPopupUtils::ConfigureTrayPopupButton(this);
  }

  ~MoreButton() override = default;

  // views::Button:
  void PaintButtonContents(gfx::Canvas* canvas) override {
    gfx::Rect rect(GetContentsBounds());
    cc::PaintFlags flags;
    flags.setAntiAlias(true);
    flags.setColor(kUnifiedMenuButtonColor);
    flags.setStyle(cc::PaintFlags::kFill_Style);
    canvas->DrawRoundRect(rect, kTrayItemSize / 2, flags);
  }

  std::unique_ptr<views::InkDrop> CreateInkDrop() override {
    return TrayPopupUtils::CreateInkDrop(this);
  }

  std::unique_ptr<views::InkDropRipple> CreateInkDropRipple() const override {
    return TrayPopupUtils::CreateInkDropRipple(
        TrayPopupInkDropStyle::FILL_BOUNDS, this,
        GetInkDropCenterBasedOnLastEvent(), kUnifiedMenuIconColor);
  }

  std::unique_ptr<views::InkDropHighlight> CreateInkDropHighlight()
      const override {
    return TrayPopupUtils::CreateInkDropHighlight(
        TrayPopupInkDropStyle::FILL_BOUNDS, this, kUnifiedMenuIconColor);
  }

  std::unique_ptr<views::InkDropMask> CreateInkDropMask() const override {
    return std::make_unique<views::RoundRectInkDropMask>(size(), gfx::Insets(),
                                                         kTrayItemSize / 2);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(MoreButton);
};

}  // namespace

UnifiedVolumeView::UnifiedVolumeView(UnifiedVolumeSliderController* controller,
                                     bool is_main_view)
    : UnifiedSliderView(controller,
                        kSystemMenuVolumeHighIcon,
                        IDS_ASH_STATUS_TRAY_VOLUME),
      more_button_(new MoreButton(controller)),
      is_main_view_(is_main_view) {
  DCHECK(CrasAudioHandler::IsInitialized());
  CrasAudioHandler::Get()->AddAudioObserver(this);
  AddChildView(more_button_);
  Update(false /* by_user */);
}

UnifiedVolumeView::~UnifiedVolumeView() {
  DCHECK(CrasAudioHandler::IsInitialized());
  CrasAudioHandler::Get()->RemoveAudioObserver(this);
}

void UnifiedVolumeView::Update(bool by_user) {
  bool is_muted = CrasAudioHandler::Get()->IsOutputMuted();
  float level = CrasAudioHandler::Get()->GetOutputVolumePercent() / 100.f;

  // Indicate that the slider is inactive when it's muted.
  slider()->UpdateState(!is_muted);

  button()->SetToggled(!is_muted);
  button()->SetVectorIcon(GetVolumeIconForLevel(is_muted ? 0.f : level));

  more_button_->SetVisible(is_main_view_ &&
                           (CrasAudioHandler::Get()->has_alternative_input() ||
                            CrasAudioHandler::Get()->has_alternative_output()));

  // Slider's value is in finer granularity than audio volume level(0.01),
  // there will be a small discrepancy between slider's value and volume level
  // on audio side. To avoid the jittering in slider UI, do not set change
  // slider value if the change is less than the threshold.
  if (std::abs(level - slider()->value()) < kSliderIgnoreUpdateThreshold)
    return;

  SetSliderValue(level, by_user);
}

void UnifiedVolumeView::OnOutputNodeVolumeChanged(uint64_t node_id,
                                                  int volume) {
  Update(true /* by_user */);
}

void UnifiedVolumeView::OnOutputMuteChanged(bool mute_on, bool system_adjust) {
  Update(true /* by_user */);
}

void UnifiedVolumeView::OnAudioNodesChanged() {
  Update(true /* by_user */);
}

void UnifiedVolumeView::OnActiveOutputNodeChanged() {
  Update(true /* by_user */);
}

void UnifiedVolumeView::OnActiveInputNodeChanged() {
  Update(true /* by_user */);
}

void UnifiedVolumeView::ChildVisibilityChanged(views::View* child) {
  Update(true /* by_user */);
}

}  // namespace ash
