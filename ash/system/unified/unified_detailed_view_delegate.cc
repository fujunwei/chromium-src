// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ash/system/unified/unified_detailed_view_delegate.h"

#include "ash/resources/vector_icons/vector_icons.h"
#include "ash/strings/grit/ash_strings.h"
#include "ash/system/tray/hover_highlight_view.h"
#include "ash/system/tray/tray_constants.h"
#include "ash/system/tray/tray_popup_item_style.h"
#include "ash/system/tray/tray_popup_utils.h"
#include "ash/system/unified/collapse_button.h"
#include "ash/system/unified/top_shortcut_button.h"
#include "ash/system/unified/unified_system_tray_controller.h"
#include "components/vector_icons/vector_icons.h"
#include "ui/base/l10n/l10n_util.h"
#include "ui/gfx/paint_vector_icon.h"
#include "ui/gfx/vector_icon_types.h"
#include "ui/views/border.h"
#include "ui/views/controls/label.h"
#include "ui/views/controls/separator.h"
#include "ui/views/layout/box_layout.h"

namespace ash {

namespace {

void ConfigureTitleTriView(TriView* tri_view, TriView::Container container) {
  std::unique_ptr<views::BoxLayout> layout;

  switch (container) {
    case TriView::Container::START:
      FALLTHROUGH;
    case TriView::Container::END:
      layout = std::make_unique<views::BoxLayout>(views::BoxLayout::kHorizontal,
                                                  gfx::Insets(),
                                                  kUnifiedTopShortcutSpacing);
      layout->set_main_axis_alignment(
          views::BoxLayout::MAIN_AXIS_ALIGNMENT_CENTER);
      layout->set_cross_axis_alignment(
          views::BoxLayout::CROSS_AXIS_ALIGNMENT_CENTER);
      break;
    case TriView::Container::CENTER:
      tri_view->SetFlexForContainer(TriView::Container::CENTER, 1.f);

      layout = std::make_unique<views::BoxLayout>(views::BoxLayout::kVertical);
      layout->set_main_axis_alignment(
          views::BoxLayout::MAIN_AXIS_ALIGNMENT_CENTER);
      layout->set_cross_axis_alignment(
          views::BoxLayout::CROSS_AXIS_ALIGNMENT_STRETCH);
      break;
  }

  tri_view->SetContainerLayout(container, std::move(layout));
  tri_view->SetMinSize(container,
                       gfx::Size(0, kUnifiedDetailedViewTitleRowHeight));
}

class BackButton : public CustomShapeButton {
 public:
  BackButton(views::ButtonListener* listener) : CustomShapeButton(listener) {
    gfx::ImageSkia image =
        gfx::CreateVectorIcon(kUnifiedMenuArrowBackIcon, kUnifiedMenuIconColor);
    SetImage(views::Button::STATE_NORMAL, image);
    SetImageAlignment(HorizontalAlignment::ALIGN_RIGHT,
                      VerticalAlignment::ALIGN_MIDDLE);
    SetTooltipText(
        l10n_util::GetStringUTF16(IDS_ASH_STATUS_TRAY_PREVIOUS_MENU));
    SetBorder(views::CreateEmptyBorder(
        gfx::Insets((kTrayItemSize - image.width()) / 2)));
  }

  ~BackButton() override = default;

  // CustomShapeButton:
  gfx::Size CalculatePreferredSize() const override {
    return gfx::Size(kTrayItemSize * 3 / 2, kTrayItemSize);
  }

  SkPath CreateCustomShapePath(const gfx::Rect& bounds) const override {
    SkPath path;
    SkScalar bottom_radius = SkIntToScalar(kTrayItemSize / 2);
    SkScalar radii[8] = {
        0, 0, bottom_radius, bottom_radius, bottom_radius, bottom_radius, 0, 0};
    path.addRoundRect(gfx::RectToSkRect(bounds), radii);
    return path;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(BackButton);
};

}  // namespace

UnifiedDetailedViewDelegate::UnifiedDetailedViewDelegate(
    UnifiedSystemTrayController* tray_controller)
    : tray_controller_(tray_controller) {}

UnifiedDetailedViewDelegate::~UnifiedDetailedViewDelegate() = default;

void UnifiedDetailedViewDelegate::TransitionToMainView(bool restore_focus) {
  tray_controller_->TransitionToMainView(restore_focus);
}

void UnifiedDetailedViewDelegate::CloseBubble() {
  tray_controller_->CloseBubble();
}

SkColor UnifiedDetailedViewDelegate::GetBackgroundColor(
    ui::NativeTheme* native_theme) {
  return SK_ColorTRANSPARENT;
}

bool UnifiedDetailedViewDelegate::IsOverflowIndicatorEnabled() const {
  return false;
}

TriView* UnifiedDetailedViewDelegate::CreateTitleRow(int string_id) {
  auto* tri_view = new TriView(kUnifiedTopShortcutSpacing);

  ConfigureTitleTriView(tri_view, TriView::Container::START);
  ConfigureTitleTriView(tri_view, TriView::Container::CENTER);
  ConfigureTitleTriView(tri_view, TriView::Container::END);

  auto* label = TrayPopupUtils::CreateDefaultLabel();
  label->SetText(l10n_util::GetStringUTF16(string_id));
  TrayPopupItemStyle style(TrayPopupItemStyle::FontStyle::TITLE,
                           true /* use_unified_theme */);
  style.SetupLabel(label);
  tri_view->AddView(TriView::Container::CENTER, label);

  tri_view->SetContainerVisible(TriView::Container::END, false);
  tri_view->SetBorder(
      views::CreateEmptyBorder(kUnifiedDetailedViewTitlePadding));

  return tri_view;
}

views::View* UnifiedDetailedViewDelegate::CreateTitleSeparator() {
  views::Separator* separator = new views::Separator();
  separator->SetColor(kUnifiedMenuSeparatorColor);
  separator->SetBorder(views::CreateEmptyBorder(
      kTitleRowProgressBarHeight - views::Separator::kThickness, 0, 0, 0));
  return separator;
}

void UnifiedDetailedViewDelegate::ShowStickyHeaderSeparator(
    views::View* view,
    bool show_separator) {
  if (show_separator) {
    view->SetBorder(views::CreatePaddedBorder(
        views::CreateSolidSidedBorder(0, 0, kSeparatorWidth, 0,
                                      kUnifiedMenuSeparatorColor),
        gfx::Insets(kMenuSeparatorVerticalPadding, 0,
                    kMenuSeparatorVerticalPadding - kSeparatorWidth, 0)));
  } else {
    view->SetBorder(views::CreateEmptyBorder(
        gfx::Insets(kMenuSeparatorVerticalPadding, 0)));
  }
  view->SchedulePaint();
}

views::Separator* UnifiedDetailedViewDelegate::CreateListSubHeaderSeparator() {
  views::Separator* separator = new views::Separator();
  separator->SetColor(kUnifiedMenuSeparatorColor);
  separator->SetBorder(views::CreateEmptyBorder(
      kMenuSeparatorVerticalPadding - views::Separator::kThickness, 0, 0, 0));
  return separator;
}

HoverHighlightView* UnifiedDetailedViewDelegate::CreateScrollListItem(
    ViewClickListener* listener,
    const gfx::VectorIcon& icon,
    const base::string16& text) {
  HoverHighlightView* item =
      new HoverHighlightView(listener, true /* use_unified_theme */);
  if (icon.is_empty())
    item->AddLabelRow(text);
  else
    item->AddIconAndLabel(gfx::CreateVectorIcon(icon, kUnifiedMenuIconColor),
                          text);
  return item;
}

views::Button* UnifiedDetailedViewDelegate::CreateBackButton(
    views::ButtonListener* listener) {
  return new BackButton(listener);
}

views::Button* UnifiedDetailedViewDelegate::CreateInfoButton(
    views::ButtonListener* listener,
    int info_accessible_name_id) {
  return new TopShortcutButton(listener, kUnifiedMenuInfoIcon,
                               info_accessible_name_id);
}

views::Button* UnifiedDetailedViewDelegate::CreateSettingsButton(
    views::ButtonListener* listener,
    int setting_accessible_name_id) {
  auto* button = new TopShortcutButton(listener, kUnifiedMenuSettingsIcon,
                                       setting_accessible_name_id);
  if (!TrayPopupUtils::CanOpenWebUISettings())
    button->SetEnabled(false);
  return button;
}

views::Button* UnifiedDetailedViewDelegate::CreateHelpButton(
    views::ButtonListener* listener) {
  auto* button = new TopShortcutButton(listener, vector_icons::kHelpOutlineIcon,
                                       IDS_ASH_STATUS_TRAY_HELP);
  // Help opens a web page, so treat it like Web UI settings.
  if (!TrayPopupUtils::CanOpenWebUISettings())
    button->SetEnabled(false);
  return button;
}

}  // namespace ash
