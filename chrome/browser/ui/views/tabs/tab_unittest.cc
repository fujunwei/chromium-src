// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/ui/views/tabs/tab.h"

#include <stddef.h>

#include "base/macros.h"
#include "base/strings/utf_string_conversions.h"
#include "chrome/browser/ui/layout_constants.h"
#include "chrome/browser/ui/tabs/tab_utils.h"
#include "chrome/browser/ui/views/tabs/alert_indicator_button.h"
#include "chrome/browser/ui/views/tabs/tab_close_button.h"
#include "chrome/browser/ui/views/tabs/tab_controller.h"
#include "chrome/browser/ui/views/tabs/tab_icon.h"
#include "chrome/grit/theme_resources.h"
#include "chrome/test/views/chrome_views_test_base.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "ui/base/material_design/material_design_controller.h"
#include "ui/base/models/list_selection_model.h"
#include "ui/gfx/color_palette.h"
#include "ui/gfx/favicon_size.h"
#include "ui/views/controls/button/image_button.h"
#include "ui/views/controls/label.h"
#include "ui/views/widget/widget.h"

using views::Widget;

class FakeTabController : public TabController {
 public:
  FakeTabController() {}

  void set_active_tab(bool value) { active_tab_ = value; }
  void set_paint_throbber_to_layer(bool value) {
    paint_throbber_to_layer_ = value;
  }

  const ui::ListSelectionModel& GetSelectionModel() const override {
    return selection_model_;
  }
  bool SupportsMultipleSelection() override { return false; }
  NewTabButtonPosition GetNewTabButtonPosition() const override {
    return LEADING;
  }
  bool ShouldHideCloseButtonForTab(Tab* tab) const override { return false; }
  bool ShouldShowCloseButtonOnHover() override { return false; }
  bool MaySetClip() override { return false; }
  void SelectTab(Tab* tab) override {}
  void ExtendSelectionTo(Tab* tab) override {}
  void ToggleSelected(Tab* tab) override {}
  void AddSelectionFromAnchorTo(Tab* tab) override {}
  void CloseTab(Tab* tab, CloseTabSource source) override {}
  void ToggleTabAudioMute(Tab* tab) override {}
  void ShowContextMenuForTab(Tab* tab,
                             const gfx::Point& p,
                             ui::MenuSourceType source_type) override {}
  bool IsActiveTab(const Tab* tab) const override { return active_tab_; }
  bool IsTabSelected(const Tab* tab) const override { return false; }
  bool IsTabPinned(const Tab* tab) const override { return false; }
  bool IsFirstVisibleTab(const Tab* tab) const override { return false; }
  bool IsLastVisibleTab(const Tab* tab) const override { return false; }
  bool SingleTabMode() const override { return false; }
  bool IsIncognito() const override { return false; }
  void MaybeStartDrag(
      Tab* tab,
      const ui::LocatedEvent& event,
      const ui::ListSelectionModel& original_selection) override {}
  void ContinueDrag(views::View* view, const ui::LocatedEvent& event) override {
  }
  bool EndDrag(EndDragReason reason) override { return false; }
  Tab* GetTabAt(const gfx::Point& point) override { return nullptr; }
  const Tab* GetSubsequentTab(const Tab* tab) override { return nullptr; }
  void OnMouseEventInTab(views::View* source,
                         const ui::MouseEvent& event) override {}
  bool ShouldPaintTab(
      const Tab* tab,
      const base::RepeatingCallback<gfx::Path(const gfx::Rect&)>&
          border_callback,
      gfx::Path* clip) override {
    return true;
  }
  bool CanPaintThrobberToLayer() const override {
    return paint_throbber_to_layer_;
  }
  SkColor GetToolbarTopSeparatorColor() const override { return SK_ColorBLACK; }
  SkColor GetTabSeparatorColor() const override { return SK_ColorBLACK; }
  SkColor GetTabBackgroundColor(TabState state) const override {
    return gfx::kPlaceholderColor;
  }
  SkColor GetTabForegroundColor(TabState state) const override {
    return gfx::kPlaceholderColor;
  }
  int GetBackgroundResourceId(bool* custom_image) const override {
    *custom_image = false;
    return IDR_THEME_TAB_BACKGROUND;
  }
  gfx::Rect GetTabAnimationTargetBounds(const Tab* tab) override {
    return tab->bounds();
  }
  base::string16 GetAccessibleTabName(const Tab* tab) const override {
    return base::string16();
  }

 private:
  ui::ListSelectionModel selection_model_;
  bool active_tab_ = false;
  bool paint_throbber_to_layer_ = true;

  DISALLOW_COPY_AND_ASSIGN(FakeTabController);
};

class TabTest : public ChromeViewsTestBase {
 public:
  TabTest() {}
  ~TabTest() override {}

  static TabIcon* GetTabIcon(const Tab& tab) { return tab.icon_; }

  static views::Label* GetTabTitle(const Tab& tab) { return tab.title_; }

  static views::ImageButton* GetAlertIndicator(const Tab& tab) {
    return tab.alert_indicator_button_;
  }

  static views::ImageButton* GetCloseButton(const Tab& tab) {
    return tab.close_button_;
  }

  static int GetTitleWidth(const Tab& tab) {
    return tab.title_->bounds().width();
  }

  static void EndTitleAnimation(Tab& tab) { tab.title_animation_.End(); }

  static void LayoutTab(Tab* tab) { tab->Layout(); }

  static int VisibleIconCount(const Tab& tab) {
    return tab.showing_icon_ + tab.showing_alert_indicator_ +
           tab.showing_close_button_;
  }

  static void CheckForExpectedLayoutAndVisibilityOfElements(const Tab& tab) {
    // Check whether elements are visible when they are supposed to be, given
    // Tab size and TabRendererData state.
    if (tab.data_.pinned) {
      EXPECT_EQ(1, VisibleIconCount(tab));
      if (tab.data_.alert_state != TabAlertState::NONE) {
        EXPECT_FALSE(tab.showing_icon_);
        EXPECT_TRUE(tab.showing_alert_indicator_);
      } else {
        EXPECT_TRUE(tab.showing_icon_);
        EXPECT_FALSE(tab.showing_alert_indicator_);
      }
      EXPECT_FALSE(tab.title_->visible());
      EXPECT_FALSE(tab.showing_close_button_);
    } else if (tab.IsActive()) {
      EXPECT_TRUE(tab.showing_close_button_);
      switch (VisibleIconCount(tab)) {
        case 1:
          EXPECT_FALSE(tab.showing_icon_);
          EXPECT_FALSE(tab.showing_alert_indicator_);
          break;
        case 2:
          if (tab.data_.alert_state != TabAlertState::NONE) {
            EXPECT_FALSE(tab.showing_icon_);
            EXPECT_TRUE(tab.showing_alert_indicator_);
          } else {
            EXPECT_TRUE(tab.showing_icon_);
            EXPECT_FALSE(tab.showing_alert_indicator_);
          }
          break;
        default:
          EXPECT_EQ(3, VisibleIconCount(tab));
          EXPECT_TRUE(tab.data_.alert_state != TabAlertState::NONE);
          break;
      }
    } else {  // Tab not active and not pinned tab.
      switch (VisibleIconCount(tab)) {
        case 1:
          EXPECT_FALSE(tab.showing_close_button_);
          if (tab.data_.alert_state == TabAlertState::NONE ||
              tab.center_favicon_)
            EXPECT_FALSE(tab.showing_alert_indicator_);
          if (tab.center_favicon_)
            EXPECT_TRUE(tab.showing_icon_);
          break;
        case 2:
          EXPECT_TRUE(tab.showing_icon_);
          if (tab.data_.alert_state != TabAlertState::NONE)
            EXPECT_TRUE(tab.showing_alert_indicator_);
          else
            EXPECT_FALSE(tab.showing_alert_indicator_);
          break;
        default:
          EXPECT_EQ(3, VisibleIconCount(tab));
          EXPECT_TRUE(tab.data_.alert_state != TabAlertState::NONE);
      }
    }

    // Check positioning of elements with respect to each other, and that they
    // are fully within the contents bounds.
    const gfx::Rect contents_bounds = tab.GetContentsBounds();
    if (tab.showing_icon_) {
      if (tab.center_favicon_) {
        EXPECT_LE(tab.icon_->x(), contents_bounds.x());
      } else {
        EXPECT_LE(contents_bounds.x(), tab.icon_->x());
      }
      if (tab.title_->visible())
        EXPECT_LE(tab.icon_->bounds().right(), tab.title_->x());
      EXPECT_LE(contents_bounds.y(), tab.icon_->y());
      EXPECT_LE(tab.icon_->bounds().bottom(), contents_bounds.bottom());
    }

    if (tab.showing_icon_ && tab.showing_alert_indicator_) {
      // When checking for overlap, other views should not overlap the main
      // favicon (covered by kFaviconSize) but can overlap the extra space
      // reserved for the attention indicator.
      int icon_visual_right = tab.icon_->bounds().x() + gfx::kFaviconSize;
      EXPECT_LE(icon_visual_right, GetAlertIndicatorBounds(tab).x());
    }

    if (tab.showing_alert_indicator_) {
      if (tab.title_->visible()) {
        EXPECT_LE(tab.title_->bounds().right(),
                  GetAlertIndicatorBounds(tab).x());
      }
      EXPECT_LE(GetAlertIndicatorBounds(tab).right(), contents_bounds.right());
      EXPECT_LE(contents_bounds.y(), GetAlertIndicatorBounds(tab).y());
      EXPECT_LE(GetAlertIndicatorBounds(tab).bottom(),
                contents_bounds.bottom());
    }
    if (tab.showing_alert_indicator_ && tab.showing_close_button_) {
      // Note: The alert indicator can overlap the left-insets of the close box,
      // but should otherwise be to the left of the close button.
      EXPECT_LE(GetAlertIndicatorBounds(tab).right(),
                tab.close_button_->bounds().x() +
                    tab.close_button_->GetInsets().left());
    }
    if (tab.showing_close_button_) {
      // Note: The title bounds can overlap the left-insets of the close box,
      // but should otherwise be to the left of the close button.
      if (tab.title_->visible()) {
        EXPECT_LE(tab.title_->bounds().right(),
                  tab.close_button_->bounds().x() +
                      tab.close_button_->GetInsets().left());
      }
      // We need to use the close button contents bounds instead of its bounds,
      // since it has an empty border around it to extend its clickable area for
      // touch.
      // Note: The close button right edge can be outside the nominal contents
      // bounds, but shouldn't leave the local bounds.
      const gfx::Rect close_bounds = tab.close_button_->GetContentsBounds();
      EXPECT_LE(close_bounds.right(), tab.GetLocalBounds().right());
      EXPECT_LE(contents_bounds.y(), close_bounds.y());
      EXPECT_LE(close_bounds.bottom(), contents_bounds.bottom());
    }
  }

  static void StopFadeAnimationIfNecessary(const Tab& tab) {
    // Stop the fade animation directly instead of waiting an unknown number of
    // seconds.
    if (gfx::Animation* fade_animation =
            tab.alert_indicator_button_->fade_animation_.get()) {
      fade_animation->Stop();
    }
  }

 protected:
  void InitWidget(Widget* widget) {
    Widget::InitParams params(CreateParams(Widget::InitParams::TYPE_WINDOW));
    params.ownership = Widget::InitParams::WIDGET_OWNS_NATIVE_WIDGET;
    params.bounds.SetRect(10, 20, 300, 400);
    widget->Init(params);
  }

 private:
  static gfx::Rect GetAlertIndicatorBounds(const Tab& tab) {
    if (!tab.alert_indicator_button_) {
      ADD_FAILURE();
      return gfx::Rect();
    }
    return tab.alert_indicator_button_->bounds();
  }

  std::string original_locale_;
};

TEST_F(TabTest, HitTestTopPixel) {
  Widget widget;
  InitWidget(&widget);

  FakeTabController tab_controller;
  Tab tab(&tab_controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);
  tab.SizeToPreferredSize();

  // Tabs are slanted, so a click halfway down the left edge won't hit it.
  int middle_y = tab.height() / 2;
  EXPECT_FALSE(tab.HitTestPoint(gfx::Point(0, middle_y)));

  // Tabs should not be hit if we click above them.
  int middle_x = tab.width() / 2;
  EXPECT_FALSE(tab.HitTestPoint(gfx::Point(middle_x, -1)));
  EXPECT_TRUE(tab.HitTestPoint(gfx::Point(middle_x, 0)));

  // Make sure top edge clicks still select the tab when the window is
  // maximized.
  widget.Maximize();
  EXPECT_TRUE(tab.HitTestPoint(gfx::Point(middle_x, 0)));

  // But clicks in the area above the slanted sides should still miss.
  EXPECT_FALSE(tab.HitTestPoint(gfx::Point(0, 0)));
  EXPECT_FALSE(tab.HitTestPoint(gfx::Point(tab.width() - 1, 0)));
}

TEST_F(TabTest, LayoutAndVisibilityOfElements) {
  static const TabAlertState kAlertStatesToTest[] = {
      TabAlertState::NONE,          TabAlertState::TAB_CAPTURING,
      TabAlertState::AUDIO_PLAYING, TabAlertState::AUDIO_MUTING,
      TabAlertState::PIP_PLAYING,
  };

  Widget widget;
  InitWidget(&widget);

  FakeTabController controller;
  Tab tab(&controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);

  SkBitmap bitmap;
  bitmap.allocN32Pixels(16, 16);
  TabRendererData data;
  data.favicon = gfx::ImageSkia::CreateFrom1xBitmap(bitmap);

  // Perform layout over all possible combinations, checking for correct
  // results.
  for (bool is_pinned_tab : {false, true}) {
    for (bool is_active_tab : {false, true}) {
      for (TabAlertState alert_state : kAlertStatesToTest) {
        SCOPED_TRACE(::testing::Message()
                     << (is_active_tab ? "Active " : "Inactive ")
                     << (is_pinned_tab ? "pinned " : "")
                     << "tab with alert indicator state "
                     << static_cast<int>(alert_state));

        data.pinned = is_pinned_tab;
        controller.set_active_tab(is_active_tab);
        data.alert_state = alert_state;
        tab.SetData(data);
        StopFadeAnimationIfNecessary(tab);

        // Test layout for every width from standard to minimum.
        int width, min_width;
        if (is_pinned_tab) {
          width = min_width = Tab::GetPinnedWidth();
        } else {
          width = Tab::GetStandardWidth();
          min_width = is_active_tab ? Tab::GetMinimumActiveWidth()
                                    : Tab::GetMinimumInactiveWidth();
        }
        const int height = GetLayoutConstant(TAB_HEIGHT);
        for (; width >= min_width; --width) {
          SCOPED_TRACE(::testing::Message() << "width=" << width);
          tab.SetBounds(0, 0, width, height);  // Invokes Tab::Layout().
          CheckForExpectedLayoutAndVisibilityOfElements(tab);
        }
      }
    }
  }
}

// Regression test for http://crbug.com/420313: Confirms that any child Views of
// Tab do not attempt to provide their own tooltip behavior/text. It also tests
// that Tab provides the expected tooltip text (according to tab_utils).
TEST_F(TabTest, TooltipProvidedByTab) {
  Widget widget;
  InitWidget(&widget);

  FakeTabController controller;
  Tab tab(&controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);
  tab.SizeToPreferredSize();

  SkBitmap bitmap;
  bitmap.allocN32Pixels(16, 16);
  TabRendererData data;
  data.favicon = gfx::ImageSkia::CreateFrom1xBitmap(bitmap);

  data.title = base::UTF8ToUTF16(
      "This is a really long tab title that would case views::Label to provide "
      "its own tooltip; but Tab should disable that feature so it can provide "
      "the tooltip instead.");

  // Test both with and without an indicator showing since the tab tooltip text
  // should include a description of the alert state when the indicator is
  // present.
  for (int i = 0; i < 2; ++i) {
    data.alert_state =
        (i == 0 ? TabAlertState::NONE : TabAlertState::AUDIO_PLAYING);
    SCOPED_TRACE(::testing::Message()
                 << "Tab with alert indicator state "
                 << static_cast<uint8_t>(data.alert_state));
    tab.SetData(data);

    for (int j = 0; j < tab.child_count(); ++j) {
      views::View& child = *tab.child_at(j);
      if (!strcmp(child.GetClassName(), "TabCloseButton"))
        continue;  // Close button is excepted.
      if (!child.visible())
        continue;
      SCOPED_TRACE(::testing::Message() << "child_at(" << j << "): "
                   << child.GetClassName());

      const gfx::Point midpoint(child.width() / 2, child.height() / 2);
      EXPECT_FALSE(child.GetTooltipHandlerForPoint(midpoint));
      const gfx::Point mouse_hover_point =
          midpoint + child.GetMirroredPosition().OffsetFromOrigin();
      base::string16 tooltip;
      EXPECT_TRUE(static_cast<views::View&>(tab).GetTooltipText(
          mouse_hover_point, &tooltip));
      EXPECT_EQ(chrome::AssembleTabTooltipText(data.title, data.alert_state),
                tooltip);
    }
  }
}

// Regression test for http://crbug.com/226253. Calling Layout() more than once
// shouldn't change the insets of the close button.
TEST_F(TabTest, CloseButtonLayout) {
  FakeTabController tab_controller;
  Tab tab(&tab_controller, nullptr);
  tab.SetBounds(0, 0, 100, 50);
  LayoutTab(&tab);
  gfx::Insets close_button_insets = GetCloseButton(tab)->GetInsets();
  LayoutTab(&tab);
  gfx::Insets close_button_insets_2 = GetCloseButton(tab)->GetInsets();
  EXPECT_EQ(close_button_insets.top(), close_button_insets_2.top());
  EXPECT_EQ(close_button_insets.left(), close_button_insets_2.left());
  EXPECT_EQ(close_button_insets.bottom(), close_button_insets_2.bottom());
  EXPECT_EQ(close_button_insets.right(), close_button_insets_2.right());

  // Also make sure the close button is sized as large as the tab.
  EXPECT_EQ(50, GetCloseButton(tab)->bounds().height());
}

// Regression test for http://crbug.com/609701. Ensure TabCloseButton does not
// get focus on right click.
TEST_F(TabTest, CloseButtonFocus) {
  Widget widget;
  InitWidget(&widget);
  FakeTabController tab_controller;
  Tab tab(&tab_controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);

  views::ImageButton* tab_close_button = GetCloseButton(tab);

  // Verify tab_close_button does not get focus on right click.
  ui::MouseEvent right_click_event(ui::ET_KEY_PRESSED, gfx::Point(),
                                   gfx::Point(), base::TimeTicks(),
                                   ui::EF_RIGHT_MOUSE_BUTTON, 0);
  tab_close_button->OnMousePressed(right_click_event);
  EXPECT_NE(tab_close_button,
            tab_close_button->GetFocusManager()->GetFocusedView());
}

// Tests expected changes to the ThrobberView state when the WebContents loading
// state changes or the animation timer (usually in BrowserView) triggers.
TEST_F(TabTest, LayeredThrobber) {
  Widget widget;
  InitWidget(&widget);

  FakeTabController tab_controller;
  Tab tab(&tab_controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);
  tab.SizeToPreferredSize();

  TabIcon* icon = GetTabIcon(tab);
  TabRendererData data;
  data.url = GURL("http://example.com");
  EXPECT_FALSE(icon->ShowingLoadingAnimation());
  EXPECT_EQ(TabNetworkState::kNone, tab.data().network_state);

  // Simulate a "normal" tab load: should paint to a layer.
  data.network_state = TabNetworkState::kWaiting;
  tab.SetData(data);
  EXPECT_TRUE(tab_controller.CanPaintThrobberToLayer());
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_TRUE(icon->layer());
  data.network_state = TabNetworkState::kLoading;
  tab.SetData(data);
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_TRUE(icon->layer());
  data.network_state = TabNetworkState::kNone;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());

  // Simulate a tab that should hide throbber.
  data.should_hide_throbber = true;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());
  data.network_state = TabNetworkState::kWaiting;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());
  data.network_state = TabNetworkState::kLoading;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());
  data.network_state = TabNetworkState::kNone;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());

  // Simulate a tab that should not hide throbber.
  data.should_hide_throbber = false;
  data.network_state = TabNetworkState::kWaiting;
  tab.SetData(data);
  EXPECT_TRUE(tab_controller.CanPaintThrobberToLayer());
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_TRUE(icon->layer());
  data.network_state = TabNetworkState::kLoading;
  tab.SetData(data);
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_TRUE(icon->layer());
  data.network_state = TabNetworkState::kNone;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());

  // After loading is done, simulate another resource starting to load.
  data.network_state = TabNetworkState::kWaiting;
  tab.SetData(data);
  EXPECT_TRUE(icon->ShowingLoadingAnimation());

  // Reset.
  data.network_state = TabNetworkState::kNone;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());

  // Simulate a drag started and stopped during a load: layer painting stops
  // temporarily.
  data.network_state = TabNetworkState::kWaiting;
  tab.SetData(data);
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_TRUE(icon->layer());
  tab_controller.set_paint_throbber_to_layer(false);
  tab.StepLoadingAnimation();
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_FALSE(icon->layer());
  tab_controller.set_paint_throbber_to_layer(true);
  tab.StepLoadingAnimation();
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_TRUE(icon->layer());
  data.network_state = TabNetworkState::kNone;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());

  // Simulate a tab load starting and stopping during tab dragging (or with
  // stacked tabs): no layer painting.
  tab_controller.set_paint_throbber_to_layer(false);
  data.network_state = TabNetworkState::kWaiting;
  tab.SetData(data);
  EXPECT_TRUE(icon->ShowingLoadingAnimation());
  EXPECT_FALSE(icon->layer());
  data.network_state = TabNetworkState::kNone;
  tab.SetData(data);
  EXPECT_FALSE(icon->ShowingLoadingAnimation());
}

TEST_F(TabTest, TitleHiddenWhenSmall) {
  FakeTabController tab_controller;
  Tab tab(&tab_controller, nullptr);
  tab.SetBounds(0, 0, 100, 50);
  EXPECT_GT(GetTitleWidth(tab), 0);
  tab.SetBounds(0, 0, 0, 50);
  EXPECT_EQ(0, GetTitleWidth(tab));
}

TEST_F(TabTest, FaviconDoesntMoveWhenShowingAlertIndicator) {
  Widget widget;
  InitWidget(&widget);

  for (bool is_active_tab : {false, true}) {
    FakeTabController controller;
    controller.set_active_tab(is_active_tab);
    Tab tab(&controller, nullptr);
    widget.GetContentsView()->AddChildView(&tab);
    tab.SizeToPreferredSize();

    views::View* icon = GetTabIcon(tab);
    int icon_x = icon->x();
    TabRendererData data;
    data.alert_state = TabAlertState::AUDIO_PLAYING;
    tab.SetData(data);
    EXPECT_EQ(icon_x, icon->x());
  }
}

TEST_F(TabTest, SmallTabsHideCloseButton) {
  Widget widget;
  InitWidget(&widget);

  FakeTabController controller;
  controller.set_active_tab(false);
  Tab tab(&controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);
  const int width = Tab::GetContentsInsets().width() +
                    Tab::kMinimumContentsWidthForCloseButtons;
  tab.SetBounds(0, 0, width, 50);
  const views::View* close = GetCloseButton(tab);
  EXPECT_TRUE(close->visible());

  const views::View* icon = GetTabIcon(tab);
  const int icon_x = icon->x();
  // Shrink the tab. The close button should disappear.
  tab.SetBounds(0, 0, width - 1, 50);
  EXPECT_FALSE(close->visible());
  // The favicon moves left because the extra padding disappears too.
  EXPECT_LT(icon->x(), icon_x);
}

TEST_F(TabTest, ExtraLeftPaddingNotShownOnSmallActiveTab) {
  Widget widget;
  InitWidget(&widget);

  FakeTabController controller;
  controller.set_active_tab(true);
  Tab tab(&controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);
  tab.SetBounds(0, 0, 200, 50);
  const views::View* close = GetCloseButton(tab);
  EXPECT_TRUE(close->visible());

  const views::View* icon = GetTabIcon(tab);
  const int icon_x = icon->x();

  tab.SetBounds(0, 0, 40, 50);
  EXPECT_TRUE(close->visible());
  // The favicon moves left because the extra padding disappears.
  EXPECT_LT(icon->x(), icon_x);
}

TEST_F(TabTest, ExtraLeftPaddingShownOnSiteWithoutFavicon) {
  Widget widget;
  InitWidget(&widget);

  FakeTabController controller;
  Tab tab(&controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);

  tab.SizeToPreferredSize();
  const views::View* icon = GetTabIcon(tab);
  const int icon_x = icon->x();

  // Remove the favicon.
  TabRendererData data;
  data.show_icon = false;
  tab.SetData(data);
  EndTitleAnimation(tab);
  EXPECT_FALSE(icon->visible());
  // Title should be placed where the favicon was.
  EXPECT_EQ(icon_x, GetTabTitle(tab)->x());
}

TEST_F(TabTest, ExtraAlertPaddingNotShownOnSmallActiveTab) {
  if (!ui::MaterialDesignController::IsRefreshUi()) {
    // Extra alert padding not shown pre-Refresh.
    return;
  }

  Widget widget;
  InitWidget(&widget);

  FakeTabController controller;
  controller.set_active_tab(true);
  Tab tab(&controller, nullptr);
  widget.GetContentsView()->AddChildView(&tab);
  TabRendererData data;
  data.alert_state = TabAlertState::AUDIO_PLAYING;
  tab.SetData(data);

  tab.SetBounds(0, 0, 200, 50);
  const views::View* close = GetCloseButton(tab);
  const views::View* alert = GetAlertIndicator(tab);
  const int original_spacing = close->x() - alert->bounds().right();

  tab.SetBounds(0, 0, 60, 50);
  EXPECT_FALSE(GetTabIcon(tab)->visible());
  EXPECT_TRUE(close->visible());
  EXPECT_TRUE(alert->visible());
  // The alert indicator moves closer because the extra padding is gone.
  EXPECT_LT(close->x() - alert->bounds().right(), original_spacing);
}
