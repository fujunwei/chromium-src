// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_UI_VIEWS_AUTOFILL_SAVE_CARD_BUBBLE_VIEWS_H_
#define CHROME_BROWSER_UI_VIEWS_AUTOFILL_SAVE_CARD_BUBBLE_VIEWS_H_

#include "chrome/browser/ui/autofill/save_card_bubble_view.h"
#include "chrome/browser/ui/sync/bubble_sync_promo_delegate.h"
#include "chrome/browser/ui/views/location_bar/location_bar_bubble_delegate_view.h"
#include "components/autofill/core/browser/ui/save_card_bubble_controller.h"

namespace content {
class WebContents;
}

namespace autofill {

// This class serves as a base view to any of the bubble views that are part of
// the flow for when the user submits a form with a credit card number that
// Autofill has not previously saved. The base view establishes the button
// handlers, the calculated size, the Super G logo, testing methods, the
// SyncPromoDelegate and the window title (controller eventually handles the
// title for each sub-class).
class SaveCardBubbleViews : public SaveCardBubbleView,
                            public LocationBarBubbleDelegateView {
 public:
  // Bubble will be anchored to |anchor_view|.
  SaveCardBubbleViews(views::View* anchor_view,
                      const gfx::Point& anchor_point,
                      content::WebContents* web_contents,
                      SaveCardBubbleController* controller);

  void Show(DisplayReason reason);

  // SaveCardBubbleView:
  void Hide() override;

  // views::BubbleDialogDelegateView:
  views::View* CreateFootnoteView() override;
  bool Accept() override;
  bool Cancel() override;
  bool Close() override;
  int GetDialogButtons() const override;

  // views::View:
  gfx::Size CalculatePreferredSize() const override;
  void AddedToWidget() override;

  // views::WidgetDelegate:
  bool ShouldShowCloseButton() const override;
  base::string16 GetWindowTitle() const override;
  void WindowClosing() override;

  // Returns the footnote view, so it can be searched for clickable views.
  // Exists for testing (specifically, browsertests).
  views::View* GetFootnoteViewForTesting();

 protected:
  // Delegate for the personalized sync promo view used when desktop identity
  // consistency is enabled.
  class SyncPromoDelegate : public BubbleSyncPromoDelegate {
   public:
    explicit SyncPromoDelegate(SaveCardBubbleController* controller);

    // BubbleSyncPromoDelegate:
    void OnEnableSync(const AccountInfo& account,
                      bool is_default_promo_account) override;

   private:
    SaveCardBubbleController* controller_;

    DISALLOW_COPY_AND_ASSIGN(SyncPromoDelegate);
  };

  // Create the dialog's content view containing everything except for the
  // footnote.
  virtual std::unique_ptr<views::View> CreateMainContentView();

  // Set the footnote view so that its accessible for testing.
  void SetFootnoteViewForTesting(views::View* footnote_view);

  SaveCardBubbleController* controller() {
    return controller_;
  };  // Weak reference.

  // Attributes IDs to the DialogClientView and its buttons.
  void AssignIdsToDialogClientView();

  // views::BubbleDialogDelegateView:
  void Init() override;

  std::unique_ptr<SyncPromoDelegate> sync_promo_delegate_;

  ~SaveCardBubbleViews() override;

 private:
  FRIEND_TEST_ALL_PREFIXES(
      SaveCardBubbleViewsFullFormBrowserTest,
      Upload_ClickingCloseClosesBubbleIfSecondaryUiMdExpOn);
  FRIEND_TEST_ALL_PREFIXES(
      SaveCardBubbleViewsFullFormBrowserTest,
      Upload_DecliningUploadDoesNotLogUserAcceptedCardOriginUMA);

  views::View* footnote_view_ = nullptr;

  SaveCardBubbleController* controller_;  // Weak reference.

  DISALLOW_COPY_AND_ASSIGN(SaveCardBubbleViews);
};

}  // namespace autofill

#endif  // CHROME_BROWSER_UI_VIEWS_AUTOFILL_SAVE_CARD_BUBBLE_VIEWS_H_
