// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef ASH_SYSTEM_UNIFIED_IME_MODE_VIEW_H_
#define ASH_SYSTEM_UNIFIED_IME_MODE_VIEW_H_

#include "ash/system/ime/ime_observer.h"
#include "ash/system/tray/tray_item_view.h"
#include "ash/wm/tablet_mode/tablet_mode_observer.h"
#include "base/macros.h"

namespace ash {

// An IME mode icon view in UnifiedSystemTray button.
class ImeModeView : public TrayItemView,
                    public IMEObserver,
                    public TabletModeObserver {
 public:
  ImeModeView();
  ~ImeModeView() override;

  // IMEObserver:
  void OnIMERefresh() override;
  void OnIMEMenuActivationChanged(bool is_active) override;

  // TabletModeObserver:
  void OnTabletModeStarted() override;
  void OnTabletModeEnded() override;

 private:
  void Update();

  bool ime_menu_on_shelf_activated_ = false;

  DISALLOW_COPY_AND_ASSIGN(ImeModeView);
};

}  // namespace ash

#endif  // ASH_SYSTEM_UNIFIED_IME_MODE_VIEW_H_
