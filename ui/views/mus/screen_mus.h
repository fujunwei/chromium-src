// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef UI_VIEWS_MUS_SCREEN_MUS_H_
#define UI_VIEWS_MUS_SCREEN_MUS_H_

#include "services/ui/public/interfaces/screen_provider.mojom.h"
#include "ui/display/screen_base.h"
#include "ui/views/mus/mus_export.h"

namespace views {

class ScreenMusDelegate;

// Screen implementation that gets information from
// ui::mojom::ScreenProviderObserver.
class VIEWS_MUS_EXPORT ScreenMus : public display::ScreenBase,
                                   public ui::mojom::ScreenProviderObserver {
 public:
  explicit ScreenMus(ScreenMusDelegate* delegate);
  ~ScreenMus() override;

  // ui::mojom::ScreenProviderObserver:
  void OnDisplaysChanged(std::vector<ui::mojom::WsDisplayPtr> ws_displays,
                         int64_t primary_display_id,
                         int64_t internal_display_id) override;

 private:
  friend class ScreenMusTestApi;

  // display::Screen:
  display::Display GetDisplayNearestWindow(
      gfx::NativeWindow window) const override;
  gfx::Point GetCursorScreenPoint() override;
  bool IsWindowUnderCursor(gfx::NativeWindow window) override;
  aura::Window* GetWindowAtScreenPoint(const gfx::Point& point) override;

  ScreenMusDelegate* delegate_;

  DISALLOW_COPY_AND_ASSIGN(ScreenMus);
};

}  // namespace views

#endif  // UI_VIEWS_MUS_SCREEN_MUS_H_
