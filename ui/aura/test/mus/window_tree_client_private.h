// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef UI_AURA_TEST_MUS_WINDOW_TREE_CLIENT_PRIVATE_H_
#define UI_AURA_TEST_MUS_WINDOW_TREE_CLIENT_PRIVATE_H_

#include <stdint.h>

#include <memory>

#include "base/macros.h"
#include "services/ui/public/interfaces/window_tree_constants.mojom.h"
#include "ui/aura/mus/mus_types.h"
#include "ui/aura/mus/window_tree_client.h"

namespace ui {
class Event;

namespace mojom {
class WindowTree;
}
}

namespace aura {

class EmbedRoot;
class Window;
class WindowMus;
class WindowTreeClientDelegate;
class WindowTreeClient;

enum class ChangeType;

// Use to access implementation details of WindowTreeClient.
class WindowTreeClientPrivate {
 public:
  explicit WindowTreeClientPrivate(WindowTreeClient* tree_client_impl);
  explicit WindowTreeClientPrivate(Window* window);
  ~WindowTreeClientPrivate();

  static std::unique_ptr<WindowTreeClient> CreateWindowTreeClient(
      WindowTreeClientDelegate* window_tree_delegate);

  // Calls OnEmbed() on the WindowTreeClient.
  void OnEmbed(ui::mojom::WindowTree* window_tree);

  // Simulates |event| matching a pointer watcher on the window server.
  void CallOnPointerEventObserved(Window* window,
                                  std::unique_ptr<ui::Event> event);

  void CallOnCaptureChanged(Window* new_capture, Window* old_capture);

  // Simulates the EmbedRoot receiving the token from the WindowTree and then
  // the WindowTree calling OnEmbedFromToken().
  void CallOnEmbedFromToken(EmbedRoot* embed_root);

  // Sets the WindowTree.
  void SetTree(ui::mojom::WindowTree* window_tree);

  bool HasPointerWatcher();

  Window* GetWindowByServerId(ui::Id id);

  WindowMus* NewWindowFromWindowData(WindowMus* parent,
                                     const ui::mojom::WindowData& window_data);

  bool HasInFlightChanges();

  bool HasChangeInFlightOfType(ChangeType type);

 private:
  ui::mojom::WindowDataPtr CreateWindowDataForEmbed();

  WindowTreeClient* tree_client_impl_;
  uint16_t next_window_id_ = 1u;

  DISALLOW_COPY_AND_ASSIGN(WindowTreeClientPrivate);
};

}  // namespace aura

#endif  // UI_AURA_TEST_MUS_WINDOW_TREE_CLIENT_PRIVATE_H_
