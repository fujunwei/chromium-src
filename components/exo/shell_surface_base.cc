// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/exo/shell_surface_base.h"

#include <algorithm>

#include "ash/frame/custom_frame_view_ash.h"
#include "ash/public/cpp/config.h"
#include "ash/public/cpp/shelf_types.h"
#include "ash/public/cpp/shell_window_ids.h"
#include "ash/public/cpp/window_properties.h"
#include "ash/public/cpp/window_state_type.h"
#include "ash/public/interfaces/window_pin_type.mojom.h"
#include "ash/shell.h"
#include "ash/wm/drag_window_resizer.h"
#include "ash/wm/window_resizer.h"
#include "ash/wm/window_state.h"
#include "ash/wm/window_state_delegate.h"
#include "ash/wm/window_util.h"
#include "base/logging.h"
#include "base/macros.h"
#include "base/memory/ptr_util.h"
#include "base/strings/utf_string_conversions.h"
#include "base/trace_event/trace_event.h"
#include "base/trace_event/trace_event_argument.h"
#include "cc/trees/layer_tree_frame_sink.h"
#include "components/exo/surface.h"
#include "components/exo/wm_helper.h"
#include "services/ui/public/interfaces/window_tree_constants.mojom.h"
#include "ui/accessibility/ax_node_data.h"
#include "ui/aura/client/aura_constants.h"
#include "ui/aura/client/cursor_client.h"
#include "ui/aura/window.h"
#include "ui/aura/window_event_dispatcher.h"
#include "ui/aura/window_observer.h"
#include "ui/aura/window_targeter.h"
#include "ui/aura/window_tree_host.h"
#include "ui/base/accelerators/accelerator.h"
#include "ui/base/class_property.h"
#include "ui/compositor/compositor.h"
#include "ui/compositor/dip_util.h"
#include "ui/compositor_extra/shadow.h"
#include "ui/display/display.h"
#include "ui/display/screen.h"
#include "ui/gfx/geometry/vector2d_conversions.h"
#include "ui/gfx/path.h"
#include "ui/views/widget/widget.h"
#include "ui/wm/core/capture_controller.h"
#include "ui/wm/core/coordinate_conversion.h"
#include "ui/wm/core/shadow_controller.h"
#include "ui/wm/core/shadow_types.h"
#include "ui/wm/core/window_animations.h"
#include "ui/wm/core/window_util.h"

namespace exo {
namespace {

DEFINE_LOCAL_UI_CLASS_PROPERTY_KEY(Surface*, kMainSurfaceKey, nullptr)

// Application Id set by the client.
DEFINE_OWNED_UI_CLASS_PROPERTY_KEY(std::string, kApplicationIdKey, nullptr);

// Application Id set by the client.
DEFINE_OWNED_UI_CLASS_PROPERTY_KEY(std::string, kStartupIdKey, nullptr);

const int32_t kInvalidChildAxTreeId = -1;

// The accelerator keys used to close ShellSurfaces.
const struct {
  ui::KeyboardCode keycode;
  int modifiers;
} kCloseWindowAccelerators[] = {
    {ui::VKEY_W, ui::EF_CONTROL_DOWN},
    {ui::VKEY_W, ui::EF_SHIFT_DOWN | ui::EF_CONTROL_DOWN},
    {ui::VKEY_F4, ui::EF_ALT_DOWN}};

class ShellSurfaceWidget : public views::Widget {
 public:
  explicit ShellSurfaceWidget(ShellSurfaceBase* shell_surface)
      : shell_surface_(shell_surface) {}

  // Overridden from views::Widget:
  void Close() override { shell_surface_->Close(); }
  void OnKeyEvent(ui::KeyEvent* event) override {
    // Handle only accelerators. Do not call Widget::OnKeyEvent that eats focus
    // management keys (like the tab key) as well.
    if (GetFocusManager()->ProcessAccelerator(ui::Accelerator(*event)))
      event->SetHandled();
  }

 private:
  ShellSurfaceBase* const shell_surface_;

  DISALLOW_COPY_AND_ASSIGN(ShellSurfaceWidget);
};

class CustomFrameView : public ash::CustomFrameViewAsh,
                        public aura::WindowObserver {
 public:
  using ShapeRects = std::vector<gfx::Rect>;

  CustomFrameView(views::Widget* widget,
                  ShellSurfaceBase* shell_surface,
                  bool enabled,
                  bool client_controlled_move_resize)
      : CustomFrameViewAsh(widget),
        shell_surface_(shell_surface),
        client_controlled_move_resize_(client_controlled_move_resize) {
    SetEnabled(enabled);
    SetVisible(enabled);
    if (!enabled)
      CustomFrameViewAsh::SetShouldPaintHeader(false);

    frame()->GetNativeWindow()->AddObserver(this);
  }

  ~CustomFrameView() override {
    if (frame() && frame()->GetNativeWindow() &&
        frame()->GetNativeWindow()->HasObserver(this)) {
      frame()->GetNativeWindow()->RemoveObserver(this);
    }
  }

  // Overridden from ash::CustomFrameViewAsh:
  void SetShouldPaintHeader(bool paint) override {
    if (visible()) {
      CustomFrameViewAsh::SetShouldPaintHeader(paint);
      return;
    }
    // TODO(oshima): The caption area will be unknown
    // if a client draw a caption. (It may not even be
    // rectangular). Remove mask.
    aura::Window* window = GetWidget()->GetNativeWindow();
    ui::Layer* layer = window->layer();
    if (paint) {
      if (layer->alpha_shape()) {
        layer->SetAlphaShape(nullptr);
        layer->SetMasksToBounds(false);
      }
      return;
    }

    int inset = window->GetProperty(aura::client::kTopViewInset);
    if (inset <= 0)
      return;

    gfx::Rect bound(bounds().size());
    bound.Inset(0, inset, 0, 0);
    std::unique_ptr<ShapeRects> shape = std::make_unique<ShapeRects>();
    shape->push_back(bound);
    layer->SetAlphaShape(std::move(shape));
    layer->SetMasksToBounds(true);
  }

  // Overridden from aura::WindowObserver:
  void OnWindowBoundsChanged(aura::Window* window,
                             const gfx::Rect& old_bounds,
                             const gfx::Rect& new_bounds,
                             ui::PropertyChangeReason reason) override {
    // When window bounds are changed, we need to update the header view so that
    // the window mask layer bounds can be set correctly in function
    // SetShouldPaintHeader(). Note: this can be removed if the layer mask in
    // CustomFrameView becomes unnecessary.
    // TODO(oshima): Investigate if we can eliminate this.
    CustomFrameViewAsh::UpdateHeaderView();
  }

  void OnWindowDestroying(aura::Window* window) override {
    DCHECK_EQ(frame()->GetNativeWindow(), window);
    window->RemoveObserver(this);
  }

  // Overridden from views::NonClientFrameView:
  gfx::Rect GetBoundsForClientView() const override {
    if (visible())
      return ash::CustomFrameViewAsh::GetBoundsForClientView();
    return bounds();
  }
  gfx::Rect GetWindowBoundsForClientBounds(
      const gfx::Rect& client_bounds) const override {
    if (visible()) {
      return ash::CustomFrameViewAsh::GetWindowBoundsForClientBounds(
          client_bounds);
    }
    return client_bounds;
  }
  int NonClientHitTest(const gfx::Point& point) override {
    if (visible() || !client_controlled_move_resize_)
      return ash::CustomFrameViewAsh::NonClientHitTest(point);
    return GetWidget()->client_view()->NonClientHitTest(point);
  }
  void GetWindowMask(const gfx::Size& size, gfx::Path* window_mask) override {
    if (visible())
      return ash::CustomFrameViewAsh::GetWindowMask(size, window_mask);
  }
  void ResetWindowControls() override {
    if (visible())
      return ash::CustomFrameViewAsh::ResetWindowControls();
  }
  void UpdateWindowIcon() override {
    if (visible())
      return ash::CustomFrameViewAsh::ResetWindowControls();
  }
  void UpdateWindowTitle() override {
    if (visible())
      return ash::CustomFrameViewAsh::UpdateWindowTitle();
  }
  void SizeConstraintsChanged() override {
    if (visible())
      return ash::CustomFrameViewAsh::SizeConstraintsChanged();
  }
  gfx::Size GetMinimumSize() const override {
    gfx::Size minimum_size = shell_surface_->GetMinimumSize();
    if (visible()) {
      return ash::CustomFrameViewAsh::GetWindowBoundsForClientBounds(
                 gfx::Rect(minimum_size))
          .size();
    }
    return minimum_size;
  }

 private:
  ShellSurfaceBase* const shell_surface_;
  // TODO(oshima): Remove this once the transition to new drag/resize
  // is complete. https://crbug.com/801666.
  const bool client_controlled_move_resize_;

  DISALLOW_COPY_AND_ASSIGN(CustomFrameView);
};

class CustomWindowTargeter : public aura::WindowTargeter {
 public:
  CustomWindowTargeter(views::Widget* widget,
                       bool client_controlled_move_resize)
      : widget_(widget),
        client_controlled_move_resize_(client_controlled_move_resize) {}
  ~CustomWindowTargeter() override {}

  // Overridden from aura::WindowTargeter:
  bool EventLocationInsideBounds(aura::Window* window,
                                 const ui::LocatedEvent& event) const override {
    gfx::Point local_point = event.location();

    if (window->parent()) {
      aura::Window::ConvertPointToTarget(window->parent(), window,
                                         &local_point);
    }

    if (IsInResizeHandle(window, event, local_point))
      return true;

    Surface* surface = ShellSurfaceBase::GetMainSurface(window);
    if (!surface)
      return false;

    int component =
        widget_->non_client_view()
            ? widget_->non_client_view()->NonClientHitTest(local_point)
            : HTNOWHERE;
    if (component != HTNOWHERE && component != HTCLIENT &&
        component != HTBORDER) {
      return true;
    }

    aura::Window::ConvertPointToTarget(window, surface->window(), &local_point);
    return surface->HitTest(local_point);
  }

 private:
  bool IsInResizeHandle(aura::Window* window,
                        const ui::LocatedEvent& event,
                        const gfx::Point& local_point) const {
    if (window != widget_->GetNativeWindow() ||
        !widget_->widget_delegate()->CanResize()) {
      return false;
    }

    // Use ash's resize handle detection logic if
    // a) ClientControlledShellSurface uses server side resize or
    // b) xdg shell is using the server side decoration.
    if (ash::wm::GetWindowState(widget_->GetNativeWindow())
                ->allow_set_bounds_direct()
            ? client_controlled_move_resize_
            : !widget_->non_client_view()->frame_view()->visible()) {
      return false;
    }

    ui::EventTarget* parent =
        static_cast<ui::EventTarget*>(window)->GetParentTarget();
    if (parent) {
      aura::WindowTargeter* parent_targeter =
          static_cast<aura::WindowTargeter*>(parent->GetEventTargeter());

      if (parent_targeter) {
        gfx::Rect mouse_rect;
        gfx::Rect touch_rect;

        if (parent_targeter->GetHitTestRects(window, &mouse_rect,
                                             &touch_rect)) {
          const gfx::Vector2d offset = -window->bounds().OffsetFromOrigin();
          mouse_rect.Offset(offset);
          touch_rect.Offset(offset);
          if (event.IsTouchEvent() || event.IsGestureEvent()
                  ? touch_rect.Contains(local_point)
                  : mouse_rect.Contains(local_point)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  views::Widget* const widget_;
  const bool client_controlled_move_resize_;

  DISALLOW_COPY_AND_ASSIGN(CustomWindowTargeter);
};

// A place holder to disable default implementation created by
// ash::CustomFrameViewAsh, which triggers immersive fullscreen etc, which
// we don't need.
class CustomWindowStateDelegate : public ash::wm::WindowStateDelegate {
 public:
  CustomWindowStateDelegate() {}
  ~CustomWindowStateDelegate() override {}

  // Overridden from ash::wm::WindowStateDelegate:
  bool ToggleFullscreen(ash::wm::WindowState* window_state) override {
    return false;
  }
  bool RestoreAlwaysOnTop(ash::wm::WindowState* window_state) override {
    return false;
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(CustomWindowStateDelegate);
};

}  // namespace

// Surface state associated with each configure request.
struct ShellSurfaceBase::Config {
  Config(uint32_t serial,
         const gfx::Vector2d& origin_offset,
         int resize_component,
         std::unique_ptr<ui::CompositorLock> compositor_lock);
  ~Config();

  uint32_t serial;
  gfx::Vector2d origin_offset;
  int resize_component;
  std::unique_ptr<ui::CompositorLock> compositor_lock;
};

////////////////////////////////////////////////////////////////////////////////
// ShellSurfaceBase, Config:

ShellSurfaceBase::Config::Config(
    uint32_t serial,
    const gfx::Vector2d& origin_offset,
    int resize_component,
    std::unique_ptr<ui::CompositorLock> compositor_lock)
    : serial(serial),
      origin_offset(origin_offset),
      resize_component(resize_component),
      compositor_lock(std::move(compositor_lock)) {}

ShellSurfaceBase::Config::~Config() {}

////////////////////////////////////////////////////////////////////////////////
// ShellSurfaceBase, ScopedConfigure:

ShellSurfaceBase::ScopedConfigure::ScopedConfigure(
    ShellSurfaceBase* shell_surface,
    bool force_configure)
    : shell_surface_(shell_surface), force_configure_(force_configure) {
  // ScopedConfigure instances cannot be nested.
  DCHECK(!shell_surface_->scoped_configure_);
  shell_surface_->scoped_configure_ = this;
}

ShellSurfaceBase::ScopedConfigure::~ScopedConfigure() {
  DCHECK_EQ(shell_surface_->scoped_configure_, this);
  shell_surface_->scoped_configure_ = nullptr;
  if (needs_configure_ || force_configure_)
    shell_surface_->Configure();
  // ScopedConfigure instance might have suppressed a widget bounds update.
  if (shell_surface_->widget_) {
    shell_surface_->UpdateWidgetBounds();
    shell_surface_->UpdateShadow();
  }
}

////////////////////////////////////////////////////////////////////////////////
// ShellSurfaceBase, public:

ShellSurfaceBase::ShellSurfaceBase(Surface* surface,
                                   const gfx::Point& origin,
                                   bool activatable,
                                   bool can_minimize,
                                   int container)
    : SurfaceTreeHost("ExoShellSurfaceHost"),
      origin_(origin),
      container_(container),
      activatable_(activatable),
      can_minimize_(can_minimize) {
  WMHelper::GetInstance()->AddActivationObserver(this);
  surface->AddSurfaceObserver(this);
  SetRootSurface(surface);
  host_window()->Show();
  set_owned_by_client();
}

ShellSurfaceBase::~ShellSurfaceBase() {
  DCHECK(!scoped_configure_);
  // Remove activation observer before hiding widget to prevent it from
  // casuing the configure callback to be called.
  WMHelper::GetInstance()->RemoveActivationObserver(this);
  if (widget_) {
    widget_->GetNativeWindow()->RemoveObserver(this);
    // Remove transient children so they are not automatically destroyed.
    for (auto* child : wm::GetTransientChildren(widget_->GetNativeWindow()))
      wm::RemoveTransientChild(widget_->GetNativeWindow(), child);
    if (widget_->IsVisible())
      widget_->Hide();
    widget_->CloseNow();
  }
  if (parent_)
    parent_->RemoveObserver(this);
  if (root_surface())
    root_surface()->RemoveSurfaceObserver(this);
  if (has_grab_)
    wm::CaptureController::Get()->RemoveObserver(this);
}

void ShellSurfaceBase::AcknowledgeConfigure(uint32_t serial) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::AcknowledgeConfigure", "serial",
               serial);

  // Apply all configs that are older or equal to |serial|. The result is that
  // the origin of the main surface will move and the resize direction will
  // change to reflect the acknowledgement of configure request with |serial|
  // at the next call to Commit().
  while (!pending_configs_.empty()) {
    std::unique_ptr<Config> config = std::move(pending_configs_.front());
    pending_configs_.pop_front();

    // Add the config offset to the accumulated offset that will be applied when
    // Commit() is called.
    pending_origin_offset_ += config->origin_offset;

    // Set the resize direction that will be applied when Commit() is called.
    pending_resize_component_ = config->resize_component;

    if (config->serial == serial)
      break;
  }

  if (widget_) {
    UpdateWidgetBounds();
    UpdateShadow();
  }
}

void ShellSurfaceBase::Activate() {
  TRACE_EVENT0("exo", "ShellSurfaceBase::Activate");

  if (!widget_ || widget_->IsActive())
    return;

  widget_->Activate();
}

void ShellSurfaceBase::SetTitle(const base::string16& title) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetTitle", "title",
               base::UTF16ToUTF8(title));

  title_ = title;
  if (widget_)
    widget_->UpdateWindowTitle();
}

void ShellSurfaceBase::SetIcon(const gfx::ImageSkia& icon) {
  TRACE_EVENT0("exo", "ShellSurfaceBase::SetIcon");

  icon_ = icon;
  if (widget_)
    widget_->UpdateWindowIcon();
}

void ShellSurfaceBase::SetSystemModal(bool system_modal) {
  // System modal container is used by clients to implement client side
  // managed system modal dialogs using a single ShellSurface instance.
  // Hit-test region will be non-empty when at least one dialog exists on
  // the client side. Here we detect the transition between no client side
  // dialog and at least one dialog so activatable state is properly
  // updated.
  if (container_ != ash::kShellWindowId_SystemModalContainer) {
    LOG(ERROR)
        << "Only a window in SystemModalContainer can change the modality";
    return;
  }

  if (system_modal == system_modal_)
    return;

  bool non_system_modal_window_was_active =
      !system_modal_ && widget_ && widget_->IsActive();

  system_modal_ = system_modal;

  if (widget_) {
    UpdateSystemModal();
    // Deactivate to give the focus back to normal windows.
    if (!system_modal_ && !non_system_modal_window_was_active_) {
      widget_->Deactivate();
    }
  }

  non_system_modal_window_was_active_ = non_system_modal_window_was_active;
}

void ShellSurfaceBase::UpdateSystemModal() {
  DCHECK(widget_);
  DCHECK_EQ(container_, ash::kShellWindowId_SystemModalContainer);
  widget_->GetNativeWindow()->SetProperty(
      aura::client::kModalKey,
      system_modal_ ? ui::MODAL_TYPE_SYSTEM : ui::MODAL_TYPE_NONE);
}

// static
void ShellSurfaceBase::SetApplicationId(aura::Window* window,
                                        const base::Optional<std::string>& id) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetApplicationId", "application_id",
               id ? *id : "null");

  if (id)
    window->SetProperty(kApplicationIdKey, new std::string(*id));
  else
    window->ClearProperty(kApplicationIdKey);
}

// static
const std::string* ShellSurfaceBase::GetApplicationId(
    const aura::Window* window) {
  return window->GetProperty(kApplicationIdKey);
}

void ShellSurfaceBase::SetApplicationId(const char* application_id) {
  // Store the value in |application_id_| in case the window does not exist yet.
  if (application_id)
    application_id_ = std::string(application_id);
  else
    application_id_.reset();

  if (widget_ && widget_->GetNativeWindow())
    SetApplicationId(widget_->GetNativeWindow(), application_id_);
}

// static
void ShellSurfaceBase::SetStartupId(aura::Window* window,
                                    const base::Optional<std::string>& id) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetStartupId", "startup_id",
               id ? *id : "null");

  if (id)
    window->SetProperty(kStartupIdKey, new std::string(*id));
  else
    window->ClearProperty(kStartupIdKey);
}

// static
const std::string* ShellSurfaceBase::GetStartupId(aura::Window* window) {
  return window->GetProperty(kStartupIdKey);
}

void ShellSurfaceBase::SetStartupId(const char* startup_id) {
  // Store the value in |startup_id_| in case the window does not exist yet.
  if (startup_id)
    startup_id_ = std::string(startup_id);
  else
    startup_id_.reset();

  if (widget_ && widget_->GetNativeWindow())
    SetStartupId(widget_->GetNativeWindow(), startup_id_);
}

void ShellSurfaceBase::SetChildAxTreeId(int32_t child_ax_tree_id) {
  // We don't expect that child ax tree id is changed once it's set.
  DCHECK_EQ(child_ax_tree_id_, kInvalidChildAxTreeId);
  DCHECK_NE(child_ax_tree_id, kInvalidChildAxTreeId);

  child_ax_tree_id_ = child_ax_tree_id;

  this->NotifyAccessibilityEvent(ax::mojom::Event::kChildrenChanged, false);
}

void ShellSurfaceBase::Close() {
  if (!close_callback_.is_null())
    close_callback_.Run();
}

void ShellSurfaceBase::SetGeometry(const gfx::Rect& geometry) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetGeometry", "geometry",
               geometry.ToString());

  if (geometry.IsEmpty()) {
    DLOG(WARNING) << "Surface geometry must be non-empty";
    return;
  }

  pending_geometry_ = geometry;
}

void ShellSurfaceBase::SetOrigin(const gfx::Point& origin) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetOrigin", "origin",
               origin.ToString());

  origin_ = origin;
}

void ShellSurfaceBase::SetActivatable(bool activatable) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetActivatable", "activatable",
               activatable);

  activatable_ = activatable;
}

void ShellSurfaceBase::SetContainer(int container) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetContainer", "container", container);

  container_ = container;
}

void ShellSurfaceBase::SetMaximumSize(const gfx::Size& size) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetMaximumSize", "size",
               size.ToString());

  pending_maximum_size_ = size;
}

void ShellSurfaceBase::SetMinimumSize(const gfx::Size& size) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetMinimumSize", "size",
               size.ToString());

  pending_minimum_size_ = size;
}

void ShellSurfaceBase::SetCanMinimize(bool can_minimize) {
  TRACE_EVENT1("exo", "ShellSurfaceBase::SetCanMinimize", "can_minimize",
               can_minimize);

  can_minimize_ = can_minimize;
}

void ShellSurfaceBase::DisableMovement() {
  movement_disabled_ = true;

  if (widget_)
    widget_->set_movement_disabled(true);
}

// static
void ShellSurfaceBase::SetMainSurface(aura::Window* window, Surface* surface) {
  window->SetProperty(kMainSurfaceKey, surface);
}

// static
Surface* ShellSurfaceBase::GetMainSurface(const aura::Window* window) {
  return window->GetProperty(kMainSurfaceKey);
}

// static
Surface* ShellSurfaceBase::GetTargetSurfaceForLocatedEvent(
    ui::LocatedEvent* event) {
  aura::Window* window = wm::CaptureController::Get()->GetCaptureWindow();
  gfx::PointF location_in_target = event->location_f();

  if (!window)
    return Surface::AsSurface(static_cast<aura::Window*>(event->target()));

  Surface* main_surface = ShellSurfaceBase::GetMainSurface(window);
  // Skip if the event is captured by non exo windwows.
  if (!main_surface)
    return nullptr;

  while (true) {
    aura::Window* focused = window->GetEventHandlerForPoint(
        gfx::ToFlooredPoint(location_in_target));

    if (focused) {
      aura::Window::ConvertPointToTarget(window, focused, &location_in_target);
      return Surface::AsSurface(focused);
    }

    aura::Window* parent_window = wm::GetTransientParent(window);

    if (!parent_window) {
      location_in_target = event->location_f();
      return main_surface;
    }
    aura::Window::ConvertPointToTarget(window, parent_window,
                                       &location_in_target);
    window = parent_window;
  }
}

std::unique_ptr<base::trace_event::TracedValue>
ShellSurfaceBase::AsTracedValue() const {
  std::unique_ptr<base::trace_event::TracedValue> value(
      new base::trace_event::TracedValue());
  value->SetString("title", base::UTF16ToUTF8(title_));
  if (GetWidget() && GetWidget()->GetNativeWindow()) {
    const std::string* application_id =
        GetApplicationId(GetWidget()->GetNativeWindow());

    if (application_id)
      value->SetString("application_id", *application_id);

    const std::string* startup_id =
        GetStartupId(GetWidget()->GetNativeWindow());

    if (startup_id)
      value->SetString("startup_id", *startup_id);
  }
  return value;
}

////////////////////////////////////////////////////////////////////////////////
// SurfaceDelegate overrides:

void ShellSurfaceBase::OnSurfaceCommit() {
  // SetShadowBounds requires synchronizing shadow bounds with the next frame,
  // so submit the next frame to a new surface and let the host window use the
  // new surface.
  if (shadow_bounds_changed_)
    host_window()->AllocateLocalSurfaceId();

  SurfaceTreeHost::OnSurfaceCommit();

  if (enabled() && !widget_) {
    // Defer widget creation until surface contains some contents.
    if (host_window()->bounds().IsEmpty()) {
      Configure();
      return;
    }

    CreateShellSurfaceWidget(ui::SHOW_STATE_NORMAL);
  }

  // Apply the accumulated pending origin offset to reflect acknowledged
  // configure requests.
  origin_offset_ += pending_origin_offset_;
  pending_origin_offset_ = gfx::Vector2d();

  // Update resize direction to reflect acknowledged configure requests.
  resize_component_ = pending_resize_component_;

  // Apply new window geometry.
  geometry_ = pending_geometry_;

  // Apply new minimum/maximium size.
  bool size_constraint_changed = minimum_size_ != pending_minimum_size_ ||
                                 maximum_size_ != pending_maximum_size_;
  minimum_size_ = pending_minimum_size_;
  maximum_size_ = pending_maximum_size_;

  if (widget_) {
    UpdateWidgetBounds();
    UpdateShadow();

    // System modal container is used by clients to implement overlay
    // windows using a single ShellSurface instance.  If hit-test
    // region is empty, then it is non interactive window and won't be
    // activated.
    if (container_ == ash::kShellWindowId_SystemModalContainer) {
      // Prevent window from being activated when hit test region is empty.
      bool activatable = activatable_ && HasHitTestRegion();
      if (activatable != CanActivate()) {
        set_can_activate(activatable);
        // Activate or deactivate window if activation state changed.
        if (activatable) {
          // Automatically activate only if the window is modal.
          // Non modal window should be activated by a user action.
          // TODO(oshima): Non modal system window does not have an associated
          // task ID, and as a result, it cannot be activated from client side.
          // Fix this (b/65460424) and remove this if condition.
          if (system_modal_)
            wm::ActivateWindow(widget_->GetNativeWindow());
        } else if (widget_->IsActive()) {
          wm::DeactivateWindow(widget_->GetNativeWindow());
        }
      }
    }

    UpdateSurfaceBounds();

    // Show widget if needed.
    if (pending_show_widget_) {
      DCHECK(!widget_->IsClosed());
      DCHECK(!widget_->IsVisible());
      pending_show_widget_ = false;
      widget_->Show();
      if (has_grab_)
        StartCapture();

      if (container_ == ash::kShellWindowId_SystemModalContainer)
        UpdateSystemModal();
    }
  }

  SubmitCompositorFrame();

  if (size_constraint_changed)
    widget_->OnSizeConstraintsChanged();
}

bool ShellSurfaceBase::IsInputEnabled(Surface*) const {
  return true;
}

void ShellSurfaceBase::OnSetFrame(SurfaceFrameType frame_type) {
  if (is_popup_) {
    // TODO(oshima): Consider supporting shadow type.
    DLOG(WARNING) << "popup does not support frame decoration";
    return;
  }

  bool frame_was_disabled = !frame_enabled();
  frame_type_ = frame_type;
  switch (frame_type) {
    case SurfaceFrameType::NONE:
      shadow_bounds_.reset();
      break;
    case SurfaceFrameType::NORMAL:
    case SurfaceFrameType::AUTOHIDE:
    case SurfaceFrameType::OVERLAY:
      // Initialize the shadow if it didn't exist.  Do not reset if
      // the frame type just switched from another enabled type.
      if (!shadow_bounds_ || frame_was_disabled)
        shadow_bounds_ = gfx::Rect();
      break;
    case SurfaceFrameType::SHADOW:
      shadow_bounds_ = gfx::Rect();
      break;
  }
  if (!widget_)
    return;
  CustomFrameView* frame_view =
      static_cast<CustomFrameView*>(widget_->non_client_view()->frame_view());
  if (frame_view->enabled() == frame_enabled())
    return;

  frame_view->SetEnabled(frame_enabled());
  frame_view->SetVisible(frame_enabled());
  frame_view->SetShouldPaintHeader(frame_enabled());
  frame_view->SetHeaderHeight(base::nullopt);
  widget_->GetRootView()->Layout();
  // TODO(oshima): We probably should wait applying these if the
  // window is animating.
  UpdateWidgetBounds();
  UpdateSurfaceBounds();
}

void ShellSurfaceBase::OnSetFrameColors(SkColor active_color,
                                        SkColor inactive_color) {
  has_frame_colors_ = true;
  active_frame_color_ = SkColorSetA(active_color, SK_AlphaOPAQUE);
  inactive_frame_color_ = SkColorSetA(inactive_color, SK_AlphaOPAQUE);
  if (widget_) {
    widget_->GetNativeWindow()->SetProperty(ash::kFrameActiveColorKey,
                                            active_frame_color_);
    widget_->GetNativeWindow()->SetProperty(ash::kFrameInactiveColorKey,
                                            inactive_frame_color_);
  }
}

void ShellSurfaceBase::OnSetParent(Surface* parent,
                                   const gfx::Point& position) {
  views::Widget* parent_widget =
      parent ? views::Widget::GetTopLevelWidgetForNativeView(parent->window())
             : nullptr;
  if (parent_widget) {
    // Set parent window if using default container and the container itself
    // is not the parent.
    if (container_ == ash::kShellWindowId_DefaultContainer)
      SetParentWindow(parent_widget->GetNativeWindow());

    origin_ = position;
    views::View::ConvertPointToScreen(
        parent_widget->widget_delegate()->GetContentsView(), &origin_);

    if (!widget_)
      return;

    ash::wm::WindowState* window_state =
        ash::wm::GetWindowState(widget_->GetNativeWindow());
    if (window_state->is_dragged())
      return;

    gfx::Rect widget_bounds = widget_->GetWindowBoundsInScreen();
    gfx::Rect new_widget_bounds(origin_, widget_bounds.size());
    if (new_widget_bounds != widget_bounds) {
      base::AutoReset<bool> auto_ignore_window_bounds_changes(
          &ignore_window_bounds_changes_, true);
      widget_->SetBounds(new_widget_bounds);
      UpdateSurfaceBounds();
    }
  } else {
    SetParentWindow(nullptr);
  }
}

void ShellSurfaceBase::OnSetStartupId(const char* startup_id) {
  SetStartupId(startup_id);
}

void ShellSurfaceBase::OnSetApplicationId(const char* application_id) {
  SetApplicationId(application_id);
}

////////////////////////////////////////////////////////////////////////////////
// SurfaceObserver overrides:

void ShellSurfaceBase::OnSurfaceDestroying(Surface* surface) {
  DCHECK_EQ(root_surface(), surface);
  surface->RemoveSurfaceObserver(this);
  SetRootSurface(nullptr);

  if (widget_)
    SetMainSurface(widget_->GetNativeWindow(), nullptr);

  // Hide widget before surface is destroyed. This allows hide animations to
  // run using the current surface contents.
  if (widget_) {
    // Remove transient children so they are not automatically hidden.
    for (auto* child : wm::GetTransientChildren(widget_->GetNativeWindow()))
      wm::RemoveTransientChild(widget_->GetNativeWindow(), child);
    widget_->Hide();
  }

  // Note: In its use in the Wayland server implementation, the surface
  // destroyed callback may destroy the ShellSurface instance. This call needs
  // to be last so that the instance can be destroyed.
  if (!surface_destroyed_callback_.is_null())
    std::move(surface_destroyed_callback_).Run();
}

////////////////////////////////////////////////////////////////////////////////
// views::WidgetDelegate overrides:

bool ShellSurfaceBase::CanResize() const {
  if (movement_disabled_)
    return false;
  // The shell surface is resizable by default when min/max size is empty,
  // othersize it's resizable when min size != max size.
  return minimum_size_.IsEmpty() || minimum_size_ != maximum_size_;
}

bool ShellSurfaceBase::CanMaximize() const {
  // Shell surfaces in system modal container cannot be maximized.
  if (container_ != ash::kShellWindowId_DefaultContainer)
    return false;

  // Non-transient shell surfaces can be maximized.
  return !parent_;
}

bool ShellSurfaceBase::CanMinimize() const {
  // Non-transient shell surfaces can be minimized.
  return !parent_ && can_minimize_;
}

base::string16 ShellSurfaceBase::GetWindowTitle() const {
  return title_;
}

bool ShellSurfaceBase::ShouldShowWindowTitle() const {
  return false;
}

gfx::ImageSkia ShellSurfaceBase::GetWindowIcon() {
  return icon_;
}

void ShellSurfaceBase::WindowClosing() {
  SetEnabled(false);
  widget_ = nullptr;
}

views::Widget* ShellSurfaceBase::GetWidget() {
  return widget_;
}

const views::Widget* ShellSurfaceBase::GetWidget() const {
  return widget_;
}

views::View* ShellSurfaceBase::GetContentsView() {
  return this;
}

views::NonClientFrameView* ShellSurfaceBase::CreateNonClientFrameView(
    views::Widget* widget) {
  aura::Window* window = widget_->GetNativeWindow();
  // ShellSurfaces always use immersive mode.
  window->SetProperty(aura::client::kImmersiveFullscreenKey, true);
  ash::wm::WindowState* window_state = ash::wm::GetWindowState(window);
  if (!frame_enabled() && !window_state->HasDelegate()) {
    window_state->SetDelegate(std::make_unique<CustomWindowStateDelegate>());
  }
  CustomFrameView* frame_view = new CustomFrameView(
      widget, this, frame_enabled(), client_controlled_move_resize_);
  if (has_frame_colors_)
    frame_view->SetFrameColors(active_frame_color_, inactive_frame_color_);
  return frame_view;
}

bool ShellSurfaceBase::WidgetHasHitTestMask() const {
  return true;
}

void ShellSurfaceBase::GetWidgetHitTestMask(gfx::Path* mask) const {
  GetHitTestMask(mask);

  gfx::Point origin = host_window()->bounds().origin();
  SkMatrix matrix;
  float scale = GetScale();
  matrix.setScaleTranslate(
      SkFloatToScalar(1.0f / scale), SkFloatToScalar(1.0f / scale),
      SkIntToScalar(origin.x()), SkIntToScalar(origin.y()));
  mask->transform(matrix);
}

void ShellSurfaceBase::OnCaptureChanged(aura::Window* lost_capture,
                                        aura::Window* gained_capture) {
  if (lost_capture == widget_->GetNativeWindow() && is_popup_) {
    wm::CaptureController::Get()->RemoveObserver(this);
    if (gained_capture &&
        lost_capture == wm::GetTransientParent(gained_capture)) {
      // Don't close if the capture has been transferred to the child popup.
      return;
    }
    aura::Window* parent = wm::GetTransientParent(lost_capture);
    if (parent) {
      // The capture needs to be transferred to the parent if it had grab.
      views::Widget* parent_widget =
          views::Widget::GetWidgetForNativeWindow(parent);
      ShellSurfaceBase* parent_shell_surface = static_cast<ShellSurfaceBase*>(
          parent_widget->widget_delegate()->GetContentsView());
      if (parent_shell_surface->has_grab_)
        parent_shell_surface->StartCapture();
    }
    widget_->Close();
  }
}

////////////////////////////////////////////////////////////////////////////////
// views::Views overrides:

gfx::Size ShellSurfaceBase::CalculatePreferredSize() const {
  if (!geometry_.IsEmpty())
    return geometry_.size();

  return host_window()->bounds().size();
}

gfx::Size ShellSurfaceBase::GetMinimumSize() const {
  return minimum_size_.IsEmpty() ? gfx::Size(1, 1) : minimum_size_;
}

gfx::Size ShellSurfaceBase::GetMaximumSize() const {
  // On ChromeOS, non empty maximum size will make the window
  // non maximizable.
  return maximum_size_;
}

void ShellSurfaceBase::GetAccessibleNodeData(ui::AXNodeData* node_data) {
  node_data->role = ax::mojom::Role::kClient;

  if (child_ax_tree_id_ == kInvalidChildAxTreeId)
    return;

  node_data->AddIntAttribute(ax::mojom::IntAttribute::kChildTreeId,
                             child_ax_tree_id_);
}

////////////////////////////////////////////////////////////////////////////////
// aura::WindowObserver overrides:

void ShellSurfaceBase::OnWindowBoundsChanged(aura::Window* window,
                                             const gfx::Rect& old_bounds,
                                             const gfx::Rect& new_bounds,
                                             ui::PropertyChangeReason reason) {
  if (!widget_ || !root_surface() || ignore_window_bounds_changes_)
    return;

  if (window == widget_->GetNativeWindow()) {
    if (new_bounds.size() == old_bounds.size())
      return;

    // If size changed then give the client a chance to produce new contents
    // before origin on screen is changed. Retain the old origin by reverting
    // the origin delta until the next configure is acknowledged.
    gfx::Vector2d delta = new_bounds.origin() - old_bounds.origin();
    origin_offset_ -= delta;
    pending_origin_offset_accumulator_ += delta;

    UpdateSurfaceBounds();

    // The shadow size may be updated to match the widget. Change it back
    // to the shadow content size. Note that this relies on wm::ShadowController
    // being notified of the change before |this|.
    UpdateShadow();

    Configure();
  }
}

void ShellSurfaceBase::OnWindowDestroying(aura::Window* window) {
  if (window == parent_) {
    parent_ = nullptr;
    // |parent_| being set to null effects the ability to maximize the window.
    if (widget_)
      widget_->OnSizeConstraintsChanged();
  }
  window->RemoveObserver(this);
}

////////////////////////////////////////////////////////////////////////////////
// wm::ActivationChangeObserver overrides:

void ShellSurfaceBase::OnWindowActivated(ActivationReason reason,
                                         aura::Window* gained_active,
                                         aura::Window* lost_active) {
  if (!widget_)
    return;

  if (gained_active == widget_->GetNativeWindow() ||
      lost_active == widget_->GetNativeWindow()) {
    DCHECK(CanActivate());
    Configure();
    UpdateShadow();
  }
}

////////////////////////////////////////////////////////////////////////////////
// ui::AcceleratorTarget overrides:

bool ShellSurfaceBase::AcceleratorPressed(const ui::Accelerator& accelerator) {
  for (const auto& entry : kCloseWindowAccelerators) {
    if (ui::Accelerator(entry.keycode, entry.modifiers) == accelerator) {
      if (!close_callback_.is_null())
        close_callback_.Run();
      return true;
    }
  }
  return views::View::AcceleratorPressed(accelerator);
}

////////////////////////////////////////////////////////////////////////////////
// ShellSurfaceBase, protected:

void ShellSurfaceBase::CreateShellSurfaceWidget(
    ui::WindowShowState show_state) {
  DCHECK(enabled());
  DCHECK(!widget_);

  views::Widget::InitParams params;
  params.type = is_popup_ ? views::Widget::InitParams::TYPE_POPUP
                          : views::Widget::InitParams::TYPE_WINDOW;
  params.ownership = views::Widget::InitParams::NATIVE_WIDGET_OWNS_WIDGET;
  params.delegate = this;
  params.shadow_type = views::Widget::InitParams::SHADOW_TYPE_NONE;
  params.opacity = views::Widget::InitParams::TRANSLUCENT_WINDOW;
  params.show_state = show_state;
  // Make shell surface a transient child if |parent_| has been set.
  params.parent =
      parent_ ? parent_
              : ash::Shell::GetContainer(
                    ash::Shell::GetRootWindowForNewWindows(), container_);
  params.bounds = gfx::Rect(origin_, gfx::Size());
  bool activatable = activatable_;
  // ShellSurfaces in system modal container are only activatable if input
  // region is non-empty. See OnCommitSurface() for more details.
  if (container_ == ash::kShellWindowId_SystemModalContainer)
    activatable &= HasHitTestRegion();
  // Transient child needs to have an application id to be activatable.
  if (parent_)
    activatable &= application_id_.has_value();
  params.activatable = activatable ? views::Widget::InitParams::ACTIVATABLE_YES
                                   : views::Widget::InitParams::ACTIVATABLE_NO;
  // Note: NativeWidget owns this widget.
  widget_ = new ShellSurfaceWidget(this);
  widget_->Init(params);

  aura::Window* window = widget_->GetNativeWindow();
  window->SetName("ExoShellSurface");
  // TODO(yawano): This needs to be set to false if TalkBack is handling this
  //               window.
  window->SetProperty(aura::client::kAccessibilityFocusFallsbackToWidgetKey,
                      true);
  window->AddChild(host_window());
  // Use DESCENDANTS_ONLY event targeting policy for mus/mash.
  // TODO(https://crbug.com/839521): Revisit after event dispatching code is
  //     changed for mus/mash.
  window->SetEventTargetingPolicy(
      ash::Shell::GetAshConfig() == ash::Config::CLASSIC
          ? ui::mojom::EventTargetingPolicy::TARGET_AND_DESCENDANTS
          : ui::mojom::EventTargetingPolicy::DESCENDANTS_ONLY);
  window->SetEventTargeter(base::WrapUnique(
      new CustomWindowTargeter(widget_, client_controlled_move_resize_)));
  SetApplicationId(window, application_id_);
  SetStartupId(window, startup_id_);
  SetMainSurface(window, root_surface());

  // Start tracking changes to window bounds and window state.
  window->AddObserver(this);
  ash::wm::WindowState* window_state = ash::wm::GetWindowState(window);
  InitializeWindowState(window_state);

  // AutoHide shelf in fullscreen state.
  window_state->SetHideShelfWhenFullscreen(false);

  // Fade visibility animations for non-activatable windows.
  if (!CanActivate()) {
    wm::SetWindowVisibilityAnimationType(
        window, wm::WINDOW_VISIBILITY_ANIMATION_TYPE_FADE);
  }

  // Register close window accelerators.
  views::FocusManager* focus_manager = widget_->GetFocusManager();
  for (const auto& entry : kCloseWindowAccelerators) {
    focus_manager->RegisterAccelerator(
        ui::Accelerator(entry.keycode, entry.modifiers),
        ui::AcceleratorManager::kNormalPriority, this);
  }

  // Show widget next time Commit() is called.
  pending_show_widget_ = true;
}

void ShellSurfaceBase::Configure() {
  // Delay configure callback if |scoped_configure_| is set.
  if (scoped_configure_) {
    scoped_configure_->set_needs_configure();
    return;
  }

  gfx::Vector2d origin_offset = pending_origin_offset_accumulator_;
  pending_origin_offset_accumulator_ = gfx::Vector2d();

  int resize_component = HTCAPTION;
  if (widget_) {
    ash::wm::WindowState* window_state =
        ash::wm::GetWindowState(widget_->GetNativeWindow());

    // If surface is being resized, save the resize direction.
    if (window_state->is_dragged())
      resize_component = window_state->drag_details()->window_component;
  }

  uint32_t serial = 0;
  if (!configure_callback_.is_null()) {
    if (widget_) {
      serial = configure_callback_.Run(
          GetClientViewBounds().size(),
          ash::wm::GetWindowState(widget_->GetNativeWindow())->GetStateType(),
          IsResizing(), widget_->IsActive(), origin_offset);
    } else {
      serial = configure_callback_.Run(gfx::Size(),
                                       ash::mojom::WindowStateType::NORMAL,
                                       false, false, origin_offset);
    }
  }

  if (!serial) {
    pending_origin_offset_ += origin_offset;
    pending_resize_component_ = resize_component;
    return;
  }

  // Apply origin offset and resize component at the first Commit() after this
  // configure request has been acknowledged.
  pending_configs_.push_back(
      std::make_unique<Config>(serial, origin_offset, resize_component,
                               std::move(configure_compositor_lock_)));
  LOG_IF(WARNING, pending_configs_.size() > 100)
      << "Number of pending configure acks for shell surface has reached: "
      << pending_configs_.size();
}

bool ShellSurfaceBase::IsResizing() const {
  ash::wm::WindowState* window_state =
      ash::wm::GetWindowState(widget_->GetNativeWindow());
  if (!window_state->is_dragged())
    return false;
  return window_state->drag_details() &&
         (window_state->drag_details()->bounds_change &
          ash::WindowResizer::kBoundsChange_Resizes);
}

void ShellSurfaceBase::UpdateWidgetBounds() {
  DCHECK(widget_);

  aura::Window* window = widget_->GetNativeWindow();
  ash::wm::WindowState* window_state = ash::wm::GetWindowState(window);
  // Return early if the shell is currently managing the bounds of the widget.
  if (!window_state->allow_set_bounds_direct()) {
    // 1) When a window is either maximized/fullscreen/pinned.
    if (window_state->IsMaximizedOrFullscreenOrPinned())
      return;
    // 2) When a window is snapped.
    if (window_state->IsSnapped())
      return;
    // 3) When a window is being interactively resized.
    if (IsResizing())
      return;
    // 4) When a window's bounds are being animated.
    if (window->layer()->GetAnimator()->IsAnimatingProperty(
            ui::LayerAnimationElement::BOUNDS))
      return;
  }

  // Return early if there is pending configure requests.
  if (!pending_configs_.empty() || scoped_configure_)
    return;

  gfx::Rect new_widget_bounds = GetWidgetBounds();

  // Set |ignore_window_bounds_changes_| as this change to window bounds
  // should not result in a configure request.
  DCHECK(!ignore_window_bounds_changes_);
  ignore_window_bounds_changes_ = true;
  if (new_widget_bounds != widget_->GetWindowBoundsInScreen())
    SetWidgetBounds(new_widget_bounds);
  ignore_window_bounds_changes_ = false;
}

void ShellSurfaceBase::SetWidgetBounds(const gfx::Rect& bounds) {
  widget_->SetBounds(bounds);
  UpdateSurfaceBounds();
}

void ShellSurfaceBase::UpdateSurfaceBounds() {
  gfx::Point origin = GetClientViewBounds().origin();

  origin += GetSurfaceOrigin().OffsetFromOrigin();
  origin -= ToFlooredVector2d(ScaleVector2d(
      root_surface_origin().OffsetFromOrigin(), 1.f / GetScale()));

  host_window()->SetBounds(gfx::Rect(origin, host_window()->bounds().size()));
  // The host window might have not been added to the widget yet.
  if (host_window()->parent()) {
    ui::SnapLayerToPhysicalPixelBoundary(widget_->GetNativeWindow()->layer(),
                                         host_window()->layer());
  }
}

void ShellSurfaceBase::UpdateShadow() {
  if (!widget_ || !root_surface())
    return;

  shadow_bounds_changed_ = false;

  aura::Window* window = widget_->GetNativeWindow();

  if (!shadow_bounds_) {
    wm::SetShadowElevation(window, wm::kShadowElevationNone);
  } else {
    wm::SetShadowElevation(window, wm::kShadowElevationDefault);

    ui::Shadow* shadow = wm::ShadowController::GetShadowForWindow(window);
    // Maximized/Fullscreen window does not create a shadow.
    if (!shadow)
      return;

    shadow->SetContentBounds(GetShadowBounds());
    // Surfaces that can't be activated are usually menus and tooltips. Use a
    // small style shadow for them.
    if (!CanActivate())
      shadow->SetElevation(wm::kShadowElevationMenuOrTooltip);
    // We don't have rounded corners unless frame is enabled.
    if (!frame_enabled())
      shadow->SetRoundedCornerRadius(0);
  }
}

gfx::Rect ShellSurfaceBase::GetVisibleBounds() const {
  // Use |geometry_| if set, otherwise use the visual bounds of the surface.
  if (!geometry_.IsEmpty())
    return geometry_;

  return root_surface() ? gfx::Rect(root_surface()->content_size())
                        : gfx::Rect();
}

gfx::Point ShellSurfaceBase::GetMouseLocation() const {
  aura::Window* const root_window = widget_->GetNativeWindow()->GetRootWindow();
  gfx::Point location =
      root_window->GetHost()->dispatcher()->GetLastMouseLocationInRoot();
  aura::Window::ConvertPointToTarget(
      root_window, widget_->GetNativeWindow()->parent(), &location);
  return location;
}

gfx::Rect ShellSurfaceBase::GetClientViewBounds() const {
  return widget_->non_client_view()
             ? widget_->non_client_view()
                   ->frame_view()
                   ->GetBoundsForClientView()
             : gfx::Rect(widget_->GetWindowBoundsInScreen().size());
}

gfx::Rect ShellSurfaceBase::GetShadowBounds() const {
  return shadow_bounds_->IsEmpty()
             ? gfx::Rect(widget_->GetNativeWindow()->bounds().size())
             : gfx::ScaleToEnclosedRect(*shadow_bounds_, 1.f / GetScale());
}

////////////////////////////////////////////////////////////////////////////////
// ShellSurfaceBase, private:

float ShellSurfaceBase::GetScale() const {
  return 1.f;
}

gfx::Rect ShellSurfaceBase::GetWidgetBounds() const {
  gfx::Rect visible_bounds = GetVisibleBounds();
  gfx::Rect new_widget_bounds =
      widget_->non_client_view()
          ? widget_->non_client_view()->GetWindowBoundsForClientBounds(
                visible_bounds)
          : visible_bounds;

  if (movement_disabled_) {
    new_widget_bounds.set_origin(origin_);
  } else if (resize_component_ == HTCAPTION) {
    // Preserve widget position.
    new_widget_bounds.set_origin(widget_->GetWindowBoundsInScreen().origin());
  } else {
    // Compute widget origin using surface origin if the current location of
    // surface is being anchored to one side of the widget as a result of a
    // resize operation.
    gfx::Rect visible_bounds = GetVisibleBounds();
    gfx::Point origin = GetSurfaceOrigin() + visible_bounds.OffsetFromOrigin();
    wm::ConvertPointToScreen(widget_->GetNativeWindow(), &origin);
    new_widget_bounds.set_origin(origin);
  }
  return new_widget_bounds;
}

gfx::Point ShellSurfaceBase::GetSurfaceOrigin() const {
  DCHECK(!movement_disabled_ || resize_component_ == HTCAPTION);

  gfx::Rect visible_bounds = GetVisibleBounds();
  gfx::Rect client_bounds = GetClientViewBounds();

  switch (resize_component_) {
    case HTCAPTION:
      return gfx::Point() + origin_offset_ - visible_bounds.OffsetFromOrigin();
    case HTBOTTOM:
    case HTRIGHT:
    case HTBOTTOMRIGHT:
      return gfx::Point() - visible_bounds.OffsetFromOrigin();
    case HTTOP:
    case HTTOPRIGHT:
      return gfx::Point(0, client_bounds.height() - visible_bounds.height()) -
             visible_bounds.OffsetFromOrigin();
    case HTLEFT:
    case HTBOTTOMLEFT:
      return gfx::Point(client_bounds.width() - visible_bounds.width(), 0) -
             visible_bounds.OffsetFromOrigin();
    case HTTOPLEFT:
      return gfx::Point(client_bounds.width() - visible_bounds.width(),
                        client_bounds.height() - visible_bounds.height()) -
             visible_bounds.OffsetFromOrigin();
    default:
      NOTREACHED();
      return gfx::Point();
  }
}

void ShellSurfaceBase::SetParentWindow(aura::Window* parent) {
  if (parent_) {
    parent_->RemoveObserver(this);
    if (widget_)
      wm::RemoveTransientChild(parent_, widget_->GetNativeWindow());
  }
  parent_ = parent;
  if (parent_) {
    parent_->AddObserver(this);
    if (widget_)
      wm::AddTransientChild(parent_, widget_->GetNativeWindow());
  }

  // If |parent_| is set effects the ability to maximize the window.
  if (widget_)
    widget_->OnSizeConstraintsChanged();
}

void ShellSurfaceBase::StartCapture() {
  widget_->set_auto_release_capture(false);
  wm::CaptureController::Get()->AddObserver(this);
  // Just capture on the window.
  widget_->SetCapture(nullptr /* view */);
}

}  // namespace exo
