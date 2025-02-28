// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_XR_XR_DEVICE_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_XR_XR_DEVICE_H_

#include "device/vr/public/mojom/vr_service.mojom-blink.h"
#include "mojo/public/cpp/bindings/binding.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise.h"
#include "third_party/blink/renderer/bindings/core/v8/script_promise_resolver.h"
#include "third_party/blink/renderer/modules/xr/xr_session_creation_options.h"
#include "third_party/blink/renderer/platform/bindings/script_wrappable.h"
#include "third_party/blink/renderer/platform/heap/handle.h"
#include "third_party/blink/renderer/platform/wtf/forward.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"

namespace blink {

class ScriptPromiseResolver;
class XR;
class XRFrameProvider;
class XRSession;

class XRDevice final : public ScriptWrappable,
                       public device::mojom::blink::VRDisplayClient {
  DEFINE_WRAPPERTYPEINFO();

 public:
  XRDevice(XR*,
           device::mojom::blink::VRDisplayHostPtr,
           device::mojom::blink::VRDisplayClientRequest,
           device::mojom::blink::VRDisplayInfoPtr);
  XR* xr() const { return xr_; }

  bool external() const { return is_external_; }

  ScriptPromise supportsSession(ScriptState*, const XRSessionCreationOptions&);
  ScriptPromise requestSession(ScriptState*, const XRSessionCreationOptions&);

  // XRDisplayClient
  void OnChanged(device::mojom::blink::VRDisplayInfoPtr) override;
  void OnExitPresent() override;
  void OnBlur() override;
  void OnFocus() override;
  void OnActivate(device::mojom::blink::VRDisplayEventReason,
                  OnActivateCallback on_handled) override;
  void OnDeactivate(device::mojom::blink::VRDisplayEventReason) override;

  XRFrameProvider* frameProvider();

  void Dispose();

  const device::mojom::blink::VRDisplayHostPtr& xrDisplayHostPtr() const {
    return display_;
  }
  const device::mojom::blink::XRFrameDataProviderPtr& xrMagicWindowProviderPtr()
      const {
    return magic_window_provider_;
  }
  const device::mojom::blink::XREnviromentIntegrationProviderPtr&
  xrEnviromentProviderPtr() const {
    return enviroment_provider_;
  }
  const device::mojom::blink::VRDisplayInfoPtr& xrDisplayInfoPtr() const {
    return display_info_;
  }
  // Incremented every time display_info_ is changed, so that other objects that
  // depend on it can know when they need to update.
  unsigned int xrDisplayInfoPtrId() const { return display_info_id_; }

  void OnFrameFocusChanged();
  // The device may report focus to us - for example if another application is
  // using the headset, or some browsing UI is shown, we may not have device
  // focus.
  bool HasDeviceFocus() { return has_device_focus_; }
  bool HasDeviceAndFrameFocus() { return IsFrameFocused() && HasDeviceFocus(); }

  bool SupportsImmersive() { return supports_immersive_; }

  int64_t GetSourceId() const;

  void Trace(blink::Visitor*) override;

 private:
  void SetXRDisplayInfo(device::mojom::blink::VRDisplayInfoPtr);

  const char* checkSessionSupport(const XRSessionCreationOptions&) const;

  void OnRequestSessionReturned(ScriptPromiseResolver* resolver,
                                const XRSessionCreationOptions& options,
                                device::mojom::blink::XRSessionPtr session);
  void OnSupportsSessionReturned(ScriptPromiseResolver* resolver,
                                 bool supports_session);

  // There are two components to focus - whether the frame itself has
  // traditional focus and whether the device reports that we have focus. These
  // are aggregated so we can hand out focus/blur events on sessions and
  // determine when to call animation frame callbacks.
  void OnFocusChanged();
  bool IsFrameFocused();

  Member<XR> xr_;
  Member<XRFrameProvider> frame_provider_;
  HeapHashSet<WeakMember<XRSession>> sessions_;
  bool is_external_ = false;
  bool supports_immersive_ = false;
  bool supports_ar_ = false;
  bool has_device_focus_ = true;

  // Indicates whether we've already logged a request for an immersive session.
  bool did_log_request_immersive_session_ = false;

  device::mojom::blink::XRFrameDataProviderPtr magic_window_provider_;
  device::mojom::blink::XREnviromentIntegrationProviderPtr enviroment_provider_;
  device::mojom::blink::VRDisplayHostPtr display_;
  device::mojom::blink::VRDisplayInfoPtr display_info_;
  unsigned int display_info_id_ = 0;

  mojo::Binding<device::mojom::blink::VRDisplayClient> display_client_binding_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_XR_XR_DEVICE_H_
