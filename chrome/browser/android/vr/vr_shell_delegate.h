// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_ANDROID_VR_VR_SHELL_DELEGATE_H_
#define CHROME_BROWSER_ANDROID_VR_VR_SHELL_DELEGATE_H_

#include <jni.h>

#include <map>
#include <memory>

#include "base/android/jni_weak_ref.h"
#include "base/callback.h"
#include "base/cancelable_callback.h"
#include "base/macros.h"
#include "chrome/browser/android/vr/vr_core_info.h"
#include "chrome/browser/vr/content_input_delegate.h"
#include "chrome/browser/vr/metrics/session_metrics_helper.h"
#include "device/vr/android/gvr/gvr_delegate_provider.h"
#include "device/vr/public/mojom/vr_service.mojom.h"
#include "device/vr/vr_device.h"
#include "third_party/gvr-android-sdk/src/libraries/headers/vr/gvr/capi/include/gvr_types.h"

namespace device {
class GvrDevice;
}

namespace vr {

// GENERATED_JAVA_ENUM_PACKAGE: org.chromium.chrome.browser.vr
enum class VrSupportLevel : int {
  kVrDisabled = 0,
  kVrNeedsUpdate = 1,  // VR Support is available, but needs update.
  kVrCardboard = 2,
  kVrDaydream = 3,  // Supports both Cardboard and Daydream viewer.
};

class VrShell;

class VrShellDelegate : public device::GvrDelegateProvider {
 public:
  VrShellDelegate(JNIEnv* env, jobject obj);
  ~VrShellDelegate() override;

  static device::GvrDelegateProvider* CreateVrShellDelegate();

  static VrShellDelegate* GetNativeVrShellDelegate(
      JNIEnv* env,
      const base::android::JavaRef<jobject>& jdelegate);

  void SetDelegate(VrShell* vr_shell, gvr::ViewerType viewer_type);
  void RemoveDelegate();

  void SetPresentResult(JNIEnv* env,
                        const base::android::JavaParamRef<jobject>& obj,
                        jboolean success);
  void RecordVrStartAction(JNIEnv* env,
                           const base::android::JavaParamRef<jobject>& obj,
                           jint start_action);
  void DisplayActivate(JNIEnv* env,
                       const base::android::JavaParamRef<jobject>& obj);
  void OnPause(JNIEnv* env, const base::android::JavaParamRef<jobject>& obj);
  void OnResume(JNIEnv* env, const base::android::JavaParamRef<jobject>& obj);
  bool IsClearActivatePending(JNIEnv* env,
                              const base::android::JavaParamRef<jobject>& obj);
  void Destroy(JNIEnv* env, const base::android::JavaParamRef<jobject>& obj);

  device::GvrDevice* GetDevice();

  void SendRequestPresentReply(device::mojom::XRSessionPtr session);

  // device::GvrDelegateProvider implementation.
  void ExitWebVRPresent() override;

 private:
  // device::GvrDelegateProvider implementation.
  bool ShouldDisableGvrDevice() override;
  void SetDeviceId(unsigned int device_id) override;
  void StartWebXRPresentation(
      device::mojom::VRDisplayInfoPtr display_info,
      device::mojom::XRDeviceRuntimeSessionOptionsPtr options,
      base::OnceCallback<void(device::mojom::XRSessionPtr)> callback) override;
  void OnListeningForActivateChanged(bool listening) override;

  void OnActivateDisplayHandled(bool will_not_present);
  void SetListeningForActivate(bool listening);
  void OnPresentResult(
      device::mojom::VRDisplayInfoPtr display_info,
      device::mojom::XRDeviceRuntimeSessionOptionsPtr options,
      base::OnceCallback<void(device::mojom::XRSessionPtr)> callback,
      bool success);

  std::unique_ptr<VrCoreInfo> MakeVrCoreInfo(JNIEnv* env);

  base::android::ScopedJavaGlobalRef<jobject> j_vr_shell_delegate_;
  unsigned int device_id_ = 0;
  VrShell* vr_shell_ = nullptr;

  // Deferred callback stored for later use in cases where vr_shell
  // wasn't ready yet. Used once SetDelegate is called.
  base::OnceCallback<void(bool)> on_present_result_callback_;

  // Mojo callback waiting for request present response. This is temporarily
  // stored here from OnPresentResult's outgoing ConnectPresentingService call
  // until the reply arguments are received by SendRequestPresentReply.
  base::OnceCallback<void(device::mojom::XRSessionPtr)>
      request_present_response_callback_;

  bool pending_successful_present_request_ = false;
  base::Optional<VrStartAction> pending_vr_start_action_;
  base::Optional<PresentationStartAction> possible_presentation_start_action_;

  scoped_refptr<base::SingleThreadTaskRunner> task_runner_;

  base::WeakPtrFactory<VrShellDelegate> weak_ptr_factory_;

  DISALLOW_COPY_AND_ASSIGN(VrShellDelegate);
};

}  // namespace vr

#endif  // CHROME_BROWSER_ANDROID_VR_VR_SHELL_DELEGATE_H_
