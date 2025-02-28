// Copyright (c) 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "device/vr/oculus/oculus_device.h"

#include <math.h>

#include <utility>

#include "base/memory/ptr_util.h"
#include "base/numerics/math_constants.h"
#include "build/build_config.h"
#include "device/vr/oculus/oculus_render_loop.h"
#include "device/vr/oculus/oculus_type_converters.h"
#include "third_party/libovr/src/Include/OVR_CAPI.h"
#include "third_party/libovr/src/Include/OVR_CAPI_D3D.h"
#include "ui/gfx/geometry/angle_conversions.h"

namespace device {

namespace {

mojom::VREyeParametersPtr GetEyeDetails(ovrSession session,
                                        const ovrHmdDesc& hmd_desc,
                                        ovrEyeType eye) {
  auto eye_parameters = mojom::VREyeParameters::New();
  auto render_desc =
      ovr_GetRenderDesc(session, eye, hmd_desc.DefaultEyeFov[eye]);
  eye_parameters->fieldOfView = mojom::VRFieldOfView::New();
  eye_parameters->fieldOfView->upDegrees =
      gfx::RadToDeg(atanf(render_desc.Fov.UpTan));
  eye_parameters->fieldOfView->downDegrees =
      gfx::RadToDeg(atanf(render_desc.Fov.DownTan));
  eye_parameters->fieldOfView->leftDegrees =
      gfx::RadToDeg(atanf(render_desc.Fov.LeftTan));
  eye_parameters->fieldOfView->rightDegrees =
      gfx::RadToDeg(atanf(render_desc.Fov.RightTan));

  auto offset = render_desc.HmdToEyeOffset;
  eye_parameters->offset = std::vector<float>({offset.x, offset.y, offset.z});

  auto texture_size =
      ovr_GetFovTextureSize(session, eye, render_desc.Fov, 1.0f);
  eye_parameters->renderWidth = texture_size.w;
  eye_parameters->renderHeight = texture_size.h;

  return eye_parameters;
}

mojom::VRDisplayInfoPtr CreateVRDisplayInfo(unsigned int id,
                                            ovrSession session) {
  mojom::VRDisplayInfoPtr display_info = mojom::VRDisplayInfo::New();
  display_info->index = id;
  display_info->displayName = std::string("Oculus");
  display_info->capabilities = mojom::VRDisplayCapabilities::New();
  display_info->capabilities->hasPosition = true;
  display_info->capabilities->hasExternalDisplay = true;
  display_info->capabilities->canPresent = true;
  display_info->webvr_default_framebuffer_scale = 1.0;
  display_info->webxr_default_framebuffer_scale = 1.0;

  ovrHmdDesc hmdDesc = ovr_GetHmdDesc(session);
  display_info->leftEye = GetEyeDetails(session, hmdDesc, ovrEye_Left);
  display_info->rightEye = GetEyeDetails(session, hmdDesc, ovrEye_Right);

  display_info->stageParameters = mojom::VRStageParameters::New();
  ovr_SetTrackingOriginType(session, ovrTrackingOrigin_FloorLevel);
  ovrTrackingState ovr_state = ovr_GetTrackingState(session, 0, true);
  float floor_height = ovr_state.HeadPose.ThePose.Position.y;
  ovr_SetTrackingOriginType(session, ovrTrackingOrigin_EyeLevel);

  display_info->stageParameters->standingTransform = {
      1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, floor_height, 0, 1};

  ovrVector3f boundary_size;
  ovr_GetBoundaryDimensions(session, ovrBoundary_PlayArea, &boundary_size);
  display_info->stageParameters->sizeX = boundary_size.x;
  display_info->stageParameters->sizeZ = boundary_size.z;

  return display_info;
}

}  // namespace

OculusDevice::OculusDevice()
    : VRDeviceBase(VRDeviceId::OCULUS_DEVICE_ID),
      main_thread_task_runner_(base::ThreadTaskRunnerHandle::Get()),
      exclusive_controller_binding_(this),
      weak_ptr_factory_(this) {
  StartOvrSession();
  if (!session_) {
    return;
  }

  SetVRDisplayInfo(CreateVRDisplayInfo(GetId(), session_));

  render_loop_ = std::make_unique<OculusRenderLoop>(
      base::BindRepeating(&OculusDevice::OnPresentationEnded,
                          weak_ptr_factory_.GetWeakPtr()),
      base::BindRepeating(&OculusDevice::OnControllerUpdated,
                          weak_ptr_factory_.GetWeakPtr()));
}

OculusDevice::~OculusDevice() {
  StopOvrSession();

  device::GamepadDataFetcherManager::GetInstance()->RemoveSourceFactory(
      device::GAMEPAD_SOURCE_OCULUS);
  data_fetcher_ = nullptr;
}

void OculusDevice::RequestSession(
    mojom::XRDeviceRuntimeSessionOptionsPtr options,
    mojom::XRRuntime::RequestSessionCallback callback) {
  if (!options->immersive) {
    ReturnNonImmersiveSession(std::move(callback));
    return;
  }

  StopOvrSession();

  if (!render_loop_->IsRunning())
    render_loop_->Start();

  if (!render_loop_->IsRunning()) {
    std::move(callback).Run(nullptr, nullptr);
    StartOvrSession();
    return;
  }

  auto on_request_present_result =
      base::BindOnce(&OculusDevice::OnRequestSessionResult,
                     weak_ptr_factory_.GetWeakPtr(), std::move(callback));

  render_loop_->task_runner()->PostTask(
      FROM_HERE, base::BindOnce(&OculusRenderLoop::RequestSession,
                                render_loop_->GetWeakPtr(), std::move(options),
                                std::move(on_request_present_result)));
}

void OculusDevice::OnRequestSessionResult(
    mojom::XRRuntime::RequestSessionCallback callback,
    bool result,
    mojom::XRSessionPtr session) {
  if (!result) {
    std::move(callback).Run(nullptr, nullptr);

    // Start magic window again.
    StartOvrSession();
    return;
  }

  OnStartPresenting();

  mojom::XRSessionControllerPtr session_controller;
  exclusive_controller_binding_.Bind(mojo::MakeRequest(&session_controller));

  // Unretained is safe because the error handler won't be called after the
  // binding has been destroyed.
  exclusive_controller_binding_.set_connection_error_handler(
      base::BindOnce(&OculusDevice::OnPresentingControllerMojoConnectionError,
                     base::Unretained(this)));

  std::move(callback).Run(std::move(session), std::move(session_controller));

  if (!oculus_gamepad_factory_) {
    oculus_gamepad_factory_ =
        new OculusGamepadDataFetcher::Factory(GetId(), this);
    GamepadDataFetcherManager::GetInstance()->AddFactory(
        oculus_gamepad_factory_);
  }
}

void OculusDevice::OnControllerUpdated(ovrInputState input,
                                       ovrInputState remote,
                                       ovrTrackingState tracking,
                                       bool has_touch,
                                       bool has_remote) {
  if (data_fetcher_)
    data_fetcher_->UpdateGamepadData(
        {input, remote, tracking, has_touch, has_remote});
}

// XRSessionController
void OculusDevice::SetFrameDataRestricted(bool restricted) {
  // Presentation sessions can not currently be restricted.
  DCHECK(false);
}

void OculusDevice::OnPresentingControllerMojoConnectionError() {
  render_loop_->task_runner()->PostTask(
      FROM_HERE, base::BindOnce(&OculusRenderLoop::ExitPresent,
                                render_loop_->GetWeakPtr()));
  OnExitPresent();
  exclusive_controller_binding_.Close();
}

void OculusDevice::OnPresentationEnded() {
  StartOvrSession();
}

void OculusDevice::StartOvrSession() {
  ovrInitParams initParams = {ovrInit_RequestVersion | ovrInit_Invisible,
                              OVR_MINOR_VERSION, NULL, 0, 0};
  ovrResult result = ovr_Initialize(&initParams);
  if (OVR_FAILURE(result)) {
    return;
  }

  ovrGraphicsLuid luid;
  result = ovr_Create(&session_, &luid);
  if (OVR_FAILURE(result)) {
    return;
  }
}

void OculusDevice::StopOvrSession() {
  if (session_) {
    // Shut down our current session so the presentation session can begin.
    ovr_Destroy(session_);
    session_ = nullptr;
    ovr_Shutdown();
  }
}

void OculusDevice::OnMagicWindowFrameDataRequest(
    mojom::XRFrameDataProvider::GetFrameDataCallback callback) {
  if (!session_) {
    std::move(callback).Run(nullptr);
    return;
  }

  ovrTrackingState state = ovr_GetTrackingState(session_, 0, false);

  mojom::XRFrameDataPtr frame_data = mojom::XRFrameData::New();
  frame_data->pose = mojo::ConvertTo<mojom::VRPosePtr>(state.HeadPose.ThePose);
  std::move(callback).Run(std::move(frame_data));
}

void OculusDevice::RegisterDataFetcher(OculusGamepadDataFetcher* data_fetcher) {
  data_fetcher_ = data_fetcher;
}

}  // namespace device
