// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef DEVICE_FIDO_GET_ASSERTION_TASK_H_
#define DEVICE_FIDO_GET_ASSERTION_TASK_H_

#include <stdint.h>

#include <memory>
#include <vector>

#include "base/callback.h"
#include "base/component_export.h"
#include "base/macros.h"
#include "base/memory/weak_ptr.h"
#include "base/optional.h"
#include "device/fido/ctap_get_assertion_request.h"
#include "device/fido/device_operation.h"
#include "device/fido/fido_constants.h"
#include "device/fido/fido_task.h"

namespace device {

class AuthenticatorGetAssertionResponse;

// Represents per device sign operation on CTAP1/CTAP2 devices.
// https://fidoalliance.org/specs/fido-v2.0-rd-20161004/fido-client-to-authenticator-protocol-v2.0-rd-20161004.html#authenticatorgetassertion
class COMPONENT_EXPORT(DEVICE_FIDO) GetAssertionTask : public FidoTask {
 public:
  using GetAssertionTaskCallback = base::OnceCallback<void(
      CtapDeviceResponseCode,
      base::Optional<AuthenticatorGetAssertionResponse>)>;
  using SignOperation = DeviceOperation<CtapGetAssertionRequest,
                                        AuthenticatorGetAssertionResponse>;

  GetAssertionTask(FidoDevice* device,
                   CtapGetAssertionRequest request,
                   GetAssertionTaskCallback callback);
  ~GetAssertionTask() override;

 private:
  // FidoTask:
  void StartTask() override;

  void GetAssertion();
  void U2fSign();

  CtapGetAssertionRequest request_;
  std::unique_ptr<SignOperation> sign_operation_;
  GetAssertionTaskCallback callback_;
  base::WeakPtrFactory<GetAssertionTask> weak_factory_;

  DISALLOW_COPY_AND_ASSIGN(GetAssertionTask);
};

}  // namespace device

#endif  // DEVICE_FIDO_GET_ASSERTION_TASK_H_
