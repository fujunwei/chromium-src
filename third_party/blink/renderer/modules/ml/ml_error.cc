// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/ml_error.h"

#include "services/ml/public/mojom/constants.mojom-blink.h"

namespace blink {

namespace {

const char kNoErrorMessage[] = "No Error.";
const char kBadDataErrorMessage[] = "Bad data.";
const char kOperationFailedErrorMessage[] = "Operation failed.";
const char kBadStateErrorMessage[] = "Bad state.";
const char kUnsupportedPlatformErrorMessage[] =
    "Current platform isn't supported.";
const char kUnsupportedPreferenceErrorMessage[] =
    "The preference isn't supported";
const char kUnsupportedOperationErrorMessage[] = "Operation is not supported.";
const char kUnsupportedFuseCodeErrorMessage[] = "Fuse code is not supproted.";
const char kDepthwiseMultiplierErrorMessage[] =
    "Output channels in depthwise convolution descriptor must be multiplie"
    " of input channels.";
const char kUnsupportedBetaErrorMessage[] =
    "The value of Beta must be 1 in softmax Operation.";
const char kUnsupportedAxisErrorMessage[] =
    "The value of axis must be 3 in concatentation Operation.";
const char kDifferentBatchSizeErrorMessage[] =
    "The batch size for arithmetic must be same.";
const char kIntegerResizeBilinearErrorMessage[] =
    "The upsampling factor for the x/y must be integer in resize bilinear.";
const char kInvalidContextErrorMessage[] = "Invalid context.";

DOMExceptionCode ErrorCode(int result_code) {
  switch (result_code) {
    case ml::mojom::blink::NOT_ERROR:
      return DOMExceptionCode::kNoError;
    case ml::mojom::blink::OP_FAILED:
      return DOMExceptionCode::kOperationError;
    case ml::mojom::blink::BAD_STATE:
    case ml::mojom::blink::INVALID_CONTEXT:
      return DOMExceptionCode::kInvalidStateError;
    case ml::mojom::blink::UNSUPPORTED_PLATFORM:
    case ml::mojom::blink::UNSUPPORTED_PREFERENCE:
    case ml::mojom::blink::UNSUPPORTED_OPERATION:
    case ml::mojom::blink::UNSUPPORTED_FUSE_CODE:
    case ml::mojom::blink::UNSUPPORTED_BETA:
    case ml::mojom::blink::UNSUPPORTED_AXIS:
      return DOMExceptionCode::kNotSupportedError;
    case ml::mojom::blink::BAD_DATA:
    case ml::mojom::blink::DEPTHWISE_MULTIPLIER:
    case ml::mojom::blink::DIFFERENT_BATCH_SIZE:
    case ml::mojom::blink::INTEGER_RESIZE_BILINEAR:
      return DOMExceptionCode::kDataError;
    default:
      NOTREACHED();
      return DOMExceptionCode::kUnknownError;
  }
}

const char* ErrorMessage(int result_code) {
  switch (result_code) {
    case ml::mojom::blink::NOT_ERROR:
      return kNoErrorMessage;
    case ml::mojom::blink::BAD_DATA:
      return kBadDataErrorMessage;
    case ml::mojom::blink::OP_FAILED:
      return kOperationFailedErrorMessage;
    case ml::mojom::blink::BAD_STATE:
      return kBadStateErrorMessage;
    case ml::mojom::blink::UNSUPPORTED_PLATFORM:
      return kUnsupportedPlatformErrorMessage;
    case ml::mojom::blink::UNSUPPORTED_PREFERENCE:
      return kUnsupportedPreferenceErrorMessage;
    case ml::mojom::blink::UNSUPPORTED_OPERATION:
      return kUnsupportedOperationErrorMessage;
    case ml::mojom::blink::UNSUPPORTED_FUSE_CODE:
      return kUnsupportedFuseCodeErrorMessage;
    case ml::mojom::blink::UNSUPPORTED_BETA:
      return kUnsupportedBetaErrorMessage;
    case ml::mojom::blink::UNSUPPORTED_AXIS:
      return kUnsupportedAxisErrorMessage;
    case ml::mojom::blink::DEPTHWISE_MULTIPLIER:
      return kDepthwiseMultiplierErrorMessage;
    case ml::mojom::blink::DIFFERENT_BATCH_SIZE:
      return kDifferentBatchSizeErrorMessage;
    case ml::mojom::blink::INTEGER_RESIZE_BILINEAR:
      return kIntegerResizeBilinearErrorMessage;
    case ml::mojom::blink::INVALID_CONTEXT:
      return kInvalidContextErrorMessage;
    default:
      NOTREACHED();
      return nullptr;
  }
}

}  // namespace

namespace ml_error {

DOMException* CreateException(int result_code, const String& operation) {
  return DOMException::Create(ErrorCode(result_code),
                              "\"" + operation + "\" function fails because " +
                                  ErrorMessage(result_code));
}

}  // namespace ml_error

}  // namespace blink
