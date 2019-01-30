// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_ERRORCOMPILATION_H_
#define THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_ERRORCOMPILATION_H_

#include "third_party/blink/renderer/core/dom/dom_exception.h"

namespace blink {

namespace ml_error {

DOMException* CreateException(int, const String&);
}

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_MODULES_ML_ML_ERRORCOMPILATION_H_