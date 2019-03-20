// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/cl_dnn_symbol_table.h"

namespace ml {

LATE_BINDING_SYMBOL_TABLE_DEFINE_BEGIN(ClDnnSymbolTable,
                                       "/opt/intel/computer_vision_sdk/"
                                       "inference_engine/external/cldnn/lib/"
                                       "libclDNN64.so")
#define X(sym) LATE_BINDING_SYMBOL_TABLE_DEFINE_ENTRY(ClDnnSymbolTable, sym)
CL_DNN_SYMBOLS_LIST
#undef X
LATE_BINDING_SYMBOL_TABLE_DEFINE_END(ClDnnSymbolTable)

}  // namespace ml
