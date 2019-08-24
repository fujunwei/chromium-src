// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_ML_COMPILATION_DELEGATE_MKL_DNN_H_
#define SERVICES_ML_COMPILATION_DELEGATE_MKL_DNN_H_

#include <map>
#include <memory>
#include <vector>

#include "base/macros.h"
#include "mojo/public/cpp/bindings/strong_binding.h"
#include "services/ml/common.h"
#include "services/ml/compilation_impl.h"
#include "services/ml/public/interfaces/model.mojom.h"
#include "third_party/mkl-dnn/include/mkldnn.h"

namespace ml {

struct MemoryMklDnn {
  mkldnn_primitive_t primitive;
  void* buffer;
};

struct OperationMklDnn {
  explicit OperationMklDnn(mkldnn_primitive_t);
  ~OperationMklDnn();
  OperationMklDnn(const OperationMklDnn&);

  mkldnn_primitive_t primitive;
  int32_t type;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs; 
};

struct CompiledModelMklDnn {
  CompiledModelMklDnn();
  ~CompiledModelMklDnn();

  std::map<std::string, MemoryMklDnn> memories;
  std::vector<OperationMklDnn> operations;
  mkldnn_engine_t engine;
};

class CompilationDelegateMklDnn : public CompilationDelegate {
 public:
  explicit CompilationDelegateMklDnn(const CompilationImpl*);
  ~CompilationDelegateMklDnn() override;

  int32_t Compile() override;
  std::unique_ptr<mojom::Execution> CreateExecution(
      mojom::ExecutionInitParamsPtr params) override;

 private:
  friend class ExecutionImplMklDnn;
  int32_t MkldnnInit();
  int32_t MkldnnCompile();
  int32_t MkldnnGetMemoryFormat(const std::vector<uint32_t>&, mkldnn_memory_format_t*);
  int32_t MkldnnGetDataType(int32_t, mkldnn_data_type_t*);
  int32_t MkldnnGetDims(const std::vector<uint32_t>&, std::vector<int32_t>&);
  int32_t MkldnnAddMemory(uint32_t index, mkldnn_memory_format_t* format = nullptr);
  int32_t MkldnnAddInput(uint32_t index);
  int32_t MkldnnAddOutput(uint32_t index);
  int32_t MkldnnAddReorder(const std::string& input_name,
                           const std::string& output_name,
                           bool run = false);
  int32_t MkldnnAddActivationByFusedCode(const std::string& input,
                                        const std::string& id,
                                        int32_t fuse_code);
  int32_t MkldnnAddElementwise(int32_t type,
                              const std::vector<uint32_t>& inputs,
                              const std::vector<uint32_t>& outputs);
  int32_t MkldnnAddConvolution(int32_t type,
                              const std::vector<uint32_t>& inputs,
                              const std::vector<uint32_t>& outputs);
  int32_t MkldnnAddPooling(int32_t type,
                          const std::vector<uint32_t>& inputs,
                          const std::vector<uint32_t>& outputs);
  int32_t MkldnnAddSoftmax(int32_t type,
                          const std::vector<uint32_t>& inputs,
                          const std::vector<uint32_t>& outputs);
  int32_t MkldnnAddReshape(int32_t type,
                          const std::vector<uint32_t>& inputs,
                          const std::vector<uint32_t>& outputs);
  int32_t MkldnnAddConcatenation(int32_t type,
                                const std::vector<uint32_t>& inputs,
                                const std::vector<uint32_t>& outputs);
  int32_t MkldnnAddFullyConnected(int32_t type,
                                 const std::vector<uint32_t>& inputs,
                                 const std::vector<uint32_t>& outputs);
  int32_t MkldnnAddResizeBilinear(int32_t type,
                                 const std::vector<uint32_t>& inputs,
                                 const std::vector<uint32_t>& outputs);

 private:
  const CompilationImpl* compilation_;

  std::shared_ptr<CompiledModelMklDnn> compiled_model_;

  DISALLOW_COPY_AND_ASSIGN(CompilationDelegateMklDnn);
};

}  // namespace ml

#endif  // SERVICES_ML_COMPILATION_DELEGATE_MKL_DNN_H_