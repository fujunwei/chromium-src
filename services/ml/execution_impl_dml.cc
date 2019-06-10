// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "services/ml/execution_impl_dml.h"

#include <utility>

#include "services/ml/common.h"
#include "services/ml/dml_d3dx12_utils.h"
#include "services/ml/float16_compressor.h"
#include "services/ml/public/mojom/constants.mojom.h"

namespace ml {

ExecutionImplDML::ExecutionImplDML(scoped_refptr<CompiledModelDML> dml,
                                   mojom::ExecutionInitParamsPtr params)
    : params_(std::move(params)), dml_(dml) {}

ExecutionImplDML::~ExecutionImplDML() = default;

void ExecutionImplDML::StartCompute(StartComputeCallback callback) {
  // Bind and execute the operator on the GPU.
  ID3D12DescriptorHeap* d3D12_descriptor_heaps[] = {
      dml_->descriptor_heap_.Get()};
  dml_->command_list_->SetDescriptorHeaps(ARRAYSIZE(d3D12_descriptor_heaps),
                                          d3D12_descriptor_heaps);

  unsigned long long sys_1 = GetCurrentTimeMsec();
  uint32_t memory_offset = 0;
  HRESULT hr = S_OK;
  for (size_t i = 0; i < params_->inputs.size(); ++i) {
    const mojom::OperandInfoPtr& operand = params_->inputs[i];
    uint32_t offset = memory_offset;
    uint32_t length = GetRequiredSize(operand);
    memory_offset += length;
    auto mapping = params_->memory->MapAtOffset(length, offset);
    ComPtr<ID3D12Resource> upload_resource =
        dml_->operand_map_[operand->index]->upload_resource_;
    ComPtr<ID3D12Resource> operand_resource =
        dml_->operand_map_[operand->index]->operand_resource_;
    hr = UploadTensorResource(static_cast<void*>(mapping.get()), length,
                              upload_resource, operand_resource,
                              dml_->command_list_);
    if (FAILED(hr)) {
      LOG(ERROR) << "Failed uploading tensor resource for inputs data.";
    }
  }
  // hr = CloseExecuteResetWait(dml_->d3d12_device_, dml_->command_queue_,
  //                            dml_->command_allocator_, dml_->command_list_);
  // if (FAILED(hr)) {
  //   LOG(ERROR) << "Failed executing command list for compiled operators.";
  //   return;
  // }
  unsigned long long sys_2 = GetCurrentTimeMsec();
  LOG(ERROR) << L"======upload time: " << sys_2 - sys_1 << "\n";
  for (size_t i = 0; i < dml_->operations_.size(); ++i) {
    hr = ExecuteCompiledOperator(dml_->operations_[i]->compiled_operator_.Get(),
                                 dml_->operations_[i], i);
    if (FAILED(hr)) {
      LOG(ERROR) << "Failed executing operator.";
      std::move(callback).Run(mojom::OP_FAILED);
      return;
    }
  }
  hr = CloseExecuteResetWait(dml_->d3d12_device_, dml_->command_queue_,
                             dml_->command_allocator_, dml_->command_list_);
  if (FAILED(hr)) {
    LOG(ERROR) << "Failed executing command list for compiled operators.";
    return;
  }
  sys_1 = GetCurrentTimeMsec();
  LOG(ERROR) << L"======predict time: " << sys_1 - sys_2 << "\n";

  hr = ReadResultBack(memory_offset);
  if (FAILED(hr)) {
    LOG(ERROR) << "Failed reading result.";
    std::move(callback).Run(mojom::OP_FAILED);
    return;
  }

  std::move(callback).Run(mojom::NOT_ERROR);
}

HRESULT ExecutionImplDML::ExecuteCompiledOperator(
    IDMLCompiledOperator* compiled_operator,
    const std::unique_ptr<OperationDML>& operation,
    uint32_t operation_index) {
  // Record execution of the compiled operator.
  dml_->command_recorder_->RecordDispatch(dml_->command_list_.Get(),
                                          compiled_operator,
                                          operation->binding_table_.Get());
  CD3DX12_RESOURCE_BARRIER resource_barrier =
      CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
  dml_->command_list_->ResourceBarrier(1, &resource_barrier);

  return S_OK;
}

HRESULT ExecutionImplDML::ReadResultBack(uint32_t memory_offset) {
  // The output buffer now contains the result of the identity operator,
  // so read it back if you want the CPU to access it.
  HRESULT hr = S_OK;
  for (size_t i = 0; i < params_->outputs.size(); ++i) {
    size_t output_index = params_->outputs[i]->index;
    ComPtr<ID3D12Resource> output_resource =
        dml_->operand_map_[output_index]->operand_resource_;
    ComPtr<ID3D12Resource> readback_buffer =
        dml_->operand_map_[output_index]->readback_resource_;
    CD3DX12_RESOURCE_BARRIER resource_barrier =
        CD3DX12_RESOURCE_BARRIER::Transition(
            output_resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    dml_->command_list_->ResourceBarrier(1, &resource_barrier);

    dml_->command_list_->CopyResource(readback_buffer.Get(),
                                      output_resource.Get());
  }

  CloseExecuteResetWait(dml_->d3d12_device_, dml_->command_queue_,
                        dml_->command_allocator_, dml_->command_list_);

  for (size_t i = 0; i < params_->outputs.size(); ++i) {
    size_t output_index = params_->outputs[i]->index;
    ComPtr<ID3D12Resource> readback_buffer =
        dml_->operand_map_[output_index]->readback_resource_;
    D3D12_RANGE tensor_buffer_range = {
        0, dml_->operand_map_[output_index]->SizeInBytes()};
    void* output_buffer_data = nullptr;
    hr = readback_buffer->Map(0, &tensor_buffer_range, &output_buffer_data);
    if (FAILED(hr)) {
      LOG(ERROR) << "Failed map buffer for reading result.";
      return hr;
    }
    // Convert float16 to float32.
    uint16_t* float16_data = static_cast<uint16_t*>(output_buffer_data);
    std::vector<float> float32_data;
    const mojom::OperandInfoPtr& operand = params_->outputs[i];
    size_t length = product(operand->dimensions);
    for (size_t i = 0; i < length; ++i) {
      float32_data.push_back(Float16Compressor::decompress(float16_data[i]));
    }

    // Copy to out memory.
    const uint32_t offset = memory_offset;
    const uint32_t output_buffer_size = GetRequiredSize(operand);
    memory_offset += output_buffer_size;
    auto mapping = params_->memory->MapAtOffset(output_buffer_size, offset);
    memcpy(mapping.get(), float32_data.data(), output_buffer_size);

    D3D12_RANGE empty_range{0, 0};
    readback_buffer->Unmap(0, &empty_range);
  }

  return S_OK;
}

}  // namespace ml
