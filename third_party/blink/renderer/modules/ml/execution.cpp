// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/execution.h"

#import <OpenGL/gl3.h>

#include "gpu/GLES2/gl2extchromium.h"
#include "gpu/command_buffer/client/gles2_interface.h"
#include "gpu/command_buffer/client/gles2_lib.h"
#include "gpu/command_buffer/client/gpu_memory_buffer_manager.h"
#include "gpu/command_buffer/client/shared_image_interface.h"
#include "gpu/command_buffer/common/shared_image_usage.h"
#include "mojo/public/cpp/bindings/interface_ptr.h"
#include "mojo/public/cpp/system/platform_handle.h"
#include "services/ml/public/mojom/constants.mojom-blink.h"
#include "services/service_manager/public/cpp/interface_provider.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"
#include "third_party/blink/renderer/platform/graphics/gpu/drawing_buffer.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"
#include "ui/gfx/mojo/buffer_types.mojom-blink.h"

namespace blink {

namespace {

uint32_t product(const WTF::Vector<uint32_t>& dims) {
  uint32_t prod = 1;

  for (wtf_size_t i = 0; i < dims.size(); ++i)
    prod *= dims[i];

  return prod;
}

uint32_t requiredSize(int32_t type, const WTF::Vector<uint32_t>& dimensions) {
  if (type == ml::mojom::blink::FLOAT32) {
    return sizeof(float);
  } else if (type == ml::mojom::blink::INT32) {
    return sizeof(int32_t);
  } else if (type == ml::mojom::blink::UINT32) {
    return sizeof(uint32_t);
  } else if (type == ml::mojom::blink::TENSOR_FLOAT32) {
    return product(dimensions) * sizeof(float);
  } else if (type == ml::mojom::blink::TENSOR_INT32) {
    return product(dimensions) * sizeof(int32_t);
  } else if (type == ml::mojom::blink::TENSOR_QUANT8_ASYMM) {
    return product(dimensions) * sizeof(int8_t);
  } else {
    NOTREACHED();
  }

  return 0;
}

}  // namespace

Execution::Execution(ml::mojom::blink::ExecutionInitParamsPtr init_params) {
  execution_.Bind(std::move(init_params->execution));
  execution_.set_connection_error_handler(
      WTF::Bind(&Execution::OnConnectionError, WrapWeakPersistent(this)));

  uint32_t total_length = 0;
  memory_ = std::move(init_params->memory);
  for (wtf_size_t i = 0; i < init_params->inputs.size(); ++i) {
    uint32_t offset = total_length;
    uint32_t length = requiredSize(init_params->inputs[i]->type,
                                   init_params->inputs[i]->dimensions);
    inputs_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }

  for (wtf_size_t i = 0; i < init_params->outputs.size(); ++i) {
    uint32_t offset = total_length;
    uint32_t length = requiredSize(init_params->outputs[i]->type,
                                   init_params->outputs[i]->dimensions);
    outputs_.push_back(std::make_unique<OperandInfo>(
        offset, length, memory_->MapAtOffset(length, offset)));
    total_length += length;
  }

  output_buffer_views_.resize(init_params->outputs.size());
}

Execution::~Execution() = default;

void Execution::setInput(uint32_t index,
                         MaybeShared<DOMArrayBufferView> data,
                         ExceptionState& exception_state) {
  if (index >= inputs_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid index");
    return;
  }

  std::unique_ptr<OperandInfo>& info = inputs_.at(index);
  uint32_t length = data.View()->byteLength();
  if (info->length != length) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid data");
    return;
  }

  memcpy(static_cast<void*>(info->mapping.get()), data.View()->BaseAddress(),
         length);
}

void Execution::setInput(uint32_t index,
                         WebGLRenderingContext* context,
                         WebGLTexture* texture,
                         ExceptionState& exception_state) {
  if (index >= inputs_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid index");
    return;
  }

  std::unique_ptr<OperandInfo>& info = inputs_.at(index);
  // 32 is the length of GLuint type, see "typedef unsigned int GLuint".
  if (info->length < 32) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid data");
    return;
  }

  DrawingBuffer* drawing_buffer = context->GetDrawingBuffer();
  gpu::gles2::GLES2Interface* gl = drawing_buffer->ContextGL();
  WebGraphicsContext3DProvider* context_provider =
      drawing_buffer->ContextProvider();

  gpu::SharedImageInterface* sii = context_provider->SharedImageInterface();
  gpu::GpuMemoryBufferManager* gpu_memory_buffer_manager =
      Platform::Current()->GetGpuMemoryBufferManager();

  gpu::Mailbox mailbox;
  GLuint texture_id = 0;
  std::unique_ptr<gfx::GpuMemoryBuffer> gpu_memory_buffer;
  IntSize size(100, 100);  // Only for testing
  gpu_memory_buffer = gpu_memory_buffer_manager->CreateGpuMemoryBuffer(
      gfx::Size(size), gfx::BufferFormat::RGBA_8888, gfx::BufferUsage::SCANOUT,
      gpu::kNullSurfaceHandle);
  if (!gpu_memory_buffer)
    return;

  mailbox = sii->CreateSharedImage(
      gpu_memory_buffer.get(), gpu_memory_buffer_manager,
      context->ColorParams().GetStorageGfxColorSpace(),
      gpu::SHARED_IMAGE_USAGE_GLES2 |
          gpu::SHARED_IMAGE_USAGE_GLES2_FRAMEBUFFER_HINT |
          gpu::SHARED_IMAGE_USAGE_DISPLAY | gpu::SHARED_IMAGE_USAGE_SCANOUT);

  // Import the allocated SharedImage into GL.
  gpu::SyncToken sync_token = sii->GenUnverifiedSyncToken();
  gl->WaitSyncTokenCHROMIUM(sync_token.GetConstData());
  texture_id = gl->CreateAndTexStorage2DSharedImageCHROMIUM(mailbox.name);
  gl->BindTexture(GL_TEXTURE_RECTANGLE, texture_id);
  gl->BeginSharedImageAccessDirectCHROMIUM(
      texture_id, GL_SHARED_IMAGE_ACCESS_MODE_READWRITE_CHROMIUM);

  // Testing.
  if (false) {
    GLuint fbo = 0;
    gl->GenFramebuffers(1, &fbo);
    gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);
    gl->FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_RECTANGLE, texture_id, 0);
    gl->ClearColor(0, 1, 0, 1);
    gl->Clear(GL_COLOR_BUFFER_BIT);
    gl->Flush();

    uint32_t result;
    gl->ReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &result);
    // The expected result is 4278255360.
  }

  gfx::GpuMemoryBufferHandle handle = gpu_memory_buffer->CloneHandle();
  auto buffer_handle = gfx::mojom::blink::GpuMemoryBufferHandle::New();
  buffer_handle->id = gfx::mojom::blink::GpuMemoryBufferId::New(handle.id.id);
  buffer_handle->offset = handle.offset;
  buffer_handle->stride = handle.stride;
  buffer_handle->platform_handle =
      gfx::mojom::blink::GpuMemoryBufferPlatformHandle::NewMachPort(
          mojo::WrapMachPort(handle.mach_port.get()));

  execution_->SetGpuMemoryBufferHandle(std::move(buffer_handle));
}

void Execution::setOutput(uint32_t index,
                          MaybeShared<DOMArrayBufferView> data,
                          ExceptionState& exception_state) {
  if (index >= output_buffer_views_.size()) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid index");
    return;
  }

  std::unique_ptr<OperandInfo>& info = outputs_.at(index);
  uint32_t length = data.View()->byteLength();
  if (info->length != length) {
    exception_state.ThrowDOMException(DOMExceptionCode::kInvalidStateError,
                                      "Invalid data");
    return;
  }

  output_buffer_views_[index] = data.View();
}

ScriptPromise Execution::startCompute(ScriptState* script_state) {
  ScriptPromiseResolver* resolver = ScriptPromiseResolver::Create(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!execution_) {
    resolver->Reject(
        DOMException::Create(DOMExceptionCode::kNotSupportedError,
                             "Neural Network service unavailable."));
    return promise;
  }

  requests_.insert(resolver);

  execution_->StartCompute(WTF::Bind(&Execution::OnStartCompute,
                                     WrapPersistent(this),
                                     WrapPersistent(resolver)));
  return promise;
}

void Execution::OnStartCompute(ScriptPromiseResolver* resolver,
                               int32_t result_code) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code != ml::mojom::blink::NOT_ERROR) {
    return resolver->Reject(DOMException::Create(
        DOMExceptionCode::kInvalidStateError,
        "startCompute fails " + String::Number(result_code)));
  }

  for (wtf_size_t i = 0; i < outputs_.size(); ++i) {
    DOMArrayBufferView* view = output_buffer_views_.at(i);
    if (view) {
      uint32_t length = view->byteLength();
      std::unique_ptr<OperandInfo>& info = outputs_.at(i);
      memcpy(view->BaseAddress(), static_cast<const void*>(info->mapping.get()),
             length);
    }
  }
  resolver->Resolve(result_code);
}

void Execution::OnResultCode(ScriptPromiseResolver* resolver,
                             const String& operation_name,
                             int32_t result_code) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code != ml::mojom::blink::NOT_ERROR) {
    return resolver->Reject(DOMException::Create(
        DOMExceptionCode::kInvalidStateError,
        operation_name + "fails: " + String::Number(result_code)));
  }

  resolver->Resolve(result_code);
}

void Execution::Trace(blink::Visitor* visitor) {
  visitor->Trace(requests_);
  visitor->Trace(output_buffer_views_);
  ScriptWrappable::Trace(visitor);
}

void Execution::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(DOMException::Create(DOMExceptionCode::kNotSupportedError,
                                         "Execution is not implemented."));
  }

  requests_.clear();
  execution_.reset();
}

}  // namespace blink
