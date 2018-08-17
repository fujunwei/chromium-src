// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/blink/renderer/modules/ml/neural_network_context.h"

#include "services/service_manager/public/cpp/interface_provider.h"
#include "third_party/blink/renderer/core/dom/document.h"
#include "third_party/blink/renderer/core/dom/dom_exception.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/platform/bindings/exception_code.h"

#include "third_party/blink/renderer/modules/ml/model.h"
#include "third_party/blink/renderer/modules/ml/navigator_ml.h"

namespace blink {

NeuralNetworkContext::NeuralNetworkContext(NavigatorML* navigator_ml)
    : ContextLifecycleObserver(navigator_ml->GetDocument()) {
  navigator_ml->GetDocument()->GetFrame()->GetInterfaceProvider().GetInterface(
      mojo::MakeRequest(&neural_network_));
  neural_network_.set_connection_error_handler(
      WTF::Bind(&NeuralNetworkContext::OnConnectionError, WrapWeakPersistent(this)));
}

NeuralNetworkContext::~NeuralNetworkContext() {}

void NeuralNetworkContext::Dispose() {}

void NeuralNetworkContext::ContextDestroyed(ExecutionContext*) {
  Dispose();
}

ScriptPromise NeuralNetworkContext::createModel(ScriptState* script_state) {
  ScriptPromiseResolver* resolver = ScriptPromiseResolver::Create(script_state);
  ScriptPromise promise = resolver->Promise();
  if (!neural_network_) {
    resolver->Reject(
        DOMException::Create(DOMExceptionCode::kNotSupportedError,
                             "Neural Network service unavailable."));
    return promise;
  }
  requests_.insert(resolver);

  neural_network_->CreateModel(WTF::Bind(&NeuralNetworkContext::OnCreateModel,
                                         WrapPersistent(this),
                                         WrapPersistent(resolver)));
  return promise;
}

void NeuralNetworkContext::OnCreateModel(
    ScriptPromiseResolver* resolver, int32_t result_code, ml::mojom::blink::ModelInitParamsPtr init_params) {
  DCHECK(requests_.Contains(resolver));
  requests_.erase(resolver);

  if (result_code == ml::mojom::blink::NOT_ERROR) {
    resolver->Resolve(new Model(std::move(init_params->model)));
  } else {
    String msg("createModel fails: ");
    msg.append(String::Number(result_code));
    resolver->Reject(
        DOMException::Create(DOMExceptionCode::kInvalidStateError, msg));
  }
}

void NeuralNetworkContext::Trace(blink::Visitor* visitor) {
  visitor->Trace(requests_);
  ScriptWrappable::Trace(visitor);
  ContextLifecycleObserver::Trace(visitor);
}

void NeuralNetworkContext::OnConnectionError() {
  for (const auto& request : requests_) {
    request->Reject(DOMException::Create(DOMExceptionCode::kNotSupportedError,
                                         "Neural Network is not implemented."));
  }
  requests_.clear();
  neural_network_.reset();
}


}  // namespace blink
