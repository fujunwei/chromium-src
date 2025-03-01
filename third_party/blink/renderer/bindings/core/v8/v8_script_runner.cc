/*
 * Copyright (C) 2009 Google Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE INC. AND ITS CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL APPLE INC. OR ITS CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "third_party/blink/renderer/bindings/core/v8/v8_script_runner.h"

#include "build/build_config.h"
#include "third_party/blink/renderer/bindings/core/v8/binding_security.h"
#include "third_party/blink/renderer/bindings/core/v8/referrer_script_info.h"
#include "third_party/blink/renderer/bindings/core/v8/script_source_code.h"
#include "third_party/blink/renderer/bindings/core/v8/script_streamer.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_binding_for_core.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_code_cache.h"
#include "third_party/blink/renderer/bindings/core/v8/v8_initializer.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/frame/local_dom_window.h"
#include "third_party/blink/renderer/core/frame/local_frame.h"
#include "third_party/blink/renderer/core/inspector/inspector_trace_events.h"
#include "third_party/blink/renderer/core/probe/core_probes.h"
#include "third_party/blink/renderer/platform/bindings/script_forbidden_scope.h"
#include "third_party/blink/renderer/platform/bindings/v8_throw_exception.h"
#include "third_party/blink/renderer/platform/histogram.h"
#include "third_party/blink/renderer/platform/instrumentation/tracing/trace_event.h"
#include "third_party/blink/renderer/platform/loader/fetch/cached_metadata.h"
#include "third_party/blink/renderer/platform/wtf/assertions.h"
#include "third_party/blink/renderer/platform/wtf/text/text_encoding.h"
#include "third_party/blink/renderer/platform/wtf/time.h"

namespace blink {

namespace {

// Used to throw an exception before we exceed the C++ stack and crash.
// This limit was arrived at arbitrarily. crbug.com/449744
const int kMaxRecursionDepth = 44;

// In order to make sure all pending messages to be processed in
// v8::Function::Call, we don't call throwStackOverflowException
// directly. Instead, we create a v8::Function of
// throwStackOverflowException and call it.
void ThrowStackOverflowException(
    const v8::FunctionCallbackInfo<v8::Value>& info) {
  V8ThrowException::ThrowRangeError(info.GetIsolate(),
                                    "Maximum call stack size exceeded.");
}

void ThrowScriptForbiddenException(v8::Isolate* isolate) {
  V8ThrowException::ThrowError(isolate, "Script execution is forbidden.");
}

v8::Local<v8::Value> ThrowStackOverflowExceptionIfNeeded(v8::Isolate* isolate) {
  if (V8PerIsolateData::From(isolate)->IsHandlingRecursionLevelError()) {
    // If we are already handling a recursion level error, we should
    // not invoke v8::Function::Call.
    return v8::Undefined(isolate);
  }
  v8::MicrotasksScope microtasks_scope(
      isolate, v8::MicrotasksScope::kDoNotRunMicrotasks);
  V8PerIsolateData::From(isolate)->SetIsHandlingRecursionLevelError(true);

  ScriptForbiddenScope::AllowUserAgentScript allow_script;
  v8::Local<v8::Value> result =
      v8::Function::New(isolate->GetCurrentContext(),
                        ThrowStackOverflowException, v8::Local<v8::Value>(), 0,
                        v8::ConstructorBehavior::kThrow)
          .ToLocalChecked()
          ->Call(v8::Undefined(isolate), 0, nullptr);

  V8PerIsolateData::From(isolate)->SetIsHandlingRecursionLevelError(false);
  return result;
}

v8::MaybeLocal<v8::Script> CompileScriptInternal(
    v8::Isolate* isolate,
    ExecutionContext* execution_context,
    const ScriptSourceCode& source_code,
    v8::ScriptOrigin origin,
    v8::ScriptCompiler::CompileOptions compile_options,
    v8::ScriptCompiler::NoCacheReason no_cache_reason,
    InspectorCompileScriptEvent::V8CacheResult* cache_result) {
  v8::Local<v8::String> code = V8String(isolate, source_code.Source());

  if (ScriptStreamer* streamer = source_code.Streamer()) {
    // Final compile call for a streamed compilation.
    // Streaming compilation may involve use of code cache.
    // TODO(leszeks): Add compile timer to streaming compilation.
    DCHECK(streamer->IsFinished());
    DCHECK(!streamer->StreamingSuppressed());
    return v8::ScriptCompiler::Compile(isolate->GetCurrentContext(),
                                       streamer->Source(), code, origin);
  }

  // Allow inspector to use its own compilation cache store.
  v8::ScriptCompiler::CachedData* inspector_data = nullptr;
  probe::consumeCompilationCache(execution_context, source_code,
                                 &inspector_data);
  if (inspector_data) {
    v8::ScriptCompiler::Source source(code, origin, inspector_data);
    v8::MaybeLocal<v8::Script> script =
        v8::ScriptCompiler::Compile(isolate->GetCurrentContext(), &source,
                                    v8::ScriptCompiler::kConsumeCodeCache);
    return script;
  }

  switch (compile_options) {
    case v8::ScriptCompiler::kNoCompileOptions:
    case v8::ScriptCompiler::kEagerCompile: {
      v8::ScriptCompiler::Source source(code, origin);
      return v8::ScriptCompiler::Compile(isolate->GetCurrentContext(), &source,
                                         compile_options, no_cache_reason);
    }

    case v8::ScriptCompiler::kConsumeCodeCache: {
      // Compile a script, and consume a V8 cache that was generated previously.
      SingleCachedMetadataHandler* cache_handler = source_code.CacheHandler();
      v8::ScriptCompiler::CachedData* cached_data =
          V8CodeCache::CreateCachedData(cache_handler);
      v8::ScriptCompiler::Source source(code, origin, cached_data);
      v8::MaybeLocal<v8::Script> script =
          v8::ScriptCompiler::Compile(isolate->GetCurrentContext(), &source,
                                      v8::ScriptCompiler::kConsumeCodeCache);

      if (cached_data->rejected) {
        cache_handler->ClearCachedMetadata(
            CachedMetadataHandler::kSendToPlatform);
      }
      if (cache_result) {
        cache_result->consume_result = base::make_optional(
            InspectorCompileScriptEvent::V8CacheResult::ConsumeResult(
                v8::ScriptCompiler::kConsumeCodeCache, cached_data->length,
                cached_data->rejected));
      }
      return script;
    }
    case v8::ScriptCompiler::kProduceCodeCache:
    case v8::ScriptCompiler::kProduceFullCodeCache:
    case v8::ScriptCompiler::kProduceParserCache:
    case v8::ScriptCompiler::kConsumeParserCache:
      NOTREACHED();
      break;
  }

  // All switch branches should return and we should never get here.
  // But some compilers aren't sure, hence this default.
  NOTREACHED();
  return v8::MaybeLocal<v8::Script>();
}

}  // namespace

v8::MaybeLocal<v8::Script> V8ScriptRunner::CompileScript(
    ScriptState* script_state,
    const ScriptSourceCode& source,
    AccessControlStatus access_control_status,
    v8::ScriptCompiler::CompileOptions compile_options,
    v8::ScriptCompiler::NoCacheReason no_cache_reason,
    const ReferrerScriptInfo& referrer_info) {
  v8::Isolate* isolate = script_state->GetIsolate();
  if (source.Source().length() >= v8::String::kMaxLength) {
    V8ThrowException::ThrowError(isolate, "Source file too large.");
    return v8::Local<v8::Script>();
  }

  const String& file_name = source.Url();
  const TextPosition& script_start_position = source.StartPosition();

  constexpr const char* kTraceEventCategoryGroup = "v8,devtools.timeline";
  TRACE_EVENT_BEGIN1(kTraceEventCategoryGroup, "v8.compile", "fileName",
                     file_name.Utf8());
  ExecutionContext* execution_context = ExecutionContext::From(script_state);
  probe::V8Compile probe(execution_context, file_name,
                         script_start_position.line_.ZeroBasedInt(),
                         script_start_position.column_.ZeroBasedInt());

  // NOTE: For compatibility with WebCore, ScriptSourceCode's line starts at
  // 1, whereas v8 starts at 0.
  v8::ScriptOrigin origin(
      V8String(isolate, file_name),
      v8::Integer::New(isolate, script_start_position.line_.ZeroBasedInt()),
      v8::Integer::New(isolate, script_start_position.column_.ZeroBasedInt()),
      v8::Boolean::New(isolate, access_control_status == kSharableCrossOrigin),
      v8::Local<v8::Integer>(), V8String(isolate, source.SourceMapUrl()),
      v8::Boolean::New(isolate, access_control_status == kOpaqueResource),
      v8::False(isolate),  // is_wasm
      v8::False(isolate),  // is_module
      referrer_info.ToV8HostDefinedOptions(isolate));

  if (!*TRACE_EVENT_API_GET_CATEGORY_GROUP_ENABLED(kTraceEventCategoryGroup)) {
    return CompileScriptInternal(isolate, execution_context, source, origin,
                                 compile_options, no_cache_reason, nullptr);
  }

  InspectorCompileScriptEvent::V8CacheResult cache_result;
  v8::MaybeLocal<v8::Script> script =
      CompileScriptInternal(isolate, execution_context, source, origin,
                            compile_options, no_cache_reason, &cache_result);
  TRACE_EVENT_END1(
      kTraceEventCategoryGroup, "v8.compile", "data",
      InspectorCompileScriptEvent::Data(file_name, script_start_position,
                                        cache_result, source.Streamer()));
  return script;
}

v8::MaybeLocal<v8::Module> V8ScriptRunner::CompileModule(
    v8::Isolate* isolate,
    const String& source,
    const String& file_name,
    AccessControlStatus access_control_status,
    const TextPosition& start_position,
    const ReferrerScriptInfo& referrer_info) {
  TRACE_EVENT1("v8,devtools.timeline", "v8.compileModule", "fileName",
               file_name.Utf8());

  v8::ScriptOrigin origin(
      V8String(isolate, file_name),
      v8::Integer::New(isolate, start_position.line_.ZeroBasedInt()),
      v8::Integer::New(isolate, start_position.column_.ZeroBasedInt()),
      v8::Boolean::New(isolate, access_control_status == kSharableCrossOrigin),
      v8::Local<v8::Integer>(),    // script id
      v8::String::Empty(isolate),  // source_map_url
      v8::Boolean::New(isolate, access_control_status == kOpaqueResource),
      v8::False(isolate),  // is_wasm
      v8::True(isolate),   // is_module
      referrer_info.ToV8HostDefinedOptions(isolate));

  v8::ScriptCompiler::Source script_source(V8String(isolate, source), origin);
  return v8::ScriptCompiler::CompileModule(isolate, &script_source);
}

v8::MaybeLocal<v8::Value> V8ScriptRunner::RunCompiledScript(
    v8::Isolate* isolate,
    v8::Local<v8::Script> script,
    ExecutionContext* context) {
  DCHECK(!script.IsEmpty());
  ScopedFrameBlamer frame_blamer(
      context->IsDocument() ? ToDocument(context)->GetFrame() : nullptr);

  v8::Local<v8::Value> script_name =
      script->GetUnboundScript()->GetScriptName();
  TRACE_EVENT1("v8", "v8.run", "fileName",
               TRACE_STR_COPY(*v8::String::Utf8Value(isolate, script_name)));
  RuntimeCallStatsScopedTracer rcs_scoped_tracer(isolate);
  RUNTIME_CALL_TIMER_SCOPE(isolate, RuntimeCallStats::CounterId::kV8);

  if (v8::MicrotasksScope::GetCurrentDepth(isolate) >= kMaxRecursionDepth)
    return ThrowStackOverflowExceptionIfNeeded(isolate);

  CHECK(!context->IsIteratingOverObservers());

  // Run the script and keep track of the current recursion depth.
  v8::MaybeLocal<v8::Value> result;
  {
    if (ScriptForbiddenScope::IsScriptForbidden()) {
      ThrowScriptForbiddenException(isolate);
      return v8::MaybeLocal<v8::Value>();
    }
    v8::Isolate::SafeForTerminationScope safe_for_termination(isolate);
    v8::MicrotasksScope microtasks_scope(isolate,
                                         v8::MicrotasksScope::kRunMicrotasks);
    // ToCoreString here should be zero copy due to externalized string
    // unpacked.
    String script_url = ToCoreString(script_name->ToString(isolate));
    probe::ExecuteScript probe(context, script_url);
    result = script->Run(isolate->GetCurrentContext());
  }

  CHECK(!isolate->IsDead());
  return result;
}

v8::MaybeLocal<v8::Value> V8ScriptRunner::CompileAndRunInternalScript(
    v8::Isolate* isolate,
    ScriptState* script_state,
    const ScriptSourceCode& source_code) {
  DCHECK_EQ(isolate, script_state->GetIsolate());

  v8::ScriptCompiler::CompileOptions compile_options;
  V8CodeCache::ProduceCacheOptions produce_cache_options;
  v8::ScriptCompiler::NoCacheReason no_cache_reason;
  std::tie(compile_options, produce_cache_options, no_cache_reason) =
      V8CodeCache::GetCompileOptions(kV8CacheOptionsDefault, source_code);
  // Currently internal scripts don't have cache handlers. So we should not
  // produce cache for them.
  DCHECK_EQ(produce_cache_options,
            V8CodeCache::ProduceCacheOptions::kNoProduceCache);
  v8::Local<v8::Script> script;
  // Use default ScriptReferrerInfo here:
  // - nonce: empty for internal script, and
  // - parser_state: always "not parser inserted" for internal scripts.
  if (!V8ScriptRunner::CompileScript(script_state, source_code,
                                     kSharableCrossOrigin, compile_options,
                                     no_cache_reason, ReferrerScriptInfo())
           .ToLocal(&script))
    return v8::MaybeLocal<v8::Value>();

  TRACE_EVENT0("v8", "v8.run");
  RuntimeCallStatsScopedTracer rcs_scoped_tracer(isolate);
  RUNTIME_CALL_TIMER_SCOPE(isolate, RuntimeCallStats::CounterId::kV8);
  v8::Isolate::SafeForTerminationScope safe_for_termination(isolate);
  v8::MicrotasksScope microtasks_scope(
      isolate, v8::MicrotasksScope::kDoNotRunMicrotasks);
  v8::MaybeLocal<v8::Value> result = script->Run(isolate->GetCurrentContext());
  CHECK(!isolate->IsDead());
  return result;
}

v8::MaybeLocal<v8::Value> V8ScriptRunner::CallAsConstructor(
    v8::Isolate* isolate,
    v8::Local<v8::Object> constructor,
    ExecutionContext* context,
    int argc,
    v8::Local<v8::Value> argv[]) {
  TRACE_EVENT0("v8", "v8.callAsConstructor");
  RUNTIME_CALL_TIMER_SCOPE(isolate, RuntimeCallStats::CounterId::kV8);

  int depth = v8::MicrotasksScope::GetCurrentDepth(isolate);
  if (depth >= kMaxRecursionDepth)
    return v8::MaybeLocal<v8::Value>(
        ThrowStackOverflowExceptionIfNeeded(isolate));

  CHECK(!context->IsIteratingOverObservers());

  if (ScriptForbiddenScope::IsScriptForbidden()) {
    ThrowScriptForbiddenException(isolate);
    return v8::MaybeLocal<v8::Value>();
  }

  // TODO(dominicc): When inspector supports tracing object
  // invocation, change this to use v8::Object instead of
  // v8::Function. All callers use functions because
  // CustomElementRegistry#define's IDL signature is Function.
  CHECK(constructor->IsFunction());
  v8::Local<v8::Function> function = constructor.As<v8::Function>();

  v8::Isolate::SafeForTerminationScope safe_for_termination(isolate);
  v8::MicrotasksScope microtasks_scope(isolate,
                                       v8::MicrotasksScope::kRunMicrotasks);
  probe::CallFunction probe(context, function, depth);
  v8::MaybeLocal<v8::Value> result =
      constructor->CallAsConstructor(isolate->GetCurrentContext(), argc, argv);
  CHECK(!isolate->IsDead());
  return result;
}

v8::MaybeLocal<v8::Value> V8ScriptRunner::CallFunction(
    v8::Local<v8::Function> function,
    ExecutionContext* context,
    v8::Local<v8::Value> receiver,
    int argc,
    v8::Local<v8::Value> args[],
    v8::Isolate* isolate) {
  LocalFrame* frame =
      context->IsDocument() ? ToDocument(context)->GetFrame() : nullptr;
  ScopedFrameBlamer frame_blamer(frame);
  TRACE_EVENT0("v8", "v8.callFunction");
  RuntimeCallStatsScopedTracer rcs_scoped_tracer(isolate);
  RUNTIME_CALL_TIMER_SCOPE(isolate, RuntimeCallStats::CounterId::kV8);

  int depth = v8::MicrotasksScope::GetCurrentDepth(isolate);
  if (depth >= kMaxRecursionDepth)
    return v8::MaybeLocal<v8::Value>(
        ThrowStackOverflowExceptionIfNeeded(isolate));

  CHECK(!context->IsIteratingOverObservers());

  if (ScriptForbiddenScope::IsScriptForbidden()) {
    ThrowScriptForbiddenException(isolate);
    return v8::MaybeLocal<v8::Value>();
  }

  DCHECK(!frame || BindingSecurity::ShouldAllowAccessToFrame(
                       ToLocalDOMWindow(function->CreationContext()), frame,
                       BindingSecurity::ErrorReportOption::kDoNotReport));
  CHECK(!ThreadState::Current()->IsWrapperTracingForbidden());
  v8::Isolate::SafeForTerminationScope safe_for_termination(isolate);
  v8::MicrotasksScope microtasks_scope(isolate,
                                       v8::MicrotasksScope::kRunMicrotasks);
  probe::CallFunction probe(context, function, depth);
  v8::MaybeLocal<v8::Value> result =
      function->Call(isolate->GetCurrentContext(), receiver, argc, args);
  CHECK(!isolate->IsDead());

  return result;
}

v8::MaybeLocal<v8::Value> V8ScriptRunner::CallInternalFunction(
    v8::Isolate* isolate,
    v8::Local<v8::Function> function,
    v8::Local<v8::Value> receiver,
    int argc,
    v8::Local<v8::Value> args[]) {
  TRACE_EVENT0("v8", "v8.callFunction");
  RuntimeCallStatsScopedTracer rcs_scoped_tracer(isolate);
  RUNTIME_CALL_TIMER_SCOPE(isolate, RuntimeCallStats::CounterId::kV8);

  CHECK(!ThreadState::Current()->IsWrapperTracingForbidden());
  v8::Isolate::SafeForTerminationScope safe_for_termination(isolate);
  v8::MicrotasksScope microtasks_scope(
      isolate, v8::MicrotasksScope::kDoNotRunMicrotasks);
  v8::MaybeLocal<v8::Value> result =
      function->Call(isolate->GetCurrentContext(), receiver, argc, args);
  CHECK(!isolate->IsDead());
  return result;
}

v8::MaybeLocal<v8::Value> V8ScriptRunner::EvaluateModule(
    v8::Isolate* isolate,
    v8::Local<v8::Module> module,
    v8::Local<v8::Context> context) {
  TRACE_EVENT0("v8,devtools.timeline", "v8.evaluateModule");
  RUNTIME_CALL_TIMER_SCOPE(isolate, RuntimeCallStats::CounterId::kV8);
  v8::Isolate::SafeForTerminationScope safe_for_termination(isolate);
  v8::MicrotasksScope microtasks_scope(isolate,
                                       v8::MicrotasksScope::kRunMicrotasks);
  return module->Evaluate(context);
}

void V8ScriptRunner::ReportException(v8::Isolate* isolate,
                                     v8::Local<v8::Value> exception) {
  DCHECK(!exception.IsEmpty());

  // https://html.spec.whatwg.org/multipage/webappapis.html#report-the-error
  v8::Local<v8::Message> message =
      v8::Exception::CreateMessage(isolate, exception);
  if (IsMainThread())
    V8Initializer::MessageHandlerInMainThread(message, exception);
  else
    V8Initializer::MessageHandlerInWorker(message, exception);
}

v8::MaybeLocal<v8::Value> V8ScriptRunner::CallExtraHelper(
    ScriptState* script_state,
    const char* name,
    size_t num_args,
    v8::Local<v8::Value>* args) {
  v8::Isolate* isolate = script_state->GetIsolate();
  v8::Local<v8::Value> function_value;
  v8::Local<v8::Context> context = script_state->GetContext();
  if (!context->GetExtrasBindingObject()
           ->Get(context, V8AtomicString(isolate, name))
           .ToLocal(&function_value))
    return v8::MaybeLocal<v8::Value>();
  v8::Local<v8::Function> function = function_value.As<v8::Function>();
  return V8ScriptRunner::CallInternalFunction(
      isolate, function, v8::Undefined(isolate), num_args, args);
}

}  // namespace blink
