// Copyright 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "gin/arguments.h"

#include "base/strings/stringprintf.h"
#include "gin/converter.h"

namespace gin {

Arguments::Arguments()
    : isolate_(NULL),
      info_(NULL),
      next_(0),
      insufficient_arguments_(false) {
}

Arguments::Arguments(const v8::FunctionCallbackInfo<v8::Value>& info)
    : isolate_(info.GetIsolate()),
      info_(&info),
      next_(0),
      insufficient_arguments_(false) {
}

Arguments::~Arguments() = default;

v8::Local<v8::Value> Arguments::PeekNext() const {
  if (next_ >= info_->Length())
    return v8::Local<v8::Value>();
  return (*info_)[next_];
}

std::vector<v8::Local<v8::Value>> Arguments::GetAll() const {
  std::vector<v8::Local<v8::Value>> result;
  int length = info_->Length();
  if (length == 0)
    return result;

  result.reserve(length);
  for (int i = 0; i < length; ++i)
    result.push_back((*info_)[i]);

  return result;
}

v8::Local<v8::Context> Arguments::GetHolderCreationContext() {
  return info_->Holder()->CreationContext();
}

std::string V8TypeAsString(v8::Isolate* isolate, v8::Local<v8::Value> value) {
  if (value.IsEmpty())
    return "<empty handle>";
  if (value->IsUndefined())
    return "undefined";
  if (value->IsNull())
    return "null";
  std::string result;
  if (!ConvertFromV8(isolate, value, &result))
    return std::string();
  return result;
}

void Arguments::ThrowError() const {
  if (insufficient_arguments_)
    return ThrowTypeError("Insufficient number of arguments.");

  return ThrowTypeError(base::StringPrintf(
      "Error processing argument at index %d, conversion failure from %s",
      next_ - 1, V8TypeAsString(isolate_, (*info_)[next_ - 1]).c_str()));
}

void Arguments::ThrowTypeError(const std::string& message) const {
  isolate_->ThrowException(v8::Exception::TypeError(
      StringToV8(isolate_, message)));
}

bool Arguments::IsConstructCall() const {
  return info_->IsConstructCall();
}

}  // namespace gin
