// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LiICENSE file.

#include "third_party/blink/renderer/modules/background_fetch/background_fetch_icon_loader.h"

#include "skia/ext/image_operations.h"
#include "third_party/blink/public/common/manifest/manifest_icon_selector.h"
#include "third_party/blink/public/platform/web_size.h"
#include "third_party/blink/public/platform/web_url_request.h"
#include "third_party/blink/renderer/core/execution_context/execution_context.h"
#include "third_party/blink/renderer/core/loader/threadable_loader.h"
#include "third_party/blink/renderer/modules/background_fetch/background_fetch_bridge.h"
#include "third_party/blink/renderer/modules/manifest/image_resource.h"
#include "third_party/blink/renderer/modules/manifest/image_resource_type_converters.h"
#include "third_party/blink/renderer/platform/graphics/color_behavior.h"
#include "third_party/blink/renderer/platform/heap/heap_allocator.h"
#include "third_party/blink/renderer/platform/image-decoders/image_decoder.h"
#include "third_party/blink/renderer/platform/image-decoders/image_frame.h"
#include "third_party/blink/renderer/platform/loader/fetch/resource_loader_options.h"
#include "third_party/blink/renderer/platform/loader/fetch/resource_request.h"
#include "third_party/blink/renderer/platform/weborigin/kurl.h"
#include "third_party/blink/renderer/platform/wtf/text/string_impl.h"
#include "third_party/blink/renderer/platform/wtf/text/wtf_string.h"
#include "third_party/blink/renderer/platform/wtf/threading.h"

namespace blink {

namespace {

const unsigned long kIconFetchTimeoutInMs = 30000;
const int kMinimumIconSizeInPx = 0;

}  // namespace

BackgroundFetchIconLoader::BackgroundFetchIconLoader() = default;
BackgroundFetchIconLoader::~BackgroundFetchIconLoader() {
  // We should've called Stop() before the destructor is invoked.
  DCHECK(stopped_ || icon_callback_.is_null());
}

void BackgroundFetchIconLoader::Start(BackgroundFetchBridge* bridge,
                                      ExecutionContext* execution_context,
                                      HeapVector<ManifestImageResource> icons,
                                      IconCallback icon_callback) {
  DCHECK(!stopped_);
  DCHECK_GE(icons.size(), 1u);
  DCHECK(bridge);

  icons_ = std::move(icons);
  bridge->GetIconDisplaySize(
      WTF::Bind(&BackgroundFetchIconLoader::DidGetIconDisplaySizeIfSoLoadIcon,
                WrapWeakPersistent(this), WrapWeakPersistent(execution_context),
                std::move(icon_callback)));
}

void BackgroundFetchIconLoader::DidGetIconDisplaySizeIfSoLoadIcon(
    ExecutionContext* execution_context,
    IconCallback icon_callback,
    const WebSize& icon_display_size_pixels) {
  if (icon_display_size_pixels.IsEmpty()) {
    std::move(icon_callback).Run(SkBitmap());
    return;
  }

  KURL best_icon_url =
      PickBestIconForDisplay(execution_context, icon_display_size_pixels);
  if (best_icon_url.IsEmpty()) {
    // None of the icons provided was suitable.
    std::move(icon_callback).Run(SkBitmap());
    return;
  }

  icon_callback_ = std::move(icon_callback);

  ThreadableLoaderOptions threadable_loader_options;
  threadable_loader_options.timeout_milliseconds = kIconFetchTimeoutInMs;

  ResourceLoaderOptions resource_loader_options;
  if (execution_context->IsWorkerGlobalScope())
    resource_loader_options.request_initiator_context = kWorkerContext;

  ResourceRequest resource_request(best_icon_url);
  resource_request.SetRequestContext(WebURLRequest::kRequestContextImage);
  resource_request.SetPriority(ResourceLoadPriority::kMedium);
  resource_request.SetRequestorOrigin(execution_context->GetSecurityOrigin());

  threadable_loader_ = ThreadableLoader::Create(*execution_context, this,
                                                threadable_loader_options,
                                                resource_loader_options);

  threadable_loader_->Start(resource_request);
}

KURL BackgroundFetchIconLoader::PickBestIconForDisplay(
    ExecutionContext* execution_context,
    const WebSize& icon_display_size_pixels) {
  std::vector<Manifest::ImageResource> icons;
  for (const auto& icon : icons_)
    icons.emplace_back(blink::ConvertManifestImageResource(icon));

  return KURL(ManifestIconSelector::FindBestMatchingIcon(
      std::move(icons), icon_display_size_pixels.height, kMinimumIconSizeInPx,
      Manifest::ImageResource::Purpose::ANY));
}

void BackgroundFetchIconLoader::Stop() {
  if (stopped_)
    return;

  stopped_ = true;
  if (threadable_loader_) {
    threadable_loader_->Cancel();
    threadable_loader_ = nullptr;
  }
}

void BackgroundFetchIconLoader::DidReceiveData(const char* data,
                                               unsigned length) {
  if (!data_)
    data_ = SharedBuffer::Create();
  data_->Append(data, length);
}

void BackgroundFetchIconLoader::DidFinishLoading(
    unsigned long resource_identifier) {
  if (stopped_)
    return;
  if (data_) {
    // Decode data.
    const bool data_complete = true;
    std::unique_ptr<ImageDecoder> decoder = ImageDecoder::Create(
        data_, data_complete, ImageDecoder::kAlphaPremultiplied,
        ImageDecoder::kDefaultBitDepth, ColorBehavior::TransformToSRGB());
    if (decoder) {
      // the |ImageFrame*| is owned by the decoder.
      ImageFrame* image_frame = decoder->DecodeFrameBufferAtIndex(0);
      if (image_frame) {
        std::move(icon_callback_).Run(image_frame->Bitmap());
        return;
      }
    }
  }
  RunCallbackWithEmptyBitmap();
}

void BackgroundFetchIconLoader::DidFail(const ResourceError& error) {
  RunCallbackWithEmptyBitmap();
}

void BackgroundFetchIconLoader::DidFailRedirectCheck() {
  RunCallbackWithEmptyBitmap();
}

void BackgroundFetchIconLoader::RunCallbackWithEmptyBitmap() {
  // If this has been stopped it is not desirable to trigger further work,
  // there is a shutdown of some sort in progress.
  if (stopped_)
    return;

  std::move(icon_callback_).Run(SkBitmap());
}

}  // namespace blink
