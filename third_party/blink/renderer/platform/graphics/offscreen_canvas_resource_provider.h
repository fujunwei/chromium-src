// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_OFFSCREEN_CANVAS_RESOURCE_PROVIDER_H_
#define THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_OFFSCREEN_CANVAS_RESOURCE_PROVIDER_H_

#include "components/viz/common/resources/returned_resource.h"
#include "components/viz/common/resources/transferable_resource.h"
#include "third_party/blink/renderer/platform/graphics/static_bitmap_image.h"

namespace viz {
class SingleReleaseCallback;
namespace mojom {
namespace blink {
class CompositorFrameSink;
}
}  // namespace mojom
}  // namespace viz

namespace blink {

class CanvasResource;
class CanvasResourceDispatcher;

class PLATFORM_EXPORT OffscreenCanvasResourceProvider {
 public:
  // The CompositorFrameSink given here must be kept alive as long as this
  // class is, as it is used to free the software-backed resources in the
  // display compositor.
  OffscreenCanvasResourceProvider(int width,
                                  int height,
                                  CanvasResourceDispatcher*);

  ~OffscreenCanvasResourceProvider();

  void SetTransferableResource(viz::TransferableResource* out_resource,
                               scoped_refptr<CanvasResource>);

  void ReclaimResource(unsigned resource_id);
  void ReclaimResources(const WTF::Vector<viz::ReturnedResource>& resources);
  void IncNextResourceId() { next_resource_id_++; }
  unsigned GetNextResourceId() { return next_resource_id_; }

  void Reshape(int width, int height) {
    width_ = width;
    height_ = height;
    // TODO(junov): Prevent recycling resources of the wrong size.
  }

 private:
  struct FrameResource {
    FrameResource() = default;
    ~FrameResource();

    // TODO(junov):  What does this do?
    bool spare_lock = true;

    // Back-pointer to the OffscreenCanvasResourceProvider. FrameResource does
    // not outlive the provider.
    OffscreenCanvasResourceProvider* provider = nullptr;
    std::unique_ptr<viz::SingleReleaseCallback> release_callback;
    gpu::SyncToken sync_token;
    bool is_lost = false;
  };

  using ResourceMap = HashMap<unsigned, std::unique_ptr<FrameResource>>;

  void SetNeedsBeginFrameInternal();
  std::unique_ptr<FrameResource> CreateOrRecycleFrameResource();
  void ReclaimResourceInternal(const ResourceMap::iterator&);

  CanvasResourceDispatcher* frame_dispatcher_;
  int width_;
  int height_;
  unsigned next_resource_id_ = 0;
  std::unique_ptr<FrameResource> recyclable_resource_;
  ResourceMap resources_;
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_OFFSCREEN_CANVAS_RESOURCE_PROVIDER_H_
