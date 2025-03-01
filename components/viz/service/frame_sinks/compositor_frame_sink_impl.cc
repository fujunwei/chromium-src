// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/viz/service/frame_sinks/compositor_frame_sink_impl.h"

#include <utility>

#include "components/viz/service/frame_sinks/frame_sink_manager_impl.h"

namespace viz {

CompositorFrameSinkImpl::CompositorFrameSinkImpl(
    FrameSinkManagerImpl* frame_sink_manager,
    const FrameSinkId& frame_sink_id,
    mojom::CompositorFrameSinkRequest request,
    mojom::CompositorFrameSinkClientPtr client)
    : compositor_frame_sink_client_(std::move(client)),
      compositor_frame_sink_binding_(this, std::move(request)),
      support_(std::make_unique<CompositorFrameSinkSupport>(
          compositor_frame_sink_client_.get(),
          frame_sink_manager,
          frame_sink_id,
          false /* is_root */,
          true /* needs_sync_points */)) {
  compositor_frame_sink_binding_.set_connection_error_handler(
      base::Bind(&CompositorFrameSinkImpl::OnClientConnectionLost,
                 base::Unretained(this)));
}

CompositorFrameSinkImpl::~CompositorFrameSinkImpl() = default;

void CompositorFrameSinkImpl::SetNeedsBeginFrame(bool needs_begin_frame) {
  support_->SetNeedsBeginFrame(needs_begin_frame);
}

void CompositorFrameSinkImpl::SetWantsAnimateOnlyBeginFrames() {
  support_->SetWantsAnimateOnlyBeginFrames();
}

void CompositorFrameSinkImpl::SubmitCompositorFrame(
    const LocalSurfaceId& local_surface_id,
    CompositorFrame frame,
    base::Optional<HitTestRegionList> hit_test_region_list,
    uint64_t submit_time) {
  SubmitCompositorFrameInternal(local_surface_id, std::move(frame),
                                std::move(hit_test_region_list), submit_time,
                                SubmitCompositorFrameSyncCallback());
}

void CompositorFrameSinkImpl::SubmitCompositorFrameSync(
    const LocalSurfaceId& local_surface_id,
    CompositorFrame frame,
    base::Optional<HitTestRegionList> hit_test_region_list,
    uint64_t submit_time,
    SubmitCompositorFrameSyncCallback callback) {
  SubmitCompositorFrameInternal(local_surface_id, std::move(frame),
                                std::move(hit_test_region_list), submit_time,
                                std::move(callback));
}

void CompositorFrameSinkImpl::SubmitCompositorFrameInternal(
    const LocalSurfaceId& local_surface_id,
    CompositorFrame frame,
    base::Optional<HitTestRegionList> hit_test_region_list,
    uint64_t submit_time,
    mojom::CompositorFrameSink::SubmitCompositorFrameSyncCallback callback) {
  const auto result = support_->MaybeSubmitCompositorFrame(
      local_surface_id, std::move(frame), std::move(hit_test_region_list),
      submit_time, std::move(callback));
  if (result == CompositorFrameSinkSupport::ACCEPTED)
    return;

  const char* reason =
      CompositorFrameSinkSupport::GetSubmitResultAsString(result);
  DLOG(ERROR) << "SubmitCompositorFrame failed for " << local_surface_id
              << " because " << reason;
  compositor_frame_sink_binding_.CloseWithReason(static_cast<uint32_t>(result),
                                                 reason);
  OnClientConnectionLost();
}

void CompositorFrameSinkImpl::DidNotProduceFrame(
    const BeginFrameAck& begin_frame_ack) {
  support_->DidNotProduceFrame(begin_frame_ack);
}

void CompositorFrameSinkImpl::DidAllocateSharedBitmap(
    mojo::ScopedSharedBufferHandle buffer,
    const SharedBitmapId& id) {
  if (!support_->DidAllocateSharedBitmap(std::move(buffer), id)) {
    DLOG(ERROR) << "DidAllocateSharedBitmap failed for duplicate "
                << "SharedBitmapId";
    compositor_frame_sink_binding_.Close();
    OnClientConnectionLost();
  }
}

void CompositorFrameSinkImpl::DidDeleteSharedBitmap(const SharedBitmapId& id) {
  support_->DidDeleteSharedBitmap(id);
}

void CompositorFrameSinkImpl::OnClientConnectionLost() {
  // The client that owns this CompositorFrameSink is either shutting down or
  // has done something invalid and the connection to the client was terminated.
  // Destroy |this| to free up resources as it's no longer useful.
  FrameSinkId frame_sink_id = support_->frame_sink_id();
  support_->frame_sink_manager()->DestroyCompositorFrameSink(frame_sink_id,
                                                             base::DoNothing());
}

}  // namespace viz
