// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_VIDEO_FRAME_SUBMITTER_H_
#define THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_VIDEO_FRAME_SUBMITTER_H_

#include <memory>
#include <utility>

#include "base/memory/weak_ptr.h"
#include "base/threading/thread_checker.h"
#include "components/viz/client/shared_bitmap_reporter.h"
#include "components/viz/common/gpu/context_provider.h"
#include "components/viz/common/resources/shared_bitmap.h"
#include "mojo/public/cpp/bindings/binding.h"
#include "mojo/public/cpp/system/buffer.h"
#include "services/viz/public/interfaces/compositing/compositor_frame_sink.mojom-blink.h"
#include "third_party/blink/public/platform/web_video_frame_submitter.h"
#include "third_party/blink/renderer/platform/graphics/video_frame_resource_provider.h"
#include "third_party/blink/renderer/platform/platform_export.h"
#include "third_party/blink/renderer/platform/wtf/functional.h"

namespace blink {

// This single-threaded class facilitates the communication between the media
// stack and mojo, providing compositor frames containing video frames to the
// |compositor_frame_sink_|. This class has dependencies on classes that use
// the media thread's OpenGL ContextProvider, and thus, besides construction,
// should be consistently ran from the same media SingleThreadTaskRunner.
class PLATFORM_EXPORT VideoFrameSubmitter
    : public WebVideoFrameSubmitter,
      public viz::ContextLostObserver,
      public viz::SharedBitmapReporter,
      public viz::mojom::blink::CompositorFrameSinkClient {
 public:
  explicit VideoFrameSubmitter(WebContextProviderCallback,
                               std::unique_ptr<VideoFrameResourceProvider>);

  ~VideoFrameSubmitter() override;

  bool Rendering() { return is_rendering_; }
  cc::VideoFrameProvider* Provider() { return provider_; }
  mojo::Binding<viz::mojom::blink::CompositorFrameSinkClient>* Binding() {
    return &binding_;
  }
  void SetSink(viz::mojom::blink::CompositorFrameSinkPtr* sink) {
    compositor_frame_sink_ = std::move(*sink);
  }
  void SetSurfaceId(viz::SurfaceId id) { surface_id_ = id; }

  void OnReceivedContextProvider(
      bool,
      scoped_refptr<ui::ContextProviderCommandBuffer>);

  // cc::VideoFrameProvider::Client implementation.
  void StopUsingProvider() override;
  void StartRendering() override;
  void StopRendering() override;
  void DidReceiveFrame() override;

  // WebVideoFrameSubmitter implementation.
  void Initialize(cc::VideoFrameProvider*) override;
  void SetRotation(media::VideoRotation) override;
  void EnableSubmission(viz::SurfaceId, WebFrameSinkDestroyedCallback) override;
  void UpdateSubmissionState(bool) override;
  void SetForceSubmit(bool) override;

  // viz::ContextLostObserver implementation.
  void OnContextLost() override;

  // cc::mojom::CompositorFrameSinkClient implementation.
  void DidReceiveCompositorFrameAck(
      const WTF::Vector<viz::ReturnedResource>& resources) override;
  void DidPresentCompositorFrame(
      uint32_t presentation_token,
      ::gfx::mojom::blink::PresentationFeedbackPtr feedback) final;
  void OnBeginFrame(const viz::BeginFrameArgs&) override;
  void OnBeginFramePausedChanged(bool paused) override {}
  void ReclaimResources(
      const WTF::Vector<viz::ReturnedResource>& resources) override;

  // viz::SharedBitmapReporter implementation.
  void DidAllocateSharedBitmap(mojo::ScopedSharedBufferHandle,
                               const viz::SharedBitmapId&) override;
  void DidDeleteSharedBitmap(const viz::SharedBitmapId&) override;

 private:
  FRIEND_TEST_ALL_PREFIXES(VideoFrameSubmitterTest, ContextLostDuringSubmit);
  FRIEND_TEST_ALL_PREFIXES(VideoFrameSubmitterTest,
                           ShouldSubmitPreventsSubmission);
  FRIEND_TEST_ALL_PREFIXES(VideoFrameSubmitterTest,
                           SetForceSubmitForcesSubmission);

  void StartSubmitting();
  void UpdateSubmissionStateInternal();
  void SubmitFrame(const viz::BeginFrameAck&, scoped_refptr<media::VideoFrame>);
  void SubmitEmptyFrame();

  // Pulls frame and submits it to compositor.
  // Used in cases like PaintSingleFrame, which occurs before video rendering
  // has started to post a poster image, or to submit a final frame before
  // ending rendering.
  void SubmitSingleFrame();

  // Return whether the submitter should submit frames based on its current
  // state.
  bool ShouldSubmit() const;

  cc::VideoFrameProvider* provider_ = nullptr;
  scoped_refptr<ui::ContextProviderCommandBuffer> context_provider_;
  viz::mojom::blink::CompositorFrameSinkPtr compositor_frame_sink_;
  mojo::Binding<viz::mojom::blink::CompositorFrameSinkClient> binding_;
  WebContextProviderCallback context_provider_callback_;
  std::unique_ptr<VideoFrameResourceProvider> resource_provider_;
  WebFrameSinkDestroyedCallback frame_sink_destroyed_callback_;
  viz::SurfaceId surface_id_;
  bool waiting_for_compositor_ack_ = false;

  bool is_rendering_;
  // If we are not on screen, we should not submit.
  bool should_submit_internal_ = false;
  // Whether frames should always be submitted.
  bool force_submit_ = false;
  media::VideoRotation rotation_;

  // Size of the video frame being submitted. It is set the first time a frame
  // is submitted and is expected to never change aftewards. Used to send an
  // empty frame when the video is out of view.
  gfx::Rect frame_size_;

  THREAD_CHECKER(media_thread_checker_);

  base::WeakPtrFactory<VideoFrameSubmitter> weak_ptr_factory_;

  DISALLOW_COPY_AND_ASSIGN(VideoFrameSubmitter);
};

}  // namespace blink

#endif  // THIRD_PARTY_BLINK_RENDERER_PLATFORM_GRAPHICS_VIDEO_FRAME_SUBMITTER_H_
