// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/viz/service/display_embedder/gpu_display_provider.h"

#include <utility>

#include "base/command_line.h"
#include "base/compiler_specific.h"
#include "base/threading/thread_task_runner_handle.h"
#include "cc/base/switches.h"
#include "components/viz/common/display/renderer_settings.h"
#include "components/viz/common/frame_sinks/begin_frame_source.h"
#include "components/viz/service/display/display.h"
#include "components/viz/service/display/display_scheduler.h"
#include "components/viz/service/display_embedder/gl_output_surface.h"
#include "components/viz/service/display_embedder/in_process_gpu_memory_buffer_manager.h"
#include "components/viz/service/display_embedder/server_shared_bitmap_manager.h"
#include "components/viz/service/display_embedder/skia_output_surface_impl.h"
#include "components/viz/service/display_embedder/software_output_surface.h"
#include "components/viz/service/display_embedder/viz_process_context_provider.h"
#include "components/viz/service/gl/gpu_service_impl.h"
#include "gpu/command_buffer/client/shared_memory_limits.h"
#include "gpu/command_buffer/service/image_factory.h"
#include "gpu/ipc/command_buffer_task_executor.h"
#include "gpu/ipc/common/surface_handle.h"
#include "gpu/ipc/service/gpu_channel_manager.h"
#include "gpu/ipc/service/gpu_channel_manager_delegate.h"
#include "gpu/ipc/service/gpu_memory_buffer_factory.h"
#include "ui/base/ui_base_switches.h"

#if defined(OS_WIN)
#include "components/viz/service/display_embedder/gl_output_surface_win.h"
#include "components/viz/service/display_embedder/software_output_device_win.h"
#endif

#if defined(OS_ANDROID)
#include "components/viz/service/display_embedder/gl_output_surface_android.h"
#endif

#if defined(OS_MACOSX)
#include "components/viz/service/display_embedder/gl_output_surface_mac.h"
#include "components/viz/service/display_embedder/software_output_device_mac.h"
#include "ui/base/cocoa/remote_layer_api.h"
#endif

#if defined(USE_X11)
#include "components/viz/service/display_embedder/software_output_device_x11.h"
#endif

#if defined(USE_OZONE)
#include "components/viz/service/display_embedder/gl_output_surface_ozone.h"
#include "components/viz/service/display_embedder/software_output_device_ozone.h"
#include "gpu/command_buffer/client/gles2_interface.h"
#include "ui/ozone/public/ozone_platform.h"
#include "ui/ozone/public/surface_factory_ozone.h"
#include "ui/ozone/public/surface_ozone_canvas.h"
#endif

namespace {

gpu::ImageFactory* GetImageFactory(gpu::GpuChannelManager* channel_manager) {
  auto* buffer_factory = channel_manager->gpu_memory_buffer_factory();
  return buffer_factory ? buffer_factory->AsImageFactory() : nullptr;
}

}  // namespace

namespace viz {

GpuDisplayProvider::GpuDisplayProvider(
    uint32_t restart_id,
    GpuServiceImpl* gpu_service_impl,
    scoped_refptr<gpu::CommandBufferTaskExecutor> task_executor,
    gpu::GpuChannelManager* gpu_channel_manager,
    ServerSharedBitmapManager* server_shared_bitmap_manager,
    bool headless,
    bool wait_for_all_pipeline_stages_before_draw)
    : restart_id_(restart_id),
      gpu_service_impl_(gpu_service_impl),
      task_executor_(std::move(task_executor)),
      gpu_channel_manager_delegate_(gpu_channel_manager->delegate()),
      gpu_memory_buffer_manager_(
          std::make_unique<InProcessGpuMemoryBufferManager>(
              gpu_channel_manager)),
      image_factory_(GetImageFactory(gpu_channel_manager)),
      server_shared_bitmap_manager_(server_shared_bitmap_manager),
      task_runner_(base::ThreadTaskRunnerHandle::Get()),
      headless_(headless),
      wait_for_all_pipeline_stages_before_draw_(
          wait_for_all_pipeline_stages_before_draw) {
  DCHECK_NE(restart_id_, BeginFrameSource::kNotRestartableId);
}

GpuDisplayProvider::~GpuDisplayProvider() = default;

std::unique_ptr<Display> GpuDisplayProvider::CreateDisplay(
    const FrameSinkId& frame_sink_id,
    gpu::SurfaceHandle surface_handle,
    bool gpu_compositing,
    mojom::DisplayClient* display_client,
    ExternalBeginFrameSource* external_begin_frame_source,
    SyntheticBeginFrameSource* synthetic_begin_frame_source,
    const RendererSettings& renderer_settings,
    bool send_swap_size_notifications) {
  BeginFrameSource* begin_frame_source =
      synthetic_begin_frame_source
          ? static_cast<BeginFrameSource*>(synthetic_begin_frame_source)
          : static_cast<BeginFrameSource*>(external_begin_frame_source);

  // TODO(penghuang): Merge two output surfaces into one when GLRenderer and
  // software compositor is removed.
  std::unique_ptr<OutputSurface> output_surface;
  SkiaOutputSurface* skia_output_surface = nullptr;

  if (!gpu_compositing) {
    output_surface = std::make_unique<SoftwareOutputSurface>(
        CreateSoftwareOutputDeviceForPlatform(surface_handle, display_client));
  } else if (renderer_settings.use_skia_deferred_display_list) {
#if defined(OS_MACOSX) || defined(OS_WIN)
    // TODO(penghuang): Support DDL for all platforms.
    NOTIMPLEMENTED();
    return nullptr;
#else
    output_surface = std::make_unique<SkiaOutputSurfaceImpl>(
        gpu_service_impl_, surface_handle, synthetic_begin_frame_source);
    skia_output_surface = static_cast<SkiaOutputSurface*>(output_surface.get());
#endif
  } else {
    scoped_refptr<VizProcessContextProvider> context_provider;

    // Retry creating and binding |context_provider| on transient failures.
    gpu::ContextResult context_result = gpu::ContextResult::kTransientFailure;
    while (context_result != gpu::ContextResult::kSuccess) {
      context_provider = base::MakeRefCounted<VizProcessContextProvider>(
          task_executor_, surface_handle, gpu_memory_buffer_manager_.get(),
          image_factory_, gpu_channel_manager_delegate_,
          gpu::SharedMemoryLimits());
      context_result = context_provider->BindToCurrentThread();

      if (context_result == gpu::ContextResult::kFatalFailure) {
        gpu_service_impl_->DisableGpuCompositing();
        return nullptr;
      }
    }

    if (context_provider->ContextCapabilities().surfaceless) {
#if defined(USE_OZONE)
      output_surface = std::make_unique<GLOutputSurfaceOzone>(
          std::move(context_provider), surface_handle,
          synthetic_begin_frame_source, gpu_memory_buffer_manager_.get(),
          GL_TEXTURE_2D, GL_BGRA_EXT);
#elif defined(OS_MACOSX)
      output_surface = std::make_unique<GLOutputSurfaceMac>(
          std::move(context_provider), surface_handle,
          synthetic_begin_frame_source, gpu_memory_buffer_manager_.get(),
          renderer_settings.allow_overlays);
#else
      NOTREACHED();
#endif
    } else {
#if defined(OS_WIN)
      const auto& capabilities = context_provider->ContextCapabilities();
      const bool use_overlays =
          capabilities.dc_layers && capabilities.use_dc_overlays_for_video;
      output_surface = std::make_unique<GLOutputSurfaceWin>(
          std::move(context_provider), synthetic_begin_frame_source,
          use_overlays);
#elif defined(OS_ANDROID)
      output_surface = std::make_unique<GLOutputSurfaceAndroid>(
          std::move(context_provider), synthetic_begin_frame_source);
#else
      output_surface = std::make_unique<GLOutputSurface>(
          std::move(context_provider), synthetic_begin_frame_source);
#endif
    }
  }

  // If we need swap size notifications tell the output surface now.
  output_surface->SetNeedsSwapSizeNotifications(send_swap_size_notifications);

  int max_frames_pending = output_surface->capabilities().max_frames_pending;
  DCHECK_GT(max_frames_pending, 0);

  auto scheduler = std::make_unique<DisplayScheduler>(
      begin_frame_source, task_runner_.get(), max_frames_pending,
      wait_for_all_pipeline_stages_before_draw_);

  return std::make_unique<Display>(
      server_shared_bitmap_manager_, renderer_settings, frame_sink_id,
      std::move(output_surface), std::move(scheduler), task_runner_,
      skia_output_surface);
}

uint32_t GpuDisplayProvider::GetRestartId() const {
  return restart_id_;
}

std::unique_ptr<SoftwareOutputDevice>
GpuDisplayProvider::CreateSoftwareOutputDeviceForPlatform(
    gpu::SurfaceHandle surface_handle,
    mojom::DisplayClient* display_client) {
  if (headless_)
    return std::make_unique<SoftwareOutputDevice>();

#if defined(OS_WIN)
  HWND child_hwnd;
  auto device = CreateSoftwareOutputDeviceWinGpu(
      surface_handle, &output_device_backing_, display_client, &child_hwnd);

  // If |child_hwnd| isn't null then a new child HWND was created. Send an IPC
  // to browser process for SetParent() syscall.
  if (child_hwnd) {
    gpu_channel_manager_delegate_->SendCreatedChildWindow(surface_handle,
                                                          child_hwnd);
  }

  return device;
#elif defined(OS_MACOSX)
  return std::make_unique<SoftwareOutputDeviceMac>(task_runner_);
#elif defined(OS_ANDROID)
  // Android does not do software compositing, so we can't get here.
  NOTREACHED();
  return nullptr;
#elif defined(USE_OZONE)
  ui::SurfaceFactoryOzone* factory =
      ui::OzonePlatform::GetInstance()->GetSurfaceFactoryOzone();
  std::unique_ptr<ui::SurfaceOzoneCanvas> surface_ozone =
      factory->CreateCanvasForWidget(surface_handle);
  CHECK(surface_ozone);
  return std::make_unique<SoftwareOutputDeviceOzone>(std::move(surface_ozone));
#elif defined(USE_X11)
  return std::make_unique<SoftwareOutputDeviceX11>(surface_handle);
#endif
}

}  // namespace viz
