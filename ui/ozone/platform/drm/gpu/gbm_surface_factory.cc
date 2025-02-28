// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ui/ozone/platform/drm/gpu/gbm_surface_factory.h"

#include <gbm.h>

#include <utility>

#include "base/files/file_path.h"
#include "build/build_config.h"
#include "third_party/khronos/EGL/egl.h"
#include "ui/gfx/buffer_format_util.h"
#include "ui/gfx/native_pixmap.h"
#include "ui/gl/gl_surface_egl.h"
#include "ui/ozone/common/egl_util.h"
#include "ui/ozone/common/gl_ozone_egl.h"
#include "ui/ozone/common/gl_ozone_osmesa.h"
#include "ui/ozone/platform/drm/common/drm_util.h"
#include "ui/ozone/platform/drm/gpu/drm_thread_proxy.h"
#include "ui/ozone/platform/drm/gpu/drm_window_proxy.h"
#include "ui/ozone/platform/drm/gpu/gbm_buffer.h"
#include "ui/ozone/platform/drm/gpu/gbm_overlay_surface.h"
#include "ui/ozone/platform/drm/gpu/gbm_surface.h"
#include "ui/ozone/platform/drm/gpu/gbm_surfaceless.h"
#include "ui/ozone/platform/drm/gpu/proxy_helpers.h"
#include "ui/ozone/platform/drm/gpu/screen_manager.h"
#include "ui/ozone/public/surface_ozone_canvas.h"

#if BUILDFLAG(ENABLE_VULKAN)
#include "gpu/vulkan/vulkan_function_pointers.h"
#include "ui/ozone/platform/drm/gpu/vulkan_implementation_gbm.h"
#if defined(OS_CHROMEOS)
#define VK_STRUCTURE_TYPE_DMA_BUF_IMAGE_CREATE_INFO_INTEL 1024
typedef struct VkDmaBufImageCreateInfo_ {
  VkStructureType sType;
  const void* pNext;
  int fd;
  VkFormat format;
  VkExtent3D extent;
  uint32_t strideInBytes;
} VkDmaBufImageCreateInfo;

typedef VkResult(VKAPI_PTR* PFN_vkCreateDmaBufImageINTEL)(
    VkDevice device,
    const VkDmaBufImageCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDeviceMemory* pMem,
    VkImage* pImage);
#endif
#endif

namespace ui {

namespace {

class GLOzoneEGLGbm : public GLOzoneEGL {
 public:
  GLOzoneEGLGbm(GbmSurfaceFactory* surface_factory,
                DrmThreadProxy* drm_thread_proxy)
      : surface_factory_(surface_factory),
        drm_thread_proxy_(drm_thread_proxy) {}
  ~GLOzoneEGLGbm() override {}

  scoped_refptr<gl::GLSurface> CreateViewGLSurface(
      gfx::AcceleratedWidget window) override {
    return gl::InitializeGLSurface(new GbmSurface(
        surface_factory_, drm_thread_proxy_->CreateDrmWindowProxy(window),
        window));
  }

  scoped_refptr<gl::GLSurface> CreateSurfacelessViewGLSurface(
      gfx::AcceleratedWidget window) override {
    return gl::InitializeGLSurface(new GbmSurfaceless(
        surface_factory_, drm_thread_proxy_->CreateDrmWindowProxy(window),
        window));
  }

  scoped_refptr<gl::GLSurface> CreateOffscreenGLSurface(
      const gfx::Size& size) override {
    DCHECK_EQ(size.width(), 0);
    DCHECK_EQ(size.height(), 0);
    return gl::InitializeGLSurface(new gl::SurfacelessEGL(size));
  }

 protected:
  intptr_t GetNativeDisplay() override { return EGL_DEFAULT_DISPLAY; }

  bool LoadGLES2Bindings(gl::GLImplementation impl) override {
    return LoadDefaultEGLGLES2Bindings(impl);
  }

 private:
  GbmSurfaceFactory* surface_factory_;
  DrmThreadProxy* drm_thread_proxy_;

  DISALLOW_COPY_AND_ASSIGN(GLOzoneEGLGbm);
};

}  // namespace

GbmSurfaceFactory::GbmSurfaceFactory(DrmThreadProxy* drm_thread_proxy)
    : egl_implementation_(
          std::make_unique<GLOzoneEGLGbm>(this, drm_thread_proxy)),
      osmesa_implementation_(std::make_unique<GLOzoneOSMesa>()),
      drm_thread_proxy_(drm_thread_proxy) {}

GbmSurfaceFactory::~GbmSurfaceFactory() {
  DCHECK(thread_checker_.CalledOnValidThread());
}

void GbmSurfaceFactory::RegisterSurface(gfx::AcceleratedWidget widget,
                                        GbmSurfaceless* surface) {
  DCHECK(thread_checker_.CalledOnValidThread());
  widget_to_surface_map_.insert(std::make_pair(widget, surface));
}

void GbmSurfaceFactory::UnregisterSurface(gfx::AcceleratedWidget widget) {
  DCHECK(thread_checker_.CalledOnValidThread());
  widget_to_surface_map_.erase(widget);
}

GbmSurfaceless* GbmSurfaceFactory::GetSurface(
    gfx::AcceleratedWidget widget) const {
  DCHECK(thread_checker_.CalledOnValidThread());
  auto it = widget_to_surface_map_.find(widget);
  DCHECK(it != widget_to_surface_map_.end());
  return it->second;
}

std::vector<gl::GLImplementation>
GbmSurfaceFactory::GetAllowedGLImplementations() {
  DCHECK(thread_checker_.CalledOnValidThread());
  return std::vector<gl::GLImplementation>{gl::kGLImplementationEGLGLES2,
                                           gl::kGLImplementationOSMesaGL,
                                           gl::kGLImplementationSwiftShaderGL};
}

GLOzone* GbmSurfaceFactory::GetGLOzone(gl::GLImplementation implementation) {
  switch (implementation) {
    case gl::kGLImplementationEGLGLES2:
    case gl::kGLImplementationSwiftShaderGL:
      return egl_implementation_.get();
    case gl::kGLImplementationOSMesaGL:
      return osmesa_implementation_.get();
    default:
      return nullptr;
  }
}

#if BUILDFLAG(ENABLE_VULKAN)
std::unique_ptr<gpu::VulkanImplementation>
GbmSurfaceFactory::CreateVulkanImplementation() {
  return std::make_unique<ui::VulkanImplementationGbm>();
}

scoped_refptr<gfx::NativePixmap> GbmSurfaceFactory::CreateNativePixmapForVulkan(
    gfx::AcceleratedWidget widget,
    gfx::Size size,
    gfx::BufferFormat format,
    gfx::BufferUsage usage,
    VkDevice vk_device,
    VkDeviceMemory* vk_device_memory,
    VkImage* vk_image) {
#if defined(OS_CHROMEOS)
  scoped_refptr<GbmBuffer> buffer = drm_thread_proxy_->CreateBuffer(
      widget, size, format, usage, GbmBuffer::kFlagNoModifiers);
  if (!buffer.get())
    return nullptr;

  PFN_vkCreateDmaBufImageINTEL create_dma_buf_image_intel =
      reinterpret_cast<PFN_vkCreateDmaBufImageINTEL>(
          vkGetDeviceProcAddr(vk_device, "vkCreateDmaBufImageINTEL"));
  if (!create_dma_buf_image_intel) {
    LOG(ERROR) << "Scanout buffers can only be imported into vulkan when "
                  "vkCreateDmaBufImageINTEL is available.";
    return nullptr;
  }

  DCHECK(buffer->AreFdsValid());
  DCHECK_EQ(buffer->GetFdCount(), 1U);

  base::ScopedFD vk_image_fd(dup(buffer->GetFd(0)));
  DCHECK(vk_image_fd.is_valid());

  VkDmaBufImageCreateInfo dma_buf_image_create_info = {
      .sType = static_cast<VkStructureType>(
          VK_STRUCTURE_TYPE_DMA_BUF_IMAGE_CREATE_INFO_INTEL),
      .fd = vk_image_fd.release(),
      .format = VK_FORMAT_B8G8R8A8_SRGB,
      .extent = (VkExtent3D){size.width(), size.height(), 1},
      .strideInBytes = buffer->GetStride(0),
  };

  VkResult result =
      create_dma_buf_image_intel(vk_device, &dma_buf_image_create_info, nullptr,
                                 vk_device_memory, vk_image);
  if (result != VK_SUCCESS) {
    LOG(ERROR) << "Failed to create a Vulkan image from a dmabuf.";
    return nullptr;
  }

  return base::MakeRefCounted<GbmPixmap>(this, buffer);
#else
  return nullptr;
#endif
}
#endif

std::unique_ptr<OverlaySurface> GbmSurfaceFactory::CreateOverlaySurface(
    gfx::AcceleratedWidget window) {
  return std::make_unique<GbmOverlaySurface>(
      drm_thread_proxy_->CreateDrmWindowProxy(window));
}

std::unique_ptr<SurfaceOzoneCanvas> GbmSurfaceFactory::CreateCanvasForWidget(
    gfx::AcceleratedWidget widget) {
  DCHECK(thread_checker_.CalledOnValidThread());
  LOG(ERROR) << "Software rendering mode is not supported with GBM platform";
  return nullptr;
}

std::vector<gfx::BufferFormat> GbmSurfaceFactory::GetScanoutFormats(
    gfx::AcceleratedWidget widget) {
  std::vector<gfx::BufferFormat> scanout_formats;
  drm_thread_proxy_->GetScanoutFormats(widget, &scanout_formats);
  return scanout_formats;
}

scoped_refptr<gfx::NativePixmap> GbmSurfaceFactory::CreateNativePixmap(
    gfx::AcceleratedWidget widget,
    gfx::Size size,
    gfx::BufferFormat format,
    gfx::BufferUsage usage) {
  scoped_refptr<GbmBuffer> buffer = drm_thread_proxy_->CreateBuffer(
      widget, size, format, usage, 0 /* flags */);
  if (!buffer.get())
    return nullptr;

  return base::MakeRefCounted<GbmPixmap>(this, buffer);
}

scoped_refptr<gfx::NativePixmap>
GbmSurfaceFactory::CreateNativePixmapFromHandleInternal(
    gfx::AcceleratedWidget widget,
    gfx::Size size,
    gfx::BufferFormat format,
    const gfx::NativePixmapHandle& handle) {
  size_t num_planes = gfx::NumberOfPlanesForBufferFormat(format);
  if (handle.planes.size() != num_planes ||
      (handle.fds.size() != 1 && handle.fds.size() != num_planes)) {
    return nullptr;
  }
  std::vector<base::ScopedFD> scoped_fds;
  for (auto& fd : handle.fds) {
    scoped_fds.emplace_back(fd.fd);
  }

  std::vector<gfx::NativePixmapPlane> planes;
  for (const auto& plane : handle.planes) {
    planes.push_back(plane);
  }

  scoped_refptr<GbmBuffer> buffer = drm_thread_proxy_->CreateBufferFromFds(
      widget, size, format, std::move(scoped_fds), planes);
  if (!buffer)
    return nullptr;

  return base::MakeRefCounted<GbmPixmap>(this, buffer);
}

scoped_refptr<gfx::NativePixmap>
GbmSurfaceFactory::CreateNativePixmapFromHandle(
    gfx::AcceleratedWidget widget,
    gfx::Size size,
    gfx::BufferFormat format,
    const gfx::NativePixmapHandle& handle) {
  // Query the external service (if available), whether it recognizes this
  // NativePixmapHandle, and whether it can provide a corresponding NativePixmap
  // backing it. If so, the handle is consumed. Otherwise, the handle remains
  // valid and can be further importer by standard means.
  if (!get_protected_native_pixmap_callback_.is_null()) {
    auto protected_pixmap = get_protected_native_pixmap_callback_.Run(handle);
    if (protected_pixmap)
      return protected_pixmap;
  }

  return CreateNativePixmapFromHandleInternal(widget, size, format, handle);
}

scoped_refptr<gfx::NativePixmap>
GbmSurfaceFactory::CreateNativePixmapForProtectedBufferHandle(
    gfx::AcceleratedWidget widget,
    gfx::Size size,
    gfx::BufferFormat format,
    const gfx::NativePixmapHandle& handle) {
  // Create a new NativePixmap without querying the external service for any
  // existing mappings.
  return CreateNativePixmapFromHandleInternal(widget, size, format, handle);
}

void GbmSurfaceFactory::SetGetProtectedNativePixmapDelegate(
    const GetProtectedNativePixmapCallback&
        get_protected_native_pixmap_callback) {
  get_protected_native_pixmap_callback_ = get_protected_native_pixmap_callback;
}

}  // namespace ui
