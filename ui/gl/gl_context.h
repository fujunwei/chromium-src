// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef UI_GL_GL_CONTEXT_H_
#define UI_GL_GL_CONTEXT_H_

#include <memory>
#include <string>

#include "base/atomicops.h"
#include "base/cancelable_callback.h"
#include "base/macros.h"
#include "base/memory/ref_counted.h"
#include "base/synchronization/cancellation_flag.h"
#include "build/build_config.h"
#include "ui/gfx/extension_set.h"
#include "ui/gl/gl_export.h"
#include "ui/gl/gl_share_group.h"
#include "ui/gl/gl_state_restorer.h"
#include "ui/gl/gl_workarounds.h"
#include "ui/gl/gpu_preference.h"

namespace gfx {
class ColorSpace;
}  // namespace gfx

namespace gl {
class YUVToRGBConverter;
}  // namespace gl

namespace gpu {
class GLContextVirtual;
}  // namespace gpu

namespace gl {

struct CurrentGL;
class DebugGLApi;
struct DriverGL;
class GLApi;
class GLSurface;
class GPUTiming;
class GPUTimingClient;
struct GLVersionInfo;
class RealGLApi;
class TraceGLApi;

// Where available, choose a GL context priority for devices that support it.
// Currently this requires the EGL_IMG_context_priority extension that is
// present on Daydream ready Android devices. Default is Medium, and the
// attribute is ignored if the extension is missing.
//
// "High" priority must only be used for special cases with strong realtime
// requirements, it is incompatible with other critical system GL work such as
// the GVR library's asynchronous reprojection for VR viewing. Please avoid
// using it for any GL contexts that may be used during VR presentation,
// see crbug.com/727800.
//
// Instead, consider using "Low" priority for possibly-slow GL work such as
// user WebGL content.
enum ContextPriority {
  ContextPriorityLow,
  ContextPriorityMedium,
  ContextPriorityHigh
};

struct GLContextAttribs {
  GpuPreference gpu_preference = PreferIntegratedGpu;
  bool bind_generates_resource = true;
  bool webgl_compatibility_context = false;
  bool global_texture_share_group = false;
  bool robust_resource_initialization = false;
  bool robust_buffer_access = false;
  int client_major_es_version = 3;
  int client_minor_es_version = 0;
  ContextPriority context_priority = ContextPriorityMedium;
};

// Encapsulates an OpenGL context, hiding platform specific management.
class GL_EXPORT GLContext : public base::RefCounted<GLContext> {
 public:
  explicit GLContext(GLShareGroup* share_group);

  static int32_t TotalGLContexts();

  static bool SwitchableGPUsSupported();
  // This should be called at most once at GPU process startup time.
  // By default, GPU switching is not supported unless this is called.
  static void SetSwitchableGPUsSupported();

  // This should be called at most once at GPU process startup time.
  static void SetForcedGpuPreference(GpuPreference gpu_preference);
  // If a gpu preference is forced (by GPU driver bug workaround, etc), return
  // it. Otherwise, return the original input preference.
  static GpuPreference AdjustGpuPreference(GpuPreference gpu_preference);

  // Initializes the GL context to be compatible with the given surface. The GL
  // context can be made with other surface's of the same type. The compatible
  // surface is only needed for certain platforms like WGL, OSMesa and GLX. It
  // should be specific for all platforms though.
  virtual bool Initialize(GLSurface* compatible_surface,
                          const GLContextAttribs& attribs) = 0;

  // Makes the GL context and a surface current on the current thread.
  virtual bool MakeCurrent(GLSurface* surface) = 0;

  // Releases this GL context and surface as current on the current thread.
  virtual void ReleaseCurrent(GLSurface* surface) = 0;

  // Returns true if this context and surface is current. Pass a null surface
  // if the current surface is not important.
  virtual bool IsCurrent(GLSurface* surface) = 0;

  // Get the underlying platform specific GL context "handle".
  virtual void* GetHandle() = 0;

  // Creates a GPUTimingClient class which abstracts various GPU Timing exts.
  virtual scoped_refptr<GPUTimingClient> CreateGPUTimingClient() = 0;

  // Set the GL workarounds.
  void SetGLWorkarounds(const GLWorkarounds& workarounds);

  void SetDisabledGLExtensions(const std::string& disabled_gl_extensions);

  // Gets the GLStateRestorer for the context.
  GLStateRestorer* GetGLStateRestorer();

  // Sets the GLStateRestorer for the context (takes ownership).
  void SetGLStateRestorer(GLStateRestorer* state_restorer);

  // Returns set of extensions. The context must be current.
  virtual const gfx::ExtensionSet& GetExtensions() = 0;

  // Indicate that it is safe to force this context to switch GPUs, since
  // transitioning can cause corruption and hangs (OS X only).
  virtual void SetSafeToForceGpuSwitch();

  // Attempt to force the context to move to the GPU of its sharegroup. Return
  // false only in the event of an unexpected error on the context.
  virtual bool ForceGpuSwitchIfNeeded();

  // Indicate that the real context switches should unbind the FBO first
  // (For an Android work-around only).
  virtual void SetUnbindFboOnMakeCurrent();

  // Returns whether the current context supports the named extension. The
  // context must be current.
  bool HasExtension(const char* name);

  // Returns version info of the underlying GL context. The context must be
  // current.
  const GLVersionInfo* GetVersionInfo();

  GLShareGroup* share_group();

  static bool LosesAllContextsOnContextLost();

  // Returns the last GLContext made current, virtual or real.
  static GLContext* GetCurrent();

  virtual bool WasAllocatedUsingRobustnessExtension();

  // Make this context current when used for context virtualization.
  bool MakeVirtuallyCurrent(GLContext* virtual_context, GLSurface* surface);

  // Notify this context that |virtual_context|, that was using us, is
  // being released or destroyed.
  void OnReleaseVirtuallyCurrent(GLContext* virtual_context);

  // Returns the GL version string. The context must be current.
  virtual std::string GetGLVersion();

  // Returns the GL renderer string. The context must be current.
  virtual std::string GetGLRenderer();

  // Returns a helper structure to convert the YUV color space |color_space|
  // to its associated full-range RGB color space.
  virtual YUVToRGBConverter* GetYUVToRGBConverter(
      const gfx::ColorSpace& color_space);

  // Get the CurrentGL object for this context containing the driver, version
  // and API.
  CurrentGL* GetCurrentGL();

  // Reinitialize the dynamic bindings of this context.  Needed when the driver
  // may be exposing different extensions compared to when it was initialized.
  // TODO(geofflang): Try to make this call uncessessary by pre-loading all
  // extension entry points.
  void ReinitializeDynamicBindings();

  // Forces this context, which must be a virtual context, to be no
  // longer considered virtually current. The real context remains
  // current.
  virtual void ForceReleaseVirtuallyCurrent();

#if defined(OS_MACOSX)
  // Create a fence for all work submitted to this context so far, and return a
  // monotonically increasing handle to it. This returned handle never needs to
  // be freed. This method is used to create backpressure to throttle GL work
  // on macOS, so that we do not starve CoreAnimation.
  virtual uint64_t BackpressureFenceCreate();
  // Perform a client-side wait on a previously-created fence.
  virtual void BackpressureFenceWait(uint64_t fence);
#endif

 protected:
  virtual ~GLContext();

  // Create the GLApi for this context using the provided driver. Creates a
  // RealGLApi by default.
  virtual GLApi* CreateGLApi(DriverGL* driver);

  // Will release the current context when going out of scope, unless canceled.
  class ScopedReleaseCurrent {
   public:
    ScopedReleaseCurrent();
    ~ScopedReleaseCurrent();

    void Cancel();

   private:
    bool canceled_;
  };

  // Sets the GL api to the real hardware API (vs the VirtualAPI)
  void BindGLApi();
  virtual void SetCurrent(GLSurface* surface);

  // Initialize function pointers to functions where the bound version depends
  // on GL version or supported extensions. Should be called immediately after
  // this context is made current.
  void InitializeDynamicBindings();

  // Returns the last real (non-virtual) GLContext made current.
  static GLContext* GetRealCurrent();

  virtual void ResetExtensions() = 0;

  GLApi* gl_api() { return gl_api_.get(); }

 private:
  friend class base::RefCounted<GLContext>;

  // For GetRealCurrent.
  friend class gpu::GLContextVirtual;

  std::unique_ptr<GLVersionInfo> GenerateGLVersionInfo();

  static base::subtle::Atomic32 total_gl_contexts_;

  static bool switchable_gpus_supported_;

  static GpuPreference forced_gpu_preference_;

  GLWorkarounds gl_workarounds_;
  std::string disabled_gl_extensions_;

  bool static_bindings_initialized_ = false;
  bool dynamic_bindings_initialized_ = false;
  std::unique_ptr<DriverGL> driver_gl_;
  std::unique_ptr<GLApi> gl_api_;
  std::unique_ptr<TraceGLApi> trace_gl_api_;
  std::unique_ptr<DebugGLApi> debug_gl_api_;
  std::unique_ptr<CurrentGL> current_gl_;

  // Copy of the real API (if one was created) for dynamic initialization
  RealGLApi* real_gl_api_ = nullptr;

  scoped_refptr<GLShareGroup> share_group_;
  GLContext* current_virtual_context_ = nullptr;
  bool state_dirtied_externally_ = false;
  std::unique_ptr<GLStateRestorer> state_restorer_;
  std::unique_ptr<GLVersionInfo> version_info_;

  DISALLOW_COPY_AND_ASSIGN(GLContext);
};

class GL_EXPORT GLContextReal : public GLContext {
 public:
  explicit GLContextReal(GLShareGroup* share_group);
  scoped_refptr<GPUTimingClient> CreateGPUTimingClient() override;
  const gfx::ExtensionSet& GetExtensions() override;

 protected:
  ~GLContextReal() override;

  void ResetExtensions() override;

  void SetCurrent(GLSurface* surface) override;
  void SetExtensionsFromString(std::string extensions);
  const std::string& extension_string() { return extensions_string_; }

 private:
  std::unique_ptr<GPUTiming> gpu_timing_;
  std::string extensions_string_;
  gfx::ExtensionSet extensions_;
  bool extensions_initialized_ = false;
  DISALLOW_COPY_AND_ASSIGN(GLContextReal);
};

// Wraps GLContext in scoped_refptr and tries to initializes it. Returns a
// scoped_refptr containing the initialized GLContext or nullptr if
// initialization fails.
GL_EXPORT scoped_refptr<GLContext> InitializeGLContext(
    scoped_refptr<GLContext> context,
    GLSurface* compatible_surface,
    const GLContextAttribs& attribs);

}  // namespace gl

#endif  // UI_GL_GL_CONTEXT_H_
