// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <vector>

#include "base/at_exit.h"
#include "base/bind.h"
#include "base/command_line.h"
#include "base/logging.h"
#include "base/memory/ref_counted.h"
#include "base/strings/string_piece.h"
#include "base/strings/string_util.h"
#include "build/build_config.h"
#include "gpu/command_buffer/common/constants.h"
#include "gpu/command_buffer/common/context_creation_attribs.h"
#include "gpu/command_buffer/common/sync_token.h"
#include "gpu/command_buffer/service/buffer_manager.h"
#include "gpu/command_buffer/service/command_buffer_direct.h"
#include "gpu/command_buffer/service/context_group.h"
#include "gpu/command_buffer/service/gles2_cmd_decoder.h"
#include "gpu/command_buffer/service/gpu_switches.h"
#include "gpu/command_buffer/service/gpu_tracer.h"
#include "gpu/command_buffer/service/image_manager.h"
#include "gpu/command_buffer/service/logger.h"
#include "gpu/command_buffer/service/mailbox_manager_impl.h"
#include "gpu/command_buffer/service/raster_decoder.h"
#include "gpu/command_buffer/service/raster_decoder_context_state.h"
#include "gpu/command_buffer/service/service_discardable_manager.h"
#include "gpu/command_buffer/service/sync_point_manager.h"
#include "gpu/command_buffer/service/transfer_buffer_manager.h"
#include "ui/gfx/geometry/size.h"
#include "ui/gl/gl_context.h"
#include "ui/gl/gl_context_egl.h"
#include "ui/gl/gl_context_stub.h"
#include "ui/gl/gl_image_ref_counted_memory.h"
#include "ui/gl/gl_share_group.h"
#include "ui/gl/gl_stub_api.h"
#include "ui/gl/gl_surface.h"
#include "ui/gl/gl_surface_egl.h"
#include "ui/gl/gl_surface_stub.h"
#include "ui/gl/init/gl_factory.h"
#include "ui/gl/test/gl_surface_test_support.h"

namespace gpu {
namespace {

const size_t kCommandBufferSize = 16384;
const size_t kTransferBufferSize = 16384;
const size_t kSmallTransferBufferSize = 16;
const size_t kTinyTransferBufferSize = 3;

#if !defined(GPU_FUZZER_USE_ANGLE) && !defined(GPU_FUZZER_USE_SWIFTSHADER)
#define GPU_FUZZER_USE_STUB
#endif

constexpr const char* kExtensions[] = {
    "GL_AMD_compressed_ATC_texture",
    "GL_ANGLE_client_arrays",
    "GL_ANGLE_depth_texture",
    "GL_ANGLE_framebuffer_multisample",
    "GL_ANGLE_instanced_arrays",
    "GL_ANGLE_request_extension",
    "GL_ANGLE_robust_client_memory",
    "GL_ANGLE_robust_resource_initialization",
    "GL_ANGLE_texture_compression_dxt3",
    "GL_ANGLE_texture_compression_dxt5",
    "GL_ANGLE_texture_rectangle",
    "GL_ANGLE_texture_usage",
    "GL_ANGLE_translated_shader_source",
    "GL_ANGLE_webgl_compatibility",
    "GL_APPLE_texture_format_BGRA8888",
    "GL_APPLE_vertex_array_object",
    "GL_APPLE_ycbcr_422",
    "GL_ARB_blend_func_extended",
    "GL_ARB_depth_texture",
    "GL_ARB_draw_buffers",
    "GL_ARB_draw_instanced",
    "GL_ARB_ES3_compatibility",
    "GL_ARB_explicit_attrib_location",
    "GL_ARB_explicit_uniform_location",
    "GL_ARB_framebuffer_sRGB",
    "GL_ARB_instanced_arrays",
    "GL_ARB_map_buffer_range",
    "GL_ARB_occlusion_query",
    "GL_ARB_occlusion_query2",
    "GL_ARB_pixel_buffer_object",
    "GL_ARB_program_interface_query",
    "GL_ARB_robustness",
    "GL_ARB_sampler_objects",
    "GL_ARB_shader_image_load_store",
    "GL_ARB_shading_language_420pack",
    "GL_ARB_texture_float",
    "GL_ARB_texture_gather",
    "GL_ARB_texture_non_power_of_two",
    "GL_ARB_texture_rectangle",
    "GL_ARB_texture_rg",
    "GL_ARB_texture_storage",
    "GL_ARB_timer_query",
    "GL_ARB_vertex_array_object",
    "GL_ATI_texture_compression_atitc",
    "GL_CHROMIUM_bind_generates_resource",
    "GL_CHROMIUM_color_buffer_float_rgb",
    "GL_CHROMIUM_color_buffer_float_rgba",
    "GL_CHROMIUM_copy_compressed_texture",
    "GL_CHROMIUM_copy_texture",
    "GL_CHROMIUM_texture_filtering_hint",
    "GL_EXT_blend_func_extended",
    "GL_EXT_blend_minmax",
    "GL_EXT_color_buffer_float",
    "GL_EXT_color_buffer_half_float",
    "GL_EXT_debug_marker",
    "GL_EXT_direct_state_access",
    "GL_EXT_discard_framebuffer",
    "GL_EXT_disjoint_timer_query",
    "GL_EXT_draw_buffers",
    "GL_EXT_frag_depth",
    "GL_EXT_framebuffer_multisample",
    "GL_EXT_framebuffer_sRGB",
    "GL_EXT_map_buffer_range",
    "GL_EXT_multisample_compatibility",
    "GL_EXT_multisampled_render_to_texture",
    "GL_EXT_occlusion_query_boolean",
    "GL_EXT_packed_depth_stencil",
    "GL_EXT_read_format_bgra",
    "GL_EXT_robustness",
    "GL_EXT_shader_texture_lod",
    "GL_EXT_sRGB",
    "GL_EXT_sRGB_write_control",
    "GL_EXT_texture_compression_dxt1",
    "GL_EXT_texture_compression_s3tc",
    "GL_EXT_texture_compression_s3tc_srgb",
    "GL_EXT_texture_filter_anisotropic",
    "GL_EXT_texture_format_BGRA8888",
    "GL_EXT_texture_norm16",
    "GL_EXT_texture_rg",
    "GL_EXT_texture_sRGB",
    "GL_EXT_texture_sRGB_decode",
    "GL_EXT_texture_storage",
    "GL_EXT_timer_query",
    "GL_IMG_multisampled_render_to_texture",
    "GL_IMG_texture_compression_pvrtc",
    "GL_INTEL_framebuffer_CMAA",
    "GL_KHR_blend_equation_advanced",
    "GL_KHR_blend_equation_advanced_coherent",
    "GL_KHR_debug",
    "GL_KHR_robustness",
    "GL_KHR_texture_compression_astc_ldr",
    "GL_NV_blend_equation_advanced",
    "GL_NV_blend_equation_advanced_coherent",
    "GL_NV_draw_buffers",
    "GL_NV_EGL_stream_consumer_external",
    "GL_NV_fence",
    "GL_NV_framebuffer_mixed_samples",
    "GL_NV_path_rendering",
    "GL_NV_pixel_buffer_object",
    "GL_NV_sRGB_formats",
    "GL_OES_compressed_ETC1_RGB8_texture",
    "GL_OES_depth24",
    "GL_OES_depth_texture",
    "GL_OES_EGL_image_external",
    "GL_OES_element_index_uint",
    "GL_OES_packed_depth_stencil",
    "GL_OES_rgb8_rgba8",
    "GL_OES_standard_derivatives",
    "GL_OES_texture_float",
    "GL_OES_texture_float_linear",
    "GL_OES_texture_half_float",
    "GL_OES_texture_half_float_linear",
    "GL_OES_texture_npot",
    "GL_OES_vertex_array_object"};
constexpr size_t kExtensionCount = arraysize(kExtensions);

#if defined(GPU_FUZZER_USE_STUB)
constexpr const char* kDriverVersions[] = {"OpenGL ES 2.0", "OpenGL ES 3.1",
                                           "2.0", "4.5.0"};
#endif

class BitIterator {
 public:
  BitIterator(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  bool GetBit() {
    if (offset_ == size_)
      return false;
    bool value = !!(data_[offset_] & (1u << bit_));
    if (++bit_ == 8) {
      bit_ = 0;
      ++offset_;
    }
    return value;
  }

  void Advance(size_t bits) {
    bit_ += bits;
    offset_ += bit_ / 8;
    if (offset_ >= size_) {
      offset_ = size_;
      bit_ = 0;
    } else {
      bit_ = bit_ % 8;
    }
  }

  size_t consumed_bytes() const { return offset_ + (bit_ + 7) / 8; }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t offset_ = 0;
  size_t bit_ = 0;
};

struct Config {
  size_t MakeFromBits(const uint8_t* bits, size_t size) {
    BitIterator it(bits, size);
    attrib_helper.red_size = 8;
    attrib_helper.green_size = 8;
    attrib_helper.blue_size = 8;
    attrib_helper.alpha_size = it.GetBit() ? 8 : 0;
    attrib_helper.depth_size = it.GetBit() ? 24 : 0;
    attrib_helper.stencil_size = it.GetBit() ? 8 : 0;
    attrib_helper.buffer_preserved = it.GetBit();
    attrib_helper.bind_generates_resource = it.GetBit();
    attrib_helper.single_buffer = it.GetBit();
    bool es3 = it.GetBit();
    attrib_helper.context_type =
        es3 ? CONTEXT_TYPE_OPENGLES3 : CONTEXT_TYPE_OPENGLES2;
    attrib_helper.enable_oop_rasterization = it.GetBit();

#if defined(GPU_FUZZER_USE_STUB)
    std::vector<base::StringPiece> enabled_extensions;
    enabled_extensions.reserve(kExtensionCount);
    for (const char* extension : kExtensions) {
      if (it.GetBit())
        enabled_extensions.push_back(extension);
    }
    extensions = base::JoinString(enabled_extensions, " ");

    bool desktop_driver = it.GetBit();
    size_t version_index = (desktop_driver ? 2 : 0) + (es3 ? 1 : 0);
    version = kDriverVersions[version_index];
#else
    // We consume bits even if we don't use them, so that the same inputs can be
    // used for every fuzzer variation
    it.Advance(kExtensionCount + 1);
#endif

#define GPU_OP(type, name) workarounds.name = it.GetBit();
    GPU_DRIVER_BUG_WORKAROUNDS(GPU_OP)
#undef GPU_OP

#if defined(GPU_FUZZER_USE_PASSTHROUGH_CMD_DECODER)
    gl_context_attribs.bind_generates_resource =
        attrib_helper.bind_generates_resource;
    gl_context_attribs.webgl_compatibility_context =
        IsWebGLContextType(attrib_helper.context_type);
    gl_context_attribs.global_texture_share_group = true;
    gl_context_attribs.robust_resource_initialization = true;
    gl_context_attribs.robust_buffer_access = true;
    gl_context_attribs.client_major_es_version =
        IsWebGL2OrES3ContextType(attrib_helper.context_type) ? 3 : 2;
    gl_context_attribs.client_minor_es_version = 0;
#endif

    return it.consumed_bytes();
  }

  GpuDriverBugWorkarounds workarounds;
  ContextCreationAttribs attrib_helper;
  gl::GLContextAttribs gl_context_attribs;
#if defined(GPU_FUZZER_USE_STUB)
  const char* version;
  std::string extensions;
#endif
};

GpuPreferences GetGpuPreferences() {
  GpuPreferences preferences;
#if defined(GPU_FUZZER_USE_PASSTHROUGH_CMD_DECODER)
  preferences.use_passthrough_cmd_decoder = true;
#endif
  return preferences;
}

class CommandBufferSetup {
 public:
  CommandBufferSetup()
      : atexit_manager_(),
        gpu_preferences_(GetGpuPreferences()),
        share_group_(new gl::GLShareGroup),
        translator_cache_(gpu_preferences_) {
    logging::SetMinLogLevel(logging::LOG_FATAL);
    base::CommandLine::Init(0, NULL);

    auto* command_line = base::CommandLine::ForCurrentProcess();
    ALLOW_UNUSED_LOCAL(command_line);

#if defined(GPU_FUZZER_USE_ANGLE)
    command_line->AppendSwitchASCII(switches::kUseGL,
                                    gl::kGLImplementationANGLEName);
    command_line->AppendSwitchASCII(switches::kUseANGLE,
                                    gl::kANGLEImplementationNullName);
    CHECK(gl::init::InitializeGLOneOffImplementation(
        gl::kGLImplementationEGLGLES2, false, false, false, true));
#elif defined(GPU_FUZZER_USE_SWIFTSHADER)
    command_line->AppendSwitchASCII(switches::kUseGL,
                                    gl::kGLImplementationSwiftShaderName);
    CHECK(gl::init::InitializeGLOneOffImplementation(
        gl::kGLImplementationSwiftShaderGL, false, false, false, true));
#elif defined(GPU_FUZZER_USE_STUB)
    gl::GLSurfaceTestSupport::InitializeOneOffWithStubBindings();
    // Because the context depends on configuration bits, we want to recreate
    // it every time.
    recreate_context_ = true;
#else
#error invalid configuration
#endif
    discardable_manager_ = std::make_unique<ServiceDiscardableManager>();

    if (gpu_preferences_.use_passthrough_cmd_decoder)
      recreate_context_ = true;

    surface_ = gl::init::CreateOffscreenGLSurface(gfx::Size());
    if (!recreate_context_) {
      InitContext();
    }
  }

  bool InitDecoder() {
    if (recreate_context_) {
      InitContext();
    }

    context_->MakeCurrent(surface_.get());
    GpuFeatureInfo gpu_feature_info;
#if defined(GPU_FUZZER_USE_RASTER_DECODER)
    // Cause feature_info's |chromium_raster_transport| to be enabled.
    gpu_feature_info.status_values[GPU_FEATURE_TYPE_OOP_RASTERIZATION] =
        kGpuFeatureStatusEnabled;
#endif
    scoped_refptr<gles2::FeatureInfo> feature_info =
        new gles2::FeatureInfo(config_.workarounds, gpu_feature_info);
    scoped_refptr<gles2::ContextGroup> context_group = new gles2::ContextGroup(
        gpu_preferences_, true, &mailbox_manager_, nullptr /* memory_tracker */,
        &translator_cache_, &completeness_cache_, feature_info,
        config_.attrib_helper.bind_generates_resource, &image_manager_,
        nullptr /* image_factory */, nullptr /* progress_reporter */,
        gpu_feature_info, discardable_manager_.get());
    command_buffer_.reset(new CommandBufferDirect(
        context_group->transfer_buffer_manager(), &sync_point_manager_));

#if defined(GPU_FUZZER_USE_RASTER_DECODER)
    CHECK(feature_info->feature_flags().chromium_raster_transport);
    scoped_refptr<raster::RasterDecoderContextState> context_state =
        new raster::RasterDecoderContextState(
            share_group_, surface_, context_,
            config_.workarounds.use_virtualized_gl_contexts);
    context_state->InitializeGrContext(config_.workarounds, nullptr);
    decoder_.reset(raster::RasterDecoder::Create(
        command_buffer_.get(), command_buffer_->service(), &outputter_,
        context_group.get(), std::move(context_state)));
#else
    decoder_.reset(gles2::GLES2Decoder::Create(
        command_buffer_.get(), command_buffer_->service(), &outputter_,
        context_group.get()));
#endif

    decoder_->GetLogger()->set_log_synthesized_gl_errors(false);

    auto result = decoder_->Initialize(surface_.get(), context_.get(), true,
                                       gles2::DisallowedFeatures(),
                                       config_.attrib_helper);
    if (result != gpu::ContextResult::kSuccess)
      return false;

    command_buffer_->set_handler(decoder_.get());
    InitializeInitialCommandBuffer();

    decoder_->set_max_bucket_size(8 << 20);
    context_group->buffer_manager()->set_max_buffer_size(8 << 20);
    return decoder_->MakeCurrent();
  }

  void ResetDecoder() {
#if !defined(GPU_FUZZER_USE_RASTER_DECODER)
    // Keep a reference to the translators, which keeps them in the cache even
    // after the decoder is reset. They are expensive to initialize, but they
    // don't keep state.
    scoped_refptr<gles2::ShaderTranslatorInterface> translator =
        decoder_->GetTranslator(GL_VERTEX_SHADER);
    if (translator)
      translator->AddRef();
    translator = decoder_->GetTranslator(GL_FRAGMENT_SHADER);
    if (translator)
      translator->AddRef();
#endif
    decoder_->Destroy(true);
    decoder_.reset();
    if (recreate_context_) {
      context_->ReleaseCurrent(nullptr);
      context_ = nullptr;
    }
    command_buffer_.reset();
  }

  void RunCommandBuffer(const uint8_t* data, size_t size) {
    size_t consumed = config_.MakeFromBits(data, size);
    consumed = (consumed + 3) & ~3;
    if (consumed > size)
      return;
    data += consumed;
    size -= consumed;
    // The commands are flushed at a uint32_t granularity. If the data is not
    // a full command, we zero-pad it.
    size_t padded_size = (size + 3) & ~3;
    // crbug.com/638836 The -max_len argument is sometimes not respected, so the
    // fuzzer may give us too much data. Bail ASAP in that case.
    if (padded_size > kCommandBufferSize)
      return;

    if (!InitDecoder())
      return;

    size_t buffer_size = buffer_->size();
    CHECK_LE(padded_size, buffer_size);
    command_buffer_->SetGetBuffer(buffer_id_);
    auto* memory = static_cast<char*>(buffer_->memory());
    memcpy(memory, data, size);
    if (size < buffer_size)
      memset(memory + size, 0, buffer_size - size);
    command_buffer_->Flush(padded_size / 4);
    ResetDecoder();
  }

 private:
  void CreateTransferBuffer(size_t size, int32_t id) {
    scoped_refptr<Buffer> buffer =
        command_buffer_->CreateTransferBufferWithId(size, id);
    memset(buffer->memory(), 0, size);
  }

  void InitializeInitialCommandBuffer() {
    buffer_id_ = 1;
    buffer_ = command_buffer_->CreateTransferBufferWithId(kCommandBufferSize,
                                                          buffer_id_);
    CHECK(buffer_);
    // Create some transfer buffers. This is somewhat arbitrary, but having a
    // reasonably sized buffer in slot 4 allows us to prime the corpus with data
    // extracted from unit tests.
    CreateTransferBuffer(kTransferBufferSize, 2);
    CreateTransferBuffer(kSmallTransferBufferSize, 3);
    CreateTransferBuffer(kTransferBufferSize, 4);
    CreateTransferBuffer(kTinyTransferBufferSize, 5);
  }

  void InitContext() {
#if !defined(GPU_FUZZER_USE_STUB)
    context_ = new gl::GLContextEGL(share_group_.get());
    context_->Initialize(surface_.get(), config_.gl_context_attribs);
#else
    scoped_refptr<gl::GLContextStub> context_stub =
        new gl::GLContextStub(share_group_.get());
    context_stub->SetGLVersionString(config_.version);
    context_stub->SetExtensionsString(config_.extensions.c_str());
    context_stub->SetUseStubApi(true);
    context_ = context_stub;
#endif

// When not using the passthrough decoder, ANGLE should not be generating
// errors (the decoder should prevent those from happening). We register a
// callback to catch them if it does.
#if defined(GPU_FUZZER_USE_ANGLE) && \
    !defined(GPU_FUZZER_USE_PASSTHROUGH_CMD_DECODER)
    context_->MakeCurrent(surface_.get());
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr,
                          GL_FALSE);
    glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR,
                          GL_DEBUG_SEVERITY_HIGH, 0, nullptr, GL_TRUE);

    glDebugMessageCallback(&LogGLDebugMessage, nullptr);
#endif
  }

  static void APIENTRY LogGLDebugMessage(GLenum source,
                                         GLenum type,
                                         GLuint id,
                                         GLenum severity,
                                         GLsizei length,
                                         const GLchar* message,
                                         const GLvoid* user_param) {
    LOG_IF(FATAL, (id != GL_OUT_OF_MEMORY)) << "GL Driver Message: " << message;
  }

  base::AtExitManager atexit_manager_;

  GpuPreferences gpu_preferences_;

  Config config_;

  gles2::MailboxManagerImpl mailbox_manager_;
  gles2::TraceOutputter outputter_;
  scoped_refptr<gl::GLShareGroup> share_group_;
  SyncPointManager sync_point_manager_;
  gles2::ImageManager image_manager_;
  std::unique_ptr<ServiceDiscardableManager> discardable_manager_;

  bool recreate_context_ = false;
  scoped_refptr<gl::GLSurface> surface_;
  scoped_refptr<gl::GLContext> context_;

  gles2::ShaderTranslatorCache translator_cache_;
  gles2::FramebufferCompletenessCache completeness_cache_;

  std::unique_ptr<CommandBufferDirect> command_buffer_;

#if defined(GPU_FUZZER_USE_RASTER_DECODER)
  std::unique_ptr<raster::RasterDecoder> decoder_;
#else
  std::unique_ptr<gles2::GLES2Decoder> decoder_;
#endif

  scoped_refptr<Buffer> buffer_;
  int32_t buffer_id_ = 0;
};

// Intentionally leaked because asan tries to exit cleanly after a crash, but
// the decoder is usually in a bad state at that point.
// We need to load ANGLE libraries before the fuzzer infrastructure starts,
// because it gets confused about new coverage counters being dynamically
// registered, causing crashes.
CommandBufferSetup* g_setup = new CommandBufferSetup();

}  // anonymous namespace
}  // namespace gpu

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  gpu::g_setup->RunCommandBuffer(data, size);
  return 0;
}
