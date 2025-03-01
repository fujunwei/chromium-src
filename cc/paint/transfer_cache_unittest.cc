// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <vector>

#include "base/command_line.h"
#include "base/threading/thread_task_runner_handle.h"
#include "build/build_config.h"
#include "cc/paint/image_transfer_cache_entry.h"
#include "cc/paint/raw_memory_transfer_cache_entry.h"
#include "cc/paint/transfer_cache_entry.h"
#include "cc/test/test_in_process_context_provider.h"
#include "gpu/command_buffer/client/client_transfer_cache.h"
#include "gpu/command_buffer/client/gles2_cmd_helper.h"
#include "gpu/command_buffer/client/gles2_implementation.h"
#include "gpu/command_buffer/client/raster_interface.h"
#include "gpu/command_buffer/client/shared_memory_limits.h"
#include "gpu/command_buffer/common/context_creation_attribs.h"
#include "gpu/command_buffer/service/service_transfer_cache.h"
#include "gpu/config/gpu_switches.h"
#include "gpu/ipc/raster_in_process_context.h"
#include "testing/gtest/include/gtest/gtest.h"
#include "third_party/skia/include/core/SkImage.h"
#include "ui/gl/gl_implementation.h"

namespace cc {
namespace {

class TransferCacheTest : public testing::Test {
 public:
  TransferCacheTest() : test_client_entry_(std::vector<uint8_t>(100)) {}

  void SetUp() override {
    gpu::ContextCreationAttribs attribs;
    attribs.alpha_size = -1;
    attribs.depth_size = 24;
    attribs.stencil_size = 8;
    attribs.samples = 0;
    attribs.sample_buffers = 0;
    attribs.fail_if_major_perf_caveat = false;
    attribs.bind_generates_resource = false;
    // Enable OOP rasterization.
    attribs.enable_oop_rasterization = true;
    attribs.enable_raster_interface = true;
    attribs.enable_gles2_interface = false;

    context_ = std::make_unique<gpu::RasterInProcessContext>();
    auto result = context_->Initialize(
        /*service=*/nullptr, attribs, gpu::SharedMemoryLimits(),
        &gpu_memory_buffer_manager_, &image_factory_,
        /*gpu_channel_manager_delegate=*/nullptr, nullptr, nullptr);

    ASSERT_EQ(result, gpu::ContextResult::kSuccess);
    ASSERT_TRUE(context_->GetCapabilities().supports_oop_raster);
  }

  void TearDown() override { context_.reset(); }

  gpu::ServiceTransferCache* ServiceTransferCache() {
    return context_->GetTransferCacheForTest();
  }

  int decoder_id() { return context_->GetRasterDecoderIdForTest(); }

  gpu::raster::RasterInterface* ri() { return context_->GetImplementation(); }

  gpu::ContextSupport* ContextSupport() {
    return context_->GetContextSupport();
  }

  const ClientRawMemoryTransferCacheEntry& test_client_entry() const {
    return test_client_entry_;
  }
  void CreateEntry(const ClientTransferCacheEntry& entry) {
    auto* context_support = ContextSupport();
    size_t size = entry.SerializedSize();
    void* data = context_support->MapTransferCacheEntry(size);
    ASSERT_TRUE(data);
    entry.Serialize(base::make_span(static_cast<uint8_t*>(data), size));
    context_support->UnmapAndCreateTransferCacheEntry(entry.UnsafeType(),
                                                      entry.Id());
  }

 private:
  viz::TestGpuMemoryBufferManager gpu_memory_buffer_manager_;
  TestImageFactory image_factory_;
  std::unique_ptr<gpu::RasterInProcessContext> context_;
  gl::DisableNullDrawGLBindings enable_pixel_output_;
  ClientRawMemoryTransferCacheEntry test_client_entry_;
};

TEST_F(TransferCacheTest, Basic) {
  auto* service_cache = ServiceTransferCache();
  auto* context_support = ContextSupport();

  // Create an entry.
  const auto& entry = test_client_entry();
  CreateEntry(entry);
  ri()->Finish();

  // Validate service-side state.
  EXPECT_NE(nullptr,
            service_cache->GetEntry(gpu::ServiceTransferCache::EntryKey(
                decoder_id(), entry.Type(), entry.Id())));

  // Unlock on client side and flush to service.
  context_support->UnlockTransferCacheEntries(
      {{entry.UnsafeType(), entry.Id()}});
  ri()->Finish();

  // Re-lock on client side and validate state. No need to flush as lock is
  // local.
  EXPECT_TRUE(context_support->ThreadsafeLockTransferCacheEntry(
      entry.UnsafeType(), entry.Id()));

  // Delete on client side, flush, and validate that deletion reaches service.
  context_support->DeleteTransferCacheEntry(entry.UnsafeType(), entry.Id());
  ri()->Finish();
  EXPECT_EQ(nullptr,
            service_cache->GetEntry(gpu::ServiceTransferCache::EntryKey(
                decoder_id(), entry.Type(), entry.Id())));
}

TEST_F(TransferCacheTest, MemoryEviction) {
  auto* service_cache = ServiceTransferCache();
  auto* context_support = ContextSupport();

  const auto& entry = test_client_entry();
  // Create an entry.
  CreateEntry(entry);
  ri()->Finish();

  // Validate service-side state.
  EXPECT_NE(nullptr,
            service_cache->GetEntry(gpu::ServiceTransferCache::EntryKey(
                decoder_id(), entry.Type(), entry.Id())));

  // Unlock on client side and flush to service.
  context_support->UnlockTransferCacheEntries(
      {{entry.UnsafeType(), entry.Id()}});
  ri()->Finish();

  // Evict on the service side.
  service_cache->SetCacheSizeLimitForTesting(0);
  EXPECT_EQ(nullptr,
            service_cache->GetEntry(gpu::ServiceTransferCache::EntryKey(
                decoder_id(), entry.Type(), entry.Id())));

  // Try to re-lock on the client side. This should fail.
  EXPECT_FALSE(context_support->ThreadsafeLockTransferCacheEntry(
      entry.UnsafeType(), entry.Id()));
}

TEST_F(TransferCacheTest, CountEviction) {
  auto* service_cache = ServiceTransferCache();
  auto* context_support = ContextSupport();

  // Create 10 entries and leave them all unlocked.
  std::vector<std::unique_ptr<ClientRawMemoryTransferCacheEntry>> entries;
  for (int i = 0; i < 10; ++i) {
    entries.emplace_back(std::make_unique<ClientRawMemoryTransferCacheEntry>(
        std::vector<uint8_t>(4)));
    CreateEntry(*entries[i]);
    context_support->UnlockTransferCacheEntries(
        {{entries[i]->UnsafeType(), entries[i]->Id()}});
    ri()->Finish();
  }

  // These entries should be using up space.
  EXPECT_EQ(service_cache->cache_size_for_testing(), 40u);

  // Evict on the service side.
  service_cache->SetMaxCacheEntriesForTesting(5);

  // Half the entries should be evicted.
  EXPECT_EQ(service_cache->cache_size_for_testing(), 20u);
}

TEST_F(TransferCacheTest, RawMemoryTransfer) {
  auto* service_cache = ServiceTransferCache();

  // Create an entry with some initialized data.
  std::vector<uint8_t> data;
  data.resize(100);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  // Add the entry to the transfer cache
  ClientRawMemoryTransferCacheEntry client_entry(data);
  CreateEntry(client_entry);
  ri()->Finish();

  // Validate service-side data matches.
  ServiceTransferCacheEntry* service_entry =
      service_cache->GetEntry(gpu::ServiceTransferCache::EntryKey(
          decoder_id(), client_entry.Type(), client_entry.Id()));
  EXPECT_EQ(service_entry->Type(), client_entry.Type());
  const std::vector<uint8_t> service_data =
      static_cast<ServiceRawMemoryTransferCacheEntry*>(service_entry)->data();
  EXPECT_EQ(data, service_data);
}

TEST_F(TransferCacheTest, ImageMemoryTransfer) {
// TODO(ericrk): This test doesn't work on Android. crbug.com/777628
#if defined(OS_ANDROID)
  return;
#endif

  auto* service_cache = ServiceTransferCache();

  // Create a 10x10 image.
  SkImageInfo info = SkImageInfo::MakeN32Premul(10, 10);
  std::vector<uint8_t> data;
  data.resize(info.width() * info.height() * 4);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i;
  }
  SkPixmap pixmap(info, data.data(), info.minRowBytes());

  // Add the entry to the transfer cache
  ClientImageTransferCacheEntry client_entry(&pixmap, nullptr, false);
  CreateEntry(client_entry);
  ri()->Finish();

  // Validate service-side data matches.
  ServiceTransferCacheEntry* service_entry =
      service_cache->GetEntry(gpu::ServiceTransferCache::EntryKey(
          decoder_id(), client_entry.Type(), client_entry.Id()));
  EXPECT_EQ(service_entry->Type(), client_entry.Type());
  sk_sp<SkImage> service_image =
      static_cast<ServiceImageTransferCacheEntry*>(service_entry)->image();
  EXPECT_TRUE(service_image->isTextureBacked());

  std::vector<uint8_t> service_data;
  service_data.resize(data.size());
  service_image->readPixels(info, service_data.data(), info.minRowBytes(), 0,
                            0);

  EXPECT_EQ(data, service_data);
}

}  // namespace
}  // namespace cc
