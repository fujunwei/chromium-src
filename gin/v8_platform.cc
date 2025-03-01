// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "gin/public/v8_platform.h"

#include <algorithm>

#include "base/allocator/partition_allocator/address_space_randomization.h"
#include "base/allocator/partition_allocator/page_allocator.h"
#include "base/bind.h"
#include "base/bit_cast.h"
#include "base/bits.h"
#include "base/debug/stack_trace.h"
#include "base/location.h"
#include "base/logging.h"
#include "base/rand_util.h"
#include "base/sys_info.h"
#include "base/task_scheduler/post_task.h"
#include "base/task_scheduler/task_scheduler.h"
#include "base/task_scheduler/task_traits.h"
#include "base/trace_event/trace_event.h"
#include "build/build_config.h"
#include "gin/per_isolate_data.h"

namespace gin {

namespace {

base::LazyInstance<V8Platform>::Leaky g_v8_platform = LAZY_INSTANCE_INITIALIZER;

constexpr base::TaskTraits kDefaultTaskTraits = {
    base::TaskPriority::USER_VISIBLE};

constexpr base::TaskTraits kBlockingTaskTraits = {
    base::TaskPriority::USER_BLOCKING};

void PrintStackTrace() {
  base::debug::StackTrace trace;
  trace.Print();
}

class ConvertableToTraceFormatWrapper final
    : public base::trace_event::ConvertableToTraceFormat {
 public:
  explicit ConvertableToTraceFormatWrapper(
      std::unique_ptr<v8::ConvertableToTraceFormat>& inner)
      : inner_(std::move(inner)) {}
  ~ConvertableToTraceFormatWrapper() override = default;
  void AppendAsTraceFormat(std::string* out) const final {
    inner_->AppendAsTraceFormat(out);
  }

 private:
  std::unique_ptr<v8::ConvertableToTraceFormat> inner_;

  DISALLOW_COPY_AND_ASSIGN(ConvertableToTraceFormatWrapper);
};

class EnabledStateObserverImpl final
    : public base::trace_event::TraceLog::EnabledStateObserver {
 public:
  EnabledStateObserverImpl() = default;

  void OnTraceLogEnabled() final {
    base::AutoLock lock(mutex_);
    for (auto* o : observers_) {
      o->OnTraceEnabled();
    }
  }

  void OnTraceLogDisabled() final {
    base::AutoLock lock(mutex_);
    for (auto* o : observers_) {
      o->OnTraceDisabled();
    }
  }

  void AddObserver(v8::TracingController::TraceStateObserver* observer) {
    {
      base::AutoLock lock(mutex_);
      DCHECK(!observers_.count(observer));
      if (observers_.empty()) {
        base::trace_event::TraceLog::GetInstance()->AddEnabledStateObserver(
            this);
      }
      observers_.insert(observer);
    }
    // Fire the observer if recording is already in progress.
    if (base::trace_event::TraceLog::GetInstance()->IsEnabled())
      observer->OnTraceEnabled();
  }

  void RemoveObserver(v8::TracingController::TraceStateObserver* observer) {
    base::AutoLock lock(mutex_);
    DCHECK(observers_.count(observer) == 1);
    observers_.erase(observer);
    if (observers_.empty()) {
      base::trace_event::TraceLog::GetInstance()->RemoveEnabledStateObserver(
          this);
    }
  }

 private:
  base::Lock mutex_;
  std::unordered_set<v8::TracingController::TraceStateObserver*> observers_;

  DISALLOW_COPY_AND_ASSIGN(EnabledStateObserverImpl);
};

base::LazyInstance<EnabledStateObserverImpl>::Leaky g_trace_state_dispatcher =
    LAZY_INSTANCE_INITIALIZER;

// TODO(skyostil): Deduplicate this with the clamper in Blink.
class TimeClamper {
 public:
  static constexpr double kResolutionSeconds = 0.001;

  TimeClamper() : secret_(base::RandUint64()) {}

  double ClampTimeResolution(double time_seconds) const {
    bool was_negative = false;
    if (time_seconds < 0) {
      was_negative = true;
      time_seconds = -time_seconds;
    }
    // For each clamped time interval, compute a pseudorandom transition
    // threshold. The reported time will either be the start of that interval or
    // the next one depending on which side of the threshold |time_seconds| is.
    double interval = floor(time_seconds / kResolutionSeconds);
    double clamped_time = interval * kResolutionSeconds;
    double tick_threshold = ThresholdFor(clamped_time);

    if (time_seconds >= tick_threshold)
      clamped_time = (interval + 1) * kResolutionSeconds;
    if (was_negative)
      clamped_time = -clamped_time;
    return clamped_time;
  }

 private:
  inline double ThresholdFor(double clamped_time) const {
    uint64_t time_hash = MurmurHash3(bit_cast<int64_t>(clamped_time) ^ secret_);
    return clamped_time + kResolutionSeconds * ToDouble(time_hash);
  }

  static inline double ToDouble(uint64_t value) {
    // Exponent for double values for [1.0 .. 2.0]
    static const uint64_t kExponentBits = uint64_t{0x3FF0000000000000};
    static const uint64_t kMantissaMask = uint64_t{0x000FFFFFFFFFFFFF};
    uint64_t random = (value & kMantissaMask) | kExponentBits;
    return bit_cast<double>(random) - 1;
  }

  static inline uint64_t MurmurHash3(uint64_t value) {
    value ^= value >> 33;
    value *= uint64_t{0xFF51AFD7ED558CCD};
    value ^= value >> 33;
    value *= uint64_t{0xC4CEB9FE1A85EC53};
    value ^= value >> 33;
    return value;
  }

  const uint64_t secret_;
  DISALLOW_COPY_AND_ASSIGN(TimeClamper);
};

base::LazyInstance<TimeClamper>::Leaky g_time_clamper =
    LAZY_INSTANCE_INITIALIZER;

#if BUILDFLAG(USE_PARTITION_ALLOC)
base::PageAccessibilityConfiguration GetPageConfig(
    v8::PageAllocator::Permission permission) {
  switch (permission) {
    case v8::PageAllocator::Permission::kRead:
      return base::PageRead;
    case v8::PageAllocator::Permission::kReadWrite:
      return base::PageReadWrite;
    case v8::PageAllocator::Permission::kReadWriteExecute:
      return base::PageReadWriteExecute;
    case v8::PageAllocator::Permission::kReadExecute:
      return base::PageReadExecute;
    default:
      DCHECK_EQ(v8::PageAllocator::Permission::kNoAccess, permission);
      return base::PageInaccessible;
  }
}

class PageAllocator : public v8::PageAllocator {
 public:
  ~PageAllocator() override = default;

  size_t AllocatePageSize() override {
    return base::kPageAllocationGranularity;
  }

  size_t CommitPageSize() override { return base::kSystemPageSize; }

  void SetRandomMmapSeed(int64_t seed) override {
    base::SetRandomPageBaseSeed(seed);
  }

  void* GetRandomMmapAddr() override { return base::GetRandomPageBase(); }

  void* AllocatePages(void* address,
                      size_t length,
                      size_t alignment,
                      v8::PageAllocator::Permission permissions) override {
    base::PageAccessibilityConfiguration config = GetPageConfig(permissions);
    bool commit = (permissions != v8::PageAllocator::Permission::kNoAccess);
    return base::AllocPages(address, length, alignment, config,
                            base::PageTag::kV8, commit);
  }

  bool FreePages(void* address, size_t length) override {
    base::FreePages(address, length);
    return true;
  }

  bool ReleasePages(void* address, size_t length, size_t new_length) override {
    DCHECK_LT(new_length, length);
    uint8_t* release_base = reinterpret_cast<uint8_t*>(address) + new_length;
    size_t release_size = length - new_length;
#if defined(OS_POSIX)
    // On POSIX, we can unmap the trailing pages.
    base::FreePages(release_base, release_size);
#else  // defined(OS_WIN)
    // On Windows, we can only de-commit the trailing pages.
    base::DecommitSystemPages(release_base, release_size);
#endif
    return true;
  }

  bool SetPermissions(void* address,
                      size_t length,
                      Permission permissions) override {
    // If V8 sets permissions to none, we can discard the memory.
    if (permissions == v8::PageAllocator::Permission::kNoAccess) {
      base::DecommitSystemPages(address, length);
      return true;
    } else {
      return base::SetSystemPagesAccess(address, length,
                                        GetPageConfig(permissions));
    }
  }
};

base::LazyInstance<PageAllocator>::Leaky g_page_allocator =
    LAZY_INSTANCE_INITIALIZER;

#endif  // BUILDFLAG(USE_PARTITION_ALLOC)

}  // namespace

class V8Platform::TracingControllerImpl : public v8::TracingController {
 public:
  TracingControllerImpl() = default;
  ~TracingControllerImpl() override = default;

  // TracingController implementation.
  const uint8_t* GetCategoryGroupEnabled(const char* name) override {
    return TRACE_EVENT_API_GET_CATEGORY_GROUP_ENABLED(name);
  }
  uint64_t AddTraceEvent(
      char phase,
      const uint8_t* category_enabled_flag,
      const char* name,
      const char* scope,
      uint64_t id,
      uint64_t bind_id,
      int32_t num_args,
      const char** arg_names,
      const uint8_t* arg_types,
      const uint64_t* arg_values,
      std::unique_ptr<v8::ConvertableToTraceFormat>* arg_convertables,
      unsigned int flags) override {
    std::unique_ptr<base::trace_event::ConvertableToTraceFormat>
        convertables[2];
    if (num_args > 0 && arg_types[0] == TRACE_VALUE_TYPE_CONVERTABLE) {
      convertables[0].reset(
          new ConvertableToTraceFormatWrapper(arg_convertables[0]));
    }
    if (num_args > 1 && arg_types[1] == TRACE_VALUE_TYPE_CONVERTABLE) {
      convertables[1].reset(
          new ConvertableToTraceFormatWrapper(arg_convertables[1]));
    }
    DCHECK_LE(num_args, 2);
    base::trace_event::TraceEventHandle handle =
        TRACE_EVENT_API_ADD_TRACE_EVENT_WITH_BIND_ID(
            phase, category_enabled_flag, name, scope, id, bind_id, num_args,
            arg_names, arg_types, (const long long unsigned int*)arg_values,
            convertables, flags);
    uint64_t result;
    memcpy(&result, &handle, sizeof(result));
    return result;
  }
  void UpdateTraceEventDuration(const uint8_t* category_enabled_flag,
                                const char* name,
                                uint64_t handle) override {
    base::trace_event::TraceEventHandle traceEventHandle;
    memcpy(&traceEventHandle, &handle, sizeof(handle));
    TRACE_EVENT_API_UPDATE_TRACE_EVENT_DURATION(category_enabled_flag, name,
                                                traceEventHandle);
  }
  void AddTraceStateObserver(TraceStateObserver* observer) override {
    g_trace_state_dispatcher.Get().AddObserver(observer);
  }
  void RemoveTraceStateObserver(TraceStateObserver* observer) override {
    g_trace_state_dispatcher.Get().RemoveObserver(observer);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(TracingControllerImpl);
};

// static
V8Platform* V8Platform::Get() { return g_v8_platform.Pointer(); }

V8Platform::V8Platform() : tracing_controller_(new TracingControllerImpl) {}

V8Platform::~V8Platform() = default;

#if BUILDFLAG(USE_PARTITION_ALLOC)
v8::PageAllocator* V8Platform::GetPageAllocator() {
  return g_page_allocator.Pointer();
}

void V8Platform::OnCriticalMemoryPressure() {
// We only have a reservation on 32-bit Windows systems.
// TODO(bbudge) Make the #if's in BlinkInitializer match.
#if defined(OS_WIN) && defined(ARCH_CPU_32_BITS)
  base::ReleaseReservation();
#endif
}
#endif  // BUILDFLAG(USE_PARTITION_ALLOC)

std::shared_ptr<v8::TaskRunner> V8Platform::GetForegroundTaskRunner(
    v8::Isolate* isolate) {
  PerIsolateData* data = PerIsolateData::From(isolate);
  return data->task_runner();
}

int V8Platform::NumberOfWorkerThreads() {
  // V8Platform assumes the scheduler uses the same set of workers for default
  // and user blocking tasks.
  const int num_foreground_workers =
      base::TaskScheduler::GetInstance()
          ->GetMaxConcurrentNonBlockedTasksWithTraitsDeprecated(
              kDefaultTaskTraits);
  DCHECK_EQ(num_foreground_workers,
            base::TaskScheduler::GetInstance()
                ->GetMaxConcurrentNonBlockedTasksWithTraitsDeprecated(
                    kBlockingTaskTraits));
  return std::max(1, num_foreground_workers);
}

void V8Platform::CallOnWorkerThread(std::unique_ptr<v8::Task> task) {
  base::PostTaskWithTraits(FROM_HERE, kDefaultTaskTraits,
                           base::BindOnce(&v8::Task::Run, std::move(task)));
}

void V8Platform::CallBlockingTaskOnWorkerThread(
    std::unique_ptr<v8::Task> task) {
  base::PostTaskWithTraits(FROM_HERE, kBlockingTaskTraits,
                           base::BindOnce(&v8::Task::Run, std::move(task)));
}

void V8Platform::CallDelayedOnWorkerThread(std::unique_ptr<v8::Task> task,
                                           double delay_in_seconds) {
  base::PostDelayedTaskWithTraits(
      FROM_HERE, kDefaultTaskTraits,
      base::BindOnce(&v8::Task::Run, std::move(task)),
      base::TimeDelta::FromSecondsD(delay_in_seconds));
}

void V8Platform::CallOnForegroundThread(v8::Isolate* isolate, v8::Task* task) {
  PerIsolateData* data = PerIsolateData::From(isolate);
  data->task_runner()->PostTask(std::unique_ptr<v8::Task>(task));
}

void V8Platform::CallDelayedOnForegroundThread(v8::Isolate* isolate,
                                               v8::Task* task,
                                               double delay_in_seconds) {
  PerIsolateData* data = PerIsolateData::From(isolate);
  data->task_runner()->PostDelayedTask(std::unique_ptr<v8::Task>(task),
                                       delay_in_seconds);
}

void V8Platform::CallIdleOnForegroundThread(v8::Isolate* isolate,
                                            v8::IdleTask* task) {
  PerIsolateData* data = PerIsolateData::From(isolate);
  data->task_runner()->PostIdleTask(std::unique_ptr<v8::IdleTask>(task));
}

bool V8Platform::IdleTasksEnabled(v8::Isolate* isolate) {
  return PerIsolateData::From(isolate)->task_runner()->IdleTasksEnabled();
}

double V8Platform::MonotonicallyIncreasingTime() {
  return base::TimeTicks::Now().ToInternalValue() /
      static_cast<double>(base::Time::kMicrosecondsPerSecond);
}

double V8Platform::CurrentClockTimeMillis() {
  double now_seconds = base::Time::Now().ToJsTime() / 1000;
  return g_time_clamper.Get().ClampTimeResolution(now_seconds) * 1000;
}

v8::TracingController* V8Platform::GetTracingController() {
  return tracing_controller_.get();
}

v8::Platform::StackTracePrinter V8Platform::GetStackTracePrinter() {
  return PrintStackTrace;
}

}  // namespace gin
