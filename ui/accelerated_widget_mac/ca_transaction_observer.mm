// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ui/accelerated_widget_mac/ca_transaction_observer.h"

#include "base/no_destructor.h"
#include "base/time/default_tick_clock.h"
#include "base/trace_event/trace_event.h"
#include "ui/accelerated_widget_mac/window_resize_helper_mac.h"

#import <AppKit/AppKit.h>
#import <CoreFoundation/CoreFoundation.h>
#import <QuartzCore/QuartzCore.h>

typedef enum {
  kCATransactionPhasePreLayout,
  kCATransactionPhasePreCommit,
  kCATransactionPhasePostCommit,
} CATransactionPhase;

API_AVAILABLE(macos(10.11))
@interface CATransaction ()
+ (void)addCommitHandler:(void (^)(void))block
                forPhase:(CATransactionPhase)phase;
@end

namespace ui {

namespace {
NSString* kRunLoopMode = @"Chrome CATransactionCoordinator commit handler";
constexpr auto kPostCommitTimeout = base::TimeDelta::FromMilliseconds(50);
}  // namespace

CATransactionCoordinator& CATransactionCoordinator::Get() {
  static base::NoDestructor<CATransactionCoordinator> instance;
  return *instance;
}

void CATransactionCoordinator::SynchronizeImpl() {
  static bool registeredRunLoopMode = false;
  if (!registeredRunLoopMode) {
    CFRunLoopAddCommonMode(CFRunLoopGetCurrent(),
                           static_cast<CFStringRef>(kRunLoopMode));
    registeredRunLoopMode = true;
  }
  if (active_)
    return;
  active_ = true;

  for (auto& observer : post_commit_observers_)
    observer.OnActivateForTransaction();

  [CATransaction addCommitHandler:^{
    PreCommitHandler();
  }
                         forPhase:kCATransactionPhasePreCommit];

  [CATransaction addCommitHandler:^{
    PostCommitHandler();
  }
                         forPhase:kCATransactionPhasePostCommit];
}

void CATransactionCoordinator::PreCommitHandler() {
  TRACE_EVENT0("ui", "CATransactionCoordinator: pre-commit handler");
  auto* clock = base::DefaultTickClock::GetInstance();
  const base::TimeTicks start_time = clock->NowTicks();
  while (true) {
    bool continue_waiting = false;
    base::TimeTicks deadline = start_time;
    for (auto& observer : pre_commit_observers_) {
      if (observer.ShouldWaitInPreCommit()) {
        continue_waiting = true;
        deadline = std::max(deadline, start_time + observer.PreCommitTimeout());
      }
    }
    if (!continue_waiting)
      break;  // success

    base::TimeDelta time_left = deadline - clock->NowTicks();
    if (time_left <= base::TimeDelta::FromSeconds(0))
      break;  // timeout

    ui::WindowResizeHelperMac::Get()->WaitForSingleTaskToRun(time_left);
  }
}

void CATransactionCoordinator::PostCommitHandler() {
  TRACE_EVENT0("ui", "CATransactionCoordinator: post-commit handler");

  for (auto& observer : post_commit_observers_)
    observer.OnEnterPostCommit();

  auto* clock = base::DefaultTickClock::GetInstance();
  const base::TimeTicks deadline = clock->NowTicks() + kPostCommitTimeout;
  while (true) {
    bool continue_waiting = std::any_of(
        post_commit_observers_.begin(), post_commit_observers_.end(),
        std::mem_fn(&PostCommitObserver::ShouldWaitInPostCommit));
    if (!continue_waiting)
      break;  // success

    base::TimeDelta time_left = deadline - clock->NowTicks();
    if (time_left <= base::TimeDelta::FromSeconds(0))
      break;  // timeout

    ui::WindowResizeHelperMac::Get()->WaitForSingleTaskToRun(time_left);
  }
  active_ = false;
}

CATransactionCoordinator::CATransactionCoordinator() = default;
CATransactionCoordinator::~CATransactionCoordinator() = default;

void CATransactionCoordinator::Synchronize() {
  if (disabled_for_testing_)
    return;
  if (@available(macos 10.11, *))
    SynchronizeImpl();
}

void CATransactionCoordinator::AddPreCommitObserver(
    PreCommitObserver* observer) {
  pre_commit_observers_.AddObserver(observer);
}

void CATransactionCoordinator::RemovePreCommitObserver(
    PreCommitObserver* observer) {
  pre_commit_observers_.RemoveObserver(observer);
}

void CATransactionCoordinator::AddPostCommitObserver(
    PostCommitObserver* observer) {
  post_commit_observers_.AddObserver(observer);
}

void CATransactionCoordinator::RemovePostCommitObserver(
    PostCommitObserver* observer) {
  post_commit_observers_.RemoveObserver(observer);
}

}  // namespace ui
