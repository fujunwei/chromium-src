// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "extensions/browser/test_event_router_observer.h"

#include "base/memory/ptr_util.h"

namespace extensions {

TestEventRouterObserver::TestEventRouterObserver(EventRouter* event_router)
    : event_router_(event_router) {
  event_router_->AddObserverForTesting(this);
}

TestEventRouterObserver::~TestEventRouterObserver() {
  // Note: can't use ScopedObserver<> here because the method is
  // RemoveObserverForTesting() instead of RemoveObserver().
  event_router_->RemoveObserverForTesting(this);
}

void TestEventRouterObserver::ClearEvents() {
  events_.clear();
}

void TestEventRouterObserver::OnWillDispatchEvent(const Event& event) {
  DCHECK(!event.event_name.empty());
  events_[event.event_name] = base::WrapUnique(event.DeepCopy());
}

}  // namespace extensions
