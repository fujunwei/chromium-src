// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "extensions/renderer/extension_throttle_manager.h"

#include <utility>

#include "base/logging.h"
#include "base/metrics/field_trial.h"
#include "base/metrics/histogram.h"
#include "base/strings/string_util.h"
#include "extensions/common/constants.h"
#include "extensions/renderer/extension_url_loader_throttle.h"
#include "net/base/url_util.h"
#include "third_party/blink/public/platform/web_url.h"
#include "third_party/blink/public/platform/web_url_request.h"

namespace extensions {

const unsigned int ExtensionThrottleManager::kMaximumNumberOfEntries = 1500;
const unsigned int ExtensionThrottleManager::kRequestsBetweenCollecting = 200;

ExtensionThrottleManager::ExtensionThrottleManager()
    : requests_since_last_gc_(0),
      ignore_user_gesture_load_flag_for_tests_(false) {
  url_id_replacements_.ClearPassword();
  url_id_replacements_.ClearUsername();
  url_id_replacements_.ClearQuery();
  url_id_replacements_.ClearRef();
}

ExtensionThrottleManager::~ExtensionThrottleManager() {
  base::AutoLock auto_lock(lock_);
  // Delete all entries.
  url_entries_.clear();
}

std::unique_ptr<content::URLLoaderThrottle>
ExtensionThrottleManager::MaybeCreateURLLoaderThrottle(
    const blink::WebURLRequest& request) {
  if (!request.SiteForCookies().ProtocolIs(extensions::kExtensionScheme))
    return nullptr;
  return std::make_unique<ExtensionURLLoaderThrottle>(this);
}

ExtensionThrottleEntry* ExtensionThrottleManager::RegisterRequestUrl(
    const GURL& url) {
  // Internal function, no locking.

  // Normalize the url.
  std::string url_id = GetIdFromUrl(url);

  // Periodically garbage collect old entries.
  GarbageCollectEntriesIfNecessary();

  // Find the entry in the map or create a new null entry.
  std::unique_ptr<ExtensionThrottleEntry>& entry = url_entries_[url_id];

  // If the entry exists but could be garbage collected at this point, we
  // start with a fresh entry so that we possibly back off a bit less
  // aggressively (i.e. this resets the error count when the entry's URL
  // hasn't been requested in long enough).
  if (entry && entry->IsEntryOutdated())
    entry.reset();

  // Create the entry if needed.
  if (!entry) {
    if (backoff_policy_for_tests_) {
      entry.reset(
          new ExtensionThrottleEntry(url_id, backoff_policy_for_tests_.get(),
                                     ignore_user_gesture_load_flag_for_tests_));
    } else {
      entry.reset(new ExtensionThrottleEntry(
          url_id, ignore_user_gesture_load_flag_for_tests_));
    }

    // We only disable back-off throttling on an entry that we have
    // just constructed.  This is to allow unit tests to explicitly override
    // the entry for localhost URLs.
    if (net::IsLocalhost(url)) {
      // TODO(joi): Once sliding window is separate from back-off throttling,
      // we can simply return a dummy implementation of
      // ExtensionThrottleEntry here that never blocks anything.
      entry->DisableBackoffThrottling();
    }
  }

  return entry.get();
}

bool ExtensionThrottleManager::ShouldRejectRequest(const GURL& request_url,
                                                   int request_load_flags) {
  base::AutoLock auto_lock(lock_);
  return RegisterRequestUrl(request_url)
      ->ShouldRejectRequest(request_load_flags);
}

bool ExtensionThrottleManager::ShouldRejectRedirect(
    const GURL& request_url,
    int request_load_flags,
    const net::RedirectInfo& redirect_info) {
  {
    base::AutoLock auto_lock(lock_);
    const std::string url_id = GetIdFromUrl(request_url);
    ExtensionThrottleEntry* entry = url_entries_[url_id].get();
    DCHECK(entry);
    // TODO(crbug.com/866798) Temporarily checking for null ptr.
    if (entry)
      entry->UpdateWithResponse(redirect_info.status_code);
  }
  return ShouldRejectRequest(redirect_info.new_url, request_load_flags);
}

void ExtensionThrottleManager::WillProcessResponse(
    const GURL& response_url,
    const network::ResourceResponseHead& response_head) {
  if (response_head.network_accessed) {
    base::AutoLock auto_lock(lock_);
    const std::string url_id = GetIdFromUrl(response_url);
    ExtensionThrottleEntry* entry = url_entries_[url_id].get();
    DCHECK(entry);
    // TODO(crbug.com/866798) Temporarily checking for null ptr.
    if (entry)
      entry->UpdateWithResponse(response_head.headers->response_code());
  }
}

void ExtensionThrottleManager::SetBackoffPolicyForTests(
    std::unique_ptr<net::BackoffEntry::Policy> policy) {
  base::AutoLock auto_lock(lock_);
  backoff_policy_for_tests_ = std::move(policy);
}

void ExtensionThrottleManager::OverrideEntryForTests(
    const GURL& url,
    std::unique_ptr<ExtensionThrottleEntry> entry) {
  base::AutoLock auto_lock(lock_);
  // Normalize the url.
  std::string url_id = GetIdFromUrl(url);

  // Periodically garbage collect old entries.
  GarbageCollectEntriesIfNecessary();

  url_entries_[url_id] = std::move(entry);
}

void ExtensionThrottleManager::EraseEntryForTests(const GURL& url) {
  base::AutoLock auto_lock(lock_);
  // Normalize the url.
  std::string url_id = GetIdFromUrl(url);
  url_entries_.erase(url_id);
}

void ExtensionThrottleManager::SetIgnoreUserGestureLoadFlagForTests(
    bool ignore_user_gesture_load_flag_for_tests) {
  base::AutoLock auto_lock(lock_);
  ignore_user_gesture_load_flag_for_tests_ = true;
}

void ExtensionThrottleManager::SetOnline(bool is_online) {
  // When we switch from online to offline or change IP addresses, we
  // clear all back-off history. This is a precaution in case the change in
  // online state now lets us communicate without error with servers that
  // we were previously getting 500 or 503 responses from (perhaps the
  // responses are from a badly-written proxy that should have returned a
  // 502 or 504 because it's upstream connection was down or it had no route
  // to the server).
  // Remove all entries.  Any entries that in-flight requests have a reference
  // to will live until those requests end, and these entries may be
  // inconsistent with new entries for the same URLs, but since what we
  // want is a clean slate for the new connection type, this is OK.
  base::AutoLock auto_lock(lock_);
  url_entries_.clear();
  requests_since_last_gc_ = 0;
}

std::string ExtensionThrottleManager::GetIdFromUrl(const GURL& url) const {
  if (!url.is_valid())
    return url.possibly_invalid_spec();

  GURL id = url.ReplaceComponents(url_id_replacements_);
  return base::ToLowerASCII(id.spec());
}

void ExtensionThrottleManager::GarbageCollectEntriesIfNecessary() {
  requests_since_last_gc_++;
  if (requests_since_last_gc_ < kRequestsBetweenCollecting)
    return;
  requests_since_last_gc_ = 0;

  GarbageCollectEntries();
}

void ExtensionThrottleManager::GarbageCollectEntries() {
  base::EraseIf(url_entries_, [](const auto& entry) {
    return entry.second->IsEntryOutdated();
  });

  // In case something broke we want to make sure not to grow indefinitely.
  while (url_entries_.size() > kMaximumNumberOfEntries) {
    url_entries_.erase(url_entries_.begin());
  }
}

}  // namespace extensions
