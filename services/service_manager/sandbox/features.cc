// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/service_manager/sandbox/features.h"

#include "build/build_config.h"

namespace service_manager {
namespace features {

#if defined(OS_WIN)
// Enables Windows audio service sandbox.
// (Only causes an effect when feature kAudioServiceOutOfProcess is enabled.)
const base::Feature kAudioServiceWindowsSandbox{
    "AudioServiceWindowsSandbox", base::FEATURE_DISABLED_BY_DEFAULT};

// Enables Windows network service sandbox lockdown.
// (Only causes an effect when feature kNetworkService is enabled.)
const base::Feature kNetworkServiceWindowsSandbox{
    "NetworkServiceWindowsSandbox", base::FEATURE_DISABLED_BY_DEFAULT};

// Emergency "off switch" for new Windows sandbox security mitigation,
// sandbox::MITIGATION_EXTENSION_POINT_DISABLE.
const base::Feature kWinSboxDisableExtensionPoints{
    "WinSboxDisableExtensionPoint", base::FEATURE_ENABLED_BY_DEFAULT};
#endif  // defined(OS_WIN)

}  // namespace features
}  // namespace service_manager
