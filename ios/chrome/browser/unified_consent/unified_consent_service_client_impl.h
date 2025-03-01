// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef IOS_CHROME_BROWSER_UNIFIED_CONSENT_UNIFIED_CONSENT_SERVICE_CLIENT_IMPL_H_
#define IOS_CHROME_BROWSER_UNIFIED_CONSENT_UNIFIED_CONSENT_SERVICE_CLIENT_IMPL_H_

#include "base/macros.h"
#include "components/unified_consent/unified_consent_service_client.h"

class PrefService;

// iOS implementation for UnifiedConsentServiceClient.
class UnifiedConsentServiceClientImpl
    : public unified_consent::UnifiedConsentServiceClient {
 public:
  explicit UnifiedConsentServiceClientImpl(PrefService* pref_service);
  ~UnifiedConsentServiceClientImpl() override = default;

  void SetAlternateErrorPagesEnabled(bool enabled) override;
  void SetMetricsReportingEnabled(bool enabled) override;
  void SetSearchSuggestEnabled(bool enabled) override;
  void SetSafeBrowsingEnabled(bool enabled) override;
  void SetSafeBrowsingExtendedReportingEnabled(bool enabled) override;
  void SetNetworkPredictionEnabled(bool enabled) override;
  void SetSpellCheckEnabled(bool enabled) override;

 private:
  PrefService* pref_service_;

  DISALLOW_COPY_AND_ASSIGN(UnifiedConsentServiceClientImpl);
};

#endif  // IOS_CHROME_BROWSER_UNIFIED_CONSENT_UNIFIED_CONSENT_SERVICE_CLIENT_IMPL_H_
