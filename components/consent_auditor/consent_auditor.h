// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef COMPONENTS_CONSENT_AUDITOR_CONSENT_AUDITOR_H_
#define COMPONENTS_CONSENT_AUDITOR_CONSENT_AUDITOR_H_

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "base/memory/weak_ptr.h"
#include "components/keyed_service/core/keyed_service.h"
#include "components/sync/model/model_type_sync_bridge.h"

namespace consent_auditor {

// Feature for which a consent moment is to be recorded.
//
// This enum is used in histograms. Entries should not be renumbered and
// numeric values should never be reused.
//
// A Java counterpart will be generated for this enum.
// GENERATED_JAVA_ENUM_PACKAGE: org.chromium.chrome.browser.consent_auditor
// GENERATED_JAVA_CLASS_NAME_OVERRIDE: ConsentAuditorFeature
enum class Feature {
  CHROME_SYNC = 0,
  PLAY_STORE = 1,
  BACKUP_AND_RESTORE = 2,
  GOOGLE_LOCATION_SERVICE = 3,
  CHROME_UNIFIED_CONSENT = 4,

  FEATURE_LAST = CHROME_UNIFIED_CONSENT
};

// Whether a consent is given or not given.
//
// A Java counterpart will be generated for this enum.
// GENERATED_JAVA_ENUM_PACKAGE: org.chromium.chrome.browser.consent_auditor
enum class ConsentStatus { NOT_GIVEN, GIVEN };

// TODO(vitaliii): Delete user-event-related code once USER_CONSENTS type is
// fully launched.
class ConsentAuditor : public KeyedService {
 public:
  ConsentAuditor();
  ~ConsentAuditor() override;

  // Records a consent for |feature| for the signed-in GAIA account with
  // the ID |account_id| (as defined in AccountInfo).
  // Consent text consisted of strings with |consent_grd_ids|, and the UI
  // element the user clicked had the ID |confirmation_grd_id|.
  // Whether the consent was GIVEN or NOT_GIVEN is passed as |status|.
  //
  // DEPRECATED
  // TODO(markusheintz): Make this method private once all clients have been
  // migrated to the new API.
  virtual void RecordGaiaConsent(const std::string& account_id,
                                 Feature feature,
                                 const std::vector<int>& description_grd_ids,
                                 int confirmation_grd_id,
                                 ConsentStatus status) = 0;

  // Records the ARC Play |consent| for the signed-in GAIA account with the ID
  // |account_id| (as defined in AccountInfo).
  virtual void RecordArcPlayConsent(
      const std::string& account_id,
      const sync_pb::UserConsentTypes::ArcPlayTermsOfServiceConsent&
          consent) = 0;

  // Records the ARC Google Location Service |consent| for the signed-in GAIA
  // account with the ID |account_id| (as defined in AccountInfo).
  virtual void RecordArcGoogleLocationServiceConsent(
      const std::string& account_id,
      const sync_pb::UserConsentTypes::ArcGoogleLocationServiceConsent&
          consent) = 0;

  // Records the ARC Backup and Restore |consent| for the signed-in GAIA
  // account with the ID |account_id| (as defined in AccountInfo).
  virtual void RecordArcBackupAndRestoreConsent(
      const std::string& account_id,
      const sync_pb::UserConsentTypes::ArcBackupAndRestoreConsent& consent) = 0;

  // Records the Sync |consent| for the signed-in GAIA account with the ID
  // |account_id| (as defined in AccountInfo).
  virtual void RecordSyncConsent(
      const std::string& account_id,
      const sync_pb::UserConsentTypes::SyncConsent& consent) = 0;

  // Records the Chrome Unified |consent| for the signed-in GAIA account with
  // the ID |accounts_id| (as defined in Account Info).
  virtual void RecordUnifiedConsent(
      const std::string& account_id,
      const sync_pb::UserConsentTypes::UnifiedConsent& consent) = 0;

  // Records that the user consented to a |feature|. The user was presented with
  // |description_text| and accepted it by interacting |confirmation_text|
  // (e.g. clicking on a button; empty if not applicable).
  // Returns true if successful.
  virtual void RecordLocalConsent(const std::string& feature,
                                  const std::string& description_text,
                                  const std::string& confirmation_text) = 0;

  // Returns the underlying Sync integration point.
  virtual base::WeakPtr<syncer::ModelTypeControllerDelegate>
  GetControllerDelegateOnUIThread() = 0;

 private:
  DISALLOW_COPY_AND_ASSIGN(ConsentAuditor);
};

}  // namespace consent_auditor

#endif  // COMPONENTS_CONSENT_AUDITOR_CONSENT_AUDITOR_H_
