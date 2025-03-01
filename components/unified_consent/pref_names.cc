// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/unified_consent/pref_names.h"

namespace unified_consent {
namespace prefs {

// Boolean that is true when the user opted into unified consent.
const char kUnifiedConsentGiven[] = "unified_consent_given";

// Integer indicating the migration state of unified consent, defined in
// unified_consent::MigrationState.
const char kUnifiedConsentMigrationState[] = "unified_consent.migration_state";

const char kUrlKeyedAnonymizedDataCollectionEnabled[] =
    "url_keyed_anonymized_data_collection.enabled";

}  // namespace prefs
}  // namespace unified_consent
