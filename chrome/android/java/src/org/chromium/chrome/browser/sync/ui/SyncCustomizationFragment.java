// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.sync.ui;

import android.accounts.Account;
import android.app.DialogFragment;
import android.app.FragmentManager;
import android.app.FragmentTransaction;
import android.content.Context;
import android.content.Intent;
import android.content.res.Resources;
import android.net.Uri;
import android.os.Bundle;
import android.preference.CheckBoxPreference;
import android.preference.Preference;
import android.preference.Preference.OnPreferenceChangeListener;
import android.preference.Preference.OnPreferenceClickListener;
import android.preference.PreferenceFragment;
import android.preference.SwitchPreference;
import android.support.annotation.IntDef;
import android.support.annotation.Nullable;
import android.support.v4.app.FragmentActivity;
import android.text.Spannable;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;

import org.chromium.base.ApiCompatibilityUtils;
import org.chromium.base.BuildInfo;
import org.chromium.base.ContextUtils;
import org.chromium.base.ThreadUtils;
import org.chromium.base.VisibleForTesting;
import org.chromium.base.metrics.RecordHistogram;
import org.chromium.chrome.R;
import org.chromium.chrome.browser.autofill.PersonalDataManager;
import org.chromium.chrome.browser.invalidation.InvalidationController;
import org.chromium.chrome.browser.preferences.ChromeSwitchPreference;
import org.chromium.chrome.browser.preferences.SyncedAccountPreference;
import org.chromium.chrome.browser.profiles.Profile;
import org.chromium.chrome.browser.signin.SigninManager;
import org.chromium.chrome.browser.signin.SigninUtils;
import org.chromium.chrome.browser.sync.GoogleServiceAuthError;
import org.chromium.chrome.browser.sync.ProfileSyncService;
import org.chromium.chrome.browser.sync.SyncAccountSwitcher;
import org.chromium.components.signin.AccountManagerFacade;
import org.chromium.components.signin.ChromeSigninController;
import org.chromium.components.sync.AndroidSyncSettings;
import org.chromium.components.sync.ModelType;
import org.chromium.components.sync.PassphraseType;
import org.chromium.components.sync.ProtocolErrorClientAction;
import org.chromium.components.sync.StopSource;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.HashSet;
import java.util.Set;

/**
 * Settings fragment to customize Sync options (data types, encryption).
 */
public class SyncCustomizationFragment extends PreferenceFragment
        implements PassphraseDialogFragment.Listener, PassphraseCreationDialogFragment.Listener,
                   PassphraseTypeDialogFragment.Listener, OnPreferenceClickListener,
                   OnPreferenceChangeListener, ProfileSyncService.SyncStateChangedListener {
    private static final String TAG = "SyncCustomizationFragment";

    @VisibleForTesting
    public static final String FRAGMENT_ENTER_PASSPHRASE = "enter_password";
    @VisibleForTesting
    public static final String FRAGMENT_CUSTOM_PASSPHRASE = "custom_password";
    @VisibleForTesting
    public static final String FRAGMENT_PASSPHRASE_TYPE = "password_type";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_EVERYTHING = "sync_everything";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_AUTOFILL = "sync_autofill";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_BOOKMARKS = "sync_bookmarks";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_OMNIBOX = "sync_omnibox";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_PASSWORDS = "sync_passwords";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_RECENT_TABS = "sync_recent_tabs";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_SETTINGS = "sync_settings";
    @VisibleForTesting
    public static final String PREFERENCE_PAYMENTS_INTEGRATION = "payments_integration";
    @VisibleForTesting
    public static final String PREFERENCE_ENCRYPTION = "encryption";
    @VisibleForTesting
    public static final String PREF_SYNC_SWITCH = "sync_switch";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_MANAGE_DATA = "sync_manage_data";
    @VisibleForTesting
    public static final String PREFERENCE_SYNC_ACCOUNT_LIST = "synced_account";
    public static final String PREFERENCE_SYNC_ERROR_CARD = "sync_error_card";

    @IntDef({SyncError.NO_ERROR, SyncError.ANDROID_SYNC_DISABLED, SyncError.AUTH_ERROR,
            SyncError.PASSPHRASE_REQUIRED, SyncError.CLIENT_OUT_OF_DATE, SyncError.OTHER_ERRORS})
    @Retention(RetentionPolicy.SOURCE)
    private @interface SyncError {
        int NO_ERROR = -1;
        int ANDROID_SYNC_DISABLED = 0;
        int AUTH_ERROR = 1;
        int PASSPHRASE_REQUIRED = 2;
        int CLIENT_OUT_OF_DATE = 3;
        int OTHER_ERRORS = 128;
    }

    private ChromeSwitchPreference mSyncSwitchPreference;
    private boolean mIsEngineInitialized;
    private boolean mIsPassphraseRequired;

    private static final String DASHBOARD_URL = "https://www.google.com/settings/chrome/sync";

    private SwitchPreference mSyncEverything;
    private CheckBoxPreference mSyncAutofill;
    private CheckBoxPreference mSyncBookmarks;
    private CheckBoxPreference mSyncOmnibox;
    private CheckBoxPreference mSyncPasswords;
    private CheckBoxPreference mSyncRecentTabs;
    private CheckBoxPreference mSyncSettings;
    private CheckBoxPreference mPaymentsIntegration;
    private Preference mSyncEncryption;
    private Preference mManageSyncData;
    private Preference mSyncErrorCard;
    private CheckBoxPreference[] mAllTypes;
    private SyncedAccountPreference mSyncedAccountPreference;

    private ProfileSyncService mProfileSyncService;

    @SyncError
    private int mCurrentSyncError = SyncError.NO_ERROR;

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        mProfileSyncService = ProfileSyncService.get();
        assert mProfileSyncService != null;
        mIsEngineInitialized = mProfileSyncService.isEngineInitialized();
        mIsPassphraseRequired =
                mIsEngineInitialized && mProfileSyncService.isPassphraseRequiredForDecryption();

        getActivity().setTitle(R.string.sign_in_sync);
        addPreferencesFromResource(R.xml.sync_customization_preferences);
        mSyncEverything = (SwitchPreference) findPreference(PREFERENCE_SYNC_EVERYTHING);
        mSyncAutofill = (CheckBoxPreference) findPreference(PREFERENCE_SYNC_AUTOFILL);
        mSyncBookmarks = (CheckBoxPreference) findPreference(PREFERENCE_SYNC_BOOKMARKS);
        mSyncOmnibox = (CheckBoxPreference) findPreference(PREFERENCE_SYNC_OMNIBOX);
        mSyncPasswords = (CheckBoxPreference) findPreference(PREFERENCE_SYNC_PASSWORDS);
        mSyncRecentTabs = (CheckBoxPreference) findPreference(PREFERENCE_SYNC_RECENT_TABS);
        mSyncSettings = (CheckBoxPreference) findPreference(PREFERENCE_SYNC_SETTINGS);
        mPaymentsIntegration = (CheckBoxPreference) findPreference(PREFERENCE_PAYMENTS_INTEGRATION);

        mSyncEncryption = findPreference(PREFERENCE_ENCRYPTION);
        mSyncEncryption.setOnPreferenceClickListener(this);
        mManageSyncData = findPreference(PREFERENCE_SYNC_MANAGE_DATA);
        mManageSyncData.setOnPreferenceClickListener(this);
        mSyncErrorCard = findPreference(PREFERENCE_SYNC_ERROR_CARD);
        mSyncErrorCard.setOnPreferenceClickListener(this);

        mAllTypes = new CheckBoxPreference[] {
                mSyncAutofill, mSyncBookmarks, mSyncOmnibox, mSyncPasswords,
                mSyncRecentTabs, mSyncSettings, mPaymentsIntegration
        };

        mSyncEverything.setOnPreferenceChangeListener(this);
        for (CheckBoxPreference type : mAllTypes) {
            type.setOnPreferenceChangeListener(this);
        }

        mSyncSwitchPreference = (ChromeSwitchPreference) findPreference(PREF_SYNC_SWITCH);
        mSyncSwitchPreference.setOnPreferenceChangeListener((preference, newValue) -> {
            assert canDisableSync();
            if ((boolean) newValue) {
                mProfileSyncService.requestStart();
            } else {
                stopSync();
            }
            // Must be done asynchronously because the switch state isn't updated
            // until after this function exits.
            ThreadUtils.postOnUiThread(this::updateSyncStateFromSwitch);
            return true;
        });

        mSyncedAccountPreference =
                (SyncedAccountPreference) findPreference(PREFERENCE_SYNC_ACCOUNT_LIST);

        // TODO(https://crbug.com/710657): Migrate to SyncCustomizationFragment to
        // extend android.support.v7.preference.Preference and remove this cast.
        FragmentActivity fragmentActivity = (FragmentActivity) getActivity();
        mSyncedAccountPreference.setOnPreferenceChangeListener(
                new SyncAccountSwitcher(fragmentActivity, mSyncedAccountPreference));
    }

    @Override
    public boolean onPreferenceChange(Preference preference, Object newValue) {
        if (preference == mSyncEverything) {
            ThreadUtils.postOnUiThread(this::updateDataTypeState);
            return true;
        }
        if (isSyncTypePreference(preference)) {
            final boolean syncAutofillToggled = preference == mSyncAutofill;
            final boolean preferenceChecked = (boolean) newValue;
            ThreadUtils.postOnUiThread(() -> {
                if (syncAutofillToggled) {
                    // If the user checks the autofill sync checkbox, then enable and check the
                    // payments integration checkbox.
                    //
                    // If the user unchecks the autofill sync checkbox, then disable and uncheck
                    // the payments integration checkbox.
                    mPaymentsIntegration.setEnabled(preferenceChecked);
                    mPaymentsIntegration.setChecked(preferenceChecked);
                }
                maybeDisableSync();
            });
            return true;
        }
        return false;
    }

    /**
     * @return Whether Sync can be disabled.
     */
    private boolean canDisableSync() {
        return !Profile.getLastUsedProfile().isChild();
    }

    private boolean isSyncTypePreference(Preference preference) {
        for (Preference pref : mAllTypes) {
            if (pref == preference) return true;
        }
        return false;
    }

    @Override
    public void onStart() {
        super.onStart();
        // The current account may have been switched on a different screen so ensure the synced
        // account preference displays the correct signed in account.
        mSyncedAccountPreference.update();

        mIsEngineInitialized = mProfileSyncService.isEngineInitialized();
        mIsPassphraseRequired =
                mIsEngineInitialized && mProfileSyncService.isPassphraseRequiredForDecryption();
        // This prevents sync from actually syncing until the dialog is closed.
        mProfileSyncService.setSetupInProgress(true);
        mProfileSyncService.addSyncStateChangedListener(this);
        updateSyncState();
    }

    @Override
    public void onStop() {
        super.onStop();

        mProfileSyncService.removeSyncStateChangedListener(this);
        // If this activity is closing, apply configuration changes and tell sync that
        // the user is done configuring sync.
        if (!getActivity().isChangingConfigurations()) {
            // Only save state if the switch and external state match. If a stop and clear comes
            // while the dialog is open, this will be false and settings won't be saved.
            if (mSyncSwitchPreference.isChecked() && AndroidSyncSettings.isSyncEnabled()) {
                // Save the new data type state.
                configureSyncDataTypes();
                // Inform sync that the user has finished setting up sync at least once.
                mProfileSyncService.setFirstSetupComplete();
            }
            PersonalDataManager.setPaymentsIntegrationEnabled(mPaymentsIntegration.isChecked());
            // Setup is done. This was preventing sync from turning on even if it was enabled.
            // TODO(crbug/557784): This needs to be set only when we think the user is done with
            // setting up. This means: 1) If the user leaves the Sync Settings screen (via back)
            // or, 2) If the user leaves the screen by tapping on "Manage Synced Data"
            mProfileSyncService.setSetupInProgress(false);
        }
    }

    /**
     * Update the state of all settings from sync.
     *
     * This sets the state of the sync switch from external sync state and then calls
     * updateSyncStateFromSwitch, which uses that as its source of truth.
     */
    private void updateSyncState() {
        boolean isSyncEnabled = AndroidSyncSettings.isSyncEnabled();
        mSyncSwitchPreference.setChecked(isSyncEnabled);
        mSyncSwitchPreference.setEnabled(canDisableSync());
        updateSyncStateFromSwitch();
    }

    private void updateSyncAccountsListState() {
        SyncedAccountPreference accountList =
                (SyncedAccountPreference) findPreference(PREFERENCE_SYNC_ACCOUNT_LIST);

        // We remove the the SyncedAccountPreference if there's only 1 account on the device, so
        // it's possible for accountList to be null
        if (accountList != null) {
            Account[] accounts = AccountManagerFacade.get().tryGetGoogleAccounts();
            if (accounts.length <= 1) {
                getPreferenceScreen().removePreference(accountList);
            } else {
                accountList.setEnabled(mSyncSwitchPreference.isChecked());
            }
        }
    }

    /**
     * Update the state of settings using the switch state to determine if sync is enabled.
     */
    private void updateSyncStateFromSwitch() {
        updateSyncEverythingState();
        updateDataTypeState();
        updateEncryptionState();
        updateSyncAccountsListState();
        updateSyncErrorCard();
    }

    /**
     * Update the encryption state.
     *
     * If sync's engine is initialized, the button is enabled and the dialog will present the
     * valid encryption options for the user. Otherwise, any encryption dialogs will be closed
     * and the button will be disabled because the engine is needed in order to know and
     * modify the encryption state.
     */
    private void updateEncryptionState() {
        boolean isSyncEnabled = mSyncSwitchPreference.isChecked();
        boolean isEngineInitialized = mProfileSyncService.isEngineInitialized();
        mSyncEncryption.setEnabled(isSyncEnabled && isEngineInitialized);
        mSyncEncryption.setSummary(null);
        if (!isEngineInitialized) {
            // If sync is not initialized, encryption state is unavailable and can't be changed.
            // Leave the button disabled and the summary empty. Additionally, close the dialogs in
            // case they were open when a stop and clear comes.
            closeDialogIfOpen(FRAGMENT_CUSTOM_PASSPHRASE);
            closeDialogIfOpen(FRAGMENT_ENTER_PASSPHRASE);
            return;
        }
        if (!mProfileSyncService.isPassphraseRequiredForDecryption()) {
            closeDialogIfOpen(FRAGMENT_ENTER_PASSPHRASE);
        }
        if (mProfileSyncService.isPassphraseRequiredForDecryption() && isAdded()) {
            mSyncEncryption.setSummary(
                    errorSummary(getString(R.string.sync_need_passphrase), getActivity()));
        }
    }

    /**
     * Applies a span to the given string to give it an error color.
     */
    private static Spannable errorSummary(String string, Context context) {
        SpannableString summary = new SpannableString(string);
        summary.setSpan(new ForegroundColorSpan(
                ApiCompatibilityUtils.getColor(
                        context.getResources(), R.color.input_underline_error_color)),
                0, summary.length(), 0);
        return summary;
    }

    private void configureSyncDataTypes() {
        if (maybeDisableSync()) return;

        boolean syncEverything = mSyncEverything.isChecked();
        mProfileSyncService.setPreferredDataTypes(syncEverything, getSelectedModelTypes());
        // Update the invalidation listener with the set of types we are enabling.
        InvalidationController invController = InvalidationController.get();
        invController.ensureStartedAndUpdateRegisteredTypes();
    }

    private Set<Integer> getSelectedModelTypes() {
        Set<Integer> types = new HashSet<>();
        if (mSyncAutofill.isChecked()) types.add(ModelType.AUTOFILL);
        if (mSyncBookmarks.isChecked()) types.add(ModelType.BOOKMARKS);
        if (mSyncOmnibox.isChecked()) types.add(ModelType.TYPED_URLS);
        if (mSyncPasswords.isChecked()) types.add(ModelType.PASSWORDS);
        if (mSyncRecentTabs.isChecked()) types.add(ModelType.PROXY_TABS);
        if (mSyncSettings.isChecked()) types.add(ModelType.PREFERENCES);
        return types;
    }

    private void displayPassphraseTypeDialog() {
        FragmentTransaction ft = getFragmentManager().beginTransaction();
        PassphraseTypeDialogFragment dialog = PassphraseTypeDialogFragment.create(
                mProfileSyncService.getPassphraseType(),
                mProfileSyncService.getExplicitPassphraseTime(),
                mProfileSyncService.isEncryptEverythingAllowed());
        dialog.show(ft, FRAGMENT_PASSPHRASE_TYPE);
        dialog.setTargetFragment(this, -1);
    }

    private void displayPassphraseDialog() {
        FragmentTransaction ft = getFragmentManager().beginTransaction();
        PassphraseDialogFragment.newInstance(this).show(ft, FRAGMENT_ENTER_PASSPHRASE);
    }

    private void displayCustomPassphraseDialog() {
        FragmentTransaction ft = getFragmentManager().beginTransaction();
        PassphraseCreationDialogFragment dialog = new PassphraseCreationDialogFragment();
        dialog.setTargetFragment(this, -1);
        dialog.show(ft, FRAGMENT_CUSTOM_PASSPHRASE);
    }

    private void closeDialogIfOpen(String tag) {
        FragmentManager manager = getFragmentManager();
        if (manager == null) {
            // Do nothing if the manager doesn't exist yet; see http://crbug.com/480544.
            return;
        }
        DialogFragment df = (DialogFragment) manager.findFragmentByTag(tag);
        if (df != null) {
            df.dismiss();
        }
    }

    /**
     * @return whether the passphrase successfully decrypted the pending keys.
     */
    private boolean handleDecryption(String passphrase) {
        if (!passphrase.isEmpty() && mProfileSyncService.setDecryptionPassphrase(passphrase)) {
            // PassphraseDialogFragment doesn't handle closing itself, so do it here. This is
            // not done in updateSyncState() because that happens onResume and possibly in other
            // cases where the dialog should stay open.
            closeDialogIfOpen(FRAGMENT_ENTER_PASSPHRASE);
            // Update our configuration UI.
            updateSyncState();
            return true;
        }
        return false;
    }

    /**
     * Callback for PassphraseDialogFragment.Listener
     */
    @Override
    public boolean onPassphraseEntered(String passphrase) {
        if (!mProfileSyncService.isEngineInitialized()) {
            // If the engine was shut down since the dialog was opened, do nothing.
            return false;
        }
        return handleDecryption(passphrase);
    }

    /**
     * Callback for PassphraseDialogFragment.Listener
     */
    @Override
    public void onPassphraseCanceled() {
    }

    /**
     * Callback for PassphraseCreationDialogFragment.Listener
     */
    @Override
    public void onPassphraseCreated(String passphrase) {
        if (!mProfileSyncService.isEngineInitialized()) {
            // If the engine was shut down since the dialog was opened, do nothing.
            return;
        }
        mProfileSyncService.enableEncryptEverything();
        mProfileSyncService.setEncryptionPassphrase(passphrase);
        // Configure the current set of data types - this tells the sync engine to
        // apply our encryption configuration changes.
        configureSyncDataTypes();
        // Re-display our config UI to properly reflect the new state.
        updateSyncState();
    }

    /**
     * Callback for PassphraseTypeDialogFragment.Listener
     */
    @Override
    public void onPassphraseTypeSelected(PassphraseType type) {
        if (!mProfileSyncService.isEngineInitialized()) {
            // If the engine was shut down since the dialog was opened, do nothing.
            return;
        }

        boolean isAllDataEncrypted = mProfileSyncService.isEncryptEverythingEnabled();
        boolean isUsingSecondaryPassphrase = mProfileSyncService.isUsingSecondaryPassphrase();

        // The passphrase type should only ever be selected if the account doesn't have
        // full encryption enabled. Otherwise both options should be disabled.
        assert !isAllDataEncrypted;
        assert !isUsingSecondaryPassphrase;
        displayCustomPassphraseDialog();
    }

    /**
     * Callback for OnPreferenceClickListener
     */
    @Override
    public boolean onPreferenceClick(Preference preference) {
        if (!isResumed()) {
            // This event could come in after onPause if the user clicks back and the preference at
            // roughly the same time. See http://b/5983282
            return false;
        }
        if (preference == mSyncEncryption && mProfileSyncService.isEngineInitialized()) {
            if (mProfileSyncService.isPassphraseRequiredForDecryption()) {
                displayPassphraseDialog();
            } else {
                displayPassphraseTypeDialog();
                return true;
            }
        } else if (preference == mManageSyncData) {
            openDashboardTabInNewActivityStack();
            return true;
        } else if (preference == mSyncErrorCard) {
            onSyncErrorCardClicked();
            return true;
        }
        return false;
    }

    /**
     * Opens the Google Dashboard where the user can control the data stored for the account.
     */
    private void openDashboardTabInNewActivityStack() {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(DASHBOARD_URL));
        intent.setPackage(getActivity().getPackageName());
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
    }

    /**
     * Update the state of the sync everything switch.
     *
     * If sync is on, load the pref from native. Otherwise display sync everything as on but
     * disable the switch.
     */
    private void updateSyncEverythingState() {
        boolean isSyncEnabled = mSyncSwitchPreference.isChecked();
        mSyncEverything.setEnabled(isSyncEnabled);
        mSyncEverything.setChecked(!isSyncEnabled
                || mProfileSyncService.hasKeepEverythingSynced());
    }

    /**
     * Update the data type switch state.
     *
     * If sync is on, load the prefs from native. Otherwise, all data types are disabled and
     * checked. Note that the Password data type will be shown as disabled and unchecked between
     * sync being turned on and the engine initialization completing.
     */
    private void updateDataTypeState() {
        boolean isSyncEnabled = mSyncSwitchPreference.isChecked();
        boolean syncEverything = mSyncEverything.isChecked();
        boolean passwordSyncConfigurable = mProfileSyncService.isEngineInitialized()
                && mProfileSyncService.isCryptographerReady();
        Set<Integer> syncTypes = mProfileSyncService.getPreferredDataTypes();
        boolean syncAutofill = syncTypes.contains(ModelType.AUTOFILL);
        for (CheckBoxPreference pref : mAllTypes) {
            boolean canSyncType = true;
            if (pref == mSyncPasswords) canSyncType = passwordSyncConfigurable;
            if (pref == mPaymentsIntegration) {
                canSyncType = syncAutofill || syncEverything;
            }

            if (!isSyncEnabled) {
                pref.setChecked(true);
            } else if (syncEverything) {
                pref.setChecked(canSyncType);
            }

            pref.setEnabled(isSyncEnabled && !syncEverything && canSyncType);
        }
        if (isSyncEnabled && !syncEverything) {
            mSyncAutofill.setChecked(syncAutofill);
            mSyncBookmarks.setChecked(syncTypes.contains(ModelType.BOOKMARKS));
            mSyncOmnibox.setChecked(syncTypes.contains(ModelType.TYPED_URLS));
            mSyncPasswords.setChecked(passwordSyncConfigurable
                    && syncTypes.contains(ModelType.PASSWORDS));
            mSyncRecentTabs.setChecked(syncTypes.contains(ModelType.PROXY_TABS));
            mSyncSettings.setChecked(syncTypes.contains(ModelType.PREFERENCES));
            mPaymentsIntegration.setChecked(
                    syncAutofill && PersonalDataManager.isPaymentsIntegrationEnabled());
        }
    }

    private void updateSyncErrorCard() {
        mCurrentSyncError = getSyncError();
        if (mCurrentSyncError != SyncError.NO_ERROR) {
            String summary = getSyncErrorHint(mCurrentSyncError);
            mSyncErrorCard.setSummary(summary);
            getPreferenceScreen().addPreference(mSyncErrorCard);
        } else {
            getPreferenceScreen().removePreference(mSyncErrorCard);
        }
    }

    @SyncError
    private int getSyncError() {
        if (!AndroidSyncSettings.isMasterSyncEnabled()) {
            return SyncError.ANDROID_SYNC_DISABLED;
        }

        if (!mSyncSwitchPreference.isChecked()) {
            return SyncError.NO_ERROR;
        }

        if (mProfileSyncService.getAuthError()
                == GoogleServiceAuthError.State.INVALID_GAIA_CREDENTIALS) {
            return SyncError.AUTH_ERROR;
        }

        if (mProfileSyncService.getProtocolErrorClientAction()
                == ProtocolErrorClientAction.UPGRADE_CLIENT) {
            return SyncError.CLIENT_OUT_OF_DATE;
        }

        if (mProfileSyncService.getAuthError() != GoogleServiceAuthError.State.NONE
                || mProfileSyncService.hasUnrecoverableError()) {
            return SyncError.OTHER_ERRORS;
        }

        if (mProfileSyncService.isSyncActive()
                && mProfileSyncService.isPassphraseRequiredForDecryption()) {
            return SyncError.PASSPHRASE_REQUIRED;
        }

        return SyncError.NO_ERROR;
    }

    /**
     * Gets hint message to resolve sync error.
     * @param error The sync error.
     */
    private String getSyncErrorHint(@SyncError int error) {
        Resources res = getActivity().getResources();
        switch (error) {
            case SyncError.ANDROID_SYNC_DISABLED:
                return res.getString(R.string.hint_android_sync_disabled);
            case SyncError.AUTH_ERROR:
                return res.getString(R.string.hint_sync_auth_error);
            case SyncError.CLIENT_OUT_OF_DATE:
                return res.getString(
                        R.string.hint_client_out_of_date, BuildInfo.getInstance().hostPackageLabel);
            case SyncError.OTHER_ERRORS:
                return res.getString(R.string.hint_other_sync_errors);
            case SyncError.PASSPHRASE_REQUIRED:
                return res.getString(R.string.hint_passphrase_required);
            case SyncError.NO_ERROR:
            default:
                return null;
        }
    }

    private void onSyncErrorCardClicked() {
        if (mCurrentSyncError == SyncError.NO_ERROR) {
            return;
        }

        if (mCurrentSyncError == SyncError.ANDROID_SYNC_DISABLED) {
            SigninUtils.openAccountSettingsPage(
                    getActivity(), ChromeSigninController.get().getSignedInAccountName());
            return;
        }

        if (mCurrentSyncError == SyncError.AUTH_ERROR) {
            AccountManagerFacade.get().updateCredentials(
                    ChromeSigninController.get().getSignedInUser(), getActivity(), null);
            return;
        }

        if (mCurrentSyncError == SyncError.CLIENT_OUT_OF_DATE) {
            // Opens the client in play store for update.
            Intent intent = new Intent(Intent.ACTION_VIEW);
            intent.setData(Uri.parse("market://details?id="
                    + ContextUtils.getApplicationContext().getPackageName()));
            startActivity(intent);
            return;
        }

        if (mCurrentSyncError == SyncError.OTHER_ERRORS) {
            final Account account = ChromeSigninController.get().getSignedInUser();
            SigninManager.get().signOut(() -> SigninManager.get().signIn(account, null, null));
            return;
        }

        if (mCurrentSyncError == SyncError.PASSPHRASE_REQUIRED) {
            displayPassphraseDialog();
            return;
        }
    }

    /**
     * Listen to sync state changes.
     *
     * If the user has just turned on sync, this listener is needed in order to enable
     * the encryption settings once the engine has initialized.
     */
    @Override
    public void syncStateChanged() {
        boolean wasSyncInitialized = mIsEngineInitialized;
        boolean wasPassphraseRequired = mIsPassphraseRequired;
        mIsEngineInitialized = mProfileSyncService.isEngineInitialized();
        mIsPassphraseRequired =
                mIsEngineInitialized && mProfileSyncService.isPassphraseRequiredForDecryption();
        if (mIsEngineInitialized != wasSyncInitialized
                || mIsPassphraseRequired != wasPassphraseRequired) {
            // Update all because Password syncability is also affected by the engine.
            updateSyncStateFromSwitch();
        } else {
            updateSyncErrorCard();
        }
    }

    /**
     * Disables Sync if all data types have been disabled.
     *
     * @return true if Sync has been disabled, false otherwise.
     */
    private boolean maybeDisableSync() {
        if (mSyncEverything.isChecked()
                || !getSelectedModelTypes().isEmpty()
                || !canDisableSync()) {
            return false;
        }
        stopSync();
        mSyncSwitchPreference.setChecked(false);
        // setChecked doesn't trigger the callback, so update manually.
        updateSyncStateFromSwitch();
        return true;
    }

    private void stopSync() {
        if (mProfileSyncService.isSyncRequested()) {
            RecordHistogram.recordEnumeratedHistogram("Sync.StopSource",
                    StopSource.CHROME_SYNC_SETTINGS, StopSource.STOP_SOURCE_LIMIT);
            mProfileSyncService.requestStop();
        }
    }
}
