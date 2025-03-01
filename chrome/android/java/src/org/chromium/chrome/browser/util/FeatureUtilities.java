// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.util;

import android.annotation.SuppressLint;
import android.annotation.TargetApi;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.os.Build;
import android.os.Bundle;
import android.os.UserManager;
import android.speech.RecognizerIntent;
import android.text.TextUtils;

import org.chromium.base.CommandLine;
import org.chromium.base.ContextUtils;
import org.chromium.base.StrictModeContext;
import org.chromium.base.SysUtils;
import org.chromium.base.ThreadUtils;
import org.chromium.base.VisibleForTesting;
import org.chromium.base.annotations.CalledByNative;
import org.chromium.base.library_loader.LibraryLoader;
import org.chromium.chrome.browser.ChromeFeatureList;
import org.chromium.chrome.browser.ChromeSwitches;
import org.chromium.chrome.browser.firstrun.FirstRunUtils;
import org.chromium.chrome.browser.locale.LocaleManager;
import org.chromium.chrome.browser.partnercustomizations.PartnerBrowserCustomizations;
import org.chromium.chrome.browser.preferences.ChromePreferenceManager;
import org.chromium.chrome.browser.tabmodel.DocumentModeAssassin;
import org.chromium.chrome.browser.toolbar.ToolbarLayout;
import org.chromium.components.signin.AccountManagerFacade;
import org.chromium.components.variations.VariationsAssociatedData;
import org.chromium.ui.base.DeviceFormFactor;

import java.util.List;

/**
 * A utility {@code class} meant to help determine whether or not certain features are supported by
 * this device.
 */
public class FeatureUtilities {
    private static final String TAG = "FeatureUtilities";
    private static final Integer CONTEXTUAL_SUGGESTIONS_TOOLBAR_MIN_DP = 320;

    private static Boolean sHasGoogleAccountAuthenticator;
    private static Boolean sHasRecognitionIntentHandler;
    private static String sChromeHomeSwipeLogicType;

    private static Boolean sIsSoleEnabled;
    private static Boolean sIsChromeModernDesignEnabled;
    private static Boolean sIsHomePageButtonForceEnabled;
    private static Boolean sIsHomepageTileEnabled;
    private static Boolean sIsNewTabPageButtonEnabled;
    private static Boolean sIsBottomToolbarEnabled;

    private static final String NTP_BUTTON_TRIAL_NAME = "NewTabPage";
    private static final String NTP_BUTTON_VARIANT_PARAM_NAME = "variation";

    /**
     * Determines whether or not the {@link RecognizerIntent#ACTION_WEB_SEARCH} {@link Intent}
     * is handled by any {@link android.app.Activity}s in the system.  The result will be cached for
     * future calls.  Passing {@code false} to {@code useCachedValue} will force it to re-query any
     * {@link android.app.Activity}s that can process the {@link Intent}.
     * @param context        The {@link Context} to use to check to see if the {@link Intent} will
     *                       be handled.
     * @param useCachedValue Whether or not to use the cached value from a previous result.
     * @return {@code true} if recognition is supported.  {@code false} otherwise.
     */
    public static boolean isRecognitionIntentPresent(Context context, boolean useCachedValue) {
        ThreadUtils.assertOnUiThread();
        if (sHasRecognitionIntentHandler == null || !useCachedValue) {
            PackageManager pm = context.getPackageManager();
            List<ResolveInfo> activities = pm.queryIntentActivities(
                    new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH), 0);
            sHasRecognitionIntentHandler = activities.size() > 0;
        }

        return sHasRecognitionIntentHandler;
    }

    /**
     * Determines whether or not the user has a Google account (so we can sync) or can add one.
     * @param context The {@link Context} that we should check accounts under.
     * @return Whether or not sync is allowed on this device.
     */
    public static boolean canAllowSync(Context context) {
        return (hasGoogleAccountAuthenticator(context) && hasSyncPermissions(context))
                || hasGoogleAccounts(context);
    }

    @VisibleForTesting
    static boolean hasGoogleAccountAuthenticator(Context context) {
        if (sHasGoogleAccountAuthenticator == null) {
            AccountManagerFacade accountHelper = AccountManagerFacade.get();
            sHasGoogleAccountAuthenticator = accountHelper.hasGoogleAccountAuthenticator();
        }
        return sHasGoogleAccountAuthenticator;
    }

    @VisibleForTesting
    static boolean hasGoogleAccounts(Context context) {
        return AccountManagerFacade.get().hasGoogleAccounts();
    }

    @SuppressLint("InlinedApi")
    @TargetApi(Build.VERSION_CODES.JELLY_BEAN_MR2)
    private static boolean hasSyncPermissions(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.JELLY_BEAN_MR2) return true;

        UserManager manager = (UserManager) context.getSystemService(Context.USER_SERVICE);
        Bundle userRestrictions = manager.getUserRestrictions();
        return !userRestrictions.getBoolean(UserManager.DISALLOW_MODIFY_ACCOUNTS, false);
    }

    /**
     * Check whether Chrome should be running on document mode.
     * @param context The context to use for checking configuration.
     * @return Whether Chrome should be running on document mode.
     */
    public static boolean isDocumentMode(Context context) {
        return isDocumentModeEligible(context) && !DocumentModeAssassin.isOptedOutOfDocumentMode();
    }

    /**
     * Whether the device could possibly run in Document mode (may return true even if the document
     * mode is turned off).
     *
     * This function can't be changed to return false (even if document mode is deleted) because we
     * need to know whether a user needs to be migrated away.
     *
     * @param context The context to use for checking configuration.
     * @return Whether the device could possibly run in Document mode.
     */
    public static boolean isDocumentModeEligible(Context context) {
        return Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP
                && !DeviceFormFactor.isTablet();
    }

    /**
     * Records the current custom tab visibility state with native-side feature utilities.
     * @param visible Whether a custom tab is visible.
     */
    public static void setCustomTabVisible(boolean visible) {
        nativeSetCustomTabVisible(visible);
    }

    /**
     * Records whether the activity is in multi-window mode with native-side feature utilities.
     * @param isInMultiWindowMode Whether the activity is in Android N multi-window mode.
     */
    public static void setIsInMultiWindowMode(boolean isInMultiWindowMode) {
        nativeSetIsInMultiWindowMode(isInMultiWindowMode);
    }

    /**
     * Caches flags that must take effect on startup but are set via native code.
     */
    public static void cacheNativeFlags() {
        cacheSoleEnabled();
        cacheCommandLineOnNonRootedEnabled();
        FirstRunUtils.cacheFirstRunPrefs();
        cacheChromeModernDesignEnabled();
        cacheHomePageButtonForceEnabled();
        cacheHomepageTileEnabled();
        cacheNewTabPageButtonEnabledAndMaybeVariant();
        cacheBottomToolbarEnabled();

        // Propagate DONT_PREFETCH_LIBRARIES feature value to LibraryLoader. This can't
        // be done in LibraryLoader itself because it lives in //base and can't depend
        // on ChromeFeatureList.
        LibraryLoader.setDontPrefetchLibrariesOnNextRuns(
                ChromeFeatureList.isEnabled(ChromeFeatureList.DONT_PREFETCH_LIBRARIES));
    }

    /**
     * @return True if tab model merging for Android N+ is enabled.
     */
    public static boolean isTabModelMergingEnabled() {
        if (CommandLine.getInstance().hasSwitch(ChromeSwitches.DISABLE_TAB_MERGING_FOR_TESTING)) {
            return false;
        }
        return Build.VERSION.SDK_INT > Build.VERSION_CODES.M;
    }

    /**
     * Cache whether or not modern design is enabled so on next startup, the value can be made
     * available immediately.
     */
    public static void cacheChromeModernDesignEnabled() {
        boolean isModernEnabled =
                ChromeFeatureList.isEnabled(ChromeFeatureList.CHROME_MODERN_DESIGN);

        ChromePreferenceManager manager = ChromePreferenceManager.getInstance();
        manager.setChromeModernDesignEnabled(isModernEnabled);
    }

    /**
     * Cache whether or not the home page button is force enabled so on next startup, the value can
     * be made available immediately.
     */
    public static void cacheHomePageButtonForceEnabled() {
        if (PartnerBrowserCustomizations.isHomepageProviderAvailableAndEnabled()) return;
        ChromePreferenceManager.getInstance().setHomePageButtonForceEnabled(
                ChromeFeatureList.isEnabled(ChromeFeatureList.HOME_PAGE_BUTTON_FORCE_ENABLED));
    }

    /**
     * @return Whether or not the home page button is force enabled.
     */
    public static boolean isHomePageButtonForceEnabled() {
        if (sIsHomePageButtonForceEnabled == null) {
            ChromePreferenceManager prefManager = ChromePreferenceManager.getInstance();

            try (StrictModeContext unused = StrictModeContext.allowDiskReads()) {
                sIsHomePageButtonForceEnabled = prefManager.isHomePageButtonForceEnabled();
            }
        }
        return sIsHomePageButtonForceEnabled;
    }

    /**
     * Resets whether the home page button is enabled for tests. After this is called, the next
     * call to #isHomePageButtonForceEnabled() will retrieve the value from shared preferences.
     */
    public static void resetHomePageButtonForceEnabledForTests() {
        sIsHomePageButtonForceEnabled = null;
    }

    /**
     * Cache whether or not the new tab page button is enabled so on next startup, the value can
     * be made available immediately.
     */
    public static void cacheHomepageTileEnabled() {
        ChromePreferenceManager.getInstance().setHomepageTileEnabled(
                ChromeFeatureList.isEnabled(ChromeFeatureList.HOMEPAGE_TILE));
    }

    /**
     * @return Whether or not the new tab page button is enabled.
     */
    public static boolean isHomepageTileEnabled() {
        if (sIsHomepageTileEnabled == null) {
            ChromePreferenceManager prefManager = ChromePreferenceManager.getInstance();

            try (StrictModeContext unused = StrictModeContext.allowDiskReads()) {
                sIsHomepageTileEnabled = prefManager.isHomepageTileEnabled();
            }
        }
        return sIsHomepageTileEnabled;
    }

    /**
     * Cache whether or not the new tab page button is enabled and, if it is, the button's variant
     * as well so that on next startup, both values can be made available immediately.
     */
    public static void cacheNewTabPageButtonEnabledAndMaybeVariant() {
        boolean isNTPButtonEnabled = ChromeFeatureList.isEnabled(ChromeFeatureList.NTP_BUTTON);
        ChromePreferenceManager.getInstance().setNewTabPageButtonEnabled(isNTPButtonEnabled);
        if (isNTPButtonEnabled) {
            String iconVariant = getNTPButtonVariant();
            if (TextUtils.isEmpty(iconVariant)) iconVariant = ToolbarLayout.NTP_BUTTON_HOME_VARIANT;
            ChromePreferenceManager.getInstance().setNewTabPageButtonVariant(iconVariant);
        }
    }

    /**
     * Gets the new tab page button variant from variations associated data.
     * Native must be initialized before this method is called.
     * @return The new tab page button variant.
     */
    public static String getNTPButtonVariant() {
        return VariationsAssociatedData.getVariationParamValue(
                NTP_BUTTON_TRIAL_NAME, NTP_BUTTON_VARIANT_PARAM_NAME);
    }

    /**
     * @return Whether or not the new tab page button is enabled.
     */
    public static boolean isNewTabPageButtonEnabled() {
        if (sIsNewTabPageButtonEnabled == null) {
            ChromePreferenceManager prefManager = ChromePreferenceManager.getInstance();

            try (StrictModeContext unused = StrictModeContext.allowDiskReads()) {
                sIsNewTabPageButtonEnabled = prefManager.isNewTabPageButtonEnabled();
            }
        }
        return sIsNewTabPageButtonEnabled;
    }

    /**
     * Cache whether or not the bottom toolbar is enabled so on next startup, the value can
     * be made available immediately.
     */
    public static void cacheBottomToolbarEnabled() {
        ChromePreferenceManager.getInstance().setBottomToolbarEnabled(
                ChromeFeatureList.isEnabled(ChromeFeatureList.CHROME_DUET));
    }

    /**
     * @return Whether or not the bottom toolbar is enabled.
     */
    public static boolean isBottomToolbarEnabled() {
        if (sIsBottomToolbarEnabled == null) {
            ChromePreferenceManager prefManager = ChromePreferenceManager.getInstance();

            try (StrictModeContext unused = StrictModeContext.allowDiskReads()) {
                sIsBottomToolbarEnabled = prefManager.isBottomToolbarEnabled()
                        && !DeviceFormFactor.isNonMultiDisplayContextOnTablet(
                                   ContextUtils.getApplicationContext());
            }
        }
        return sIsBottomToolbarEnabled;
    }

    /**
     * Cache whether or not command line is enabled on non-rooted devices.
     */
    private static void cacheCommandLineOnNonRootedEnabled() {
        boolean isCommandLineOnNonRootedEnabled =
                ChromeFeatureList.isEnabled(ChromeFeatureList.COMMAND_LINE_ON_NON_ROOTED);
        ChromePreferenceManager manager = ChromePreferenceManager.getInstance();
        manager.setCommandLineOnNonRootedEnabled(isCommandLineOnNonRootedEnabled);
    }

    /**
     * @return Whether or not the download progress infobar is enabled.
     */
    public static boolean isDownloadProgressInfoBarEnabled() {
        return ChromeFeatureList.isEnabled(ChromeFeatureList.DOWNLOAD_PROGRESS_INFOBAR);
    }

    /**
     * @return The type of swipe logic used for opening the bottom sheet in Chrome Home. Null is
     *         returned if the command line is not initialized or no experiment is specified.
     */
    public static String getChromeHomeSwipeLogicType() {
        if (sChromeHomeSwipeLogicType == null) {
            CommandLine instance = CommandLine.getInstance();
            sChromeHomeSwipeLogicType =
                    instance.getSwitchValue(ChromeSwitches.CHROME_HOME_SWIPE_LOGIC);
        }

        return sChromeHomeSwipeLogicType;
    }

    /**
     * Cache whether or not Sole integration is enabled.
     */
    public static void cacheSoleEnabled() {
        boolean featureEnabled = ChromeFeatureList.isEnabled(ChromeFeatureList.SOLE_INTEGRATION);
        ChromePreferenceManager prefManager = ChromePreferenceManager.getInstance();
        boolean prefEnabled = prefManager.isSoleEnabled();
        if (featureEnabled == prefEnabled) return;

        prefManager.setSoleEnabled(featureEnabled);
    }

    /**
     * @return Whether or not Sole integration is enabled.
     */
    public static boolean isSoleEnabled() {
        if (sIsSoleEnabled == null) {
            ChromePreferenceManager prefManager = ChromePreferenceManager.getInstance();

            // Allow disk access for preferences while Sole is in experimentation.
            try (StrictModeContext unused = StrictModeContext.allowDiskReads()) {
                sIsSoleEnabled = prefManager.isSoleEnabled();
            }
        }
        return sIsSoleEnabled;
    }

    /**
     * Resets whether Chrome modern design is enabled for tests. After this is called, the next
     * call to #isChromeModernDesignEnabled() will retrieve the value from shared preferences.
     */
    public static void resetChromeModernDesignEnabledForTests() {
        sIsChromeModernDesignEnabled = null;
    }

    /**
     * @return Whether Chrome modern design is enabled. This returns true if Chrome Home is enabled.
     */
    @CalledByNative
    public static boolean isChromeModernDesignEnabled() {
        if (sIsChromeModernDesignEnabled == null) {
            ChromePreferenceManager prefManager = ChromePreferenceManager.getInstance();
            try (StrictModeContext unused = StrictModeContext.allowDiskReads()) {
                sIsChromeModernDesignEnabled = prefManager.isChromeModernDesignEnabled();
            }
        }

        return sIsChromeModernDesignEnabled;
    }

    /**
     * @param activityContext The context for the containing activity.
     * @return Whether contextual suggestions are enabled.
     */
    public static boolean areContextualSuggestionsEnabled(Context activityContext) {
        int smallestScreenWidth =
                activityContext.getResources().getConfiguration().smallestScreenWidthDp;
        return !DeviceFormFactor.isNonMultiDisplayContextOnTablet(activityContext)
                && !LocaleManager.getInstance().needToCheckForSearchEnginePromo()
                && isChromeModernDesignEnabled()
                && (ChromeFeatureList.isEnabled(
                            ChromeFeatureList.CONTEXTUAL_SUGGESTIONS_BOTTOM_SHEET)
                    || (smallestScreenWidth >= CONTEXTUAL_SUGGESTIONS_TOOLBAR_MIN_DP
                        && ChromeFeatureList.isEnabled(
                                ChromeFeatureList.CONTEXTUAL_SUGGESTIONS_BUTTON)));
    }

    /**
     * @return Whether this device is running Android Go. This is assumed when we're running Android
     * O or later and we're on a low-end device.
     */
    public static boolean isAndroidGo() {
        return SysUtils.isLowEndDevice()
                && android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O;
    }

    private static native void nativeSetCustomTabVisible(boolean visible);
    private static native void nativeSetIsInMultiWindowMode(boolean isInMultiWindowMode);
}
