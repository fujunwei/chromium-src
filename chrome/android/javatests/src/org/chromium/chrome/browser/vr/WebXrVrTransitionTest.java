// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.vr;

import static org.chromium.chrome.browser.vr.XrTestFramework.PAGE_LOAD_TIMEOUT_S;
import static org.chromium.chrome.browser.vr.XrTestFramework.POLL_CHECK_INTERVAL_LONG_MS;
import static org.chromium.chrome.browser.vr.XrTestFramework.POLL_CHECK_INTERVAL_SHORT_MS;
import static org.chromium.chrome.browser.vr.XrTestFramework.POLL_TIMEOUT_LONG_MS;
import static org.chromium.chrome.browser.vr.XrTestFramework.POLL_TIMEOUT_SHORT_MS;
import static org.chromium.chrome.test.util.ChromeRestriction.RESTRICTION_TYPE_DON_ENABLED;
import static org.chromium.chrome.test.util.ChromeRestriction.RESTRICTION_TYPE_VIEWER_DAYDREAM;

import android.annotation.TargetApi;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;
import android.os.SystemClock;
import android.support.test.InstrumentationRegistry;
import android.support.test.filters.LargeTest;
import android.support.test.filters.MediumTest;
import android.support.test.uiautomator.UiDevice;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.RuleChain;
import org.junit.runner.RunWith;

import org.chromium.base.test.params.ParameterAnnotations.ClassParameter;
import org.chromium.base.test.params.ParameterAnnotations.UseRunnerDelegate;
import org.chromium.base.test.params.ParameterSet;
import org.chromium.base.test.params.ParameterizedRunner;
import org.chromium.base.test.util.CommandLineFlags;
import org.chromium.base.test.util.MinAndroidSdkLevel;
import org.chromium.base.test.util.Restriction;
import org.chromium.base.test.util.RetryOnFailure;
import org.chromium.chrome.browser.ChromeActivity;
import org.chromium.chrome.browser.ChromeSwitches;
import org.chromium.chrome.browser.vr.rules.XrActivityRestriction;
import org.chromium.chrome.browser.vr.util.NfcSimUtils;
import org.chromium.chrome.browser.vr.util.VrTestRuleUtils;
import org.chromium.chrome.browser.vr.util.VrTransitionUtils;
import org.chromium.chrome.test.ChromeActivityTestRule;
import org.chromium.chrome.test.ChromeJUnit4RunnerDelegate;
import org.chromium.content.browser.test.util.CriteriaHelper;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * End-to-end tests for transitioning between WebVR and WebXR's magic window and
 * presentation modes.
 */
@RunWith(ParameterizedRunner.class)
@UseRunnerDelegate(ChromeJUnit4RunnerDelegate.class)
@CommandLineFlags.Add({ChromeSwitches.DISABLE_FIRST_RUN_EXPERIENCE, "enable-webvr"})
@MinAndroidSdkLevel(Build.VERSION_CODES.KITKAT) // WebVR and WebXR are only supported on K+
@TargetApi(Build.VERSION_CODES.KITKAT) // Necessary to allow taking screenshots with UiAutomation
public class WebXrVrTransitionTest {
    @ClassParameter
    private static List<ParameterSet> sClassParams =
            VrTestRuleUtils.generateDefaultTestRuleParameters();
    @Rule
    public RuleChain mRuleChain;

    private ChromeActivityTestRule mTestRule;
    private WebXrVrTestFramework mWebXrVrTestFramework;
    private WebVrTestFramework mWebVrTestFramework;

    public WebXrVrTransitionTest(Callable<ChromeActivityTestRule> callable) throws Exception {
        mTestRule = callable.call();
        mRuleChain = VrTestRuleUtils.wrapRuleInXrActivityRestrictionRule(mTestRule);
    }

    @Before
    public void setUp() throws Exception {
        mWebXrVrTestFramework = new WebXrVrTestFramework(mTestRule);
        mWebVrTestFramework = new WebVrTestFramework(mTestRule);
    }

    /**
     * Tests that a successful requestPresent call actually enters VR
     */
    @Test
    @MediumTest
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testRequestPresentEntersVr() throws InterruptedException {
        testPresentationEntryImpl(
                WebVrTestFramework.getFileUrlForHtmlTestFile("generic_webvr_page"),
                mWebVrTestFramework);
    }

    /**
     * Tests that a successful request for an immersive session actually enters VR.
     */
    @Test
    @MediumTest
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
            public void testRequestSessionEntersVr() throws InterruptedException {
        testPresentationEntryImpl(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile("generic_webxr_page"),
                mWebXrVrTestFramework);
    }

    private void testPresentationEntryImpl(String url, WebXrVrTestFramework framework)
            throws InterruptedException {
        framework.loadUrlAndAwaitInitialization(url, PAGE_LOAD_TIMEOUT_S);
        framework.enterSessionWithUserGestureOrFail();
        Assert.assertTrue("Browser did not enter VR", VrShellDelegate.isInVr());

        // Initial Pixel Test - Verify that the Canvas is blue.
        // The Canvas is set to blue while presenting.
        final UiDevice uiDevice =
                UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());

        CriteriaHelper.pollInstrumentationThread(
                ()
                        -> {
                    Bitmap screenshot = InstrumentationRegistry.getInstrumentation()
                                                .getUiAutomation()
                                                .takeScreenshot();

                    if (screenshot != null) {
                        // Calculate center of eye coordinates.
                        int height = uiDevice.getDisplayHeight() / 2;
                        int width = uiDevice.getDisplayWidth() / 4;

                        // Verify screen is blue.
                        int pixel = screenshot.getPixel(width, height);
                        // Workaround for the immersive mode popup sometimes being rendered over
                        // the screen on K, which causes the pure blue to be darkened to (0, 0,
                        // 127).
                        // TODO(https://crbug.com/819021): Only check pure blue.
                        return pixel == Color.BLUE || pixel == Color.rgb(0, 0, 127);
                    }
                    return false;
                },
                "Immersive session started, but browser not visibly in VR", POLL_TIMEOUT_LONG_MS,
                POLL_CHECK_INTERVAL_LONG_MS);
    }

    /**
     * Tests that WebVR is not exposed if the flag is not on and the page does
     * not have an origin trial token.
     */
    @Test
    @MediumTest
    @CommandLineFlags.Remove({"enable-webvr"})
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testWebVrDisabledWithoutFlagSet() throws InterruptedException {
        // TODO(bsheedy): Remove this test once WebVR is on by default without
        // requiring an origin trial.
        apiDisabledWithoutFlagSetImpl(WebVrTestFramework.getFileUrlForHtmlTestFile(
                                              "test_webvr_disabled_without_flag_set"),
                mWebVrTestFramework);
    }

    /**
     * Tests that WebXR is not exposed if the flag is not on and the page does
     * not have an origin trial token.
     */
    @Test
    @MediumTest
    @CommandLineFlags.Remove({"enable-webvr"})
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testWebXrDisabledWithoutFlagSet() throws InterruptedException {
        // TODO(bsheedy): Remove this test once WebXR is on by default without
        // requiring an origin trial.
        apiDisabledWithoutFlagSetImpl(WebXrVrTestFramework.getFileUrlForHtmlTestFile(
                                              "test_webxr_disabled_without_flag_set"),
                mWebXrVrTestFramework);
    }

    private void apiDisabledWithoutFlagSetImpl(String url, WebXrVrTestFramework framework)
            throws InterruptedException {
        framework.loadUrlAndAwaitInitialization(url, PAGE_LOAD_TIMEOUT_S);
        framework.waitOnJavaScriptStep();
        framework.endTest();
    }

    /**
     * Tests that scanning the Daydream View NFC tag on supported devices fires the
     * vrdisplayactivate event and the event allows presentation without a user gesture.
     */
    @Test
    @LargeTest
    @Restriction(RESTRICTION_TYPE_VIEWER_DAYDREAM)
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testNfcFiresVrdisplayactivate() throws InterruptedException {
        mWebVrTestFramework.loadUrlAndAwaitInitialization(
                WebVrTestFramework.getFileUrlForHtmlTestFile("test_nfc_fires_vrdisplayactivate"),
                PAGE_LOAD_TIMEOUT_S);
        mWebVrTestFramework.runJavaScriptOrFail("addListener()", POLL_TIMEOUT_LONG_MS);
        NfcSimUtils.simNfcScanUntilVrEntry(mTestRule.getActivity());
        mWebVrTestFramework.waitOnJavaScriptStep();
        mWebVrTestFramework.endTest();
        // VrCore has a 2000 ms debounce timeout on NFC scans. When run multiple times in different
        // activities, it is possible for a latter test to be run in the 2 seconds after the
        // previous test's NFC scan, causing it to fail flakily. So, wait 2 seconds to ensure that
        // can't happen.
        SystemClock.sleep(2000);
    }

    /**
     * Tests that the requestPresent promise doesn't resolve if the DON flow is
     * not completed.
     */
    @Test
    @MediumTest
    @Restriction({RESTRICTION_TYPE_VIEWER_DAYDREAM, RESTRICTION_TYPE_DON_ENABLED})
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testPresentationPromiseUnresolvedDuringDon() throws InterruptedException {
        presentationPromiseUnresolvedDuringDonImpl(
                WebVrTestFramework.getFileUrlForHtmlTestFile(
                        "test_presentation_promise_unresolved_during_don"),
                mWebVrTestFramework);
    }

    /**
     * Tests that the immersive session promise doesn't resolve if the DON flow is
     * not completed.
     */
    @Test
    @MediumTest
    @Restriction({RESTRICTION_TYPE_VIEWER_DAYDREAM, RESTRICTION_TYPE_DON_ENABLED})
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
            public void testPresentationPromiseUnresolvedDuringDon_WebXr()
            throws InterruptedException {
        presentationPromiseUnresolvedDuringDonImpl(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile(
                        "webxr_test_presentation_promise_unresolved_during_don"),
                mWebXrVrTestFramework);
    }

    private void presentationPromiseUnresolvedDuringDonImpl(
            String url, WebXrVrTestFramework framework) throws InterruptedException {
        framework.loadUrlAndAwaitInitialization(url, PAGE_LOAD_TIMEOUT_S);
        framework.enterSessionWithUserGestureAndWait();
        framework.endTest();
    }

    /**
     * Tests that the requestPresent promise is rejected if the DON flow is canceled.
     */
    @Test
    @MediumTest
    @Restriction({RESTRICTION_TYPE_VIEWER_DAYDREAM, RESTRICTION_TYPE_DON_ENABLED})
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testPresentationPromiseRejectedIfDonCanceled() throws InterruptedException {
        presentationPromiseRejectedIfDonCanceledImpl(
                WebVrTestFramework.getFileUrlForHtmlTestFile(
                        "test_presentation_promise_rejected_if_don_canceled"),
                mWebVrTestFramework);
    }

    /**
     * Tests that the immersive session promise is rejected if the DON flow is canceled.
     */
    @Test
    @MediumTest
    @Restriction({RESTRICTION_TYPE_VIEWER_DAYDREAM, RESTRICTION_TYPE_DON_ENABLED})
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
            public void testPresentationPromiseRejectedIfDonCanceled_WebXr()
            throws InterruptedException {
        presentationPromiseRejectedIfDonCanceledImpl(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile(
                        "webxr_test_presentation_promise_rejected_if_don_canceled"),
                mWebXrVrTestFramework);
    }

    private void presentationPromiseRejectedIfDonCanceledImpl(
            String url, WebXrVrTestFramework framework) throws InterruptedException {
        framework.loadUrlAndAwaitInitialization(url, PAGE_LOAD_TIMEOUT_S);
        final UiDevice uiDevice =
                UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
        framework.enterSessionWithUserGesture();
        // Wait until the DON flow appears to be triggered
        // TODO(bsheedy): Make this less hacky if there's ever an explicit way to check if the
        // DON flow is currently active https://crbug.com/758296
        CriteriaHelper.pollUiThread(() -> {
            return uiDevice.getCurrentPackageName().equals("com.google.vr.vrcore");
        }, "DON flow did not start", POLL_TIMEOUT_LONG_MS, POLL_CHECK_INTERVAL_SHORT_MS);
        uiDevice.pressBack();
        framework.waitOnJavaScriptStep();
        framework.endTest();
    }

    /**
     * Tests that the omnibox reappears after exiting VR.
     */
    @Test
    @MediumTest
    public void testControlsVisibleAfterExitingVr() throws InterruptedException {
        controlsVisibleAfterExitingVrImpl(
                WebVrTestFramework.getFileUrlForHtmlTestFile("generic_webvr_page"),
                mWebVrTestFramework);
    }

    /**
     * Tests that the omnibox reappears after exiting an immersive session.
     */
    @Test
    @MediumTest
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            public void testControlsVisibleAfterExitingVr_WebXr() throws InterruptedException {
        controlsVisibleAfterExitingVrImpl(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile("generic_webxr_page"),
                mWebXrVrTestFramework);
    }

    private void controlsVisibleAfterExitingVrImpl(String url, final WebXrVrTestFramework framework)
            throws InterruptedException {
        framework.loadUrlAndAwaitInitialization(url, PAGE_LOAD_TIMEOUT_S);
        framework.enterSessionWithUserGestureOrFail();
        VrTransitionUtils.forceExitVr();
        // The hiding of the controls may only propagate after VR has exited, so give it a chance
        // to propagate. In the worst case this test will erroneously pass, but should never
        // erroneously fail, and should only be flaky if omnibox showing is broken.
        Thread.sleep(100);
        CriteriaHelper.pollUiThread(
                ()
                        -> {
                    ChromeActivity activity = framework.getRule().getActivity();
                    return activity.getFullscreenManager().getBrowserControlHiddenRatio() == 0.0;
                },
                "Browser controls did not unhide after exiting VR", POLL_TIMEOUT_SHORT_MS,
                POLL_CHECK_INTERVAL_SHORT_MS);
    }

    /**
     * Tests that window.requestAnimationFrame stops firing while in WebVR presentation, but resumes
     * afterwards.
     */
    @Test
    @MediumTest
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    @RetryOnFailure
    public void testWindowRafStopsFiringWhilePresenting() throws InterruptedException {
        windowRafStopsFiringWhilePresentingImpl(
                WebVrTestFramework.getFileUrlForHtmlTestFile(
                        "test_window_raf_stops_firing_while_presenting"),
                mWebVrTestFramework);
    }

    /**
     * Tests that window.requestAnimationFrame stops firing while in a WebXR immersive session, but
     * resumes afterwards.
     */
    @Test
    @MediumTest
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
            public void testWindowRafStopsFiringWhilePresenting_WebXr()
            throws InterruptedException {
        windowRafStopsFiringWhilePresentingImpl(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile(
                        "webxr_test_window_raf_stops_firing_during_immersive_session"),
                mWebXrVrTestFramework);
    }

    private void windowRafStopsFiringWhilePresentingImpl(String url, WebXrVrTestFramework framework)
            throws InterruptedException {
        framework.loadUrlAndAwaitInitialization(url, PAGE_LOAD_TIMEOUT_S);
        framework.executeStepAndWait("stepVerifyBeforePresent()");
        // Pausing of window.rAF is done asynchronously, so wait until that's done.
        final CountDownLatch vsyncPausedLatch = new CountDownLatch(1);
        TestVrShellDelegate.getInstance().setVrShellOnVSyncPausedCallback(
                () -> { vsyncPausedLatch.countDown(); });
        framework.enterSessionWithUserGestureOrFail();
        vsyncPausedLatch.await(POLL_TIMEOUT_SHORT_MS, TimeUnit.MILLISECONDS);
        framework.executeStepAndWait("stepVerifyDuringPresent()");
        VrTransitionUtils.forceExitVr();
        framework.executeStepAndWait("stepVerifyAfterPresent()");
        framework.endTest();
    }

    /**
     * Tests renderer crashes while in WebVR presentation stay in VR.
     */
    @Test
    @MediumTest
    @Restriction(RESTRICTION_TYPE_VIEWER_DAYDREAM)
    public void testRendererKilledInWebVrStaysInVr()
            throws IllegalArgumentException, InterruptedException, TimeoutException {
        rendererKilledInVrStaysInVrImpl(
                WebVrTestFramework.getFileUrlForHtmlTestFile("generic_webvr_page"),
                mWebVrTestFramework);
    }

    /**
     * Tests renderer crashes while in WebXR presentation stay in VR.
     */
    @Test
    @MediumTest
    @Restriction(RESTRICTION_TYPE_VIEWER_DAYDREAM)
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            public void testRendererKilledInWebXrStaysInVr()
            throws IllegalArgumentException, InterruptedException, TimeoutException {
        rendererKilledInVrStaysInVrImpl(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile("generic_webxr_page"),
                mWebXrVrTestFramework);
    }

    private void rendererKilledInVrStaysInVrImpl(String url, WebXrVrTestFramework framework)
            throws InterruptedException {
        framework.loadUrlAndAwaitInitialization(url, PAGE_LOAD_TIMEOUT_S);
        framework.enterSessionWithUserGestureOrFail();
        framework.simulateRendererKilled();
        Assert.assertTrue("Browser did not enter VR", VrShellDelegate.isInVr());
    }

    /**
     * Tests that window.rAF continues to fire when we have a non-immersive session.
     */
    @Test
    @MediumTest
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
            public void testWindowRafFiresDuringNonImmersiveSession() throws InterruptedException {
        mWebXrVrTestFramework.loadUrlAndAwaitInitialization(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile(
                        "test_window_raf_fires_during_non_immersive_session"),
                PAGE_LOAD_TIMEOUT_S);
        mWebXrVrTestFramework.waitOnJavaScriptStep();
        mWebXrVrTestFramework.endTest();
    }

    /**
     * Tests that non-immersive sessions stop receiving rAFs during an immersive session, but resume
     * once the immersive session ends.
     */
    @Test
    @MediumTest
    @CommandLineFlags
            .Remove({"enable-webvr"})
            @CommandLineFlags.Add({"enable-features=WebXR"})
            @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
            public void testNonImmersiveStopsDuringImmersive() throws InterruptedException {
        mWebXrVrTestFramework.loadUrlAndAwaitInitialization(
                WebXrVrTestFramework.getFileUrlForHtmlTestFile(
                        "test_non_immersive_stops_during_immersive"),
                PAGE_LOAD_TIMEOUT_S);
        mWebXrVrTestFramework.executeStepAndWait("stepBeforeImmersive()");
        mWebXrVrTestFramework.enterSessionWithUserGestureOrFail();
        mWebXrVrTestFramework.executeStepAndWait("stepDuringImmersive()");
        VrTransitionUtils.forceExitVr();
        mWebXrVrTestFramework.executeStepAndWait("stepAfterImmersive()");
        mWebXrVrTestFramework.endTest();
    }
}
