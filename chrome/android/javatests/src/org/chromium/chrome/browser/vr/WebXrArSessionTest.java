// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.vr;

import static org.chromium.chrome.browser.vr.WebXrArTestFramework.PAGE_LOAD_TIMEOUT_S;
import static org.chromium.chrome.browser.vr.WebXrArTestFramework.POLL_TIMEOUT_LONG_MS;

import android.os.Build;
import android.support.test.InstrumentationRegistry;
import android.support.test.filters.MediumTest;

import org.junit.After;
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
import org.chromium.chrome.browser.ChromeSwitches;
import org.chromium.chrome.browser.vr.rules.XrActivityRestriction;
import org.chromium.chrome.browser.vr.util.XrTestRuleUtils;
import org.chromium.chrome.test.ChromeActivityTestRule;
import org.chromium.chrome.test.ChromeJUnit4RunnerDelegate;
import org.chromium.net.test.EmbeddedTestServer;

import java.util.List;
import java.util.concurrent.Callable;

/**
 * End-to-end tests for testing WebXR for AR's requestSession behavior.
 */
@RunWith(ParameterizedRunner.class)
@UseRunnerDelegate(ChromeJUnit4RunnerDelegate.class)
@CommandLineFlags.
Add({ChromeSwitches.DISABLE_FIRST_RUN_EXPERIENCE, "enable-features=WebXR,WebXRHitTest"})
@MinAndroidSdkLevel(Build.VERSION_CODES.N) // WebXR for AR is only supported on N+
public class WebXrArSessionTest {
    @ClassParameter
    private static List<ParameterSet> sClassParams =
            XrTestRuleUtils.generateDefaultTestRuleParameters();
    @Rule
    public RuleChain mRuleChain;

    private ChromeActivityTestRule mTestRule;
    private WebXrArTestFramework mWebXrArTestFramework;
    private EmbeddedTestServer mServer;

    private boolean mShouldCreateServer;

    public WebXrArSessionTest(Callable<ChromeActivityTestRule> callable) throws Exception {
        mTestRule = callable.call();
        mRuleChain = XrTestRuleUtils.wrapRuleInXrActivityRestrictionRule(mTestRule);
    }

    @Before
    public void setUp() throws Exception {
        mWebXrArTestFramework = new WebXrArTestFramework(mTestRule);
        // WebappActivityTestRule automatically creates a test server, and creating multiple causes
        // it to crash hitting a DCHECK. So, only handle the server ourselves if whatever test rule
        // we're using doesn't create one itself.
        mServer = mTestRule.getTestServer();
        if (mServer == null) {
            mShouldCreateServer = true;
            mServer = EmbeddedTestServer.createAndStartServer(InstrumentationRegistry.getContext());
        }
    }

    @After
    public void tearDown() throws Exception {
        if (mServer != null && mShouldCreateServer) {
            mServer.stopAndDestroyServer();
        }
    }

    /**
     * Tests that a session request for AR succeeds.
     */
    @Test
    @MediumTest
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testArRequestSessionSucceeds() throws InterruptedException {
        mWebXrArTestFramework.loadUrlAndAwaitInitialization(
                mServer.getURL(WebXrArTestFramework.getEmbeddedServerPathForHtmlTestFile(
                        "test_ar_request_session_succeeds")),
                PAGE_LOAD_TIMEOUT_S);
        mWebXrArTestFramework.enterSessionWithUserGestureOrFail();
        mWebXrArTestFramework.assertNoJavaScriptErrors();
    }

    /**
     * Tests that repeatedly starting and stopping AR sessions does not cause any unexpected
     * behavior. Regression test for https://crbug.com/837894.
     */
    @Test
    @MediumTest
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testRepeatedArSessionsSucceed() throws InterruptedException {
        mWebXrArTestFramework.loadUrlAndAwaitInitialization(
                mServer.getURL(WebXrArTestFramework.getEmbeddedServerPathForHtmlTestFile(
                        "test_ar_request_session_succeeds")),
                PAGE_LOAD_TIMEOUT_S);
        for (int i = 0; i < 2; i++) {
            mWebXrArTestFramework.enterSessionWithUserGestureOrFail();
            mWebXrArTestFramework.endSession();
        }
        mWebXrArTestFramework.assertNoJavaScriptErrors();
    }

    /**
     * Tests that repeated calls to requestSession on the same page only prompts the user for
     * camera permissions once.
     */
    @Test
    @MediumTest
    @XrActivityRestriction({XrActivityRestriction.SupportedActivity.ALL})
    public void testRepeatedArSessionsOnlyPromptPermissionsOnce() throws InterruptedException {
        mWebXrArTestFramework.loadUrlAndAwaitInitialization(
                mServer.getURL(WebXrArTestFramework.getEmbeddedServerPathForHtmlTestFile(
                        "test_ar_request_session_succeeds")),
                PAGE_LOAD_TIMEOUT_S);
        Assert.assertTrue("First AR session request did not trigger permission prompt",
                mWebXrArTestFramework.arSessionRequestWouldTriggerPermissionPrompt());
        mWebXrArTestFramework.enterSessionWithUserGestureOrFail();
        mWebXrArTestFramework.endSession();
        // Manually run through the same steps as enterArSessionOrFail so that we don't trigger
        // its automatic permission acceptance.
        Assert.assertFalse("Second AR session request triggered permission prompt",
                mWebXrArTestFramework.arSessionRequestWouldTriggerPermissionPrompt());
        mWebXrArTestFramework.enterSessionWithUserGesture();
        mWebXrArTestFramework.pollJavaScriptBooleanOrFail(
                "sessionInfos[sessionTypes.AR].currentSession != null", POLL_TIMEOUT_LONG_MS);
    }
}