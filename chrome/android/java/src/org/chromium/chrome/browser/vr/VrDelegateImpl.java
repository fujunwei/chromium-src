// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.vr;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;

import org.chromium.chrome.browser.ChromeActivity;

/**
 * {@link VrDelegate} and {@link VrIntentDelegate} implementation if the VR module is available.
 * Forwards calls to VR classes that implement them (mostly {@link VrShellDelegate} and {@link
 * VrIntentUtils}).
 */
/* package */ class VrDelegateImpl implements VrDelegate, VrIntentDelegate {
    @Override
    public void forceExitVrImmediately() {
        VrShellDelegate.forceExitVrImmediately();
    }

    @Override
    public boolean onActivityResultWithNative(int requestCode, int resultCode) {
        return VrShellDelegate.onActivityResultWithNative(requestCode, resultCode);
    }

    @Override
    public void onNativeLibraryAvailable() {
        VrShellDelegate.onNativeLibraryAvailable();
    }

    @Override
    public boolean isInVr() {
        return VrShellDelegate.isInVr();
    }

    @Override
    public boolean canLaunch2DIntents() {
        return VrShellDelegate.canLaunch2DIntents();
    }

    @Override
    public boolean onBackPressed() {
        return VrShellDelegate.onBackPressed();
    }

    @Override
    public boolean enterVrIfNecessary() {
        return VrShellDelegate.enterVrIfNecessary();
    }

    @Override
    public void maybeRegisterVrEntryHook(final ChromeActivity activity) {
        VrShellDelegate.maybeRegisterVrEntryHook(activity);
    }

    @Override
    public void maybeUnregisterVrEntryHook() {
        VrShellDelegate.maybeUnregisterVrEntryHook();
    }

    @Override
    public void onMultiWindowModeChanged(boolean isInMultiWindowMode) {
        VrShellDelegate.onMultiWindowModeChanged(isInMultiWindowMode);
    }

    @Override
    public void requestToExitVrForSearchEnginePromoDialog(
            OnExitVrRequestListener listener, Activity activity) {
        VrShellDelegate.requestToExitVrForSearchEnginePromoDialog(listener, activity);
    }

    @Override
    public void requestToExitVr(OnExitVrRequestListener listener) {
        VrShellDelegate.requestToExitVr(listener);
    }

    @Override
    public void requestToExitVr(OnExitVrRequestListener listener, @UiUnsupportedMode int reason) {
        VrShellDelegate.requestToExitVr(listener, reason);
    }

    @Override
    public void requestToExitVrAndRunOnSuccess(Runnable onSuccess) {
        VrShellDelegate.requestToExitVrAndRunOnSuccess(onSuccess);
    }

    @Override
    public void requestToExitVrAndRunOnSuccess(Runnable onSuccess, @UiUnsupportedMode int reason) {
        VrShellDelegate.requestToExitVrAndRunOnSuccess(onSuccess, reason);
    }

    @Override
    public void onActivityShown(ChromeActivity activity) {
        VrShellDelegate.onActivityShown(activity);
    }

    @Override
    public void onActivityHidden(ChromeActivity activity) {
        VrShellDelegate.onActivityHidden(activity);
    }

    @Override
    public boolean onDensityChanged(int oldDpi, int newDpi) {
        return VrShellDelegate.onDensityChanged(oldDpi, newDpi);
    }

    @Override
    public void rawTopContentOffsetChanged(float topContentOffset) {
        VrShellDelegate.rawTopContentOffsetChanged(topContentOffset);
    }

    @Override
    public void onNewIntentWithNative(ChromeActivity activity, Intent intent) {
        VrShellDelegate.onNewIntentWithNative(activity, intent);
    }

    @Override
    public void maybeHandleVrIntentPreNative(ChromeActivity activity, Intent intent) {
        VrShellDelegate.maybeHandleVrIntentPreNative(activity, intent);
    }

    @Override
    public void setVrModeEnabled(Activity activity, boolean enabled) {
        VrShellDelegate.setVrModeEnabled(activity, enabled);
    }

    @Override
    public boolean bootsToVr() {
        return VrShellDelegate.bootsToVr();
    }

    @Override
    public boolean isDaydreamReadyDevice() {
        return VrShellDelegate.isDaydreamReadyDevice();
    }

    @Override
    public boolean isDaydreamCurrentViewer() {
        return VrShellDelegate.isDaydreamCurrentViewer();
    }

    @Override
    public boolean isVrIntent(Intent intent) {
        return VrIntentUtils.isVrIntent(intent);
    }

    @Override
    public boolean isLaunchingIntoVr(Activity activity, Intent intent) {
        return VrIntentUtils.isLaunchingIntoVr(activity, intent);
    }

    @Override
    public Intent setupVrFreIntent(Context context, Intent freIntent) {
        return VrIntentUtils.setupVrFreIntent(context, freIntent);
    }

    @Override
    public Bundle getVrIntentOptions(Context context) {
        return VrIntentUtils.getVrIntentOptions(context);
    }
}
