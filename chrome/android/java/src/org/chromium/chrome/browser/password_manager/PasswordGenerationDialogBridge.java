// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chrome.browser.password_manager;

import org.chromium.base.annotations.CalledByNative;
import org.chromium.chrome.browser.ChromeActivity;
import org.chromium.chrome.browser.password_manager.PasswordGenerationDialogCoordinator.SaveExplanationText;
import org.chromium.ui.base.WindowAndroid;

/**
 * JNI call glue between native password generation and Java objects.
 */
public class PasswordGenerationDialogBridge {
    private long mNativePasswordGenerationDialogViewAndroid;
    private final PasswordGenerationDialogCoordinator mPasswordGenerationDialog;
    // TODO(ioanap): Get the generated password from the model once editing is in place.
    private String mGeneratedPassword;

    private PasswordGenerationDialogBridge(
            WindowAndroid windowAndroid, long nativePasswordGenerationDialogViewAndroid) {
        mNativePasswordGenerationDialogViewAndroid = nativePasswordGenerationDialogViewAndroid;
        ChromeActivity activity = (ChromeActivity) windowAndroid.getActivity().get();
        mPasswordGenerationDialog = new PasswordGenerationDialogCoordinator(activity);
    }

    @CalledByNative
    public static PasswordGenerationDialogBridge create(
            WindowAndroid windowAndroid, long nativeDialog) {
        return new PasswordGenerationDialogBridge(windowAndroid, nativeDialog);
    }

    @CalledByNative
    public void showDialog(String generatedPassword, String explanationString, int linkRangeStart,
            int linkRangeEnd) {
        mGeneratedPassword = generatedPassword;
        mPasswordGenerationDialog.showDialog(generatedPassword,
                new SaveExplanationText(explanationString, linkRangeStart, linkRangeEnd,
                        (view) -> onSavedPasswordsLinkClicked()),
                this::onPasswordAcceptedOrRejected);
    }

    @CalledByNative
    private void destroy() {
        mNativePasswordGenerationDialogViewAndroid = 0;
        mPasswordGenerationDialog.dismissDialog();
    }

    private void onPasswordAcceptedOrRejected(boolean accepted) {
        if (mNativePasswordGenerationDialogViewAndroid == 0) return;

        if (accepted) {
            nativePasswordAccepted(mNativePasswordGenerationDialogViewAndroid, mGeneratedPassword);
        } else {
            nativePasswordRejected(mNativePasswordGenerationDialogViewAndroid);
        }
        mPasswordGenerationDialog.dismissDialog();
    }

    private void onSavedPasswordsLinkClicked() {
        if (mNativePasswordGenerationDialogViewAndroid == 0) return;
        nativeOnSavedPasswordsLinkClicked(mNativePasswordGenerationDialogViewAndroid);
    }

    private native void nativePasswordAccepted(
            long nativePasswordGenerationDialogViewAndroid, String generatedPassword);
    private native void nativePasswordRejected(long nativePasswordGenerationDialogViewAndroid);
    private native void nativeOnSavedPasswordsLinkClicked(
            long nativePasswordGenerationDialogViewAndroid);
}
