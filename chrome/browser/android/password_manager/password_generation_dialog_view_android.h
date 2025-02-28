// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_ANDROID_PASSWORD_MANAGER_PASSWORD_GENERATION_DIALOG_VIEW_ANDROID_H_
#define CHROME_BROWSER_ANDROID_PASSWORD_MANAGER_PASSWORD_GENERATION_DIALOG_VIEW_ANDROID_H_

#include <jni.h>

#include "base/android/scoped_java_ref.h"
#include "base/strings/string16.h"
#include "chrome/browser/password_manager/password_generation_dialog_view_interface.h"

class PasswordAccessoryController;

// Modal dialog displaying a generated password with options to accept or
// reject it. Communicates events to its Java counterpart and passes responses
// back to the |PasswordAccessoryController|.
// TODO(crbug.com/835234): Add a specialized dialog controller.
class PasswordGenerationDialogViewAndroid
    : public PasswordGenerationDialogViewInterface {
 public:
  // Builds the UI for the |controller|
  explicit PasswordGenerationDialogViewAndroid(
      PasswordAccessoryController* controller);

  ~PasswordGenerationDialogViewAndroid() override;

  // Called to show the dialog. |password| is the generated password.
  void Show(base::string16& password) override;

  // Called from Java via JNI.
  void PasswordAccepted(JNIEnv* env,
                        const base::android::JavaParamRef<jobject>& obj,
                        const base::android::JavaParamRef<jstring>& password);

  // Called from Java via JNI.
  void PasswordRejected(JNIEnv* env,
                        const base::android::JavaParamRef<jobject>& obj);

  // Called from Java via JNI.
  void OnSavedPasswordsLinkClicked(
      JNIEnv* env,
      const base::android::JavaParamRef<jobject>& obj);

 private:
  // The controller provides data for this view and owns it.
  PasswordAccessoryController* controller_;

  // The corresponding java object.
  base::android::ScopedJavaGlobalRef<jobject> java_object_;

  DISALLOW_COPY_AND_ASSIGN(PasswordGenerationDialogViewAndroid);
};

#endif  // CHROME_BROWSER_ANDROID_PASSWORD_MANAGER_PASSWORD_GENERATION_DIALOG_VIEW_ANDROID_H_
