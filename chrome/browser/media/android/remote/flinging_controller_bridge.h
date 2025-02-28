// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_MEDIA_ANDROID_REMOTE_FLINGING_CONTROLLER_BRIDGE_H_
#define CHROME_BROWSER_MEDIA_ANDROID_REMOTE_FLINGING_CONTROLLER_BRIDGE_H_

#include "base/android/scoped_java_ref.h"
#include "base/time/time.h"
#include "media/base/flinging_controller.h"
#include "media/base/media_controller.h"

namespace media_router {

// Allows native code to call into a Java FlingingController.
class FlingingControllerBridge : public media::FlingingController,
                                 public media::MediaController {
 public:
  explicit FlingingControllerBridge(
      base::android::ScopedJavaGlobalRef<jobject> controller);
  ~FlingingControllerBridge() override;

  // FlingingController implementation.
  media::MediaController* GetMediaController() override;
  void AddMediaStatusObserver(media::MediaStatusObserver* observer) override;
  void RemoveMediaStatusObserver(media::MediaStatusObserver* observer) override;
  base::TimeDelta GetApproximateCurrentTime() override;

  // MediaController implementation.
  void Play() override;
  void Pause() override;
  void SetMute(bool mute) override;
  void SetVolume(float volume) override;
  void Seek(base::TimeDelta time) override;

 private:
  // Java MediaControllerBridge instance.
  base::android::ScopedJavaGlobalRef<jobject> j_flinging_controller_bridge_;

  DISALLOW_COPY_AND_ASSIGN(FlingingControllerBridge);
};

}  // namespace media_router

#endif  // CHROME_BROWSER_MEDIA_ANDROID_REMOTE_FLINGING_CONTROLLER_BRIDGE_H_
