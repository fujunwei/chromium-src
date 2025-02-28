// Copyright 2015 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package org.chromium.chromecast.shell;

import android.annotation.SuppressLint;
import android.content.Context;
import android.media.AudioFocusRequest;
import android.media.AudioManager;
import android.os.Build;

import org.chromium.base.Log;
import org.chromium.base.VisibleForTesting;
import org.chromium.chromecast.base.Controller;
import org.chromium.chromecast.base.Observable;
import org.chromium.chromecast.base.Unit;

/**
 * Wrapper for Cast code to use a single AudioManager instance.
 * Muting and unmuting streams must be invoke on the same AudioManager instance.
 */
public class CastAudioManager {
    private static final String TAG = "CastAudioManager";
    // TODO(sanfin): This class should encapsulate SDK-dependent implementation details of
    // android.media.AudioManager.
    private static CastAudioManager sInstance = null;

    public static CastAudioManager getAudioManager(Context context) {
        if (sInstance == null) {
            sInstance = new CastAudioManager(
                    (AudioManager) context.getSystemService(Context.AUDIO_SERVICE));
        }
        return sInstance;
    }

    private final AudioManager mAudioManager;

    @VisibleForTesting
    CastAudioManager(AudioManager audioManager) {
        mAudioManager = audioManager;
    }

    /**
     * Requests audio focus whenever the given Observable is activated.
     *
     * Returns an Observable that is activated whenever the audio focus is granted.
     *
     * TODO(sanfin): Distinguish between transient, ducking, and full audio focus losses.
     */
    public Observable<Unit> requestAudioFocusWhen(
            Observable<?> event, int streamType, int durationHint) {
        Controller<Unit> audioFocusState = new Controller<>();
        event.watch(() -> {
            AudioManager.OnAudioFocusChangeListener listener = (int focusChange) -> {
                switch (focusChange) {
                    case AudioManager.AUDIOFOCUS_GAIN:
                        audioFocusState.set(Unit.unit());
                        return;
                    default:
                        audioFocusState.reset();
                        return;
                }
            };
            // Request audio focus when the source event is activated.
            if (requestAudioFocus(listener, streamType, durationHint)
                    != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                Log.e(TAG, "Failed to get audio focus");
            }
            // Abandon audio focus when the source event is deactivated.
            return () -> {
                if (abandonAudioFocus(listener) != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                    Log.e(TAG, "Failed to abandon audio focus");
                }
                audioFocusState.reset();
            };
        });
        return audioFocusState;
    }

    // Only called on Lollipop and below, in an Activity's onPause() event.
    // On Lollipop and below, setStreamMute() calls are cumulative and per-application, and if
    // Activities don't unmute the streams that they mute, the stream remains muted to other
    // applications, which are unable to unmute the stream themselves. Therefore, when an Activity
    // is paused, it must unmute any streams it had muted.
    // More context in b/19964892 and b/22204758.
    @SuppressWarnings("deprecation")
    public void releaseStreamMuteIfNecessary(int streamType) {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.LOLLIPOP) {
            // On L, if we try to unmute a stream that is not muted, a warning Toast appears.
            // Check the stream mute state to determine whether to unmute.
            boolean isMuted = false;
            try {
                // isStreamMute() was only made public in M, but it can be accessed through
                // reflection in L.
                isMuted = (Boolean) mAudioManager.getClass()
                                  .getMethod("isStreamMute", int.class)
                                  .invoke(mAudioManager, streamType);
            } catch (Exception e) {
                Log.e(TAG, "Can not call AudioManager.isStreamMute().", e);
            }

            if (isMuted) {
                // Note: this is a no-op on fixed-volume devices.
                mAudioManager.setStreamMute(streamType, false);
            }
        }
    }

    @SuppressLint("NewApi")
    public int requestAudioFocus(AudioFocusRequest focusRequest) {
        return mAudioManager.requestAudioFocus(focusRequest);
    }

    @SuppressWarnings("deprecation")
    public int requestAudioFocus(
            AudioManager.OnAudioFocusChangeListener l, int streamType, int durationHint) {
        return mAudioManager.requestAudioFocus(l, streamType, durationHint);
    }

    @SuppressLint("NewApi")
    public int abandonAudioFocusRequest(AudioFocusRequest focusRequest) {
        return mAudioManager.abandonAudioFocusRequest(focusRequest);
    }

    @SuppressWarnings("deprecation")
    public int abandonAudioFocus(AudioManager.OnAudioFocusChangeListener l) {
        return mAudioManager.abandonAudioFocus(l);
    }

    public int getStreamMaxVolume(int streamType) {
        return mAudioManager.getStreamMaxVolume(streamType);
    }

    // TODO(sanfin): Do not expose this. All needed AudioManager methods can be adapted with
    // CastAudioManager.
    public AudioManager getInternal() {
        return mAudioManager;
    }
}
