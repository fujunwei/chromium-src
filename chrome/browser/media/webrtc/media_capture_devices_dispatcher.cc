// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "chrome/browser/media/webrtc/media_capture_devices_dispatcher.h"

#include "base/command_line.h"
#include "base/logging.h"
#include "base/metrics/field_trial.h"
#include "base/strings/string_number_conversions.h"
#include "base/strings/string_util.h"
#include "base/strings/utf_string_conversions.h"
#include "build/build_config.h"
#include "chrome/browser/media/media_access_handler.h"
#include "chrome/browser/media/webrtc/media_stream_capture_indicator.h"
#include "chrome/browser/media/webrtc/permission_bubble_media_access_handler.h"
#include "chrome/browser/profiles/profile.h"
#include "chrome/browser/ui/browser.h"
#include "chrome/browser/ui/browser_window.h"
#include "chrome/common/chrome_switches.h"
#include "chrome/common/pref_names.h"
#include "components/pref_registry/pref_registry_syncable.h"
#include "components/prefs/pref_service.h"
#include "components/prefs/scoped_user_pref_update.h"
#include "content/public/browser/browser_thread.h"
#include "content/public/browser/media_capture_devices.h"
#include "content/public/browser/notification_source.h"
#include "content/public/browser/render_frame_host.h"
#include "content/public/browser/render_process_host.h"
#include "content/public/browser/web_contents.h"
#include "content/public/common/media_stream_request.h"
#include "extensions/buildflags/buildflags.h"
#include "extensions/common/constants.h"
#include "media/base/media_switches.h"

#if defined(OS_CHROMEOS)
#include "ash/shell.h"
#include "chrome/browser/media/chromeos_login_media_access_handler.h"
#include "chrome/browser/media/public_session_media_access_handler.h"
#include "chrome/browser/media/public_session_tab_capture_access_handler.h"
#endif  // defined(OS_CHROMEOS)

#if BUILDFLAG(ENABLE_EXTENSIONS)
#include "chrome/browser/media/extension_media_access_handler.h"
#include "chrome/browser/media/webrtc/desktop_capture_access_handler.h"
#include "chrome/browser/media/webrtc/tab_capture_access_handler.h"
#include "extensions/browser/extension_registry.h"
#include "extensions/common/extension.h"
#include "extensions/common/permissions/permissions_data.h"
#endif

using content::BrowserThread;
using content::MediaCaptureDevices;
using content::MediaStreamDevices;

namespace {

// Finds a device in |devices| that has |device_id|, or NULL if not found.
const content::MediaStreamDevice* FindDeviceWithId(
    const content::MediaStreamDevices& devices,
    const std::string& device_id) {
  content::MediaStreamDevices::const_iterator iter = devices.begin();
  for (; iter != devices.end(); ++iter) {
    if (iter->id == device_id) {
      return &(*iter);
    }
  }
  return NULL;
}

content::WebContents* WebContentsFromIds(int render_process_id,
                                         int render_frame_id) {
  content::WebContents* web_contents =
      content::WebContents::FromRenderFrameHost(
          content::RenderFrameHost::FromID(render_process_id, render_frame_id));
  return web_contents;
}

#if BUILDFLAG(ENABLE_EXTENSIONS)
inline CaptureAccessHandlerBase* ToCaptureAccessHandlerBase(
    MediaAccessHandler* handler) {
  return static_cast<CaptureAccessHandlerBase*>(handler);
}
#endif
}  // namespace

MediaCaptureDevicesDispatcher* MediaCaptureDevicesDispatcher::GetInstance() {
  return base::Singleton<MediaCaptureDevicesDispatcher>::get();
}

MediaCaptureDevicesDispatcher::MediaCaptureDevicesDispatcher()
    : is_device_enumeration_disabled_(false),
      media_stream_capture_indicator_(new MediaStreamCaptureIndicator()) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);

#if BUILDFLAG(ENABLE_EXTENSIONS)
#if defined(OS_CHROMEOS)
  media_access_handlers_.push_back(
      std::make_unique<ChromeOSLoginMediaAccessHandler>());
  // Wrapper around ExtensionMediaAccessHandler used in Public Sessions.
  media_access_handlers_.push_back(
      std::make_unique<PublicSessionMediaAccessHandler>());
#else
  media_access_handlers_.push_back(
      std::make_unique<ExtensionMediaAccessHandler>());
#endif
  media_access_handlers_.push_back(
      std::make_unique<DesktopCaptureAccessHandler>());
#if defined(OS_CHROMEOS)
  // Wrapper around TabCaptureAccessHandler used in Public Sessions.
  media_access_handlers_.push_back(
      std::make_unique<PublicSessionTabCaptureAccessHandler>());
#else
  media_access_handlers_.push_back(std::make_unique<TabCaptureAccessHandler>());
#endif
#endif
  media_access_handlers_.push_back(
      std::make_unique<PermissionBubbleMediaAccessHandler>());
}

MediaCaptureDevicesDispatcher::~MediaCaptureDevicesDispatcher() {}

void MediaCaptureDevicesDispatcher::RegisterProfilePrefs(
    user_prefs::PrefRegistrySyncable* registry) {
  registry->RegisterStringPref(prefs::kDefaultAudioCaptureDevice,
                               std::string());
  registry->RegisterStringPref(prefs::kDefaultVideoCaptureDevice,
                               std::string());
}

bool MediaCaptureDevicesDispatcher::IsOriginForCasting(const GURL& origin) {
  // Whitelisted tab casting extensions.
  return
      // Media Router Dev
      origin.spec() == "chrome-extension://enhhojjnijigcajfphajepfemndkmdlo/" ||
      // Media Router Stable
      origin.spec() == "chrome-extension://pkedcjkdefgpdelpbcmbmeomcjbeemfm/";
}

void MediaCaptureDevicesDispatcher::AddObserver(Observer* observer) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  if (!observers_.HasObserver(observer))
    observers_.AddObserver(observer);
}

void MediaCaptureDevicesDispatcher::RemoveObserver(Observer* observer) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  observers_.RemoveObserver(observer);
}

const MediaStreamDevices&
MediaCaptureDevicesDispatcher::GetAudioCaptureDevices() {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  if (is_device_enumeration_disabled_ || !test_audio_devices_.empty())
    return test_audio_devices_;

  return MediaCaptureDevices::GetInstance()->GetAudioCaptureDevices();
}

const MediaStreamDevices&
MediaCaptureDevicesDispatcher::GetVideoCaptureDevices() {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  if (is_device_enumeration_disabled_ || !test_video_devices_.empty())
    return test_video_devices_;

  return MediaCaptureDevices::GetInstance()->GetVideoCaptureDevices();
}

void MediaCaptureDevicesDispatcher::ProcessMediaAccessRequest(
    content::WebContents* web_contents,
    const content::MediaStreamRequest& request,
    content::MediaResponseCallback callback,
    const extensions::Extension* extension) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);

  for (const auto& handler : media_access_handlers_) {
    if (handler->SupportsStreamType(web_contents, request.video_type,
                                    extension) ||
        handler->SupportsStreamType(web_contents, request.audio_type,
                                    extension)) {
      handler->HandleRequest(web_contents, request, std::move(callback),
                             extension);
      return;
    }
  }
  std::move(callback).Run(content::MediaStreamDevices(),
                          content::MEDIA_DEVICE_NOT_SUPPORTED, nullptr);
}

bool MediaCaptureDevicesDispatcher::CheckMediaAccessPermission(
    content::RenderFrameHost* render_frame_host,
    const GURL& security_origin,
    content::MediaStreamType type) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  return CheckMediaAccessPermission(render_frame_host, security_origin, type,
                                    nullptr);
}

bool MediaCaptureDevicesDispatcher::CheckMediaAccessPermission(
    content::RenderFrameHost* render_frame_host,
    const GURL& security_origin,
    content::MediaStreamType type,
    const extensions::Extension* extension) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  for (const auto& handler : media_access_handlers_) {
    if (handler->SupportsStreamType(
            content::WebContents::FromRenderFrameHost(render_frame_host), type,
            extension)) {
      return handler->CheckMediaAccessPermission(
          render_frame_host, security_origin, type, extension);
    }
  }
  return false;
}

void MediaCaptureDevicesDispatcher::GetDefaultDevicesForProfile(
    Profile* profile,
    bool audio,
    bool video,
    content::MediaStreamDevices* devices) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  DCHECK(audio || video);

  PrefService* prefs = profile->GetPrefs();
  std::string default_device;
  if (audio) {
    default_device = prefs->GetString(prefs::kDefaultAudioCaptureDevice);
    const content::MediaStreamDevice* device =
        GetRequestedAudioDevice(default_device);
    if (!device)
      device = GetFirstAvailableAudioDevice();
    if (device)
      devices->push_back(*device);
  }

  if (video) {
    default_device = prefs->GetString(prefs::kDefaultVideoCaptureDevice);
    const content::MediaStreamDevice* device =
        GetRequestedVideoDevice(default_device);
    if (!device)
      device = GetFirstAvailableVideoDevice();
    if (device)
      devices->push_back(*device);
  }
}

std::string MediaCaptureDevicesDispatcher::GetDefaultDeviceIDForProfile(
    Profile* profile,
    content::MediaStreamType type) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  PrefService* prefs = profile->GetPrefs();
  if (type == content::MEDIA_DEVICE_AUDIO_CAPTURE)
    return prefs->GetString(prefs::kDefaultAudioCaptureDevice);
  else if (type == content::MEDIA_DEVICE_VIDEO_CAPTURE)
    return prefs->GetString(prefs::kDefaultVideoCaptureDevice);
  else
    return std::string();
}

const content::MediaStreamDevice*
MediaCaptureDevicesDispatcher::GetRequestedAudioDevice(
    const std::string& requested_audio_device_id) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  const content::MediaStreamDevices& audio_devices = GetAudioCaptureDevices();
  const content::MediaStreamDevice* const device =
      FindDeviceWithId(audio_devices, requested_audio_device_id);
  return device;
}

const content::MediaStreamDevice*
MediaCaptureDevicesDispatcher::GetFirstAvailableAudioDevice() {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  const content::MediaStreamDevices& audio_devices = GetAudioCaptureDevices();
  if (audio_devices.empty())
    return NULL;
  return &(*audio_devices.begin());
}

const content::MediaStreamDevice*
MediaCaptureDevicesDispatcher::GetRequestedVideoDevice(
    const std::string& requested_video_device_id) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  const content::MediaStreamDevices& video_devices = GetVideoCaptureDevices();
  const content::MediaStreamDevice* const device =
      FindDeviceWithId(video_devices, requested_video_device_id);
  return device;
}

const content::MediaStreamDevice*
MediaCaptureDevicesDispatcher::GetFirstAvailableVideoDevice() {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  const content::MediaStreamDevices& video_devices = GetVideoCaptureDevices();
  if (video_devices.empty())
    return NULL;
  return &(*video_devices.begin());
}

void MediaCaptureDevicesDispatcher::DisableDeviceEnumerationForTesting() {
  is_device_enumeration_disabled_ = true;
}

scoped_refptr<MediaStreamCaptureIndicator>
MediaCaptureDevicesDispatcher::GetMediaStreamCaptureIndicator() {
  return media_stream_capture_indicator_;
}

void MediaCaptureDevicesDispatcher::OnAudioCaptureDevicesChanged() {
  DCHECK_CURRENTLY_ON(BrowserThread::IO);
  BrowserThread::PostTask(
      BrowserThread::UI, FROM_HERE,
      base::BindOnce(
          &MediaCaptureDevicesDispatcher::NotifyAudioDevicesChangedOnUIThread,
          base::Unretained(this)));
}

void MediaCaptureDevicesDispatcher::OnVideoCaptureDevicesChanged() {
  DCHECK_CURRENTLY_ON(BrowserThread::IO);
  BrowserThread::PostTask(
      BrowserThread::UI, FROM_HERE,
      base::BindOnce(
          &MediaCaptureDevicesDispatcher::NotifyVideoDevicesChangedOnUIThread,
          base::Unretained(this)));
}

void MediaCaptureDevicesDispatcher::OnMediaRequestStateChanged(
    int render_process_id,
    int render_frame_id,
    int page_request_id,
    const GURL& security_origin,
    content::MediaStreamType stream_type,
    content::MediaRequestState state) {
  DCHECK_CURRENTLY_ON(BrowserThread::IO);
  BrowserThread::PostTask(
      BrowserThread::UI, FROM_HERE,
      base::BindOnce(
          &MediaCaptureDevicesDispatcher::UpdateMediaRequestStateOnUIThread,
          base::Unretained(this), render_process_id, render_frame_id,
          page_request_id, security_origin, stream_type, state));
}

void MediaCaptureDevicesDispatcher::OnCreatingAudioStream(
    int render_process_id,
    int render_frame_id) {
  // TODO(https://crbug.com/837606): Figure out how to simplify threading here.
  // Currently, this will either always be called on the UI thread, or always
  // on the IO thread, depending on how far along the work to migrate to the
  // audio service has progressed. The rest of the methods of the
  // content::MediaObserver are always called on the IO thread.
  if (BrowserThread::CurrentlyOn(BrowserThread::UI)) {
    OnCreatingAudioStreamOnUIThread(render_process_id, render_frame_id);
    return;
  }

  DCHECK_CURRENTLY_ON(BrowserThread::IO);
  BrowserThread::PostTask(
      BrowserThread::UI, FROM_HERE,
      base::BindOnce(
          &MediaCaptureDevicesDispatcher::OnCreatingAudioStreamOnUIThread,
          base::Unretained(this), render_process_id, render_frame_id));
}

void MediaCaptureDevicesDispatcher::NotifyAudioDevicesChangedOnUIThread() {
  MediaStreamDevices devices = GetAudioCaptureDevices();
  for (auto& observer : observers_)
    observer.OnUpdateAudioDevices(devices);
}

void MediaCaptureDevicesDispatcher::NotifyVideoDevicesChangedOnUIThread() {
  MediaStreamDevices devices = GetVideoCaptureDevices();
  for (auto& observer : observers_)
    observer.OnUpdateVideoDevices(devices);
}

void MediaCaptureDevicesDispatcher::UpdateMediaRequestStateOnUIThread(
    int render_process_id,
    int render_frame_id,
    int page_request_id,
    const GURL& security_origin,
    content::MediaStreamType stream_type,
    content::MediaRequestState state) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  for (const auto& handler : media_access_handlers_) {
    if (handler->SupportsStreamType(
            WebContentsFromIds(render_process_id, render_frame_id), stream_type,
            nullptr)) {
      handler->UpdateMediaRequestState(render_process_id, render_frame_id,
                                       page_request_id, stream_type, state);
      break;
    }
  }

#if defined(OS_CHROMEOS)
  if (IsOriginForCasting(security_origin) && IsVideoMediaType(stream_type)) {
    // Notify ash that casting state has changed.
    if (state == content::MEDIA_REQUEST_STATE_DONE) {
      ash::Shell::Get()->OnCastingSessionStartedOrStopped(true);
    } else if (state == content::MEDIA_REQUEST_STATE_CLOSING) {
      ash::Shell::Get()->OnCastingSessionStartedOrStopped(false);
    }
  }
#endif

  for (auto& observer : observers_) {
    observer.OnRequestUpdate(render_process_id, render_frame_id, stream_type,
                             state);
  }
}

void MediaCaptureDevicesDispatcher::OnCreatingAudioStreamOnUIThread(
    int render_process_id,
    int render_frame_id) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  for (auto& observer : observers_)
    observer.OnCreatingAudioStream(render_process_id, render_frame_id);
}

bool MediaCaptureDevicesDispatcher::IsInsecureCapturingInProgress(
    int render_process_id,
    int render_frame_id) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
#if BUILDFLAG(ENABLE_EXTENSIONS)
  for (const auto& handler : media_access_handlers_) {
    if (handler->SupportsStreamType(
            WebContentsFromIds(render_process_id, render_frame_id),
            content::MEDIA_DESKTOP_VIDEO_CAPTURE, nullptr) ||
        handler->SupportsStreamType(
            WebContentsFromIds(render_process_id, render_frame_id),
            content::MEDIA_TAB_VIDEO_CAPTURE, nullptr)) {
      if (ToCaptureAccessHandlerBase(handler.get())
              ->IsInsecureCapturingInProgress(render_process_id,
                                              render_frame_id))
        return true;
    }
  }
#endif
  return false;
}

void MediaCaptureDevicesDispatcher::SetTestAudioCaptureDevices(
    const MediaStreamDevices& devices) {
  test_audio_devices_ = devices;
}

void MediaCaptureDevicesDispatcher::SetTestVideoCaptureDevices(
    const MediaStreamDevices& devices) {
  test_video_devices_ = devices;
}

void MediaCaptureDevicesDispatcher::OnSetCapturingLinkSecured(
    int render_process_id,
    int render_frame_id,
    int page_request_id,
    content::MediaStreamType stream_type,
    bool is_secure) {
  DCHECK_CURRENTLY_ON(BrowserThread::IO);
  if (stream_type != content::MEDIA_TAB_VIDEO_CAPTURE &&
      stream_type != content::MEDIA_DESKTOP_VIDEO_CAPTURE)
    return;

  BrowserThread::PostTask(
      BrowserThread::UI, FROM_HERE,
      base::BindOnce(&MediaCaptureDevicesDispatcher::UpdateCapturingLinkSecured,
                     base::Unretained(this), render_process_id, render_frame_id,
                     page_request_id, stream_type, is_secure));
}

void MediaCaptureDevicesDispatcher::UpdateCapturingLinkSecured(
    int render_process_id,
    int render_frame_id,
    int page_request_id,
    content::MediaStreamType stream_type,
    bool is_secure) {
  DCHECK_CURRENTLY_ON(BrowserThread::UI);
  if (stream_type != content::MEDIA_TAB_VIDEO_CAPTURE &&
      stream_type != content::MEDIA_DESKTOP_VIDEO_CAPTURE)
    return;

#if BUILDFLAG(ENABLE_EXTENSIONS)
  for (const auto& handler : media_access_handlers_) {
    if (handler->SupportsStreamType(
            WebContentsFromIds(render_process_id, render_frame_id), stream_type,
            nullptr)) {
      ToCaptureAccessHandlerBase(handler.get())
          ->UpdateCapturingLinkSecured(render_process_id, render_frame_id,
                                       page_request_id, is_secure);
      break;
    }
  }
#endif
}
