// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "content/renderer/media/webrtc/rtc_peer_connection_handler.h"

#include <ctype.h>
#include <string.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "base/command_line.h"
#include "base/location.h"
#include "base/logging.h"
#include "base/metrics/histogram_functions.h"
#include "base/metrics/histogram_macros.h"
#include "base/stl_util.h"
#include "base/strings/string_util.h"
#include "base/strings/utf_string_conversions.h"
#include "base/threading/thread_task_runner_handle.h"
#include "base/trace_event/trace_event.h"
#include "base/unguessable_token.h"
#include "content/public/common/content_features.h"
#include "content/public/common/content_switches.h"
#include "content/renderer/media/stream/media_stream_constraints_util.h"
#include "content/renderer/media/stream/media_stream_track.h"
#include "content/renderer/media/webrtc/peer_connection_dependency_factory.h"
#include "content/renderer/media/webrtc/peer_connection_tracker.h"
#include "content/renderer/media/webrtc/rtc_certificate.h"
#include "content/renderer/media/webrtc/rtc_data_channel_handler.h"
#include "content/renderer/media/webrtc/rtc_dtmf_sender_handler.h"
#include "content/renderer/media/webrtc/rtc_event_log_output_sink.h"
#include "content/renderer/media/webrtc/rtc_event_log_output_sink_proxy.h"
#include "content/renderer/media/webrtc/rtc_stats.h"
#include "content/renderer/media/webrtc/webrtc_audio_device_impl.h"
#include "content/renderer/media/webrtc/webrtc_set_description_observer.h"
#include "content/renderer/media/webrtc/webrtc_uma_histograms.h"
#include "content/renderer/render_thread_impl.h"
#include "media/base/media_switches.h"
#include "third_party/blink/public/platform/web_media_constraints.h"
#include "third_party/blink/public/platform/web_rtc_answer_options.h"
#include "third_party/blink/public/platform/web_rtc_configuration.h"
#include "third_party/blink/public/platform/web_rtc_data_channel_init.h"
#include "third_party/blink/public/platform/web_rtc_ice_candidate.h"
#include "third_party/blink/public/platform/web_rtc_legacy_stats.h"
#include "third_party/blink/public/platform/web_rtc_offer_options.h"
#include "third_party/blink/public/platform/web_rtc_rtp_sender.h"
#include "third_party/blink/public/platform/web_rtc_rtp_transceiver.h"
#include "third_party/blink/public/platform/web_rtc_session_description.h"
#include "third_party/blink/public/platform/web_rtc_session_description_request.h"
#include "third_party/blink/public/platform/web_rtc_void_request.h"
#include "third_party/blink/public/platform/web_url.h"
#include "third_party/webrtc/api/rtceventlogoutput.h"
#include "third_party/webrtc/pc/mediasession.h"

using webrtc::DataChannelInterface;
using webrtc::IceCandidateInterface;
using webrtc::MediaStreamInterface;
using webrtc::PeerConnectionInterface;
using webrtc::PeerConnectionObserver;
using webrtc::StatsReport;
using webrtc::StatsReports;

namespace content {
namespace {

// Used to back histogram value of "WebRTC.PeerConnection.RtcpMux",
// so treat as append-only.
enum RtcpMux {
  RTCP_MUX_DISABLED,
  RTCP_MUX_ENABLED,
  RTCP_MUX_NO_MEDIA,
  RTCP_MUX_MAX
};

// Converter functions from libjingle types to WebKit types.
blink::WebRTCPeerConnectionHandlerClient::ICEGatheringState
GetWebKitIceGatheringState(
    webrtc::PeerConnectionInterface::IceGatheringState state) {
  using blink::WebRTCPeerConnectionHandlerClient;
  switch (state) {
    case webrtc::PeerConnectionInterface::kIceGatheringNew:
      return WebRTCPeerConnectionHandlerClient::kICEGatheringStateNew;
    case webrtc::PeerConnectionInterface::kIceGatheringGathering:
      return WebRTCPeerConnectionHandlerClient::kICEGatheringStateGathering;
    case webrtc::PeerConnectionInterface::kIceGatheringComplete:
      return WebRTCPeerConnectionHandlerClient::kICEGatheringStateComplete;
    default:
      NOTREACHED();
      return WebRTCPeerConnectionHandlerClient::kICEGatheringStateNew;
  }
}

blink::WebRTCPeerConnectionHandlerClient::ICEConnectionState
GetWebKitIceConnectionState(
    webrtc::PeerConnectionInterface::IceConnectionState ice_state) {
  using blink::WebRTCPeerConnectionHandlerClient;
  switch (ice_state) {
    case webrtc::PeerConnectionInterface::kIceConnectionNew:
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateStarting;
    case webrtc::PeerConnectionInterface::kIceConnectionChecking:
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateChecking;
    case webrtc::PeerConnectionInterface::kIceConnectionConnected:
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateConnected;
    case webrtc::PeerConnectionInterface::kIceConnectionCompleted:
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateCompleted;
    case webrtc::PeerConnectionInterface::kIceConnectionFailed:
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateFailed;
    case webrtc::PeerConnectionInterface::kIceConnectionDisconnected:
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateDisconnected;
    case webrtc::PeerConnectionInterface::kIceConnectionClosed:
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateClosed;
    default:
      NOTREACHED();
      return WebRTCPeerConnectionHandlerClient::kICEConnectionStateClosed;
  }
}

blink::WebRTCSessionDescription CreateWebKitSessionDescription(
    const std::string& sdp, const std::string& type) {
  blink::WebRTCSessionDescription description;
  description.Initialize(blink::WebString::FromUTF8(type),
                         blink::WebString::FromUTF8(sdp));
  return description;
}

blink::WebRTCSessionDescription
CreateWebKitSessionDescription(
    const webrtc::SessionDescriptionInterface* native_desc) {
  if (!native_desc) {
    LOG(ERROR) << "Native session description is null.";
    return blink::WebRTCSessionDescription();
  }

  std::string sdp;
  if (!native_desc->ToString(&sdp)) {
    LOG(ERROR) << "Failed to get SDP string of native session description.";
    return blink::WebRTCSessionDescription();
  }

  return CreateWebKitSessionDescription(sdp, native_desc->type());
}

void RunClosureWithTrace(const base::Closure& closure,
                         const char* trace_event_name) {
  TRACE_EVENT0("webrtc", trace_event_name);
  closure.Run();
}

void RunSynchronousClosure(const base::Closure& closure,
                           const char* trace_event_name,
                           base::WaitableEvent* event) {
  {
    TRACE_EVENT0("webrtc", trace_event_name);
    closure.Run();
  }
  event->Signal();
}

void GetSdpAndTypeFromSessionDescription(
    const base::Callback<const webrtc::SessionDescriptionInterface*()>&
        description_callback,
    std::string* sdp, std::string* type) {
  const webrtc::SessionDescriptionInterface* description =
      description_callback.Run();
  if (description) {
    description->ToString(sdp);
    *type = description->type();
  }
}

// Converter functions from Blink types to WebRTC types.

// This function doesn't assume |webrtc_config| is empty. Any fields in
// |blink_config| replace the corresponding fields in |webrtc_config|, but
// fields that only exist in |webrtc_config| are left alone.
void GetNativeRtcConfiguration(
    const blink::WebRTCConfiguration& blink_config,
    webrtc::PeerConnectionInterface::RTCConfiguration* webrtc_config) {
  DCHECK(webrtc_config);

  webrtc_config->servers.clear();
  for (const blink::WebRTCIceServer& blink_server : blink_config.ice_servers) {
    webrtc::PeerConnectionInterface::IceServer server;
    server.username = blink_server.username.Utf8();
    server.password = blink_server.credential.Utf8();
    server.uri = blink_server.url.GetString().Utf8();
    webrtc_config->servers.push_back(server);
  }

  switch (blink_config.ice_transport_policy) {
    case blink::WebRTCIceTransportPolicy::kRelay:
      webrtc_config->type = webrtc::PeerConnectionInterface::kRelay;
      break;
    case blink::WebRTCIceTransportPolicy::kAll:
      webrtc_config->type = webrtc::PeerConnectionInterface::kAll;
      break;
    default:
      NOTREACHED();
  }

  switch (blink_config.bundle_policy) {
    case blink::WebRTCBundlePolicy::kBalanced:
      webrtc_config->bundle_policy =
          webrtc::PeerConnectionInterface::kBundlePolicyBalanced;
      break;
    case blink::WebRTCBundlePolicy::kMaxBundle:
      webrtc_config->bundle_policy =
          webrtc::PeerConnectionInterface::kBundlePolicyMaxBundle;
      break;
    case blink::WebRTCBundlePolicy::kMaxCompat:
      webrtc_config->bundle_policy =
          webrtc::PeerConnectionInterface::kBundlePolicyMaxCompat;
      break;
    default:
      NOTREACHED();
  }

  switch (blink_config.rtcp_mux_policy) {
    case blink::WebRTCRtcpMuxPolicy::kNegotiate:
      webrtc_config->rtcp_mux_policy =
          webrtc::PeerConnectionInterface::kRtcpMuxPolicyNegotiate;
      break;
    case blink::WebRTCRtcpMuxPolicy::kRequire:
      webrtc_config->rtcp_mux_policy =
          webrtc::PeerConnectionInterface::kRtcpMuxPolicyRequire;
      break;
    default:
      NOTREACHED();
  }

  switch (blink_config.sdp_semantics) {
    case blink::WebRTCSdpSemantics::kPlanB:
      webrtc_config->sdp_semantics = webrtc::SdpSemantics::kPlanB;
      break;
    case blink::WebRTCSdpSemantics::kUnifiedPlan:
      webrtc_config->sdp_semantics = webrtc::SdpSemantics::kUnifiedPlan;
      break;
    case blink::WebRTCSdpSemantics::kDefault:
      NOTREACHED();
  }

  webrtc_config->certificates.clear();
  for (const std::unique_ptr<blink::WebRTCCertificate>& blink_certificate :
       blink_config.certificates) {
    webrtc_config->certificates.push_back(
        static_cast<RTCCertificate*>(blink_certificate.get())
            ->rtcCertificate());
  }

  webrtc_config->ice_candidate_pool_size = blink_config.ice_candidate_pool_size;
}

absl::optional<bool> ConstraintToOptional(
    const blink::WebMediaConstraints& constraints,
    const blink::BooleanConstraint blink::WebMediaTrackConstraintSet::*picker) {
  bool value;
  if (GetConstraintValueAsBoolean(constraints, picker, &value)) {
    return absl::optional<bool>(value);
  }
  return absl::nullopt;
}

void CopyConstraintsIntoRtcConfiguration(
    const blink::WebMediaConstraints constraints,
    webrtc::PeerConnectionInterface::RTCConfiguration* configuration) {
  // Copy info from constraints into configuration, if present.
  if (constraints.IsEmpty()) {
    return;
  }

  bool the_value;
  if (GetConstraintValueAsBoolean(
          constraints, &blink::WebMediaTrackConstraintSet::enable_i_pv6,
          &the_value)) {
    configuration->disable_ipv6 = !the_value;
  } else {
    // Note: IPv6 WebRTC value is "disable" while Blink is "enable".
    configuration->disable_ipv6 = false;
  }

  if (GetConstraintValueAsBoolean(
          constraints, &blink::WebMediaTrackConstraintSet::enable_dscp,
          &the_value)) {
    configuration->set_dscp(the_value);
  }

  if (GetConstraintValueAsBoolean(
          constraints,
          &blink::WebMediaTrackConstraintSet::goog_cpu_overuse_detection,
          &the_value)) {
    configuration->set_cpu_adaptation(the_value);
  }

  if (GetConstraintValueAsBoolean(
          constraints,
          &blink::WebMediaTrackConstraintSet::
              goog_enable_video_suspend_below_min_bitrate,
          &the_value)) {
    configuration->set_suspend_below_min_bitrate(the_value);
  }

  if (!GetConstraintValueAsBoolean(
          constraints,
          &blink::WebMediaTrackConstraintSet::enable_rtp_data_channels,
          &configuration->enable_rtp_data_channel)) {
    configuration->enable_rtp_data_channel = false;
  }
  int rate;
  if (GetConstraintValueAsInteger(
          constraints,
          &blink::WebMediaTrackConstraintSet::goog_screencast_min_bitrate,
          &rate)) {
    configuration->screencast_min_bitrate = rate;
  }
  configuration->combined_audio_video_bwe = ConstraintToOptional(
      constraints,
      &blink::WebMediaTrackConstraintSet::goog_combined_audio_video_bwe);
  configuration->enable_dtls_srtp = ConstraintToOptional(
      constraints, &blink::WebMediaTrackConstraintSet::enable_dtls_srtp);
}

// Class mapping responses from calls to libjingle CreateOffer/Answer and
// the blink::WebRTCSessionDescriptionRequest.
class CreateSessionDescriptionRequest
    : public webrtc::CreateSessionDescriptionObserver {
 public:
  explicit CreateSessionDescriptionRequest(
      const scoped_refptr<base::SingleThreadTaskRunner>& main_thread,
      const blink::WebRTCSessionDescriptionRequest& request,
      const base::WeakPtr<RTCPeerConnectionHandler>& handler,
      const base::WeakPtr<PeerConnectionTracker>& tracker,
      PeerConnectionTracker::Action action)
      : main_thread_(main_thread),
        webkit_request_(request),
        handler_(handler),
        tracker_(tracker),
        action_(action) {}

  void OnSuccess(webrtc::SessionDescriptionInterface* desc) override {
    if (!main_thread_->BelongsToCurrentThread()) {
      main_thread_->PostTask(
          FROM_HERE, base::BindOnce(&CreateSessionDescriptionRequest::OnSuccess,
                                    this, desc));
      return;
    }

    if (tracker_ && handler_) {
      std::string value;
      if (desc) {
        desc->ToString(&value);
        value = "type: " + desc->type() + ", sdp: " + value;
      }
      tracker_->TrackSessionDescriptionCallback(handler_.get(), action_,
                                                "OnSuccess", value);
    }
    webkit_request_.RequestSucceeded(CreateWebKitSessionDescription(desc));
    webkit_request_.Reset();
    delete desc;
  }
  void OnFailure(webrtc::RTCError error) override {
    if (!main_thread_->BelongsToCurrentThread()) {
      main_thread_->PostTask(
          FROM_HERE, base::BindOnce(&CreateSessionDescriptionRequest::OnFailure,
                                    this, std::move(error)));
      return;
    }

    if (handler_ && tracker_) {
      tracker_->TrackSessionDescriptionCallback(handler_.get(), action_,
                                                "OnFailure", error.message());
    }
    // TODO(hta): Convert CreateSessionDescriptionRequest.OnFailure
    webkit_request_.RequestFailed(error);
    webkit_request_.Reset();
  }

 protected:
  ~CreateSessionDescriptionRequest() override {
    // This object is reference counted and its callback methods |OnSuccess| and
    // |OnFailure| will be invoked on libjingle's signaling thread and posted to
    // the main thread. Since the main thread may complete before the signaling
    // thread has deferenced this object there is no guarantee that this object
    // is destructed on the main thread.
    DLOG_IF(ERROR, !webkit_request_.IsNull())
        << "CreateSessionDescriptionRequest not completed. Shutting down?";
  }

  const scoped_refptr<base::SingleThreadTaskRunner> main_thread_;
  blink::WebRTCSessionDescriptionRequest webkit_request_;
  const base::WeakPtr<RTCPeerConnectionHandler> handler_;
  const base::WeakPtr<PeerConnectionTracker> tracker_;
  PeerConnectionTracker::Action action_;
};

blink::WebRTCLegacyStatsMemberType
WebRTCLegacyStatsMemberTypeFromStatsValueType(
    webrtc::StatsReport::Value::Type type) {
  switch (type) {
    case StatsReport::Value::kInt:
      return blink::kWebRTCLegacyStatsMemberTypeInt;
    case StatsReport::Value::kInt64:
      return blink::kWebRTCLegacyStatsMemberTypeInt64;
    case StatsReport::Value::kFloat:
      return blink::kWebRTCLegacyStatsMemberTypeFloat;
    case StatsReport::Value::kString:
    case StatsReport::Value::kStaticString:
      return blink::kWebRTCLegacyStatsMemberTypeString;
    case StatsReport::Value::kBool:
      return blink::kWebRTCLegacyStatsMemberTypeBool;
    case StatsReport::Value::kId:
      return blink::kWebRTCLegacyStatsMemberTypeId;
  }
  NOTREACHED();
  return blink::kWebRTCLegacyStatsMemberTypeInt;
}

// Class mapping responses from calls to libjingle
// GetStats into a blink::WebRTCStatsCallback.
class StatsResponse : public webrtc::StatsObserver {
 public:
  StatsResponse(const scoped_refptr<LocalRTCStatsRequest>& request,
                scoped_refptr<base::SingleThreadTaskRunner> task_runner)
      : request_(request.get()), main_thread_(task_runner) {
    // Measure the overall time it takes to satisfy a getStats request.
    TRACE_EVENT_ASYNC_BEGIN0("webrtc", "getStats_Native", this);
    signaling_thread_checker_.DetachFromThread();
  }

  void OnComplete(const StatsReports& reports) override {
    DCHECK(signaling_thread_checker_.CalledOnValidThread());
    TRACE_EVENT0("webrtc", "StatsResponse::OnComplete");
    // We can't use webkit objects directly since they use a single threaded
    // heap allocator.
    std::vector<Report*>* report_copies = new std::vector<Report*>();
    report_copies->reserve(reports.size());
    for (auto* r : reports)
      report_copies->push_back(new Report(r));

    main_thread_->PostTaskAndReply(
        FROM_HERE,
        base::BindOnce(&StatsResponse::DeliverCallback, this,
                       base::Unretained(report_copies)),
        base::BindOnce(&StatsResponse::DeleteReports,
                       base::Unretained(report_copies)));
  }

 private:
  class Report : public blink::WebRTCLegacyStats {
   public:
    class MemberIterator : public blink::WebRTCLegacyStatsMemberIterator {
     public:
      MemberIterator(const StatsReport::Values::const_iterator& it,
                     const StatsReport::Values::const_iterator& end)
          : it_(it), end_(end) {}

      // blink::WebRTCLegacyStatsMemberIterator
      bool IsEnd() const override { return it_ == end_; }
      void Next() override { ++it_; }
      blink::WebString GetName() const override {
        return blink::WebString::FromUTF8(it_->second->display_name());
      }
      blink::WebRTCLegacyStatsMemberType GetType() const override {
        return WebRTCLegacyStatsMemberTypeFromStatsValueType(
            it_->second->type());
      }
      int ValueInt() const override { return it_->second->int_val(); }
      int64_t ValueInt64() const override { return it_->second->int64_val(); }
      float ValueFloat() const override { return it_->second->float_val(); }
      blink::WebString ValueString() const override {
        const StatsReport::ValuePtr& value = it_->second;
        if (value->type() == StatsReport::Value::kString)
          return blink::WebString::FromUTF8(value->string_val());
        DCHECK_EQ(value->type(), StatsReport::Value::kStaticString);
        return blink::WebString::FromUTF8(value->static_string_val());
      }
      bool ValueBool() const override { return it_->second->bool_val(); }
      blink::WebString ValueToString() const override {
        const StatsReport::ValuePtr& value = it_->second;
        if (value->type() == StatsReport::Value::kString)
          return blink::WebString::FromUTF8(value->string_val());
        if (value->type() == StatsReport::Value::kStaticString)
          return blink::WebString::FromUTF8(value->static_string_val());
        return blink::WebString::FromUTF8(value->ToString());
      }

     private:
      StatsReport::Values::const_iterator it_;
      StatsReport::Values::const_iterator end_;
    };

    Report(const StatsReport* report)
        : thread_checker_(),
          id_(report->id()->ToString()),
          type_(report->type()),
          type_name_(report->TypeToString()),
          timestamp_(report->timestamp()),
          values_(report->values()) {}

    ~Report() override {
      // Since the values vector holds pointers to const objects that are bound
      // to the signaling thread, they must be released on the same thread.
      DCHECK(thread_checker_.CalledOnValidThread());
    }

    // blink::WebRTCLegacyStats
    blink::WebString Id() const override {
      return blink::WebString::FromUTF8(id_);
    }
    blink::WebString GetType() const override {
      return blink::WebString::FromUTF8(type_name_);
    }
    double Timestamp() const override { return timestamp_; }
    blink::WebRTCLegacyStatsMemberIterator* Iterator() const override {
      return new MemberIterator(values_.cbegin(), values_.cend());
    }

    bool HasValues() const {
      return values_.size() > 0;
    }

   private:
    const base::ThreadChecker thread_checker_;
    const std::string id_;
    const StatsReport::StatsType type_;
    const std::string type_name_;
    const double timestamp_;
    const StatsReport::Values values_;
  };

  static void DeleteReports(std::vector<Report*>* reports) {
    TRACE_EVENT0("webrtc", "StatsResponse::DeleteReports");
    for (auto* p : *reports)
      delete p;
    delete reports;
  }

  void DeliverCallback(const std::vector<Report*>* reports) {
    DCHECK(main_thread_->BelongsToCurrentThread());
    TRACE_EVENT0("webrtc", "StatsResponse::DeliverCallback");

    rtc::scoped_refptr<LocalRTCStatsResponse> response(
        request_->createResponse().get());
    for (const auto* report : *reports) {
      if (report->HasValues())
        AddReport(response.get(), *report);
    }

    // Record the getStats operation as done before calling into Blink so that
    // we don't skew the perf measurements of the native code with whatever the
    // callback might be doing.
    TRACE_EVENT_ASYNC_END0("webrtc", "getStats_Native", this);
    request_->requestSucceeded(response);
    request_ = nullptr;  // must be freed on the main thread.
  }

  void AddReport(LocalRTCStatsResponse* response, const Report& report) {
    response->addStats(report);
  }

  rtc::scoped_refptr<LocalRTCStatsRequest> request_;
  const scoped_refptr<base::SingleThreadTaskRunner> main_thread_;
  base::ThreadChecker signaling_thread_checker_;
};

void GetStatsOnSignalingThread(
    const scoped_refptr<webrtc::PeerConnectionInterface>& pc,
    webrtc::PeerConnectionInterface::StatsOutputLevel level,
    const scoped_refptr<webrtc::StatsObserver>& observer,
    rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> selector) {
  TRACE_EVENT0("webrtc", "GetStatsOnSignalingThread");

  if (selector) {
    bool belongs_to_pc = false;
    for (const auto& sender : pc->GetSenders()) {
      if (sender->track() == selector) {
        belongs_to_pc = true;
        break;
      }
    }
    if (!belongs_to_pc) {
      for (const auto& receiver : pc->GetReceivers()) {
        if (receiver->track() == selector) {
          belongs_to_pc = true;
          break;
        }
      }
    }
    if (!belongs_to_pc) {
      DVLOG(1) << "GetStats: Track not found.";
      observer->OnComplete(StatsReports());
      return;
    }
  }

  if (!pc->GetStats(observer.get(), selector.get(), level)) {
    DVLOG(1) << "GetStats failed.";
    observer->OnComplete(StatsReports());
  }
}

void GetRTCStatsOnSignalingThread(
    const scoped_refptr<base::SingleThreadTaskRunner>& main_thread,
    scoped_refptr<webrtc::PeerConnectionInterface> native_peer_connection,
    std::unique_ptr<blink::WebRTCStatsReportCallback> callback) {
  TRACE_EVENT0("webrtc", "GetRTCStatsOnSignalingThread");

  native_peer_connection->GetStats(
      RTCStatsCollectorCallbackImpl::Create(main_thread, std::move(callback)));
}

void ConvertOfferOptionsToWebrtcOfferOptions(
    const blink::WebRTCOfferOptions& options,
    webrtc::PeerConnectionInterface::RTCOfferAnswerOptions* output) {
  output->offer_to_receive_audio = options.OfferToReceiveAudio();
  output->offer_to_receive_video = options.OfferToReceiveVideo();
  output->voice_activity_detection = options.VoiceActivityDetection();
  output->ice_restart = options.IceRestart();
}

void ConvertAnswerOptionsToWebrtcAnswerOptions(
    const blink::WebRTCAnswerOptions& options,
    webrtc::PeerConnectionInterface::RTCOfferAnswerOptions* output) {
  output->voice_activity_detection = options.VoiceActivityDetection();
}

void ConvertConstraintsToWebrtcOfferOptions(
    const blink::WebMediaConstraints& constraints,
    webrtc::PeerConnectionInterface::RTCOfferAnswerOptions* output) {
  if (constraints.IsEmpty()) {
    return;
  }
  std::string failing_name;
  if (constraints.Basic().HasMandatoryOutsideSet(
          {constraints.Basic().offer_to_receive_audio.GetName(),
           constraints.Basic().offer_to_receive_video.GetName(),
           constraints.Basic().voice_activity_detection.GetName(),
           constraints.Basic().ice_restart.GetName()},
          failing_name)) {
    // TODO(hta): Reject the calling operation with "constraint error"
    // https://crbug.com/594894
    DLOG(ERROR) << "Invalid mandatory constraint to CreateOffer/Answer: "
                << failing_name;
  }
  GetConstraintValueAsInteger(
      constraints, &blink::WebMediaTrackConstraintSet::offer_to_receive_audio,
      &output->offer_to_receive_audio);
  GetConstraintValueAsInteger(
      constraints, &blink::WebMediaTrackConstraintSet::offer_to_receive_video,
      &output->offer_to_receive_video);
  GetConstraintValueAsBoolean(
      constraints, &blink::WebMediaTrackConstraintSet::voice_activity_detection,
      &output->voice_activity_detection);
  GetConstraintValueAsBoolean(constraints,
                              &blink::WebMediaTrackConstraintSet::ice_restart,
                              &output->ice_restart);
}

std::set<RTCPeerConnectionHandler*>* GetPeerConnectionHandlers() {
  static std::set<RTCPeerConnectionHandler*>* handlers =
      new std::set<RTCPeerConnectionHandler*>();
  return handlers;
}

// Counts the number of senders that have |stream_id| as an associated stream.
size_t GetLocalStreamUsageCount(
    const std::vector<std::unique_ptr<RTCRtpSender>>& rtp_senders,
    const std::string stream_id) {
  size_t usage_count = 0;
  for (const auto& sender : rtp_senders) {
    for (const auto& sender_stream_id : sender->state().stream_ids()) {
      if (sender_stream_id == stream_id) {
        ++usage_count;
        break;
      }
    }
  }
  return usage_count;
}

bool IsRemoteStream(
    const std::map<uintptr_t, std::unique_ptr<RTCRtpReceiver>>& rtp_receivers,
    const std::string& stream_id) {
  for (const auto& receiver_entry : rtp_receivers) {
    for (const auto& receiver_stream_id :
         receiver_entry.second->state().stream_ids()) {
      if (stream_id == receiver_stream_id)
        return true;
    }
  }
  return false;
}

enum SdpSemanticRequested {
  kSdpSemanticRequestedDefault,
  kSdpSemanticRequestedPlanB,
  kSdpSemanticRequestedUnifiedPlan,
  kSdpSemanticRequestedMax
};

SdpSemanticRequested GetSdpSemanticRequested(
    blink::WebRTCSdpSemantics sdp_semantics) {
  switch (sdp_semantics) {
    case blink::WebRTCSdpSemantics::kDefault:
      return kSdpSemanticRequestedDefault;
    case blink::WebRTCSdpSemantics::kPlanB:
      return kSdpSemanticRequestedPlanB;
    case blink::WebRTCSdpSemantics::kUnifiedPlan:
      return kSdpSemanticRequestedUnifiedPlan;
  }
  NOTREACHED();
  return kSdpSemanticRequestedDefault;
}

MediaStreamTrackMetrics::Kind MediaStreamTrackMetricsKind(
    const blink::WebMediaStreamTrack& track) {
  return track.Source().GetType() == blink::WebMediaStreamSource::kTypeAudio
             ? MediaStreamTrackMetrics::Kind::kAudio
             : MediaStreamTrackMetrics::Kind::kVideo;
}

}  // namespace

// Implementation of LocalRTCStatsRequest.
LocalRTCStatsRequest::LocalRTCStatsRequest(blink::WebRTCStatsRequest impl)
    : impl_(impl) {
}

LocalRTCStatsRequest::LocalRTCStatsRequest() {}
LocalRTCStatsRequest::~LocalRTCStatsRequest() {}

bool LocalRTCStatsRequest::hasSelector() const {
  return impl_.HasSelector();
}

blink::WebMediaStreamTrack LocalRTCStatsRequest::component() const {
  return impl_.Component();
}

scoped_refptr<LocalRTCStatsResponse> LocalRTCStatsRequest::createResponse() {
  return scoped_refptr<LocalRTCStatsResponse>(
      new rtc::RefCountedObject<LocalRTCStatsResponse>(impl_.CreateResponse()));
}

void LocalRTCStatsRequest::requestSucceeded(
    const LocalRTCStatsResponse* response) {
  impl_.RequestSucceeded(response->webKitStatsResponse());
}

// Implementation of LocalRTCStatsResponse.
blink::WebRTCStatsResponse LocalRTCStatsResponse::webKitStatsResponse() const {
  return impl_;
}

void LocalRTCStatsResponse::addStats(const blink::WebRTCLegacyStats& stats) {
  impl_.AddStats(stats);
}

// Processes the resulting state changes of a SetLocalDescription() or
// SetRemoteDescription() call.
class RTCPeerConnectionHandler::WebRtcSetDescriptionObserverImpl
    : public WebRtcSetDescriptionObserver {
 public:
  WebRtcSetDescriptionObserverImpl(
      base::WeakPtr<RTCPeerConnectionHandler> handler,
      blink::WebRTCVoidRequest web_request,
      base::WeakPtr<PeerConnectionTracker> tracker,
      scoped_refptr<base::SingleThreadTaskRunner> task_runner,
      PeerConnectionTracker::Action action,
      blink::WebRTCSdpSemantics sdp_semantics)
      : handler_(handler),
        main_thread_(task_runner),
        web_request_(web_request),
        tracker_(tracker),
        action_(action),
        sdp_semantics_(sdp_semantics) {
    DCHECK(sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB ||
           sdp_semantics_ == blink::WebRTCSdpSemantics::kUnifiedPlan);
  }

  void OnSetDescriptionComplete(
      webrtc::RTCError error,
      WebRtcSetDescriptionObserver::States states) override {
    if (!error.ok()) {
      if (tracker_ && handler_) {
        tracker_->TrackSessionDescriptionCallback(handler_.get(), action_,
                                                  "OnFailure", error.message());
      }
      web_request_.RequestFailed(error);
      web_request_.Reset();
      return;
    }

    if (handler_) {
      handler_->OnSignalingChange(states.signaling_state);

      // Process the rest of the state changes differently depending on SDP
      // semantics.
      if (sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB) {
        ProcessStateChangesPlanB(std::move(states));
      } else {
        DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
        ProcessStateChangesUnifiedPlan(std::move(states));
      }

      if (tracker_) {
        tracker_->TrackSessionDescriptionCallback(handler_.get(), action_,
                                                  "OnSuccess", "");
      }
    }
    if (action_ == PeerConnectionTracker::ACTION_SET_REMOTE_DESCRIPTION) {
      // Resolve the promise in a post to ensure any events scheduled for
      // dispatching will have fired by the time the promise is resolved.
      // TODO(hbos): Don't schedule/post to fire events/resolve the promise.
      // Instead, do it all synchronously. This must happen as the last step
      // before returning so that all effects of SRD have occurred when the
      // event executes. https://crbug.com/788558
      main_thread_->PostTask(
          FROM_HERE,
          base::BindOnce(&RTCPeerConnectionHandler::
                             WebRtcSetDescriptionObserverImpl::ResolvePromise,
                         this));
    } else {
      // Resolve promise immediately if we can. https://crbug.com/788558 still
      // needs to be addressed for "setLocalDescription(answer)" rejecting a
      // transceiver in Unified Plan, but this is a minor edge-case.
      ResolvePromise();
    }
  }

 private:
  ~WebRtcSetDescriptionObserverImpl() override {}

  void ResolvePromise() {
    web_request_.RequestSucceeded();
    web_request_.Reset();
  }

  void ProcessStateChangesPlanB(WebRtcSetDescriptionObserver::States states) {
    DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kPlanB);
    // Determine which receivers have been removed before processing the
    // removal as to not invalidate the iterator.
    std::vector<RTCRtpReceiver*> removed_receivers;
    for (auto it = handler_->rtp_receivers_.begin();
         it != handler_->rtp_receivers_.end(); ++it) {
      if (ReceiverWasRemoved(*it->second, states.transceiver_states))
        removed_receivers.push_back(it->second.get());
    }

    // Process the addition of remote receivers/tracks.
    for (auto& transceiver_state : states.transceiver_states) {
      if (ReceiverWasAdded(transceiver_state)) {
        handler_->OnAddReceiverPlanB(transceiver_state.MoveReceiverState());
      }
    }
    // Process the removal of remote receivers/tracks.
    for (auto* removed_receiver : removed_receivers) {
      handler_->OnRemoveReceiverPlanB(RTCRtpReceiver::getId(
          removed_receiver->state().webrtc_receiver().get()));
    }
  }

  bool ReceiverWasAdded(const RtpTransceiverState& transceiver_state) {
    return !base::ContainsKey(
        handler_->rtp_receivers_,
        RTCRtpReceiver::getId(
            transceiver_state.receiver_state()->webrtc_receiver().get()));
  }

  bool ReceiverWasRemoved(
      const RTCRtpReceiver& receiver,
      const std::vector<RtpTransceiverState>& transceiver_states) {
    for (const auto& transceiver_state : transceiver_states) {
      if (transceiver_state.receiver_state()->webrtc_receiver() ==
          receiver.state().webrtc_receiver()) {
        return false;
      }
    }
    return true;
  }

  void ProcessStateChangesUnifiedPlan(
      WebRtcSetDescriptionObserver::States states) {
    DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
    handler_->OnModifyTransceivers(
        std::move(states.transceiver_states),
        action_ == PeerConnectionTracker::ACTION_SET_REMOTE_DESCRIPTION);
  }

  base::WeakPtr<RTCPeerConnectionHandler> handler_;
  scoped_refptr<base::SequencedTaskRunner> main_thread_;
  blink::WebRTCVoidRequest web_request_;
  base::WeakPtr<PeerConnectionTracker> tracker_;
  PeerConnectionTracker::Action action_;
  blink::WebRTCSdpSemantics sdp_semantics_;
};

// Receives notifications from a PeerConnection object about state changes. The
// callbacks we receive here come on the webrtc signaling thread, so this class
// takes care of delivering them to an RTCPeerConnectionHandler instance on the
// main thread. In order to do safe PostTask-ing, the class is reference counted
// and checks for the existence of the RTCPeerConnectionHandler instance before
// delivering callbacks on the main thread.
class RTCPeerConnectionHandler::Observer
    : public base::RefCountedThreadSafe<RTCPeerConnectionHandler::Observer>,
      public PeerConnectionObserver,
      public RtcEventLogOutputSink {
 public:
  Observer(const base::WeakPtr<RTCPeerConnectionHandler>& handler,
           scoped_refptr<base::SingleThreadTaskRunner> task_runner)
      : handler_(handler), main_thread_(task_runner) {}

  // When an RTC event log is sent back from PeerConnection, it arrives here.
  void OnWebRtcEventLogWrite(const std::string& output) override {
    if (!main_thread_->BelongsToCurrentThread()) {
      main_thread_->PostTask(
          FROM_HERE,
          base::BindOnce(
              &RTCPeerConnectionHandler::Observer::OnWebRtcEventLogWrite, this,
              output));
    } else if (handler_) {
      handler_->OnWebRtcEventLogWrite(output);
    }
  }

 protected:
  friend class base::RefCountedThreadSafe<RTCPeerConnectionHandler::Observer>;
  ~Observer() override = default;

  // TODO(hbos): Remove once no longer mandatory to implement.
  void OnSignalingChange(PeerConnectionInterface::SignalingState) override {}
  void OnAddStream(rtc::scoped_refptr<MediaStreamInterface>) override {}
  void OnRemoveStream(rtc::scoped_refptr<MediaStreamInterface>) override {}

  void OnDataChannel(
      rtc::scoped_refptr<DataChannelInterface> data_channel) override {
    std::unique_ptr<RtcDataChannelHandler> handler(
        new RtcDataChannelHandler(main_thread_, data_channel));
    main_thread_->PostTask(
        FROM_HERE,
        base::BindOnce(&RTCPeerConnectionHandler::Observer::OnDataChannelImpl,
                       this, std::move(handler)));
  }

  void OnRenegotiationNeeded() override {
    if (!main_thread_->BelongsToCurrentThread()) {
      main_thread_->PostTask(
          FROM_HERE,
          base::BindOnce(
              &RTCPeerConnectionHandler::Observer::OnRenegotiationNeeded,
              this));
    } else if (handler_) {
      handler_->OnRenegotiationNeeded();
    }
  }

  void OnIceConnectionChange(
      PeerConnectionInterface::IceConnectionState new_state) override {
    if (!main_thread_->BelongsToCurrentThread()) {
      main_thread_->PostTask(
          FROM_HERE,
          base::BindOnce(
              &RTCPeerConnectionHandler::Observer::OnIceConnectionChange, this,
              new_state));
    } else if (handler_) {
      handler_->OnIceConnectionChange(new_state);
    }
  }

  void OnIceGatheringChange(
      PeerConnectionInterface::IceGatheringState new_state) override {
    if (!main_thread_->BelongsToCurrentThread()) {
      main_thread_->PostTask(
          FROM_HERE,
          base::BindOnce(
              &RTCPeerConnectionHandler::Observer::OnIceGatheringChange, this,
              new_state));
    } else if (handler_) {
      handler_->OnIceGatheringChange(new_state);
    }
  }

  void OnIceCandidate(const IceCandidateInterface* candidate) override {
    std::string sdp;
    if (!candidate->ToString(&sdp)) {
      NOTREACHED() << "OnIceCandidate: Could not get SDP string.";
      return;
    }

    main_thread_->PostTask(
        FROM_HERE,
        base::BindOnce(&RTCPeerConnectionHandler::Observer::OnIceCandidateImpl,
                       this, sdp, candidate->sdp_mid(),
                       candidate->sdp_mline_index(),
                       candidate->candidate().component(),
                       candidate->candidate().address().family()));
  }

  void OnDataChannelImpl(std::unique_ptr<RtcDataChannelHandler> handler) {
    DCHECK(main_thread_->BelongsToCurrentThread());
    if (handler_)
      handler_->OnDataChannel(std::move(handler));
  }

  void OnIceCandidateImpl(const std::string& sdp, const std::string& sdp_mid,
      int sdp_mline_index, int component, int address_family) {
    DCHECK(main_thread_->BelongsToCurrentThread());
    if (handler_) {
      handler_->OnIceCandidate(sdp, sdp_mid, sdp_mline_index, component,
          address_family);
    }
  }

 private:
  const base::WeakPtr<RTCPeerConnectionHandler> handler_;
  const scoped_refptr<base::SingleThreadTaskRunner> main_thread_;
};

RTCPeerConnectionHandler::RTCPeerConnectionHandler(
    blink::WebRTCPeerConnectionHandlerClient* client,
    PeerConnectionDependencyFactory* dependency_factory,
    scoped_refptr<base::SingleThreadTaskRunner> task_runner)
    : id_(base::ToUpperASCII(base::UnguessableToken::Create().ToString())),
      initialize_called_(false),
      client_(client),
      is_closed_(false),
      dependency_factory_(dependency_factory),
      track_adapter_map_(
          new WebRtcMediaStreamTrackAdapterMap(dependency_factory_,
                                               task_runner)),
      // This will be overwritten to the actual value used at Initialize().
      sdp_semantics_(blink::WebRTCSdpSemantics::kDefault),
      task_runner_(std::move(task_runner)),
      weak_factory_(this) {
  CHECK(client_);

  GetPeerConnectionHandlers()->insert(this);
}

RTCPeerConnectionHandler::~RTCPeerConnectionHandler() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());

  Stop();

  GetPeerConnectionHandlers()->erase(this);
  if (peer_connection_tracker_)
    peer_connection_tracker_->UnregisterPeerConnection(this);

  UMA_HISTOGRAM_COUNTS_10000(
      "WebRTC.NumDataChannelsPerPeerConnection", num_data_channels_created_);
}

void RTCPeerConnectionHandler::associateWithFrame(blink::WebLocalFrame* frame) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  DCHECK(frame);
  frame_ = frame;
}

bool RTCPeerConnectionHandler::Initialize(
    const blink::WebRTCConfiguration& server_configuration,
    const blink::WebMediaConstraints& options,
    blink::WebRTCSdpSemantics original_sdp_semantics_value) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  DCHECK(frame_);

  CHECK(!initialize_called_);
  initialize_called_ = true;

  peer_connection_tracker_ =
      RenderThreadImpl::current()->peer_connection_tracker()->AsWeakPtr();

  sdp_semantics_ = server_configuration.sdp_semantics;
  DCHECK(sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB ||
         sdp_semantics_ == blink::WebRTCSdpSemantics::kUnifiedPlan);
  GetNativeRtcConfiguration(server_configuration, &configuration_);

  // Choose between RTC smoothness algorithm and prerenderer smoothing.
  // Prerenderer smoothing is turned on if RTC smoothness is turned off.
  configuration_.set_prerenderer_smoothing(
      base::CommandLine::ForCurrentProcess()->HasSwitch(
          switches::kDisableRTCSmoothnessAlgorithm));

  configuration_.set_experiment_cpu_load_estimator(
      base::FeatureList::IsEnabled(media::kNewEncodeCpuLoadEstimator));

  // Copy all the relevant constraints into |config|.
  CopyConstraintsIntoRtcConfiguration(options, &configuration_);

  peer_connection_observer_ =
      new Observer(weak_factory_.GetWeakPtr(), task_runner_);
  native_peer_connection_ = dependency_factory_->CreatePeerConnection(
      configuration_, frame_, peer_connection_observer_.get());

  if (!native_peer_connection_.get()) {
    LOG(ERROR) << "Failed to initialize native PeerConnection.";
    return false;
  }

  if (peer_connection_tracker_) {
    peer_connection_tracker_->RegisterPeerConnection(this, configuration_,
                                                     options, frame_);
  }

  UMA_HISTOGRAM_ENUMERATION(
      "WebRTC.PeerConnection.SdpSemanticRequested",
      GetSdpSemanticRequested(original_sdp_semantics_value),
      kSdpSemanticRequestedMax);

  return true;
}

bool RTCPeerConnectionHandler::InitializeForTest(
    const blink::WebRTCConfiguration& server_configuration,
    const blink::WebMediaConstraints& options,
    const base::WeakPtr<PeerConnectionTracker>& peer_connection_tracker) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());

  CHECK(!initialize_called_);
  initialize_called_ = true;

  sdp_semantics_ = server_configuration.sdp_semantics;
  DCHECK(sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB ||
         sdp_semantics_ == blink::WebRTCSdpSemantics::kUnifiedPlan);
  GetNativeRtcConfiguration(server_configuration, &configuration_);

  peer_connection_observer_ =
      new Observer(weak_factory_.GetWeakPtr(), task_runner_);
  CopyConstraintsIntoRtcConfiguration(options, &configuration_);

  native_peer_connection_ = dependency_factory_->CreatePeerConnection(
      configuration_, nullptr, peer_connection_observer_.get());
  if (!native_peer_connection_.get()) {
    LOG(ERROR) << "Failed to initialize native PeerConnection.";
    return false;
  }
  peer_connection_tracker_ = peer_connection_tracker;
  return true;
}

void RTCPeerConnectionHandler::CreateOffer(
    const blink::WebRTCSessionDescriptionRequest& request,
    const blink::WebMediaConstraints& options) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::createOffer");

  scoped_refptr<CreateSessionDescriptionRequest> description_request(
      new rtc::RefCountedObject<CreateSessionDescriptionRequest>(
          task_runner_, request, weak_factory_.GetWeakPtr(),
          peer_connection_tracker_,
          PeerConnectionTracker::ACTION_CREATE_OFFER));

  // TODO(tommi): Do this asynchronously via e.g. PostTaskAndReply.
  webrtc::PeerConnectionInterface::RTCOfferAnswerOptions webrtc_options;
  ConvertConstraintsToWebrtcOfferOptions(options, &webrtc_options);
  native_peer_connection_->CreateOffer(description_request.get(),
                                       webrtc_options);

  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackCreateOffer(this, options);
}

void RTCPeerConnectionHandler::CreateOffer(
    const blink::WebRTCSessionDescriptionRequest& request,
    const blink::WebRTCOfferOptions& options) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::createOffer");

  scoped_refptr<CreateSessionDescriptionRequest> description_request(
      new rtc::RefCountedObject<CreateSessionDescriptionRequest>(
          task_runner_, request, weak_factory_.GetWeakPtr(),
          peer_connection_tracker_,
          PeerConnectionTracker::ACTION_CREATE_OFFER));

  // TODO(tommi): Do this asynchronously via e.g. PostTaskAndReply.
  webrtc::PeerConnectionInterface::RTCOfferAnswerOptions webrtc_options;
  ConvertOfferOptionsToWebrtcOfferOptions(options, &webrtc_options);
  native_peer_connection_->CreateOffer(description_request.get(),
                                       webrtc_options);

  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackCreateOffer(this, options);
}

void RTCPeerConnectionHandler::CreateAnswer(
    const blink::WebRTCSessionDescriptionRequest& request,
    const blink::WebMediaConstraints& options) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::createAnswer");
  scoped_refptr<CreateSessionDescriptionRequest> description_request(
      new rtc::RefCountedObject<CreateSessionDescriptionRequest>(
          task_runner_, request, weak_factory_.GetWeakPtr(),
          peer_connection_tracker_,
          PeerConnectionTracker::ACTION_CREATE_ANSWER));
  webrtc::PeerConnectionInterface::RTCOfferAnswerOptions webrtc_options;
  ConvertConstraintsToWebrtcOfferOptions(options, &webrtc_options);
  // TODO(tommi): Do this asynchronously via e.g. PostTaskAndReply.
  native_peer_connection_->CreateAnswer(description_request.get(),
                                        webrtc_options);

  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackCreateAnswer(this, options);
}

void RTCPeerConnectionHandler::CreateAnswer(
    const blink::WebRTCSessionDescriptionRequest& request,
    const blink::WebRTCAnswerOptions& options) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::createAnswer");
  scoped_refptr<CreateSessionDescriptionRequest> description_request(
      new rtc::RefCountedObject<CreateSessionDescriptionRequest>(
          task_runner_, request, weak_factory_.GetWeakPtr(),
          peer_connection_tracker_,
          PeerConnectionTracker::ACTION_CREATE_ANSWER));
  // TODO(tommi): Do this asynchronously via e.g. PostTaskAndReply.
  webrtc::PeerConnectionInterface::RTCOfferAnswerOptions webrtc_options;
  ConvertAnswerOptionsToWebrtcAnswerOptions(options, &webrtc_options);
  native_peer_connection_->CreateAnswer(description_request.get(),
                                        webrtc_options);

  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackCreateAnswer(this, options);
}

bool IsOfferOrAnswer(const webrtc::SessionDescriptionInterface* native_desc) {
  DCHECK(native_desc);
  return native_desc->type() == "offer" || native_desc->type() == "answer";
}

void RTCPeerConnectionHandler::SetLocalDescription(
    const blink::WebRTCVoidRequest& request,
    const blink::WebRTCSessionDescription& description) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::setLocalDescription");

  std::string sdp = description.Sdp().Utf8();
  std::string type = description.GetType().Utf8();

  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackSetSessionDescription(
        this, sdp, type, PeerConnectionTracker::SOURCE_LOCAL);
  }

  webrtc::SdpParseError error;
  // Since CreateNativeSessionDescription uses the dependency factory, we need
  // to make this call on the current thread to be safe.
  webrtc::SessionDescriptionInterface* native_desc =
      CreateNativeSessionDescription(sdp, type, &error);
  if (!native_desc) {
    std::string reason_str = "Failed to parse SessionDescription. ";
    reason_str.append(error.line);
    reason_str.append(" ");
    reason_str.append(error.description);
    LOG(ERROR) << reason_str;
    request.RequestFailed(webrtc::RTCError(webrtc::RTCErrorType::INTERNAL_ERROR,
                                           std::move(reason_str)));
    if (peer_connection_tracker_) {
      peer_connection_tracker_->TrackSessionDescriptionCallback(
          this, PeerConnectionTracker::ACTION_SET_LOCAL_DESCRIPTION,
          "OnFailure", reason_str);
    }
    return;
  }

  if (!first_local_description_ && IsOfferOrAnswer(native_desc)) {
    first_local_description_.reset(new FirstSessionDescription(native_desc));
    if (first_remote_description_) {
      ReportFirstSessionDescriptions(
          *first_local_description_,
          *first_remote_description_);
    }
  }

  scoped_refptr<WebRtcSetDescriptionObserverImpl> content_observer(
      new WebRtcSetDescriptionObserverImpl(
          weak_factory_.GetWeakPtr(), request, peer_connection_tracker_,
          task_runner_, PeerConnectionTracker::ACTION_SET_LOCAL_DESCRIPTION,
          sdp_semantics_));

  bool surface_receivers_only =
      (sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB);
  scoped_refptr<webrtc::SetSessionDescriptionObserver> webrtc_observer(
      WebRtcSetLocalDescriptionObserverHandler::Create(
          task_runner_, signaling_thread(), native_peer_connection_,
          track_adapter_map_, content_observer, surface_receivers_only)
          .get());

  signaling_thread()->PostTask(
      FROM_HERE,
      base::BindOnce(
          &RunClosureWithTrace,
          base::Bind(&webrtc::PeerConnectionInterface::SetLocalDescription,
                     native_peer_connection_,
                     base::RetainedRef(webrtc_observer),
                     base::Unretained(native_desc)),
          "SetLocalDescription"));
}

void RTCPeerConnectionHandler::SetRemoteDescription(
    const blink::WebRTCVoidRequest& request,
    const blink::WebRTCSessionDescription& description) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::setRemoteDescription");
  std::string sdp = description.Sdp().Utf8();
  std::string type = description.GetType().Utf8();

  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackSetSessionDescription(
        this, sdp, type, PeerConnectionTracker::SOURCE_REMOTE);
  }

  webrtc::SdpParseError error;
  // Since CreateNativeSessionDescription uses the dependency factory, we need
  // to make this call on the current thread to be safe.
  std::unique_ptr<webrtc::SessionDescriptionInterface> native_desc(
      CreateNativeSessionDescription(sdp, type, &error));
  if (!native_desc) {
    std::string reason_str = "Failed to parse SessionDescription. ";
    reason_str.append(error.line);
    reason_str.append(" ");
    reason_str.append(error.description);
    LOG(ERROR) << reason_str;
    request.RequestFailed(webrtc::RTCError(
        webrtc::RTCErrorType::UNSUPPORTED_OPERATION, std::move(reason_str)));
    if (peer_connection_tracker_) {
      peer_connection_tracker_->TrackSessionDescriptionCallback(
          this, PeerConnectionTracker::ACTION_SET_REMOTE_DESCRIPTION,
          "OnFailure", reason_str);
    }
    return;
  }

  if (!first_remote_description_ && IsOfferOrAnswer(native_desc.get())) {
    first_remote_description_.reset(
        new FirstSessionDescription(native_desc.get()));
    if (first_local_description_) {
      ReportFirstSessionDescriptions(
          *first_local_description_,
          *first_remote_description_);
    }
  }

  scoped_refptr<WebRtcSetDescriptionObserverImpl> content_observer(
      new WebRtcSetDescriptionObserverImpl(
          weak_factory_.GetWeakPtr(), request, peer_connection_tracker_,
          task_runner_, PeerConnectionTracker::ACTION_SET_REMOTE_DESCRIPTION,
          sdp_semantics_));

  bool surface_receivers_only =
      (sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB);
  rtc::scoped_refptr<webrtc::SetRemoteDescriptionObserverInterface>
      webrtc_observer(WebRtcSetRemoteDescriptionObserverHandler::Create(
                          task_runner_, signaling_thread(),
                          native_peer_connection_, track_adapter_map_,
                          content_observer, surface_receivers_only)
                          .get());

  signaling_thread()->PostTask(
      FROM_HERE,
      base::BindOnce(
          &RunClosureWithTrace,
          base::Bind(
              static_cast<void (webrtc::PeerConnectionInterface::*)(
                  std::unique_ptr<webrtc::SessionDescriptionInterface>,
                  rtc::scoped_refptr<
                      webrtc::SetRemoteDescriptionObserverInterface>)>(
                  &webrtc::PeerConnectionInterface::SetRemoteDescription),
              native_peer_connection_, base::Passed(&native_desc),
              webrtc_observer),
          "SetRemoteDescription"));
}

blink::WebRTCSessionDescription RTCPeerConnectionHandler::LocalDescription() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::localDescription");

  // Since local_description returns a pointer to a non-reference-counted object
  // that lives on the signaling thread, we cannot fetch a pointer to it and use
  // it directly here. Instead, we access the object completely on the signaling
  // thread.
  std::string sdp, type;
  base::Callback<const webrtc::SessionDescriptionInterface*()> description_cb =
      base::Bind(&webrtc::PeerConnectionInterface::local_description,
                 native_peer_connection_);
  RunSynchronousClosureOnSignalingThread(
      base::Bind(&GetSdpAndTypeFromSessionDescription,
                 std::move(description_cb), base::Unretained(&sdp),
                 base::Unretained(&type)),
      "localDescription");

  return CreateWebKitSessionDescription(sdp, type);
}

blink::WebRTCSessionDescription RTCPeerConnectionHandler::RemoteDescription() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::remoteDescription");
  // Since local_description returns a pointer to a non-reference-counted object
  // that lives on the signaling thread, we cannot fetch a pointer to it and use
  // it directly here. Instead, we access the object completely on the signaling
  // thread.
  std::string sdp, type;
  base::Callback<const webrtc::SessionDescriptionInterface*()> description_cb =
      base::Bind(&webrtc::PeerConnectionInterface::remote_description,
                 native_peer_connection_);
  RunSynchronousClosureOnSignalingThread(
      base::Bind(&GetSdpAndTypeFromSessionDescription,
                 std::move(description_cb), base::Unretained(&sdp),
                 base::Unretained(&type)),
      "remoteDescription");

  return CreateWebKitSessionDescription(sdp, type);
}

webrtc::RTCErrorType RTCPeerConnectionHandler::SetConfiguration(
    const blink::WebRTCConfiguration& blink_config) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::setConfiguration");
  GetNativeRtcConfiguration(blink_config, &configuration_);

  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackSetConfiguration(this, configuration_);

  webrtc::RTCError webrtc_error;
  bool ret =
      native_peer_connection_->SetConfiguration(configuration_, &webrtc_error);
  // The boolean return value is made redundant by the error output param; just
  // DCHECK that they're consistent.
  DCHECK_EQ(ret, webrtc_error.type() == webrtc::RTCErrorType::NONE);
  return webrtc_error.type();
}

bool RTCPeerConnectionHandler::AddICECandidate(
    const blink::WebRTCVoidRequest& request,
    scoped_refptr<blink::WebRTCICECandidate> candidate) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::addICECandidate");
  // Libjingle currently does not accept callbacks for addICECandidate.
  // For that reason we are going to call callbacks from here.

  // TODO(tommi): Instead of calling addICECandidate here, we can do a
  // PostTaskAndReply kind of a thing.
  bool result = AddICECandidate(std::move(candidate));
  task_runner_->PostTask(
      FROM_HERE,
      base::BindOnce(&RTCPeerConnectionHandler::OnaddICECandidateResult,
                     weak_factory_.GetWeakPtr(), request, result));
  // On failure callback will be triggered.
  return true;
}

bool RTCPeerConnectionHandler::AddICECandidate(
    scoped_refptr<blink::WebRTCICECandidate> candidate) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::addICECandidate");
  std::unique_ptr<webrtc::IceCandidateInterface> native_candidate(
      dependency_factory_->CreateIceCandidate(candidate->SdpMid().Utf8(),
                                              candidate->SdpMLineIndex(),
                                              candidate->Candidate().Utf8()));
  bool return_value = false;

  if (native_candidate) {
    return_value =
        native_peer_connection_->AddIceCandidate(native_candidate.get());
    LOG_IF(ERROR, !return_value) << "Error processing ICE candidate.";
  } else {
    LOG(ERROR) << "Could not create native ICE candidate.";
  }

  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackAddIceCandidate(
        this, std::move(candidate), PeerConnectionTracker::SOURCE_REMOTE,
        return_value);
  }
  return return_value;
}

void RTCPeerConnectionHandler::OnaddICECandidateResult(
    const blink::WebRTCVoidRequest& webkit_request, bool result) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnaddICECandidateResult");
  if (!result) {
    // We don't have the actual error code from the libjingle, so for now
    // using a generic error string.
    return webkit_request.RequestFailed(
        webrtc::RTCError(webrtc::RTCErrorType::UNSUPPORTED_OPERATION,
                         std::move("Error processing ICE candidate")));
  }

  return webkit_request.RequestSucceeded();
}

void RTCPeerConnectionHandler::GetStats(
    const blink::WebRTCStatsRequest& request) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  scoped_refptr<LocalRTCStatsRequest> inner_request(
      new rtc::RefCountedObject<LocalRTCStatsRequest>(request));
  getStats(inner_request);
}

void RTCPeerConnectionHandler::getStats(
    const scoped_refptr<LocalRTCStatsRequest>& request) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::getStats");

  rtc::scoped_refptr<webrtc::StatsObserver> observer(
      new rtc::RefCountedObject<StatsResponse>(request, task_runner_));

  rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> selector;
  if (request->hasSelector()) {
    auto track_adapter_ref =
        track_adapter_map_->GetLocalTrackAdapter(request->component());
    if (!track_adapter_ref) {
      track_adapter_ref =
          track_adapter_map_->GetRemoteTrackAdapter(request->component());
    }
    if (track_adapter_ref)
      selector = track_adapter_ref->webrtc_track();
  }

  GetStats(observer, webrtc::PeerConnectionInterface::kStatsOutputLevelStandard,
           std::move(selector));
}

// TODO(tommi,hbos): It's weird to have three {g|G}etStats methods for the
// legacy stats collector API and even more for the new stats API. Clean it up.
// TODO(hbos): Rename old |getStats| and related functions to "getLegacyStats",
// rename new |getStats|'s helper functions from "GetRTCStats*" to "GetStats*".
void RTCPeerConnectionHandler::GetStats(
    webrtc::StatsObserver* observer,
    webrtc::PeerConnectionInterface::StatsOutputLevel level,
    rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> selector) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  signaling_thread()->PostTask(
      FROM_HERE,
      base::BindOnce(&GetStatsOnSignalingThread, native_peer_connection_, level,
                     base::WrapRefCounted(observer), std::move(selector)));
}

void RTCPeerConnectionHandler::GetStats(
    std::unique_ptr<blink::WebRTCStatsReportCallback> callback) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  signaling_thread()->PostTask(
      FROM_HERE, base::BindOnce(&GetRTCStatsOnSignalingThread, task_runner_,
                                native_peer_connection_, std::move(callback)));
}

webrtc::RTCErrorOr<std::unique_ptr<blink::WebRTCRtpTransceiver>>
RTCPeerConnectionHandler::AddTransceiverWithTrack(
    const blink::WebMediaStreamTrack& web_track,
    const webrtc::RtpTransceiverInit& init) {
  DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  std::unique_ptr<WebRtcMediaStreamTrackAdapterMap::AdapterRef> track_ref =
      track_adapter_map_->GetOrCreateLocalTrackAdapter(web_track);
  TransceiverStateSurfacer transceiver_state_surfacer(task_runner_,
                                                      signaling_thread());
  webrtc::RTCErrorOr<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>>
      error_or_transceiver;
  RunSynchronousClosureOnSignalingThread(
      base::BindRepeating(
          &RTCPeerConnectionHandler::AddTransceiverWithTrackOnSignalingThread,
          base::Unretained(this), base::RetainedRef(track_ref->webrtc_track()),
          base::ConstRef(init), base::Unretained(&transceiver_state_surfacer),
          base::Unretained(&error_or_transceiver)),
      "AddTransceiverWithTrackOnSignalingThread");
  if (!error_or_transceiver.ok()) {
    // Don't leave the surfacer in a pending state.
    transceiver_state_surfacer.ObtainStates();
    return error_or_transceiver.MoveError();
  }

  auto transceiver_states = transceiver_state_surfacer.ObtainStates();
  auto transceiver =
      CreateOrUpdateTransceiver(std::move(transceiver_states[0]));
  std::unique_ptr<blink::WebRTCRtpTransceiver> web_transceiver =
      std::move(transceiver);
  return std::move(web_transceiver);
}

void RTCPeerConnectionHandler::AddTransceiverWithTrackOnSignalingThread(
    rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> webrtc_track,
    webrtc::RtpTransceiverInit init,
    TransceiverStateSurfacer* transceiver_state_surfacer,
    webrtc::RTCErrorOr<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>>*
        error_or_transceiver) {
  *error_or_transceiver =
      native_peer_connection_->AddTransceiver(webrtc_track, init);
  std::vector<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>> transceivers;
  if (error_or_transceiver->ok())
    transceivers.push_back(error_or_transceiver->value());
  transceiver_state_surfacer->Initialize(track_adapter_map_, transceivers);
}

webrtc::RTCErrorOr<std::unique_ptr<blink::WebRTCRtpTransceiver>>
RTCPeerConnectionHandler::AddTransceiverWithKind(
    std::string kind,
    const webrtc::RtpTransceiverInit& init) {
  DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  cricket::MediaType media_type;
  if (kind == webrtc::MediaStreamTrackInterface::kAudioKind) {
    media_type = cricket::MEDIA_TYPE_AUDIO;
  } else {
    DCHECK_EQ(kind, webrtc::MediaStreamTrackInterface::kVideoKind);
    media_type = cricket::MEDIA_TYPE_VIDEO;
  }
  TransceiverStateSurfacer transceiver_state_surfacer(task_runner_,
                                                      signaling_thread());
  webrtc::RTCErrorOr<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>>
      error_or_transceiver;
  RunSynchronousClosureOnSignalingThread(
      base::BindRepeating(&RTCPeerConnectionHandler::
                              AddTransceiverWithMediaTypeOnSignalingThread,
                          base::Unretained(this), base::ConstRef(media_type),
                          base::ConstRef(init),
                          base::Unretained(&transceiver_state_surfacer),
                          base::Unretained(&error_or_transceiver)),
      "AddTransceiverWithMediaTypeOnSignalingThread");
  if (!error_or_transceiver.ok()) {
    // Don't leave the surfacer in a pending state.
    transceiver_state_surfacer.ObtainStates();
    return error_or_transceiver.MoveError();
  }

  auto transceiver_states = transceiver_state_surfacer.ObtainStates();
  auto transceiver =
      CreateOrUpdateTransceiver(std::move(transceiver_states[0]));
  std::unique_ptr<blink::WebRTCRtpTransceiver> web_transceiver =
      std::move(transceiver);
  return std::move(web_transceiver);
}

void RTCPeerConnectionHandler::AddTransceiverWithMediaTypeOnSignalingThread(
    cricket::MediaType media_type,
    webrtc::RtpTransceiverInit init,
    TransceiverStateSurfacer* transceiver_state_surfacer,
    webrtc::RTCErrorOr<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>>*
        error_or_transceiver) {
  *error_or_transceiver =
      native_peer_connection_->AddTransceiver(media_type, init);
  std::vector<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>> transceivers;
  if (error_or_transceiver->ok())
    transceivers.push_back(error_or_transceiver->value());
  transceiver_state_surfacer->Initialize(track_adapter_map_, transceivers);
}

webrtc::RTCErrorOr<std::unique_ptr<blink::WebRTCRtpTransceiver>>
RTCPeerConnectionHandler::AddTrack(
    const blink::WebMediaStreamTrack& track,
    const blink::WebVector<blink::WebMediaStream>& streams) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::AddTrack");

  std::unique_ptr<WebRtcMediaStreamTrackAdapterMap::AdapterRef> track_ref =
      track_adapter_map_->GetOrCreateLocalTrackAdapter(track);
  std::vector<std::string> stream_ids(streams.size());
  for (size_t i = 0; i < streams.size(); ++i)
    stream_ids[i] = streams[i].Id().Utf8();

  // Invoke native AddTrack() on the signaling thread and surface the resulting
  // transceiver (Plan B: sender only).
  // TODO(hbos): Implement and surface full transceiver support under Unified
  // Plan. https://crbug.com/777617
  TransceiverStateSurfacer transceiver_state_surfacer(task_runner_,
                                                      signaling_thread());
  webrtc::RTCErrorOr<rtc::scoped_refptr<webrtc::RtpSenderInterface>>
      error_or_sender;
  RunSynchronousClosureOnSignalingThread(
      base::BindRepeating(
          &RTCPeerConnectionHandler::AddTrackOnSignalingThread,
          base::Unretained(this), base::RetainedRef(track_ref->webrtc_track()),
          std::move(stream_ids), base::Unretained(&transceiver_state_surfacer),
          base::Unretained(&error_or_sender)),
      "AddTrackOnSignalingThread");
  DCHECK(transceiver_state_surfacer.is_initialized());
  if (!error_or_sender.ok()) {
    // Don't leave the surfacer in a pending state.
    transceiver_state_surfacer.ObtainStates();
    return error_or_sender.MoveError();
  }
  track_metrics_.AddTrack(MediaStreamTrackMetrics::Direction::kSend,
                          MediaStreamTrackMetricsKind(track),
                          track.Id().Utf8());

  auto transceiver_states = transceiver_state_surfacer.ObtainStates();
  DCHECK_EQ(transceiver_states.size(), 1u);
  auto transceiver_state = std::move(transceiver_states[0]);

  std::unique_ptr<blink::WebRTCRtpTransceiver> web_transceiver;
  const blink::WebRTCRtpSender* web_sender = nullptr;
  const blink::WebRTCRtpReceiver* web_receiver = nullptr;
  if (sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB) {
    // Plan B: Create sender only.
    DCHECK(transceiver_state.sender_state());
    auto webrtc_sender = transceiver_state.sender_state()->webrtc_sender();
    DCHECK(FindSender(RTCRtpSender::getId(webrtc_sender.get())) ==
           rtp_senders_.end());
    RtpSenderState sender_state = transceiver_state.MoveSenderState();
    DCHECK(sender_state.is_initialized());
    rtp_senders_.push_back(std::make_unique<RTCRtpSender>(
        native_peer_connection_, track_adapter_map_, std::move(sender_state)));
    web_sender = rtp_senders_.back().get();
    web_transceiver = std::make_unique<RTCRtpSenderOnlyTransceiver>(
        rtp_senders_.back()->ShallowCopy());
  } else {
    DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
    // Unified Plan: Create or recycle a transceiver.
    auto transceiver = CreateOrUpdateTransceiver(std::move(transceiver_state));
    web_sender = transceiver->content_sender();
    web_receiver = transceiver->content_receiver();
    web_transceiver = std::move(transceiver);
  }
  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackAddTransceiver(
        this, PeerConnectionTracker::TransceiverUpdatedReason::kAddTrack,
        web_sender, web_receiver);
  }
  for (const auto& stream_id : rtp_senders_.back()->state().stream_ids()) {
    if (GetLocalStreamUsageCount(rtp_senders_, stream_id) == 1u) {
      // This is the first occurrence of this stream.
      PerSessionWebRTCAPIMetrics::GetInstance()->IncrementStreamCounter();
    }
  }
  return web_transceiver;
}

void RTCPeerConnectionHandler::AddTrackOnSignalingThread(
    rtc::scoped_refptr<webrtc::MediaStreamTrackInterface> track,
    std::vector<std::string> stream_ids,
    TransceiverStateSurfacer* transceiver_state_surfacer,
    webrtc::RTCErrorOr<rtc::scoped_refptr<webrtc::RtpSenderInterface>>*
        error_or_sender) {
  *error_or_sender = native_peer_connection_->AddTrack(track, stream_ids);
  std::vector<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>> transceivers;
  if (error_or_sender->ok()) {
    auto sender = error_or_sender->value();
    if (sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB) {
      transceivers = {new SurfaceSenderStateOnly(sender)};
    } else {
      DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
      rtc::scoped_refptr<webrtc::RtpTransceiverInterface>
          transceiver_for_sender = nullptr;
      for (const auto& transceiver :
           native_peer_connection_->GetTransceivers()) {
        if (transceiver->sender() == sender) {
          transceiver_for_sender = transceiver;
          break;
        }
      }
      DCHECK(transceiver_for_sender);
      transceivers = {transceiver_for_sender};
    }
  }
  transceiver_state_surfacer->Initialize(track_adapter_map_,
                                         std::move(transceivers));
}

webrtc::RTCErrorOr<std::unique_ptr<blink::WebRTCRtpTransceiver>>
RTCPeerConnectionHandler::RemoveTrack(blink::WebRTCRtpSender* web_sender) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::RemoveTrack");
  if (sdp_semantics_ == blink::WebRTCSdpSemantics::kPlanB) {
    if (RemoveTrackPlanB(web_sender)) {
      // In Plan B, null indicates success.
      std::unique_ptr<blink::WebRTCRtpTransceiver> web_transceiver = nullptr;
      return std::move(web_transceiver);
    }
    // TODO(hbos): Surface RTCError from third_party/webrtc when
    // peerconnectioninterface.h is updated. https://crbug.com/webrtc/9534
    return webrtc::RTCError(webrtc::RTCErrorType::INVALID_STATE);
  }
  DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
  return RemoveTrackUnifiedPlan(web_sender);
}

bool RTCPeerConnectionHandler::RemoveTrackPlanB(
    blink::WebRTCRtpSender* web_sender) {
  DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kPlanB);
  auto web_track = web_sender->Track();
  auto it = FindSender(web_sender->Id());
  if (it == rtp_senders_.end())
    return false;
  if (!(*it)->RemoveFromPeerConnection(native_peer_connection_.get()))
    return false;
  track_metrics_.RemoveTrack(MediaStreamTrackMetrics::Direction::kSend,
                             MediaStreamTrackMetricsKind(web_track),
                             web_track.Id().Utf8());
  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackRemoveTransceiver(
        this, PeerConnectionTracker::TransceiverUpdatedReason::kRemoveTrack,
        (*it).get(), nullptr);
  }
  std::vector<std::string> stream_ids = (*it)->state().stream_ids();
  rtp_senders_.erase(it);
  for (const auto& stream_id : stream_ids) {
    if (GetLocalStreamUsageCount(rtp_senders_, stream_id) == 0u) {
      // This was the last occurrence of this stream.
      PerSessionWebRTCAPIMetrics::GetInstance()->DecrementStreamCounter();
    }
  }
  return true;
}

webrtc::RTCErrorOr<std::unique_ptr<blink::WebRTCRtpTransceiver>>
RTCPeerConnectionHandler::RemoveTrackUnifiedPlan(
    blink::WebRTCRtpSender* web_sender) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
  auto it = FindSender(web_sender->Id());
  if (it == rtp_senders_.end())
    return webrtc::RTCError(webrtc::RTCErrorType::INVALID_PARAMETER);
  const auto& sender = *it;
  auto webrtc_sender = sender->state().webrtc_sender();

  TransceiverStateSurfacer transceiver_state_surfacer(task_runner_,
                                                      signaling_thread());
  bool result;
  RunSynchronousClosureOnSignalingThread(
      base::BindRepeating(
          &RTCPeerConnectionHandler::RemoveTrackUnifiedPlanOnSignalingThread,
          base::Unretained(this), base::RetainedRef(webrtc_sender),
          base::Unretained(&transceiver_state_surfacer),
          base::Unretained(&result)),
      "RemoveTrackUnifiedPlanOnSignalingThread");
  DCHECK(transceiver_state_surfacer.is_initialized());
  if (!result) {
    // Don't leave the surfacer in a pending state.
    transceiver_state_surfacer.ObtainStates();
    // TODO(hbos): Surface RTCError from third_party/webrtc when
    // peerconnectioninterface.h is updated. https://crbug.com/webrtc/9534
    return webrtc::RTCError(webrtc::RTCErrorType::INTERNAL_ERROR);
  }

  auto transceiver_states = transceiver_state_surfacer.ObtainStates();
  DCHECK_EQ(transceiver_states.size(), 1u);
  auto transceiver_state = std::move(transceiver_states[0]);

  // Return the update transceiver state.
  auto transceiver = CreateOrUpdateTransceiver(std::move(transceiver_state));
  std::unique_ptr<blink::WebRTCRtpTransceiver> web_transceiver =
      std::move(transceiver);
  return web_transceiver;
}

void RTCPeerConnectionHandler::RemoveTrackUnifiedPlanOnSignalingThread(
    rtc::scoped_refptr<webrtc::RtpSenderInterface> sender,
    TransceiverStateSurfacer* transceiver_state_surfacer,
    bool* result) {
  *result = native_peer_connection_->RemoveTrack(sender);
  std::vector<rtc::scoped_refptr<webrtc::RtpTransceiverInterface>> transceivers;
  if (*result) {
    rtc::scoped_refptr<webrtc::RtpTransceiverInterface> transceiver_for_sender =
        nullptr;
    for (const auto& transceiver : native_peer_connection_->GetTransceivers()) {
      if (transceiver->sender() == sender) {
        transceiver_for_sender = transceiver;
        break;
      }
    }
    DCHECK(transceiver_for_sender);
    transceivers = {transceiver_for_sender};
  }
  transceiver_state_surfacer->Initialize(track_adapter_map_,
                                         std::move(transceivers));
}

void RTCPeerConnectionHandler::CloseClientPeerConnection() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  if (!is_closed_)
    client_->ClosePeerConnection();
}

void RTCPeerConnectionHandler::StartEventLog(IPC::PlatformFileForTransit file,
                                             int64_t max_file_size_bytes) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  DCHECK(file != IPC::InvalidPlatformFileForTransit());
  // TODO(eladalon): StartRtcEventLog() return value is not useful; remove it
  // or find a way to be able to use it.
  // https://crbug.com/775415
  native_peer_connection_->StartRtcEventLog(
      IPC::PlatformFileForTransitToPlatformFile(file), max_file_size_bytes);
}

void RTCPeerConnectionHandler::StartEventLog() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  // TODO(eladalon): StartRtcEventLog() return value is not useful; remove it
  // or find a way to be able to use it.
  // https://crbug.com/775415
  native_peer_connection_->StartRtcEventLog(
      std::make_unique<RtcEventLogOutputSinkProxy>(
          peer_connection_observer_.get()),
      webrtc::RtcEventLog::kImmediateOutput);
}

void RTCPeerConnectionHandler::StopEventLog() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  native_peer_connection_->StopRtcEventLog();
}

void RTCPeerConnectionHandler::OnWebRtcEventLogWrite(
    const std::string& output) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackRtcEventLogWrite(this, output);
  }
}

blink::WebRTCDataChannelHandler* RTCPeerConnectionHandler::CreateDataChannel(
    const blink::WebString& label,
    const blink::WebRTCDataChannelInit& init) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::createDataChannel");
  DVLOG(1) << "createDataChannel label " << label.Utf8();

  webrtc::DataChannelInit config;
  // TODO(jiayl): remove the deprecated reliable field once Libjingle is updated
  // to handle that.
  config.reliable = false;
  config.id = init.id;
  config.ordered = init.ordered;
  config.negotiated = init.negotiated;
  config.maxRetransmits = init.max_retransmits;
  config.maxRetransmitTime = init.max_retransmit_time;
  config.protocol = init.protocol.Utf8();

  rtc::scoped_refptr<webrtc::DataChannelInterface> webrtc_channel(
      native_peer_connection_->CreateDataChannel(label.Utf8(), &config));
  if (!webrtc_channel) {
    DLOG(ERROR) << "Could not create native data channel.";
    return nullptr;
  }
  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackCreateDataChannel(
        this, webrtc_channel.get(), PeerConnectionTracker::SOURCE_LOCAL);
  }

  ++num_data_channels_created_;

  return new RtcDataChannelHandler(task_runner_, webrtc_channel);
}

void RTCPeerConnectionHandler::Stop() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  DVLOG(1) << "RTCPeerConnectionHandler::stop";

  if (is_closed_ || !native_peer_connection_.get())
    return;  // Already stopped.

  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackStop(this);

  native_peer_connection_->Close();

  // This object may no longer forward call backs to blink.
  is_closed_ = true;
}

blink::WebString RTCPeerConnectionHandler::Id() const {
  return blink::WebString::FromASCII(id_);
}

void RTCPeerConnectionHandler::OnSignalingChange(
    webrtc::PeerConnectionInterface::SignalingState new_state) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnSignalingChange");

  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackSignalingStateChange(this, new_state);
  if (!is_closed_)
    client_->DidChangeSignalingState(new_state);
}

// Called any time the IceConnectionState changes
void RTCPeerConnectionHandler::OnIceConnectionChange(
    webrtc::PeerConnectionInterface::IceConnectionState new_state) {
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnIceConnectionChange");
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  ReportICEState(new_state);
  if (new_state == webrtc::PeerConnectionInterface::kIceConnectionChecking) {
    ice_connection_checking_start_ = base::TimeTicks::Now();
  } else if (new_state ==
      webrtc::PeerConnectionInterface::kIceConnectionConnected) {
    // If the state becomes connected, send the time needed for PC to become
    // connected from checking to UMA. UMA data will help to know how much
    // time needed for PC to connect with remote peer.
    if (ice_connection_checking_start_.is_null()) {
      // From UMA, we have observed a large number of calls falling into the
      // overflow buckets. One possibility is that the Checking is not signaled
      // before Connected. This is to guard against that situation to make the
      // metric more robust.
      UMA_HISTOGRAM_MEDIUM_TIMES("WebRTC.PeerConnection.TimeToConnect",
                                 base::TimeDelta());
    } else {
    UMA_HISTOGRAM_MEDIUM_TIMES(
        "WebRTC.PeerConnection.TimeToConnect",
        base::TimeTicks::Now() - ice_connection_checking_start_);
    }
  }

  track_metrics_.IceConnectionChange(new_state);
  blink::WebRTCPeerConnectionHandlerClient::ICEConnectionState state =
      GetWebKitIceConnectionState(new_state);
  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackIceConnectionStateChange(this, state);
  if (!is_closed_)
    client_->DidChangeICEConnectionState(state);
}

// Called any time the IceGatheringState changes
void RTCPeerConnectionHandler::OnIceGatheringChange(
    webrtc::PeerConnectionInterface::IceGatheringState new_state) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnIceGatheringChange");

  if (new_state == webrtc::PeerConnectionInterface::kIceGatheringComplete) {
    UMA_HISTOGRAM_COUNTS_100("WebRTC.PeerConnection.IPv4LocalCandidates",
                             num_local_candidates_ipv4_);

    UMA_HISTOGRAM_COUNTS_100("WebRTC.PeerConnection.IPv6LocalCandidates",
                             num_local_candidates_ipv6_);
  } else if (new_state ==
             webrtc::PeerConnectionInterface::kIceGatheringGathering) {
    // ICE restarts will change gathering state back to "gathering",
    // reset the counter.
    ResetUMAStats();
  }

  blink::WebRTCPeerConnectionHandlerClient::ICEGatheringState state =
      GetWebKitIceGatheringState(new_state);
  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackIceGatheringStateChange(this, state);
  if (!is_closed_)
    client_->DidChangeICEGatheringState(state);
}

void RTCPeerConnectionHandler::OnRenegotiationNeeded() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnRenegotiationNeeded");
  if (peer_connection_tracker_)
    peer_connection_tracker_->TrackOnRenegotiationNeeded(this);
  if (!is_closed_)
    client_->NegotiationNeeded();
}

void RTCPeerConnectionHandler::OnAddReceiverPlanB(
    RtpReceiverState receiver_state) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  DCHECK(receiver_state.is_initialized());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnAddReceiverPlanB");
  auto web_track = receiver_state.track_ref()->web_track();
  // Update metrics.
  track_metrics_.AddTrack(MediaStreamTrackMetrics::Direction::kReceive,
                          MediaStreamTrackMetricsKind(web_track),
                          web_track.Id().Utf8());
  for (const auto& stream_id : receiver_state.stream_ids()) {
    // New remote stream?
    if (!IsRemoteStream(rtp_receivers_, stream_id))
      PerSessionWebRTCAPIMetrics::GetInstance()->IncrementStreamCounter();
  }
  uintptr_t receiver_id =
      RTCRtpReceiver::getId(receiver_state.webrtc_receiver().get());
  DCHECK(rtp_receivers_.find(receiver_id) == rtp_receivers_.end());
  const std::unique_ptr<RTCRtpReceiver>& rtp_receiver =
      rtp_receivers_
          .insert(std::make_pair(receiver_id, std::make_unique<RTCRtpReceiver>(
                                                  native_peer_connection_,
                                                  std::move(receiver_state))))
          .first->second;
  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackAddTransceiver(
        this,
        PeerConnectionTracker::TransceiverUpdatedReason::kSetRemoteDescription,
        nullptr, rtp_receiver.get());
  }
  if (!is_closed_)
    client_->DidAddReceiverPlanB(rtp_receiver->ShallowCopy());
}

void RTCPeerConnectionHandler::OnRemoveReceiverPlanB(uintptr_t receiver_id) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnRemoveReceiverPlanB");

  auto it = rtp_receivers_.find(receiver_id);
  DCHECK(it != rtp_receivers_.end());
  auto receiver = it->second->ShallowCopy();
  rtp_receivers_.erase(it);
  // Update metrics.
  track_metrics_.RemoveTrack(MediaStreamTrackMetrics::Direction::kReceive,
                             MediaStreamTrackMetricsKind(receiver->Track()),
                             receiver->Track().Id().Utf8());
  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackRemoveTransceiver(
        this,
        PeerConnectionTracker::TransceiverUpdatedReason::kSetRemoteDescription,
        nullptr /* sender */, receiver.get() /* receiver */);
  }
  for (const auto& stream_id : receiver->state().stream_ids()) {
    // This was the last occurence of the stream?
    if (!IsRemoteStream(rtp_receivers_, stream_id))
      PerSessionWebRTCAPIMetrics::GetInstance()->IncrementStreamCounter();
  }
  if (!is_closed_)
    client_->DidRemoveReceiverPlanB(std::move(receiver));
}

void RTCPeerConnectionHandler::OnModifyTransceivers(
    std::vector<RtpTransceiverState> transceiver_states,
    bool is_remote_description) {
  DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
  std::vector<std::unique_ptr<blink::WebRTCRtpTransceiver>> web_transceivers(
      transceiver_states.size());
  for (size_t i = 0; i < transceiver_states.size(); ++i) {
    // TODO(hbos): Figure out if a transceiver was added, modified, removed or
    // stayed the same and use the |peer_connection_tracker_| to track
    // transceiver updates for debugging aid in chrome://webrtc-internals.
    // https://crbug.com/864912
    web_transceivers[i] =
        CreateOrUpdateTransceiver(std::move(transceiver_states[i]));
  }
  if (!is_closed_) {
    client_->DidModifyTransceivers(std::move(web_transceivers),
                                   is_remote_description);
  }
}

void RTCPeerConnectionHandler::OnDataChannel(
    std::unique_ptr<RtcDataChannelHandler> handler) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnDataChannelImpl");

  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackCreateDataChannel(
        this, handler->channel().get(), PeerConnectionTracker::SOURCE_REMOTE);
  }

  if (!is_closed_)
    client_->DidAddRemoteDataChannel(handler.release());
}

void RTCPeerConnectionHandler::OnIceCandidate(
    const std::string& sdp, const std::string& sdp_mid, int sdp_mline_index,
    int component, int address_family) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  TRACE_EVENT0("webrtc", "RTCPeerConnectionHandler::OnIceCandidateImpl");
  scoped_refptr<blink::WebRTCICECandidate> web_candidate =
      blink::WebRTCICECandidate::Create(blink::WebString::FromUTF8(sdp),
                                        blink::WebString::FromUTF8(sdp_mid),
                                        sdp_mline_index);
  if (peer_connection_tracker_) {
    peer_connection_tracker_->TrackAddIceCandidate(
        this, web_candidate, PeerConnectionTracker::SOURCE_LOCAL, true);
  }

  // Only the first m line's first component is tracked to avoid
  // miscounting when doing BUNDLE or rtcp mux.
  if (sdp_mline_index == 0 && component == 1) {
    if (address_family == AF_INET) {
      ++num_local_candidates_ipv4_;
    } else if (address_family == AF_INET6) {
      ++num_local_candidates_ipv6_;
    } else {
      NOTREACHED();
    }
  }
  if (!is_closed_)
    client_->DidGenerateICECandidate(std::move(web_candidate));
}

webrtc::SessionDescriptionInterface*
RTCPeerConnectionHandler::CreateNativeSessionDescription(
    const std::string& sdp, const std::string& type,
    webrtc::SdpParseError* error) {
  webrtc::SessionDescriptionInterface* native_desc =
      dependency_factory_->CreateSessionDescription(type, sdp, error);

  LOG_IF(ERROR, !native_desc) << "Failed to create native session description."
                              << " Type: " << type << " SDP: " << sdp;

  return native_desc;
}

RTCPeerConnectionHandler::FirstSessionDescription::FirstSessionDescription(
    const webrtc::SessionDescriptionInterface* sdesc) {
  DCHECK(sdesc);

  for (const auto& content : sdesc->description()->contents()) {
    if (content.type == cricket::NS_JINGLE_RTP) {
      const auto* mdesc =
          static_cast<cricket::MediaContentDescription*>(content.description);
      audio = audio || (mdesc->type() == cricket::MEDIA_TYPE_AUDIO);
      video = video || (mdesc->type() == cricket::MEDIA_TYPE_VIDEO);
      rtcp_mux = rtcp_mux || mdesc->rtcp_mux();
    }
  }
}

void RTCPeerConnectionHandler::ReportFirstSessionDescriptions(
    const FirstSessionDescription& local,
    const FirstSessionDescription& remote) {
  RtcpMux rtcp_mux = RTCP_MUX_ENABLED;
  if ((!local.audio && !local.video) || (!remote.audio && !remote.video)) {
    rtcp_mux = RTCP_MUX_NO_MEDIA;
  } else if (!local.rtcp_mux || !remote.rtcp_mux) {
    rtcp_mux = RTCP_MUX_DISABLED;
  }

  UMA_HISTOGRAM_ENUMERATION(
      "WebRTC.PeerConnection.RtcpMux", rtcp_mux, RTCP_MUX_MAX);

  // TODO(pthatcher): Reports stats about whether we have audio and
  // video or not.
}

std::vector<std::unique_ptr<RTCRtpSender>>::iterator
RTCPeerConnectionHandler::FindSender(uintptr_t id) {
  for (auto it = rtp_senders_.begin(); it != rtp_senders_.end(); ++it) {
    if ((*it)->Id() == id)
      return it;
  }
  return rtp_senders_.end();
}

std::vector<std::unique_ptr<RTCRtpTransceiver>>::iterator
RTCPeerConnectionHandler::FindTransceiver(uintptr_t id) {
  for (auto it = rtp_transceivers_.begin(); it != rtp_transceivers_.end();
       ++it) {
    if ((*it)->Id() == id)
      return it;
  }
  return rtp_transceivers_.end();
}

std::unique_ptr<RTCRtpTransceiver>
RTCPeerConnectionHandler::CreateOrUpdateTransceiver(
    RtpTransceiverState transceiver_state) {
  DCHECK_EQ(sdp_semantics_, blink::WebRTCSdpSemantics::kUnifiedPlan);
  DCHECK(transceiver_state.is_initialized());
  DCHECK(transceiver_state.sender_state());
  DCHECK(transceiver_state.receiver_state());
  auto webrtc_transceiver = transceiver_state.webrtc_transceiver();
  auto webrtc_sender = transceiver_state.sender_state()->webrtc_sender();
  auto webrtc_receiver = transceiver_state.receiver_state()->webrtc_receiver();

  std::unique_ptr<RTCRtpTransceiver> transceiver;
  auto it = FindTransceiver(RTCRtpTransceiver::GetId(webrtc_transceiver.get()));
  if (it == rtp_transceivers_.end()) {
    // Create a new transceiver, including a sender and a receiver.
    transceiver = std::make_unique<RTCRtpTransceiver>(
        native_peer_connection_, track_adapter_map_,
        std::move(transceiver_state));
    rtp_transceivers_.push_back(transceiver->ShallowCopy());
    DCHECK(FindSender(RTCRtpSender::getId(webrtc_sender.get())) ==
           rtp_senders_.end());
    rtp_senders_.push_back(transceiver->content_sender()->ShallowCopy());
    uintptr_t receiver_id = RTCRtpReceiver::getId(webrtc_receiver.get());
    DCHECK(rtp_receivers_.find(receiver_id) == rtp_receivers_.end());
    rtp_receivers_.insert(std::make_pair(
        receiver_id, transceiver->content_receiver()->ShallowCopy()));
  } else {
    // Update the transceiver. This also updates the sender and receiver.
    transceiver = (*it)->ShallowCopy();
    transceiver->set_state(std::move(transceiver_state));
    DCHECK(FindSender(RTCRtpSender::getId(webrtc_sender.get())) !=
           rtp_senders_.end());
    DCHECK(rtp_receivers_.find(RTCRtpReceiver::getId(webrtc_receiver.get())) !=
           rtp_receivers_.end());
  }
  return transceiver;
}

scoped_refptr<base::SingleThreadTaskRunner>
RTCPeerConnectionHandler::signaling_thread() const {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  return dependency_factory_->GetWebRtcSignalingThread();
}

void RTCPeerConnectionHandler::RunSynchronousClosureOnSignalingThread(
    const base::Closure& closure,
    const char* trace_event_name) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  scoped_refptr<base::SingleThreadTaskRunner> thread(signaling_thread());
  if (!thread.get() || thread->BelongsToCurrentThread()) {
    TRACE_EVENT0("webrtc", trace_event_name);
    closure.Run();
  } else {
    base::WaitableEvent event(base::WaitableEvent::ResetPolicy::AUTOMATIC,
                              base::WaitableEvent::InitialState::NOT_SIGNALED);
    thread->PostTask(FROM_HERE,
                     base::BindOnce(&RunSynchronousClosure, closure,
                                    base::Unretained(trace_event_name),
                                    base::Unretained(&event)));
    event.Wait();
  }
}

void RTCPeerConnectionHandler::ReportICEState(
    webrtc::PeerConnectionInterface::IceConnectionState new_state) {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  if (ice_state_seen_[new_state])
    return;
  ice_state_seen_[new_state] = true;
  UMA_HISTOGRAM_ENUMERATION("WebRTC.PeerConnection.ConnectionState", new_state,
                            webrtc::PeerConnectionInterface::kIceConnectionMax);
}

void RTCPeerConnectionHandler::ResetUMAStats() {
  DCHECK(task_runner_->RunsTasksInCurrentSequence());
  num_local_candidates_ipv6_ = 0;
  num_local_candidates_ipv4_ = 0;
  ice_connection_checking_start_ = base::TimeTicks();
  memset(ice_state_seen_, 0, sizeof(ice_state_seen_));
}
}  // namespace content
