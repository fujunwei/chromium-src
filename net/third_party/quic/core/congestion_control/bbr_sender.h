// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

// BBR (Bottleneck Bandwidth and RTT) congestion control algorithm.

#ifndef NET_THIRD_PARTY_QUIC_CORE_CONGESTION_CONTROL_BBR_SENDER_H_
#define NET_THIRD_PARTY_QUIC_CORE_CONGESTION_CONTROL_BBR_SENDER_H_

#include <cstdint>
#include <ostream>

#include "base/macros.h"
#include "net/third_party/quic/core/congestion_control/bandwidth_sampler.h"
#include "net/third_party/quic/core/congestion_control/send_algorithm_interface.h"
#include "net/third_party/quic/core/congestion_control/windowed_filter.h"
#include "net/third_party/quic/core/crypto/quic_random.h"
#include "net/third_party/quic/core/quic_bandwidth.h"
#include "net/third_party/quic/core/quic_packets.h"
#include "net/third_party/quic/core/quic_time.h"
#include "net/third_party/quic/core/quic_unacked_packet_map.h"
#include "net/third_party/quic/platform/api/quic_export.h"
#include "net/third_party/quic/platform/api/quic_string.h"

namespace quic {

class RttStats;

typedef uint64_t QuicRoundTripCount;

// BbrSender implements BBR congestion control algorithm.  BBR aims to estimate
// the current available Bottleneck Bandwidth and RTT (hence the name), and
// regulates the pacing rate and the size of the congestion window based on
// those signals.
//
// BBR relies on pacing in order to function properly.  Do not use BBR when
// pacing is disabled.
//
// TODO(vasilvv): implement traffic policer (long-term sampling) mode.
class QUIC_EXPORT_PRIVATE BbrSender : public SendAlgorithmInterface {
 public:
  enum Mode {
    // Startup phase of the connection.
    STARTUP,
    // After achieving the highest possible bandwidth during the startup, lower
    // the pacing rate in order to drain the queue.
    DRAIN,
    // Cruising mode.
    PROBE_BW,
    // Temporarily slow down sending in order to empty the buffer and measure
    // the real minimum RTT.
    PROBE_RTT,
  };

  // Indicates how the congestion control limits the amount of bytes in flight.
  enum RecoveryState {
    // Do not limit.
    NOT_IN_RECOVERY,
    // Allow an extra outstanding byte for each byte acknowledged.
    CONSERVATION,
    // Allow 1.5 extra outstanding bytes for each byte acknowledged.
    MEDIUM_GROWTH,
    // Allow two extra outstanding bytes for each byte acknowledged (slow
    // start).
    GROWTH
  };

  // Debug state can be exported in order to troubleshoot potential congestion
  // control issues.
  struct DebugState {
    explicit DebugState(const BbrSender& sender);
    DebugState(const DebugState& state);

    Mode mode;
    QuicBandwidth max_bandwidth;
    QuicRoundTripCount round_trip_count;
    int gain_cycle_index;
    QuicByteCount congestion_window;

    bool is_at_full_bandwidth;
    QuicBandwidth bandwidth_at_last_round;
    QuicRoundTripCount rounds_without_bandwidth_gain;

    QuicTime::Delta min_rtt;
    QuicTime min_rtt_timestamp;

    RecoveryState recovery_state;
    QuicByteCount recovery_window;

    bool last_sample_is_app_limited;
    QuicPacketNumber end_of_app_limited_phase;
  };

  BbrSender(const RttStats* rtt_stats,
            const QuicUnackedPacketMap* unacked_packets,
            QuicPacketCount initial_tcp_congestion_window,
            QuicPacketCount max_tcp_congestion_window,
            QuicRandom* random);
  BbrSender(const BbrSender&) = delete;
  BbrSender& operator=(const BbrSender&) = delete;
  ~BbrSender() override;

  // Start implementation of SendAlgorithmInterface.
  bool InSlowStart() const override;
  bool InRecovery() const override;
  bool IsProbingForMoreBandwidth() const override;

  void SetFromConfig(const QuicConfig& config,
                     Perspective perspective) override;

  void AdjustNetworkParameters(QuicBandwidth bandwidth,
                               QuicTime::Delta rtt) override;
  void SetNumEmulatedConnections(int num_connections) override {}
  void SetInitialCongestionWindowInPackets(
      QuicPacketCount congestion_window) override;
  void OnCongestionEvent(bool rtt_updated,
                         QuicByteCount prior_in_flight,
                         QuicTime event_time,
                         const AckedPacketVector& acked_packets,
                         const LostPacketVector& lost_packets) override;
  void OnPacketSent(QuicTime sent_time,
                    QuicByteCount bytes_in_flight,
                    QuicPacketNumber packet_number,
                    QuicByteCount bytes,
                    HasRetransmittableData is_retransmittable) override;
  void OnRetransmissionTimeout(bool packets_retransmitted) override {}
  void OnConnectionMigration() override {}
  bool CanSend(QuicByteCount bytes_in_flight) override;
  QuicBandwidth PacingRate(QuicByteCount bytes_in_flight) const override;
  QuicBandwidth BandwidthEstimate() const override;
  QuicByteCount GetCongestionWindow() const override;
  QuicByteCount GetSlowStartThreshold() const override;
  CongestionControlType GetCongestionControlType() const override;
  QuicString GetDebugState() const override;
  void OnApplicationLimited(QuicByteCount bytes_in_flight) override;
  // End implementation of SendAlgorithmInterface.

  // Gets the number of RTTs BBR remains in STARTUP phase.
  QuicRoundTripCount num_startup_rtts() const { return num_startup_rtts_; }
  bool has_non_app_limited_sample() const {
    return has_non_app_limited_sample_;
  }

  // Sets the pacing gain used in STARTUP.  Must be greater than 1.
  void set_high_gain(float high_gain) {
    DCHECK_LT(1.0f, high_gain);
    high_gain_ = high_gain;
    if (mode_ == STARTUP) {
      pacing_gain_ = high_gain;
    }
  }

  // Sets the CWND gain used in STARTUP.  Must be greater than 1.
  void set_high_cwnd_gain(float high_cwnd_gain) {
    DCHECK_LT(1.0f, high_cwnd_gain);
    high_cwnd_gain_ = high_cwnd_gain;
    if (mode_ == STARTUP) {
      congestion_window_gain_ = high_cwnd_gain;
    }
  }

  // Sets the gain used in DRAIN.  Must be less than 1.
  void set_drain_gain(float drain_gain) {
    DCHECK_GT(1.0f, drain_gain);
    drain_gain_ = drain_gain;
  }

  DebugState ExportDebugState() const;

 private:
  typedef WindowedFilter<QuicBandwidth,
                         MaxFilter<QuicBandwidth>,
                         QuicRoundTripCount,
                         QuicRoundTripCount>
      MaxBandwidthFilter;

  typedef WindowedFilter<QuicByteCount,
                         MaxFilter<QuicByteCount>,
                         QuicRoundTripCount,
                         QuicRoundTripCount>
      MaxAckHeightFilter;

  // Returns the current estimate of the RTT of the connection.  Outside of the
  // edge cases, this is minimum RTT.
  QuicTime::Delta GetMinRtt() const;
  // Returns whether the connection has achieved full bandwidth required to exit
  // the slow start.
  bool IsAtFullBandwidth() const;
  // Computes the target congestion window using the specified gain.
  QuicByteCount GetTargetCongestionWindow(float gain) const;
  // The target congestion window during PROBE_RTT.
  QuicByteCount ProbeRttCongestionWindow() const;
  // Returns true if the current min_rtt should be kept and we should not enter
  // PROBE_RTT immediately.
  bool ShouldExtendMinRttExpiry() const;

  // Enters the STARTUP mode.
  void EnterStartupMode();
  // Enters the PROBE_BW mode.
  void EnterProbeBandwidthMode(QuicTime now);

  // Discards the lost packets from BandwidthSampler state.
  void DiscardLostPackets(const LostPacketVector& lost_packets);
  // Updates the round-trip counter if a round-trip has passed.  Returns true if
  // the counter has been advanced.
  bool UpdateRoundTripCounter(QuicPacketNumber last_acked_packet);
  // Updates the current bandwidth and min_rtt estimate based on the samples for
  // the received acknowledgements.  Returns true if min_rtt has expired.
  bool UpdateBandwidthAndMinRtt(QuicTime now,
                                const AckedPacketVector& acked_packets);
  // Updates the current gain used in PROBE_BW mode.
  void UpdateGainCyclePhase(QuicTime now,
                            QuicByteCount prior_in_flight,
                            bool has_losses);
  // Tracks for how many round-trips the bandwidth has not increased
  // significantly.
  void CheckIfFullBandwidthReached();
  // Transitions from STARTUP to DRAIN and from DRAIN to PROBE_BW if
  // appropriate.
  void MaybeExitStartupOrDrain(QuicTime now);
  // Decides whether to enter or exit PROBE_RTT.
  void MaybeEnterOrExitProbeRtt(QuicTime now,
                                bool is_round_start,
                                bool min_rtt_expired);
  // Determines whether BBR needs to enter, exit or advance state of the
  // recovery.
  void UpdateRecoveryState(QuicPacketNumber last_acked_packet,
                           bool has_losses,
                           bool is_round_start);

  // Updates the ack aggregation max filter in bytes.
  // Returns the most recent addition to the filter, or |newly_acked_bytes| if
  // nothing was fed in to the filter.
  QuicByteCount UpdateAckAggregationBytes(QuicTime ack_time,
                                          QuicByteCount newly_acked_bytes);

  // Determines the appropriate pacing rate for the connection.
  void CalculatePacingRate();
  // Determines the appropriate congestion window for the connection.
  void CalculateCongestionWindow(QuicByteCount bytes_acked,
                                 QuicByteCount excess_acked);
  // Determines the approriate window that constrains the in-flight during
  // recovery.
  void CalculateRecoveryWindow(QuicByteCount bytes_acked,
                               QuicByteCount bytes_lost);

  const RttStats* rtt_stats_;
  const QuicUnackedPacketMap* unacked_packets_;
  QuicRandom* random_;

  Mode mode_;

  // Bandwidth sampler provides BBR with the bandwidth measurements at
  // individual points.
  std::unique_ptr<BandwidthSamplerInterface> sampler_;

  // The number of the round trips that have occurred during the connection.
  QuicRoundTripCount round_trip_count_;

  // The packet number of the most recently sent packet.
  QuicPacketNumber last_sent_packet_;
  // Acknowledgement of any packet after |current_round_trip_end_| will cause
  // the round trip counter to advance.
  QuicPacketCount current_round_trip_end_;

  // The filter that tracks the maximum bandwidth over the multiple recent
  // round-trips.
  MaxBandwidthFilter max_bandwidth_;

  // Tracks the maximum number of bytes acked faster than the sending rate.
  MaxAckHeightFilter max_ack_height_;

  // The time this aggregation started and the number of bytes acked during it.
  QuicTime aggregation_epoch_start_time_;
  QuicByteCount aggregation_epoch_bytes_;

  // Minimum RTT estimate.  Automatically expires within 10 seconds (and
  // triggers PROBE_RTT mode) if no new value is sampled during that period.
  QuicTime::Delta min_rtt_;
  // The time at which the current value of |min_rtt_| was assigned.
  QuicTime min_rtt_timestamp_;

  // The maximum allowed number of bytes in flight.
  QuicByteCount congestion_window_;

  // The initial value of the |congestion_window_|.
  QuicByteCount initial_congestion_window_;

  // The largest value the |congestion_window_| can achieve.
  QuicByteCount max_congestion_window_;

  // The smallest value the |congestion_window_| can achieve.
  QuicByteCount min_congestion_window_;

  // The pacing gain applied during the STARTUP phase.
  float high_gain_;

  // The CWND gain applied during the STARTUP phase.
  float high_cwnd_gain_;

  // The pacing gain applied during the DRAIN phase.
  float drain_gain_;

  // The current pacing rate of the connection.
  QuicBandwidth pacing_rate_;

  // The gain currently applied to the pacing rate.
  float pacing_gain_;
  // The gain currently applied to the congestion window.
  float congestion_window_gain_;

  // The gain used for the congestion window during PROBE_BW.  Latched from
  // quic_bbr_cwnd_gain flag.
  const float congestion_window_gain_constant_;
  // The number of RTTs to stay in STARTUP mode.  Defaults to 3.
  QuicRoundTripCount num_startup_rtts_;
  // If true, exit startup if 1RTT has passed with no bandwidth increase and
  // the connection is in recovery.
  bool exit_startup_on_loss_;

  // Number of round-trips in PROBE_BW mode, used for determining the current
  // pacing gain cycle.
  int cycle_current_offset_;
  // The time at which the last pacing gain cycle was started.
  QuicTime last_cycle_start_;

  // Indicates whether the connection has reached the full bandwidth mode.
  bool is_at_full_bandwidth_;
  // Number of rounds during which there was no significant bandwidth increase.
  QuicRoundTripCount rounds_without_bandwidth_gain_;
  // The bandwidth compared to which the increase is measured.
  QuicBandwidth bandwidth_at_last_round_;

  // Set to true upon exiting quiescence.
  bool exiting_quiescence_;

  // Time at which PROBE_RTT has to be exited.  Setting it to zero indicates
  // that the time is yet unknown as the number of packets in flight has not
  // reached the required value.
  QuicTime exit_probe_rtt_at_;
  // Indicates whether a round-trip has passed since PROBE_RTT became active.
  bool probe_rtt_round_passed_;

  // Indicates whether the most recent bandwidth sample was marked as
  // app-limited.
  bool last_sample_is_app_limited_;
  // Indicates whether any non app-limited samples have been recorded.
  bool has_non_app_limited_sample_;

  // Current state of recovery.
  RecoveryState recovery_state_;
  // Receiving acknowledgement of a packet after |end_recovery_at_| will cause
  // BBR to exit the recovery mode.  A value above zero indicates at least one
  // loss has been detected, so it must not be set back to zero.
  QuicPacketNumber end_recovery_at_;
  // A window used to limit the number of bytes in flight during loss recovery.
  QuicByteCount recovery_window_;
  // If true, consider all samples in recovery app-limited.
  bool is_app_limited_recovery_;

  // When true, pace at 1.5x and disable packet conservation in STARTUP.
  bool slower_startup_;
  // When true, disables packet conservation in STARTUP.
  bool rate_based_startup_;
  // Used as the initial packet conservation mode when first entering recovery.
  RecoveryState initial_conservation_in_startup_;
  // When true, add the most recent ack aggregation measurement during STARTUP.
  bool enable_ack_aggregation_during_startup_;
  // When true, expire the windowed ack aggregation values in STARTUP when
  // bandwidth increases more than 25%.
  bool expire_ack_aggregation_in_startup_;

  // If true, will not exit low gain mode until bytes_in_flight drops below BDP
  // or it's time for high gain mode.
  bool drain_to_target_;

  // If true, use a CWND of 0.75*BDP during probe_rtt instead of 4 packets.
  bool probe_rtt_based_on_bdp_;
  // If true, skip probe_rtt and update the timestamp of the existing min_rtt to
  // now if min_rtt over the last cycle is within 12.5% of the current min_rtt.
  // Even if the min_rtt is 12.5% too low, the 25% gain cycling and 2x CWND gain
  // should overcome an overly small min_rtt.
  bool probe_rtt_skipped_if_similar_rtt_;
  // If true, disable PROBE_RTT entirely as long as the connection was recently
  // app limited.
  bool probe_rtt_disabled_if_app_limited_;
  bool app_limited_since_last_probe_rtt_;
  QuicTime::Delta min_rtt_since_last_probe_rtt_;
};

QUIC_EXPORT_PRIVATE std::ostream& operator<<(std::ostream& os,
                                             const BbrSender::Mode& mode);
QUIC_EXPORT_PRIVATE std::ostream& operator<<(
    std::ostream& os,
    const BbrSender::DebugState& state);

}  // namespace quic

#endif  // NET_THIRD_PARTY_QUIC_CORE_CONGESTION_CONTROL_BBR_SENDER_H_
