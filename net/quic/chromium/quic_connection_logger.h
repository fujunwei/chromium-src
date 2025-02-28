// Copyright (c) 2013 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef NET_QUIC_CHROMIUM_QUIC_CONNECTION_LOGGER_H_
#define NET_QUIC_CHROMIUM_QUIC_CONNECTION_LOGGER_H_

#include <stddef.h>

#include <bitset>
#include <string>

#include "base/macros.h"
#include "base/timer/timer.h"
#include "net/base/net_export.h"
#include "net/base/network_change_notifier.h"
#include "net/cert/cert_verify_result.h"
#include "net/log/net_log_with_source.h"
#include "net/socket/socket_performance_watcher.h"
#include "net/third_party/quic/core/crypto/crypto_handshake_message.h"
#include "net/third_party/quic/core/http/quic_spdy_session.h"
#include "net/third_party/quic/core/quic_connection.h"
#include "net/third_party/quic/core/quic_packets.h"

namespace base {
class HistogramBase;
}

namespace net {

// This class is a debug visitor of a quic::QuicConnection which logs
// events to |net_log|.
class NET_EXPORT_PRIVATE QuicConnectionLogger
    : public quic::QuicConnectionDebugVisitor,
      public quic::QuicPacketCreator::DebugDelegate {
 public:
  QuicConnectionLogger(
      quic::QuicSpdySession* session,
      const char* const connection_description,
      std::unique_ptr<SocketPerformanceWatcher> socket_performance_watcher,
      const NetLogWithSource& net_log);

  ~QuicConnectionLogger() override;

  // quic::QuicPacketCreator::DebugDelegateInterface
  void OnFrameAddedToPacket(const quic::QuicFrame& frame) override;

  // QuicConnectionDebugVisitorInterface
  void OnPacketSent(const quic::SerializedPacket& serialized_packet,
                    quic::QuicPacketNumber original_packet_number,
                    quic::TransmissionType transmission_type,
                    quic::QuicTime sent_time) override;
  void OnPingSent() override;
  void OnPacketReceived(const quic::QuicSocketAddress& self_address,
                        const quic::QuicSocketAddress& peer_address,
                        const quic::QuicEncryptedPacket& packet) override;
  void OnUnauthenticatedHeader(const quic::QuicPacketHeader& header) override;
  void OnIncorrectConnectionId(quic::QuicConnectionId connection_id) override;
  void OnUndecryptablePacket() override;
  void OnDuplicatePacket(quic::QuicPacketNumber packet_number) override;
  void OnProtocolVersionMismatch(quic::ParsedQuicVersion version) override;
  void OnPacketHeader(const quic::QuicPacketHeader& header) override;
  void OnStreamFrame(const quic::QuicStreamFrame& frame) override;
  void OnAckFrame(const quic::QuicAckFrame& frame) override;
  void OnStopWaitingFrame(const quic::QuicStopWaitingFrame& frame) override;
  void OnRstStreamFrame(const quic::QuicRstStreamFrame& frame) override;
  void OnConnectionCloseFrame(
      const quic::QuicConnectionCloseFrame& frame) override;
  void OnWindowUpdateFrame(const quic::QuicWindowUpdateFrame& frame,
                           const quic::QuicTime& receive_time) override;
  void OnBlockedFrame(const quic::QuicBlockedFrame& frame) override;
  void OnGoAwayFrame(const quic::QuicGoAwayFrame& frame) override;
  void OnPingFrame(const quic::QuicPingFrame& frame) override;
  void OnPublicResetPacket(const quic::QuicPublicResetPacket& packet) override;
  void OnVersionNegotiationPacket(
      const quic::QuicVersionNegotiationPacket& packet) override;
  void OnConnectionClosed(quic::QuicErrorCode error,
                          const std::string& error_details,
                          quic::ConnectionCloseSource source) override;
  void OnSuccessfulVersionNegotiation(
      const quic::ParsedQuicVersion& version) override;
  void OnRttChanged(quic::QuicTime::Delta rtt) const override;

  void OnCryptoHandshakeMessageReceived(
      const quic::CryptoHandshakeMessage& message);
  void OnCryptoHandshakeMessageSent(
      const quic::CryptoHandshakeMessage& message);
  void UpdateReceivedFrameCounts(quic::QuicStreamId stream_id,
                                 int num_frames_received,
                                 int num_duplicate_frames_received);
  void OnCertificateVerified(const CertVerifyResult& result);

  // Returns connection's overall packet loss rate in fraction.
  float ReceivedPacketLossRate() const;

 private:
  // Do a factory get for a histogram to record a 6-packet loss-sequence as a
  // sample. The histogram will record the 64 distinct possible combinations.
  // |which_6| is used to adjust the name of the histogram to distinguish the
  // first 6 packets in a connection, vs. some later 6 packets.
  base::HistogramBase* Get6PacketHistogram(const char* which_6) const;
  // For connections longer than 21 received packets, this call will calculate
  // the overall packet loss rate, and record it into a histogram.
  void RecordAggregatePacketLossRate() const;

  void UpdateIsCapturing();

  NetLogWithSource net_log_;
  quic::QuicSpdySession* session_;  // Unowned.
  // The last packet number received.
  quic::QuicPacketNumber last_received_packet_number_;
  // The size of the most recently received packet.
  size_t last_received_packet_size_;
  // True if a PING frame has been sent and no packet has been received.
  bool no_packet_received_after_ping_;
  // The size of the previously received packet.
  size_t previous_received_packet_size_;
  // The largest packet number received.  In the case where a packet is
  // received late (out of order), this value will not be updated.
  quic::QuicPacketNumber largest_received_packet_number_;
  // Number of times that the current received packet number is
  // smaller than the last received packet number.
  size_t num_out_of_order_received_packets_;
  // Number of times that the current received packet number is
  // smaller than the last received packet number and where the
  // size of the current packet is larger than the size of the previous
  // packet.
  size_t num_out_of_order_large_received_packets_;
  // The number of times that OnPacketHeader was called.
  // If the network replicates packets, then this number may be slightly
  // different from the real number of distinct packets received.
  quic::QuicPacketCount num_packets_received_;
  // The quic::kCADR value provided by the server in ServerHello.
  IPEndPoint local_address_from_shlo_;
  // The first local address from which a packet was received.
  IPEndPoint local_address_from_self_;
  // Count of the number of frames received.
  int num_frames_received_;
  // Count of the number of duplicate frames received.
  int num_duplicate_frames_received_;
  // Count of the number of packets received with incorrect connection IDs.
  int num_incorrect_connection_ids_;
  // Count of the number of undecryptable packets received.
  int num_undecryptable_packets_;
  // Count of the number of duplicate packets received.
  int num_duplicate_packets_;
  // Count of the number of BLOCKED frames received.
  int num_blocked_frames_received_;
  // Count of the number of BLOCKED frames sent.
  int num_blocked_frames_sent_;
  // Vector of inital packets status' indexed by packet numbers, where
  // false means never received.  Zero is not a valid packet number, so
  // that offset is never used, and we'll track 150 packets.
  std::bitset<151> received_packets_;
  // Vector to indicate which of the initial 150 received packets turned out to
  // contain solo ACK frames.  An element is true iff an ACK frame was in the
  // corresponding packet, and there was very little else.
  std::bitset<151> received_acks_;
  // The available type of connection (WiFi, 3G, etc.) when connection was first
  // used.
  const char* const connection_description_;
  // Receives notifications regarding the performance of the underlying socket
  // for the QUIC connection. May be null.
  const std::unique_ptr<SocketPerformanceWatcher> socket_performance_watcher_;
  // Lower the overhead of checking whether logging is active, by
  // periodically polling and caching the result of net_log_.IsCapturing().
  bool net_log_is_capturing_;
  base::RepeatingTimer timer_;

  DISALLOW_COPY_AND_ASSIGN(QuicConnectionLogger);
};

}  // namespace net

#endif  // NET_QUIC_CHROMIUM_QUIC_CONNECTION_LOGGER_H_
