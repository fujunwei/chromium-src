// Copyright (c) 2012 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// A client specific quic::QuicSession subclass.  This class owns the underlying
// quic::QuicConnection and QuicConnectionHelper objects.  The connection stores
// a non-owning pointer to the helper so this session needs to ensure that
// the helper outlives the connection.

#ifndef NET_QUIC_CHROMIUM_QUIC_CHROMIUM_CLIENT_SESSION_H_
#define NET_QUIC_CHROMIUM_QUIC_CHROMIUM_CLIENT_SESSION_H_

#include <stddef.h>

#include <list>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "base/containers/mru_cache.h"
#include "base/macros.h"
#include "base/time/time.h"
#include "net/base/completion_once_callback.h"
#include "net/base/load_timing_info.h"
#include "net/base/net_error_details.h"
#include "net/base/net_export.h"
#include "net/base/proxy_server.h"
#include "net/cert/ct_verify_result.h"
#include "net/log/net_log_with_source.h"
#include "net/quic/chromium/quic_chromium_client_stream.h"
#include "net/quic/chromium/quic_chromium_packet_reader.h"
#include "net/quic/chromium/quic_chromium_packet_writer.h"
#include "net/quic/chromium/quic_connection_logger.h"
#include "net/quic/chromium/quic_connectivity_probing_manager.h"
#include "net/quic/chromium/quic_session_key.h"
#include "net/socket/socket_performance_watcher.h"
#include "net/spdy/http2_priority_dependencies.h"
#include "net/spdy/multiplexed_session.h"
#include "net/spdy/server_push_delegate.h"
#include "net/third_party/quic/core/http/quic_client_push_promise_index.h"
#include "net/third_party/quic/core/http/quic_spdy_client_session_base.h"
#include "net/third_party/quic/core/quic_crypto_client_stream.h"
#include "net/third_party/quic/core/quic_packets.h"
#include "net/third_party/quic/core/quic_server_id.h"
#include "net/third_party/quic/core/quic_time.h"
#include "net/traffic_annotation/network_traffic_annotation.h"

namespace net {

class CertVerifyResult;
class DatagramClientSocket;
class NetLog;
class QuicCryptoClientStreamFactory;
class QuicServerInfo;
class QuicStreamFactory;
class SSLInfo;
class TransportSecurityState;

using TokenBindingSignatureMap =
    base::MRUCache<std::pair<TokenBindingType, std::string>,
                   std::vector<uint8_t>>;

namespace test {
class QuicChromiumClientSessionPeer;
}  // namespace test

// Result of a session migration attempt.
enum class MigrationResult {
  SUCCESS,         // Migration succeeded.
  NO_NEW_NETWORK,  // Migration failed since no new network was found.
  FAILURE          // Migration failed for other reasons.
};

// Mode of connection migration.
enum class ConnectionMigrationMode {
  NO_MIGRATION,
  NO_MIGRATION_ON_PATH_DEGRADING_V1,
  FULL_MIGRATION_V1,
  NO_MIGRATION_ON_PATH_DEGRADING_V2,
  FULL_MIGRATION_V2
};

// Cause of connection migration.
enum ConnectionMigrationCause {
  UNKNOWN,
  ON_NETWORK_CONNECTED,                // No probing.
  ON_NETWORK_DISCONNECTED,             // No probing.
  ON_WRITE_ERROR,                      // No probing.
  ON_NETWORK_MADE_DEFAULT,             // With probing.
  ON_MIGRATE_BACK_TO_DEFAULT_NETWORK,  // With probing.
  ON_PATH_DEGRADING,                   // With probing.
  MIGRATION_CAUSE_MAX
};

// Result of connection migration.
enum QuicConnectionMigrationStatus {
  MIGRATION_STATUS_NO_MIGRATABLE_STREAMS,
  MIGRATION_STATUS_ALREADY_MIGRATED,
  MIGRATION_STATUS_INTERNAL_ERROR,
  MIGRATION_STATUS_TOO_MANY_CHANGES,
  MIGRATION_STATUS_SUCCESS,
  MIGRATION_STATUS_NON_MIGRATABLE_STREAM,
  MIGRATION_STATUS_NOT_ENABLED,
  MIGRATION_STATUS_NO_ALTERNATE_NETWORK,
  MIGRATION_STATUS_ON_PATH_DEGRADING_DISABLED,
  MIGRATION_STATUS_DISABLED_BY_CONFIG,
  MIGRATION_STATUS_PATH_DEGRADING_NOT_ENABLED,
  MIGRATION_STATUS_TIMEOUT,
  MIGRATION_STATUS_MAX
};

// Result of a connectivity probing attempt.
enum class ProbingResult {
  PENDING,                          // Probing started, pending result.
  DISABLED_WITH_IDLE_SESSION,       // Probing disabled with idle session.
  DISABLED_BY_CONFIG,               // Probing disabled by config.
  DISABLED_BY_NON_MIGRABLE_STREAM,  // Probing disabled by special stream.
  INTERNAL_ERROR,                   // Probing failed for internal reason.
  FAILURE,                          // Probing failed for other reason.
};

class NET_EXPORT_PRIVATE QuicChromiumClientSession
    : public quic::QuicSpdyClientSessionBase,
      public MultiplexedSession,
      public QuicConnectivityProbingManager::Delegate,
      public QuicChromiumPacketReader::Visitor,
      public QuicChromiumPacketWriter::Delegate {
 public:
  class StreamRequest;

  // Wrapper for interacting with the session in a restricted fashion which
  // hides the details of the underlying session's lifetime. All methods of
  // the Handle are safe to use even after the underlying session is destroyed.
  class NET_EXPORT_PRIVATE Handle
      : public MultiplexedSessionHandle,
        public quic::QuicClientPushPromiseIndex::Delegate {
   public:
    // Constructs a handle to |session| which was created via the alternative
    // server |destination|.
    Handle(const base::WeakPtr<QuicChromiumClientSession>& session,
           const HostPortPair& destination);
    Handle(const Handle& other) = delete;
    ~Handle() override;

    // Returns true if the session is still connected.
    bool IsConnected() const;

    // Returns true if the handshake has been confirmed.
    bool IsCryptoHandshakeConfirmed() const;

    // Starts a request to rendezvous with a promised a stream.  If OK is
    // returned, then |push_stream_| will be updated with the promised
    // stream.  If ERR_IO_PENDING is returned, then when the rendezvous is
    // eventually completed |callback| will be called.
    int RendezvousWithPromised(const spdy::SpdyHeaderBlock& headers,
                               CompletionOnceCallback callback);

    // Starts a request to create a stream.  If OK is returned, then
    // |stream_| will be updated with the newly created stream.  If
    // ERR_IO_PENDING is returned, then when the request is eventuallly
    // complete |callback| will be called.
    int RequestStream(bool requires_confirmation,
                      CompletionOnceCallback callback,
                      const NetworkTrafficAnnotationTag& traffic_annotation);

    // Releases |stream_| to the caller. Returns nullptr if the underlying
    // QuicChromiumClientSession is closed.
    std::unique_ptr<QuicChromiumClientStream::Handle> ReleaseStream();

    // Releases |push_stream_| to the caller.
    std::unique_ptr<QuicChromiumClientStream::Handle> ReleasePromisedStream();

    // Sends Rst for the stream, and makes sure that future calls to
    // IsClosedStream(id) return true, which ensures that any subsequent
    // frames related to this stream will be ignored (modulo flow
    // control accounting).
    void ResetPromised(quic::QuicStreamId id,
                       quic::QuicRstStreamErrorCode error_code);

    // Returns a new packet bundler while will cause writes to be batched up
    // until a packet is full, or the last bundler is destroyed.
    std::unique_ptr<quic::QuicConnection::ScopedPacketFlusher>
    CreatePacketBundler(quic::QuicConnection::AckBundling bundling_mode);

    // Populates network error details for this session.
    void PopulateNetErrorDetails(NetErrorDetails* details) const;

    // Returns the connection timing for the handshake of this session.
    const LoadTimingInfo::ConnectTiming& GetConnectTiming();

    // Signs the exported keying material used for Token Binding using key
    // |*key| and puts the signature in |*out|. Returns a net error code.
    Error GetTokenBindingSignature(crypto::ECPrivateKey* key,
                                   TokenBindingType tb_type,
                                   std::vector<uint8_t>* out);

    // Returns true if |other| is a handle to the same session as this handle.
    bool SharesSameSession(const Handle& other) const;

    // Returns the QUIC version used by the session.
    quic::QuicTransportVersion GetQuicVersion() const;

    // Copies the remote udp address into |address| and returns a net error
    // code.
    int GetPeerAddress(IPEndPoint* address) const;

    // Copies the local udp address into |address| and returns a net error
    // code.
    int GetSelfAddress(IPEndPoint* address) const;

    // Returns the push promise index associated with the session.
    quic::QuicClientPushPromiseIndex* GetPushPromiseIndex();

    // Returns the session's server ID.
    quic::QuicServerId server_id() const { return server_id_; }

    // Returns the alternative server used for this session.
    HostPortPair destination() const { return destination_; }

    // Returns the session's net log.
    const NetLogWithSource& net_log() const { return net_log_; }

    // Returns the session's connection migration mode.
    ConnectionMigrationMode connection_migration_mode() const {
      return session_->connection_migration_mode();
    }

    // quic::QuicClientPushPromiseIndex::Delegate implementation
    bool CheckVary(const spdy::SpdyHeaderBlock& client_request,
                   const spdy::SpdyHeaderBlock& promise_request,
                   const spdy::SpdyHeaderBlock& promise_response) override;
    void OnRendezvousResult(quic::QuicSpdyStream* stream) override;

    // Returns true if the session's connection has sent or received any bytes.
    bool WasEverUsed() const;

   private:
    friend class QuicChromiumClientSession;
    friend class QuicChromiumClientSession::StreamRequest;

    // Waits for the handshake to be confirmed and invokes |callback| when
    // that happens. If the handshake has already been confirmed, returns OK.
    // If the connection has already been closed, returns a net error. If the
    // connection closes before the handshake is confirmed, |callback| will
    // be invoked with an error.
    int WaitForHandshakeConfirmation(CompletionOnceCallback callback);

    // Called when the handshake is confirmed.
    void OnCryptoHandshakeConfirmed();

    // Called when the session is closed with a net error.
    void OnSessionClosed(quic::QuicTransportVersion quic_version,
                         int net_error,
                         quic::QuicErrorCode quic_error,
                         bool port_migration_detected,
                         LoadTimingInfo::ConnectTiming connect_timing,
                         bool was_ever_used);

    // Called by |request| to create a stream.
    int TryCreateStream(StreamRequest* request);

    // Called by |request| to cancel stream request.
    void CancelRequest(StreamRequest* request);

    // Underlying session which may be destroyed before this handle.
    base::WeakPtr<QuicChromiumClientSession> session_;

    HostPortPair destination_;

    // Stream request created by |RequestStream()|.
    std::unique_ptr<StreamRequest> stream_request_;

    // Information saved from the session which can be used even after the
    // session is destroyed.
    NetLogWithSource net_log_;
    bool was_handshake_confirmed_;
    int net_error_;
    quic::QuicErrorCode quic_error_;
    bool port_migration_detected_;
    quic::QuicServerId server_id_;
    quic::QuicTransportVersion quic_version_;
    LoadTimingInfo::ConnectTiming connect_timing_;
    quic::QuicClientPushPromiseIndex* push_promise_index_;

    // |quic::QuicClientPromisedInfo| owns this. It will be set when |Try()|
    // is asynchronous, i.e. it returned quic::QUIC_PENDING, and remains valid
    // until |OnRendezvouResult()| fires or |push_handle_->Cancel()| is
    // invoked.
    quic::QuicClientPushPromiseIndex::TryHandle* push_handle_;
    CompletionOnceCallback push_callback_;
    std::unique_ptr<QuicChromiumClientStream::Handle> push_stream_;

    bool was_ever_used_;
  };

  // A helper class used to manage a request to create a stream.
  class NET_EXPORT_PRIVATE StreamRequest {
   public:
    // Cancels any pending stream creation request and resets |stream_| if
    // it has not yet been released.
    ~StreamRequest();

    // Starts a request to create a stream.  If OK is returned, then
    // |stream_| will be updated with the newly created stream.  If
    // ERR_IO_PENDING is returned, then when the request is eventuallly
    // complete |callback| will be called.
    int StartRequest(CompletionOnceCallback callback);

    // Releases |stream_| to the caller.
    std::unique_ptr<QuicChromiumClientStream::Handle> ReleaseStream();

    const NetworkTrafficAnnotationTag traffic_annotation() {
      return traffic_annotation_;
    }

   private:
    friend class QuicChromiumClientSession;

    enum State {
      STATE_NONE,
      STATE_WAIT_FOR_CONFIRMATION,
      STATE_WAIT_FOR_CONFIRMATION_COMPLETE,
      STATE_REQUEST_STREAM,
      STATE_REQUEST_STREAM_COMPLETE,
    };

    // |session| must outlive this request.
    StreamRequest(QuicChromiumClientSession::Handle* session,
                  bool requires_confirmation,
                  const NetworkTrafficAnnotationTag& traffic_annotation);

    void OnIOComplete(int rv);
    void DoCallback(int rv);

    int DoLoop(int rv);
    int DoWaitForConfirmation();
    int DoWaitForConfirmationComplete(int rv);
    int DoRequestStream();
    int DoRequestStreamComplete(int rv);

    // Called by |session_| for an asynchronous request when the stream
    // request has finished successfully.
    void OnRequestCompleteSuccess(
        std::unique_ptr<QuicChromiumClientStream::Handle> stream);

    // Called by |session_| for an asynchronous request when the stream
    // request has finished with an error. Also called with ERR_ABORTED
    // if |session_| is destroyed while the stream request is still pending.
    void OnRequestCompleteFailure(int rv);

    QuicChromiumClientSession::Handle* session_;
    const bool requires_confirmation_;
    CompletionOnceCallback callback_;
    std::unique_ptr<QuicChromiumClientStream::Handle> stream_;
    // For tracking how much time pending stream requests wait.
    base::TimeTicks pending_start_time_;
    State next_state_;

    const NetworkTrafficAnnotationTag traffic_annotation_;

    base::WeakPtrFactory<StreamRequest> weak_factory_;

    DISALLOW_COPY_AND_ASSIGN(StreamRequest);
  };

  // Constructs a new session which will own |connection|, but not
  // |stream_factory|, which must outlive this session.
  // TODO(rch): decouple the factory from the session via a Delegate interface.
  QuicChromiumClientSession(
      quic::QuicConnection* connection,
      std::unique_ptr<DatagramClientSocket> socket,
      QuicStreamFactory* stream_factory,
      QuicCryptoClientStreamFactory* crypto_client_stream_factory,
      quic::QuicClock* clock,
      TransportSecurityState* transport_security_state,
      std::unique_ptr<QuicServerInfo> server_info,
      const QuicSessionKey& session_key,
      bool require_confirmation,
      bool migrate_sesion_early_v2,
      bool migrate_session_on_network_change_v2,
      NetworkChangeNotifier::NetworkHandle default_network,
      base::TimeDelta max_time_on_non_default_network,
      int max_migrations_to_non_default_network_on_path_degrading,
      int yield_after_packets,
      quic::QuicTime::Delta yield_after_duration,
      bool headers_include_h2_stream_dependency,
      int cert_verify_flags,
      const quic::QuicConfig& config,
      quic::QuicCryptoClientConfig* crypto_config,
      const char* const connection_description,
      base::TimeTicks dns_resolution_start_time,
      base::TimeTicks dns_resolution_end_time,
      quic::QuicClientPushPromiseIndex* push_promise_index,
      ServerPushDelegate* push_delegate,
      base::SequencedTaskRunner* task_runner,
      std::unique_ptr<SocketPerformanceWatcher> socket_performance_watcher,
      NetLog* net_log);
  ~QuicChromiumClientSession() override;

  void Initialize() override;

  void AddHandle(Handle* handle);
  void RemoveHandle(Handle* handle);

  // Returns the session's connection migration mode.
  ConnectionMigrationMode connection_migration_mode() const;

  // Waits for the handshake to be confirmed and invokes |callback| when
  // that happens. If the handshake has already been confirmed, returns OK.
  // If the connection has already been closed, returns a net error. If the
  // connection closes before the handshake is confirmed, |callback| will
  // be invoked with an error.
  int WaitForHandshakeConfirmation(CompletionOnceCallback callback);

  // Attempts to create a new stream.  If the stream can be
  // created immediately, returns OK.  If the open stream limit
  // has been reached, returns ERR_IO_PENDING, and |request|
  // will be added to the stream requets queue and will
  // be completed asynchronously.
  // TODO(rch): remove |stream| from this and use setter on |request|
  // and fix in spdy too.
  int TryCreateStream(StreamRequest* request);

  // Cancels the pending stream creation request.
  void CancelRequest(StreamRequest* request);

  // QuicChromiumPacketWriter::Delegate override.
  int HandleWriteError(int error_code,
                       scoped_refptr<QuicChromiumPacketWriter::ReusableIOBuffer>
                           last_packet) override;
  void OnWriteError(int error_code) override;
  // Called when the associated writer is unblocked. Write the cached |packet_|
  // if |packet_| is set. May send a PING packet if
  // |send_packet_after_migration_| is set and writer is not blocked after
  // writing queued packets.
  void OnWriteUnblocked() override;

  // QuicConnectivityProbingManager::Delegate override.
  void OnProbeNetworkSucceeded(
      NetworkChangeNotifier::NetworkHandle network,
      const quic::QuicSocketAddress& self_address,
      std::unique_ptr<DatagramClientSocket> socket,
      std::unique_ptr<QuicChromiumPacketWriter> writer,
      std::unique_ptr<QuicChromiumPacketReader> reader) override;

  void OnProbeNetworkFailed(
      NetworkChangeNotifier::NetworkHandle network) override;

  bool OnSendConnectivityProbingPacket(
      QuicChromiumPacketWriter* writer,
      const quic::QuicSocketAddress& peer_address) override;

  // quic::QuicSpdySession methods:
  size_t WriteHeaders(
      quic::QuicStreamId id,
      spdy::SpdyHeaderBlock headers,
      bool fin,
      spdy::SpdyPriority priority,
      quic::QuicReferenceCountedPointer<quic::QuicAckListenerInterface>
          ack_listener) override;
  void UnregisterStreamPriority(quic::QuicStreamId id, bool is_static) override;
  void UpdateStreamPriority(quic::QuicStreamId id,
                            spdy::SpdyPriority new_priority) override;

  // quic::QuicSession methods:
  void OnStreamFrame(const quic::QuicStreamFrame& frame) override;
  QuicChromiumClientStream* CreateOutgoingDynamicStream() override;
  const quic::QuicCryptoClientStream* GetCryptoStream() const override;
  quic::QuicCryptoClientStream* GetMutableCryptoStream() override;
  void CloseStream(quic::QuicStreamId stream_id) override;
  void SendRstStream(quic::QuicStreamId id,
                     quic::QuicRstStreamErrorCode error,
                     quic::QuicStreamOffset bytes_written) override;
  void OnCryptoHandshakeEvent(CryptoHandshakeEvent event) override;
  void OnCryptoHandshakeMessageSent(
      const quic::CryptoHandshakeMessage& message) override;
  void OnCryptoHandshakeMessageReceived(
      const quic::CryptoHandshakeMessage& message) override;
  void OnGoAway(const quic::QuicGoAwayFrame& frame) override;
  void OnRstStream(const quic::QuicRstStreamFrame& frame) override;

  // QuicClientSessionBase methods:
  void OnConfigNegotiated() override;
  void OnProofValid(
      const quic::QuicCryptoClientConfig::CachedState& cached) override;
  void OnProofVerifyDetailsAvailable(
      const quic::ProofVerifyDetails& verify_details) override;

  // quic::QuicConnectionVisitorInterface methods:
  void OnConnectionClosed(quic::QuicErrorCode error,
                          const std::string& error_details,
                          quic::ConnectionCloseSource source) override;
  void OnSuccessfulVersionNegotiation(
      const quic::ParsedQuicVersion& version) override;
  void OnConnectivityProbeReceived(
      const quic::QuicSocketAddress& self_address,
      const quic::QuicSocketAddress& peer_address) override;
  void OnPathDegrading() override;
  bool HasOpenDynamicStreams() const override;

  // QuicChromiumPacketReader::Visitor methods:
  void OnReadError(int result, const DatagramClientSocket* socket) override;
  bool OnPacket(const quic::QuicReceivedPacket& packet,
                const quic::QuicSocketAddress& local_address,
                const quic::QuicSocketAddress& peer_address) override;

  // MultiplexedSession methods:
  bool GetRemoteEndpoint(IPEndPoint* endpoint) override;
  bool GetSSLInfo(SSLInfo* ssl_info) const override;
  Error GetTokenBindingSignature(crypto::ECPrivateKey* key,
                                 TokenBindingType tb_type,
                                 std::vector<uint8_t>* out) override;

  // Performs a crypto handshake with the server.
  int CryptoConnect(CompletionOnceCallback callback);

  // Causes the QuicConnectionHelper to start reading from all sockets
  // and passing the data along to the quic::QuicConnection.
  void StartReading();

  // Close the session because of |net_error| and notifies the factory
  // that this session has been closed, which will delete the session.
  void CloseSessionOnError(int net_error, quic::QuicErrorCode quic_error);

  // Close the session because of |net_error| and notifies the factory
  // that this session has been closed later, which will delete the session.
  void CloseSessionOnErrorLater(int net_error, quic::QuicErrorCode quic_error);

  std::unique_ptr<base::Value> GetInfoAsValue(
      const std::set<HostPortPair>& aliases);

  const NetLogWithSource& net_log() const { return net_log_; }

  // Returns a Handle to this session.
  std::unique_ptr<QuicChromiumClientSession::Handle> CreateHandle(
      const HostPortPair& destination);

  // Returns the number of client hello messages that have been sent on the
  // crypto stream. If the handshake has completed then this is one greater
  // than the number of round-trips needed for the handshake.
  int GetNumSentClientHellos() const;

  // Returns true if |hostname| may be pooled onto this session.  If this
  // is a secure QUIC session, then |hostname| must match the certificate
  // presented during the handshake.
  bool CanPool(const std::string& hostname,
               PrivacyMode privacy_mode,
               const SocketTag& socket_tag) const;

  const quic::QuicServerId& server_id() const {
    return session_key_.server_id();
  }

  // Attempts to migrate session when |writer| encounters a write error.
  // If |writer| is no longer actively used, abort migration.
  void MigrateSessionOnWriteError(int error_code,
                                  quic::QuicPacketWriter* writer);

  // Helper method that completes connection/server migration.
  // Unblocks packet writer on network level. If the writer becomes unblocked
  // then, OnWriteUnblocked() will be invoked to send packet after migration.
  void WriteToNewSocket();

  // Migrates session over to use |peer_address| and |network|.
  // If |network| is kInvalidNetworkHandle, default network is used. If the
  // migration fails and |close_session_on_error| is true, session will be
  // closed.
  MigrationResult Migrate(NetworkChangeNotifier::NetworkHandle network,
                          IPEndPoint peer_address,
                          bool close_session_on_error,
                          const NetLogWithSource& migration_net_log);

  // Migrates session onto new socket, i.e., starts reading from
  // |socket| in addition to any previous sockets, and sets |writer|
  // to be the new default writer. Returns true if socket was
  // successfully added to the session and the session was
  // successfully migrated to using the new socket. Returns true on
  // successful migration, or false if number of migrations exceeds
  // kMaxReadersPerQuicSession. Takes ownership of |socket|, |reader|,
  // and |writer|.
  bool MigrateToSocket(std::unique_ptr<DatagramClientSocket> socket,
                       std::unique_ptr<QuicChromiumPacketReader> reader,
                       std::unique_ptr<QuicChromiumPacketWriter> writer);

  // Called when NetworkChangeNotifier notifies observers of a newly
  // connected network. Migrates this session to the newly connected
  // network if the session has a pending migration.
  void OnNetworkConnected(NetworkChangeNotifier::NetworkHandle network,
                          const NetLogWithSource& net_log);

  // Called when NetworkChangeNotifier broadcasts to observers of
  // |disconnected_network|.
  void OnNetworkDisconnectedV2(
      NetworkChangeNotifier::NetworkHandle disconnected_network,
      const NetLogWithSource& migration_net_log);

  // Called when NetworkChangeNotifier broadcats to observers of a new default
  // network. Migrates this session to |new_network| if appropriate.
  void OnNetworkMadeDefault(NetworkChangeNotifier::NetworkHandle new_network,
                            const NetLogWithSource& migration_net_log);

  // Schedules a migration alarm to wait for a new network.
  void OnNoNewNetwork();

  // Called when migration alarm fires. If migration has not occurred
  // since alarm was set, closes session with error.
  void OnMigrationTimeout(size_t num_sockets);

  // Populates network error details for this session.
  void PopulateNetErrorDetails(NetErrorDetails* details) const;

  // Returns current default socket. This is the socket over which all
  // QUIC packets are sent. This default socket can change, so do not store the
  // returned socket.
  const DatagramClientSocket* GetDefaultSocket() const;

  bool IsAuthorized(const std::string& hostname) override;

  bool HandlePromised(quic::QuicStreamId associated_id,
                      quic::QuicStreamId promised_id,
                      const spdy::SpdyHeaderBlock& headers) override;

  void DeletePromised(quic::QuicClientPromisedInfo* promised) override;

  void OnPushStreamTimedOut(quic::QuicStreamId stream_id) override;

  // Cancels the push if the push stream for |url| has not been claimed and is
  // still active. Otherwise, no-op.
  void CancelPush(const GURL& url);

  const LoadTimingInfo::ConnectTiming& GetConnectTiming();

  quic::QuicTransportVersion GetQuicVersion() const;

  // Returns the estimate of dynamically allocated memory in bytes.
  // See base/trace_event/memory_usage_estimator.h.
  // TODO(xunjieli): It only tracks |packet_readers_|. Write a better estimate.
  size_t EstimateMemoryUsage() const;

  bool require_confirmation() const { return require_confirmation_; }

 protected:
  // quic::QuicSession methods:
  bool ShouldCreateIncomingDynamicStream(quic::QuicStreamId id) override;
  bool ShouldCreateOutgoingDynamicStream() override;

  QuicChromiumClientStream* CreateIncomingDynamicStream(
      quic::QuicStreamId id) override;

 private:
  friend class test::QuicChromiumClientSessionPeer;

  typedef std::set<Handle*> HandleSet;
  typedef std::list<StreamRequest*> StreamRequestQueue;

  bool WasConnectionEverUsed();

  QuicChromiumClientStream* CreateOutgoingReliableStreamImpl(
      const NetworkTrafficAnnotationTag& traffic_annotation);
  QuicChromiumClientStream* CreateIncomingReliableStreamImpl(
      quic::QuicStreamId id,
      const NetworkTrafficAnnotationTag& traffic_annotation);
  // A completion callback invoked when a read completes.
  void OnReadComplete(int result);

  void OnClosedStream();

  void CloseAllStreams(int net_error);
  void CloseAllHandles(int net_error);
  void CancelAllRequests(int net_error);
  void NotifyRequestsOfConfirmation(int net_error);

  ProbingResult StartProbeNetwork(NetworkChangeNotifier::NetworkHandle network,
                                  IPEndPoint peer_address,
                                  const NetLogWithSource& migration_net_log);

  // Called when there is only one possible working network: |network|, If any
  // error encountered, this session will be cloed. When the migration succeeds:
  //  - If we are no longer on the default interface, migrate back to default
  //    network timer will be set.
  //  - If we are now on the default interface, migrate back to default network
  //    timer will be cancelled.
  void MigrateImmediately(NetworkChangeNotifier::NetworkHandle network);

  void StartMigrateBackToDefaultNetworkTimer(base::TimeDelta delay);
  void CancelMigrateBackToDefaultNetworkTimer();
  void TryMigrateBackToDefaultNetwork(base::TimeDelta timeout);
  void MaybeRetryMigrateBackToDefaultNetwork();

  // Returns true if session is migratable. If not, a task is posted to
  // close the session later if |close_session_if_not_migratable| is true.
  bool IsSessionMigratable(bool close_session_if_not_migratable);
  // Close non-migratable streams in both directions by sending reset stream to
  // peer when connection migration attempts to migrate to the alternate
  // network.
  void ResetNonMigratableStreams();
  void LogMetricsOnNetworkDisconnected();
  void LogMetricsOnNetworkMadeDefault();
  void LogConnectionMigrationResultToHistogram(
      QuicConnectionMigrationStatus status);
  void LogHandshakeStatusOnConnectionMigrationSignal() const;
  void HistogramAndLogMigrationFailure(const NetLogWithSource& net_log,
                                       QuicConnectionMigrationStatus status,
                                       quic::QuicConnectionId connection_id,
                                       const std::string& reason);
  void HistogramAndLogMigrationSuccess(const NetLogWithSource& net_log,
                                       quic::QuicConnectionId connection_id);

  // Notifies the factory that this session is going away and no more streams
  // should be created from it.  This needs to be called before closing any
  // streams, because closing a stream may cause a new stream to be created.
  void NotifyFactoryOfSessionGoingAway();

  // Posts a task to notify the factory that this session has been closed.
  void NotifyFactoryOfSessionClosedLater();

  // Notifies the factory that this session has been closed which will
  // delete |this|.
  void NotifyFactoryOfSessionClosed();

  QuicSessionKey session_key_;
  bool require_confirmation_;
  bool migrate_session_early_v2_;
  bool migrate_session_on_network_change_v2_;
  base::TimeDelta max_time_on_non_default_network_;
  // Maximum allowed number of migrations to non-default network triggered by
  // path degrading per default network.
  int max_migrations_to_non_default_network_on_path_degrading_;
  int current_migrations_to_non_default_network_on_path_degrading_;
  quic::QuicClock* clock_;  // Unowned.
  int yield_after_packets_;
  quic::QuicTime::Delta yield_after_duration_;

  base::TimeTicks most_recent_path_degrading_timestamp_;
  base::TimeTicks most_recent_network_disconnected_timestamp_;

  int most_recent_write_error_;
  base::TimeTicks most_recent_write_error_timestamp_;

  std::unique_ptr<quic::QuicCryptoClientStream> crypto_stream_;
  QuicStreamFactory* stream_factory_;
  std::vector<std::unique_ptr<DatagramClientSocket>> sockets_;
  TransportSecurityState* transport_security_state_;
  std::unique_ptr<QuicServerInfo> server_info_;
  std::unique_ptr<CertVerifyResult> cert_verify_result_;
  std::unique_ptr<ct::CTVerifyResult> ct_verify_result_;
  std::string pinning_failure_log_;
  bool pkp_bypassed_;
  bool is_fatal_cert_error_;
  HandleSet handles_;
  StreamRequestQueue stream_requests_;
  std::vector<CompletionOnceCallback> waiting_for_confirmation_callbacks_;
  CompletionOnceCallback callback_;
  size_t num_total_streams_;
  base::SequencedTaskRunner* task_runner_;
  NetLogWithSource net_log_;
  std::vector<std::unique_ptr<QuicChromiumPacketReader>> packet_readers_;
  LoadTimingInfo::ConnectTiming connect_timing_;
  std::unique_ptr<QuicConnectionLogger> logger_;
  // True when the session is going away, and streams may no longer be created
  // on this session. Existing stream will continue to be processed.
  bool going_away_;
  // True when the session receives a go away from server due to port migration.
  bool port_migration_detected_;
  TokenBindingSignatureMap token_binding_signatures_;
  // Not owned. |push_delegate_| outlives the session and handles server pushes
  // received by session.
  ServerPushDelegate* push_delegate_;
  // UMA histogram counters for streams pushed to this session.
  int streams_pushed_count_;
  int streams_pushed_and_claimed_count_;
  uint64_t bytes_pushed_count_;
  uint64_t bytes_pushed_and_unclaimed_count_;
  // Stores the packet that witnesses socket write error. This packet will be
  // written to an alternate socket when the migration completes and the
  // alternate socket is unblocked.
  scoped_refptr<QuicChromiumPacketWriter::ReusableIOBuffer> packet_;
  // Stores the latest default network platform marks.
  NetworkChangeNotifier::NetworkHandle default_network_;
  QuicConnectivityProbingManager probing_manager_;
  int retry_migrate_back_count_;
  base::OneShotTimer migrate_back_to_default_timer_;
  ConnectionMigrationCause current_connection_migration_cause_;
  // True if a packet needs to be sent when packet writer is unblocked to
  // complete connection migration. The packet can be a cached packet if
  // |packet_| is set, a queued packet, or a PING packet.
  bool send_packet_after_migration_;
  // True if migration is triggered, and there is no alternate network to
  // migrate to.
  bool wait_for_new_network_;
  // True if read errors should be ignored. Set when migration on write error is
  // posted and unset until the first packet is written after migration.
  bool ignore_read_error_;

  // If true, client headers will include HTTP/2 stream dependency info derived
  // from spdy::SpdyPriority.
  bool headers_include_h2_stream_dependency_;
  Http2PriorityDependencies priority_dependency_state_;

  base::WeakPtrFactory<QuicChromiumClientSession> weak_factory_;

  DISALLOW_COPY_AND_ASSIGN(QuicChromiumClientSession);
};

}  // namespace net

#endif  // NET_QUIC_CHROMIUM_QUIC_CHROMIUM_CLIENT_SESSION_H_
