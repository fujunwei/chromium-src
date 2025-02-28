// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef DEVICE_FIDO_FIDO_CABLE_DISCOVERY_H_
#define DEVICE_FIDO_FIDO_CABLE_DISCOVERY_H_

#include <stdint.h>

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/component_export.h"
#include "base/containers/span.h"
#include "base/macros.h"
#include "base/memory/ref_counted.h"
#include "base/memory/weak_ptr.h"
#include "device/fido/ble/fido_ble_discovery_base.h"

namespace device {

class FidoCableDevice;
class BluetoothDevice;
class BluetoothAdvertisement;
class FidoCableHandshakeHandler;

class COMPONENT_EXPORT(DEVICE_FIDO) FidoCableDiscovery
    : public FidoBleDiscoveryBase {
 public:
  static constexpr size_t kEphemeralIdSize = 16;
  static constexpr size_t kSessionPreKeySize = 32;
  using EidArray = std::array<uint8_t, kEphemeralIdSize>;
  using SessionPreKeyArray = std::array<uint8_t, kSessionPreKeySize>;

  // Encapsulates information required to discover Cable device per single
  // credential. When multiple credentials are enrolled to a single account
  // (i.e. more than one phone has been enrolled to an user account as a
  // security key), then FidoCableDiscovery must advertise for all of the client
  // EID received from the relying party.
  // TODO(hongjunchoi): Add discovery data required for MakeCredential request.
  // See: https://crbug.com/837088
  struct COMPONENT_EXPORT(DEVICE_FIDO) CableDiscoveryData {
    CableDiscoveryData(uint8_t version,
                       const EidArray& client_eid,
                       const EidArray& authenticator_eid,
                       const SessionPreKeyArray& session_pre_key);
    CableDiscoveryData(const CableDiscoveryData& data);
    CableDiscoveryData& operator=(const CableDiscoveryData& other);
    ~CableDiscoveryData();

    uint8_t version;
    EidArray client_eid;
    EidArray authenticator_eid;
    SessionPreKeyArray session_pre_key;
  };

  FidoCableDiscovery(std::vector<CableDiscoveryData> discovery_data);
  ~FidoCableDiscovery() override;

 protected:
  virtual std::unique_ptr<FidoCableHandshakeHandler> CreateHandshakeHandler(
      FidoCableDevice* device,
      base::span<const uint8_t, kSessionPreKeySize> session_pre_key,
      base::span<const uint8_t, 8> nonce);

 private:
  FRIEND_TEST_ALL_PREFIXES(FidoCableDiscoveryTest,
                           TestUnregisterAdvertisementUponDestruction);

  // BluetoothAdapter::Observer:
  void DeviceAdded(BluetoothAdapter* adapter, BluetoothDevice* device) override;
  void DeviceChanged(BluetoothAdapter* adapter,
                     BluetoothDevice* device) override;
  void DeviceRemoved(BluetoothAdapter* adapter,
                     BluetoothDevice* device) override;

  // FidoBleDiscoveryBase:
  void OnSetPowered() override;

  void StartAdvertisement();
  void OnAdvertisementRegistered(
      const EidArray& client_eid,
      scoped_refptr<BluetoothAdvertisement> advertisement);
  void OnAdvertisementRegisterError(
      BluetoothAdvertisement::ErrorCode error_code);
  // Keeps a counter of success/failure of advertisements done by the client.
  // If all advertisements fail, then immediately stop discovery process and
  // invoke NotifyDiscoveryStarted(false). Otherwise kick off discovery session
  // once all advertisements has been processed.
  void RecordAdvertisementResult(bool is_success);
  void CableDeviceFound(BluetoothAdapter* adapter, BluetoothDevice* device);
  void ConductEncryptionHandshake(
      std::unique_ptr<FidoCableDevice> device,
      base::span<const uint8_t, kSessionPreKeySize> session_pre_key,
      base::span<const uint8_t, 8> nonce);
  void ValidateAuthenticatorHandshakeMessage(
      std::unique_ptr<FidoCableDevice> cable_device,
      FidoCableHandshakeHandler* handshake_handler,
      base::Optional<std::vector<uint8_t>> handshake_response);

  const CableDiscoveryData* GetFoundCableDiscoveryData(
      const BluetoothDevice* device) const;

  std::vector<CableDiscoveryData> discovery_data_;
  size_t advertisement_success_counter_ = 0;
  size_t advertisement_failure_counter_ = 0;
  std::map<EidArray, scoped_refptr<BluetoothAdvertisement>> advertisements_;
  std::map<std::string, std::unique_ptr<FidoCableHandshakeHandler>>
      cable_handshake_handlers_;
  base::WeakPtrFactory<FidoCableDiscovery> weak_factory_;

  DISALLOW_COPY_AND_ASSIGN(FidoCableDiscovery);
};

}  // namespace device

#endif  // DEVICE_FIDO_FIDO_CABLE_DISCOVERY_H_
