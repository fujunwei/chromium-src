// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef DEVICE_BLUETOOTH_TEST_FAKE_BLUETOOTH_LE_MANUFACTURER_DATA_WINRT_H_
#define DEVICE_BLUETOOTH_TEST_FAKE_BLUETOOTH_LE_MANUFACTURER_DATA_WINRT_H_

#include <windows.devices.bluetooth.advertisement.h>
#include <wrl/implements.h>

#include <stdint.h>

#include <vector>

#include "base/macros.h"

namespace device {

class FakeBluetoothLEManufacturerData
    : public Microsoft::WRL::RuntimeClass<
          Microsoft::WRL::RuntimeClassFlags<
              Microsoft::WRL::WinRt | Microsoft::WRL::InhibitRoOriginateError>,
          ABI::Windows::Devices::Bluetooth::Advertisement::
              IBluetoothLEManufacturerData> {
 public:
  FakeBluetoothLEManufacturerData(uint16_t company_id,
                                  std::vector<uint8_t> data);
  ~FakeBluetoothLEManufacturerData() override;

  // IBluetoothLEManufacturerData:
  IFACEMETHODIMP get_CompanyId(uint16_t* value) override;
  IFACEMETHODIMP put_CompanyId(uint16_t value) override;
  IFACEMETHODIMP get_Data(
      ABI::Windows::Storage::Streams::IBuffer** value) override;
  IFACEMETHODIMP put_Data(
      ABI::Windows::Storage::Streams::IBuffer* value) override;

 private:
  uint16_t company_id_;
  std::vector<uint8_t> data_;

  DISALLOW_COPY_AND_ASSIGN(FakeBluetoothLEManufacturerData);
};

}  // namespace device

#endif  // DEVICE_BLUETOOTH_TEST_FAKE_BLUETOOTH_LE_MANUFACTURER_DATA_WINRT_H_
