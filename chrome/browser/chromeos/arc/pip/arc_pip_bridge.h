// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROME_BROWSER_CHROMEOS_ARC_PIP_ARC_PIP_BRIDGE_H_
#define CHROME_BROWSER_CHROMEOS_ARC_PIP_ARC_PIP_BRIDGE_H_

#include "components/arc/common/pip.mojom.h"
#include "components/arc/connection_observer.h"
#include "components/keyed_service/core/keyed_service.h"

namespace content {
class BrowserContext;
}  // namespace content

namespace arc {

class ArcBridgeService;

class ArcPipBridge : public KeyedService,
                     public ConnectionObserver<mojom::PipInstance>,
                     public mojom::PipHost {
 public:
  // Returns singleton instance for the given BrowserContext,
  // or nullptr if the browser |context| is not allowed to use ARC.
  static ArcPipBridge* GetForBrowserContext(content::BrowserContext* context);

  ArcPipBridge(content::BrowserContext* context,
               ArcBridgeService* bridge_service);
  ~ArcPipBridge() override;

  // ConnectionObserver<mojom::PipInstance> overrides.
  void OnConnectionReady() override;
  void OnConnectionClosed() override;

  // PipHost overrides.
  void OnPipEvent(arc::mojom::ArcPipEvent event) override;

  // PipInstance methods:
  void ClosePip();

 private:
  ArcBridgeService* const arc_bridge_service_;

  DISALLOW_COPY_AND_ASSIGN(ArcPipBridge);
};

}  // namespace arc

#endif  // CHROME_BROWSER_CHROMEOS_ARC_PIP_ARC_PIP_BRIDGE_H_
