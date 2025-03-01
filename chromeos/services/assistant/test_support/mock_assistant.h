// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef CHROMEOS_SERVICES_ASSISTANT_TEST_SUPPORT_MOCK_ASSISTANT_H_
#define CHROMEOS_SERVICES_ASSISTANT_TEST_SUPPORT_MOCK_ASSISTANT_H_

#include "base/macros.h"
#include "chromeos/services/assistant/public/mojom/assistant.mojom.h"
#include "testing/gmock/include/gmock/gmock.h"

namespace gfx {
class Rect;
}  // namespace gfx

namespace chromeos {
namespace assistant {

class MockAssistant : public mojom::Assistant {
 public:
  MockAssistant();
  ~MockAssistant() override;

  MOCK_METHOD0(StartVoiceInteraction, void());

  MOCK_METHOD0(StopActiveInteraction, void());

  MOCK_METHOD1(SendTextQuery, void(const std::string&));

  MOCK_METHOD1(
      AddAssistantInteractionSubscriber,
      void(chromeos::assistant::mojom::AssistantInteractionSubscriberPtr));

  MOCK_METHOD1(
      AddAssistantNotificationSubscriber,
      void(chromeos::assistant::mojom::AssistantNotificationSubscriberPtr));

  MOCK_METHOD1(
      AddAssistantScreenContextSubscriber,
      void(chromeos::assistant::mojom::AssistantScreenContextSubscriberPtr));

  MOCK_METHOD2(RetrieveNotification,
               void(chromeos::assistant::mojom::AssistantNotificationPtr, int));

  MOCK_METHOD1(DismissNotification,
               void(chromeos::assistant::mojom::AssistantNotificationPtr));

  MOCK_METHOD2(RequestScreenContext, void(const gfx::Rect&, base::OnceClosure));

 private:
  DISALLOW_COPY_AND_ASSIGN(MockAssistant);
};

}  // namespace assistant
}  // namespace chromeos

#endif  // CHROMEOS_SERVICES_ASSISTANT_TEST_SUPPORT_MOCK_ASSISTANT_H_
