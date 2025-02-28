// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "components/exo/text_input.h"

#include "base/strings/utf_string_conversions.h"
#include "components/exo/buffer.h"
#include "components/exo/shell_surface.h"
#include "components/exo/surface.h"
#include "components/exo/test/exo_test_base.h"
#include "components/exo/test/exo_test_helper.h"
#include "testing/gmock/include/gmock/gmock.h"
#include "ui/aura/window_tree_host.h"
#include "ui/base/ime/composition_text.h"
#include "ui/base/ime/input_method_factory.h"
#include "ui/base/ime/input_method_observer.h"
#include "ui/views/widget/widget.h"

using testing::_;

namespace exo {

namespace {

class MockTextInputDelegate : public TextInput::Delegate {
 public:
  MockTextInputDelegate() = default;

  // TextInput::Delegate:
  MOCK_METHOD0(Activated, void());
  MOCK_METHOD0(Deactivated, void());
  MOCK_METHOD1(OnVirtualKeyboardVisibilityChanged, void(bool));
  MOCK_METHOD1(SetCompositionText, void(const ui::CompositionText&));
  MOCK_METHOD1(Commit, void(const base::string16&));
  MOCK_METHOD1(SetCursor, void(const gfx::Range&));
  MOCK_METHOD1(DeleteSurroundingText, void(const gfx::Range&));
  MOCK_METHOD1(SendKey, void(const ui::KeyEvent&));
  MOCK_METHOD1(OnLanguageChanged, void(const std::string&));
  MOCK_METHOD1(OnTextDirectionChanged,
               void(base::i18n::TextDirection direction));

 private:
  DISALLOW_COPY_AND_ASSIGN(MockTextInputDelegate);
};

class TestingInputMethodObserver : public ui::InputMethodObserver {
 public:
  TestingInputMethodObserver(ui::InputMethod* input_method)
      : input_method_(input_method) {
    input_method_->AddObserver(this);
  }

  ~TestingInputMethodObserver() override {
    input_method_->RemoveObserver(this);
  }

  // ui::InputMethodObserver
  MOCK_METHOD0(OnFocus, void());
  MOCK_METHOD0(OnBlur, void());
  MOCK_METHOD1(OnCaretBoundsChanged, void(const ui::TextInputClient*));
  MOCK_METHOD1(OnTextInputStateChanged, void(const ui::TextInputClient*));
  MOCK_METHOD1(OnInputMethodDestroyed, void(const ui::InputMethod*));
  MOCK_METHOD0(OnShowVirtualKeyboardIfEnabled, void());

 private:
  ui::InputMethod* input_method_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(TestingInputMethodObserver);
};

class TextInputTest : public test::ExoTestBase {
 public:
  TextInputTest() = default;

  void SetUp() override {
    ui::SetUpInputMethodFactoryForTesting();
    test::ExoTestBase::SetUp();
    text_input_ =
        std::make_unique<TextInput>(std::make_unique<MockTextInputDelegate>());
    SetupSurface();
  }

  void TearDown() override {
    TearDownSurface();
    text_input_.reset();
    test::ExoTestBase::TearDown();
  }

  void SetupSurface() {
    gfx::Size buffer_size(32, 32);
    buffer_ = std::make_unique<Buffer>(
        exo_test_helper()->CreateGpuMemoryBuffer(buffer_size));
    surface_ = std::make_unique<Surface>();
    shell_surface_ = std::make_unique<ShellSurface>(surface_.get());

    surface_->Attach(buffer_.get());
    surface_->Commit();

    gfx::Point origin(100, 100);
    shell_surface_->SetGeometry(gfx::Rect(origin, buffer_size));
  }

  void TearDownSurface() {
    shell_surface_.reset();
    surface_.reset();
    buffer_.reset();
  }

 protected:
  TextInput* text_input() { return text_input_.get(); }
  MockTextInputDelegate* delegate() {
    return static_cast<MockTextInputDelegate*>(text_input_->delegate());
  }
  Surface* surface() { return surface_.get(); }

  ui::InputMethod* GetInputMethod() {
    return surface_->window()->GetHost()->GetInputMethod();
  }

 private:
  std::unique_ptr<TextInput> text_input_;

  std::unique_ptr<Buffer> buffer_;
  std::unique_ptr<Surface> surface_;
  std::unique_ptr<ShellSurface> shell_surface_;

  DISALLOW_COPY_AND_ASSIGN(TextInputTest);
};

TEST_F(TextInputTest, Activate) {
  EXPECT_EQ(ui::TEXT_INPUT_TYPE_NONE, text_input()->GetTextInputType());
  EXPECT_EQ(ui::TEXT_INPUT_MODE_DEFAULT, text_input()->GetTextInputMode());

  EXPECT_CALL(*delegate(), Activated).Times(1);
  text_input()->Activate(surface());
  EXPECT_EQ(ui::TEXT_INPUT_TYPE_TEXT, text_input()->GetTextInputType());
  EXPECT_EQ(ui::TEXT_INPUT_MODE_TEXT, text_input()->GetTextInputMode());
  EXPECT_EQ(0, text_input()->GetTextInputFlags());

  EXPECT_CALL(*delegate(), Deactivated).Times(1);
  text_input()->Deactivate();
  EXPECT_EQ(ui::TEXT_INPUT_TYPE_NONE, text_input()->GetTextInputType());
  EXPECT_EQ(ui::TEXT_INPUT_MODE_DEFAULT, text_input()->GetTextInputMode());
}

TEST_F(TextInputTest, ShowVirtualKeyboardIfEnabled) {
  TestingInputMethodObserver observer(GetInputMethod());

  EXPECT_CALL(observer, OnTextInputStateChanged(text_input())).Times(1);
  EXPECT_CALL(*delegate(), Activated).Times(1);
  text_input()->Activate(surface());

  EXPECT_CALL(observer, OnShowVirtualKeyboardIfEnabled)
      .WillOnce(testing::Invoke(
          [this]() { text_input()->OnKeyboardVisibilityStateChanged(true); }));
  EXPECT_CALL(*delegate(), OnVirtualKeyboardVisibilityChanged(true)).Times(1);
  text_input()->ShowVirtualKeyboardIfEnabled();

  EXPECT_CALL(*delegate(), Deactivated).Times(1);
  text_input()->Deactivate();
}

TEST_F(TextInputTest, ShowVirtualKeyboardIfEnabledBeforeActivated) {
  TestingInputMethodObserver observer(GetInputMethod());

  // ShowVirtualKeyboardIfEnabled before activation.
  text_input()->ShowVirtualKeyboardIfEnabled();

  EXPECT_CALL(observer, OnTextInputStateChanged(text_input())).Times(1);
  EXPECT_CALL(observer, OnShowVirtualKeyboardIfEnabled)
      .WillOnce(testing::Invoke(
          [this]() { text_input()->OnKeyboardVisibilityStateChanged(true); }));
  EXPECT_CALL(*delegate(), Activated).Times(1);
  EXPECT_CALL(*delegate(), OnVirtualKeyboardVisibilityChanged(true)).Times(1);
  text_input()->Activate(surface());

  EXPECT_CALL(*delegate(), Deactivated).Times(1);
}

TEST_F(TextInputTest, SetTypeModeFlag) {
  TestingInputMethodObserver observer(GetInputMethod());

  EXPECT_CALL(observer, OnTextInputStateChanged(text_input())).Times(1);
  EXPECT_CALL(*delegate(), Activated).Times(1);
  text_input()->Activate(surface());
  EXPECT_EQ(ui::TEXT_INPUT_TYPE_TEXT, text_input()->GetTextInputType());
  EXPECT_EQ(ui::TEXT_INPUT_MODE_TEXT, text_input()->GetTextInputMode());
  EXPECT_EQ(0, text_input()->GetTextInputFlags());
  EXPECT_TRUE(text_input()->ShouldDoLearning());

  int flags = ui::TEXT_INPUT_FLAG_AUTOCOMPLETE_OFF |
              ui::TEXT_INPUT_FLAG_AUTOCAPITALIZE_NONE;
  EXPECT_CALL(observer, OnTextInputStateChanged(text_input())).Times(1);
  text_input()->SetTypeModeFlags(ui::TEXT_INPUT_TYPE_URL,
                                 ui::TEXT_INPUT_MODE_URL, flags, false);

  EXPECT_EQ(ui::TEXT_INPUT_TYPE_URL, text_input()->GetTextInputType());
  EXPECT_EQ(ui::TEXT_INPUT_MODE_URL, text_input()->GetTextInputMode());
  EXPECT_EQ(flags, text_input()->GetTextInputFlags());
  EXPECT_FALSE(text_input()->ShouldDoLearning());

  EXPECT_CALL(*delegate(), Deactivated).Times(1);
}

TEST_F(TextInputTest, CaretBounds) {
  TestingInputMethodObserver observer(GetInputMethod());

  EXPECT_CALL(observer, OnTextInputStateChanged(text_input())).Times(1);
  EXPECT_CALL(*delegate(), Activated).Times(1);
  text_input()->Activate(surface());

  gfx::Rect bounds(10, 10, 0, 16);
  EXPECT_CALL(observer, OnCaretBoundsChanged(text_input())).Times(1);
  text_input()->SetCaretBounds(bounds);

  EXPECT_EQ(bounds.size().ToString(),
            text_input()->GetCaretBounds().size().ToString());
  gfx::Point origin = surface()->window()->GetBoundsInScreen().origin();
  origin += bounds.OffsetFromOrigin();
  EXPECT_EQ(origin.ToString(),
            text_input()->GetCaretBounds().origin().ToString());

  EXPECT_CALL(*delegate(), Deactivated).Times(1);
}

TEST_F(TextInputTest, CompositionText) {
  ui::CompositionText t;
  t.text = base::ASCIIToUTF16("composition");
  t.selection = gfx::Range(1u);
  t.ime_text_spans.push_back(
      ui::ImeTextSpan(0, t.text.size(), ui::ImeTextSpan::Thickness::kThick));

  ui::CompositionText empty;
  EXPECT_CALL(*delegate(), SetCompositionText(t)).Times(1);
  EXPECT_CALL(*delegate(), SetCompositionText(empty)).Times(1);

  text_input()->SetCompositionText(t);
  text_input()->ClearCompositionText();
}

TEST_F(TextInputTest, CommitCompositionText) {
  ui::CompositionText t;
  t.text = base::ASCIIToUTF16("composition");
  t.selection = gfx::Range(1u);
  t.ime_text_spans.push_back(
      ui::ImeTextSpan(0, t.text.size(), ui::ImeTextSpan::Thickness::kThick));

  EXPECT_CALL(*delegate(), SetCompositionText(t)).Times(1);
  EXPECT_CALL(*delegate(), Commit(t.text)).Times(1);

  text_input()->SetCompositionText(t);
  text_input()->ConfirmCompositionText();
}

TEST_F(TextInputTest, Commit) {
  base::string16 s = base::ASCIIToUTF16("commit text");

  EXPECT_CALL(*delegate(), Commit(s)).Times(1);
  text_input()->InsertText(s);
}

TEST_F(TextInputTest, InsertChar) {
  ui::KeyEvent ev(ui::ET_KEY_PRESSED, ui::VKEY_RETURN, 0);

  EXPECT_CALL(*delegate(), SendKey(testing::Ref(ev))).Times(1);
  text_input()->InsertChar(ev);
}

TEST_F(TextInputTest, InsertCharNormalKey) {
  base::char16 ch = 'x';
  ui::KeyEvent ev(ch, ui::VKEY_X, 0);

  EXPECT_CALL(*delegate(), Commit(base::string16(1, ch))).Times(1);
  EXPECT_CALL(*delegate(), SendKey(_)).Times(0);
  text_input()->InsertChar(ev);
}

}  // anonymous namespace
}  // namespace exo