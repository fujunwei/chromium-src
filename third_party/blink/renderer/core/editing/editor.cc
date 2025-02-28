/*
 * Copyright (C) 2006, 2007, 2008, 2011 Apple Inc. All rights reserved.
 * Copyright (C) 2008 Nokia Corporation and/or its subsidiary(-ies)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY APPLE COMPUTER, INC. ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL APPLE COMPUTER, INC. OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "third_party/blink/renderer/core/editing/editor.h"

#include "third_party/blink/public/platform/web_scroll_into_view_params.h"
#include "third_party/blink/renderer/core/accessibility/ax_object_cache.h"
#include "third_party/blink/renderer/core/clipboard/data_object.h"
#include "third_party/blink/renderer/core/clipboard/data_transfer.h"
#include "third_party/blink/renderer/core/clipboard/data_transfer_access_policy.h"
#include "third_party/blink/renderer/core/clipboard/system_clipboard.h"
#include "third_party/blink/renderer/core/css/css_computed_style_declaration.h"
#include "third_party/blink/renderer/core/css/css_property_value_set.h"
#include "third_party/blink/renderer/core/css_property_names.h"
#include "third_party/blink/renderer/core/dom/document_fragment.h"
#include "third_party/blink/renderer/core/dom/element_traversal.h"
#include "third_party/blink/renderer/core/dom/events/scoped_event_queue.h"
#include "third_party/blink/renderer/core/dom/node_traversal.h"
#include "third_party/blink/renderer/core/dom/parser_content_policy.h"
#include "third_party/blink/renderer/core/dom/text.h"
#include "third_party/blink/renderer/core/editing/commands/apply_style_command.h"
#include "third_party/blink/renderer/core/editing/commands/delete_selection_command.h"
#include "third_party/blink/renderer/core/editing/commands/indent_outdent_command.h"
#include "third_party/blink/renderer/core/editing/commands/insert_list_command.h"
#include "third_party/blink/renderer/core/editing/commands/replace_selection_command.h"
#include "third_party/blink/renderer/core/editing/commands/simplify_markup_command.h"
#include "third_party/blink/renderer/core/editing/commands/typing_command.h"
#include "third_party/blink/renderer/core/editing/commands/undo_stack.h"
#include "third_party/blink/renderer/core/editing/editing_behavior.h"
#include "third_party/blink/renderer/core/editing/editing_style_utilities.h"
#include "third_party/blink/renderer/core/editing/editing_tri_state.h"
#include "third_party/blink/renderer/core/editing/editing_utilities.h"
#include "third_party/blink/renderer/core/editing/ephemeral_range.h"
#include "third_party/blink/renderer/core/editing/frame_selection.h"
#include "third_party/blink/renderer/core/editing/ime/input_method_controller.h"
#include "third_party/blink/renderer/core/editing/iterators/search_buffer.h"
#include "third_party/blink/renderer/core/editing/kill_ring.h"
#include "third_party/blink/renderer/core/editing/markers/document_marker.h"
#include "third_party/blink/renderer/core/editing/markers/document_marker_controller.h"
#include "third_party/blink/renderer/core/editing/selection_template.h"
#include "third_party/blink/renderer/core/editing/serializers/serialization.h"
#include "third_party/blink/renderer/core/editing/set_selection_options.h"
#include "third_party/blink/renderer/core/editing/spellcheck/spell_checker.h"
#include "third_party/blink/renderer/core/editing/visible_position.h"
#include "third_party/blink/renderer/core/editing/visible_units.h"
#include "third_party/blink/renderer/core/editing/writing_direction.h"
#include "third_party/blink/renderer/core/event_names.h"
#include "third_party/blink/renderer/core/events/keyboard_event.h"
#include "third_party/blink/renderer/core/events/text_event.h"
#include "third_party/blink/renderer/core/frame/local_frame.h"
#include "third_party/blink/renderer/core/frame/local_frame_view.h"
#include "third_party/blink/renderer/core/frame/settings.h"
#include "third_party/blink/renderer/core/frame/use_counter.h"
#include "third_party/blink/renderer/core/html/forms/html_input_element.h"
#include "third_party/blink/renderer/core/html/forms/html_text_area_element.h"
#include "third_party/blink/renderer/core/html/html_image_element.h"
#include "third_party/blink/renderer/core/html_names.h"
#include "third_party/blink/renderer/core/input/event_handler.h"
#include "third_party/blink/renderer/core/input_type_names.h"
#include "third_party/blink/renderer/core/layout/hit_test_result.h"
#include "third_party/blink/renderer/core/layout/layout_object.h"
#include "third_party/blink/renderer/core/loader/empty_clients.h"
#include "third_party/blink/renderer/core/loader/resource/image_resource_content.h"
#include "third_party/blink/renderer/core/page/drag_data.h"
#include "third_party/blink/renderer/core/page/focus_controller.h"
#include "third_party/blink/renderer/core/page/page.h"
#include "third_party/blink/renderer/platform/bindings/exception_state.h"
#include "third_party/blink/renderer/platform/scroll/scroll_alignment.h"
#include "third_party/blink/renderer/platform/weborigin/kurl.h"
#include "third_party/blink/renderer/platform/wtf/text/character_names.h"

namespace blink {

using namespace HTMLNames;

namespace {

bool IsInPasswordFieldWithUnrevealedPassword(const Position& position) {
  if (auto* input = ToHTMLInputElementOrNull(EnclosingTextControl(position))) {
    return (input->type() == InputTypeNames::password) &&
           !input->ShouldRevealPassword();
  }
  return false;
}

}  // anonymous namespace

// When an event handler has moved the selection outside of a text control
// we should use the target control's selection for this editing operation.
SelectionInDOMTree Editor::SelectionForCommand(Event* event) {
  const SelectionInDOMTree selection =
      GetFrameSelection().GetSelectionInDOMTree();
  if (!event)
    return selection;
  // If the target is a text control, and the current selection is outside of
  // its shadow tree, then use the saved selection for that text control.
  if (!IsTextControl(*event->target()->ToNode()))
    return selection;
  auto* text_control_of_selection_start =
      EnclosingTextControl(selection.Base());
  auto* text_control_of_target = ToTextControl(event->target()->ToNode());
  if (!selection.IsNone() &&
      text_control_of_target == text_control_of_selection_start)
    return selection;
  const SelectionInDOMTree& select = text_control_of_target->Selection();
  if (select.IsNone())
    return selection;
  return select;
}

// Function considers Mac editing behavior a fallback when Page or Settings is
// not available.
EditingBehavior Editor::Behavior() const {
  if (!GetFrame().GetSettings())
    return EditingBehavior(kEditingMacBehavior);

  return EditingBehavior(GetFrame().GetSettings()->GetEditingBehaviorType());
}

static bool IsCaretAtStartOfWrappedLine(const FrameSelection& selection) {
  if (!selection.ComputeVisibleSelectionInDOMTree().IsCaret())
    return false;
  if (selection.GetSelectionInDOMTree().Affinity() != TextAffinity::kDownstream)
    return false;
  const Position& position =
      selection.ComputeVisibleSelectionInDOMTree().Start();
  if (InSameLine(PositionWithAffinity(position, TextAffinity::kUpstream),
                 PositionWithAffinity(position, TextAffinity::kDownstream)))
    return false;

  // Only when the previous character is a space to avoid undesired side
  // effects. There are cases where a new line is desired even if the previous
  // character is not a space, but typing another space will do.
  Position prev =
      PreviousPositionOf(position, PositionMoveType::kGraphemeCluster);
  const Node* prev_node = prev.ComputeContainerNode();
  if (!prev_node || !prev_node->IsTextNode())
    return false;
  int prev_offset = prev.ComputeOffsetInContainerNode();
  UChar prev_char = ToText(prev_node)->data()[prev_offset];
  return prev_char == kSpaceCharacter;
}

bool Editor::HandleTextEvent(TextEvent* event) {
  // Default event handling for Drag and Drop will be handled by DragController
  // so we leave the event for it.
  if (event->IsDrop())
    return false;

  // Default event handling for IncrementalInsertion will be handled by
  // TypingCommand::insertText(), so we leave the event for it.
  if (event->IsIncrementalInsertion())
    return false;

  // TODO(editing-dev): The use of UpdateStyleAndLayoutIgnorePendingStylesheets
  // needs to be audited.  See http://crbug.com/590369 for more details.
  frame_->GetDocument()->UpdateStyleAndLayoutIgnorePendingStylesheets();

  if (event->IsPaste()) {
    if (event->PastingFragment()) {
      ReplaceSelectionWithFragment(
          event->PastingFragment(), false, event->ShouldSmartReplace(),
          event->ShouldMatchStyle(), InputEvent::InputType::kInsertFromPaste);
    } else {
      ReplaceSelectionWithText(event->data(), false,
                               event->ShouldSmartReplace(),
                               InputEvent::InputType::kInsertFromPaste);
    }
    return true;
  }

  String data = event->data();
  if (data == "\n") {
    if (event->IsLineBreak())
      return InsertLineBreak();
    return InsertParagraphSeparator();
  }

  // Typing spaces at the beginning of wrapped line is confusing, because
  // inserted spaces would appear in the previous line.
  // Insert a line break automatically so that the spaces appear at the caret.
  // TODO(kojii): rich editing has the same issue, but has more options and
  // needs coordination with JS. Enable for plaintext only for now and collect
  // feedback.
  if (data == " " && !CanEditRichly() &&
      IsCaretAtStartOfWrappedLine(GetFrameSelection())) {
    InsertLineBreak();
  }

  return InsertTextWithoutSendingTextEvent(data, false, event);
}

bool Editor::CanEdit() const {
  return GetFrame()
      .Selection()
      .ComputeVisibleSelectionInDOMTreeDeprecated()
      .RootEditableElement();
}

bool Editor::CanEditRichly() const {
  return IsRichlyEditablePosition(
      GetFrame()
          .Selection()
          .ComputeVisibleSelectionInDOMTreeDeprecated()
          .Base());
}

bool Editor::CanCut() const {
  return CanCopy() && CanDelete();
}

bool Editor::CanCopy() const {
  if (ImageElementFromImageDocument(GetFrame().GetDocument()))
    return true;
  FrameSelection& selection = GetFrameSelection();
  if (!selection.IsAvailable())
    return false;
  return selection.ComputeVisibleSelectionInDOMTreeDeprecated().IsRange() &&
         !IsInPasswordFieldWithUnrevealedPassword(
             selection.ComputeVisibleSelectionInDOMTree().Start());
}

bool Editor::CanPaste() const {
  return CanEdit();
}

bool Editor::CanDelete() const {
  FrameSelection& selection = GetFrameSelection();
  return selection.ComputeVisibleSelectionInDOMTreeDeprecated().IsRange() &&
         selection.ComputeVisibleSelectionInDOMTree().RootEditableElement();
}

bool Editor::SmartInsertDeleteEnabled() const {
  if (Settings* settings = GetFrame().GetSettings())
    return settings->GetSmartInsertDeleteEnabled();
  return false;
}

bool Editor::IsSelectTrailingWhitespaceEnabled() const {
  if (Settings* settings = GetFrame().GetSettings())
    return settings->GetSelectTrailingWhitespaceEnabled();
  return false;
}

void Editor::DeleteSelectionWithSmartDelete(
    DeleteMode delete_mode,
    InputEvent::InputType input_type,
    const Position& reference_move_position) {
  if (GetFrame()
          .Selection()
          .ComputeVisibleSelectionInDOMTreeDeprecated()
          .IsNone())
    return;

  DCHECK(GetFrame().GetDocument());
  DeleteSelectionCommand::Create(
      *GetFrame().GetDocument(),
      DeleteSelectionOptions::Builder()
          .SetSmartDelete(delete_mode == DeleteMode::kSmart)
          .SetMergeBlocksAfterDelete(true)
          .SetExpandForSpecialElements(true)
          .SetSanitizeMarkup(true)
          .Build(),
      input_type, reference_move_position)
      ->Apply();
}

void Editor::ReplaceSelectionWithFragment(DocumentFragment* fragment,
                                          bool select_replacement,
                                          bool smart_replace,
                                          bool match_style,
                                          InputEvent::InputType input_type) {
  DCHECK(!GetFrame().GetDocument()->NeedsLayoutTreeUpdate());
  const VisibleSelection& selection =
      GetFrameSelection().ComputeVisibleSelectionInDOMTree();
  if (selection.IsNone() || !selection.IsContentEditable() || !fragment)
    return;

  ReplaceSelectionCommand::CommandOptions options =
      ReplaceSelectionCommand::kPreventNesting |
      ReplaceSelectionCommand::kSanitizeFragment;
  if (select_replacement)
    options |= ReplaceSelectionCommand::kSelectReplacement;
  if (smart_replace)
    options |= ReplaceSelectionCommand::kSmartReplace;
  if (match_style)
    options |= ReplaceSelectionCommand::kMatchStyle;
  DCHECK(GetFrame().GetDocument());
  ReplaceSelectionCommand::Create(*GetFrame().GetDocument(), fragment, options,
                                  input_type)
      ->Apply();
  RevealSelectionAfterEditingOperation();
}

void Editor::ReplaceSelectionWithText(const String& text,
                                      bool select_replacement,
                                      bool smart_replace,
                                      InputEvent::InputType input_type) {
  ReplaceSelectionWithFragment(CreateFragmentFromText(SelectedRange(), text),
                               select_replacement, smart_replace, true,
                               input_type);
}

void Editor::ReplaceSelectionAfterDragging(DocumentFragment* fragment,
                                           InsertMode insert_mode,
                                           DragSourceType drag_source_type) {
  ReplaceSelectionCommand::CommandOptions options =
      ReplaceSelectionCommand::kSelectReplacement |
      ReplaceSelectionCommand::kPreventNesting;
  if (insert_mode == InsertMode::kSmart)
    options |= ReplaceSelectionCommand::kSmartReplace;
  if (drag_source_type == DragSourceType::kPlainTextSource)
    options |= ReplaceSelectionCommand::kMatchStyle;
  DCHECK(GetFrame().GetDocument());
  ReplaceSelectionCommand::Create(*GetFrame().GetDocument(), fragment, options,
                                  InputEvent::InputType::kInsertFromDrop)
      ->Apply();
}

bool Editor::DeleteSelectionAfterDraggingWithEvents(
    Element* drag_source,
    DeleteMode delete_mode,
    const Position& reference_move_position) {
  if (!drag_source || !drag_source->isConnected())
    return true;

  // Dispatch 'beforeinput'.
  const bool should_delete =
      DispatchBeforeInputEditorCommand(
          drag_source, InputEvent::InputType::kDeleteByDrag,
          TargetRangesForInputEvent(*drag_source)) ==
      DispatchEventResult::kNotCanceled;

  // 'beforeinput' event handler may destroy frame, return false to cancel
  // remaining actions;
  if (frame_->GetDocument()->GetFrame() != frame_)
    return false;

  if (should_delete && drag_source->isConnected()) {
    DeleteSelectionWithSmartDelete(delete_mode,
                                   InputEvent::InputType::kDeleteByDrag,
                                   reference_move_position);
  }

  return true;
}

bool Editor::ReplaceSelectionAfterDraggingWithEvents(
    Element* drop_target,
    DragData* drag_data,
    DocumentFragment* fragment,
    Range* drop_caret_range,
    InsertMode insert_mode,
    DragSourceType drag_source_type) {
  if (!drop_target || !drop_target->isConnected())
    return true;

  // Dispatch 'beforeinput'.
  DataTransfer* data_transfer = DataTransfer::Create(
      DataTransfer::kDragAndDrop, DataTransferAccessPolicy::kReadable,
      drag_data->PlatformData());
  data_transfer->SetSourceOperation(drag_data->DraggingSourceOperationMask());
  const bool should_insert =
      DispatchBeforeInputDataTransfer(
          drop_target, InputEvent::InputType::kInsertFromDrop, data_transfer) ==
      DispatchEventResult::kNotCanceled;

  // 'beforeinput' event handler may destroy frame, return false to cancel
  // remaining actions;
  if (frame_->GetDocument()->GetFrame() != frame_)
    return false;

  if (should_insert && drop_target->isConnected())
    ReplaceSelectionAfterDragging(fragment, insert_mode, drag_source_type);

  return true;
}

EphemeralRange Editor::SelectedRange() {
  return GetFrame()
      .Selection()
      .ComputeVisibleSelectionInDOMTreeDeprecated()
      .ToNormalizedEphemeralRange();
}

void Editor::RespondToChangedContents(const Position& position) {
  if (GetFrame().GetSettings() &&
      GetFrame().GetSettings()->GetAccessibilityEnabled()) {
    Node* node = position.AnchorNode();
    if (AXObjectCache* cache =
            GetFrame().GetDocument()->ExistingAXObjectCache())
      cache->HandleEditableTextContentChanged(node);
  }

  GetSpellChecker().RespondToChangedContents();
  frame_->Client()->DidChangeContents();
}

void Editor::RegisterCommandGroup(CompositeEditCommand* command_group_wrapper) {
  DCHECK(command_group_wrapper->IsCommandGroupWrapper());
  last_edit_command_ = command_group_wrapper;
}

void Editor::ApplyParagraphStyle(CSSPropertyValueSet* style,
                                 InputEvent::InputType input_type) {
  if (GetFrame()
          .Selection()
          .ComputeVisibleSelectionInDOMTreeDeprecated()
          .IsNone() ||
      !style)
    return;
  DCHECK(GetFrame().GetDocument());
  ApplyStyleCommand::Create(*GetFrame().GetDocument(),
                            EditingStyle::Create(style), input_type,
                            ApplyStyleCommand::kForceBlockProperties)
      ->Apply();
}

void Editor::ApplyParagraphStyleToSelection(CSSPropertyValueSet* style,
                                            InputEvent::InputType input_type) {
  if (!style || style->IsEmpty() || !CanEditRichly())
    return;

  ApplyParagraphStyle(style, input_type);
}

Editor* Editor::Create(LocalFrame& frame) {
  return new Editor(frame);
}

Editor::Editor(LocalFrame& frame)
    : frame_(&frame),
      undo_stack_(UndoStack::Create()),
      prevent_reveal_selection_(0),
      should_start_new_kill_ring_sequence_(false),
      // This is off by default, since most editors want this behavior (this
      // matches IE but not FF).
      should_style_with_css_(false),
      kill_ring_(std::make_unique<KillRing>()),
      are_marked_text_matches_highlighted_(false),
      default_paragraph_separator_(EditorParagraphSeparator::kIsDiv),
      overwrite_mode_enabled_(false) {}

Editor::~Editor() = default;

void Editor::Clear() {
  should_style_with_css_ = false;
  default_paragraph_separator_ = EditorParagraphSeparator::kIsDiv;
  last_edit_command_ = nullptr;
  undo_stack_->Clear();
}

bool Editor::InsertText(const String& text, KeyboardEvent* triggering_event) {
  return GetFrame().GetEventHandler().HandleTextInputEvent(text,
                                                           triggering_event);
}

bool Editor::InsertTextWithoutSendingTextEvent(
    const String& text,
    bool select_inserted_text,
    TextEvent* triggering_event,
    InputEvent::InputType input_type) {
  const VisibleSelection& selection =
      CreateVisibleSelection(SelectionForCommand(triggering_event));
  if (!selection.IsContentEditable())
    return false;

  EditingState editing_state;
  // Insert the text
  TypingCommand::InsertText(
      *selection.Start().GetDocument(), text, selection.AsSelection(),
      select_inserted_text ? TypingCommand::kSelectInsertedText : 0,
      &editing_state,
      triggering_event && triggering_event->IsComposition()
          ? TypingCommand::kTextCompositionConfirm
          : TypingCommand::kTextCompositionNone,
      false, input_type);
  if (editing_state.IsAborted())
    return false;

  // Reveal the current selection
  if (LocalFrame* edited_frame = selection.Start().GetDocument()->GetFrame()) {
    if (Page* page = edited_frame->GetPage()) {
      LocalFrame* focused_or_main_frame =
          ToLocalFrame(page->GetFocusController().FocusedOrMainFrame());
      focused_or_main_frame->Selection().RevealSelection(
          ScrollAlignment::kAlignCenterIfNeeded);
    }
  }

  return true;
}

bool Editor::InsertLineBreak() {
  if (!CanEdit())
    return false;

  VisiblePosition caret =
      GetFrameSelection().ComputeVisibleSelectionInDOMTree().VisibleStart();
  bool align_to_edge = IsEndOfEditableOrNonEditableContent(caret);
  DCHECK(GetFrame().GetDocument());
  if (!TypingCommand::InsertLineBreak(*GetFrame().GetDocument()))
    return false;
  RevealSelectionAfterEditingOperation(
      align_to_edge ? ScrollAlignment::kAlignToEdgeIfNeeded
                    : ScrollAlignment::kAlignCenterIfNeeded);

  return true;
}

bool Editor::InsertParagraphSeparator() {
  if (!CanEdit())
    return false;

  if (!CanEditRichly())
    return InsertLineBreak();

  VisiblePosition caret =
      GetFrameSelection().ComputeVisibleSelectionInDOMTree().VisibleStart();
  bool align_to_edge = IsEndOfEditableOrNonEditableContent(caret);
  DCHECK(GetFrame().GetDocument());
  EditingState editing_state;
  if (!TypingCommand::InsertParagraphSeparator(*GetFrame().GetDocument()))
    return false;
  RevealSelectionAfterEditingOperation(
      align_to_edge ? ScrollAlignment::kAlignToEdgeIfNeeded
                    : ScrollAlignment::kAlignCenterIfNeeded);

  return true;
}

static void CountEditingEvent(ExecutionContext* execution_context,
                              const Event* event,
                              WebFeature feature_on_input,
                              WebFeature feature_on_text_area,
                              WebFeature feature_on_content_editable,
                              WebFeature feature_on_non_node) {
  EventTarget* event_target = event->target();
  Node* node = event_target->ToNode();
  if (!node) {
    UseCounter::Count(execution_context, feature_on_non_node);
    return;
  }

  if (IsHTMLInputElement(node)) {
    UseCounter::Count(execution_context, feature_on_input);
    return;
  }

  if (IsHTMLTextAreaElement(node)) {
    UseCounter::Count(execution_context, feature_on_text_area);
    return;
  }

  TextControlElement* control = EnclosingTextControl(node);
  if (IsHTMLInputElement(control)) {
    UseCounter::Count(execution_context, feature_on_input);
    return;
  }

  if (IsHTMLTextAreaElement(control)) {
    UseCounter::Count(execution_context, feature_on_text_area);
    return;
  }

  UseCounter::Count(execution_context, feature_on_content_editable);
}

void Editor::CountEvent(ExecutionContext* execution_context,
                        const Event* event) {
  if (!execution_context)
    return;

  if (event->type() == EventTypeNames::textInput) {
    CountEditingEvent(execution_context, event,
                      WebFeature::kTextInputEventOnInput,
                      WebFeature::kTextInputEventOnTextArea,
                      WebFeature::kTextInputEventOnContentEditable,
                      WebFeature::kTextInputEventOnNotNode);
    return;
  }

  if (event->type() == EventTypeNames::webkitBeforeTextInserted) {
    CountEditingEvent(execution_context, event,
                      WebFeature::kWebkitBeforeTextInsertedOnInput,
                      WebFeature::kWebkitBeforeTextInsertedOnTextArea,
                      WebFeature::kWebkitBeforeTextInsertedOnContentEditable,
                      WebFeature::kWebkitBeforeTextInsertedOnNotNode);
    return;
  }

  if (event->type() == EventTypeNames::webkitEditableContentChanged) {
    CountEditingEvent(
        execution_context, event,
        WebFeature::kWebkitEditableContentChangedOnInput,
        WebFeature::kWebkitEditableContentChangedOnTextArea,
        WebFeature::kWebkitEditableContentChangedOnContentEditable,
        WebFeature::kWebkitEditableContentChangedOnNotNode);
  }
}

void Editor::CopyImage(const HitTestResult& result) {
  WriteImageNodeToClipboard(*result.InnerNodeOrImageMapImage(),
                            result.AltDisplayString());
}

bool Editor::CanUndo() {
  return undo_stack_->CanUndo();
}

void Editor::Undo() {
  undo_stack_->Undo();
}

bool Editor::CanRedo() {
  return undo_stack_->CanRedo();
}

void Editor::Redo() {
  undo_stack_->Redo();
}

void Editor::SetBaseWritingDirection(WritingDirection direction) {
  Element* focused_element = GetFrame().GetDocument()->FocusedElement();
  if (IsTextControl(focused_element)) {
    if (direction == WritingDirection::kNatural)
      return;
    focused_element->setAttribute(
        dirAttr, direction == WritingDirection::kLeftToRight ? "ltr" : "rtl");
    focused_element->DispatchInputEvent();
    return;
  }

  MutableCSSPropertyValueSet* style =
      MutableCSSPropertyValueSet::Create(kHTMLQuirksMode);
  style->SetProperty(
      CSSPropertyDirection,
      direction == WritingDirection::kLeftToRight
          ? "ltr"
          : direction == WritingDirection::kRightToLeft ? "rtl" : "inherit",
      /* important */ false, GetFrame().GetDocument()->GetSecureContextMode());
  ApplyParagraphStyleToSelection(
      style, InputEvent::InputType::kFormatSetBlockTextDirection);
}

void Editor::RevealSelectionAfterEditingOperation(
    const ScrollAlignment& alignment) {
  if (prevent_reveal_selection_)
    return;
  if (!GetFrameSelection().IsAvailable())
    return;
  GetFrameSelection().RevealSelection(alignment, kDoNotRevealExtent);
}

void Editor::AddToKillRing(const EphemeralRange& range) {
  if (should_start_new_kill_ring_sequence_)
    GetKillRing().StartNewSequence();

  DCHECK(!GetFrame().GetDocument()->NeedsLayoutTreeUpdate());
  String text = PlainText(range);
  GetKillRing().Append(text);
  should_start_new_kill_ring_sequence_ = false;
}

EphemeralRange Editor::RangeForPoint(const IntPoint& frame_point) const {
  const PositionWithAffinity position_with_affinity =
      GetFrame().PositionForPoint(frame_point);
  if (position_with_affinity.IsNull())
    return EphemeralRange();

  const VisiblePosition position =
      CreateVisiblePosition(position_with_affinity);
  const VisiblePosition previous = PreviousPositionOf(position);
  if (previous.IsNotNull()) {
    const EphemeralRange previous_character_range =
        MakeRange(previous, position);
    const IntRect rect = FirstRectForRange(previous_character_range);
    if (rect.Contains(frame_point))
      return EphemeralRange(previous_character_range);
  }

  const VisiblePosition next = NextPositionOf(position);
  const EphemeralRange next_character_range = MakeRange(position, next);
  if (next_character_range.IsNotNull()) {
    const IntRect rect = FirstRectForRange(next_character_range);
    if (rect.Contains(frame_point))
      return EphemeralRange(next_character_range);
  }

  return EphemeralRange();
}

void Editor::ComputeAndSetTypingStyle(CSSPropertyValueSet* style,
                                      InputEvent::InputType input_type) {
  if (!style || style->IsEmpty()) {
    ClearTypingStyle();
    return;
  }

  // Calculate the current typing style.
  if (typing_style_)
    typing_style_->OverrideWithStyle(style);
  else
    typing_style_ = EditingStyle::Create(style);

  typing_style_->PrepareToApplyAt(
      GetFrame()
          .Selection()
          .ComputeVisibleSelectionInDOMTreeDeprecated()
          .VisibleStart()
          .DeepEquivalent(),
      EditingStyle::kPreserveWritingDirection);

  // Handle block styles, substracting these from the typing style.
  EditingStyle* block_style = typing_style_->ExtractAndRemoveBlockProperties();
  if (!block_style->IsEmpty()) {
    DCHECK(GetFrame().GetDocument());
    ApplyStyleCommand::Create(*GetFrame().GetDocument(), block_style,
                              input_type)
        ->Apply();
  }
}

bool Editor::FindString(LocalFrame& frame,
                        const String& target,
                        FindOptions options) {
  VisibleSelection selection =
      frame.Selection().ComputeVisibleSelectionInDOMTreeDeprecated();

  // TODO(yosin) We should make |findRangeOfString()| to return
  // |EphemeralRange| rather than|Range| object.
  Range* const result_range =
      FindRangeOfString(*frame.GetDocument(), target,
                        EphemeralRange(selection.Start(), selection.End()),
                        static_cast<FindOptions>(options | kFindAPICall));

  if (!result_range)
    return false;

  frame.Selection().SetSelectionAndEndTyping(
      SelectionInDOMTree::Builder()
          .SetBaseAndExtent(EphemeralRange(result_range))
          .Build());
  frame.Selection().RevealSelection();
  return true;
}

// TODO(yosin) We should return |EphemeralRange| rather than |Range|. We use
// |Range| object for checking whether start and end position crossing shadow
// boundaries, however we can do it without |Range| object.
template <typename Strategy>
static Range* FindStringBetweenPositions(
    const String& target,
    const EphemeralRangeTemplate<Strategy>& reference_range,
    FindOptions options) {
  EphemeralRangeTemplate<Strategy> search_range(reference_range);

  bool forward = !(options & kBackwards);

  while (true) {
    EphemeralRangeTemplate<Strategy> result_range =
        FindPlainText(search_range, target, options);
    if (result_range.IsCollapsed())
      return nullptr;

    Range* range_object =
        Range::Create(result_range.GetDocument(),
                      ToPositionInDOMTree(result_range.StartPosition()),
                      ToPositionInDOMTree(result_range.EndPosition()));
    if (!range_object->collapsed())
      return range_object;

    // Found text spans over multiple TreeScopes. Since it's impossible to
    // return such section as a Range, we skip this match and seek for the
    // next occurrence.
    // TODO(yosin) Handle this case.
    if (forward) {
      search_range = EphemeralRangeTemplate<Strategy>(
          NextPositionOf(result_range.StartPosition(),
                         PositionMoveType::kGraphemeCluster),
          search_range.EndPosition());
    } else {
      search_range = EphemeralRangeTemplate<Strategy>(
          search_range.StartPosition(),
          PreviousPositionOf(result_range.EndPosition(),
                             PositionMoveType::kGraphemeCluster));
    }
  }

  NOTREACHED();
  return nullptr;
}

template <typename Strategy>
static Range* FindRangeOfStringAlgorithm(
    Document& document,
    const String& target,
    const EphemeralRangeTemplate<Strategy>& reference_range,
    FindOptions options) {
  if (target.IsEmpty())
    return nullptr;

  // Start from an edge of the reference range. Which edge is used depends on
  // whether we're searching forward or backward, and whether startInSelection
  // is set.
  EphemeralRangeTemplate<Strategy> document_range =
      EphemeralRangeTemplate<Strategy>::RangeOfContents(document);
  EphemeralRangeTemplate<Strategy> search_range(document_range);

  bool forward = !(options & kBackwards);
  bool start_in_reference_range = false;
  if (reference_range.IsNotNull()) {
    start_in_reference_range = options & kStartInSelection;
    if (forward && start_in_reference_range)
      search_range = EphemeralRangeTemplate<Strategy>(
          reference_range.StartPosition(), document_range.EndPosition());
    else if (forward)
      search_range = EphemeralRangeTemplate<Strategy>(
          reference_range.EndPosition(), document_range.EndPosition());
    else if (start_in_reference_range)
      search_range = EphemeralRangeTemplate<Strategy>(
          document_range.StartPosition(), reference_range.EndPosition());
    else
      search_range = EphemeralRangeTemplate<Strategy>(
          document_range.StartPosition(), reference_range.StartPosition());
  }

  Range* result_range =
      FindStringBetweenPositions(target, search_range, options);

  // If we started in the reference range and the found range exactly matches
  // the reference range, find again. Build a selection with the found range
  // to remove collapsed whitespace. Compare ranges instead of selection
  // objects to ignore the way that the current selection was made.
  if (result_range && start_in_reference_range &&
      NormalizeRange(EphemeralRangeTemplate<Strategy>(result_range)) ==
          reference_range) {
    if (forward)
      search_range = EphemeralRangeTemplate<Strategy>(
          FromPositionInDOMTree<Strategy>(result_range->EndPosition()),
          search_range.EndPosition());
    else
      search_range = EphemeralRangeTemplate<Strategy>(
          search_range.StartPosition(),
          FromPositionInDOMTree<Strategy>(result_range->StartPosition()));
    result_range = FindStringBetweenPositions(target, search_range, options);
  }

  if (!result_range && options & kWrapAround)
    return FindStringBetweenPositions(target, document_range, options);

  return result_range;
}

Range* Editor::FindRangeOfString(Document& document,
                                 const String& target,
                                 const EphemeralRange& reference,
                                 FindOptions options) {
  return FindRangeOfStringAlgorithm<EditingStrategy>(document, target,
                                                     reference, options);
}

Range* Editor::FindRangeOfString(Document& document,
                                 const String& target,
                                 const EphemeralRangeInFlatTree& reference,
                                 FindOptions options) {
  return FindRangeOfStringAlgorithm<EditingInFlatTreeStrategy>(
      document, target, reference, options);
}

void Editor::SetMarkedTextMatchesAreHighlighted(bool flag) {
  if (flag == are_marked_text_matches_highlighted_)
    return;

  are_marked_text_matches_highlighted_ = flag;
  GetFrame().GetDocument()->Markers().RepaintMarkers(
      DocumentMarker::MarkerTypes::TextMatch());
}

void Editor::RespondToChangedSelection() {
  GetSpellChecker().RespondToChangedSelection();
  frame_->Client()->DidChangeSelection(
      GetFrameSelection().GetSelectionInDOMTree().Type() != kRangeSelection);
  SetStartNewKillRingSequence(true);
}

SpellChecker& Editor::GetSpellChecker() const {
  return GetFrame().GetSpellChecker();
}

FrameSelection& Editor::GetFrameSelection() const {
  return GetFrame().Selection();
}

void Editor::SetMark() {
  mark_ = GetFrameSelection().ComputeVisibleSelectionInDOMTree();
  mark_is_directional_ = GetFrameSelection().IsDirectional();
}

void Editor::ToggleOverwriteModeEnabled() {
  overwrite_mode_enabled_ = !overwrite_mode_enabled_;
  GetFrameSelection().SetShouldShowBlockCursor(overwrite_mode_enabled_);
}

void Editor::ReplaceSelection(const String& text) {
  DCHECK(!GetFrame().GetDocument()->NeedsLayoutTreeUpdate());
  bool select_replacement = Behavior().ShouldSelectReplacement();
  bool smart_replace = true;
  ReplaceSelectionWithText(text, select_replacement, smart_replace,
                           InputEvent::InputType::kInsertReplacementText);
}

void Editor::Trace(blink::Visitor* visitor) {
  visitor->Trace(frame_);
  visitor->Trace(last_edit_command_);
  visitor->Trace(undo_stack_);
  visitor->Trace(mark_);
  visitor->Trace(typing_style_);
}

}  // namespace blink
