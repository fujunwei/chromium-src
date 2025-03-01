/*
 * Copyright (C) 1999 Lars Knoll (knoll@kde.org)
 *           (C) 2000 Simon Hausmann <hausmann@kde.org>
 *           (C) 2000 Stefan Schimanski (1Stein@gmx.de)
 * Copyright (C) 2004, 2005, 2006, 2008, 2009, 2010 Apple Inc.
 *               All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public License
 * along with this library; see the file COPYING.LIB.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 *
 */

#include "third_party/blink/renderer/core/layout/layout_embedded_object.h"

#include "third_party/blink/renderer/core/css_value_keywords.h"
#include "third_party/blink/renderer/core/exported/web_plugin_container_impl.h"
#include "third_party/blink/renderer/core/frame/local_frame.h"
#include "third_party/blink/renderer/core/frame/local_frame_view.h"
#include "third_party/blink/renderer/core/html/html_plugin_element.h"
#include "third_party/blink/renderer/core/html_names.h"
#include "third_party/blink/renderer/core/layout/intrinsic_sizing_info.h"
#include "third_party/blink/renderer/core/layout/layout_analyzer.h"
#include "third_party/blink/renderer/core/layout/layout_view.h"
#include "third_party/blink/renderer/core/page/page.h"
#include "third_party/blink/renderer/core/paint/embedded_object_paint_invalidator.h"
#include "third_party/blink/renderer/core/paint/embedded_object_painter.h"
#include "third_party/blink/renderer/platform/text/platform_locale.h"

namespace blink {

using namespace HTMLNames;

LayoutEmbeddedObject::LayoutEmbeddedObject(Element* element)
    : LayoutEmbeddedContent(element) {
  View()->GetFrameView()->SetIsVisuallyNonEmpty();
}

LayoutEmbeddedObject::~LayoutEmbeddedObject() = default;

static String LocalizedUnavailablePluginReplacementText(
    Node* node,
    LayoutEmbeddedObject::PluginAvailability availability) {
  Locale& locale =
      node ? ToElement(node)->GetLocale() : Locale::DefaultLocale();
  switch (availability) {
    case LayoutEmbeddedObject::kPluginAvailable:
      break;
    case LayoutEmbeddedObject::kPluginMissing:
      return locale.QueryString(WebLocalizedString::kMissingPluginText);
    case LayoutEmbeddedObject::kPluginBlockedByContentSecurityPolicy:
      return locale.QueryString(WebLocalizedString::kBlockedPluginText);
  }
  NOTREACHED();
  return String();
}

void LayoutEmbeddedObject::SetPluginAvailability(
    PluginAvailability availability) {
  DCHECK_EQ(kPluginAvailable, plugin_availability_);
  plugin_availability_ = availability;

  unavailable_plugin_replacement_text_ =
      LocalizedUnavailablePluginReplacementText(GetNode(), availability);

  // node() is nullptr when LayoutEmbeddedContent is being destroyed.
  if (GetNode())
    SetShouldDoFullPaintInvalidation();
}

bool LayoutEmbeddedObject::ShowsUnavailablePluginIndicator() const {
  return plugin_availability_ != kPluginAvailable;
}

void LayoutEmbeddedObject::PaintReplaced(
    const PaintInfo& paint_info,
    const LayoutPoint& paint_offset) const {
  EmbeddedObjectPainter(*this).PaintReplaced(paint_info, paint_offset);
}

PaintInvalidationReason LayoutEmbeddedObject::InvalidatePaint(
    const PaintInvalidatorContext& context) const {
  return EmbeddedObjectPaintInvalidator(*this, context).InvalidatePaint();
}

void LayoutEmbeddedObject::UpdateLayout() {
  DCHECK(NeedsLayout());
  LayoutAnalyzer::Scope analyzer(*this);

  UpdateLogicalWidth();
  UpdateLogicalHeight();

  overflow_.reset();
  AddVisualEffectOverflow();

  UpdateAfterLayout();

  if (!GetEmbeddedContentView() && GetFrameView())
    GetFrameView()->AddPartToUpdate(*this);

  ClearNeedsLayout();
}

CompositingReasons LayoutEmbeddedObject::AdditionalCompositingReasons() const {
  if (RequiresAcceleratedCompositing())
    return CompositingReason::kPlugin;
  return CompositingReason::kNone;
}

void LayoutEmbeddedObject::ComputeIntrinsicSizingInfo(
    IntrinsicSizingInfo& intrinsic_sizing_info) const {
  FrameView* frame_view = ChildFrameView();
  if (frame_view && frame_view->GetIntrinsicSizingInfo(intrinsic_sizing_info)) {
    // Handle zoom & vertical writing modes here, as the embedded document
    // doesn't know about them.
    intrinsic_sizing_info.size.Scale(Style()->EffectiveZoom());

    if (!IsHorizontalWritingMode())
      intrinsic_sizing_info.Transpose();
    return;
  }

  LayoutEmbeddedContent::ComputeIntrinsicSizingInfo(intrinsic_sizing_info);
}

bool LayoutEmbeddedObject::NeedsPreferredWidthsRecalculation() const {
  if (LayoutEmbeddedContent::NeedsPreferredWidthsRecalculation())
    return true;
  FrameView* frame_view = ChildFrameView();
  return frame_view && frame_view->HasIntrinsicSizingInfo();
}

}  // namespace blink
