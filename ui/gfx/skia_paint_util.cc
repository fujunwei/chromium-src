// Copyright (c) 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "ui/gfx/skia_paint_util.h"

#include "cc/paint/paint_image_builder.h"
#include "third_party/skia/include/core/SkColorFilter.h"
#include "third_party/skia/include/core/SkMaskFilter.h"
#include "third_party/skia/include/effects/SkGradientShader.h"
#include "third_party/skia/include/effects/SkLayerDrawLooper.h"
#include "ui/gfx/image/image_skia_rep.h"

namespace gfx {

sk_sp<cc::PaintShader> CreateImageRepShader(const gfx::ImageSkiaRep& image_rep,
                                            SkShader::TileMode tile_mode,
                                            const SkMatrix& local_matrix) {
  return CreateImageRepShaderForScale(image_rep, tile_mode, local_matrix,
                                      image_rep.scale());
}

sk_sp<cc::PaintShader> CreateImageRepShaderForScale(
    const gfx::ImageSkiaRep& image_rep,
    SkShader::TileMode tile_mode,
    const SkMatrix& local_matrix,
    SkScalar scale) {
  // Unscale matrix by |scale| such that the bitmap is drawn at the
  // correct density.
  // Convert skew and translation to pixel coordinates.
  // Thus, for |bitmap_scale| = 2:
  //   x scale = 2, x translation = 1 DIP,
  // should be converted to
  //   x scale = 1, x translation = 2 pixels.
  SkMatrix shader_scale = local_matrix;
  shader_scale.preScale(scale, scale);
  shader_scale.setScaleX(local_matrix.getScaleX() / scale);
  shader_scale.setScaleY(local_matrix.getScaleY() / scale);

  return cc::PaintShader::MakeImage(image_rep.paint_image(), tile_mode,
                                    tile_mode, &shader_scale);
}

sk_sp<cc::PaintShader> CreateGradientShader(int start_point,
                                            int end_point,
                                            SkColor start_color,
                                            SkColor end_color) {
  SkColor grad_colors[2] = {start_color, end_color};
  SkPoint grad_points[2];
  grad_points[0].iset(0, start_point);
  grad_points[1].iset(0, end_point);

  return cc::PaintShader::MakeLinearGradient(grad_points, grad_colors, nullptr,
                                             2, SkShader::kClamp_TileMode);
}

// This is copied from
// third_party/WebKit/Source/platform/graphics/skia/SkiaUtils.h
static SkScalar RadiusToSigma(double radius) {
  return radius > 0 ? SkDoubleToScalar(0.288675f * radius + 0.5f) : 0;
}

sk_sp<SkDrawLooper> CreateShadowDrawLooper(
    const std::vector<ShadowValue>& shadows) {
  if (shadows.empty())
    return nullptr;

  SkLayerDrawLooper::Builder looper_builder;

  looper_builder.addLayer();  // top layer of the original.

  SkLayerDrawLooper::LayerInfo layer_info;
  layer_info.fPaintBits |= SkLayerDrawLooper::kMaskFilter_Bit;
  layer_info.fPaintBits |= SkLayerDrawLooper::kColorFilter_Bit;
  layer_info.fColorMode = SkBlendMode::kSrc;

  for (size_t i = 0; i < shadows.size(); ++i) {
    const ShadowValue& shadow = shadows[i];

    layer_info.fOffset.set(SkIntToScalar(shadow.x()),
                           SkIntToScalar(shadow.y()));

    SkPaint* paint = looper_builder.addLayer(layer_info);
    // Skia's blur radius defines the range to extend the blur from
    // original mask, which is half of blur amount as defined in ShadowValue.
    paint->setMaskFilter(SkMaskFilter::MakeBlur(
        kNormal_SkBlurStyle, RadiusToSigma(shadow.blur() / 2)));
    paint->setColorFilter(
        SkColorFilter::MakeModeFilter(shadow.color(), SkBlendMode::kSrcIn));
  }

  return looper_builder.detach();
}

}  // namespace gfx
