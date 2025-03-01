// Copyright (c) 2010 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "pdf/pdf.h"

#include <stdint.h>

#include <utility>

#include "base/logging.h"
#include "base/macros.h"
#include "pdf/out_of_process_instance.h"
#include "pdf/pdf_ppapi.h"

namespace chrome_pdf {

namespace {

class ScopedSdkInitializer {
 public:
  ScopedSdkInitializer() {}
  ~ScopedSdkInitializer() {
#if DCHECK_IS_ON()
    DCHECK(initialized_);
#endif
    if (!IsSDKInitializedViaPepper())
      ShutdownSDK();
  }

  // Must be called.
  bool Init() {
#if DCHECK_IS_ON()
    initialized_ = true;
#endif
    return IsSDKInitializedViaPepper() || InitializeSDK();
  }

 private:
#if DCHECK_IS_ON()
  bool initialized_ = false;
#endif

  DISALLOW_COPY_AND_ASSIGN(ScopedSdkInitializer);
};

}  // namespace

#if defined(OS_WIN)
bool RenderPDFPageToDC(base::span<const uint8_t> pdf_buffer,
                       int page_number,
                       HDC dc,
                       int dpi_x,
                       int dpi_y,
                       int bounds_origin_x,
                       int bounds_origin_y,
                       int bounds_width,
                       int bounds_height,
                       bool fit_to_bounds,
                       bool stretch_to_bounds,
                       bool keep_aspect_ratio,
                       bool center_in_bounds,
                       bool autorotate,
                       bool use_color) {
  ScopedSdkInitializer scoped_sdk_initializer;
  if (!scoped_sdk_initializer.Init())
    return false;

  PDFEngineExports* engine_exports = PDFEngineExports::Get();
  PDFEngineExports::RenderingSettings settings(
      dpi_x, dpi_y,
      pp::Rect(bounds_origin_x, bounds_origin_y, bounds_width, bounds_height),
      fit_to_bounds, stretch_to_bounds, keep_aspect_ratio, center_in_bounds,
      autorotate, use_color);
  return engine_exports->RenderPDFPageToDC(pdf_buffer, page_number, settings,
                                           dc);
}

void SetPDFEnsureTypefaceCharactersAccessible(
    PDFEnsureTypefaceCharactersAccessible func) {
  PDFEngineExports::Get()->SetPDFEnsureTypefaceCharactersAccessible(func);
}

void SetPDFUseGDIPrinting(bool enable) {
  PDFEngineExports::Get()->SetPDFUseGDIPrinting(enable);
}

void SetPDFUsePrintMode(int mode) {
  PDFEngineExports::Get()->SetPDFUsePrintMode(mode);
}
#endif  // defined(OS_WIN)

bool GetPDFDocInfo(base::span<const uint8_t> pdf_buffer,
                   int* page_count,
                   double* max_page_width) {
  ScopedSdkInitializer scoped_sdk_initializer;
  if (!scoped_sdk_initializer.Init())
    return false;

  PDFEngineExports* engine_exports = PDFEngineExports::Get();
  return engine_exports->GetPDFDocInfo(pdf_buffer, page_count, max_page_width);
}

bool GetPDFPageSizeByIndex(base::span<const uint8_t> pdf_buffer,
                           int page_number,
                           double* width,
                           double* height) {
  ScopedSdkInitializer scoped_sdk_initializer;
  if (!scoped_sdk_initializer.Init())
    return false;

  chrome_pdf::PDFEngineExports* engine_exports =
      chrome_pdf::PDFEngineExports::Get();
  return engine_exports->GetPDFPageSizeByIndex(pdf_buffer, page_number, width,
                                               height);
}

bool RenderPDFPageToBitmap(base::span<const uint8_t> pdf_buffer,
                           int page_number,
                           void* bitmap_buffer,
                           int bitmap_width,
                           int bitmap_height,
                           int dpi_x,
                           int dpi_y,
                           bool autorotate,
                           bool use_color) {
  ScopedSdkInitializer scoped_sdk_initializer;
  if (!scoped_sdk_initializer.Init())
    return false;

  PDFEngineExports* engine_exports = PDFEngineExports::Get();
  PDFEngineExports::RenderingSettings settings(
      dpi_x, dpi_y, pp::Rect(bitmap_width, bitmap_height), true, false, true,
      true, autorotate, use_color);
  return engine_exports->RenderPDFPageToBitmap(pdf_buffer, page_number,
                                               settings, bitmap_buffer);
}

bool ConvertPdfPagesToNupPdf(
    std::vector<base::span<const uint8_t>> input_buffers,
    size_t pages_per_sheet,
    size_t page_size_width,
    size_t page_size_height,
    void** dest_pdf_buffer,
    size_t* dest_pdf_buffer_size) {
  ScopedSdkInitializer scoped_sdk_initializer;
  if (!scoped_sdk_initializer.Init())
    return false;

  PDFEngineExports* engine_exports = PDFEngineExports::Get();
  return engine_exports->ConvertPdfPagesToNupPdf(
      std::move(input_buffers), pages_per_sheet, page_size_width,
      page_size_height, dest_pdf_buffer, dest_pdf_buffer_size);
}

bool ConvertPdfDocumentToNupPdf(base::span<const uint8_t> input_buffer,
                                size_t pages_per_sheet,
                                size_t page_size_width,
                                size_t page_size_height,
                                void** dest_pdf_buffer,
                                size_t* dest_pdf_buffer_size) {
  ScopedSdkInitializer scoped_sdk_initializer;
  if (!scoped_sdk_initializer.Init())
    return false;

  PDFEngineExports* engine_exports = PDFEngineExports::Get();
  return engine_exports->ConvertPdfDocumentToNupPdf(
      input_buffer, pages_per_sheet, page_size_width, page_size_height,
      dest_pdf_buffer, dest_pdf_buffer_size);
}

}  // namespace chrome_pdf
