// Copyright 2017 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/ml/mps_protocols_impl.h"

#include <vector>

#include "base/logging.h"
#include "services/ml/mpscnn_context.h"

@implementation ConvDataSource

@synthesize weights_;

@synthesize bias_;

@synthesize desc_;

- (id)initWithWeight:(float*)weights
                bias:(float*)bias
                desc:(MPSCNNConvolutionDescriptor*)desc {
  self = [super init];
  self.weights_ = weights;
  self.bias_ = bias;
  self.desc_ = desc;
  return self;
}

- (float*)biasTerms {
  return self.bias_;
}

- (MPSDataType)dataType {
  return MPSDataTypeFloat32;
}

- (MPSCNNConvolutionDescriptor*)descriptor {
  return self.desc_;
}

- (NSString*)label {
  return nullptr;
}

- (BOOL)load {
  return true;
}

- (float*)lookupTableForUInt8Kernel {
  return nullptr;
}

- (void)purge {
  return;
}

- (vector_float2*)rangesForUInt8Kernel {
  return nullptr;
}

- (void*)weights {
  return self.weights_;
}

- (id)copyWithZone:(struct _NSZone*)zone {
  ConvDataSource* source = [[ConvDataSource allocWithZone:zone] init];
  source.weights_ = self.weights_;
  source.bias_ = self.bias_;
  source.desc_ = self.desc_;
  return source;
}

@end

@implementation CustomPadding

@synthesize offset = _offset;

@synthesize clipRect = _clipRect;

@synthesize edge_mode = _edge_mode;

@synthesize num = _num;

@synthesize width = _width;

@synthesize height = _height;

@synthesize channels = _channels;

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (id)initWithCoder:(NSCoder*)coder {
  self = [super init];
  return self;
}

- (void)encodeWithCoder:(NSCoder*)aCoder {
}

- (id)initWithOffset:(MPSOffset)offset
            edgeMode:(MPSImageEdgeMode)edgeMode
                 num:(uint32_t)num
               width:(uint32_t)width
              height:(uint32_t)height
            channels:(uint32_t)channels {
  self = [super init];
  self.offset = offset;
  self.edge_mode = edgeMode;
  self.num = num;
  self.width = width;
  self.height = height;
  self.channels = channels;
  return self;
}

- (id)initWithClipRect:(MTLRegion)clipRect
                offset:(MPSOffset)offset
              edgeMode:(MPSImageEdgeMode)edgeMode
                   num:(uint32_t)num
                 width:(uint32_t)width
                height:(uint32_t)height
              channels:(uint32_t)channels {
  self = [super init];
  self.clipRect = clipRect;
  self.offset = offset;
  self.edge_mode = edgeMode;
  self.num = num;
  self.width = width;
  self.height = height;
  self.channels = channels;
  return self;
}

- (MPSNNPaddingMethod)paddingMethod {
  return MPSNNPaddingMethodCustom;
}

- (MPSImageDescriptor*)
destinationImageDescriptorForSourceImages:(NSArray<MPSImage*>*)sourceImages
                             sourceStates:(NSArray<MPSState*>*)sourceStates
                                forKernel:(MPSKernel*)kernel
                      suggestedDescriptor:(MPSImageDescriptor*)inDescriptor {
  if ([kernel isKindOfClass:[MPSCNNKernel class]]) {
    MPSCNNKernel* cnn_kernel = (MPSCNNKernel*)kernel;
    [cnn_kernel setOffset:_offset];
    [cnn_kernel setEdgeMode:_edge_mode];
    if (_clipRect.size.height > 0 && _clipRect.size.width > 0) {
      [cnn_kernel setClipRect:_clipRect];
    }
  }

  [inDescriptor setChannelFormat:MPSImageFeatureChannelFormatFloat16];
  [inDescriptor setNumberOfImages:_num];
  [inDescriptor setWidth:_width];
  [inDescriptor setHeight:_height];
  [inDescriptor setFeatureChannels:_channels];
  [inDescriptor
      setUsage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

  return inDescriptor;
}

@end

@implementation OutputImageAllocator

@synthesize image = _image;

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (id)initWithCoder:(NSCoder*)coder {
  self = [super init];
  return self;
}

- (void)encodeWithCoder:(NSCoder*)aCoder {
}

- (MPSImage*)imageForCommandBuffer:(id<MTLCommandBuffer>)cmdBuf
                   imageDescriptor:(MPSImageDescriptor*)descriptor
                            kernel:(MPSKernel*)kernel {
  if (self.image)
    return self.image;

  self.image = [[MPSImage alloc] initWithDevice:ml::GetMPSCNNContext().device
                                imageDescriptor:descriptor];

  return self.image;
}

@end

@implementation TemporaryImageHandle

@synthesize label_ = _label;

+ (BOOL)supportsSecureCoding {
  return YES;
}

- (id)initWithCoder:(NSCoder*)coder {
  self = [super init];
  return self;
}

- (void)encodeWithCoder:(NSCoder*)aCoder {
}

- (id)initWithLabel:(NSString*)label {
  self = [super init];
  self.label_ = label;
  return self;
}

/*! @abstract   A label to be attached to associated MTLResources for this node
 *  @return     A human readable string for debugging purposes
 */
- (NSString*)label {
  return self.label_;
}

@end

namespace ml {

bool GetMPSImageInfo(const OperandMac& operand,
                     uint32_t& n,
                     uint32_t& width,
                     uint32_t& height,
                     uint32_t& channels) {
  const std::vector<uint32_t>& dimensions = operand.dimensions;
  if (dimensions.size() == 4) {
    n = dimensions[0];
    height = dimensions[1];
    width = dimensions[2];
    channels = dimensions[3];
    return true;
  } else if (dimensions.size() == 3) {
    n = 1;
    height = dimensions[0];
    width = dimensions[1];
    channels = dimensions[2];
    return true;
  } else if (dimensions.size() == 2) {
    n = 1;
    height = 1;
    width = dimensions[0];
    channels = dimensions[1];
    return true;
  } else if (dimensions.size() == 1) {
    n = 1;
    height = 1;
    width = 1;
    channels = dimensions[0];
    return true;
  } else {
    DLOG(ERROR) << "dimension " << dimensions.size() << " is not supported";
    return false;
  }
}
}
