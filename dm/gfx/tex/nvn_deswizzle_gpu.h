#pragma once
#include <stddef.h>
#include <stdint.h>
#include <Availability.h>
#import <Metal/Metal.h>
#include "nvn_deswizzle.h"

// The Metal 4 encoder/API declarations exist only in the macOS 26 / iOS 26 SDKs; a classic-Metal build (DM_NVN_CLASSIC, macOS 15 / iOS 18 floor) compiles without them
#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000) || \
    (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 260000)
#define DM_MTL4_SDK 1
#else
#define DM_MTL4_SDK 0
#endif

int nvn_deswz_gpu_init(id<MTLDevice> dev);
int nvn_deswz_gpu_encode(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> src, uint64_t src_off,
                         const nvn_texdesc *d, unsigned level, unsigned layer,
                         id<MTLBuffer> dst, uint64_t dst_off, size_t dst_pitch);
#if DM_MTL4_SDK
int nvn_deswz_gpu_encode_m4(id<MTL4ComputeCommandEncoder> enc, id<MTL4ArgumentTable> table,
                            id<MTLBuffer> params, size_t params_off, id<MTLBuffer> src, uint64_t src_off,
                            const nvn_texdesc *d, unsigned level, unsigned layer,
                            id<MTLBuffer> dst, uint64_t dst_off, size_t dst_pitch);
#endif
