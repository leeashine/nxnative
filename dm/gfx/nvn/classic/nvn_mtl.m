// Classic Metal (macOS 15 / iOS 18) execution layer of the NVN null device, implementing the same
// replay contract as ../nvn_mtl.m (dm/gfx/nvn/nvn_obj.h) on the classic Metal API: ordinary command
// queue / command buffers / render, compute and blit encoders, per-slot buffer bindings, automatic
// hazard tracking. Built only with DM_NVN_M3=1 (dm/host/Makefile); the Metal 4 layer stays the default.
//
// Provenance: derived from d2r-mobile's classic-Metal NVN backend (dm/gfx/nvn/nvn_mtl.m,
// commit 9393b00ea9aca52d9254214da93738f26582562f, MIT) as integrated by the Octopath Traveler II
// fork (ot2-mobile, dm/host/m3_routea_nvn_mtl.m), and adapted to the current upstream replay contract
// (object layouts, command records, reflection kinds, shader chain). The integer-fragment-entry,
// slice-GS-folding and NVN depth-mode entry handling follow upstream ports
// (nxnative/dm/gfx/nvn/nvn_mtl.m@0f99c353d3abc334c39855d691afdfab40f4639f).
// Derived from d2r-mobile (MIT, (c) 2026 jhaizhou); see README.md in this directory.
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <QuartzCore/CAMetalDisplayLink.h>
#import <ImageIO/ImageIO.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#include <dlfcn.h>
#include <TargetConditionals.h>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#include <os/proc.h>
#else
#import <AppKit/AppKit.h>
#endif
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "../nvn_mtl.h"
#include "../nvn_obj.h"
#include "../nvn_timestamp.h"
// Compile-time switch DM_NVN_M3=1: the classic API path (see the file header). This file is only
// ever compiled in classic mode (dm/host/Makefile swaps it in under DM_NVN_M3=1), so it defaults to 1.
#ifndef DM_NVN_M3
#define DM_NVN_M3 1
#endif
#if DM_NVN_M3
#define RENC id<MTLRenderCommandEncoder>
#define CENC id<MTLComputeCommandEncoder>
#define BENC id<MTLBlitCommandEncoder>
#define CBUF id<MTLCommandBuffer>
#define RPD MTLRenderPassDescriptor
#define FB_ON(o, cb) cb
#define M4_ONLY(...)
#define M3_PRESENT(cb, dr) [cb presentDrawable:dr]
#define FB_SEL addCompletedHandler
#define FB_ARG id<MTLCommandBuffer>
#else
#define RENC id<MTL4RenderCommandEncoder>
#define CENC id<MTL4ComputeCommandEncoder>
#define BENC id<MTL4ComputeCommandEncoder>
#define CBUF id<MTL4CommandBuffer>
#define RPD MTL4RenderPassDescriptor
#define FB_ON(o, cb) o
#define M4_ONLY(...) __VA_ARGS__
#define M3_PRESENT(cb, dr) [dr present]
#define FB_SEL addFeedbackHandler
#define FB_ARG id<MTL4CommitFeedback>
#endif
#include "../nvn_pace.h"
#include "nvn_copy_pitch.h"
#include "../dmsh_chain.h"
#include "../../tex/nvn_fmt_mtl.h"
#include "../../tex/nvn_deswizzle.h"
#include "../../tex/nvn_deswizzle_gpu.h"
#include "../nvn_dirty.h"
#include "../nvn_refl.h"
#include "../nvn_cover.h"
#include "../nvn_hmap.h"
#include "../../../host/dm.h"
#import <CommonCrypto/CommonDigest.h>
#include <stdarg.h>
#include <mach-o/dyld.h>
#include <libgen.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdatomic.h>
#include <errno.h>
#include <execinfo.h>
#include <os/lock.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/arm/thread_status.h>
#include <math.h>
// Function parameters in this file are never retained again in the callee (the caller / global tables already hold them, and lifetime is guaranteed by defer_obj_release): saves a retain / release pair per call on the hot path
#pragma clang attribute push(__attribute__((objc_externally_retained)), apply_to = function)

#define MLOG(...) do { fprintf(stderr, "[dm] [nvn-mtl] " __VA_ARGS__); fputc('\n', stderr); } while (0)
#define MDIE(...) do { MLOG("FATAL " __VA_ARGS__); abort(); } while (0)
static pthread_once_t g_nvn_sync_trace_once = PTHREAD_ONCE_INIT;
static int g_nvn_sync_trace;
static void nvn_sync_trace_init(void) {
    const char *e = getenv("DM_NVN_SYNC_TRACE");
    g_nvn_sync_trace = e && atoi(e) != 0;
}
static int nvn_sync_trace_on(void) {
    pthread_once(&g_nvn_sync_trace_once, nvn_sync_trace_init);
    return g_nvn_sync_trace;
}
static void nvn_sync_trace(const char *event, const char *api,
                           uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {
    if (!nvn_sync_trace_on()) return;
    uint64_t tid = 0; char name[64] = {0};
    pthread_threadid_np(NULL, &tid); pthread_getname_np(pthread_self(), name, sizeof name);
    MLOG("[DEBUG-RA-SYNC] event=%s api=%s tid=%llu thread=%s t=%llu a=%llu b=%llu c=%llu d=%llu e=%llu",
         event, api, (unsigned long long)tid, name, (unsigned long long)clock_gettime_nsec_np(CLOCK_UPTIME_RAW),
         (unsigned long long)a, (unsigned long long)b, (unsigned long long)c, (unsigned long long)d, (unsigned long long)e);
}
static pthread_once_t g_nvn_draw_trace_once = PTHREAD_ONCE_INIT;
static int g_nvn_draw_trace;
static int g_nvn_bind_trace;
static uint64_t g_nvn_draw_trace_target_serial;
static void nvn_draw_trace_init(void) {
    const char *e = getenv("DM_NVN_DRAWTRACE");
    const char *b = getenv("DM_NVN_BINDTRACE");
    const char *target = getenv("DM_NVN_DRAWTRACE_TARGET");
    g_nvn_draw_trace = e && atoi(e) != 0;
    g_nvn_bind_trace = b && atoi(b) != 0;
    if (target && *target) {
        char *end = NULL;
        unsigned long long serial = strtoull(target, &end, 10);
        if (end != target && !*end && serial) {
            g_nvn_draw_trace_target_serial = (uint64_t)serial;
            g_nvn_draw_trace = 1;
            MLOG("[DEBUG-RA-DRAW] target-only C0 serial=%llu", serial);
        } else MLOG("[RED] invalid DM_NVN_DRAWTRACE_TARGET: %s", target);
    }
}
static int nvn_draw_trace_on(void) {
    pthread_once(&g_nvn_draw_trace_once, nvn_draw_trace_init);
    return g_nvn_draw_trace;
}
static int nvn_draw_trace_detail_on(void) {
    return nvn_draw_trace_on() && !g_nvn_draw_trace_target_serial;
}
static int nvn_draw_trace_target_draw_on(void);
static int nvn_bind_trace_on(void) {
    (void)nvn_draw_trace_on();
    return g_nvn_bind_trace;
}
static void warm_start(void);
static void warm_claim(void);

static id<MTLDevice> g_dev;
static NSMutableArray *g_l3_gpu_leaks;
static int g_nvn_deswz_gpu;
static id<MTLCommandQueue> g_q;
#if !DM_NVN_M3
static id<MTL4CommandQueue> g_q4;
static dispatch_queue_t g_feedback_q;
#endif
// The classic replay owns one command queue; all submissions and presents are ordered on it. A second queue would require cross-queue synchronization, which this backend does not implement.
// M3 command-buffer WaitSync accepts only fences ordered earlier on this queue; unresolved or future waits fail the submission.
static void stream_queue_once(void) { static int n; if (++n > 1) MDIE("[RED] classic Metal created command queue %d; only one queue is supported because cross-queue synchronization is not implemented", n); }
static _Atomic uint32_t g_present_qos;
#define ALLOC_MAX 8
static int g_alloc_n, g_alloc_cur;   // In-flight slots for the main submission (Metal 4: one command allocator per slot; classic Metal: limits the number of in-flight submissions)
#if !DM_NVN_M3
static id<MTL4CommandAllocator> g_alloc4[ALLOC_MAX];   // Command allocator pool of the main submission: at start pick one the GPU has finished with, create a new one if there is none (wait only at the cap)
static id<MTL4CommandAllocator> g_up_alloc4[3];
static id<MTL4CommandAllocator> g_present_alloc4[3];
static CBUF g_present_cb[3];
static id<MTL4ArgumentTable> g_vt, g_ft, g_ct, g_up_ct;
static id<MTL4ArgumentTable> g_present_args;
#endif
static id<MTLBuffer> g_present_uv[3];
#define MP_SLOTS 9
static id<MTLBuffer> g_mp_buf[3];   // Magenta probe: one per present slot, 5 picture points + 4 color-table cells
int g_bloom_q = -1;   // Bloom 1/4 starting point (non-static: real-device same-process alternation sets it to 0 / 1 directly)
#if !DM_NVN_M3
static id<MTL4ArgumentTable> g_bloomq_args;
#endif
static id<MTLBuffer> g_bloomq_const[3];
static id<MTLSamplerState> g_bloomq_samp;
static id<MTLRenderPipelineState> g_bloomq_pso;
static MTLPixelFormat g_bloomq_fmt;
static id<MTLTexture> g_bloomq_view;
static int g_bloomq_i;
static struct { int state; uint64_t s_ser, b0_ser, b1_ser, s_h, samp_h, b1_h; const tex_t *s_tex; float threshold, coeff_g, coeff_b; } g_bloomq;
static uint64_t g_bloomq_last_b0, g_bloomq_last_b1h, g_bloomq_last_b1s;   // Most recent bloom 1/4 overlay read redirection: B0 serial → B1 handle (kept across frames, see the 2e2d6700a2569795 branch)
static uint64_t g_present_alloc_fence[3];
static pthread_mutex_t g_pq_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_pq_cv = PTHREAD_COND_INITIALIZER;
static pthread_cond_t g_link_cv = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t g_link_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_link_tick_ns, g_link_tick_n;
static uint64_t g_link_dropped;
static int g_pq_inflight;   // Under g_pq_mu: number of presents committed to the GPU whose completion callback has not arrived yet
int g_dm_present_link;   // Present pacing mode (runtime value of DM_PRESENT_LINK, modified directly during same-process alternating measurement): 0 = acquire the drawable itself; 1 = display link at 60 Hz; 2 = display link at 120 Hz, the game presents a frame every other display-link tick
static int g_present_latency = 1;
static uint64_t g_pq_done;
static uint64_t g_pq_read;
static id<MTLResidencySet> g_res;
static int g_res_dirty;
static id<MTLSharedEvent> g_ev;
static _Atomic uint64_t g_fence;                // Highest committed fence value (incremented exclusively by the submit thread)
static int g_sub_sync_failed;                    // The serialized submit thread owns this flag.
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_done;                         // Highest completed fence value (written by the completion callback)
static _Atomic uint64_t g_first_failed_fence;  // Fail closed after the first command-buffer error; later work cannot be trusted
static _Atomic int g_last_commit_error;
static _Atomic int g_up_pending;

static pthread_mutex_t g_res_mu = PTHREAD_MUTEX_INITIALIZER;   // Dedicated lock for the residency set: res_add is called on paths that hold g_mu (arena_new), so it cannot share g_mu
static NSHashTable *g_res_objects;
#if DM_NVN_M3

// The older API binds by "buffer object + offset", while the replay layer holds GPU addresses: registered buffers (zero-copy buffers of memory pools, argument blocks, staging blocks and so on, a few dozen) are looked up by address range.
// Registered by weak reference (__unsafe_unretained, removed when residency is withdrawn); the most recently hit entry is tried first - bindings within one pass almost always land on the same one or two buffers
enum { M3_BUFS = 512 };
static struct { uint64_t gpu, len; __unsafe_unretained id<MTLBuffer> buf; } g_m3_bufs[M3_BUFS]; static int g_m3_nbufs; static os_unfair_lock g_m3_lk = OS_UNFAIR_LOCK_INIT;
static void m3_buf_add(id<MTLBuffer> b) { os_unfair_lock_lock(&g_m3_lk); if (g_m3_nbufs < M3_BUFS) { g_m3_bufs[g_m3_nbufs].gpu = b.gpuAddress; g_m3_bufs[g_m3_nbufs].len = b.length; g_m3_bufs[g_m3_nbufs++].buf = b; } os_unfair_lock_unlock(&g_m3_lk); }
static void m3_buf_del(id b) { os_unfair_lock_lock(&g_m3_lk); for (int i = 0; i < g_m3_nbufs; i++) if (g_m3_bufs[i].buf == b) { g_m3_bufs[i] = g_m3_bufs[--g_m3_nbufs]; break; } os_unfair_lock_unlock(&g_m3_lk); }
static id<MTLBuffer> m3_buf(uint64_t gpu, NSUInteger *off) {
    static _Thread_local int last; id<MTLBuffer> r = nil; *off = 0; if (!gpu) return nil;
    os_unfair_lock_lock(&g_m3_lk);
    int i = last < g_m3_nbufs && gpu >= g_m3_bufs[last].gpu && gpu - g_m3_bufs[last].gpu < g_m3_bufs[last].len ? last : -1;
    for (int k = 0; i < 0 && k < g_m3_nbufs; k++) if (gpu >= g_m3_bufs[k].gpu && gpu - g_m3_bufs[k].gpu < g_m3_bufs[k].len) i = k;
    if (i >= 0) { last = i; r = g_m3_bufs[i].buf; *off = (NSUInteger)(gpu - g_m3_bufs[i].gpu); }
    os_unfair_lock_unlock(&g_m3_lk);
    return r;
}
#endif
// Test switch DM_RES_CENSUS=1 (not usually enabled): track all objects that entered the resident set (weak references); at the memory reading line, report the count and usage of objects that are still in the resident set and those that have been evicted but not yet destroyed.

// Test switch DM_RES_CENSUS=1 (not normally on): records every object that has entered the residency set (weak references); on the tick of the memory readout line it reports the count and size of those "still in the residency set" and those "removed from the residency set but not yet destroyed".
// The latter = GPU memory we no longer use but someone still holds a reference to (for example an autorelease pool on a guest thread that is never drained)
static NSHashTable *g_res_ever; static int g_res_census = -1;
static void res_add(id<MTLResource> r) {
    if (!r) return;
    pthread_mutex_lock(&g_res_mu);
    [g_res addAllocation:(id<MTLAllocation>)r];
    if (!g_res_objects) g_res_objects = [NSHashTable weakObjectsHashTable];
    [g_res_objects addObject:r];
#if DM_NVN_M3
    if ([r conformsToProtocol:@protocol(MTLBuffer)]) m3_buf_add((id<MTLBuffer>)r);
#endif
    if (g_res_census < 0) g_res_census = getenv("DM_RES_CENSUS") != NULL;
    if (g_res_census > 0) { if (!g_res_ever) g_res_ever = [NSHashTable weakObjectsHashTable]; [g_res_ever addObject:r]; }
    g_res_dirty = 1;
    pthread_mutex_unlock(&g_res_mu);
}
static void res_remove(id<MTLResource> r) {
    if (!r) return;
    pthread_mutex_lock(&g_res_mu);
    [g_res removeAllocation:(id<MTLAllocation>)r];
    [g_res_objects removeObject:r];
#if DM_NVN_M3
    m3_buf_del(r);
#endif
    g_res_dirty = 1;
    pthread_mutex_unlock(&g_res_mu);
}

// DM_RES_CHECK=1 (diagnostic): if textures / buffers the shader accesses through the argument block are not in the residency set, print the first 40 (textures are attributed to the actual allocation via parent texture / underlying buffer)
static int g_res_check_on;
static void res_check(id r, const char *what, int reg) {
    static int nbad;
    if (!g_res_check_on || !r || nbad >= 40) return;
    int is_tex = [r conformsToProtocol:@protocol(MTLTexture)], is_buf = [r conformsToProtocol:@protocol(MTLBuffer)];
    if (!is_tex && !is_buf) return;   // Samplers etc. are not allocations
    id<MTLResource> a = r; int hops = 0;
    if (is_tex) { id<MTLTexture> t = (id<MTLTexture>)a; while (t.parentTexture) { t = t.parentTexture; hops++; } a = t.buffer ? (id<MTLResource>)t.buffer : (id<MTLResource>)t; }
    pthread_mutex_lock(&g_res_mu); int ok = [g_res_objects containsObject:a]; pthread_mutex_unlock(&g_res_mu);
    if (!ok) { nbad++; id<MTLTexture> t = is_tex ? (id<MTLTexture>)r : nil;
        MLOG("[RED] missing residency #%d: %s slot %d %s %p (parent chain: %d, allocation: %lu B, storage: %lu%s", nbad, what, reg, is_tex ? "texture" : "buffer", (__bridge void *)a, hops, (unsigned long)a.allocatedSize, (unsigned long)a.storageMode,
             t ? [NSString stringWithFormat:@" format %lu %lux%lu type %lu usage %#lx", (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.textureType, (unsigned long)t.usage].UTF8String : ""); }
}
static void res_commit(void) {
    pthread_mutex_lock(&g_res_mu);
    if (g_res_dirty) { [g_res commit]; g_res_dirty = 0; }
    pthread_mutex_unlock(&g_res_mu);
}

@interface NvDeferredRelease : NSObject { @public id object; uint64_t fence; int keep_resident; } @end
@implementation NvDeferredRelease @end
static NSMutableArray<NvDeferredRelease *> *g_deferred;
static pthread_mutex_t g_deferred_mu = PTHREAD_MUTEX_INITIALIZER;
// Deferred release has two meanings that must be distinguished (the root cause of the GPU memory fault on the first frame in Tristram):
// defer_obj_release = this object is retired: release its reference and remove it from the resident set after submissions that used it have completed;
// defer_obj_unref   = I only no longer need this reference; the object itself may still be in use (the view cache for a texture ID may store the base texture itself): release only the reference, without touching the resident set.
// Previously there was only the first kind: when invalidating the view cache, textures that were still the current storage for a sequence number were also removed from the resident set. This was harmless before because invalidation only happened when the base texture was actually replaced;

// Deferred release has two meanings that must be kept apart (the root of the GPU memory fault on the first frame on entering Travincal):
//   defer_obj_release = this object is retired: drop the reference after the submissions that used it complete, and remove it from the residency set;
//   defer_obj_unref   = only the reference I hold is no longer wanted, the object itself may still be in use (the view cache of a texture ID may hold the base texture itself): only drop the reference, leave the residency set alone.
// Previously only the former existed: when the view cache was invalidated, a texture that 'is still the current storage of some serial' was also removed from the residency set. Invalidation used to happen only when the base texture was really replaced, so it was never hit;
// after the transparency chain's in-place composite swapped the storage of two textures, each of the two live textures was removed once this way, and afterwards they are used as attachments and sampled every frame as usual - the GPU accesses non-resident memory.
static void defer_obj_release_ex(id obj, int keep_resident) {
    if (!obj) return;
    NvDeferredRelease *d = [NvDeferredRelease new]; d->object = obj; d->fence = g_fence + 1; d->keep_resident = keep_resident;
    pthread_mutex_lock(&g_deferred_mu);
    if (!g_deferred) g_deferred = [NSMutableArray array];
    [g_deferred addObject:d];
    pthread_mutex_unlock(&g_deferred_mu);
}
static void defer_obj_release(id obj) { defer_obj_release_ex(obj, 0); }
static void defer_obj_unref(id obj) { defer_obj_release_ex(obj, 1); }
static void defer_drain(void) {
    NSMutableArray<NvDeferredRelease *> *ready = [NSMutableArray array];
    pthread_mutex_lock(&g_deferred_mu);
    for (NSInteger i = g_deferred.count - 1; i >= 0; i--) {
        NvDeferredRelease *d = g_deferred[(NSUInteger)i];
        if (d->fence <= g_done) { [ready addObject:d]; [g_deferred removeObjectAtIndex:(NSUInteger)i]; }
    }
    pthread_mutex_unlock(&g_deferred_mu);
    for (NvDeferredRelease *d in ready) {
        id obj = d->object;
        if (!d->keep_resident && [obj conformsToProtocol:@protocol(MTLResource)]) {
            pthread_mutex_lock(&g_res_mu);
            int registered = [g_res_objects containsObject:obj];
            pthread_mutex_unlock(&g_res_mu);
            if (registered) res_remove((id<MTLResource>)obj);
        }
        d->object = nil;
    }
}

static pthread_mutex_t g_bf_want_mu = PTHREAD_MUTEX_INITIALIZER;
static char **g_bf_want_keys; static size_t g_bf_want_n, g_bf_want_cap; static int g_bf_want_loaded;
void bf_want_note(const char *key) {
    const char *path = getenv("DM_NVN_BF_WANT");
    if (!path || !key || strlen(key) != 16) return;
    for (int i = 0; i < 16; i++) if (!((key[i] >= '0' && key[i] <= '9') || (key[i] >= 'a' && key[i] <= 'f') || (key[i] >= 'A' && key[i] <= 'F'))) return;
    pthread_mutex_lock(&g_bf_want_mu);
    if (!g_bf_want_loaded) {
        FILE *f = fopen(path, "r"); char line[64];
        if (f) { while (fgets(line, sizeof line, f)) { char *e = strpbrk(line, "\r\n"); if (e) *e = 0; if (strlen(line) != 16) continue;
                if (g_bf_want_n == g_bf_want_cap) { g_bf_want_cap = g_bf_want_cap ? g_bf_want_cap * 2 : 64; g_bf_want_keys = realloc(g_bf_want_keys, g_bf_want_cap * sizeof *g_bf_want_keys); }
                g_bf_want_keys[g_bf_want_n++] = strdup(line); } fclose(f); }
        g_bf_want_loaded = 1;
    }
    for (size_t i = 0; i < g_bf_want_n; i++) if (!strcmp(g_bf_want_keys[i], key)) { pthread_mutex_unlock(&g_bf_want_mu); return; }
    if (g_bf_want_n == g_bf_want_cap) { g_bf_want_cap = g_bf_want_cap ? g_bf_want_cap * 2 : 64; g_bf_want_keys = realloc(g_bf_want_keys, g_bf_want_cap * sizeof *g_bf_want_keys); }
    g_bf_want_keys[g_bf_want_n++] = strdup(key);
    FILE *f = fopen(path, "a"); if (f) { fprintf(f, "%s\n", key); fclose(f); }
    pthread_mutex_unlock(&g_bf_want_mu);
}


// ---------- Memory pools ----------
typedef struct { uintptr_t lo, hi; uintptr_t base; __unsafe_unretained id<MTLBuffer> buf; uint64_t gpu; const void *key; int res; } pool_ent;   // res: already put in the residency set (on demand, see pool_res)
static pool_ent *g_pools; static int g_npool, g_cap;
static NSMutableArray *g_keep;                  // Holds the buffer references (pool_ent has non-owning pointers)
static uint64_t g_pool_adds;

#define C1_BANK_SIZE (4u << 20)
#define C1_DECL_SIZE (64u << 10)
static NSMutableArray *g_c1_banks;
static size_t g_c1_bank_off, g_c1_alloc_bytes, g_c1_data_bytes;
static uint64_t g_c1_count, g_c1_exclusive;
static NSMutableSet<NSString *> *g_c1_full;
static int g_c1_full_valid;
static os_unfair_lock g_c1_lock = OS_UNFAIR_LOCK_INIT;

// DM_UDC_SELFTEST=1 (diagnostic, tests DontCare behavior on a real A19 device): first clear private Depth32Float_Stencil8 / RGBA8 (1952x900) storage to known values,

// DM_UDC_SELFTEST=1 (diagnostic, checks DontCare behavior on a real A19 device): private-storage Depth32Float_Stencil8 / RGBA8 (1952x900) are first cleared to write known values,
// then Load + DontCare without drawing, read back and compared pixel by pixel, printing one result line
static void udc_selftest(void) {
    if (!getenv("DM_UDC_SELFTEST")) return;
    id<MTLCommandQueue> q = [g_dev newCommandQueue];
    for (int depth = 1; depth >= 0; depth--) {
        NSUInteger w = 1952, h = 900;
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:depth ? MTLPixelFormatDepth32Float_Stencil8 : MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> t = [g_dev newTextureWithDescriptor:td];
        id<MTLCommandBuffer> cb = [q commandBuffer];
        for (int pass = 0; pass < 2; pass++) {
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
            MTLLoadAction la = pass ? MTLLoadActionLoad : MTLLoadActionClear; MTLStoreAction sa = pass ? MTLStoreActionDontCare : MTLStoreActionStore;
            if (depth) { rp.depthAttachment.texture = t; rp.depthAttachment.loadAction = la; rp.depthAttachment.storeAction = sa; rp.depthAttachment.clearDepth = 0.3125;
                         rp.stencilAttachment.texture = t; rp.stencilAttachment.loadAction = la; rp.stencilAttachment.storeAction = sa; rp.stencilAttachment.clearStencil = 0x5a; }
            else { rp.colorAttachments[0].texture = t; rp.colorAttachments[0].loadAction = la; rp.colorAttachments[0].storeAction = sa; rp.colorAttachments[0].clearColor = MTLClearColorMake(0.2, 0.4, 0.6, 0.8); }
            [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
        }
        id<MTLBuffer> out = [g_dev newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:out destinationOffset:0
            destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4 options:depth ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionNone];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
        const uint32_t *px = out.contents; NSUInteger bad = 0; for (NSUInteger i = 0; i < w * h; i++) if (px[i] != px[0]) bad++;
        MLOG("UDC self-test %s %s %lux%lu first pixel %08x; differing pixels %lu / %lu", g_dev.name.UTF8String, depth ? "Depth32F_S8" : "RGBA8", (unsigned long)w, (unsigned long)h, px[0], (unsigned long)bad, (unsigned long)(w * h));
    }
}
void mtl_init(void) {
    if (g_dev) return;
    const char *link = getenv("DM_PRESENT_LINK");
    g_dm_present_link = link ? atoi(link) : 1; if (g_dm_present_link < 0 || g_dm_present_link > 2) g_dm_present_link = 1;   // Default 1 (real-device same-process alternation: when load exceeds one display-link period, long frames 15% → 0 and frame intervals even out; no difference from 0 at a steady 60); DM_PRESENT_LINK=0 turns it off   // 1 = present on every tick of a 60 Hz display link; 2 = 120 Hz display link, the game presents a frame every other display-link tick (when behind, it can present on the next 8.33 ms display-link tick)
    const char *latency = getenv("DM_PRESENT_LATENCY");
    if (latency && atoi(latency) > 0) g_present_latency = atoi(latency);
    g_res_check_on = getenv("DM_RES_CHECK") != NULL;
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) MDIE("No Metal device");
    if (!g_dev.hasUnifiedMemory) MDIE("A unified memory device is required (zero-copy memory pool)");
    g_q = [g_dev newCommandQueue];
#if DM_NVN_M3
    stream_queue_once();
#endif
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.initialCapacity = 4096;
    NSError *re = nil;
    g_res = [g_dev newResidencySetWithDescriptor:rd error:&re];
    if (!g_res) MDIE("Failed to create MTLResidencySet");
    [g_q addResidencySet:g_res];
#if !DM_NVN_M3
    dispatch_queue_attr_t feedback_attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
    g_feedback_q = dispatch_queue_create("dm.nvn.mtl4-feedback", feedback_attr);
    MTL4CommandQueueDescriptor *q4d = [MTL4CommandQueueDescriptor new]; q4d.feedbackQueue = g_feedback_q;
    g_q4 = [g_dev newMTL4CommandQueueWithDescriptor:q4d error:&re];
    if (!g_q4) MDIE("Failed to create MTL4CommandQueue");
    stream_queue_once();
    [g_q4 addResidencySet:g_res];
#endif
    g_alloc_n = 3;
#if !DM_NVN_M3
    for (int i = 0; i < 3; i++) {
        MTL4CommandAllocatorDescriptor *ad = [MTL4CommandAllocatorDescriptor new];
        ad.label = [NSString stringWithFormat:@"nvn-mtl allocator %d", i];
        NSError *ae = nil;
        g_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae];
        if (!g_alloc4[i]) MDIE("Failed to create MTL4CommandAllocator[%d]: %s", i, ae.localizedDescription.UTF8String);
        g_up_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae];
        if (!g_up_alloc4[i]) MDIE("Failed to create upload MTL4CommandAllocator[%d]: %s", i, ae.localizedDescription.UTF8String);
        MTL4CommandAllocatorDescriptor *pad = [MTL4CommandAllocatorDescriptor new];
        pad.label = [NSString stringWithFormat:@"nvn-mtl present allocator %d", i];
        g_present_alloc4[i] = [g_dev newCommandAllocatorWithDescriptor:pad error:&ae];
        if (!g_present_alloc4[i]) MDIE("Failed to create render MTL4CommandAllocator[%d]: %s", i, ae.localizedDescription.UTF8String);
        g_present_cb[i] = [g_dev newCommandBuffer];
        if (!g_present_cb[i]) MDIE("Failed to create render MTL4CommandBuffer[%d]", i);
    }
    MTL4ArgumentTableDescriptor *atd = [MTL4ArgumentTableDescriptor new];
    atd.maxBufferBindCount = 31; atd.maxTextureBindCount = 1;
    NSError *ate = nil;
    g_vt = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_vt) MDIE("Failed to create vertex argument table: %s", ate.localizedDescription.UTF8String);
    g_ft = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_ft) MDIE("Failed to create fragment argument table: %s", ate.localizedDescription.UTF8String);
    g_ct = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_ct) MDIE("Failed to create compute argument table: %s", ate.localizedDescription.UTF8String);
    g_up_ct = [g_dev newArgumentTableWithDescriptor:atd error:&ate]; if (!g_up_ct) MDIE("Failed to create upload compute argument table: %s", ate.localizedDescription.UTF8String);
    MTL4ArgumentTableDescriptor *pad = [MTL4ArgumentTableDescriptor new];
    pad.maxTextureBindCount = 1; pad.maxSamplerStateBindCount = 1; pad.maxBufferBindCount = 1;
    g_present_args = [g_dev newArgumentTableWithDescriptor:pad error:&ate];
    if (!g_present_args) MDIE("Failed to create render argument table: %s", ate.localizedDescription.UTF8String);
    g_bloomq_args = [g_dev newArgumentTableWithDescriptor:pad error:&ate];
    if (!g_bloomq_args) MDIE("Failed to create bloom 1/4 argument table: %s", ate.localizedDescription.UTF8String);
#endif
    for (int i = 0; i < 3; i++) {
        g_present_uv[i] = [g_dev newBufferWithLength:sizeof(float) * 8 options:MTLResourceStorageModeShared];
        if (!g_present_uv[i]) MDIE("Failed to create render UV buffer[%d]", i);
        res_add(g_present_uv[i]);
        g_bloomq_const[i] = [g_dev newBufferWithLength:sizeof(float) * 8 options:MTLResourceStorageModeShared];
        if (!g_bloomq_const[i]) MDIE("Failed to create bloom 1/4 constant buffer[%d]", i);
        res_add(g_bloomq_const[i]);
        g_mp_buf[i] = [g_dev newBufferWithLength:MP_SLOTS * 256 options:MTLResourceStorageModeShared];   // Magenta probe (see mp_encode)
        if (!g_mp_buf[i]) MDIE("Failed to create purple flash probe buffer[%d]", i);
        res_add(g_mp_buf[i]);
    }
    g_ev = [g_dev newSharedEvent];
    g_keep = [NSMutableArray array];
    g_nvn_deswz_gpu = nvn_deswz_gpu_init(g_dev) == 0;
    dm_res_report_register(mtl_res_counts);
    if (!g_nvn_deswz_gpu) MLOG("[RED] GPU deswizzle initialization failed; pre-swizzled textures will fall back to CPU");
    MLOG("Metal device %s", g_dev.name.UTF8String);
    udc_selftest();
}

void dm_mtl_l3_tick(uint32_t tick) {
    static int leak = -1;
    if (leak < 0) leak = getenv("DM_TEST_LEAK_GPU") != NULL;
    if (leak && tick % 100 == 0) {
        if (!g_l3_gpu_leaks) g_l3_gpu_leaks = [NSMutableArray array];
        id<MTLBuffer> leak = [g_dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
        if (leak) [g_l3_gpu_leaks addObject:leak];
    }
}

uint64_t dm_mtl_l3_sample(uint32_t tick) {
    (void)tick;
    return g_dev ? g_dev.currentAllocatedSize : 0;
}

static _Atomic uint64_t g_reg_gen = 1;   // Change generation of memory pool / texture / sampler registration: one of the invalidation conditions for argument buffer reuse
// On-demand memory pool residency: zero-copy buffers enter the resident set, and iOS reads the entire pool into memory and pins it as wired (not counted in footprint and cannot be evicted), even if the game never uses the pool——
// On a real 17 Pro Max, a 512 MB render target pool was never addressed during a game session; putting it in the resident set raised system wired memory by about 530 MB. An A14 4 GB device was therefore terminated by the system for system-wide page faults a few seconds after startup (public repository #37).
// So don't put the pool in the resident set when it's created; add it only when the GPU first gets its address via mtl_addr / mtl_gpu_addr, and submit the resident set immediately

// On-demand residency of memory pools: putting a zero-copy buffer into the residency set makes iOS read the whole pool into memory and pin it as wired (not counted in footprint and never paged out), even a pool the game never uses -
// measured on a 17 Pro Max: the 512 MB render-target pool is addressed 0 times in a run, and putting it into the residency set raises device wired memory by about 530 MB; this is why an A14 4 GB device was ended by the system for running out of memory within seconds of launch (public repository #37).
// So a pool is not put into the residency set when created; it is put in, and the residency set committed on the spot, the first time the GPU gets its address through mtl_addr / mtl_gpu_addr
// (these are the only two ways the GPU gets pool addresses; committing on the spot = earlier than any command buffer commit that uses it, without relying on each commit point calling res_commit). Caller holds g_mu
static uint64_t g_cb_commits, g_pool_res_at;   // number of command buffers committed (main + upload); this number at the last on-demand residency (0 = nothing pending to check)
static void pool_res(int i) {
    if (g_pools[i].res) return;
    res_add(g_pools[i].buf); res_commit(); g_pools[i].res = 1; g_pool_res_at = g_cb_commits + 1;
    MLOG("On-demand memory-pool residency: first access to %llu MiB pool; added to the residency set and committed with %llu command buffers already committed",
         (unsigned long long)((g_pools[i].hi - g_pools[i].lo) >> 20), (unsigned long long)g_cb_commits);
}
static void cb_committed(void) {   // called after the main submission / upload submission: checks that "the residency set commit is earlier than the first command buffer commit that uses it", one line after each on-demand residency
    pthread_mutex_lock(&g_mu); uint64_t n = ++g_cb_commits, at = g_pool_res_at; if (at) g_pool_res_at = 0; pthread_mutex_unlock(&g_mu);
    if (at) MLOG("On-demand residency ordering: first following command buffer is #%llu (the residency set was committed before #%llu)", (unsigned long long)n, (unsigned long long)at);
}
void mtl_pool_add(const void *mem, uint64_t size) {
    mtl_init(); g_reg_gen++;
    uintptr_t pg = (uintptr_t)getpagesize(), lo = (uintptr_t)mem, hi = lo + size;
    uintptr_t a = lo & ~(pg - 1), b = (hi + pg - 1) & ~(pg - 1);
    id<MTLBuffer> buf = [g_dev newBufferWithBytesNoCopy:(void *)a length:b - a options:MTLResourceStorageModeShared deallocator:nil];
    if (!buf) MDIE("Failed to create zero-copy buffer for memory pool %p+%#llx", mem, (unsigned long long)size);
    pthread_mutex_lock(&g_mu);
    if (g_npool == g_cap) { g_cap = g_cap ? g_cap * 2 : 64; g_pools = realloc(g_pools, g_cap * sizeof *g_pools); }
    g_pools[g_npool++] = (pool_ent){lo, hi, a, buf, buf.gpuAddress, mem, 0};   // not in the residency set yet: put in the first time it is addressed (pool_res)
    g_pool_adds++;
    [g_keep addObject:buf];
    pthread_mutex_unlock(&g_mu);
}

void mtl_pool_del(const void *mem) {
    g_reg_gen++;
    id<MTLBuffer> gone = nil;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++)
        if (g_pools[i].key == mem) { gone = g_pools[i].buf; [g_keep removeObject:gone]; g_pools[i] = g_pools[--g_npool]; break; }
    pthread_mutex_unlock(&g_mu);
    defer_obj_release(gone);   // retire: release after the submissions that used it complete, and remove from the residency set (it used to be only removed from g_keep - the residency set holds it strongly, so the zero-copy buffer object was never destroyed)
}

int mtl_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    int ok = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++)
        if (addr >= g_pools[i].lo && addr + len <= g_pools[i].hi) {
            pool_res(i); *buf = (__bridge void *)g_pools[i].buf; *off = addr - g_pools[i].base; ok = 1; break;
        }
    pthread_mutex_unlock(&g_mu);
    return ok;
}
int mtl_gpu_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off, uint64_t *gpu) {
    int ok = 0;
    pthread_mutex_lock(&g_mu);
    for (int i = 0; i < g_npool; i++)
        if (addr >= g_pools[i].lo && addr + len <= g_pools[i].hi) {
            pool_res(i); *buf = (__bridge void *)g_pools[i].buf; *off = addr - g_pools[i].base; *gpu = g_pools[i].gpu + *off; ok = 1; break;
        }
    pthread_mutex_unlock(&g_mu);
    return ok;
}


// ---------- Submission ----------
static void defer_reset(void); static void defer_fallback(void);
static void pg_commit(id fb);
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; void *vis; } post_op;   // vis: the visibility slot table for kind 3
// ---------- Occlusion queries (NVN_COUNTER_TYPE_SAMPLES_PASSED) ----------
// NVN semantics: samples passed is a cumulative queue counter—ResetCounter clears it; each subsequent draw increments it; ReportCounter writes the cumulative value at that point (plus a timestamp) when the GPU completes.
// Metal: attach a visibility result buffer to each render pass; draws during counting use counting mode, with a new slot per pass (no cross-pass accumulation); clearing empties the slot table since the last clear,
// At report time, pass the slot table and its owning buffer to the completion callback to sum and write back; use new slots after reporting so subsequent draws are not included in the reported value.

// ---------- Occlusion queries (NVN_COUNTER_TYPE_SAMPLES_PASSED) ----------
// NVN semantics: samples passed is a cumulative counter on the queue - ResetCounter zeroes it, every draw afterwards adds to it, and ReportCounter writes out the cumulative value at that moment (+ timestamp) when the GPU completes.
// Metal: every render pass attaches a visibility result buffer; draws during counting use counting mode, with a new slot per pass (no reliance on accumulation across passes); a reset empties the 'slot table since the reset',
// and a report hands the slot table together with its buffer (retained) to the completion callback to sum and write back; after a report a new slot is used, so later draws are not counted into the reported value.
// (Previously always wrote 0: occlusion query = 'fully occluded', campfires / light sources got culled - the character select screen calls ResetCounter 21 times per frame)
typedef struct { void *buf; uint32_t slot; } vis_slot;
typedef struct { int n; vis_slot s[]; } vis_list;
#define VIS_SLOTS 8192
typedef struct { __strong id<MTLBuffer> buf; uint64_t fence; } vis_buf;
static vis_buf g_vis_ring[4]; static int g_vis_ring_n, g_vis_ring_i = -1;
static id<MTLBuffer> g_vis; static uint32_t g_vis_n;   // This submission's visibility buffer and number of slots used
static int vis_in_ring(id<MTLBuffer> b) { for (int i = 0; i < g_vis_ring_n; i++) if (g_vis_ring[i].buf == b) return 1; return 0; }
static int g_vis_on, g_vis_counting, g_vis_cur = -1; static uint64_t g_vis_cur_pass;
static vis_slot *g_vs; static int g_nvs, g_vscap;      // Slots since the last reset
static uint64_t g_cnt_report[32], g_cnt_reset[32];       // Report / reset counts per counter type (diagnostics)
static void vis_clear(void) { for (int i = 0; i < g_nvs; i++) CFRelease(g_vs[i].buf); g_nvs = 0; g_vis_cur = -1; }
static vis_list *vis_snapshot(void) {
    vis_list *l = malloc(sizeof *l + g_nvs * sizeof(vis_slot)); l->n = g_nvs;
    for (int i = 0; i < g_nvs; i++) { l->s[i] = g_vs[i]; CFRetain(g_vs[i].buf); }
    return l;
}

// Target ranges of buffer writes (copies / inline uniform updates) already encoded in this submission but not yet executed on the GPU: reading these addresses on the CPU side at encode time yields old values (uv_proof avoids them when reading uniforms)
#define PEND_MAX 256
static uint64_t g_pend_lo[PEND_MAX], g_pend_hi[PEND_MAX]; static int g_npend, g_pend_over;
static int pend_hits(uint64_t a, uint64_t n) { if (g_pend_over) return 1; for (int i = 0; i < g_npend; i++) if (a < g_pend_hi[i] && a + n > g_pend_lo[i]) return 1; return 0; }
typedef struct { uint64_t ser; uint16_t level, slice; } zskip_zero;
static zskip_zero g_zskip_zero[4096]; static int g_zskip_n, g_zskip_count;
static zskip_zero g_zskip_cand[8]; static uint8_t g_zskip_cand_ok[8];
static void zskip_clear(void);
static int tbdr_on(void);
static id g_cb;
static BENC g_blit;
static NSMutableArray<dispatch_block_t> *g_commit_blocks;
static uint64_t g_alloc_fence[ALLOC_MAX]; static uint64_t g_alloc_waits, g_alloc_grown;
int mtl_wait(uint64_t fence, uint64_t timeout_ns);
static void up_flush(void);
// The upload batch belongs exclusively to the submission thread (the thread that calls mtl_sub_begin): its encoder cannot be used across threads. Each entry point asserts that the current thread == submission thread,

// The upload batch belongs only to the submit thread (the thread that called mtl_sub_begin): the batch's encoder cannot be used across threads. Every entry point asserts current thread == submit thread,
// and on mismatch prints the stack and aborts on the spot - so that 'another thread touched the batch' is exposed deterministically on Mac too, not by timing (real iOS device stuck at the main menu: the present thread flushed the submit thread's batch via q_cb).
static pthread_t g_sub_thread; static _Atomic int g_sub_thread_set;
static int on_sub_thread(void) { return atomic_load(&g_sub_thread_set) && pthread_equal(pthread_self(), g_sub_thread); }
static void up_owner_check(const char *where) {
    if (!atomic_load(&g_sub_thread_set) || on_sub_thread()) return;
    char name[64] = ""; pthread_getname_np(pthread_self(), name, sizeof name);
    MLOG("[RED] Upload batches belong only to the submission thread: %s was called on thread \"%s\"", where, name);
    void *bt[32]; int n = backtrace(bt, 32); backtrace_symbols_fd(bt, n, 2);
    abort();
}
// Any command buffer created outside replay / the upload batch must go through it. On the submission thread: flush this batch's uploads first, ensuring commands that subsequently read or write those textures are ordered after the uploads;
// Other threads (such as the rendering thread) create command buffers without touching the batch: they do not read or write textures uploaded by this batch; commands that need these textures all run after this replay submission (the batch is flushed before it)
// Old queue command buffers (readback / diagnostics / presentation; remove after M4-4 and M4-5 migrate): uploads are on the Metal 4 queue, and there is no ordering guarantee between the two queues——

// Every command buffer created outside the replay / upload batch goes through this. On the submit thread: flush this batch's uploads first, so that commands that then read or write these textures are ordered after the uploads;
// other threads (the present thread etc.) creating a command buffer do not touch the batch: they do not read or write textures uploaded in this batch; commands that need these textures all come after this replay submission (the batch is flushed before it)
// Old-queue command buffers (readback / diagnostics / present; to be deleted once M4-4 and M4-5 are migrated away): uploads are already on the Metal 4 queue, and there is no ordering guarantee between the two queues -
// wait for the upload to complete after flushing it, and only then can the old queue read the upload result (mtl_read_texels waited first and flushed afterwards, the readback beat the upload, 4 failures on t_nvn upload content)
static id<MTLCommandBuffer> q_cb(void) { if (on_sub_thread()) { up_flush(); while (atomic_load(&g_up_pending)) usleep(200); } return [g_q commandBuffer]; }
static post_op *g_post; static int g_npost, g_postcap;
// One autorelease pool per submission: submissions run on the guest thread (long-lived, no run loop, implicit pool never drained), so objects autoreleased during Metal replay

// One autorelease pool per submission: submissions run on a guest thread (long-lived, no run loop, never drains the implicit pool), so objects autoreleased in the Metal replay
// (command buffers, encoders, pass descriptors, etc.) are never released unless put in a pool. Measured on the character select screen of a real iOS device: the host heap grows by about 47 objects per frame, 0.22 MiB/s.
void *objc_autoreleasePoolPush(void); void objc_autoreleasePoolPop(void *);
static void *g_sub_pool;
static int g_ruler_truth = -1, g_ruler_pending;
static int tbdr_on(void);

static uint64_t g_sub_serial;   // Submission index (used by the parallel precompile's 'predict only once per submission')
static void tbdr_latch(void);
void mtl_sub_begin(void) {
    mtl_init(); g_sub_serial++;
    tbdr_latch();
    if (tbdr_on() == 0 && g_ruler_pending) { g_ruler_truth = 0; g_ruler_pending = 0; MLOG("DM_NVN_RULER: switching to the mode under test"); }
    defer_drain();
    defer_reset();
    warm_claim();
    if (g_cb) MDIE("Submission reentrancy");
    g_sub_thread = pthread_self(); atomic_store(&g_sub_thread_set, 1);
    atomic_store(&g_last_commit_error, 0);
    g_sub_sync_failed = 0;
    g_sub_pool = objc_autoreleasePoolPush();
#if DM_NVN_M3
    g_cb = [g_q commandBuffer];
#else
    g_cb = [g_dev newCommandBuffer];
#endif
    // Don't wait for the GPU mid-frame (on a real device under sustained load, with a fixed 3-buffer rotation, if the GPU falls behind this blocks here every frame; the submission thread wakes and runs at a lower frequency, with instruction count −18% but CPU time +49%):

    // Do not wait for the GPU mid-frame (hot real device: with a fixed rotation of 3, when the GPU cannot keep up every frame blocks here, and the submit thread runs at low frequency after waking, instruction count −18% but CPU time +49%):
    // pick an allocator the GPU has finished with; if all are in use create a new one; only at the cap wait for the oldest one. How far ahead it runs is bounded by the wait where the game acquires the swapchain texture, same as the old interface.
    int alloc_i = -1; uint64_t done = g_done;
    for (int i = 0; i < g_alloc_n; i++) if (g_alloc_fence[i] <= done) { alloc_i = i; break; }
#if DM_NVN_M3
    if (alloc_i < 0 && g_alloc_n < ALLOC_MAX) { alloc_i = g_alloc_n++; g_alloc_grown++; }
#else
    if (alloc_i < 0 && g_alloc_n < ALLOC_MAX) { NSError *ae = nil; MTL4CommandAllocatorDescriptor *ad = [MTL4CommandAllocatorDescriptor new];
        g_alloc4[g_alloc_n] = [g_dev newCommandAllocatorWithDescriptor:ad error:&ae]; if (!g_alloc4[g_alloc_n]) MDIE("Failed to create MTL4CommandAllocator: %s", ae.localizedDescription.UTF8String);
        alloc_i = g_alloc_n++; g_alloc_grown++; }
#endif
    if (alloc_i < 0) { alloc_i = 0; for (int i = 1; i < g_alloc_n; i++) if (g_alloc_fence[i] < g_alloc_fence[alloc_i]) alloc_i = i; g_alloc_waits++; mtl_wait(g_alloc_fence[alloc_i], UINT64_MAX); }
    g_alloc_cur = alloc_i;
#if !DM_NVN_M3
    [g_alloc4[alloc_i] reset];
    [g_cb beginCommandBufferWithAllocator:g_alloc4[alloc_i]];
#endif
    g_commit_blocks = [NSMutableArray array];
    g_npost = 0; g_vis = nil; g_vis_n = 0; g_vis_cur = -1; g_npend = 0; g_pend_over = 0;
    zskip_clear(); g_zskip_count = 0;
    int pick = -1;
    for (int i = 0; i < g_vis_ring_n; i++) if (g_vis_ring[i].fence <= g_done) { pick = i; break; }
    if (pick < 0 && g_vis_ring_n < 4) pick = g_vis_ring_n++;
    if (pick < 0) { uint64_t oldest = UINT64_MAX; for (int i = 0; i < 4; i++) if (g_vis_ring[i].fence < oldest) { oldest = g_vis_ring[i].fence; pick = i; } if (oldest > g_done) mtl_wait(oldest, UINT64_MAX); }
    g_vis_ring_i = pick; g_vis_ring[pick].fence = UINT64_MAX;
    if (!g_vis_ring[pick].buf) { g_vis_ring[pick].buf = [g_dev newBufferWithLength:VIS_SLOTS * 8 options:MTLResourceStorageModeShared]; res_add(g_vis_ring[pick].buf); }
    if (g_vis && !vis_in_ring(g_vis)) defer_obj_release(g_vis);   // the overflow buffer left by the previous submission (see the place with g_vis_n >= VIS_SLOTS): retire it, otherwise it stays in the residency set forever
    g_vis = g_vis_ring[pick].buf;
    void pg_sub_begin_ext(uint64_t fence); pg_sub_begin_ext(g_fence + 1);
    void arena_begin(void); arena_begin();
}

static void rp_end(void);
// DM_NVN_PASSGPU_SERIAL=1 (for measurement): before creating each encoder on the main command buffer, encode “signal n+1 → wait for signal n+1”—the signal is issued only after all preceding work has completed,

// DM_NVN_PASSGPU_SERIAL=1 (for measurement): before creating each encoder on the main command buffer, encode 'signal n+1 → wait for signal n+1' - the signal is only raised once all prior work has completed,
// so the following encoder starts only after the previous one has fully finished; on TBDR passes no longer overlap in the pipeline, and the per-pass start-to-end span of DM_NVN_PASSGPU is that pass's exclusive time (the whole-frame total is inflated; look only at relative cost)
static void ser_gate(void) {
    static int on = -1; static id<MTLEvent> ev; static uint64_t n;
    if (on < 0) { on = getenv("DM_NVN_PASSGPU_SERIAL") != NULL; if (on) { ev = [g_dev newEvent]; MLOG("DM_NVN_PASSGPU_SERIAL: serializing passes (for measurement; higher overhead)"); } }
    if (!on) return;
    (void)ev; (void)n;
}
// Conservative Metal 4 barriers (first version): one consumer barrier at the start of the encoder (all prior work → this encoder), and one producer barrier at the end (this encoder and prior work → subsequent encoders).

// Metal 4 conservative barriers (first version): a consumer barrier at the start of an encoder (all prior work → this encoder) and a producer barrier at the end (this encoder and earlier → later encoders).
// Only the one at the start is not enough: t_rt_range 'pass 1 → pass 2 → copy' read back black, and after adding the end barrier it is 19/19. Old-queue encoders (upload / diagnostics / present) do not go through these barriers.
#if DM_NVN_M3
static inline void m4_end(id e) { [e endEncoding]; }   // the older API has no explicit barriers: dependencies come from useResource declarations plus the system's automatic tracking
static inline void m3_rbuf(id<MTLRenderCommandEncoder> e, int vertex, NSUInteger i, uint64_t gpu) {
    NSUInteger off; id<MTLBuffer> b = m3_buf(gpu, &off); if (gpu && !b) { { static int n_; if (!n_++) MLOG("[RED] classic Metal buffer bind: GPU address %#llx is not in a registered buffer (slot %lu)", (unsigned long long)gpu, (unsigned long)i); } return; }
    if (vertex) [e setVertexBuffer:b offset:off atIndex:i]; else [e setFragmentBuffer:b offset:off atIndex:i];
}
#else
static inline void m4_end(id e) {
    if ([e conformsToProtocol:@protocol(MTL4CommandEncoder)]) [(id<MTL4CommandEncoder>)e barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [e endEncoding];
}
#endif
#if DM_NVN_M3
static inline RENC enc_r_begin(RPD *d) { return [g_cb renderCommandEncoderWithDescriptor:d]; }
#else
static inline RENC enc_r_begin(RPD *d) { RENC e = [g_cb renderCommandEncoderWithDescriptor:d]; [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; [e setArgumentTable:g_vt atStages:MTLRenderStageVertex]; [e setArgumentTable:g_ft atStages:MTLRenderStageFragment]; return e; }
#endif
static inline void enc_r_end(id e) { m4_end(e); }
static inline void enc_r_pso(RENC e, id<MTLRenderPipelineState> p) { [e setRenderPipelineState:p]; }
static inline void enc_r_dss(RENC e, id<MTLDepthStencilState> s) { [e setDepthStencilState:s]; }
static inline void enc_r_viewport(RENC e, MTLViewport v) { [e setViewport:v]; }
static inline void enc_r_scissor(RENC e, MTLScissorRect r) { [e setScissorRect:r]; }
static inline void enc_r_cull(RENC e, MTLCullMode m) { [e setCullMode:m]; }
static inline void enc_r_winding(RENC e, MTLWinding w) { [e setFrontFacingWinding:w]; }
static inline void enc_r_fill(RENC e, MTLTriangleFillMode m) { [e setTriangleFillMode:m]; }
static inline void enc_r_depth_bias(RENC e, float u, float s, float c) { [e setDepthBias:u slopeScale:s clamp:c]; }
static inline void enc_r_depth_clip(RENC e, BOOL v) { [e setDepthClipMode:v ? MTLDepthClipModeClip : MTLDepthClipModeClamp]; }
static inline void enc_r_stencil_ref(RENC e, uint32_t f, uint32_t b) { (void)b; [e setStencilReferenceValue:f]; }
static inline void enc_r_blend_color(RENC e, float r, float g, float b, float a) { [e setBlendColorRed:r green:g blue:b alpha:a]; }
static inline void enc_r_visibility(RENC e, MTLVisibilityResultMode m, NSUInteger o) { [e setVisibilityResultMode:m offset:o]; }
#if DM_NVN_M3
static inline void enc_r_vbuf(RENC e, NSUInteger i, uint64_t gpu) { m3_rbuf(e, 1, i, gpu); }
static inline void enc_r_args(RENC e, MTLRenderStages s, uint64_t gpu, NSUInteger i) { m3_rbuf(e, s == MTLRenderStageVertex, i, gpu); }
#else
static inline void enc_r_vbuf(RENC e, NSUInteger i, uint64_t gpu) { (void)e; [g_vt setAddress:gpu atIndex:i]; }
static inline void enc_r_args(RENC e, MTLRenderStages s, uint64_t gpu, NSUInteger i) { (void)e; [(s == MTLRenderStageVertex ? g_vt : g_ft) setAddress:gpu atIndex:i]; }
#endif
static void draw_trace_encoded(void);
static inline void enc_r_draw(RENC e, MTLPrimitiveType p, NSUInteger v, NSUInteger n) { [e drawPrimitives:p vertexStart:v vertexCount:n]; draw_trace_encoded(); }
static inline void enc_r_draw_instanced(RENC e, MTLPrimitiveType p, NSUInteger v, NSUInteger n, NSUInteger i, NSUInteger b) { [e drawPrimitives:p vertexStart:v vertexCount:n instanceCount:i baseInstance:b]; draw_trace_encoded(); }
#if DM_NVN_M3
static inline void enc_r_draw_indexed(RENC e, MTLPrimitiveType p, NSUInteger n, MTLIndexType t, uint64_t gpu, NSUInteger len, NSUInteger i, NSInteger bv, NSUInteger bi) { (void)len; if (!n) return; NSUInteger off; id<MTLBuffer> b = m3_buf(gpu, &off); if (!b) { { static int n_; if (!n_++) MLOG("[RED] classic Metal indexed draw: index-buffer address %#llx is not in a registered buffer", (unsigned long long)gpu); } return; } [e drawIndexedPrimitives:p indexCount:n indexType:t indexBuffer:b indexBufferOffset:off instanceCount:i baseVertex:bv baseInstance:bi]; draw_trace_encoded(); }
#else
static inline void enc_r_draw_indexed(RENC e, MTLPrimitiveType p, NSUInteger n, MTLIndexType t, uint64_t gpu, NSUInteger len, NSUInteger i, NSInteger bv, NSUInteger bi) { [e drawIndexedPrimitives:p indexCount:n indexType:t indexBuffer:gpu indexBufferLength:len instanceCount:i baseVertex:bv baseInstance:bi]; draw_trace_encoded(); }
#endif
#if DM_NVN_M3
static inline CENC enc_c_begin(MTLComputePassDescriptor *d) { return d ? [g_cb computeCommandEncoderWithDescriptor:d] : [g_cb computeCommandEncoder]; }
#else
static inline CENC enc_c_begin(MTLComputePassDescriptor *d) { (void)d; CENC e = [g_cb computeCommandEncoder]; [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; [e setArgumentTable:g_ct]; return e; }
#endif
static inline id<MTLComputeCommandEncoder> enc_c_begin_concurrent(id<MTLCommandBuffer> cb) { return [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]; }
static inline void enc_c_end(id e) { m4_end(e); }
static inline void enc_c_pso(id e, id<MTLComputePipelineState> p) { [e setComputePipelineState:p]; }
#if DM_NVN_M3
static inline void enc_c_args(id e, uint64_t gpu, NSUInteger i) { NSUInteger off; id<MTLBuffer> b = m3_buf(gpu, &off); if (gpu && !b) { { static int n_; if (!n_++) MLOG("[RED] classic Metal compute bind: GPU address %#llx is not in a registered buffer (slot %lu)", (unsigned long long)gpu, (unsigned long)i); } return; } [(id<MTLComputeCommandEncoder>)e setBuffer:b offset:off atIndex:i]; }
#else
static inline void enc_c_args(id e, uint64_t gpu, NSUInteger i) { (void)e; [g_ct setAddress:gpu atIndex:i]; }
#endif
static inline void enc_c_dispatch(id e, MTLSize n, MTLSize t) { [e dispatchThreadgroups:n threadsPerThreadgroup:t]; }
#if DM_NVN_M3
static inline id enc_b_begin(id cb) { return [cb blitCommandEncoder]; }
#else
static inline id enc_b_begin(id cb) { return [cb respondsToSelector:@selector(beginCommandBufferWithAllocator:)] ? [cb computeCommandEncoder] : [cb blitCommandEncoder]; }
#endif
static inline void enc_b_end(id e) { m4_end(e); }
static inline void enc_b_copy_buffer(id e, id<MTLBuffer> s, NSUInteger so, id<MTLBuffer> d, NSUInteger doff, NSUInteger n) { [e copyFromBuffer:s sourceOffset:so toBuffer:d destinationOffset:doff size:n]; }
static inline void enc_b_copy_buffer_texture(id e, id<MTLBuffer> s, NSUInteger so, NSUInteger row, NSUInteger img, MTLSize z, id<MTLTexture> d, NSUInteger sl, NSUInteger lv, MTLOrigin o) { [e copyFromBuffer:s sourceOffset:so sourceBytesPerRow:row sourceBytesPerImage:img sourceSize:z toTexture:d destinationSlice:sl destinationLevel:lv destinationOrigin:o]; }
static inline void enc_b_copy_texture_buffer(id e, id<MTLTexture> s, NSUInteger sl, NSUInteger lv, MTLOrigin o, MTLSize z, id<MTLBuffer> d, NSUInteger doff, NSUInteger row, NSUInteger img) { [e copyFromTexture:s sourceSlice:sl sourceLevel:lv sourceOrigin:o sourceSize:z toBuffer:d destinationOffset:doff destinationBytesPerRow:row destinationBytesPerImage:img]; }
static inline void enc_b_copy_texture(id e, id<MTLTexture> s, NSUInteger sl, NSUInteger lv, MTLOrigin so, MTLSize z, id<MTLTexture> d, NSUInteger dl, NSUInteger dv, MTLOrigin doo) { [e copyFromTexture:s sourceSlice:sl sourceLevel:lv sourceOrigin:so sourceSize:z toTexture:d destinationSlice:dl destinationLevel:dv destinationOrigin:doo]; }
static void blit_note(void);   // Per-pass dump: a copy / buffer update ended the render pass (defined after PD)
extern int g_blit_bar;
#if DM_NVN_M3
static BENC blit(void) { blit_note(); rp_end(); if (!g_blit) { ser_gate(); g_blit = enc_b_begin(g_cb); } return g_blit; }
#else
// When reusing the same copy encoder, a barrier is also needed between successive copies: commands within the same Metal 4 encoder are not guaranteed to execute in recording order,

// When reusing the same blit encoder, a barrier is also needed between two copies: Metal 4 does not guarantee commands within one encoder run in recording order,
// so write-then-read / two writes to the same range can be reordered (without it, t_nvn 5c read a stale value 1 time in 3). The barrier at encoder creation only covers it against earlier encoders
static BENC blit(void) { blit_note(); rp_end();
    if (!g_blit) { ser_gate(); g_blit = enc_b_begin(g_cb); [g_blit barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; }
    else if (g_blit_bar) [g_blit barrierAfterEncoderStages:MTLStageBlit | MTLStageDispatch beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];   // With Metal 4 the copy encoder is a compute encoder, so both stages are blocked
    return g_blit; }
#endif

void mtl_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    if (g_npend < PEND_MAX) { g_pend_lo[g_npend] = dst; g_pend_hi[g_npend++] = dst + n; } else g_pend_over = 1;
    void *sb, *db; uint64_t so, dof;
    if (!mtl_addr(src, n, &sb, &so) || !mtl_addr(dst, n, &db, &dof))
        MDIE("Buffer copy address is not within any memory pool src=%#llx dst=%#llx n=%#llx", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n);
    enc_b_copy_buffer(blit(), (__bridge id<MTLBuffer>)sb, so, (__bridge id<MTLBuffer>)db, dof, n);
}
// Inline uniform update (UpdateUniformBuffer, OP_UPDATE_UB): the data was copied into command memory at record time; copy it into this layer's shared ring now
// (the game may recycle command memory before the GPU runs) and blit ring -> destination in submission order. NVN semantics keep the update ordered with the draws around it.
static id<MTLBuffer> g_ubs_buf; static size_t g_ubs_off, g_ubs_size;
void mtl_sub_update(const void *data, uint64_t dst, uint64_t n) {
    if (!n) return;
    if (g_npend < PEND_MAX) { g_pend_lo[g_npend] = dst; g_pend_hi[g_npend++] = dst + n; } else g_pend_over = 1;
    void *db; uint64_t dof;
    if (!mtl_addr(dst, n, &db, &dof)) MDIE("uniform update target is in no memory pool dst=%#llx n=%#llx", (unsigned long long)dst, (unsigned long long)n);
    size_t need = (n + 255u) & ~(size_t)255u;
    if (!g_ubs_buf || g_ubs_off + need > g_ubs_size) {
        if (g_ubs_buf) defer_obj_release(g_ubs_buf);   // copies already encoded still reference the old block: release after this submission completes
        g_ubs_size = MAX((size_t)1 << 20, need);
        g_ubs_buf = [g_dev newBufferWithLength:g_ubs_size options:MTLResourceStorageModeShared];
        if (!g_ubs_buf) MDIE("uniform update ring allocation failed %#zx", g_ubs_size);
        res_add(g_ubs_buf); g_ubs_off = 0;
    }
    memcpy((uint8_t *)g_ubs_buf.contents + g_ubs_off, data, n);
    enc_b_copy_buffer(blit(), g_ubs_buf, g_ubs_off, (__bridge id<MTLBuffer>)db, dof, n);
    g_ubs_off += need;
}
void mtl_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    if (!n) return;
    if (dst > UINT64_MAX - n) MDIE("ClearBuffer address overflow dst=%#llx n=%#llx", (unsigned long long)dst, (unsigned long long)n);
    if (g_npend < PEND_MAX) { g_pend_lo[g_npend] = dst; g_pend_hi[g_npend++] = dst + n; } else g_pend_over = 1;
    void *db; uint64_t doff;
    if (!mtl_addr(dst, n, &db, &doff)) MDIE("ClearBuffer target is not in the memory pool dst=%#llx n=%#llx", (unsigned long long)dst, (unsigned long long)n);
    id<MTLBuffer> target = (__bridge id<MTLBuffer>)db;
    uint8_t bytes[4]; memcpy(bytes, &value, sizeof bytes);
#if DM_NVN_M3
    if (bytes[0] == bytes[1] && bytes[0] == bytes[2] && bytes[0] == bytes[3]) {
        [blit() fillBuffer:target range:NSMakeRange((NSUInteger)doff, (NSUInteger)n) value:bytes[0]];
        return;
    }
#endif

    // Non-uniform byte patterns need an immutable staging source; each asynchronous copy must retain its own value until GPU completion.
    NSUInteger chunk = (NSUInteger)(n < 65536 ? n : 65536);
    id<MTLBuffer> pattern = [g_dev newBufferWithLength:chunk options:MTLResourceStorageModeShared];
    if (!pattern) MDIE("ClearBuffer staging buffer allocation failed size=%lu", (unsigned long)chunk);
    uint8_t *contents = pattern.contents;
    for (NSUInteger i = 0; i < chunk; i++) contents[i] = bytes[i & 3];
    res_add(pattern);
    [g_commit_blocks addObject:^{ res_remove(pattern); }];
    BENC enc = blit();
    for (uint64_t off = 0; off < n;) {
        NSUInteger amount = (NSUInteger)(n - off < chunk ? n - off : chunk);
        enc_b_copy_buffer(enc, pattern, 0, target, doff + off, amount);
        off += amount;
    }
}

static void post(post_op o) {
    if (g_npost == g_postcap) { g_postcap = g_postcap ? g_postcap * 2 : 64; g_post = realloc(g_post, g_postcap * sizeof *g_post); }
    g_post[g_npost++] = o;
}
void mtl_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0, NULL}); }
void mtl_sub_counter(uint64_t addr, uint32_t type) { post((post_op){2, NULL, 0, 0, addr, type, NULL}); }
void mtl_counter_stats(uint64_t rep[32], uint64_t rst[32]) { memcpy(rep, g_cnt_report, sizeof g_cnt_report); memcpy(rst, g_cnt_reset, sizeof g_cnt_reset); }
static uint64_t gpu_ticks(void) {
    uint64_t ns = guest_ns();
    if (!ns) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); ns = (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
    return nvn_timestamp_from_ns(ns);
}
static int gpu_feedback_guest_ns(uint32_t type, double gpu_start, double gpu_end, uint64_t guest, uint64_t host_now, uint64_t *out) {
    double seconds = type == NVN_COUNTER_TYPE_TIMESTAMP_TOP ? gpu_start : gpu_end;
    if (!out || !isfinite(seconds) || seconds <= 0 || seconds >= (double)UINT64_MAX / 1e9 || !guest) return 0;
    uint64_t at = (uint64_t)(seconds * 1e9);
    if (at > host_now || host_now - at > guest) return 0;
    *out = guest - (host_now - at);
    return 1;
}
static uint64_t gpu_feedback_ticks(uint32_t type, double gpu_start, double gpu_end) {
    mach_timebase_info_data_t tb;
    if (mach_timebase_info(&tb) != KERN_SUCCESS || !tb.denom) return gpu_ticks();
    uint64_t guest = guest_ns();
    uint64_t now = (uint64_t)((__uint128_t)mach_absolute_time() * tb.numer / tb.denom);
    uint64_t ns;
    if (!gpu_feedback_guest_ns(type, gpu_start, gpu_end, guest, now, &ns)) return gpu_ticks();
    return nvn_timestamp_from_ns(ns);
}
#if defined(DM_NVN_TESTS)
int mtl_test_gpu_feedback_ns(uint32_t type, double gpu_start, double gpu_end, uint64_t guest, uint64_t host_now, uint64_t *out) {
    return gpu_feedback_guest_ns(type, gpu_start, gpu_end, guest, host_now, out);
}
#endif
static void mark_fence_failed(uint64_t fence) {
    uint64_t none = 0;
    atomic_compare_exchange_strong(&g_first_failed_fence, &none, fence);
}
int mtl_fence_failed(uint64_t fence) {
    uint64_t first = atomic_load(&g_first_failed_fence);
    return first && fence >= first;
}
#if defined(DM_NVN_TESTS)
void mtl_test_mark_fence_failed(uint64_t fence) { mark_fence_failed(fence); }
void mtl_test_reset_fence_failures(void) { atomic_store(&g_first_failed_fence, 0); }
#endif


// ---------- Boot instrumentation (DM_BOOTPROF=1): metallib loading, pipeline creation (both happen synchronously during replay on the submit thread), texture upload, first-frame present ----------
static uint64_t bp_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec; }
static uint64_t g_bp_t0;   // Time of process start (when this module is loaded)
__attribute__((constructor)) static void bp_t0(void) { g_bp_t0 = bp_now(); }
static int bootprof(void) { static int v = -1; if (v < 0) v = getenv("DM_BOOTPROF") != NULL; return v; }
static uint64_t g_bp_lib_n, g_bp_lib_ns, g_bp_pso_n, g_bp_pso_ns, g_bp_up_bytes, g_bp_first_present;
static uint64_t g_up_total_ns;   // Cumulative upload time (same source as the per-frame @upload_ms)

// ---------- Per-frame GPU time aggregation ----------
#define GRING 1024
static struct { _Atomic uint64_t frame, ns, t0, t1; _Atomic int pending; } g_gpu[GRING];   // t0 / t1: the frame's earliest GPUStart / latest GPUEnd (mach absolute time, ns)
static _Atomic uint64_t g_cur_frame;
#if defined(DM_NVN_TESTS)
static _Atomic uint64_t g_test_gpu_last_frame;
uint64_t mtl_test_gpu_last_frame(void) { return atomic_load(&g_test_gpu_last_frame); }
#endif
// ---------- Same-process alternating measurement (DM_NVN_UB, off by default: without the environment variable the code below only adds one 'is it enabled' check) ----------
// On a hot real device the system keeps adjusting power / frequency, and segment comparison (segments A and B each run for a minute) cannot resolve differences of <10%; within the same run, toggle by phase in turn and take differences of adjacent phase pairs, so the drift cancels.
// DM_NVN_UB="name=item,item;name=item", one phase every DM_NVN_UB_N frames (default 180, minimum 60): even phases = baseline (nothing changed), odd phases take one category in turn. Each item of a category:
//   @set:<global variable name>=<value>   on entering the phase set this int global variable to the value, on leaving restore it (address found by symbol name; switches to be used on a real device must have a line added to the registration table below)
//   <fragment shader key prefix> or *   the phase drops all draws of these fragment shaders (the picture will be wrong; it only measures 'how much this kind of draw is worth'; a nearly idle category needs a round of its own)

// Table output: $DM_ROOT/run/ruler/ub_table.py; real-device script: $DM_ROOT/run/ruler/ub_alt.sh.
static int g_ub_n = -1, g_ub_N = 180; static _Atomic int g_ub_cur = -1; static char g_ub_name[16][16], g_ub_keys[16][16][48]; static int g_ub_nk[16];
static _Atomic int g_ubp_on, g_ubp_c[4]; static _Atomic uint64_t g_ubp_sum_us;
uint64_t (*g_ub_sub_cpu_ns)(void), (*g_ub_sub_cyc)(void);   // Registered by hle_nvn.c: cumulative CPU nanoseconds / cycles of submissions from the translation thread (0 when not registered)
static uint64_t ub_cpu(void) { return g_ub_sub_cpu_ns ? g_ub_sub_cpu_ns() : 0; }
static uint64_t ub_cyc(void) { return g_ub_sub_cyc ? g_ub_sub_cyc() : 0; }
static uint64_t g_ubx_cpu0, g_ubx_cyc0, g_ubx_f0;   // Counter start of this phase (frame 45): gives per-frame CPU and cycles of the translation thread per phase
void mtl_ub_present(double iv) {   // Called once per frame by the present thread: actual present interval in ms (−1 = dropped, 0 = no reading)
    if (!atomic_load(&g_ubp_on) || iv == 0) return;
    if (iv < 0) { atomic_fetch_add(&g_ubp_c[3], 1); return; }
    atomic_fetch_add(&g_ubp_c[iv <= 20.8 ? 0 : iv <= 29.2 ? 1 : 2], 1); atomic_fetch_add(&g_ubp_sum_us, (uint64_t)(iv * 1000));
}
// Registered runtime switches: name → address. The main executable's symbols in iOS release builds are not in the dynamic symbol table, so dlsym cannot find them (on the first real-device run, all four categories reported “global variable not found,” wasting the whole run),
// so switches to be measured on real devices must be added here; on Mac / in unit tests, unregistered ones are still found by symbol name.

int g_fx_dedup = 1;   // Render encoder fixed-function state not resent if unchanged (FX_SET in draw); 0 = sent on every draw (for in-process alternating measurement comparison)
int g_la_cache = 1;   // Per-object caching for the per-draw reflection summary / all-zero fact table / all-zero set removal (udc_note_draw, zskip_can_draw); 0 = computed every time (for in-process alternating measurement comparison)
int g_blit_bar = 1;   // Barrier when reusing the blit encoder on Metal 4 (blit()); 0 = not set, only for measuring overhead with in-process alternation (data may be reordered)
static int *ub_var(const char *nm) {
    extern int g_bloom_q, g_dm_present_link, g_tbdr_rt, g_blit_bar, g_sdx_attdedup, g_tbdr_sub;
    static const struct { const char *n; int *p; } t[] = { {"g_bloom_q", &g_bloom_q}, {"g_dm_present_link", &g_dm_present_link}, {"g_fx_dedup", &g_fx_dedup}, {"g_la_cache", &g_la_cache}, {"g_tbdr_rt", &g_tbdr_rt}, {"g_blit_bar", &g_blit_bar}, {"g_sdx_attdedup", &g_sdx_attdedup}, {"g_tbdr_sub", &g_tbdr_sub} };
    for (size_t i = 0; i < sizeof t / sizeof t[0]; i++) if (!strcmp(nm, t[i].n)) return t[i].p;
    return dlsym(RTLD_DEFAULT, nm);
}
static void ub_frame(uint64_t f) {
    if (g_ub_n < 0) { g_ub_n = 0; const char *e = getenv("DM_NVN_UB"), *n = getenv("DM_NVN_UB_N"); if (n && atoi(n) >= 60) g_ub_N = atoi(n);
        for (const char *q = e; q && *q && g_ub_n < 16; ) { const char *eq = strchr(q, '='), *end = strchr(q, ';'); if (!end) end = q + strlen(q); if (!eq || eq > end) break;
            int c = g_ub_n++; size_t l = (size_t)(eq - q); if (l > 15) l = 15; memcpy(g_ub_name[c], q, l);
            for (const char *k = eq + 1; k < end && g_ub_nk[c] < 16; ) { size_t kl = 0; while (k + kl < end && k[kl] != ',') kl++; size_t cl = kl > 47 ? 47 : kl; memcpy(g_ub_keys[c][g_ub_nk[c]++], k, cl); k += kl; if (k < end) k++; }
            q = *end ? end + 1 : end; }
        if (g_ub_n) MLOG("DM_NVN_UB: %d classes, %d frames per phase (alternating measurements in the same process; frames in the phase with dropped draws are incorrect)", g_ub_n, g_ub_N); }
    if (!g_ub_n) return;
    uint64_t ph = f / (uint64_t)g_ub_N, r = f % (uint64_t)g_ub_N;
    if (r == 45) { for (int i = 0; i < 4; i++) atomic_store(&g_ubp_c[i], 0); atomic_store(&g_ubp_sum_us, 0); atomic_store(&g_ubp_on, 1); g_ubx_cpu0 = ub_cpu(); g_ubx_cyc0 = ub_cyc(); g_ubx_f0 = f; }
    if (r != 0) return;
    if (f && atomic_load(&g_ubp_on)) { atomic_store(&g_ubp_on, 0); uint64_t pp = ph - 1; int pc = (pp & 1) ? (int)((pp / 2) % (uint64_t)g_ub_n) : -1;
        int c0 = atomic_load(&g_ubp_c[0]), c1 = atomic_load(&g_ubp_c[1]), c2 = atomic_load(&g_ubp_c[2]), c3 = atomic_load(&g_ubp_c[3]), n_ = c0 + c1 + c2;
        double nf = f > g_ubx_f0 ? (double)(f - g_ubx_f0) : 1;
        MLOG("UB presentation %llu %s frames %d avg %.3f ms 2-beat %d 3-beat %d 4-beat or more %d dropped %d translation CPU per frame %.3f ms period %.2f M", (unsigned long long)pp, pc < 0 ? "base" : g_ub_name[pc], n_, n_ ? atomic_load(&g_ubp_sum_us) / 1000.0 / n_ : 0.0, c0, c1, c2, c3,
             (ub_cpu() - g_ubx_cpu0) / 1e6 / nf, (ub_cyc() - g_ubx_cyc0) / 1e6 / nf); }
    int nc = (ph & 1) ? (int)((ph / 2) % (uint64_t)g_ub_n) : -1;
    static int *sp[16]; static int sv[16], ns;   // Variables set by the previous phase: restore first
    for (int i = 0; i < ns; i++) *sp[i] = sv[i]; ns = 0;
    for (int i = 0; nc >= 0 && i < g_ub_nk[nc] && ns < 16; i++) { const char *k = g_ub_keys[nc][i]; if (strncmp(k, "@set:", 5)) continue;
        char nm[48]; const char *eq = strchr(k, '='); if (!eq) continue; size_t l = (size_t)(eq - (k + 5)); if (l >= sizeof nm) continue; memcpy(nm, k + 5, l); nm[l] = 0;
        int *a_ = ub_var(nm); static int bad, ok;
        if (!a_) { if (bad++ < 8) MLOG("[RED]DM_NVN_UB: global variable %s not found (on real hardware, register it in the ub_var table first)", nm); continue; }
        sp[ns] = a_; sv[ns] = *a_; ns++; *a_ = atoi(eq + 1); if (ok++ < 8) MLOG("DM_NVN_UB: %s = %d (original value %d)", nm, atoi(eq + 1), sv[ns - 1]); }
    atomic_store(&g_ub_cur, nc);
}
static int ub_skip(const char *key) {   // Whether this phase should drop draws of this fragment shader
    int uc = atomic_load(&g_ub_cur); if (uc < 0) return 0;
    for (int i = 0; i < g_ub_nk[uc]; i++) { const char *k = g_ub_keys[uc][i]; if (k[0] == '@') continue; if (k[0] == '*' || !strncmp(key, k, strlen(k))) return 1; }
    return 0;
}
void mtl_set_frame(uint64_t f) { atomic_store(&g_cur_frame, f); if (g_ub_n) ub_frame(f); }
// GPU timing registration: gpu_tag_begin(frame) before commit, gpu_tag_done(frame, GPU start, GPU end) on completion (seconds on the CACurrentMediaTime base).
// Auxiliary command buffers use gpu_tag_f; mtl_sub_commit registers the main submission in its completion handler.
static void gpu_tag_begin(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) { atomic_store(&g_gpu[k].frame, f); atomic_store(&g_gpu[k].ns, 0); atomic_store(&g_gpu[k].pending, 0); atomic_store(&g_gpu[k].t0, UINT64_MAX); atomic_store(&g_gpu[k].t1, 0); }
    atomic_fetch_add(&g_gpu[k].pending, 1);
}
static void gpu_tag_done(uint64_t f, double g0, double g1) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) return;
    double d = g1 - g0;
    if (d > 0) {
        atomic_fetch_add(&g_gpu[k].ns, (uint64_t)(d * 1e9));
        uint64_t s0 = (uint64_t)(g0 * 1e9), s1 = (uint64_t)(g1 * 1e9), v;
        for (v = atomic_load(&g_gpu[k].t0); s0 < v && !atomic_compare_exchange_weak(&g_gpu[k].t0, &v, s0);) {}
        for (v = atomic_load(&g_gpu[k].t1); s1 > v && !atomic_compare_exchange_weak(&g_gpu[k].t1, &v, s1);) {}
    }
    atomic_fetch_sub(&g_gpu[k].pending, 1);
}
static void gpu_tag_f(id<MTLCommandBuffer> cb, uint64_t f) { gpu_tag_begin(f); [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { gpu_tag_done(f, c.GPUStartTime, c.GPUEndTime); }]; }
void gpu_tag(id<MTLCommandBuffer> cb) { gpu_tag_f(cb, atomic_load(&g_cur_frame)); }
double mtl_gpu_ms(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return -1;
    return atomic_load(&g_gpu[k].ns) / 1e6;
}
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return 0;
    *t0 = atomic_load(&g_gpu[k].t0); *t1 = atomic_load(&g_gpu[k].t1);
    return *t1 >= *t0 && *t1;
}
static void rp_flush(void);
static const char *cctx_snap(void);   // What is in this submission (printed into the failure line on error; defined before mtl_exec)
uint64_t mtl_sub_commit(void) {
    if (g_sub_sync_failed) defer_reset(); else defer_fallback();   // Do not replay deferred commands after an invalid in-stream wait.
    rp_flush();
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    up_flush();
    uint64_t fence = ++g_fence;
    if (g_sub_sync_failed) { atomic_store(&g_last_commit_error, 1); mark_fence_failed(fence); }
    nvn_sync_trace("commit", "fence-assigned", fence, atomic_load(&g_cur_frame), g_npost, 0, 0);
    if (g_vis_ring_i >= 0) g_vis_ring[g_vis_ring_i].fence = fence;

    // CPU-visible effects after completion: copy the op table into the callback
    int n = g_npost; post_op *ops = n ? malloc(n * sizeof *ops) : NULL;
    if (n) memcpy(ops, g_post, n * sizeof *ops);
    NSArray<dispatch_block_t> *commit_blocks = [g_commit_blocks copy];
#if !DM_NVN_M3
    MTL4CommitOptions *o = [MTL4CommitOptions new];
#endif
    [FB_ON(o, g_cb) FB_SEL:^(FB_ARG f) { (void)f; for (dispatch_block_t block in commit_blocks) block(); }];
    uint64_t gpu_f = atomic_load(&g_cur_frame); gpu_tag_begin(gpu_f);
#if defined(DM_NVN_TESTS)
    atomic_store(&g_test_gpu_last_frame, gpu_f);
#endif
    NSString *cctx = @(cctx_snap());
    [FB_ON(o, g_cb) FB_SEL:^(FB_ARG cb) {
        uint64_t done_before = 0;
        if (nvn_sync_trace_on()) {
            pthread_mutex_lock(&g_mu); done_before = g_done; pthread_mutex_unlock(&g_mu);
            nvn_sync_trace("completion-enter", "MTLCommandBuffer", fence, done_before, cb.error != nil, 0, 0);
        }
        if (cb.error) {
            atomic_store(&g_last_commit_error, 1);
            mark_fence_failed(fence);
            static int n, nv;
            int victim = [cb.error.description containsString:@"InnocentVictim"];
            if (victim ? nv++ < 1 : n++ < 8)
                MLOG("[RED] command buffer error details fence=%llu: %s | GPU %.3f-%.3f",
                     (unsigned long long)fence, cb.error.description.UTF8String, cb.GPUStartTime, cb.GPUEndTime);
            if (!victim) MLOG("[RED] failed submission state fence=%llu: %s", (unsigned long long)fence, cctx.UTF8String);
            MLOG("[RED] command buffer execution error: %s", cb.error.localizedDescription.UTF8String);
        }
        gpu_tag_done(gpu_f, cb.GPUStartTime, cb.GPUEndTime);
        int failed = mtl_fence_failed(fence);
        for (int i = 0; i < n; i++) {
            if (failed) {
                if (ops[i].kind == 3) {
                    vis_list *l = ops[i].vis;
                    for (int k = 0; k < l->n; k++) CFRelease(l->s[k].buf);
                    free(l);
                }
                continue;
            }
            if (ops[i].kind == 1) { if (ops[i].mode == 1) *ops[i].val += ops[i].v; else *ops[i].val = ops[i].v; }
            else if (ops[i].kind == 3) {   // Samples passed: the sum of all slots since the reset
                vis_list *l = ops[i].vis; uint64_t sum = 0;
                for (int k = 0; k < l->n; k++) { sum += ((const uint64_t *)((__bridge id<MTLBuffer>)l->s[k].buf).contents)[l->s[k].slot]; CFRelease(l->s[k].buf); }
                free(l); uint64_t rec[2] = {sum, gpu_feedback_ticks(0, cb.GPUStartTime, cb.GPUEndTime)}; memcpy((void *)ops[i].addr, rec, sizeof rec);
            }
            else { uint64_t t = gpu_feedback_ticks(ops[i].type, cb.GPUStartTime, cb.GPUEndTime), rec[2] = {ops[i].type == 0 || ops[i].type == 0x10 ? t : 0, t}; memcpy((void *)ops[i].addr, rec, sizeof rec); }
        }
        free(ops);
        pthread_mutex_lock(&g_mu);
        if (fence > g_done) g_done = fence;
        uint64_t done_after = g_done;
        pthread_cond_broadcast(&g_cv);
        pthread_mutex_unlock(&g_mu);
        nvn_sync_trace("completion-gdone", "MTLCommandBuffer", fence, done_before, done_after, cb.error != nil, 0);
        defer_drain();
    }];
    pg_commit(FB_ON(o, g_cb));   // Per-pass GPU timing (DM_NVN_PASSGPU): returns immediately when off
    void arena_commit(uint64_t fence); arena_commit(fence);
    res_commit();
#if DM_NVN_M3
    [g_cb encodeSignalEvent:g_ev value:fence];
    nvn_sync_trace("commit-call", "mtl_sub_commit", fence, atomic_load(&g_cur_frame), 0, 0, 0);
    [g_cb commit];
    nvn_sync_trace("commit-return", "mtl_sub_commit", fence, atomic_load(&g_cur_frame), 0, 0, 0);
#else
    [g_cb endCommandBuffer];
    CBUF buffers[] = {g_cb};
    [g_q4 commit:buffers count:1 options:o];
    [g_q4 signalEvent:g_ev value:fence];
#endif
    cb_committed();
    g_alloc_fence[g_alloc_cur] = fence;
    g_cb = nil;
    g_commit_blocks = nil;
    objc_autoreleasePoolPop(g_sub_pool); g_sub_pool = NULL;
    return fence;
}

uint64_t mtl_last_fence(void) { return atomic_load_explicit(&g_fence, memory_order_acquire); }
int mtl_last_commit_error(void) { return atomic_load(&g_last_commit_error); }

int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    int trace = nvn_sync_trace_on();
    uint64_t start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW), done_before = 0;
    if (trace) {
        pthread_mutex_lock(&g_mu); done_before = g_done; pthread_mutex_unlock(&g_mu);
        nvn_sync_trace("wait-enter", "mtl_wait", fence, done_before, timeout_ns, 0, 0);
    }
    pthread_mutex_lock(&g_mu);
    int r = 0;
    while (g_done < fence) {
        uint64_t elapsed = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start;
        if (timeout_ns != UINT64_MAX && elapsed >= timeout_ns) { r = ETIMEDOUT; break; }
        uint64_t remaining = timeout_ns == UINT64_MAX ? 3600000000000ull : timeout_ns - elapsed;
        if (remaining > 3600000000000ull) remaining = 3600000000000ull;
        struct timespec relative = {(time_t)(remaining / 1000000000ull), (long)(remaining % 1000000000ull)};
        r = pthread_cond_timedwait_relative_np(&g_cv, &g_mu, &relative);
        if (r && r != ETIMEDOUT) break;
        if (r == ETIMEDOUT && timeout_ns != UINT64_MAX &&
            clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start >= timeout_ns) break;
        r = 0;
    }
    int ok = g_done >= fence;
    uint64_t done_after = g_done;
    pthread_mutex_unlock(&g_mu);
    if (trace) nvn_sync_trace("wait-exit", "mtl_wait", fence, done_after, (uint64_t)ok,
                              clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start, done_before);
    return ok;
}
static void mtl_finish_gpu(void) {
    if (g_fence) mtl_wait(g_fence, UINT64_MAX);
    while (atomic_load(&g_up_pending)) usleep(1000);
}

void mtl_present_flush(void);
void mtl_finish(void) { mtl_present_flush(); mtl_finish_gpu(); }   // Queue semantics: already-enqueued presents also count as queue work
static void present_drop_for_suspend(void);
void mtl_drain(void) { present_drop_for_suspend(); mtl_present_flush(); mtl_init(); id<MTLCommandBuffer> cb = q_cb(); [cb commit]; [cb waitUntilCompleted]; mtl_finish_gpu(); }


// ================= Render replay =================
#include "../../../host/front.h"
static int g_red_once[64];
#define RED_ONCE(k, ...) do { if (!g_red_once[k]++) MLOG("[RED] " __VA_ARGS__); } while (0)

// ---------- Texture → MTLTexture ----------
// Current-contract helpers (upstream nvn_obj.h): the NVN swizzle moved out of the texture descriptor
// into the texture object (set_swz/swz) and the view; the storage address is pool-relative with a direct-memory fallback
static uint8_t *tex_storage(const tex_t *t) { return t && t->pool && t->pool->mem ? t->pool->mem + t->off : (t ? t->mem : NULL); }
static int tex_is_packaged(const tex_t *t) { return t->d.pkg_block_h != 0; }
static int fmt_compressed(int f) { return (f >= 0x41 && f <= 0x50) || (f >= 0x79 && f <= 0x94); }
static int fmt_depth(int f) { return f >= 0x31 && f <= 0x36; }
static MTLTextureSwizzleChannels tex_swizzle(const tex_t *t, const uint8_t format_swizzle[4]) {
    uint8_t out[4];
    for (int i = 0; i < 4; i++) {
        uint8_t selector = t->set_swz ? (uint8_t)(t->swz >> (3 * i) & 7) : (uint8_t)(NVN_TEXTURE_SWIZZLE_R + i);
        if (selector == NVN_TEXTURE_SWIZZLE_ZERO || selector == NVN_TEXTURE_SWIZZLE_ONE) out[i] = selector;
        else if (selector >= NVN_TEXTURE_SWIZZLE_R && selector <= NVN_TEXTURE_SWIZZLE_A)
            out[i] = format_swizzle[selector - NVN_TEXTURE_SWIZZLE_R];
        else { RED_ONCE(51, "Invalid texture swizzle value %u, using R", selector); out[i] = NVN_SWZ_R; }
    }
    return MTLTextureSwizzleChannelsMake((MTLTextureSwizzle)out[0], (MTLTextureSwizzle)out[1],
                                         (MTLTextureSwizzle)out[2], (MTLTextureSwizzle)out[3]);
}
static int tex_swizzle_identity(MTLTextureSwizzleChannels s) {
    return s.red == MTLTextureSwizzleRed && s.green == MTLTextureSwizzleGreen &&
        s.blue == MTLTextureSwizzleBlue && s.alpha == MTLTextureSwizzleAlpha;
}
typedef struct { MTLPixelFormat a, b; } pfv_pair;
static const pfv_pair g_pfv_srgb[] = {
    {MTLPixelFormatRGBA8Unorm, MTLPixelFormatRGBA8Unorm_sRGB},
    {MTLPixelFormatBGRA8Unorm, MTLPixelFormatBGRA8Unorm_sRGB},
    {MTLPixelFormatBC1_RGBA, MTLPixelFormatBC1_RGBA_sRGB},
    {MTLPixelFormatBC2_RGBA, MTLPixelFormatBC2_RGBA_sRGB},
    {MTLPixelFormatBC3_RGBA, MTLPixelFormatBC3_RGBA_sRGB},
    {MTLPixelFormatBC7_RGBAUnorm, MTLPixelFormatBC7_RGBAUnorm_sRGB},
};
static int tex_need_pfv(MTLPixelFormat base, MTLPixelFormat view) {
    if (base == view) return 0;
    for (size_t i = 0; i < sizeof g_pfv_srgb / sizeof *g_pfv_srgb; i++)
        if ((base == g_pfv_srgb[i].a && view == g_pfv_srgb[i].b) || (base == g_pfv_srgb[i].b && view == g_pfv_srgb[i].a)) return 0;
    return 1;
}

// Cache tables on the replay hot path always use nvn_hmap (C hash, keys not boxed); the table stores CFBridgingRetain'ed objects, CFRelease'd on replace / delete
static void hm_rel(void *v) { CFRelease(v); }
static __unsafe_unretained id hm_get_id(const nvn_hmap *m, const void *k) { return (__bridge id)nvn_hm_get(m, k); }
static void hm_put_id(nvn_hmap *m, const void *k, id o) { void *old = nvn_hm_put(m, k, (void *)CFBridgingRetain(o)); if (old) CFRelease(old); }
static void hm_del_id(nvn_hmap *m, const void *k) { void *old = nvn_hm_del(m, k); if (old) CFRelease(old); }
static nvn_hmap g_by_serial = {.klen = 8};     // Serial → MTLTexture / NvPkg / NvProg
static pthread_mutex_t g_smu = PTHREAD_MUTEX_INITIALIZER;

// Serial table generation: +1 when entries are added / removed or a packaged texture is replaced. Downstream caches (texture views, programs) record the generation at fetch time; if it is unchanged they can be used directly without another table lookup
static _Atomic uint64_t g_ser_gen = 1;
static __unsafe_unretained id serial_get(uint64_t k) { pthread_mutex_lock(&g_smu); __unsafe_unretained id o = hm_get_id(&g_by_serial, &k); pthread_mutex_unlock(&g_smu); return o; }
static void serial_put(uint64_t k, id o) { pthread_mutex_lock(&g_smu); id old = hm_get_id(&g_by_serial, &k); if (old) defer_obj_release(old); hm_put_id(&g_by_serial, &k, o); g_ser_gen++; pthread_mutex_unlock(&g_smu); }
static void serial_del(uint64_t k) { pthread_mutex_lock(&g_smu); id old = hm_get_id(&g_by_serial, &k); if (old) defer_obj_release(old); hm_del_id(&g_by_serial, &k); g_ser_gen++; pthread_mutex_unlock(&g_smu); }

// Host record of a packaged texture: the current MTLTexture, the upload stamp, and the global write generation at the last check
@interface NvPkg : NSObject { @public id<MTLTexture> tex; uint64_t stamp, seen, hash; uint8_t *edge[2]; uint32_t edge_n[2]; } @end   // hash: full hash of the uploaded content under DM_NVN_DIRTYCHECK; edge: the uploaded bytes of the storage's first/last host pages where other data shares them (pkg_edges_take)
@implementation NvPkg
- (void)dealloc { res_remove(tex); for (int i = 0; i < 2; i++) free(edge[i]); }
@end
static uint64_t g_pkg_uploads, g_pkg_reuploads, g_pkg_miss, g_pkg_edge_skips;
static uint64_t g_pfv_upgrades;

// DM_NVN_DIRTYCHECK=1 (diagnostic, compared against ground truth): fully hash the pool content on every use; changed but not detected by dirty-page tracking = a missed detection
static int dirtycheck(void) { static int v = -1; if (v < 0) v = getenv("DM_NVN_DIRTYCHECK") != NULL; return v; }
static uint64_t full_hash(const uint8_t *m, uint64_t n) { uint64_t h = 1469598103934665603ull; for (uint64_t i = 0; i < n; i++) h = (h ^ m[i]) * 1099511628211ull; return h; }
// Writes beside a texture. The tracker marks whole host pages (16 KB) dirty, and guest memory pools are only 4 KB aligned: the first and last pages of a texture's storage are shared with
// whatever lies beside it (a game may write there every frame). So the bytes of the storage in those edge pages are kept at the upload (pkg_edges_take), and a texture whose only written
// pages are its edges, with those bytes unchanged, is protected again instead of re-uploaded (pkg_edges_clean). The edges are the whole storage when it lies within two pages (no page wholly inside)
static void pkg_edge_ranges(const tex_t *t, uintptr_t r[2][2], uintptr_t *ia, uintptr_t *ie) {
    uintptr_t pg = (uintptr_t)getpagesize(), a = (uintptr_t)t->mem, e = a + t->L.size;
    *ia = (a + pg - 1) & ~(pg - 1); *ie = e & ~(pg - 1);
    if (*ie <= *ia) { r[0][0] = a; r[0][1] = e; r[1][0] = r[1][1] = 0; *ia = *ie = 0; return; }
    r[0][0] = a; r[0][1] = *ia; r[1][0] = *ie; r[1][1] = e;
}
static void pkg_edges_take(NvPkg *pk, const tex_t *t) {   // right after an upload, with its stamp in pk->stamp
    for (int i = 0; i < 2; i++) { free(pk->edge[i]); pk->edge[i] = NULL; pk->edge_n[i] = 0; }
    if (!t->mem) return;
    uintptr_t r[2][2], ia, ie; pkg_edge_ranges(t, r, &ia, &ie);
    for (int i = 0; i < 2; i++) if (r[i][1] > r[i][0]) { pk->edge_n[i] = (uint32_t)(r[i][1] - r[i][0]); pk->edge[i] = malloc(pk->edge_n[i]); memcpy(pk->edge[i], (const void *)r[i][0], pk->edge_n[i]); }
    // A write after the upload's stamp may have reached the copy but not the upload: then there is no copy to trust, and the next write re-uploads as before
    if (nvn_dirty_since(t->mem, t->L.size, pk->stamp)) for (int i = 0; i < 2; i++) { free(pk->edge[i]); pk->edge[i] = NULL; pk->edge_n[i] = 0; }
}
static int pkg_edges_clean(NvPkg *pk, const tex_t *t) {   // the covered pages were written since the upload: 1 = only edge pages, and the texture's bytes there are as uploaded
    if (!pk->edge_n[0] && !pk->edge_n[1]) return 0;
    uintptr_t r[2][2], ia, ie; pkg_edge_ranges(t, r, &ia, &ie);
    uint64_t st = nvn_dirty_now(); nvn_dirty_protect(t->mem, t->L.size);   // take the stamp, protect, then read (as pkg_upload): a later write is seen at the next check
    if (ie > ia && nvn_dirty_since((const void *)ia, ie - ia, pk->stamp)) return 0;
    for (int i = 0; i < 2; i++) if (pk->edge_n[i] && memcmp(pk->edge[i], (const void *)r[i][0], pk->edge_n[i])) return 0;
    pk->stamp = st; g_pkg_edge_skips++;
    return 1;
}
// BC formats are uploaded natively (Apple silicon supports BC from the classic floor macOS 15 / iOS 18 up,
// same as the Metal 4 layer); a device without BC support fails closed at mtl_init
static MTLTextureDescriptor *tex_desc(const tex_t *t, int packaged) {
    uint32_t pf = 0; uint8_t format_swizzle[4];
    if (!nvn_fmt_to_mtl(t->d.format, &pf, format_swizzle) || !pf) { RED_ONCE(0, "No Metal mapping for texture format %#x (first occurrence)", t->d.format); return nil; }
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.pixelFormat = (MTLPixelFormat)pf;
    d.swizzle = tex_swizzle(t, format_swizzle);
    d.width = t->d.width ? t->d.width : 1; d.height = t->d.height ? t->d.height : 1; d.depth = 1; d.arrayLength = 1;
    d.mipmapLevelCount = t->d.levels ? t->d.levels : 1;
    switch (t->d.target) {
        case 0: d.textureType = MTLTextureType1D; d.height = 1; break;
        case 2: d.textureType = MTLTextureType3D; d.depth = t->d.depth; break;
        case 3: d.textureType = MTLTextureType1DArray; d.arrayLength = t->d.height; d.height = 1; break;
        case 4: d.textureType = MTLTextureType2DArray; d.arrayLength = t->d.depth; break;
        case 5: d.textureType = MTLTextureType2DMultisample; d.sampleCount = t->d.samples; d.mipmapLevelCount = 1; break;
        case 6: d.textureType = MTLTextureType2DMultisampleArray; d.sampleCount = t->d.samples; d.arrayLength = t->d.depth; d.mipmapLevelCount = 1; break;
        case 8: d.textureType = MTLTextureTypeCube; break;
        case 9: d.textureType = MTLTextureTypeCubeArray; d.arrayLength = t->d.depth; break;
        case 10: RED_ONCE(1, "Buffer texture (TARGET_BUFFER) not implemented"); return nil;
        default: d.textureType = MTLTextureType2D; break;
    }
    d.storageMode = MTLStorageModePrivate;   // Render targets and assets both go in GPU private storage (lossless compression available)
    int stencil_view = t->stencil && (t->d.format == 0x35 || t->d.format == 0x36);
    d.usage = MTLTextureUsageShaderRead | (fmt_compressed(t->d.format) || packaged ? 0 : MTLTextureUsageRenderTarget)
            | (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE ? MTLTextureUsageShaderWrite : 0)   // Image usage: shader-writable (compute kernels generating 3D color lookup tables, etc.)
            | (stencil_view ? MTLTextureUsagePixelFormatView : 0);
    MTLTextureSwizzleChannels sw = d.swizzle;
    if ((d.usage & MTLTextureUsageRenderTarget) && !tex_swizzle_identity(sw)) {
        RED_ONCE(52, "Metal RT does not support non-identity NVN sampling swizzle; creating as identity temporarily (sampling channel semantics not simulated): obj=%p fmt=%#x pf=%u tgt=%u flags=%#x size=%ux%ux%u set_swz=%u swz=%u,%u,%u,%u usage=%#x",
                 (const void *)t, t->d.format, pf, t->d.target, t->d.flags,
                 t->d.width, t->d.height, t->d.depth,
                 t->set_swz, t->swz & 7, t->swz >> 3 & 7, t->swz >> 6 & 7, t->swz >> 9 & 7,
                 (unsigned)d.usage);
        d.swizzle = MTLTextureSwizzleChannelsDefault;
    }
    return d;
}
// In-place upgrade: when the original texture was not created with PixelFormatView (disabled by default to preserve lossless compression on Apple GPUs) but a view with a different format is needed, create a new texture with the original specification + PixelFormatView, copy every slice × level, and replace the registration.

// In-place upgrade: when the original texture was not created with PixelFormatView (off by default, to keep Apple GPU lossless compression) but a different-format view is needed, create a new one with the original spec + PixelFormatView, copy over every slice × level, and replace the registration.
// Count: across the 4 screens character select / camp / wilderness / combat, 17473 textures needed a different-format view 0 times, so this is a fallback and does not rely on predicting which textures need it
static id<MTLTexture> tex_upgrade_pfv(const tex_t *t, id<MTLTexture> base, MTLPixelFormat view_pf) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = base.textureType; d.pixelFormat = base.pixelFormat; d.width = base.width; d.height = base.height; d.depth = base.depth;   // Keep the original format, only add the format-view usage
    d.arrayLength = base.arrayLength; d.mipmapLevelCount = base.mipmapLevelCount; d.sampleCount = base.sampleCount;
    d.storageMode = base.storageMode; d.cpuCacheMode = base.cpuCacheMode; d.hazardTrackingMode = base.hazardTrackingMode;
    d.resourceOptions = base.resourceOptions; d.usage = base.usage | MTLTextureUsagePixelFormatView;
    id<MTLTexture> upgraded = [g_dev newTextureWithDescriptor:d];
    if (!upgraded) { RED_ONCE(45, "Texture upgrade failed ser=%llu format %lu → %lu", t->serial, (unsigned long)base.pixelFormat, (unsigned long)view_pf); return base; }
    res_add(upgraded);
    BENC b = blit();
    NSUInteger slices = base.textureType == MTLTextureType3D ? 1 : base.arrayLength * (base.textureType == MTLTextureTypeCube || base.textureType == MTLTextureTypeCubeArray ? 6 : 1);
    for (NSUInteger l = 0; l < base.mipmapLevelCount; l++) {
        NSUInteger w = MAX(base.width >> l, 1u), h = MAX(base.height >> l, 1u), z = base.textureType == MTLTextureType3D ? MAX(base.depth >> l, 1u) : 1;
        if (base.textureType == MTLTextureType3D)
            enc_b_copy_texture(b, base, 0, l, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, z), upgraded, 0, l, MTLOriginMake(0, 0, 0));
        else for (NSUInteger s = 0; s < slices; s++)
            enc_b_copy_texture(b, base, s, l, MTLOriginMake(0, 0, 0), MTLSizeMake(w, h, 1), upgraded, s, l, MTLOriginMake(0, 0, 0));
    }
    id holder = serial_get(t->serial);
    if ([holder isKindOfClass:[NvPkg class]]) { id<MTLTexture> old = ((NvPkg *)holder)->tex; ((NvPkg *)holder)->tex = upgraded; defer_obj_release(old); g_ser_gen++; }
    else serial_put(t->serial, upgraded);
    MLOG("Texture upgraded to a format-convertible view ser=%llu base format %lu → view format %lu (upgrade count=%llu)", t->serial, (unsigned long)base.pixelFormat, (unsigned long)view_pf, ++g_pfv_upgrades);
    return upgraded;
}
// Pre-tiled texture upload: first set read-only, then take a timestamp, then read and detile pool memory into a staging buffer, and blit into a new private texture.
// Batching: uploads within one replay submission share an upload command buffer (detiling with a compute encoder + unified blit); flush it before submitting the replay command buffer;

// Packaged texture upload: first set read-only, then take the stamp, then read the pool memory and deswizzle into a staging buffer, and blit into a new private texture.
// Batching: uploads within one replay submission share one upload command buffer (compute encoder deswizzle + a unified blit), flushed before the replay command buffer is committed;
// the same queue executes in commit order ⇒ uploads are always finished before the first sample. Uploads outside a submission are flushed on the spot. The staging area is cut from 32 MB private blocks, held by the command buffer until completion.
static _Atomic uint64_t g_upload_ns;   // Cumulative time of deswizzle + upload encoding on the submit thread (timing instrumentation)
static _Atomic uint64_t g_upload_gpu_ns;
static CBUF g_up_cb;
static CENC g_up_ce;
static _Atomic int g_up_alloc_busy[3];
static unsigned g_up_seq;
#define UP_BLOCK_SIZE (32u * 1024u * 1024u)
typedef struct {
    __unsafe_unretained id<MTLBuffer> src;
    size_t off, row, image;
    MTLSize size;
    __unsafe_unretained id<MTLTexture> dst;   // Strongly referenced by g_up_texs until the flush is encoded
    NSUInteger slice, level;
} up_blit;
static up_blit *g_up_blits; static int g_nup_blits, g_up_blitcap;
static NSMutableArray *g_up_texs;   // Target textures in the batch: before the flush, textures may be replaced / released within the same submission
static id<MTLBuffer> g_up_block;
static size_t g_up_block_off, g_up_block_size;
static NSMutableArray *g_up_keep;


// Upload command buffer discarded by the system (also discarded during GPU error recovery; one is lost each run on the first frame entering Tristram): the texture objects in this batch have been created and their timestamps updated, but their contents were never written,
// and the game will never resend them if pool memory stops changing. Record the target textures in this batch; the next time any of them is used, mark it dirty and resend it once.

// The upload command buffer was discarded by the system (dropped along during GPU error recovery; the first frame on entering Travincal drops one every run): the objects of this batch of textures are built and the timestamps updated, but the content was never written in,
// and if the pool memory on the game side no longer changes it will never be re-uploaded. Record this batch of target textures, and the next time any of them is fetched treat it as dirty and re-upload once.
// DM_TEST_UP_LOST=<batch index> (for testing): treat that batch of uploads as dropped, to verify the re-upload path.
static NSHashTable *g_up_lost; static os_unfair_lock g_up_lost_lk = OS_UNFAIR_LOCK_INIT; static _Atomic int g_up_lost_n; static _Atomic uint64_t g_up_lost_redo;   // Weak-reference table: if a texture in the record is never fetched again and is released normally, this table does not keep it alive
static int up_lost_take(id tex) {   // This texture's upload was dropped: take it out of the record and return 1
    if (!tex) return 0; os_unfair_lock_lock(&g_up_lost_lk); int hit = [g_up_lost containsObject:tex]; if (hit) { [g_up_lost removeObject:tex]; atomic_store(&g_up_lost_n, (int)g_up_lost.allObjects.count); atomic_fetch_add(&g_up_lost_redo, 1); } os_unfair_lock_unlock(&g_up_lost_lk); return hit;
}
static void up_diag(void);
static void up_flush(void) {
    up_owner_check("up_flush");
    if (!g_up_cb) return;
    if (g_up_ce) {
        m4_end(g_up_ce); g_up_ce = nil;
    }
#if DM_NVN_M3

    // Older API: copies go through the blit encoder ("staging block -> texture")
    BENC b = g_nup_blits ? [g_up_cb blitCommandEncoder] : nil;
    for (int i = 0; b && i < g_nup_blits; i++) {
        up_blit *x = &g_up_blits[i];
        enc_b_copy_buffer_texture(b, x->src, x->off, x->row, x->image, x->size, x->dst, x->slice, x->level, MTLOriginMake(0, 0, 0));
    }
    [b endEncoding];
    g_up_ce = nil;
#else
    CENC b = g_nup_blits ? g_up_ce = [g_up_cb computeCommandEncoder] : nil;
    if (b) [b barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    for (int i = 0; b && i < g_nup_blits; i++) {
        up_blit *x = &g_up_blits[i];
        enc_b_copy_buffer_texture(b, x->src, x->off, x->row, x->image, x->size, x->dst, x->slice, x->level, MTLOriginMake(0, 0, 0));
    }
    if (b) { m4_end(b); g_up_ce = nil; }
#endif
    CBUF cb = g_up_cb;
    unsigned slot = (g_up_seq - 1) % 3;
    NSMutableArray *dsts = [NSMutableArray arrayWithCapacity:(NSUInteger)g_nup_blits]; for (int i = 0; i < g_nup_blits; i++) if (g_up_blits[i].dst) [dsts addObject:g_up_blits[i].dst];
    static int lost_at = -2; static uint64_t batch_n; if (lost_at == -2) { const char *e = getenv("DM_TEST_UP_LOST"); lost_at = e ? atoi(e) : -1; } int fake_lost = dsts.count && lost_at > 0 && (int)++batch_n == lost_at;
#if !DM_NVN_M3
    MTL4CommitOptions *o = [MTL4CommitOptions new];
#endif
    atomic_fetch_add(&g_up_pending, 1);
    [FB_ON(o, cb) FB_SEL:^(FB_ARG f) {
        if (f.error) MLOG("[RED]Upload command buffer execution error: %s", f.error.description.UTF8String);
        if (f.error || fake_lost) {
            os_unfair_lock_lock(&g_up_lost_lk); if (!g_up_lost) g_up_lost = [NSHashTable weakObjectsHashTable]; for (id d in dsts) [g_up_lost addObject:d]; int n = (int)g_up_lost.allObjects.count; atomic_store(&g_up_lost_n, n); os_unfair_lock_unlock(&g_up_lost_lk);
            pthread_mutex_lock(&g_smu); g_ser_gen++; pthread_mutex_unlock(&g_smu);   // Make the view cache of every texture ID take the slow path once (on a cache hit tex_mtl is not passed through and this record is not seen)
            MLOG("%sUpload command buffer was dropped: marking this batch of %lu textures as content not written; they will be re-uploaded on next use (total pending re-uploads: %d textures)", fake_lost ? "DM_TEST_UP_LOST：" : "", (unsigned long)dsts.count, n);
        }
        double d = f.GPUEndTime - f.GPUStartTime;
        if (d > 0) atomic_fetch_add(&g_upload_gpu_ns, (uint64_t)(d * 1e9));
        atomic_store(&g_up_alloc_busy[slot], 0);
        atomic_fetch_sub(&g_up_pending, 1);
    }];
    res_commit();
#if DM_NVN_M3
    [cb commit];
#else
    [cb endCommandBuffer];
    CBUF buffers[] = {cb};
    [g_q4 commit:buffers count:1 options:o];
#endif
    cb_committed();
    g_up_cb = nil; if (g_up_block) { defer_obj_release(g_up_block); g_up_block = nil; } g_up_block_off = g_up_block_size = 0;
    for (id obj in g_up_keep) defer_obj_release(obj);
    [g_up_keep removeAllObjects];
    up_diag();
    g_nup_blits = 0;
    [g_up_texs removeAllObjects];
}

static void up_begin(void) {
    up_owner_check("up_begin");
    if (g_up_cb) return;
    unsigned slot = g_up_seq++ % 3;
    while (atomic_load(&g_up_alloc_busy[slot])) usleep(1000);
    atomic_store(&g_up_alloc_busy[slot], 1);
#if DM_NVN_M3
    g_up_cb = [g_q commandBuffer];
    g_up_ce = [g_up_cb computeCommandEncoder];
#else
    [g_up_alloc4[slot] reset];
    g_up_cb = [g_dev newCommandBuffer];
    [g_up_cb beginCommandBufferWithAllocator:g_up_alloc4[slot]];
    g_up_ce = [g_up_cb computeCommandEncoder];
    [g_up_ce barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [g_up_ce setArgumentTable:g_up_ct];
#endif
    if (!g_up_texs) g_up_texs = [NSMutableArray array];
    if (!g_up_keep) g_up_keep = [NSMutableArray array];
}

// Diagnostic readings (low frequency; log only when the number of block switches changes and at least 1 second has passed since the previous line; max 200 lines per run): how many times staging blocks ran out of space and switched mid-batch; GPU memory allocated, resident-set object count, and deferred-release queue length.

// Diagnostic readout (low frequency; printed only when the block switch count changed and ≥ 1 second since the last line, at most 200 lines per game): how many times an upload batch ran out of staging block and switched blocks midway; GPU allocated, residency set object count, deferred-release queue length.
// If the old block is not retired on a switch it stays in the residency set and can never be freed, and "GPU allocated" climbs one block at a time with the switch count
static uint64_t g_up_blocks_new, g_up_swaps, g_up_swap_bytes; static NSHashTable *g_up_old;   // g_up_old: old blocks switched out (weak references); the diagnostic line counts how many are still alive
static void up_diag(void) {

    // Print a line when the block switch count changes; then one more line 5 seconds later (that batch's commit is long done and deferred retirement drained, so "GPU allocated" is the settled number)
    static uint64_t said_swaps = ~0ull, last_ns; static int lines, follow; uint64_t now = clock_gettime_nsec_np(CLOCK_MONOTONIC);
    int chg = g_up_swaps != said_swaps, fol = follow && now - last_ns >= 5000000000ull; if ((!chg && !fol) || lines >= 200) return;
    if (chg && last_ns && now - last_ns < 1000000000ull) return; last_ns = now; said_swaps = g_up_swaps; lines++; follow = chg;
    unsigned long alive = 0, inres = 0; uint64_t abytes = 0; NSArray *olds = g_up_old.allObjects;
    pthread_mutex_lock(&g_res_mu); unsigned long nres = (unsigned long)g_res_objects.allObjects.count; for (id<MTLBuffer> o in olds) { alive++; abytes += o.length; inres += [g_res_objects containsObject:o]; } pthread_mutex_unlock(&g_res_mu);
    pthread_mutex_lock(&g_deferred_mu); unsigned long ndef = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    MLOG("Upload staging blocks%s: %llu created total, %llu swaps within batches (%llu MiB of old blocks); %lu old blocks (%llu MiB) still alive, %lu still resident; GPU allocated %llu MiB, %lu resident objects, %lu deferred releases", fol && !chg ? " (settled)" : "",
         (unsigned long long)g_up_blocks_new, (unsigned long long)g_up_swaps, (unsigned long long)(g_up_swap_bytes >> 20), alive, (unsigned long long)(abytes >> 20), inres, (unsigned long long)(g_dev.currentAllocatedSize >> 20), nres, ndef);
}
static id<MTLBuffer> up_alloc(size_t n, size_t *off) {
    size_t need = (n + 255u) & ~((size_t)255u);
    if (!g_up_block || g_up_block_off + need > g_up_block_size) {
        // Block switch: the old block is still referenced by copy / decompression commands already registered for this batch, so it cannot be released immediately; pass it to g_up_keep and defer retirement along with other staging blocks until after the flush submission (remove it from the resident set on retirement).
        // Previously, this directly replaced g_up_block with a new block: the old block was never retired but remained in the resident set, leaking one block of GPU memory (usually 32 MB) for every block switch during an upload batch.

        // Block switch: the old block is still referenced by copy / decompress commands already recorded in this batch and cannot be freed on the spot; hand it to g_up_keep, to be retired later with the other staging after the flush commits (removed from the residency set on retire).
        // Previously g_up_block was simply replaced with the new block here: nobody retired the old block, yet it stayed in the residency set, so each extra block switch in an upload batch leaked one block (usually 32 MB) of GPU memory.
        // The old block and the "current block" retired at flush are two objects, so nothing is retired twice
        g_up_blocks_new++; if (g_up_block) { g_up_swaps++; g_up_swap_bytes += g_up_block_size; if (!g_up_old) g_up_old = [NSHashTable weakObjectsHashTable]; [g_up_old addObject:g_up_block]; static int leak = -1; if (leak < 0) leak = getenv("DM_TEST_UP_LEAK") != NULL;   // test switch DM_TEST_UP_LEAK=1: do not retire the old block, as before the fix (for comparison, to measure the leak)
            if (!leak) { if (!g_up_keep) g_up_keep = [NSMutableArray array]; [g_up_keep addObject:g_up_block]; } }
        g_up_block_size = n > UP_BLOCK_SIZE ? need : UP_BLOCK_SIZE;
        g_up_block = [g_dev newBufferWithLength:g_up_block_size options:MTLResourceStorageModePrivate];
        if (!g_up_block) MDIE("Failed to create pre-swizzled upload staging buffer %#zx", g_up_block_size);
        res_add(g_up_block);
        g_up_block_off = 0;
    }
    *off = g_up_block_off;
    g_up_block_off += need;
    return g_up_block;
}


// Deswizzle parameter blocks: 64 bytes per cell, cut sequentially from a 64 KB shared-storage block; a full block is replaced by a new one, and the old block is released lazily by fence
static id<MTLBuffer> g_up_pblock; static size_t g_up_pblock_off;
static id<MTLBuffer> up_param_alloc(size_t *off) {
    if (!g_up_pblock || g_up_pblock_off + 64 > g_up_pblock.length) {
        if (g_up_pblock) defer_obj_release(g_up_pblock);
        g_up_pblock = [g_dev newBufferWithLength:64 << 10 options:MTLResourceStorageModeShared];
        if (!g_up_pblock) MDIE("Failed to create deswizzle parameter buffer");
        res_add(g_up_pblock); g_up_pblock_off = 0;
    }
    *off = g_up_pblock_off; g_up_pblock_off += 64;
    return g_up_pblock;
}
static void up_blit_add(id<MTLBuffer> src, size_t off, size_t row, size_t image, MTLSize size,
                        id<MTLTexture> dst, NSUInteger slice, NSUInteger level) {
    if (g_nup_blits == g_up_blitcap) {
        g_up_blitcap = g_up_blitcap ? g_up_blitcap * 2 : 256;
        g_up_blits = realloc(g_up_blits, g_up_blitcap * sizeof *g_up_blits);
    }
    g_up_blits[g_nup_blits++] = (up_blit){src, off, row, image, size, dst, slice, level};
}

static id<MTLTexture> pkg_upload_(const tex_t *t, uint64_t *stamp);
static id<MTLTexture> pkg_upload(const tex_t *t, uint64_t *stamp) {
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    id<MTLTexture> r = pkg_upload_(t, stamp);
    clock_gettime(CLOCK_MONOTONIC, &b); uint64_t d_ = (uint64_t)((b.tv_sec - a.tv_sec) * 1000000000ll + b.tv_nsec - a.tv_nsec);
    atomic_fetch_add(&g_upload_ns, d_); g_up_total_ns += d_;
    return r;
}
uint64_t mtl_upload_ns_take(void) { return atomic_exchange(&g_upload_ns, 0); }
uint64_t mtl_upload_gpu_ns_take(void) { return atomic_exchange(&g_upload_gpu_ns, 0); }
void mtl_create_stats(uint64_t *lib, uint64_t *pso) { *lib = g_bp_lib_n; *pso = g_bp_pso_n; }
void mtl_create_ns(uint64_t *lib_ns, uint64_t *pso_ns) { *lib_ns = g_bp_lib_ns; *pso_ns = g_bp_pso_ns; }   // Cumulative time spent synchronously creating shader libraries / pipelines on the spot (the long-frame log takes the per-frame difference)
uint64_t mtl_upload_bytes(void) { return g_bp_up_bytes; }   // Cumulative texture upload bytes
static id<MTLTexture> pkg_upload_(const tex_t *t, uint64_t *stamp) {
    MTLTextureDescriptor *d = tex_desc(t, 1);
    if (!d) return nil;
    uint8_t *mem = tex_storage(t);
    id<MTLTexture> mt = [g_dev newTextureWithDescriptor:d];
    if (!mt) { RED_ONCE(2, "Failed to create MTLTexture %ux%u format %#x target %d", t->d.width, t->d.height, t->d.format, t->d.target); return nil; }
    res_add(mt);   // Both first upload and re-upload register residency here (Metal 4 does not make resources resident automatically; a missed registration = GPU address page fault)
    nvn_dirty_protect(mem, t->L.size);
    *stamp = nvn_dirty_now();
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    unsigned layers = (t->d.target == 4 || t->d.target == 9) ? t->d.depth : t->d.target == 8 ? 6 : t->d.target == 3 ? t->d.height : 1;
    if (t->d.target == 9) layers *= 6;
    size_t total = 0;
    for (unsigned l = 0; l < d.mipmapLevelCount; l++) {
        uint32_t w = MAX(t->d.width >> l, 1u), h = MAX(t->d.height >> l, 1u), z = t->d.target == 2 ? MAX(t->d.depth >> l, 1u) : 1;
        total += (size_t)((w + bw - 1) / bw) * bpb * ((h + bh - 1) / bh) * z * layers;
    }
    g_bp_up_bytes += total;
    void *mb; uint64_t mo;
    int use_gpu = g_nvn_deswz_gpu && mtl_addr((uint64_t)mem, t->L.size, &mb, &mo);
    if (g_nvn_deswz_gpu && !use_gpu) RED_ONCE(40, "Pre-swizzled texture storage is not in the memory pool; falling back to CPU");
    if (use_gpu) {

        // The GPU reads the memory pool at execution time, which is closer to a real Switch than a snapshot at CPU upload time; the dirty-page stamp is still taken before encoding, and later writes still trigger a re-upload.
        up_begin();
        size_t stg_off;
        id<MTLBuffer> stg = up_alloc(total ? total : 16, &stg_off);
        size_t off = 0; int enc_ok = g_up_ce != nil, nb0 = g_nup_blits;
        for (unsigned l = 0; enc_ok && l < d.mipmapLevelCount; l++) {
            uint32_t w = MAX(t->d.width >> l, 1u), h = MAX(t->d.height >> l, 1u), z = t->d.target == 2 ? MAX(t->d.depth >> l, 1u) : 1;
            size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
            for (unsigned y = 0; y < layers; y++) {
#if DM_NVN_M3
                if (nvn_deswz_gpu_encode(g_up_ce, (__bridge id<MTLBuffer>)mb, mo, &t->d, l, y, stg, stg_off + off, pitch)) { enc_ok = 0; break; }
#else
                size_t poff; id<MTLBuffer> params = up_param_alloc(&poff);   // Metal 4 has no setBytes: the parameter block of each deswizzle is cut from a shared block (no new GPU buffer each time)
                if (nvn_deswz_gpu_encode_m4(g_up_ce, g_up_ct, params, poff, (__bridge id<MTLBuffer>)mb, mo, &t->d, l, y, stg, stg_off + off, pitch)) { enc_ok = 0; break; }
#endif
                up_blit_add(stg, stg_off + off, pitch, pitch * rows, MTLSizeMake(w, h, z), mt, y, l);
                off += pitch * rows * z;
            }
        }
        if (enc_ok) {
            [g_up_texs addObject:mt];
            if (!g_cb) up_flush();
            return mt;
        }
        g_nup_blits = nb0;   // Roll back this texture's recorded copies: the already-encoded deswizzle only writes the staging area and is harmless; this texture takes the CPU path below instead
        if (!g_cb) up_flush();
        RED_ONCE(41, "GPU deswizzle of pre-swizzled texture does not support format %#x target %d; falling back to CPU", t->d.format, t->d.target);
    }

    up_begin();
    id<MTLBuffer> stg = [g_dev newBufferWithLength:total ? total : 16 options:MTLResourceStorageModeShared];
    res_add(stg); [g_up_keep addObject:stg];
    size_t off = 0;
    for (unsigned l = 0; l < d.mipmapLevelCount; l++) {
        uint32_t w = MAX(t->d.width >> l, 1u), h = MAX(t->d.height >> l, 1u), z = t->d.target == 2 ? MAX(t->d.depth >> l, 1u) : 1;
        size_t pitch = (size_t)((w + bw - 1) / bw) * bpb, rows = (h + bh - 1) / bh;
        for (unsigned y = 0; y < layers; y++) {
            if (nvn_deswizzle(&t->d, l, y, mem, (uint8_t *)stg.contents + off, pitch)) { RED_ONCE(9, "Deswizzle failed format %#x target %d", t->d.format, t->d.target); break; }
            up_blit_add(stg, off, pitch, pitch * rows, MTLSizeMake(w, h, z), mt, y, l);
            off += pitch * rows * z;
        }
    }
    up_flush();
    return mt;
}
static id<MTLTexture> tex_mtl(const tex_t *ct) {
    tex_t *t = (tex_t *)ct;
    if (!t || !t->serial) return nil;
    uint8_t *mem = tex_storage(t);
    int packaged = tex_is_packaged(t) && mem;
    id have = serial_get(t->serial);
    if (packaged) {
        NvPkg *pk = have;
        if (pk) {   // Already uploaded: check the covered pages only if there has been a guest write since the last check
            uint64_t now = nvn_dirty_now();
            int tracked = nvn_dirty_tracked(mem, t->L.size);
            int clean = tracked && (now == pk->seen || !nvn_dirty_since(mem, t->L.size, pk->stamp));
            pk->seen = now;
            if (clean && atomic_load_explicit(&g_up_lost_n, memory_order_relaxed) && up_lost_take(pk->tex)) { clean = 0; uint64_t k = atomic_load(&g_up_lost_redo); if (k <= 3 || !atomic_load(&g_up_lost_n)) MLOG("Re-uploading dropped texture ser=%llu (cumulative %llu textures, %d remaining in record)", (unsigned long long)t->serial, (unsigned long long)k, atomic_load(&g_up_lost_n)); }
            if (!clean) clean = pkg_edges_clean(pk, t);   // only the pages it shares with other data were written, and the texture's bytes there are unchanged: protect it again
            if (clean) {
                if (dirtycheck() && full_hash(mem, t->L.size) != pk->hash && !g_pkg_miss++)
                    MLOG("[RED]Missed dirty page: texture ser=%llu pool contents changed but change was not detected", t->serial);
                return pk->tex;
            }
            uint64_t st; id<MTLTexture> nt = pkg_upload(t, &st);   // Pool content changed (streamed assets written into the same storage): switch to a new texture; in-flight draws still read the old one
            if (nt) { id<MTLTexture> old = pk->tex; pk->tex = nt; defer_obj_release(old); res_add(nt); g_ser_gen++; pk->stamp = st; pk->seen = nvn_dirty_now(); g_pkg_reuploads++; if (dirtycheck()) pk->hash = full_hash(mem, t->L.size); pkg_edges_take(pk, t); }
            return pk->tex;
        }
        pk = [NvPkg new];
        pk->tex = pkg_upload(t, &pk->stamp);
        if (!pk->tex) return nil;
        pk->seen = nvn_dirty_now(); g_pkg_uploads++; if (dirtycheck()) pk->hash = full_hash(mem, t->L.size); pkg_edges_take(pk, t);
        serial_put(t->serial, pk);
        return pk->tex;
    }
    if (have) return have;
    if (t->d.target == 10) {   // Buffer texture (TARGET_BUFFER): create the texture-buffer view directly on the memory pool buffer (zero copy; the GPU reads the pool's current content at execution time, same as NVN semantics)
        uint32_t pf = 0; uint8_t format_swizzle[4]; int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
        void *mb; uint64_t mo;
        if (!nvn_fmt_to_mtl(t->d.format, &pf, format_swizzle) || !pf || !bpb) { RED_ONCE(36, "Buffer texture format %#x not mapped", t->d.format); return nil; }
        if (!mem || !mtl_addr((uint64_t)mem, (uint64_t)t->d.width * bpb, &mb, &mo)) { RED_ONCE(37, "Buffer texture storage is not in the memory pool"); return nil; }
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)mb;
        NSUInteger al = [g_dev minimumLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)pf];
        if (mo % al) { RED_ONCE(38, "Buffer texture offset %#llx is not aligned to %lu bytes", (unsigned long long)mo, (unsigned long)al); return nil; }
        MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:(MTLPixelFormat)pf width:t->d.width resourceOptions:b.resourceOptions
                                                                                         usage:MTLTextureUsageShaderRead | (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE ? MTLTextureUsageShaderWrite : 0)];
        d.swizzle = tex_swizzle(t, format_swizzle);
        size_t row = ((size_t)t->d.width * bpb + al - 1) / al * al;
        id<MTLTexture> mt = [b newTextureWithDescriptor:d offset:mo bytesPerRow:row];
        if (!mt) { RED_ONCE(39, "Failed to create buffer texture: %u elements format %#x", t->d.width, t->d.format); return nil; }
        res_add(mt);
        serial_put(t->serial, mt);
        return mt;
    }
    MTLTextureDescriptor *d = tex_desc(t, 0);
    if (!d) return nil;
    id<MTLTexture> mt = [g_dev newTextureWithDescriptor:d];
    if (!mt) { RED_ONCE(2, "Failed to create MTLTexture %ux%u format %#x target %d", t->d.width, t->d.height, t->d.format, t->d.target); return nil; }
    res_add(mt);
    serial_put(t->serial, mt);
    return mt;
}
// Memory reading line (always on, low frequency, capped): resident memory, GPU memory allocated, deferred-release queue, resident-set object count, registered memory pool count / peak / tracking table capacity, cumulative texture resends, total / available game heap / largest allocatable block (host dm_guest_heap_stat), remaining memory before the system limit (iOS only).
// Purpose: identify “crashes without logs”—the system gives no notification before killing a process for memory use, so only readings recorded in advance can show which value kept rising.

// Memory readout line (always on, low frequency, capped): resident memory, GPU allocated, deferred release queue, residency set object count, registered memory pool count / peak / tracking table capacity, cumulative texture re-uploads, game heap total / free / largest allocatable block (host dm_guest_heap_stat), and how far from the system memory limit (iOS only).
// Purpose: to pin down "crashes with no log" -- the system gives no notice before killing a process for memory, and only readouts left beforehand can show which number kept climbing.
// Called by the frame hook every 300 frames; printed when resident or GPU allocated changed by >= 64 MB, or game heap free space or largest allocatable block changed by >= 16 MB (the main heap normally has only about 100 MB left), otherwise one line every 3000 frames; capped at 400 lines per run. All integers (the frame hook runs on the small stack of a guest thread, so no floating-point formatting)
static void res_census(unsigned long long frame, unsigned long long rss_mb) {
    @autoreleasepool {
        unsigned long nl[2] = {0, 0}, nr[2] = {0, 0}; unsigned long long bl[2] = {0, 0}, br[2] = {0, 0};   // [0] textures [1] buffers; l = in the residency set, r = retired but not destroyed
        pthread_mutex_lock(&g_res_mu);
        for (id o in g_res_ever.allObjects) { int k = [o conformsToProtocol:@protocol(MTLBuffer)] ? 1 : 0; unsigned long long b = (unsigned long long)[(id<MTLResource>)o allocatedSize];
            if ([g_res_objects containsObject:o]) { nl[k]++; bl[k] += b; } else { nr[k]++; br[k] += b; } }
        pthread_mutex_unlock(&g_res_mu);
        MLOG("Residency census through frame %llu: resident %llu MB, GPU allocated %llu MB; in resident set: %lu textures %llu MB, %lu buffers %llu MB; retired but not destroyed: %lu textures %llu MB, %lu buffers %llu MB", frame, rss_mb, (unsigned long long)(g_dev.currentAllocatedSize >> 20),
             nl[0], bl[0] >> 20, nl[1], bl[1] >> 20, nr[0], br[0] >> 20, nr[1], br[1] >> 20);
    }
}
unsigned long long mtl_gpu_allocated_mb(void) { id<MTLDevice> d = g_dev; return d ? (unsigned long long)(d.currentAllocatedSize >> 20) : 0; }   // GPU allocated (0 before the graphics layer starts); callable from any thread
void mtl_mem_stat(unsigned long long frame, unsigned long long rss_mb) {
    if (g_res_census > 0) res_census(frame, rss_mb);
    static unsigned long long last_rss, last_gpu, last_frame; static int lines; unsigned long long gpu = (unsigned long long)(g_dev.currentAllocatedSize >> 20);
    unsigned long long dr = rss_mb > last_rss ? rss_mb - last_rss : last_rss - rss_mb, dg = gpu > last_gpu ? gpu - last_gpu : last_gpu - gpu;
    static int all = -1; if (all < 0) all = getenv("DM_MEM_STAT_ALL") != NULL;   // For measurement: printed every time, uncapped
    if (!all && (lines >= 400 || (lines && dr < 64 && dg < 64 && frame - last_frame < 3000))) return; lines++; last_rss = rss_mb; last_gpu = gpu; last_frame = frame;
    pthread_mutex_lock(&g_deferred_mu); unsigned long nd = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    pthread_mutex_lock(&g_res_mu); unsigned long nr = (unsigned long)g_res_objects.count; pthread_mutex_unlock(&g_res_mu);
    int live = 0, peak = 0, cap = 0; nvn_dirty_stat(&live, &peak, &cap);
    MLOG("Memory through frame %llu: resident %llu MB, GPU allocated %llu MB, deferred release queue %lu, resident set %lu items, memory pools %d (peak %d / tracking table capacity %d), cumulative texture retransmissions %llu",
         frame, rss_mb, gpu, nd, nr, live, peak, cap, (unsigned long long)g_pkg_reuploads);
}
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = g_pkg_uploads; *reuploads = g_pkg_reuploads; }
uint64_t mtl_pkg_miss(void) { return g_pkg_miss; }


// ---------- Render passes: SetRenderTargets opens a new pass; clears before the first command in the pass are folded into the load action ----------
static struct {
    p_set_rt rt; int has_rt;
    uint8_t clear_c[8]; MTLClearColor cc[8];
    uint8_t clear_d, clear_s; double cd; uint32_t cs;
    p_scissor sc; int has_sc;
} R;
static int nvn_draw_trace_target_draw_on(void) {
    if (!nvn_draw_trace_on()) return 0;
    if (!g_nvn_draw_trace_target_serial) return 1;
    return R.has_rt && R.rt.n > 0 && R.rt.color[0] &&
        R.rt.color[0]->serial == g_nvn_draw_trace_target_serial;
}

// ---------- Zero-source skipping ----------
// Shader fact table (statically proven keys): “if all sampled textures are 0 and coefficients are finite ⇒ output is all 0”. A coefficient is the comp component of the vec vector in the fragment-stage constant buffer register ubo_slot.
// 4a819818b8e489c5 ps_outline_blur_p：output = (sum of several samples from the same texture along a row / column) × fp_c7._m0[1].x (fp_c7 = cbuf reg 4); homogeneous linear in the sample values, with no constant term

// ---------- Zero-source skip ----------
// Shader fact table (statically proven keys): 'the textures it samples are all 0 and the coefficient is finite ⇒ the output is all 0'. Coefficient = component comp of vector vec of fragment-stage constant buffer register ubo_slot.
//   4a819818b8e489c5 ps_outline_blur_p: output = (sum of several samples of the same texture along a row / column) × fp_c7._m0[1].x (fp_c7 = cbuf reg 4), homogeneous linear in the sampled values, no constant term
//   (source of the proof: dmsh_b/<chain version>/4a819818b8e489c5_b.metal, read line by line). Add other shaders here once they are proven.
typedef struct { char key[17]; int ubo_slot, vec, comp; } zskip_fact;
static zskip_fact g_zskip_facts[16] = {{"4a819818b8e489c5", 4, 1, 0}};
static int g_zskip_nf = 1;
void zskip_test_add(const char *key, int ubo_slot, int vec, int comp) {
    if (!key || g_zskip_nf >= 16) return;
    snprintf(g_zskip_facts[g_zskip_nf].key, sizeof g_zskip_facts[0].key, "%s", key);
    g_zskip_facts[g_zskip_nf].ubo_slot = ubo_slot; g_zskip_facts[g_zskip_nf].vec = vec; g_zskip_facts[g_zskip_nf].comp = comp; g_zskip_nf++;
}
int zskip_test_count(void) { return g_zskip_count; }
static int g_zskip_env = -1;   // Cache of DM_NVN_ZSKIP (−1 = not read). This function is called on every draw, and getenv once took about 8% of the submit path; unit tests set it back to −1 after toggling the switch mid-run
static int zskip_on(void) { if (g_zskip_env < 0) { const char *e = getenv("DM_NVN_ZSKIP"); g_zskip_env = !e || strcmp(e, "0"); } return g_zskip_env && tbdr_on(); }
static uint64_t g_zskip_gen = 1;   // The all-zero set is changed only via zskip_add / zskip_clear: with the generation unchanged, an index just removed will not be in the set again
static void zskip_clear(void) { g_zskip_n = 0; memset(g_zskip_cand_ok, 0, sizeof g_zskip_cand_ok); g_zskip_gen++; }
static void zskip_del(uint64_t ser) { for (int i = 0; i < g_zskip_n;) if (g_zskip_zero[i].ser == ser) g_zskip_zero[i] = g_zskip_zero[--g_zskip_n]; else i++; }
static int zskip_has(uint64_t ser, int lv, int sl) { for (int i = 0; i < g_zskip_n; i++) if (g_zskip_zero[i].ser == ser && g_zskip_zero[i].level == lv && g_zskip_zero[i].slice == sl) return 1; return 0; }
static void zskip_add(uint64_t ser, int lv, int sl) { if (!ser || zskip_has(ser, lv, sl) || g_zskip_n >= 4096) return; g_zskip_zero[g_zskip_n++] = (zskip_zero){ser, (uint16_t)lv, (uint16_t)sl}; g_zskip_gen++; }
static int zskip_zero_clear(int i) { return R.clear_c[i] && R.cc[i].red == 0 && R.cc[i].green == 0 && R.cc[i].blue == 0 && R.cc[i].alpha == 0; }
static void zskip_commit_clears(void) { for (int i = 0; i < 8; i++) if (g_zskip_cand_ok[i]) zskip_add(g_zskip_cand[i].ser, g_zskip_cand[i].level, g_zskip_cand[i].slice); }
static int zskip_target_zero(const tex_t *t, const view_t *v, int hv) { int lv = hv && v->set_levels ? v->base_level : 0, sl = hv && v->set_layers ? v->min_layer : 0; return t && zskip_has(t->serial, lv, sl); }
#ifdef NVN_RT_RANGE_TEST
static int g_rr_checking;
static void rr_begin(void), rr_finish(int ranged);
static void rr_snap(const p_draw *, const p_draw_idx *);
static void rr_reset(void);
#endif
static RENC g_enc;

// Target info of the current render pass: determined once from the attachments at rp_begin and shared by all draws in the pass (pipeline key, depth-stencil, viewport / scissor all depend only on it)
static struct { uint32_t cfmt[8], dfmt, sfmt, samples; int hd, hs; NSUInteger tw, th; uint64_t serial, reg_gen, ser_gen, range_dirty; int range_external, range_depth_nonro, layered; __unsafe_unretained id<MTLTexture> range_color, range_depth; } g_pass;
static int g_rp_layered;
static id<MTLRenderPipelineState> g_enc_pso; static id<MTLDepthStencilState> g_enc_dss;

// Fixed-function state already set on this encoder (not resent if unchanged; same lifetime as g_enc_pso: invalidated when a new render encoder is opened). valid bits: 0 viewport 1 scissor 2 cull 3 winding 4 fill 5 depth bias 6 blend color 7 stencil reference
static struct { uint32_t valid; MTLViewport vp; MTLScissorRect sc; int cull, wind, fill; float db[3], bc[4]; uint32_t sref; } g_enc_fx;   // Already set on this encoder (not set again if identical)
static int g_enc_vis, g_enc_has_vis;   // This encoder's current visibility count mode (0 off / slot number + 1), and whether a visibility buffer is attached
static int g_pso_dirty = 1, g_dss_dirty = 1;   // Pipeline / depth-stencil related state was changed during replay
static uint64_t g_stat_args_enc, g_stat_args_reuse;   // Argument buffer: new-encode / reuse counts
static uint64_t g_stat_draw, g_stat_pass, g_bw_load, g_bw_store, g_pass_full_rw, g_udc_saved_bytes;   // Bandwidth ledger: attachment load / store bytes, number of full load-store passes
// DM_NVN_PASSDUMP: record channel attachments / sampling / output image for each draw (textures by index) for offline data-flow analysis; channel numbers match across P / D / R / S lines (= g_stat_pass after channel tracking is enabled)

// DM_NVN_PASSDUMP: records pass attachments / samples / presents one by one (textures by serial) for offline data-flow analysis; the pass numbers of P / D / R / S lines agree (= g_stat_pass after the pass is opened)
// DM_NVN_PASSDUMP=frame number, or start-end (inclusive at both ends)
static _Atomic long g_pd_ctl_a0 = -1, g_pd_ctl_a1;   // dm_gfx_passdump_next (DM_CTL "passdump n"): dump passes of the next n frames
static _Atomic int g_origin_ll;   // NVN window origin lower-left (GL): viewport flipped, winding reversed, window image stored bottom-up (mtl_set_origin_lower_left)
static int passdump(void) {
#ifdef NVN_RT_RANGE_TEST
    if (g_rr_checking) return 0;
#endif
    static long a0 = -2, a1; if (a0 == -2) { const char *e = getenv("DM_NVN_PASSDUMP"); a0 = e ? atol(e) : -1; a1 = e && strchr(e, '-') ? atol(strchr(e, '-') + 1) : a0; }
    long f = (long)atomic_load(&g_cur_frame);
    long ca0 = atomic_load(&g_pd_ctl_a0);
    if (ca0 >= 0 && f >= ca0 && f <= atomic_load(&g_pd_ctl_a1)) return 1;
    return a0 >= 0 && f >= a0 && f <= a1;
}
void dm_gfx_passdump_next(long n) { long f = (long)atomic_load(&g_cur_frame); atomic_store(&g_pd_ctl_a1, f + n); atomic_store(&g_pd_ctl_a0, f + 1); }
#define PD(...) do { if (passdump()) { fprintf(stderr, "[pass] F%llu ", (unsigned long long)atomic_load(&g_cur_frame)); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
static void blit_note(void) { if (g_enc) PD("E %llu copy / buffer update end render channel", g_stat_pass); }
// DM_GFXUSE=<file>: after the character selection page stabilizes (starting 1 second after the first OpenPanel MainMenuButtonRibbon, requires DM_BOOTPROF=1), over 30 consecutive frames,
// record the ranges in guest memory of resources actually read by the GPU (texture storage, UBO / SSBO, vertex / index buffers), deduplicated by (type, address, length), with one line on first sight.

// DM_GFXUSE=<file>: once the character select screen is stable (starting 1 second after the first OpenPanel MainMenuButtonRibbon, requires DM_BOOTPROF=1), over 30 consecutive frames,
// the ranges in guest memory of the resources the GPU actually reads (texture storage, UBO / SSBO, vertex / index buffers), deduplicated by (kind, address, length), one line recorded on first sight.
// For intersecting with the inflate output ranges (which data decompressed before character select is really used by the character select screen). Columns: kind  guest_addr  len  first_frame  tex_id  fmt  w×h
__attribute__((weak)) uint64_t dm_prof_ribbon_ns(void) { return 0; }   // diag.c provides the strong definition; when not merged it is always 0 → the time window does not open, and a failure is reported at the end
__attribute__((weak)) void dm_shot_written(const char *path, int ok) { (void)path; (void)ok; }
__attribute__((weak)) int dm_guest_heap_stat(unsigned long long *t, unsigned long long *f, unsigned long long *m) { (void)t; (void)f; (void)m; return 0; }   // hle_svc.c provides the strong definition (game main heap readout); the graphics unit tests do not link the host and cannot read it
__attribute__((weak)) void dm_guest_frame_tick(void) {}   // hle_os.c provides the strong definition (DM_DETCLOCK deterministic guest clock); a no-op when graphics unit tests do not link the host   // diag.c provides the strong definition (DM_TICK_HOLD=E:SHOT unfreeze); a no-op when graphics unit tests do not link the host
__attribute__((weak)) int dm_detclock_enabled(void) { return 0; }   // Same as above: the host's hle_os.c provides the strong definition; when graphics unit tests do not link the host, the deterministic clock is treated as off
__attribute__((weak)) uint64_t guest_ns(void) { return 0; }          // Same as above: read only when the deterministic clock is on; never reached in graphics unit tests
enum { USE_OFF, USE_WAIT, USE_REC, USE_DONE };
static int g_use = -1; static FILE *g_use_f; static long g_use_f0, g_use_rows; static nvn_hmap g_use_seen = {.klen = 24};
#define USE_FRAMES 30
static void use_exit(void) {
    if (g_use == USE_WAIT) fprintf(stderr, "[dm] [RED] DM_GFXUSE time window never opened: dm_prof_ribbon_ns() = %llu (requires DM_BOOTPROF=1 and host with panel instrumentation)\n", (unsigned long long)dm_prof_ribbon_ns());
    else if (g_use == USE_REC) fprintf(stderr, "[dm] [RED] DM_GFXUSE recorded only %ld frames (less than %d) before exit; file is incomplete\n", (long)atomic_load(&g_cur_frame) - g_use_f0, USE_FRAMES);
    if (g_use_f) fflush(g_use_f);
}
static void use_tick(void) {   // Called on every present
    if (g_use < 0) { const char *e = getenv("DM_GFXUSE"); g_use = USE_OFF;
        if (e) { g_use_f = fopen(e, "w"); if (!g_use_f) { MLOG("[RED] DM_GFXUSE cannot open %s", e); return; }
            fprintf(g_use_f, "kind\tguest_addr\tlen\tfirst_frame\ttex_id\tfmt\twxh\n"); fflush(g_use_f); g_use = USE_WAIT; atexit(use_exit);
            MLOG("[gfxuse] Waiting for panel event: start recording %d frames 1 second after first OpenPanel MainMenuButtonRibbon → %s", USE_FRAMES, e); } }
    long f = (long)atomic_load(&g_cur_frame);
    if (g_use == USE_WAIT) {
        uint64_t t = dm_prof_ribbon_ns();
        if (t && now_ns() - t >= 1000000000ull) { g_use = USE_REC; g_use_f0 = f; MLOG("[gfxuse] Start recording: from frame %ld for %d frames", f, USE_FRAMES); }
    } else if (g_use == USE_REC && f - g_use_f0 >= USE_FRAMES) {
        g_use = USE_DONE; fclose(g_use_f); g_use_f = NULL;
        MLOG("[gfxuse] Recording complete: frames %ld–%ld, %ld intervals after deduplication", g_use_f0, f - 1, g_use_rows);
    }
}
static __attribute__((always_inline)) inline void use_rec(const char *kind, uint64_t addr, uint64_t len, int tid, const tex_t *t) {
    if (g_use != USE_REC || !addr || !len) return;
    uint64_t k[3] = {addr, len, (uint64_t)(uint8_t)kind[0]};
    if (nvn_hm_get(&g_use_seen, k)) return;
    nvn_hm_put(&g_use_seen, k, (void *)1);
    g_use_rows++;
    if (t) fprintf(g_use_f, "%s\t%#llx\t%llu\t%lld\t%d\t%#x\t%ux%u\n", kind, (unsigned long long)addr, (unsigned long long)len, (long long)atomic_load(&g_cur_frame), tid, t->d.format, t->d.width, t->d.height);
    else fprintf(g_use_f, "%s\t%#llx\t%llu\t%lld\t\t\t\n", kind, (unsigned long long)addr, (unsigned long long)len, (long long)atomic_load(&g_cur_frame));
}
static uint64_t att_bytes(id<MTLTexture> t, NSUInteger level) {
    if (!t) return 0;
    NSUInteger w = MAX(t.width >> level, 1), h = MAX(t.height >> level, 1);
    NSUInteger bpp = t.pixelFormat == MTLPixelFormatDepth32Float_Stencil8 ? 5 : t.pixelFormat == MTLPixelFormatRGBA16Float || t.pixelFormat == MTLPixelFormatRG32Float ? 8 :
                     t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : t.pixelFormat == MTLPixelFormatR8Unorm || t.pixelFormat == MTLPixelFormatStencil8 ? 1 :
                     t.pixelFormat == MTLPixelFormatRG8Unorm || t.pixelFormat == MTLPixelFormatR16Float ? 2 : 4;
    return (uint64_t)w * h * bpp * t.sampleCount;
}

// The render attachment selects a particular mip, while MTLTexture.width/height still describe the base level. Empty attachments do not take part in the bounds.
static NSUInteger mip_extent(NSUInteger size, NSUInteger level) {
    return level < sizeof(size) * 8 ? MAX(size >> level, 1) : 1;
}
static void pass_dims(RPD *pd, NSUInteger nc, NSUInteger *w, NSUInteger *h) {
    *w = *h = 0;
    for (NSUInteger i = 0; i < nc + 2; i++) {
        MTLRenderPassAttachmentDescriptor *a = i < nc ? pd.colorAttachments[i] : i == nc ? pd.depthAttachment : pd.stencilAttachment;
        if (!a.texture) continue;
        NSUInteger aw = mip_extent(a.texture.width, a.level), ah = mip_extent(a.texture.height, a.level);
        *w = *w ? MIN(*w, aw) : aw; *h = *h ? MIN(*h, ah) : ah;
    }
    if (!*w) *w = *h = 1;
}

// ---------- DM_NVN_PASSGPU=<file>: per-channel GPU time (stage-boundary timestamp counters) ----------
// Sample 4 points per render channel (vertex start / end, fragment start / end) and 2 points per compute channel; parse after command buffer completion and accumulate by “channel number within the frame”.
// Each line: frame idx kind gpu_ms inc_ms w h fmt0 dfmt load draws t0_ns t1_ns (kind: R render / C compute; load: L load / C clear for each color attachment + depth)
// Pipeline overlap between channels on TBDR (the vertex stage of a later channel starts before the fragment stage of the preceding channel finishes): gpu_ms = this channel's start-to-end span (including queueing), not additive;

// ---------- DM_NVN_PASSGPU=<file>: per-pass GPU time (stage-boundary timestamp counters) ----------
// Each render pass samples 4 points (vertex start/end, fragment start/end), a compute pass 2 points; resolved after the command buffer completes and accumulated by 'index of the pass within the frame'.
// Each line: frame idx kind gpu_ms inc_ms w h fmt0 dfmt load draws t0_ns t1_ns (kind: R render / C compute; load: L load / C clear for each color attachment + depth)
// On TBDR passes overlap in the pipeline (the later pass's vertex stage is earlier than the end of the earlier pass's fragment stage): gpu_ms = this pass's start-to-end span (including queueing) and cannot be summed;
// attribution uses the timeline increment inc_ms = t1 − max(t0, latest t1 of the passes completed so far); the sum of the increments of all passes = total time the GPU is busy (command buffers complete in submission order)
typedef struct { uint64_t frame, pass; int idx, kind, w, h, draws, slot; uint32_t cfmt0, dfmt; char load[12]; } pg_pass;   // pass: global pass number (same as the pass number in the DM_NVN_PASSDUMP lines, for joining the two records)
#define PG_SLOTS 4096   // Counter sample buffer limit 32768 bytes = 4096 timestamps (measured from the error on this machine)
#define PG_RING 8
static int g_pg = -1; static FILE *g_pg_f;
#if DM_NVN_M3
static id<MTLCounterSampleBuffer> g_pg_buf[PG_RING];
#define PG_RSLOTS 4   // Render pass sample points: vertex start/end, fragment start/end (stage boundary counter sampling)
#else
static id<MTL4CounterHeap> g_pg_buf[PG_RING]; static double g_pg_ns_tick = 1;   // Metal 4: timestamp counter heap; readings are GPU clock ticks, converted to nanoseconds by the device timestamp frequency
#define PG_RSLOTS 2   // Render pass sample points: after opening the encoder (before the vertex stage) and before ending it (after the fragment stage)
static int g_pg_copen = -1;   // Compute pass: end sample point of the currently open one
#endif
static int g_pg_b; static uint32_t g_pg_used;
static pg_pass *g_pg_pass; static int g_pg_np, g_pg_cap, g_pg_open = -1;
static uint64_t g_pg_frame = ~0ull; static int g_pg_idx;
static pthread_mutex_t g_pg_mu = PTHREAD_MUTEX_INITIALIZER;
static inline void args_dirty_all(void);
typedef struct { double sum, last; uint64_t n; pg_pass meta; } pg_stat;
static pg_stat g_pg_st[256]; static uint64_t g_pg_rep_frame;
static int pg_on(void) {
    if (g_pg >= 0) return g_pg;
    const char *e = getenv("DM_NVN_PASSGPU"); g_pg = 0;
    if (!e) return 0;
#if DM_NVN_M3
    if (![g_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { MLOG("[RED] DM_NVN_PASSGPU: this device does not support stage-boundary counter sampling"); return 0; }
    id<MTLCounterSet> ts = nil; for (id<MTLCounterSet> c in g_dev.counterSets) if ([c.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = c;
    if (!ts) { MLOG("[RED] DM_NVN_PASSGPU: no timestamp counter set"); return 0; }
    MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = ts; d.storageMode = MTLStorageModeShared; d.sampleCount = PG_SLOTS;
    for (int i = 0; i < PG_RING; i++) { NSError *er = nil; g_pg_buf[i] = [g_dev newCounterSampleBufferWithDescriptor:d error:&er]; if (!g_pg_buf[i]) { MLOG("[RED] Could not create counter sampling buffer: %s", er.localizedDescription.UTF8String); return 0; } }
#else
    MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new]; hd.type = MTL4CounterHeapTypeTimestamp; hd.count = PG_SLOTS;
    for (int i = 0; i < PG_RING; i++) { NSError *er = nil; g_pg_buf[i] = [g_dev newCounterHeapWithDescriptor:hd error:&er]; if (!g_pg_buf[i]) { MLOG("[RED] Could not create timestamp counter heap: %s", er.localizedDescription.UTF8String ?: "?"); return 0; } }
    { uint64_t fq = [g_dev queryTimestampFrequency]; g_pg_ns_tick = fq ? 1e9 / (double)fq : 1; }
#endif
    if (!(g_pg_f = fopen(e, "w"))) { MLOG("[RED] DM_NVN_PASSGPU cannot open %s", e); return 0; }
    fprintf(g_pg_f, "frame\tidx\tkind\tgpu_ms\tinc_ms\tw\th\tfmt0\tdfmt\tload\tdraws\tt0_ns\tt1_ns\tpass\n");
    MLOG("[passgpu] Per-channel GPU timing → %s", e);
    return g_pg = 1;
}
static void pg_sub_begin(uint64_t fence) {   // This submission uses block fence % PG_RING of the ring; first wait for the submission that used it PG_RING submissions ago to complete
    if (!pg_on()) return;
    if (fence > PG_RING) mtl_wait(fence - PG_RING, UINT64_MAX);
    g_pg_b = (int)(fence % PG_RING); g_pg_used = 0; g_pg_np = 0; g_pg_open = -1;
}
static pg_pass *pg_new(int kind, int nslot) {
    if (!pg_on() || g_pg_used + nslot > PG_SLOTS) return NULL;
    uint64_t f = atomic_load(&g_cur_frame); if (f != g_pg_frame) { g_pg_frame = f; g_pg_idx = 0; }
    if (g_pg_np == g_pg_cap) { g_pg_cap = g_pg_cap ? g_pg_cap * 2 : 128; g_pg_pass = realloc(g_pg_pass, g_pg_cap * sizeof *g_pg_pass); }
    pg_pass *p = &g_pg_pass[g_pg_np++]; memset(p, 0, sizeof *p);
    p->frame = f; p->pass = g_stat_pass; p->idx = g_pg_idx++; p->kind = kind; p->slot = (int)g_pg_used; p->draws = -(int)g_stat_draw;
    g_pg_used += nslot;
    return p;
}
// Record channel info (attachments, dimensions, load actions) before opening a render channel; legacy APIs attach stage-boundary samples to the descriptor, while Metal 4 writes timestamps after opening the encoder / before closing it (pg_render_begin / pg_render_finish).

// Record pass info before opening a render pass (attachments, size, load action); the legacy API attaches stage boundary sampling to the descriptor, while Metal 4 writes timestamps after opening / before ending the encoder (pg_render_begin / pg_render_finish).
// When moving to Metal 4 the descriptor lost its sample attachments, this whole block was short-circuited and the readback call at submission was deleted, so neither API produced numbers again; this restores it
static void pg_render(RPD *pd, NSUInteger nc) {
    pg_pass *p = pg_new(0, PG_RSLOTS); if (!p) return;
#if DM_NVN_M3
    MTLRenderPassSampleBufferAttachmentDescriptor *a = pd.sampleBufferAttachments[0];
    a.sampleBuffer = g_pg_buf[g_pg_b];
    a.startOfVertexSampleIndex = p->slot; a.endOfVertexSampleIndex = p->slot + 1; a.startOfFragmentSampleIndex = p->slot + 2; a.endOfFragmentSampleIndex = p->slot + 3;
#endif
    id<MTLTexture> t0 = nil; int k = 0;
    for (NSUInteger i = 0; i < nc && i < 8; i++) { id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        if (!t0) t0 = t; if (k < 10) p->load[k++] = pd.colorAttachments[i].loadAction == MTLLoadActionClear ? 'C' : 'L'; }
    id<MTLTexture> dt = pd.depthAttachment.texture;
    if (dt) { if (k < 10) { p->load[k++] = '|'; p->load[k++] = pd.depthAttachment.loadAction == MTLLoadActionClear ? 'C' : 'L'; } p->dfmt = (uint32_t)dt.pixelFormat; }
    if (!t0) t0 = dt;
    NSUInteger pw, ph; pass_dims(pd, nc, &pw, &ph);
    p->w = (int)pw; p->h = (int)ph; p->cfmt0 = t0 && t0 != dt ? (uint32_t)t0.pixelFormat : 0;
    g_pg_open = g_pg_np - 1;
}
static void pg_render_begin(RENC e) {   // After opening the render encoder: write the pass start point (before the vertex stage of the first draw)
#if DM_NVN_M3
    (void)e;
#else
    if (g_pg_open >= 0 && e) [e writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageVertex intoHeap:g_pg_buf[g_pg_b] atIndex:(NSUInteger)g_pg_pass[g_pg_open].slot];
#endif
}
static void pg_render_finish(RENC e) {   // Before ending the render encoder: write the pass end point (after the fragment stage of all draws)
#if DM_NVN_M3
    (void)e;
#else
    if (g_pg_open >= 0 && e) [e writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment intoHeap:g_pg_buf[g_pg_b] atIndex:(NSUInteger)g_pg_pass[g_pg_open].slot + 1];
#endif
}
static id<MTLComputeCommandEncoder> pg_compute(void) {
    pg_pass *p = pg_new(1, 2);
    ser_gate();
    if (!p) { args_dirty_all(); return enc_c_begin(NULL); }
#if !DM_NVN_M3
    p->draws = 0; args_dirty_all();
    CENC ce = enc_c_begin(NULL);   // Metal 4: write the start point after opening the encoder, write the end point before ending it (pg_compute_end)
    [ce writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:g_pg_buf[g_pg_b] atIndex:(NSUInteger)p->slot]; g_pg_copen = p->slot + 1;
    return (id<MTLComputeCommandEncoder>)ce;
#else
    MTLComputePassDescriptor *cd = [MTLComputePassDescriptor computePassDescriptor];
    cd.sampleBufferAttachments[0].sampleBuffer = g_pg_buf[g_pg_b];
    cd.sampleBufferAttachments[0].startOfEncoderSampleIndex = p->slot; cd.sampleBufferAttachments[0].endOfEncoderSampleIndex = p->slot + 1;
    p->draws = 0;
    args_dirty_all();
    return enc_c_begin(cd);
#endif
}
static void pg_compute_end(id e) {   // Called before ending the compute encoder opened by pg_compute
#if DM_NVN_M3
    (void)e;
#else
    if (g_pg_copen >= 0 && e) { [(CENC)e writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:g_pg_buf[g_pg_b] atIndex:(NSUInteger)g_pg_copen]; g_pg_copen = -1; }
#endif
}
static void pg_end_render(void) { if (g_pg_open >= 0) { g_pg_pass[g_pg_open].draws += (int)g_stat_draw; g_pg_open = -1; } }
static void pg_commit(id fb) {   // Hands this submission's pass table to the completion callback for parsing; fb = FB_ON(o, g_cb) (on the legacy API the command buffer completion handler, on Metal 4 the commit options feedback handler)
    if (!g_pg || !g_pg_np) return;
    int n = g_pg_np; pg_pass *ps = malloc(n * sizeof *ps); memcpy(ps, g_pg_pass, n * sizeof *ps);
    __typeof__(g_pg_buf[0]) buf = g_pg_buf[g_pg_b]; uint32_t used = g_pg_used;
#if !DM_NVN_M3
    double ns_tick = g_pg_ns_tick;
#endif
    [fb FB_SEL:^(FB_ARG c) {
        (void)c;
        NSData *d = [buf resolveCounterRange:NSMakeRange(0, used)];
        uint64_t *t = d ? calloc(used ? used : 1, sizeof *t) : NULL;   // Time of each sample point (nanoseconds), 0 = invalid
#if DM_NVN_M3
        const MTLCounterResultTimestamp *r = d.bytes;
        for (uint32_t k = 0; t && k < used && (k + 1) * sizeof *r <= d.length; k++) t[k] = r[k].timestamp == MTLCounterErrorValue ? 0 : r[k].timestamp;   // Measured on this machine to be ns, same as the CPU
#else
        const MTL4TimestampHeapEntry *r = d.bytes;
        for (uint32_t k = 0; t && k < used && (k + 1) * sizeof *r <= d.length; k++) t[k] = (uint64_t)(r[k].timestamp * ns_tick);
#endif
        pthread_mutex_lock(&g_pg_mu);
        for (int i = 0; t && i < n; i++) {
            pg_pass *p = &ps[i]; uint64_t a, b;
            #define OK(x) ((x) != 0)
#if DM_NVN_M3
            if (p->kind == 0) { a = OK(t[p->slot]) ? t[p->slot] : t[p->slot + 2]; b = OK(t[p->slot + 3]) ? t[p->slot + 3] : t[p->slot + 1]; }
            else { a = t[p->slot]; b = t[p->slot + 1]; }
#else
            a = t[p->slot]; b = t[p->slot + 1];
#endif
            if (!OK(a) || !OK(b) || b < a) continue;
            #undef OK
            static uint64_t prev_end;   // Latest end time of the passes completed so far (completion callbacks are serialized in submission order)
            double ms = (b - a) / 1e6;   // Timestamps are in ns, same as the CPU (measured 1:1 via sampleTimestamps on this machine)
            double inc = b > prev_end ? (b - (a > prev_end ? a : prev_end)) / 1e6 : 0;
            if (b > prev_end) prev_end = b;
            fprintf(g_pg_f, "%llu\t%d\t%c\t%.4f\t%.4f\t%d\t%d\t%u\t%u\t%s\t%d\t%llu\t%llu\t%llu\n", (unsigned long long)p->frame, p->idx, p->kind ? 'C' : 'R', ms, inc,
                    p->w, p->h, p->cfmt0, p->dfmt, p->load, p->draws, (unsigned long long)a, (unsigned long long)b, (unsigned long long)p->pass);
            if (p->idx < 256 && p->frame > 300) { pg_stat *st = &g_pg_st[p->idx]; st->sum += inc; st->n++; st->last = inc; st->meta = *p; }
            if (p->frame >= g_pg_rep_frame + 300 && p->frame > 300) {   // Report the top five every 300 frames (accumulated from frame 300)
                g_pg_rep_frame = p->frame; double tot = 0; int top[5] = {-1, -1, -1, -1, -1};
                for (int k = 0; k < 256; k++) if (g_pg_st[k].n) { double m = g_pg_st[k].sum / g_pg_st[k].n; tot += m;
                    for (int j = 0; j < 5; j++) if (top[j] < 0 || m > g_pg_st[top[j]].sum / g_pg_st[top[j]].n) { for (int q = 4; q > j; q--) top[q] = top[q - 1]; top[j] = k; break; } }
                char line[1024]; size_t o = 0;
                for (int j = 0; j < 5 && top[j] >= 0; j++) { pg_stat *t = &g_pg_st[top[j]];
                    o += snprintf(line + o, sizeof line - o, " #%d %.2f ms %dx%d fmt%u d%u %s %d draws; ", top[j], t->sum / t->n, t->meta.w, t->meta.h, t->meta.cfmt0, t->meta.dfmt, t->meta.load, t->meta.draws); }
                MLOG("[passgpu] Through frame %llu, total average channel time per frame %.2f ms; top five: %s", (unsigned long long)p->frame, tot, line);
                fflush(g_pg_f);
            }
        }
        pthread_mutex_unlock(&g_pg_mu);
        free(ps);
    }];
}
void ts_after(void);
// ---------- Writeback lookahead: at channel end, use subsequent commands in the same submission to determine whether writeback of each attachment is redundant ----------
// On TBDR, each attachment is written back to memory as a whole at the end; Switch (direct rendering on Maxwell) has no such overhead. If an attachment is fully cleared before being read, writeback is redundant (DontCare).

// ---------- Store lookahead: at pass end, decide from the later commands of the same submission whether each attachment's write-back is redundant ----------
// On TBDR every attachment is written back to memory in full when the pass ends; Switch (Maxwell immediate-mode rendering) has no such cost. If an attachment is cleared in full before being read, the write-back is redundant (DontCare).
// For the decision see mtl_sub_index / sdx_decide; only what can be proven within the same submission is accepted, and anything still undecided at the end of the submission is always written back.
typedef struct { uint64_t ser, bytes; uint32_t lv, sl; int idx; uint8_t depth, state, ridx, drawn, why, load, wrote, wwhy; const char *wkey; int wslot; } sd_att;   // depth: 0 color 1 depth 2 stencil (the two parts of a combined depth-stencil texture are decided separately)   // state 0 undecided 1 needed 2 redundant; idx: color attachment index, -1 for depth
static sd_att g_sd[9]; static int g_nsd;

// Sub-item switch g_tbdr_sub (bitmask, registered in the variable table for in-process alternating measurement): with the master switch on, disable one item at a time to measure each item's translation CPU cost and GPU savings, and find net-loss items. Likewise latched at the start of each submit
enum { TB_SDX = 1, TB_HOIST = 2, TB_DEFER = 4, TB_FUSE = 8, TB_DS = 16, TB_UDC = 32, TB_ALL = 63 };   // Write-back lookahead (per-submit prescan index + cross-submit ledger) / pure-clear pass hoisting / same-target reuse / readable tile merging / depth-stencil attached on demand / unwritten attachments not written back
int g_tbdr_sub = TB_ALL; static int g_tbdr_sub_eff = TB_ALL;
static int g_udc_ab = -1, g_udc_ab_on;
// Change writeback to DontCare for attachments not written by this channel (depth / stencil test only, color write mask 0, etc.): the contents are already in memory, so this avoids a whole-attachment writeback.
// On by default: on Mac, screenshots enabled / disabled in the same process and the empty control have the same distribution; on A19 Pro, readback after private Depth32F_S8 / RGBA8 DontCare differs at 0.

// Attachments this pass did not write (depth / stencil test only, color write mask 0, etc.) get their write-back changed to DontCare: the content is already in memory, saving one full write-back.
// On by default: on Mac, same-process on / off screenshots follow the same distribution as the null control; on A19 Pro, private Depth32F_S8 / RGBA8 read back after DontCare differ in 0 places.
// DM_NVN_UNWRITTEN_DC=0 turns it off (for equivalence comparison)
static int udc_on(void) {
    if (g_udc_ab < 0) { const char *e = getenv("DM_NVN_UNWRITTEN_DC"); g_udc_ab = !(e && !strcmp(e, "0"));
        const char *a = getenv("DM_NVN_UDC_AB"); if (a && !strcmp(a, "1")) { g_udc_ab = 0; g_udc_ab_on = 0; } }
    return (g_udc_ab || g_udc_ab_on) && (g_tbdr_sub_eff & TB_UDC);
}
// Diagnostic DM_NVN_TBDR=0: disable all TBDR optimizations (writeback lookahead, skippable loads, same-target continuation, pure-clear-channel preconditions, on-demand depth/stencil attachment), and replay via the simplest “attach all, load all, store all” path.
// For equivalent-result comparisons only (compare frames byte-for-byte with on / off in the same replay session); not a product configuration
// Runtime master switch g_tbdr_rt (registered in the variable table for alternating measurements in the same process): alternate on / off on real hardware to measure the net translated CPU cost and GPU / total-system power saved by this mechanism.

// Diagnostic DM_NVN_TBDR=0: turns off all TBDR optimizations (writeback lookahead, load elision, same-target continuation, clear-only pass precondition, on-demand depth stencil attachment); replay takes the plainest "attach all, load all, store all" path.
// Only for equivalence comparison (same replay run on / off, comparing frames byte for byte), not a product configuration
// Runtime master switch g_tbdr_rt (registered in the variable table of in-process alternating measurement): toggled on / off on a real device to measure the net of translation CPU spent vs GPU / whole-device watts saved by this mechanism.
// Latched only at the start of each submission (g_tbdr_eff) and constant within a submission; when the latched value changes, the cross-submission lookahead state (writeback check ledger) is cleared; the all-zero set is already cleared at the start of each submission
int g_tbdr_rt = 1; static int g_tbdr_eff = 1;
static int tbdr_env(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_TBDR"), *r = getenv("DM_NVN_RULER"); v = !(e && !strcmp(e, "0")); g_ruler_truth = r && !strcmp(r, "1"); if (!v) MLOG("DM_NVN_TBDR=0: all TBDR optimizations disabled (for equivalent comparison)"); } return v; }
static int tbdr_on(void) { return tbdr_env() && g_tbdr_eff && !g_ruler_truth; }
static int tb_sub(int bit) { return tbdr_on() && (g_tbdr_sub_eff & bit); }
int dm_nvn_ruler_truth(void) { tbdr_on(); return g_ruler_truth; }   // Visual ruler ground-truth phase (host-side settings under test yield accordingly, e.g. the 3D render scale)
static int rt_range_on(void) { static int v = -1; if (v < 0) { const char *e=getenv("DM_NVN_RT_RANGE"); v=e && !strcmp(e,"1"); } return v; }   // Narrow candidate, off by default
static uint64_t g_sd_dead_bytes, g_sd_n_dead, g_sd_n_all;
// ---------- Cross-submission attachment validity ledger ----------
// Entry = subresource (texture index, mip, layer) + state, persisted across submissions:
// XV_INVALID       Contents not written back / not generated; must not be read. The next access must be a full Clear, or use as an attachment with load action DontCare / Clear; otherwise it is a violation.
// XV_PENDING_CLEAR Reserved for “pure clear pending across submissions” (not implemented): accepts only full-surface, unclipped clears with all write masks enabled; clearing again while pending = replace the clear value without executing it;
// Use as an attachment with load action DontCare / Clear → cancel pending clear; load action Load → convert to Clear load; track depth and stencil planes separately.
// Violation = report error + permanently disable that stable key (format / width / height / mip) (in memory + persisted to DM_XV_FILE, loaded on next startup) + delete all entries for that texture. This fallback can only detect, not recover (that frame's attachment contains old contents).
// First use of this: do not write back color attachments “still unresolved at the end” of a submission (sd_finish)—conditions: its first touch in this submission is a full clear / DontCare load (reset before use each frame), it is not a window texture, and the key is not disabled.
// Disable with DM_NVN_XV=0; inactive when tbdr_on()==0 (visual reference truth mode).
// Trial period (XV_PROBE): for the first XV_TRIAL occurrences of a key, only “shadow-account” for it—write back as usual, but also track its next touch; only begin truly skipping writeback after XV_TRIAL consecutive compliant touches.

// ---------- Cross-submission attachment validity ledger ----------
// Entry = subresource (texture serial, level, layer) + state, surviving across submissions:
//   XV_INVALID       content was not written back / not generated and must not be read. The next touch must be a full Clear, or use as an attachment with load action DontCare / Clear; otherwise it is a violation.
//   XV_PENDING_CLEAR reserved for 'clear-only pending across submissions' (not implemented): accepts only full, unscissored clears with the write mask fully open; clearing again while pending = replace the clear value without landing it;
//                    as an attachment with load DontCare / Clear → the pending clear is voided, with load Load → changed to a Clear load; depth-stencil is recorded per plane.
// Violation = report a failure + that stable key (format / width / height / level) permanently disabled (in memory + persisted to DM_XV_FILE, read at the next launch) + all entries of the same texture deleted. The fallback can only detect, not undo (in that frame the attachment holds old content).
// Its first use: color attachments 'scanned to the end undecided' at the end of a submission are not written back (sd_finish) - conditions: its first touch in this submission is a full clear / DontCare load (reset every frame before use), it is not a window texture, and the key is not disabled.
// Switch DM_NVN_XV=0 turns it off; has no effect when tbdr_on()==0 (visual ruler ground-truth mode).
// Trial period (XV_PROBE): for a key's first XV_TRIAL times it is only 'shadow-accounted' - written back as usual, but its next touch is tracked all the same; only after XV_TRIAL consecutive compliant touches does the write-back really start being skipped.
//   On a read / Load during the trial period: only disable that key (within this process, not persisted, no failure), the picture is intact - in practice there are readers as early as the launch / menu screens (upscale target 36), and not every fresh install should get one wrong frame.
enum { XV_INVALID = 1, XV_PENDING_CLEAR = 2, XV_PROBE = 3 };
#define XV_TRIAL 3
typedef struct { uint64_t ser, tag; uint32_t lv, sl; uint8_t state; } xv_ent;
static xv_ent g_xv[64]; static int g_xv_n, g_xv_block, g_xv_init;
static void tbdr_latch(void) {   // At the start of each submission: latch the TBDR runtime switch, and clear the cross-submission writeback check ledger if it changed
    if (g_tbdr_sub != g_tbdr_sub_eff) { if ((g_tbdr_sub ^ g_tbdr_sub_eff) & TB_SDX) g_xv_n = 0; g_tbdr_sub_eff = g_tbdr_sub; static int n; if (n++ < 4) MLOG("TBDR subfeature switches → 0x%x (starting at submission %llu)", g_tbdr_sub_eff, (unsigned long long)g_sub_serial); }
    if ((g_tbdr_rt != 0) == g_tbdr_eff) return;
    g_tbdr_eff = g_tbdr_rt != 0; g_xv_n = 0; static int n; if (n++ < 4) MLOG("TBDR runtime switch → %d (starting at submission %llu)", g_tbdr_eff, (unsigned long long)g_sub_serial);
}
static uint64_t g_xv_disabled[128]; static int g_xv_ndis;
static struct { uint64_t tag; int good; } g_xv_trial[128]; static int g_xv_ntrial;   // Number of consecutive compliant touches for each key
static int xv_trial_get(uint64_t tag, int add) { for (int i = 0; i < g_xv_ntrial; i++) if (g_xv_trial[i].tag == tag) return g_xv_trial[i].good += add; if (g_xv_ntrial < 128) { g_xv_trial[g_xv_ntrial].tag = tag; return g_xv_trial[g_xv_ntrial++].good = add; } return 0; }
static uint64_t g_xv_present[128]; static int g_xv_npresent;
static uint64_t g_xv_dc, g_xv_viol;
typedef struct { uint64_t ser; uint32_t lv, sl; uint8_t good; } xv_touch;
static xv_touch g_xv_touch[128]; static int g_xv_ntouch;
static uint64_t xv_tag(const tex_t *t, uint32_t lv) { return ((uint64_t)t->d.format << 40) ^ ((uint64_t)t->d.width << 24) ^ ((uint64_t)t->d.height << 8) ^ lv; }

// Stable key (unchanged across launches, does not use the texture serial): format / width / height / level, then mixed with the formats of the color attachments of the pass where the write-back is skipped and this attachment's index - textures of the same format and size but different purpose (e.g. transmittance 27 and upscale target 36) do not share a key
static uint64_t xv_tag_pass(const tex_t *t, uint32_t lv, int idx, const p_set_rt *rt) { uint64_t h = xv_tag(t, lv) ^ ((uint64_t)(idx + 1) << 60); for (int i = 0; i < rt->n && i < 8; i++) h = (h ^ (rt->color[i] ? (uint64_t)rt->color[i]->d.format + 1 : 0)) * 0x100000001b3ull; return h; }
static void xv_init(void) {
    if (g_xv_init) return; g_xv_init = 1;
    const char *p = getenv("DM_XV_FILE"); if (!p || !*p) return;
    FILE *f = fopen(p, "r"); if (!f) return;
    char line[256], why[160]; unsigned long long tag; int ch, n = 0;
    while (fgets(line, sizeof line, f) && g_xv_ndis < 128) if (sscanf(line, "%llx\t%159[^\t]\t%d", &tag, why, &ch) >= 1) g_xv_disabled[g_xv_ndis++] = tag, n++;
    fclose(f); MLOG("Read %d disable records", n);
}
static int xv_disabled(uint64_t tag) { xv_init(); for (int i = 0; i < g_xv_ndis; i++) if (g_xv_disabled[i] == tag) return 1; return 0; }
static xv_ent *xv_find(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_n; i++) if (g_xv[i].ser == ser && g_xv[i].lv == lv && g_xv[i].sl == sl) return &g_xv[i]; return NULL; }
static void xv_drop(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_n; i++) if (g_xv[i].ser == ser && g_xv[i].lv == lv && g_xv[i].sl == sl) { g_xv[i] = g_xv[--g_xv_n]; return; } }
static void xv_drop_all_of(uint64_t ser) { for (int i = 0; i < g_xv_n;) if (g_xv[i].ser == ser) g_xv[i] = g_xv[--g_xv_n]; else i++; }
static void xv_set(uint64_t ser, uint32_t lv, uint32_t sl, uint8_t state, uint64_t tag) {
    xv_init(); if (g_xv_block || xv_disabled(tag)) return;
    xv_ent *e = xv_find(ser, lv, sl); if (e) { e->state = state; e->tag = tag; return; }
    if (g_xv_n < 64) g_xv[g_xv_n++] = (xv_ent){ser, tag, lv, sl, state};
}
static void xv_violation(const xv_ent *e, const char *why) {
    xv_init();
    if (e->state == XV_PROBE) {   // Trial period: the content is still there and the picture is intact - only disable the key (this process)
        if (!xv_disabled(e->tag) && g_xv_ndis < 128) g_xv_disabled[g_xv_ndis++] = e->tag;
        PD("V xv trial-period witness: key=%016llx texture=%llu level=%u reason=%s → do not process this key", (unsigned long long)e->tag, (unsigned long long)e->ser, e->lv, why);
        xv_drop_all_of(e->ser); return; }
    g_xv_viol++;
    int seen = xv_disabled(e->tag); if (!seen && g_xv_ndis < 128) g_xv_disabled[g_xv_ndis++] = e->tag;
    MLOG("[RED]xv tag=%016llx texture=%llu level=%u layer=%u reason=%s channel=%llu", (unsigned long long)e->tag, (unsigned long long)e->ser, e->lv, e->sl, why, (unsigned long long)g_stat_pass);
    const char *p = getenv("DM_XV_FILE"); if (p && *p) { FILE *f = fopen(p, "a"); if (f) { fprintf(f, "%016llx\t%s\t%llu\n", (unsigned long long)e->tag, why, (unsigned long long)g_stat_pass); fclose(f); } }
    xv_drop_all_of(e->ser);
}
static void xv_check(uint64_t ser, uint32_t lv, uint32_t sl, const char *why) { xv_ent *e = xv_find(ser, lv, sl); if (e && (e->state == XV_INVALID || e->state == XV_PROBE)) { xv_ent copy = *e; xv_violation(&copy, why); } }
static void xv_good(xv_ent *e) { if (e->state == XV_PROBE) xv_trial_get(e->tag, 1); xv_drop(e->ser, e->lv, e->sl); }   // Compliant touch: counts once toward the trial period, and the entry returns to valid
static void xv_touch_add(uint64_t ser, uint32_t lv, uint32_t sl, int good) { for (int i = 0; i < g_xv_ntouch; i++) if (g_xv_touch[i].ser == ser && g_xv_touch[i].lv == lv && g_xv_touch[i].sl == sl) return; if (g_xv_ntouch < 128) g_xv_touch[g_xv_ntouch++] = (xv_touch){ser, lv, sl, (uint8_t)good}; }
static int xv_first_good(uint64_t ser, uint32_t lv, uint32_t sl) { for (int i = 0; i < g_xv_ntouch; i++) if (g_xv_touch[i].ser == ser && g_xv_touch[i].lv == lv && g_xv_touch[i].sl == sl) return g_xv_touch[i].good; return 0; }
static int xv_present(uint64_t ser) { for (int i = 0; i < g_xv_npresent; i++) if (g_xv_present[i] == ser) return 1; return 0; }
static void xv_forward(void);
static int tbdr_on(void);
static int xv_on(void) { static int off = -1; if (off < 0) { const char *e = getenv("DM_NVN_XV"); off = e && !strcmp(e, "0"); } return !off && tbdr_on(); }
void mtl_xv_window_tex(uint64_t serial) { if (!xv_present(serial) && g_xv_npresent < 128) g_xv_present[g_xv_npresent++] = serial; }   // Window (swapchain) texture: never enters the ledger
void xv_test_counts(int *dc, int *viol) { if (dc) *dc = (int)g_xv_dc; if (viol) *viol = (int)g_xv_viol; }
static void sd_finish(void);
// Skippable load (full-coverage first draw): if the draw that opens the channel can be proven to cover an entire color attachment (first_draw_covers), change that attachment's load action to DontCare—

// Load elision (fully covering first draw): if the draw that opens the pass can be proven to cover a color attachment in full (first_draw_covers), that attachment's load becomes DontCare -
// the first draw does not blend and writes all channels, every covered pixel is rewritten by it, and the original content is not visible (Switch immediate-mode rendering has no such cost; on TBDR load = reading the whole thing from memory into the tile)
static const p_draw *g_fd_dw; static const p_draw_idx *g_fd_di;   // The draw that opens the pass (one of non-indexed / indexed); non-null only while draw() is opening a pass
static uint64_t g_fd_n;         // Cumulative number of attachments judged elidable
static int g_fd_why;            // Reason for the most recent failure (Q line of PASSDUMP: 1 no program 2 shader 3 attachment 4 blend/mask/raster 5 depth-stencil test 6 vertex count 7 position attribute 8 format/stream 9 out of bounds 10 address 11 geometry)
static int first_draw_covers(int i, id<MTLTexture> t, NSUInteger level);
static int pf_is_float(MTLPixelFormat f) {   // Float color formats that can store NaN (sentinel choice for the load-elision probe; RGB9E5 cannot store NaN and does not count)
    switch (f) { case MTLPixelFormatR16Float: case MTLPixelFormatRG16Float: case MTLPixelFormatRGBA16Float: case MTLPixelFormatR32Float:
        case MTLPixelFormatRG32Float: case MTLPixelFormatRGBA32Float: case MTLPixelFormatRG11B10Float: return 1; default: return 0; }
}
static int fz_load_covers(int i, id<MTLTexture> t, NSUInteger level);
static NSMutableArray *g_lp;    // DM_NVN_LOADPROBE=1 / 2 / 3 (two constant sentinel values / NaN - under float formats normal shader output cannot reach NaN, ruling out numeric coincidence): attachments judged elidable are cleared to the sentinel color instead, and at pass end the remaining sentinel pixels are read back and counted (diagnostic: should be 0)
static void lp_finish(void);
// Same-target continuation: when the game sets the same render-target set again (with no copy / dispatch / clear in between), do not end the channel—saves one whole-attachment writeback + load on TBDR.
// Basis for equivalence: NVN requires a Barrier between two channels so the latter sees all writes from the former. Within the same Metal render encoder, subsequent draws' reads and writes to the attachments themselves (blending, depth / stencil tests)
// are ordered with earlier draws according to submission order; however, “sampling” an attachment texture is not included (reading and writing the same attachment within one channel is undefined). Therefore, continue only when subsequent draws do not sample these attachments:

// Same-target continuation: when the game calls SetRenderTargets again on the same set of targets (with no copy / dispatch / clear in between) the pass is not ended - on TBDR this saves one full write-back + load.
// Basis for equivalence: NVN's Barrier between two passes requires the latter to see all writes of the former. Within one Metal render encoder, later draws' reads and writes of the attachment itself (blend, depth / stencil test)
// are ordered with earlier draws in submission order; but 'sampling' the attachment texture is not covered (reading and writing the same attachment within one pass is undefined). So continuation happens only when later draws do not sample these attachments:
// if during continuation a draw needs to sample them, or a clear arrives, the pass is ended before it, returning to the separate semantics. Copies / dispatches end the render encoder anyway
static int g_enc_mode;          // How the current encoder was opened (rp_begin_ex argument: 0 all attachments 1 only for clears 2 depth omitted)
static int g_rt_range_active, g_rt_range_pending;
static int g_rt_merged;         // The current pass has already been continued (at least one same-target SetRenderTargets was merged in)
static uint64_t g_rt_merge_n;   // Cumulative continuation count
static int g_defer_replaying;
static void fz_end(void);   // Wrap-up of the readable-tile pass merge (defined before draw)
static void defer_fallback(void);
// DM_NVN_ENC_LABEL=1: label each render encoder “p<render channel number within frame> <width>x<height> <first 8 chars of first draw's fragment key> d<draw count>”, in Instruments (xctrace Game Performance / Metal System Trace)

// DM_NVN_ENC_LABEL=1: label every render encoder with 'p<index of the render pass within the frame> <width>x<height> <first 8 digits of the first draw's fragment key> d<draw count>'; in Instruments (xctrace Game Performance / Metal System Trace)
// the encoder track and the GPU interval table carry this name directly, and shader_map.tsv maps it back to the shader; off by default (one string allocation per pass)
static int g_el_w, g_el_h, g_el_draws; static char g_el_key[12];
static int enc_label_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_NVN_ENC_LABEL"); v = e && atoi(e); } return v; }
static void ghost_flush(void);
static void rp_end(void) { if (g_enc) {
    ghost_flush();
    if (enc_label_on()) { static uint64_t lf = ~0ull, base; uint64_t f = atomic_load(&g_cur_frame); if (f != lf) { lf = f; base = g_stat_pass - 1; }   // Intra-frame serial: the same pass has the same name every frame, so the table can aggregate by name
        [(id)g_enc setLabel:[NSString stringWithFormat:@"p%02llu %dx%d %s d%d", (unsigned long long)(g_stat_pass - base), g_el_w, g_el_h, g_el_key[0] ? g_el_key : "-", g_el_draws]]; }
#ifdef NVN_RT_RANGE_TEST
    int ranged=g_rt_range_active && !g_rt_range_pending;
#endif
    sd_finish(); pg_render_finish(g_enc); enc_r_end(g_enc); zskip_commit_clears(); g_enc = nil; g_rt_merged = 0; g_rt_range_active = g_rt_range_pending = 0; pg_end_render(); ts_after(); if (g_lp.count) lp_finish(); fz_end();
#ifdef NVN_RT_RANGE_TEST
    rr_finish(ranged);
#endif
} }
void pg_sub_begin_ext(uint64_t fence) { pg_sub_begin(fence); }


// The format a colour attachment is actually attached with: the view's format if the view changes the format and it can be attached, otherwise the base texture's format (shared by attaching and pipeline prediction, so both agree). warn: log red when it cannot be attached
static MTLPixelFormat rt_color_fmt(id<MTLTexture> base, const view_t *v, int has_v, int warn) {
    if (!base || !has_v || !v->set_format) return base.pixelFormat;
    uint32_t pf = 0; uint8_t swz[4];
    if (!nvn_fmt_to_mtl(v->format, &pf, swz) || !pf) return base.pixelFormat;
    if (pf != (uint32_t)base.pixelFormat && tex_need_pfv(base.pixelFormat, (MTLPixelFormat)pf) && !(base.usage & MTLTextureUsagePixelFormatView)) {
        if (warn) RED_ONCE(48, "Could not attach render-target view format %#x (base image %lu lacks mutable-format usage); rendering with base image format (first occurrence)", v->format, (unsigned long)base.pixelFormat);
        return base.pixelFormat; }
    return (MTLPixelFormat)pf;
}
// color: if the color attachment view format changes (e.g. writing to an RGBA8_SRGB view of an RGBA8 base texture), attach a Metal view with that format—on Switch, writes and clears are encoded according to the view format,
// With only the base texture attached, the sRGB encoding is dropped entirely and the image appears dark (public repo #36, tested by seedds in another game); the pipeline color format is taken from the attached texture format (g_pass.cfmt), keeping them consistent.

// color: when the view of a colour attachment changes the format (e.g. an RGBA8 base texture written through an RGBA8_SRGB view), attach a Metal view of that format - on the Switch writes and clears are encoded by the view's format,
// and attaching only the base texture drops the sRGB encoding entirely, making the picture too dark (public repository #36, measured by seedds on another game); the pipeline colour formats take the attached texture's format (g_pass.cfmt), so they follow.
// The view lives only until this submission completes (Metal 4 does not retain resources; the view shares storage with the base texture and is not made resident separately). Depth / stencil attachments do not change format
static void attach(MTLRenderPassAttachmentDescriptor *a, const tex_t *t, const view_t *v, int has_v, int color) {
    id<MTLTexture> base = tex_mtl(t); a.texture = base;
    if (has_v && v->set_levels) a.level = v->base_level;
    if (has_v && v->set_layers) { if (t->d.target == 2) a.depthPlane = v->min_layer; else a.slice = v->min_layer; }
    if (!color || !base) return;
    MTLPixelFormat pf = rt_color_fmt(base, v, has_v, 1);
    if (pf == base.pixelFormat) return;
    id<MTLTexture> vw = [base newTextureViewWithPixelFormat:pf];
    if (!vw) { RED_ONCE(49, "Could not create render-target view: format %#x (base image %lu); rendering with base image format (first occurrence)", v->format, (unsigned long)base.pixelFormat); return; }
    a.texture = vw; defer_obj_unref(vw);
    { static int once; if (!once++) MLOG("Attached render target using mutable-format view: view format %#x, base image %lu → attached %lu (first occurrence)", v->format, (unsigned long)base.pixelFormat, (unsigned long)pf); }
}
// mode 1 (clear_only): a pass opened only to apply pending clears; attach only attachments with clears. mode 2: look ahead to prove this pass does not use depth/stencil, and omit depth.
// clear_only: a pass opened only to apply pending clears; attach only attachments with clears—the pass does not touch the other attachments, and attaching them would cause needless reads and writes on TBDR.

// mode 1 (clear_only): a pass opened only to land pending clears, attaching only the attachments that have clears; mode 2: the lookahead proves this pass does not use depth-stencil, so depth is not attached.
// clear_only: a pass opened only to land pending clears, attaching only the attachments that have clears - the other attachments are not touched by this pass, and attaching them costs a pointless load and store on TBDR
// (Mac character select screen: the color-clear pass drags along a 400x225 depth load and store)
static RENC rp_begin_ex(int clear_only);
static RENC rp_begin(void) { return rp_begin_ex(0); }
static RENC rp_begin_ex(int clear_only) {
    if (g_enc) return g_enc;
    memset(g_zskip_cand_ok, 0, sizeof g_zskip_cand_ok);
    g_enc_mode = clear_only;
    if (!R.has_rt) { RED_ONCE(3, "No render target set before draw"); return nil; }

    // The version of the attachment-range candidate is taken before capturing any attachment; if it changes during setup, there is no continuation.
    uint64_t reg0 = 0, ser0 = 0, dirty0 = 0;
    if (rt_range_on()) { reg0 = g_reg_gen; ser0 = g_ser_gen; }
    if (rt_range_on()) dirty0=nvn_dirty_now();
    if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    RPD *pd = [RPD new];
    int any = 0;
    for (int i = 0; i < R.rt.n; i++) {
        if (!R.rt.color[i] || (clear_only == 1 && !R.clear_c[i])) continue;
        attach(pd.colorAttachments[i], R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i], 1);
        if (!pd.colorAttachments[i].texture) continue;
        any = 1;
        pd.colorAttachments[i].loadAction = R.clear_c[i] ? MTLLoadActionClear : MTLLoadActionLoad;
        pd.colorAttachments[i].clearColor = R.cc[i];
        int cov = tbdr_on() && !R.clear_c[i] && (g_fd_dw || g_fd_di) && fz_load_covers(i, pd.colorAttachments[i].texture, pd.colorAttachments[i].level);
        if (!R.clear_c[i] && (g_fd_dw || g_fd_di) && !cov) PD("Q %llu c%d readback not elidable why=%d", g_stat_pass + 1, i, g_fd_why);
        if (cov) {
            static int probe = -1; if (probe < 0) { probe = getenv("DM_NVN_LOADPROBE") ? atoi(getenv("DM_NVN_LOADPROBE")) : 0; if (probe) MLOG("DM_NVN_LOADPROBE: clearing elidable readback attachments to sentinel color and verifying by reading back"); }
            if (probe) { pd.colorAttachments[i].loadAction = MTLLoadActionClear; int pm = probe == 1 && pf_is_float(pd.colorAttachments[i].texture.pixelFormat) ? 3 : probe;   // Float formats use a NaN sentinel by default: a fixed sentinel value can be hit by real output (it has been hit in practice)
                pd.colorAttachments[i].clearColor = pm == 3 ? MTLClearColorMake(NAN, NAN, NAN, NAN) : probe == 2 ? MTLClearColorMake(0.777, 0.222, 0.555, 0.888) : MTLClearColorMake(0.123, 0.456, 0.789, 0.321);
                if (!g_lp) g_lp = [NSMutableArray array];
                [g_lp addObject:@[pd.colorAttachments[i].texture, @(pd.colorAttachments[i].level), @(pd.colorAttachments[i].slice), @(g_stat_pass + 1), @(i), @(R.rt.color[i]->serial), @(pm)]]; }
            else pd.colorAttachments[i].loadAction = MTLLoadActionDontCare;
            g_fd_n++;
        }
        xv_touch_add(R.rt.color[i]->serial, (uint32_t)pd.colorAttachments[i].level, (uint32_t)pd.colorAttachments[i].slice,
                     pd.colorAttachments[i].loadAction != MTLLoadActionLoad);
        if (g_xv_n) { xv_ent *xe = xv_find(R.rt.color[i]->serial, (uint32_t)pd.colorAttachments[i].level, (uint32_t)pd.colorAttachments[i].slice);
            if (xe && (xe->state == XV_INVALID || xe->state == XV_PROBE)) { if (pd.colorAttachments[i].loadAction == MTLLoadActionLoad) { xv_ent copy = *xe; xv_violation(&copy, "Read old contents as attachment"); } else xv_good(xe); } }
        pd.colorAttachments[i].storeAction = MTLStoreActionStore;
    }

    // clear_only: 0 attach all 1 only for clears 2 neither depth nor stencil attached 3 stencil only 4 depth only (3 / 4: the pass_ds_need lookahead proves the whole pass uses only that one)
    if (R.rt.depth && clear_only == 3) {   // Stencil only: the depth plane of the combined format does not enter the tile and is neither read nor written (Metal allows an empty depth attachment with the combined texture on the stencil attachment)
        attach(pd.stencilAttachment, R.rt.depth, &R.rt.dview, R.rt.has_dview, 0);
        MTLPixelFormat pf = pd.stencilAttachment.texture.pixelFormat;
        if (pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatStencil8) {
            any = 1; pd.stencilAttachment.loadAction = R.clear_s ? MTLLoadActionClear : MTLLoadActionLoad;
            pd.stencilAttachment.clearStencil = R.cs; pd.stencilAttachment.storeAction = MTLStoreActionStore;
        } else pd.stencilAttachment.texture = nil;   // No stencil plane: the stencil test always passes, attach nothing
    } else if (R.rt.depth && clear_only != 2 && !(clear_only == 1 && !R.clear_d && !R.clear_s)) {
        attach(pd.depthAttachment, R.rt.depth, &R.rt.dview, R.rt.has_dview, 0);
        if (pd.depthAttachment.texture) {
            any = 1;
            pd.depthAttachment.loadAction = R.clear_d ? MTLLoadActionClear : MTLLoadActionLoad;
            pd.depthAttachment.clearDepth = R.cd; pd.depthAttachment.storeAction = MTLStoreActionStore;
            MTLPixelFormat pf = pd.depthAttachment.texture.pixelFormat;
            if (clear_only != 4 && (pf == MTLPixelFormatDepth32Float_Stencil8 || pf == MTLPixelFormatStencil8)) {   // 4: depth only
                pd.stencilAttachment.texture = pd.depthAttachment.texture; pd.stencilAttachment.level = pd.depthAttachment.level;
                pd.stencilAttachment.slice = pd.depthAttachment.slice;
                pd.stencilAttachment.loadAction = R.clear_s ? MTLLoadActionClear : MTLLoadActionLoad;
                pd.stencilAttachment.clearStencil = R.cs; pd.stencilAttachment.storeAction = MTLStoreActionStore;
            }
        }
    }
    if (!any) { RED_ONCE(4, "Could not create MTLTexture for any render target"); return nil; }
    for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i] && zskip_zero_clear(i)) {
        g_zskip_cand_ok[i] = 1; g_zskip_cand[i] = (zskip_zero){R.rt.color[i]->serial, (uint16_t)(R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0), (uint16_t)(R.rt.has_cview[i] && R.rt.cview[i].set_layers ? R.rt.cview[i].min_layer : 0)};
    }

    // Bandwidth ledger: attachments with Load count as load, all attachments count as store (currently always Store)
    int nload = 0, natt = 0;
    for (int i = 0; i < R.rt.n; i++) {
        MTLRenderPassColorAttachmentDescriptor *c = pd.colorAttachments[i];
        if (!c.texture) continue;
        uint64_t b = att_bytes(c.texture, c.level); natt++; g_bw_store += b; if (c.loadAction == MTLLoadActionLoad) { g_bw_load += b; nload++; }
        if (R.rt.has_cview[i] && R.rt.cview[i].set_format) PD("V %llu c%d view format %#x (base texture NVN format %#x)", g_stat_pass + 1, i, R.rt.cview[i].format, R.rt.color[i]->d.format);
        PD("P %llu c%d ser=%llu fmt=%lu %lux%lu ms%lu lv%lu sl%lu %s", g_stat_pass + 1, i, R.rt.color[i]->serial, (unsigned long)c.texture.pixelFormat,
           (unsigned long)c.texture.width, (unsigned long)c.texture.height, (unsigned long)c.texture.sampleCount, (unsigned long)c.level, (unsigned long)c.slice,
           c.loadAction == MTLLoadActionClear ? "clear" : c.loadAction == MTLLoadActionDontCare ? "dontcare" : "load");
    }
    {   // Depth and stencil planes are counted separately (combined format 5 bytes = depth 4 + stencil 1; when only one is attached only that one is counted)
        id<MTLTexture> dt = pd.depthAttachment.texture, stt = pd.stencilAttachment.texture;
        uint64_t whole = att_bytes(dt ? dt : stt, dt ? pd.depthAttachment.level : pd.stencilAttachment.level);
        int comb = (dt ? dt : stt).pixelFormat == MTLPixelFormatDepth32Float_Stencil8;
        uint64_t db = dt ? (comb ? whole / 5 * 4 : whole) : 0, sb = stt && stt != dt ? (comb ? whole / 5 : whole) : stt ? whole - db : 0;
        if (dt) { natt++; g_bw_store += db; if (pd.depthAttachment.loadAction == MTLLoadActionLoad) { g_bw_load += db; nload++; } }
        if (stt) { natt += !dt; g_bw_store += sb; if (pd.stencilAttachment.loadAction == MTLLoadActionLoad) { g_bw_load += sb; nload += !dt; } }
        if (dt) PD("P %llu d ser=%llu fmt=%lu %lux%lu %s", g_stat_pass + 1, R.rt.depth->serial, (unsigned long)dt.pixelFormat,
                   (unsigned long)dt.width, (unsigned long)dt.height, pd.depthAttachment.loadAction == MTLLoadActionClear ? "clear" : "load");
        else if (stt) PD("P %llu s ser=%llu fmt=%lu %lux%lu %s (template attachment only)", g_stat_pass + 1, R.rt.depth->serial, (unsigned long)stt.pixelFormat,
                   (unsigned long)stt.width, (unsigned long)stt.height, pd.stencilAttachment.loadAction == MTLLoadActionClear ? "clear" : "load");
    }
    if (natt && nload == natt) g_pass_full_rw++;
    g_nsd = 0;
    {
        for (int i = 0; i < R.rt.n && i < 8; i++) { MTLRenderPassColorAttachmentDescriptor *c = pd.colorAttachments[i]; if (!c.texture) continue;
            g_sd[g_nsd] = (sd_att){0}; g_sd[g_nsd].ser = R.rt.color[i]->serial; g_sd[g_nsd].bytes = att_bytes(c.texture, c.level); g_sd[g_nsd].lv = (uint32_t)c.level; g_sd[g_nsd].sl = (uint32_t)(c.texture.textureType == MTLTextureType3D ? c.depthPlane : c.slice); g_sd[g_nsd].idx = i; g_sd[g_nsd].load = c.loadAction == MTLLoadActionLoad; g_sd[g_nsd].wrote = R.clear_c[i] != 0; g_nsd++;
            c.storeAction = MTLStoreActionUnknown; }
        id<MTLTexture> dt = pd.depthAttachment.texture, stt = pd.stencilAttachment.texture;
        uint64_t db = att_bytes(dt, pd.depthAttachment.level), sb = 0;
        if (stt) { int comb = stt.pixelFormat == MTLPixelFormatDepth32Float_Stencil8; sb = att_bytes(stt, pd.stencilAttachment.level) / (comb ? 5 : 1); if (stt == dt) db -= sb; }   // Combined format 5 bytes = depth 4 + stencil 1 (when only stencil is attached, only 1 is counted)
        if (dt) { g_sd[g_nsd] = (sd_att){0}; g_sd[g_nsd].ser = R.rt.depth->serial; g_sd[g_nsd].bytes = db; g_sd[g_nsd].lv = (uint32_t)pd.depthAttachment.level; g_sd[g_nsd].sl = (uint32_t)pd.depthAttachment.slice; g_sd[g_nsd].idx = -1; g_sd[g_nsd].depth = 1; g_sd[g_nsd].load = pd.depthAttachment.loadAction == MTLLoadActionLoad; g_sd[g_nsd].wrote = R.clear_d != 0; g_nsd++;
                  pd.depthAttachment.storeAction = MTLStoreActionUnknown; }
        if (stt) { g_sd[g_nsd] = (sd_att){0}; g_sd[g_nsd].ser = R.rt.depth->serial; g_sd[g_nsd].bytes = sb; g_sd[g_nsd].lv = (uint32_t)pd.stencilAttachment.level; g_sd[g_nsd].sl = (uint32_t)pd.stencilAttachment.slice; g_sd[g_nsd].idx = -1; g_sd[g_nsd].depth = 2; g_sd[g_nsd].load = pd.stencilAttachment.loadAction == MTLLoadActionLoad; g_sd[g_nsd].wrote = R.clear_s != 0; g_nsd++;
                   pd.stencilAttachment.storeAction = MTLStoreActionUnknown; }
    }
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    {   // Pass info (same convention as the former per-draw fetch: colors take each attachment's base texture format; sample count takes the last color attachment, or depth if there is depth)
        uint64_t ser = g_pass.serial + 1; memset(&g_pass, 0, sizeof g_pass); g_pass.serial = ser;
        for (int i = 0; i < R.rt.n && i < 8; i++) { id<MTLTexture> t = R.rt.color[i] ? pd.colorAttachments[i].texture : nil;
            if (t) { g_pass.cfmt[i] = (uint32_t)t.pixelFormat; g_pass.samples = (uint32_t)t.sampleCount; } }
        id<MTLTexture> dt = R.rt.depth ? pd.depthAttachment.texture : nil, stt = R.rt.depth ? pd.stencilAttachment.texture : nil;   // Depth / stencil attachments are fetched separately (possibly only one of them is attached)
        g_pass.hd = dt && dt.pixelFormat != MTLPixelFormatStencil8; g_pass.hs = stt != nil;
        g_pass.dfmt = g_pass.hd ? (uint32_t)dt.pixelFormat : 0; g_pass.sfmt = stt ? (uint32_t)stt.pixelFormat : 0;
        if (dt || stt) g_pass.samples = (uint32_t)(dt ? dt : stt).sampleCount;
        pass_dims(pd, (NSUInteger)R.rt.n, &g_pass.tw, &g_pass.th);
        if (rt_range_on()) { g_pass.reg_gen = reg0; g_pass.ser_gen = ser0; }
        if (rt_range_on()) { g_pass.range_dirty=dirty0; g_pass.range_color=pd.colorAttachments[0].texture; g_pass.range_depth=pd.depthAttachment.texture; }
        g_enc_pso = nil; g_enc_dss = nil; g_enc_fx.valid = 0;
    }
    if (g_vis_on) {
        // If slots are exhausted (a single submission exceeds VIS_SLOTS counter channels), create an overflow buffer and continue using it. Do not retire buffers in the ring (they are reused in rotation; retiring one would remove it from the resident set);

        // Slots used up (one submission exceeds VIS_SLOTS counting passes): create an overflow buffer and continue with it. Buffers in the ring must not be handed to retirement (they are still used in rotation, and retiring would remove them from the residency set);
        // only the previous overflow buffer may be retired. The overflow buffer itself is retired when the next submission switches back to a ring buffer (mtl_sub_begin)
        if (!g_vis || g_vis_n >= VIS_SLOTS) { if (g_vis && !vis_in_ring(g_vis)) defer_obj_release(g_vis); g_vis = [g_dev newBufferWithLength:VIS_SLOTS * 8 options:MTLResourceStorageModeShared]; res_add(g_vis); g_vis_n = 0; }
        pd.visibilityResultBuffer = g_vis;
    }
    g_enc_vis = 0; g_enc_has_vis = g_vis_on;
    g_pass.layered = g_rp_layered;
    if (g_rp_layered) {
        NSUInteger nl = NSUIntegerMax;
        for (int i = 0; i < 8; i++) {
            MTLRenderPassColorAttachmentDescriptor *c = pd.colorAttachments[i];
            id<MTLTexture> t = c.texture; if (!t) continue;
            NSUInteger total = t.textureType == MTLTextureType3D ? MAX(t.depth >> c.level, 1u) :
                t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
            NSUInteger first = t.textureType == MTLTextureType3D ? c.depthPlane : c.slice;
            if (first >= total) { MLOG("[RED]Layered pass color attachment range is invalid"); return nil; }
            nl = MIN(nl, total - first);
        }
        if (pd.depthAttachment.texture) {
            id<MTLTexture> t = pd.depthAttachment.texture;
            NSUInteger total = t.arrayLength * (t.textureType == MTLTextureTypeCube || t.textureType == MTLTextureTypeCubeArray ? 6 : 1);
            if (pd.depthAttachment.slice >= total) { MLOG("[RED]Layered pass depth attachment range is invalid"); return nil; }
            nl = MIN(nl, total - pd.depthAttachment.slice);
        }
        if (nl != NSUIntegerMax && nl > 1) {
            pd.renderTargetArrayLength = nl;
            PD("L %llu layered pass: %lu layers", g_stat_pass + 1, (unsigned long)nl);
            static int logged; if (logged++ < 4) MLOG("Layered pass: %lu layers", (unsigned long)nl);
        }
    }
    pg_render(pd, (NSUInteger)R.rt.n);
    { void td_note(id pd, NSUInteger nc); td_note(pd, (NSUInteger)R.rt.n);
      void ts_note(id pd, NSUInteger nc); ts_note(pd, (NSUInteger)R.rt.n); }
    ser_gate();
    g_enc = enc_r_begin(pd); pg_render_begin(g_enc);
    args_dirty_all();
    g_stat_pass++;
    if (enc_label_on()) { id<MTLTexture> lt = pd.colorAttachments[0].texture ?: pd.depthAttachment.texture ?: pd.stencilAttachment.texture; g_el_w = (int)lt.width; g_el_h = (int)lt.height; g_el_draws = 0; g_el_key[0] = 0; }
    return g_enc;
}

// Empty-scissor precheck before opening a pass: the size of the selected level of the first valid target is an upper bound on the minimum bounds of the actual pass.
static void rt_dims(NSUInteger *tw, NSUInteger *th) {
    id<MTLTexture> t0 = nil; NSUInteger level = 0;
    for (int i = 0; i < R.rt.n && i < 8 && !t0; i++) if (R.rt.color[i]) {
        t0 = tex_mtl(R.rt.color[i]); level = R.rt.has_cview[i] && R.rt.cview[i].set_levels ? R.rt.cview[i].base_level : 0;
    }
    if (!t0 && R.rt.depth) { t0 = tex_mtl(R.rt.depth); level = R.rt.has_dview && R.rt.dview.set_levels ? R.rt.dview.base_level : 0; }
    *tw = t0 ? mip_extent(t0.width, level) : 1; *th = t0 ? mip_extent(t0.height, level) : 1;
}

// There are pending clears but no draw in the pass: open a pass once to land the clears
static void rp_flush(void) {
    int pend = R.clear_d || R.clear_s;
    for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
    if (pend && !g_enc && g_cb) rp_begin_ex(1);
    rp_end();
}
static int scissor_full(const tex_t *t, const view_t *v, int has_v) {
    NSUInteger level = has_v && v->set_levels ? v->base_level : 0;
    NSUInteger w = mip_extent(t->d.width, level), h = mip_extent(t->d.target == 0 || t->d.target == 3 ? 1 : t->d.height, level);
    return !R.has_sc || (R.sc.x <= 0 && R.sc.y <= 0 && (int64_t)R.sc.x + R.sc.w >= (int64_t)w && (int64_t)R.sc.y + R.sc.h >= (int64_t)h);
}



// ================= Shaders / descriptor pool / pipelines / draws =================
static pthread_mutex_t g_rmu = PTHREAD_MUTEX_INITIALIZER;

// ---------- Shader mapping table: Switch control section hash → PC metallib (produced by dm/gfx/tools/shader_map.py) ----------
static NSMutableDictionary<NSString *, NSArray *> *g_shmap;
static NSString *g_shmap_path, *g_msl_dir;
static NSString *exe_rel(const char *rel) {
    char buf[4096]; uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n)) return nil;
    char *dir = dirname(buf);
    return [NSString stringWithFormat:@"%s/%s", dir, rel];
}

// Shader mapping table (output of tools/shader_map.py): at runtime only 'Switch control section key → stage' is used; shaders are always translations of the Switch originals (tools/dmsh_build.py produces <key>_b.metallib + reflection offline)
static void shmap_load(void) {
    if (g_shmap) return;
    const char *e = getenv("DM_NVN_SHMAP");
    NSString *path = e ? @(e) : exe_rel("../gfx/nvn/shader_map.tsv");
    NSString *txt = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (!txt) MDIE("Could not read shader mapping table %s (DM_NVN_SHMAP can be used to specify one)", path.UTF8String);
    g_shmap = [NSMutableDictionary dictionary];
    for (NSString *ln in [txt componentsSeparatedByString:@"\n"]) {
        if (!ln.length || [ln hasPrefix:@"#"] || [ln hasPrefix:@"package"]) continue;
        NSArray *c = [ln componentsSeparatedByString:@"\t"];
        if (c.count > 3) g_shmap[c[3]] = c[2];
    }
    g_shmap_path = path;
    const char *m = getenv("DM_MSL_DIR");

    // Library directory = <DM_MSL_DIR or default cache>/b-<translation chain version>: the version comes from this source tree's dmsh_chain.h (tools/dmsh_chain.py); a host built from an old source tree only accepts libraries of the old version
    g_msl_dir = [NSString stringWithFormat:@"%@/b-%s", m ? @(m) : [NSString stringWithFormat:@"%s%s/build/cache/metal_shader_cache", getenv("DM_ROOT") ?: getenv("HOME"), getenv("DM_ROOT") ? "" : "/dm-work"], DMSH_CHAIN];
    if (![NSFileManager.defaultManager fileExistsAtPath:g_msl_dir]) MLOG("[RED]Shader library directory %s does not exist: the translation-chain version %s for this source tree has not been packaged yet (dm/gfx/tools/dmsh_build.py)", g_msl_dir.UTF8String, DMSH_CHAIN);
    NSString *cp = [[path stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"c1_full.tsv"];   // Same directory as the mapping table (Mac: dm/gfx/nvn; iOS: dm/ inside the App, copied along by run.sh)
    NSString *ct = cp ? [NSString stringWithContentsOfFile:cp encoding:NSUTF8StringEncoding error:nil] : nil;
    NSString *head = ct ? [[ct componentsSeparatedByString:@"\n"] firstObject] : nil;
    NSString *want = [NSString stringWithFormat:@"# chain b-%s", DMSH_CHAIN];
    if (!head || ![head hasPrefix:want]) {
        MLOG("[RED]c1_full.tsv manifest is stale or missing; all c1 buffers are exclusively 64 KiB");
    } else {
        g_c1_full_valid = 1; g_c1_full = [NSMutableSet set];
        for (NSString *line in [ct componentsSeparatedByString:@"\n"]) if (line.length && ![line hasPrefix:@"#"]) [g_c1_full addObject:[line stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]];
    }
    MLOG("Shader path B (translated from original Switch): %lu mappings (%s), metallib directory %s", (unsigned long)g_shmap.count, path.UTF8String, g_msl_dir.UTF8String);
    warm_start();
}

@interface NvShader : NSObject { @public id<MTLFunction> fn, fn_u, fn_m; uint8_t mask_m, color_known; nvn_refl rf; id<MTLArgumentEncoder> ae; NSUInteger arg_len; int stage, range_ro; const char *builtin; uint64_t logged; char key[65]; id<MTLBuffer> c1buf; uint64_t c1_gpu; size_t c1off;
    int gs_loc, lay_loc, layer_out; char gs_comp, lay_comp;

    // The argument buffer encoded last time
    id<MTLBuffer> ab_buf; uint64_t ab_gpu; size_t ab_off; uint64_t ab_pass;
    int8_t zf_state, zf_nf; zskip_fact zf;   // Cached lookup result in the all-zero skip fact table (0 not looked up / 1 absent / 2 present); with the key unchanged and the table length (unit tests may append) unchanged, no re-lookup   // ab_pass: render pass index where this argument block was encoded; reused only within the same pass (across passes / submissions the arena holding the block may have been recycled and rewritten)
    id<MTLLibrary> bflib; int bflib_tried; NSMutableDictionary<NSString *, id<MTLFunction>> *bffn; } @end   // bflib: readable-tile version <key>_bf (merge B step 2)
@implementation NvShader @end
#include "../nvn_pso_skey.h"
#include "../nvn_timestamp.h"
@interface NvProg : NSObject { @public NvShader *sh[6]; int gs_unresolved; __unsafe_unretained NvShader *fl_sh[6]; uint8_t fl_ok, fl_img, fl_img_w; uint32_t fl_cmask; } @end   // fl_*: reflection summary used per draw (udc_note_draw), reused after checking against sh[]
@implementation NvProg @end
@interface NvBoundProgram : NSObject { @public prog_t program; NvProg *prog; } @end
@implementation NvBoundProgram @end
typedef struct draw_trace_ctx {
    struct draw_trace_ctx *prev;
    const prog_t *nvn_program;
    __unsafe_unretained NvProg *program;
    uint64_t seq;
    uint64_t frame;
    uint64_t metal_draws;
    const char *pipeline;
    const char *result;
    int enabled;
} draw_trace_ctx;
static _Thread_local draw_trace_ctx *g_draw_trace;
static _Atomic uint64_t g_draw_trace_seq;
static void draw_trace_encoded(void) { if (g_draw_trace) g_draw_trace->metal_draws++; }
static const char *draw_trace_shader_load(NvShader *s) {
    if (!s) return "missing";
    if (s->fn) return "metallib-hit";
    return s->builtin ? "builtin" : "no-function";
}
static void draw_trace_targets(char *out, size_t cap) {
    size_t n = 0;
    if (!R.has_rt) { snprintf(out, cap, "none"); return; }
    int nc = R.rt.n;
    if (nc < 0) nc = 0;
    if (nc > 8) nc = 8;
    for (int i = 0; i < nc; i++) {
        const tex_t *t = R.rt.color[i];
        if (!t) continue;
        int w = snprintf(out + n, cap - n, "%sC%d=#%llu(%ux%u/f%#x)", n ? "," : "",
                         i, (unsigned long long)t->serial, t->d.width, t->d.height, (unsigned)t->d.format);
        if (w < 0 || (size_t)w >= cap - n) { n = cap - 1; break; }
        n += (size_t)w;
    }
    if (R.rt.depth && n < cap - 1) {
        int w = snprintf(out + n, cap - n, "%sD=#%llu(%ux%u/f%#x)", n ? "," : "",
                         (unsigned long long)R.rt.depth->serial, R.rt.depth->d.width,
                         R.rt.depth->d.height, (unsigned)R.rt.depth->d.format);
        if (w > 0 && (size_t)w < cap - n) n += (size_t)w;
    }
    if (!n) snprintf(out, cap, "empty");
}
enum { DRAW_SOURCE_STAT_CAP = 4096 };
typedef struct {
    uint64_t serial, frame, draws, pso_ready, metal_draws, skipped, missing_shader, no_program;
} draw_source_stat;
static pthread_mutex_t g_draw_source_mu = PTHREAD_MUTEX_INITIALIZER;
static draw_source_stat g_draw_source_stats[DRAW_SOURCE_STAT_CAP];
static uint64_t g_draw_source_frame, g_draw_source_overflow;
static void draw_trace_source_record(const draw_trace_ctx *t) {
    if (!t->enabled || !R.has_rt) return;
    NvShader *vs = t->program ? t->program->sh[0] : nil;
    NvShader *fs = t->program ? t->program->sh[1] : nil;
    int pso_ready = t->pipeline &&
        (!strcmp(t->pipeline, "cache-hit") || !strcmp(t->pipeline, "prefetched") ||
         !strcmp(t->pipeline, "built") || !strcmp(t->pipeline, "encoder-reuse"));
    int missing_shader = !strcmp(draw_trace_shader_load(vs), "missing") ||
        !strcmp(draw_trace_shader_load(vs), "no-function") ||
        !strcmp(draw_trace_shader_load(fs), "missing") ||
        !strcmp(draw_trace_shader_load(fs), "no-function");
    uint64_t seen[8] = {0};
    pthread_mutex_lock(&g_draw_source_mu);
    if (g_draw_source_frame != t->frame) {
        memset(g_draw_source_stats, 0, sizeof g_draw_source_stats);
        g_draw_source_frame = t->frame;
    }
    int nc = R.rt.n < 8 ? R.rt.n : 8;
    for (int i = 0; i < nc; i++) {
        const tex_t *target = R.rt.color[i];
        if (!target || !target->serial) continue;
        int duplicate = 0;
        for (int j = 0; j < i; j++) if (seen[j] == target->serial) { duplicate = 1; break; }
        if (duplicate) continue;
        seen[i] = target->serial;
        uint64_t h = (target->serial ^ (target->serial >> 32)) * 0x9e3779b97f4a7c15ull;
        draw_source_stat *slot = NULL;
        for (size_t probe = 0; probe < DRAW_SOURCE_STAT_CAP; probe++) {
            draw_source_stat *candidate = &g_draw_source_stats[(h + probe) & (DRAW_SOURCE_STAT_CAP - 1)];
            if (candidate->serial == target->serial || !candidate->serial) { slot = candidate; break; }
        }
        if (!slot) { g_draw_source_overflow++; continue; }
        if (slot->serial != target->serial) { memset(slot, 0, sizeof *slot); slot->serial = target->serial; slot->frame = t->frame; }
        slot->draws++;
        slot->pso_ready += pso_ready;
        slot->metal_draws += t->metal_draws;
        slot->skipped += t->metal_draws == 0;
        slot->missing_shader += missing_shader;
        slot->no_program += t->result && !strcmp(t->result, "no-program");
    }
    pthread_mutex_unlock(&g_draw_source_mu);
}
static void draw_trace_present_source(const tex_t *source, uint64_t frame) {
    if (!source || !nvn_draw_trace_on()) return;
    draw_source_stat stat = {0};
    int matched = 0;
    pthread_mutex_lock(&g_draw_source_mu);
    if (g_draw_source_frame == frame) {
        uint64_t h = (source->serial ^ (source->serial >> 32)) * 0x9e3779b97f4a7c15ull;
        for (size_t probe = 0; probe < DRAW_SOURCE_STAT_CAP; probe++) {
            draw_source_stat *candidate = &g_draw_source_stats[(h + probe) & (DRAW_SOURCE_STAT_CAP - 1)];
            if (!candidate->serial) break;
            if (candidate->serial == source->serial) {
                stat = *candidate; memset(candidate, 0, sizeof *candidate); matched = 1; break;
            }
        }
    }
    uint64_t overflow = g_draw_source_overflow;
    pthread_mutex_unlock(&g_draw_source_mu);
    MLOG("[DEBUG-RA-DRAW] event=present-source frame=%llu serial=%llu size=%ux%u format=%#x target_draws=%llu matched=%d pso_ready=%llu metal_draws=%llu skipped=%llu missing_shader=%llu no_program=%llu stat_overflow=%llu",
         (unsigned long long)frame, (unsigned long long)source->serial,
         source->d.width, source->d.height, (unsigned)source->d.format,
         (unsigned long long)stat.draws, matched,
         (unsigned long long)stat.pso_ready, (unsigned long long)stat.metal_draws,
         (unsigned long long)stat.skipped, (unsigned long long)stat.missing_shader,
         (unsigned long long)stat.no_program,
         (unsigned long long)overflow);
}
static void draw_trace_shader_stages(const prog_t *program, char *out, size_t cap) {
    if (!cap) return;
    size_t used = 0;
    if (!program || !program->nstage) { snprintf(out, cap, "none"); return; }
    for (unsigned i = 0; i < program->nstage && i < 6; i++) {
        uint8_t md[8];
        int have_key = prog_key(program, i, md);
        uint32_t word700 = 0, word704 = 0;
        if (program->sd[i].control) {
            memcpy(&word700, (const uint8_t *)program->sd[i].control + 0x700, sizeof word700);
            memcpy(&word704, (const uint8_t *)program->sd[i].control + 0x704, sizeof word704);
        }
        int n = have_key
            ? snprintf(out + used, cap - used, "%u:key=%02x%02x%02x%02x%02x%02x%02x%02x:word700=%#x:word704=%#x",
                       i, md[0], md[1], md[2], md[3], md[4], md[5], md[6], md[7], word700, word704)
            : snprintf(out + used, cap - used, "%u:no-key:word700=%#x:word704=%#x", i, word700, word704);
        if (n < 0 || (size_t)n >= cap - used) { out[cap - 1] = 0; return; }
        used += (size_t)n;
    }
}
static void draw_trace_finish(draw_trace_ctx *t) {
    if (!t->enabled) return;
    draw_trace_source_record(t);
    if (!nvn_draw_trace_target_draw_on()) { g_draw_trace = t->prev; return; }
    if (t->metal_draws) t->result = "encoded";
    else if (!strcmp(t->result, "not-encoded")) t->result = "no-metal-draw";
    NvShader *vs = t->program ? t->program->sh[0] : nil;
    NvShader *fs = t->program ? t->program->sh[1] : nil;
    char targets[512]; draw_trace_targets(targets, sizeof targets);
    char stages[2048]; draw_trace_shader_stages(t->nvn_program, stages, sizeof stages);
    MLOG("[DEBUG-RA-DRAW] event=draw frame=%llu seq=%llu pass=%llu pso=%s metal=%llu result=%s vs=%s:%.16s fs=%s:%.16s programStages=%s rt=%s",
         (unsigned long long)t->frame, (unsigned long long)t->seq,
         (unsigned long long)g_stat_pass, t->pipeline ?: "unknown",
         (unsigned long long)t->metal_draws, t->result ?: "unknown",
         draw_trace_shader_load(vs), vs ? vs->key : "-", draw_trace_shader_load(fs),
         fs ? fs->key : "-", stages, targets);
    g_draw_trace = t->prev;
}

// Warm-up entry: built in the background, waiting to be taken by the submit thread. vs / fs are strong references (in pso they are only raw pointers, and the background dictionary is released when the task ends)
@interface WarmItem : NSObject { @public int kind; NSString *key, *vs_key, *fs_key; NvShader *shader, *vs, *fs; pso_key pso; id<MTLFunction> fsfn; id object; } @end
@implementation WarmItem @end
static NSMutableDictionary<NSString *, NvShader *> *g_shcache; static _Atomic uint32_t g_shcache_n;   // The tables are read and written only on the submit thread; a separate atomic count is kept for other threads to read (resource counts)

static void c1_store(NvShader *o) {
    size_t len = o->rf.c1_len;
    NSString *guardKey = [NSString stringWithUTF8String:o->key];
    if ([guardKey hasSuffix:@"_b"]) guardKey = [guardKey substringToIndex:guardKey.length - 2];
    os_unfair_lock_lock(&g_c1_lock);   // Both the warm-up thread and the submit thread load shaders
    if (!g_c1_full_valid || [g_c1_full containsObject:guardKey]) {   // Runtime index not proven bounded: exclusive 64 KiB of all zeros (same as the original implementation)
        o->c1buf = [g_dev newBufferWithLength:C1_DECL_SIZE options:MTLResourceStorageModeShared]; o->c1_gpu = o->c1buf.gpuAddress;
        if (!o->c1buf) { os_unfair_lock_unlock(&g_c1_lock); MDIE("c1 dedicated buffer allocation failed"); }
        res_add(o->c1buf);
        memset(o->c1buf.contents, 0, C1_DECL_SIZE); memcpy(o->c1buf.contents, o->rf.c1, len);
        g_c1_alloc_bytes += C1_DECL_SIZE; g_c1_data_bytes += len; g_c1_count++; g_c1_exclusive++;
        os_unfair_lock_unlock(&g_c1_lock);
        return;
    }
    size_t need = (len + 255u) & ~((size_t)255u);
    if (!g_c1_banks) g_c1_banks = [NSMutableArray array];
    if (!g_c1_banks.count || g_c1_bank_off + need + 256u + C1_DECL_SIZE > C1_BANK_SIZE) {
        id<MTLBuffer> b = [g_dev newBufferWithLength:C1_BANK_SIZE options:MTLResourceStorageModeShared];
        if (!b) { os_unfair_lock_unlock(&g_c1_lock); MDIE("c1 buffer block allocation failed"); }
        res_add(b);
        memset(b.contents, 0, C1_BANK_SIZE);
        [g_c1_banks addObject:b];
        g_c1_bank_off = 0;
        g_c1_alloc_bytes += C1_BANK_SIZE;
    }
    id<MTLBuffer> b = g_c1_banks.lastObject;
    size_t off = g_c1_bank_off;
    memcpy((uint8_t *)b.contents + off, o->rf.c1, len);
    g_c1_bank_off += need + 256u;
    g_c1_data_bytes += len;
    g_c1_count++;
    o->c1buf = b;
    o->c1_gpu = b.gpuAddress;
    o->c1off = off;
    os_unfair_lock_unlock(&g_c1_lock);
}

static NSMutableArray<WarmItem *> *g_warm_items;
static os_unfair_lock g_warm_lock = OS_UNFAIR_LOCK_INIT;
static _Atomic int g_warm_ready;
static int g_warm_started;
static uint64_t g_warm_t0;
static _Atomic int g_warm_running;   // Background warm-up has not finished yet (the loading screen on entering a run waits a little longer based on this, see mtl_warm_wait)
static uint64_t g_warm_shaders, g_warm_rps, g_warm_cps, g_warm_skip_fz;
static uint64_t g_warm_claim_shaders, g_warm_claim_rps, g_warm_claim_cps;
static id<MTLFunction> bf_fn(NvShader *fs, const int *fx, uint8_t fbmask, const uint8_t *lmap);
static int rt_range_bindings_ro(NSDictionary *j);
static FILE *g_pso_record; static int g_pso_record_checked;
static FILE *pso_record_file(void) {
    if (g_pso_record_checked) return g_pso_record;
    g_pso_record_checked = 1;
    const char *path = getenv("DM_PSO_RECORD");
    if (path && *path) g_pso_record = fopen(path, "a");
    return g_pso_record;
}
static void pso_record(const char *fmt, ...) {
    FILE *f = pso_record_file();
    if (!f) return;
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap); fflush(f);
}
// User manifest: shaders/pipelines created synchronously by the submission thread at runtime on this device are warmed up on the next launch alongside the seed manifest.

// User manifest: shaders / pipelines created synchronously on the submit thread while running on this machine, warmed up at the next launch together with the seed manifest.
// Path = DM_PSO_USER, or pso_user.tsv under the parent of the save directory (DM_SAVE, Documents/save on iOS); tests use a fresh save copy per run, so they do not pollute each other
static NSString *pso_user_path(void) {
    const char *u = getenv("DM_PSO_USER"), *save = getenv("DM_SAVE");
    if (u && *u) return @(u);
    return save && *save ? [[@(save) stringByDeletingLastPathComponent] stringByAppendingPathComponent:@"pso_user.tsv"] : nil;
}
static FILE *g_pso_user; static int g_pso_user_checked;
static NSMutableSet<NSString *> *g_pso_user_seen;
static FILE *pso_user_file(void) {
    if (g_pso_user_checked) return g_pso_user;
    g_pso_user_checked = 1;
    NSString *path = pso_user_path();
    if (path) g_pso_user = fopen(path.fileSystemRepresentation, "a");
    return g_pso_user;
}
static void pso_user_record(const char *fmt, ...) {
    FILE *f = pso_user_file();
    if (!f) return;
    char line[8192]; va_list ap; va_start(ap, fmt); vsnprintf(line, sizeof line, fmt, ap); va_end(ap);
    if (!g_pso_user_seen) {
        g_pso_user_seen = [NSMutableSet set];
        NSString *seed = g_shmap_path ? [g_shmap_path.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"pso_manifest.tsv"] : nil;
        NSMutableArray<NSString *> *paths = [NSMutableArray array]; if (seed) [paths addObject:seed]; NSString *user = pso_user_path(); if (user) [paths addObject:user];
        for (NSString *path in paths) { NSString *txt = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            for (NSString *old in [txt componentsSeparatedByString:@"\n"]) if (old.length && ![old hasPrefix:@"#"]) [g_pso_user_seen addObject:old]; }
    }
    size_t ll = strlen(line); NSString *s = [[NSString alloc] initWithBytes:line length:ll && line[ll - 1] == '\n' ? ll - 1 : ll encoding:NSUTF8StringEncoding];   // Compared with the manifest file line by line (without the newline)
    if (!s || [g_pso_user_seen containsObject:s]) return;
    [g_pso_user_seen addObject:s]; fputs(line, f); fflush(f);
}
// Create a library from a .metallib file without keeping the file descriptor open: newLibraryWithURL keeps the file open as long as the library object lives (one per game shader—of the 258 descriptors kept open during an iOS session, 213 are these,

// Build the library from a .metallib file without permanently occupying a file descriptor: newLibraryWithURL keeps that file open for as long as the library object lives (one per game shader - of the 258 descriptors the iOS app permanently holds in a run, 213 are these,
// while the default soft limit of an iOS process is 256; new shaders created in other acts raise it further). Here the file is mapped into memory and closed immediately, and the library is built from the in-memory data; the mapping is released when the library's data object is freed
static _Thread_local int t_lib_layer;
static id<MTLLibrary> lib_from_file(NSString *path, NSError **err) {
    int fd = open(path.fileSystemRepresentation, O_RDONLY); struct stat st;
    if (fd < 0 || fstat(fd, &st) || st.st_size <= 0) { if (fd >= 0) close(fd); if (err) *err = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Could not open %@", path]}]; return nil; }
    size_t len = (size_t)st.st_size; void *m = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    t_lib_layer = m != MAP_FAILED && memmem(m, len, "render_target_array_index", 25) != NULL;
    if (m == MAP_FAILED) { if (err) *err = [NSError errorWithDomain:NSPOSIXErrorDomain code:errno userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Mapping failed %@", path]}]; return nil; }
#ifdef DM_SHELL
    // Wrapper: when the user's computer does not have Apple's shader compiler (no-compiler mode in tools/d2rm_pack.py, gfx/tools/nometal_run.py), the "library file" contains
    // shader source instead of a compiled library (compiled libraries start with MTLB). Compile it on the spot; the system caches the compiled result, so the next launch with the same shader does not incur this delay.

    // Shell: when the user's computer has no Apple shader compiler (the no-compiler mode of tools/d2rm_pack.py, gfx/tools/nometal_run.py), the "library files" contain
    // shader source rather than compiled libraries (a compiled library starts with MTLB). They are compiled here on the spot; the system caches the compiled result, and from the second launch on the same shader no longer costs this time.
    // Measured on a real device (cold): about 80 ms each on average, about 450 ms for the longest single one; one session uses several hundred.
    if (len < 4 || memcmp(m, "MTLB", 4)) { NSString *src = [[NSString alloc] initWithBytes:m length:len encoding:NSUTF8StringEncoding]; munmap(m, len);
        if (!src) { if (err) *err = [NSError errorWithDomain:NSPOSIXErrorDomain code:EILSEQ userInfo:@{NSLocalizedDescriptionKey: [NSString stringWithFormat:@"Neither a compiled library nor source code %@", path]}]; return nil; }
        static _Atomic int n; static _Atomic long long us; CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent();
        id<MTLLibrary> l = [g_dev newLibraryWithSource:src options:nil error:err]; long long dt = (long long)((CFAbsoluteTimeGetCurrent() - t0) * 1e6);
        int k = atomic_fetch_add(&n, 1) + 1; long long tot = atomic_fetch_add(&us, dt) + dt;
        if (k == 1 || k % 100 == 0 || dt >= 300000) MLOG("Compiling shader now: #%d %.0f ms, total %.1f seconds", k, dt / 1e3, tot / 1e6);
        return l; }
#endif
    dispatch_data_t d = dispatch_data_create(m, len, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{ munmap(m, len); });
    return [g_dev newLibraryWithData:d error:err];
}
// Program stage keys come from prog_key (hle_nvn.c): the first 8 bytes of sha256(control section + code area),
// looked up in the offline shader map (DM_NVN_SHMAP) and resolved against the b-<chain> library directory.
// nxnative MIT 0f99c35, dm/gfx/nvn/nvn_mtl.m:3040-3048,3072-3078.
// Mode is established by DeviceSetDepthMode before any shader warm-up.
static _Atomic int g_depth_zero, g_vs_loaded;
void mtl_set_depth_zero(int on) {
    if (atomic_exchange(&g_depth_zero, on) == on) return;
    MLOG("Depth entry mode: %s", on ? "near=0 → main0_z/main0_lz" : "near=-w → main0/main0_l");
    int n = atomic_load(&g_vs_loaded);
    if (n) MLOG("[RED] %d VS already loaded when depth mode changes; keeping the old entry-point convention", n);
}

static NvShader *shader_load(NSString *pc, int stage, int layer) {
    NSString *dir = g_msl_dir;
    NSError *e = nil;
    NSString *lp = [NSString stringWithFormat:@"%@/%@.metallib", dir, pc];
    NSData *jd = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%@/%@.json", dir, pc]];
    id decoded = jd ? [NSJSONSerialization JSONObjectWithData:jd options:0 error:nil] : nil;
    NSDictionary *j = [decoded isKindOfClass:[NSDictionary class]] ? decoded : nil;
    if (stage == 2) {
        NSDictionary *slice = j[@"gs_slice"];
        NSString *comp = [slice isKindOfClass:[NSDictionary class]] ? slice[@"comp"] : nil;
        int loc = [slice isKindOfClass:[NSDictionary class]] ? [slice[@"loc"] intValue] : -1;
        if ([j[@"chain"] isEqual:@DMSH_CHAIN] && loc >= 0 && loc < 32 &&
            [comp isKindOfClass:[NSString class]] && comp.length == 1 && strchr("xyzw", comp.UTF8String[0])) {
            NvShader *o = [NvShader new]; o->stage = 2; o->lay_loc = -1;
            o->gs_loc = loc; o->gs_comp = comp.UTF8String[0]; strlcpy(o->key, pc.UTF8String, sizeof o->key);
            return o;
        }
        MLOG("[RED] geometry %s is not a verified slice writer; refusing to execute", pc.UTF8String);
        return nil;
    }
    if (layer && ![j[@"entry_layer"] isKindOfClass:[NSString class]]) return nil;
    id<MTLLibrary> lib = lib_from_file(lp, &e);
    if (!lib || ![j isKindOfClass:[NSDictionary class]]) { MLOG("[RED] Failed to load shader %s: %s", pc.UTF8String, e ? e.localizedDescription.UTF8String : "No reflection json"); return nil; }
    if (![j[@"chain"] isEqual:@DMSH_CHAIN]) { MLOG("[RED] Shader %s translation chain version %s does not match the version expected by this host: %s", pc.UTF8String, [[j[@"chain"] description] UTF8String], DMSH_CHAIN); return nil; }
    NvShader *o = [NvShader new];
    o->gs_loc = o->lay_loc = -1;
    NSDictionary *from = j[@"layer_from"];
    if (stage == 0 && [from isKindOfClass:[NSDictionary class]]) {
        o->lay_loc = [from[@"loc"] intValue];
        NSString *comp = from[@"comp"];
        if ([comp isKindOfClass:[NSString class]] && comp.length == 1) o->lay_comp = comp.UTF8String[0];
    }
    char err[160];
    if (nvn_refl_from_json(j, &o->rf, err, sizeof err)) { MLOG("[RED] Failed to convert reflection for shader %s: %s", pc.UTF8String, err); return nil; }
    if (rt_range_on()) o->range_ro=rt_range_bindings_ro(j);
    NSString *entry = j[@"entry"];
    if (stage == 0) {
        atomic_fetch_add(&g_vs_loaded, 1);
        int zero = atomic_load(&g_depth_zero);
        NSString *field = layer ? (zero ? @"entry_layer_z" : @"entry_layer") : (zero ? @"entry_z" : @"entry");
        if ([j[field] isKindOfClass:[NSString class]]) entry = j[field];
        else {
            static _Atomic int missing_depth_entry;
            if (!atomic_exchange(&missing_depth_entry, 1))
                MLOG("[RED] Vertex shader %s is missing depth entry point %s", pc.UTF8String, field.UTF8String);
            if (layer) return nil;
        }
        static _Atomic int logged[2];
        if (!atomic_fetch_add(&logged[layer], 1)) MLOG("VS depth entry point: %s shader=%s layer=%d", entry.UTF8String, pc.UTF8String, layer);
    }
    o->fn = [lib newFunctionWithName:entry]; o->stage = stage; strlcpy(o->key, pc.UTF8String, sizeof o->key);
    if (!o->fn) { MLOG("[RED] Shader %s has no entry point %s", pc.UTF8String, entry.UTF8String); return nil; }
    if (stage == 0) o->layer_out = layer || (t_lib_layer && !j[@"entry_layer"]);
    // Keep the external classic-Metal nvn_refl ABI unchanged.
    NSDictionary *raster = j[@"raster"];
    o->color_known = [raster isKindOfClass:[NSDictionary class]] && [raster[@"colors"] isKindOfClass:[NSArray class]];
    if (stage == 1 && j[@"entry_u"]) {
        o->fn_u = [lib newFunctionWithName:j[@"entry_u"]];
        if (!o->fn_u) MLOG("[RED] Shader %s has no integer output entry point %s", pc.UTF8String, [j[@"entry_u"] UTF8String]);
    }
    if (stage == 1 && j[@"entry_m"]) {
        o->fn_m = [lib newFunctionWithName:j[@"entry_m"]];
        o->mask_m = (uint8_t)[j[@"uint_mask_m"] unsignedIntValue];
        if (!o->fn_m) MLOG("[RED] Shader %s has no blended integer output entry point %s", pc.UTF8String, [j[@"entry_m"] UTF8String]);
    }
    if (o->fn.functionConstantsDictionary.count) {
        MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
        id<MTLFunction> f = [lib newFunctionWithName:entry constantValues:cv error:&e];
        if (f) o->fn = f; else MLOG("[RED] Function constant specialization failed for shader %s: %s", pc.UTF8String, e.localizedDescription.UTF8String);
        if (o->fn_u) o->fn_u = [lib newFunctionWithName:j[@"entry_u"] constantValues:cv error:&e];
        if (o->fn_m) o->fn_m = [lib newFunctionWithName:j[@"entry_m"] constantValues:cv error:&e];
    }
    if (o->rf.nb) {
        o->ae = [o->fn newArgumentEncoderWithBufferIndex:o->rf.abi];
        int max_aid = -1;
        for (int i = 0; i < o->rf.nb; i++) max_aid = MAX(max_aid, o->rf.b[i].aid);
        o->arg_len = o->ae.encodedLength;
        if (o->arg_len != (NSUInteger)(8 * (max_aid + 1)) || (o->arg_len & 7))
            MDIE("Parameter block layout error in shader %s: encodedLength=%lu maxAid=%d", o->key, (unsigned long)o->arg_len, max_aid);
    }
    if (o->rf.has_c1) c1_store(o);
    return o;
}

static int stage_of(NSString *name) {
    if ([name isEqualToString:@"Vertex"] || [name isEqualToString:@"vertex"]) return 0;
    if ([name isEqualToString:@"Pixel"] || [name isEqualToString:@"pixel"]) return 1;
    if ([name isEqualToString:@"geometry"]) return 2;
    if ([name isEqualToString:@"Hull"] || [name isEqualToString:@"hull"]) return 3;
    if ([name isEqualToString:@"Domain"] || [name isEqualToString:@"domain"]) return 4;
    if ([name isEqualToString:@"Compute"] || [name isEqualToString:@"compute"]) return 5;
    return -1;
}
#define NVN_RT_RANGE_RO_IMPL 1

// Private continuation accepts only B reflection that is explicitly RO; a missing rw on sbuf does not let the default 0 count as proof; does not change the existing reflection / translation chain.
static int rt_range_bindings_ro(NSDictionary *j) {
    if (![j[@"route"] isEqual:@"B"] || ![j[@"bindings"] isKindOfClass:[NSArray class]] ||
        ![j[@"drv_reads"] isKindOfClass:[NSDictionary class]] || [j[@"drv_reads"] count]) return 0;
    for (NSDictionary *b in j[@"bindings"]) {
        if (![b isKindOfClass:[NSDictionary class]] || ![b[@"kind"] isKindOfClass:[NSString class]] || ![b[@"msl_type"] isKindOfClass:[NSString class]]) return 0;
        NSString *k = b[@"kind"], *t = b[@"msl_type"];
        if (b[@"rw"] && (![b[@"rw"] isKindOfClass:[NSNumber class]] || [b[@"rw"] boolValue])) return 0;
        if ([k isEqual:@"cbuf"] || [k isEqual:@"c1"]) { if (![t hasPrefix:@"constant "]) return 0; }
        else if ([k isEqual:@"sbuf"]) { if (![b[@"rw"] isKindOfClass:[NSNumber class]] || [b[@"rw"] boolValue] || ![t hasPrefix:@"const device "]) return 0; }
        else if ([k isEqual:@"samp"] || [k isEqual:@"sampt"]) { if (![t isEqual:@"sampler"]) return 0; }
        else if ([k isEqual:@"tex"] || [k isEqual:@"tbuf"]) {
            if (![t hasPrefix:@"texture"] || ([t containsString:@"access::"] && ![t containsString:@"access::sample>"] && ![t containsString:@"access::read>"])) return 0; }
        else return 0;
    }
    return 1;
}
static int rt_range_pr_ro(NvProg *p) {
    if (!p || !p->sh[0] || !p->sh[1]) return 0;
    for (int k=0;k<6;k++) if (p->sh[k] && (k>1 || p->sh[k]->stage != k || !p->sh[k]->range_ro || p->sh[k]->rf.drv_read[0] || p->sh[k]->rf.drv_read[2])) return 0;
    return 1;
}
static NvShader *shader_get(NSString *pc, int stage) {
    if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
    NvShader *o = g_shcache[pc];
    if (o) return o;
    uint64_t bt = bp_now(); o = shader_load(pc, stage, 0);
    if (!o) return nil;
    g_shcache[pc] = o; atomic_fetch_add(&g_shcache_n, 1);
    pso_record("S\t%s\t%d\n", pc.UTF8String, stage);
    pso_user_record("S\t%s\t%d\n", pc.UTF8String, stage);
    g_bp_lib_n++; g_bp_lib_ns += bp_now() - bt;
    return o;
}
static NvShader *shader_get_layer(NvShader *vs) {
    NSString *pc = @(vs->key), *key = [pc stringByAppendingString:@"#l"];
    NvShader *o = g_shcache[key];
    if (!o && (o = shader_load(pc, 0, 1))) g_shcache[key] = o;
    return o;
}
static void prog_fold_gs(NvProg *o) {
    NvShader *g = o->sh[2], *v = o->sh[0];
    if (!g || g->fn || !v) return;
    NvShader *layer = v->lay_loc == g->gs_loc && v->lay_comp == g->gs_comp ? shader_get_layer(v) : nil;
    if (!layer) return;
    o->sh[0] = layer; o->sh[2] = nil; o->gs_unresolved = 0;
    static int once; if (!once++) MLOG("Slice geometry %s → VS %s layer entry point (layered pass)", g->key, v->key);
}
static __unsafe_unretained NvProg *prog_get(const prog_t *cp) {   // Called only on the submit thread
    prog_t *p = (prog_t *)cp;
    static const prog_t *lp; static uint64_t lser, lgen; static NvProg *lpr;
    uint64_t gen = g_ser_gen;
    if (p == lp && p->serial == lser && gen == lgen && lpr) return lpr;
    __unsafe_unretained NvProg *have = p->serial ? serial_get(p->serial) : nil;
    if (have) { lp = p; lser = p->serial; lgen = gen; lpr = have; return have; }
    NvProg *o = [NvProg new];
    shmap_load();
    for (unsigned i = 0; i < p->nstage; i++) {
        uint8_t md[8];
        if (!prog_key(p, i, md)) continue;
        NSString *k = [NSString stringWithFormat:@"%02x%02x%02x%02x%02x%02x%02x%02x", md[0], md[1], md[2], md[3], md[4], md[5], md[6], md[7]];
        NSString *zh = (NSString *)g_shmap[k];
        if (!zh) { MLOG("[RED] Shader %s is not in the offline mapping table", k.UTF8String); continue; }
        int st = stage_of(zh);
        NvShader *loaded = st >= 0 ? shader_get([k stringByAppendingString:@"_b"], st) : nil;
        if (loaded) o->sh[st] = loaded; else MLOG("[RED] Failed to load shader %s (%s): run dm/gfx/tools/dmsh_build.py first to generate the package", k.UTF8String, zh.UTF8String);
        if (st == 2) o->gs_unresolved = 1;
    }
    prog_fold_gs(o);
    if (p->serial) serial_put(p->serial, o);
    return o;
}
void mtl_prog_release(prog_t *p) {
    if (!p->serial) return;
    serial_del(p->serial);
}

// ---------- Descriptor pool: texture ID / sampler ID → object ----------

// ---------- Descriptor pool: texture ID / sampler ID → object ----------
// At registration the whole texture object is snapshotted (NVN semantics: the descriptor has been written into the pool); view: the MTLTexture for that ID (a view or the base texture, held via CFBridgingRetain) and the serial-table / guest-write generations at the time it was obtained
typedef struct { tex_t t; int ok; view_t v; int has_v; void *view; uint64_t view_id, view_rid, view_ser, view_dirty; } treg_t;
static treg_t *g_treg; static int g_ntreg;
// Lock-free copy of texture ID → index (the replay hot path only needs the index: write-ahead lookahead index, bind set for the pure-clear case): texture IDs are 20 bits, fixed 1M entries (BSS, only pages touched are committed),

// Lock-free copy of texture ID → serial (the replay hot path only needs the serial: store lookahead index, binding set of the clear-only hoist): texture IDs are 20 bits, fixed 1M entries (BSS, only touched pages are used),
// release write at registration, relaxed atomic read on the reader side, without entering g_rmu (the former h_ser locked and unlocked on every lookup, thousands of times per frame in the camp)
static _Atomic uint64_t g_tser[1 << 20];
static samp_par *g_sreg; static uint8_t *g_sreg_ok; static void **g_sview; static uint64_t *g_srid; static int g_nsreg;   // g_sview: sampler ID → MTLSamplerState (retained)
// A CPU Finalize may precede playback of already-recorded commands. Cache
// eviction waits for snapshot retirement and the last descriptor reference.
// Both maps are protected by g_rmu; values are integer state/counts, not CF ids.
static nvn_hmap g_texture_retired = {.klen = 8}, g_texture_descriptor_refs = {.klen = 8};
static void texture_collect_locked(uint64_t serial) {
    if ((uintptr_t)nvn_hm_get(&g_texture_retired, &serial) == 2 &&
        !nvn_hm_get(&g_texture_descriptor_refs, &serial)) {
        serial_del(serial);
        nvn_hm_del(&g_texture_retired, &serial);
    }
}
void mtl_tex_release(tex_t *t) {
    if (!t->serial) return;
    pthread_mutex_lock(&g_rmu);
    // The current contract keeps no separate snapshot object (command records and pool registrations snapshot
    // the texture by value): retire immediately; descriptor references hold the serial until they drop.
    nvn_hm_put(&g_texture_retired, &t->serial, (void *)(uintptr_t)2);
    texture_collect_locked(t->serial);
    pthread_mutex_unlock(&g_rmu);
}
static void texture_descriptor_ref_locked(uint64_t serial, int delta) {
    if (!serial) return;
    uintptr_t count = (uintptr_t)nvn_hm_get(&g_texture_descriptor_refs, &serial);
    if (delta > 0) nvn_hm_put(&g_texture_descriptor_refs, &serial, (void *)(count + 1));
    else if (count > 1) nvn_hm_put(&g_texture_descriptor_refs, &serial, (void *)(count - 1));
    else {
        nvn_hm_del(&g_texture_descriptor_refs, &serial);
        texture_collect_locked(serial);
    }
}
void mtl_reg_tex(int tid, const tex_t *t, const view_t *v) {
    int id_ = tid; if (id_ < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (id_ >= g_ntreg) { int n = id_ + 1024; g_treg = realloc(g_treg, n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (n - g_ntreg) * sizeof *g_treg); g_ntreg = n; }
    g_reg_gen++;
    if (g_treg[id_].view) { defer_obj_unref((__bridge id)g_treg[id_].view); CFRelease(g_treg[id_].view); g_treg[id_].view = NULL; g_treg[id_].view_id = g_treg[id_].view_rid = 0; }
    uint64_t old_serial = g_treg[id_].ok ? g_treg[id_].t.serial : 0;
    uint64_t new_serial = t ? t->serial : 0;
    if (old_serial != new_serial) {
        texture_descriptor_ref_locked(old_serial, -1);
        texture_descriptor_ref_locked(new_serial, 1);
    }
    g_treg[id_].ok = t != NULL; if (t) g_treg[id_].t = *t; g_treg[id_].has_v = v != NULL; if (v) g_treg[id_].v = *v;
    if (id_ < (1 << 20)) atomic_store_explicit(&g_tser[id_], t ? t->serial : 0, memory_order_release);
    pthread_mutex_unlock(&g_rmu);
}
// RegisterImage declares image use even when the builder omitted IMAGE.
void mtl_reg_image(int tid, const tex_t *t, const view_t *v) {
    if (!t) { mtl_reg_tex(tid, NULL, v); return; }
    tex_t image = *t; image.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;
    mtl_reg_tex(tid, &image, v);
}
void mtl_reg_samp(int sid, const samp_par *sp) {
    int id_ = sid; if (id_ < 0) return;
    pthread_mutex_lock(&g_rmu);
    if (id_ >= g_nsreg) { int n = id_ + 256; g_sreg = realloc(g_sreg, n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, n); g_sview = realloc(g_sview, n * sizeof *g_sview); g_srid = realloc(g_srid, n * sizeof *g_srid);
        memset(g_sreg_ok + g_nsreg, 0, n - g_nsreg); memset(g_sview + g_nsreg, 0, (n - g_nsreg) * sizeof *g_sview); memset(g_srid + g_nsreg, 0, (n - g_nsreg) * sizeof *g_srid); g_nsreg = n; }
    if (g_sview[id_]) { CFRelease(g_sview[id_]); g_sview[id_] = NULL; g_srid[id_] = 0; }
    g_reg_gen++;
    g_sreg[id_] = *sp; g_sreg_ok[id_] = 1;
    pthread_mutex_unlock(&g_rmu);
}
static MTLTextureType ttype(int target, int fallback) {
    switch (target) { case 0: return MTLTextureType1D; case 1: case 7: return MTLTextureType2D; case 2: return MTLTextureType3D;
        case 3: return MTLTextureType1DArray; case 4: return MTLTextureType2DArray; case 5: return MTLTextureType2DMultisample;
        case 6: return MTLTextureType2DMultisampleArray; case 8: return MTLTextureTypeCube; case 9: return MTLTextureTypeCubeArray; }
    return (MTLTextureType)fallback;
}
// Texture IDs outside the pool = empty slots: the game itself uses GetSeparateTextureHandle(0xFFFFF) to get a handle and bind it to an unused slot (call site observed at main+0x1746820,

// Out-of-pool texture ID = empty slot: the game itself uses GetSeparateTextureHandle(0xFFFFF) to get a handle and bind it to unused slots (measured call site main+0x1746820,
// texture pool capacity 15256) and never registers that ID. Read 0 per D3D / empty-slot semantics (the same all-zero dummy textures as unbound), no failure; unregistered IDs inside the pool are still reported a failure
static int g_tpool_n; static uint64_t g_stat_tex_oob;
void mtl_set_tex_pool_n(int n) { g_tpool_n = n; }
static id<MTLTexture> tex_by_id(int tid) {   // Caller holds g_rmu
    if (g_tpool_n && tid >= g_tpool_n) { if (!g_stat_tex_oob++) MLOG("Texture ID %d is outside the pool (capacity %d); treating as an empty slot: returning 0", tid, g_tpool_n); return nil; }
    if (tid < 0 || tid >= g_ntreg || !g_treg[tid].ok) { RED_ONCE(10, "Texture ID %d is not registered", tid); return nil; }
    treg_t *rp = &g_treg[tid];
    uint64_t sg = g_ser_gen, dg = nvn_dirty_now();   // Take the generations first and then look up: if a change happens in between, the next time must take the slow path
    if (rp->view && rp->view_ser == sg && rp->view_dirty == dg) return (__bridge id<MTLTexture>)rp->view;
    id<MTLTexture> v = (__bridge id<MTLTexture>)rp->view;
    if (v) {   // The serial table or guest memory has changed: check whether the base texture was re-uploaded and replaced; if so the view is invalidated
        id<MTLTexture> cur = tex_mtl(&rp->t);
        if (cur == v || v.parentTexture == cur) { rp->view_ser = sg; rp->view_dirty = dg; return v; }
        defer_obj_unref(v); CFRelease(rp->view); rp->view = NULL; rp->view_rid = 0; v = nil;
    }
    treg_t r = *rp;
    id<MTLTexture> base = tex_mtl(&rp->t);
    if (!base) return nil;
    v = base;
    int stencil_view = r.t.stencil && (r.t.d.format == 0x35 || r.t.d.format == 0x36);
    MTLTextureSwizzleChannels sample_swizzle = MTLTextureSwizzleChannelsDefault;
    int rt_swizzle_view = 0;
    if (base.usage & MTLTextureUsageRenderTarget) {
        uint32_t swizzle_format = r.has_v && r.v.set_format ? r.v.format : r.t.d.format;
        uint32_t swizzle_pf = 0; uint8_t swizzle_map[4];
        if (nvn_fmt_to_mtl(swizzle_format, &swizzle_pf, swizzle_map) && swizzle_pf) {
            sample_swizzle = tex_swizzle(&r.t, swizzle_map);
            rt_swizzle_view = !tex_swizzle_identity(sample_swizzle);
        } else {
            RED_ONCE(53, "Cannot parse RT sampling swizzle format %#x", swizzle_format);
            return nil;
        }
    }
    if ((r.has_v && (r.v.set_format || r.v.set_levels || r.v.set_layers || r.v.set_target)) ||
        stencil_view || rt_swizzle_view) {
        uint32_t pf = (uint32_t)base.pixelFormat; uint8_t swz[4];
        if (stencil_view) pf = MTLPixelFormatX32_Stencil8;
        if (r.has_v && r.v.set_format && !(nvn_fmt_to_mtl(r.v.format, &pf, swz) && pf)) pf = (uint32_t)base.pixelFormat;
        if (r.has_v && r.v.set_format && tex_need_pfv(base.pixelFormat, (MTLPixelFormat)pf) && !(base.usage & MTLTextureUsagePixelFormatView)) {
            base = tex_upgrade_pfv(&rp->t, base, (MTLPixelFormat)pf);
            sg = g_ser_gen;
        }
        NSUInteger l0 = r.has_v && r.v.set_levels ? r.v.base_level : 0;
        NSUInteger nl = r.has_v && r.v.set_levels && r.v.nlevels ? r.v.nlevels : base.mipmapLevelCount - l0;
        NSUInteger s0 = r.has_v && r.v.set_layers ? r.v.min_layer : 0;
        NSUInteger ns = base.textureType == MTLTextureType3D ? 1 :
            (r.has_v && r.v.set_layers && r.v.nlayers ? r.v.nlayers :
             base.arrayLength * (base.textureType == MTLTextureTypeCube || base.textureType == MTLTextureTypeCubeArray ? 6 : 1) - s0);
        MTLTextureType tt = r.has_v && r.v.set_target ? ttype(r.v.target, base.textureType) : base.textureType;
        if (tt == MTLTextureType2D && ns > 1) ns = 1;
        v = rt_swizzle_view
            ? [base newTextureViewWithPixelFormat:(MTLPixelFormat)pf textureType:tt levels:NSMakeRange(l0, nl) slices:NSMakeRange(s0, ns) swizzle:sample_swizzle]
            : [base newTextureViewWithPixelFormat:(MTLPixelFormat)pf textureType:tt levels:NSMakeRange(l0, nl) slices:NSMakeRange(s0, ns)];
        if (!v) {
            RED_ONCE(11, "Failed to create texture view for id %d", tid);
            if (rt_swizzle_view) return nil;
            v = base;
        }
        if (rt_swizzle_view) {
            static int rt_swizzle_view_logged;
            if (!rt_swizzle_view_logged++)
                MLOG("Route-A sampling view carries NVN swizzle: fmt=%#x base=%lu view=%lu channels=%u,%u,%u,%u",
                     r.has_v && r.v.set_format ? r.v.format : r.t.d.format,
                     (unsigned long)base.pixelFormat, (unsigned long)v.pixelFormat,
                     (unsigned)sample_swizzle.red, (unsigned)sample_swizzle.green,
                     (unsigned)sample_swizzle.blue, (unsigned)sample_swizzle.alpha);
        }
    }
    rp->view = (void *)CFBridgingRetain(v); rp->view_rid = v.gpuResourceID._impl; rp->view_ser = sg; rp->view_dirty = dg;
    return v;
}

// Hot-path object fetch: if the table entry is valid, return the table-held pointer directly (without ARC retain / autorelease), otherwise take the slow path to build / validate. Caller holds g_rmu.
static inline __attribute__((always_inline)) void *tex_fast(int tid) {
    if (tid > 0 && tid < g_ntreg && g_treg[tid].ok && g_treg[tid].view && g_treg[tid].view_ser == g_ser_gen && g_treg[tid].view_dirty == nvn_dirty_now()) return g_treg[tid].view;
    return (__bridge void *)tex_by_id(tid);
}
static inline __attribute__((always_inline)) uint64_t tex_fast_rid(int tid, void **out) {
    if (tid > 0 && tid < g_ntreg && g_treg[tid].ok && g_treg[tid].view && g_treg[tid].view_ser == g_ser_gen && g_treg[tid].view_dirty == nvn_dirty_now()) { if (out) *out = g_treg[tid].view; return g_treg[tid].view_rid; }
    id<MTLTexture> t = tex_by_id(tid); if (out) *out = (__bridge void *)t; return t ? t.gpuResourceID._impl : 0;
}
static MTLSamplerAddressMode wrapm(int w) {
    switch (w) { case 0: case 7: return MTLSamplerAddressModeClampToEdge; case 1: return MTLSamplerAddressModeRepeat;
        case 3: case 2: return MTLSamplerAddressModeMirrorClampToEdge; case 5: case 4: return MTLSamplerAddressModeClampToBorderColor;
        case 6: return MTLSamplerAddressModeMirrorRepeat; }
    return MTLSamplerAddressModeRepeat;
}
static id<MTLSamplerState> samp_by_id(int sid) {   // Caller holds g_rmu
    if (sid < 0 || sid >= g_nsreg || !g_sreg_ok[sid]) { RED_ONCE(12, "Sampler ID %d is not registered", sid); return nil; }
    if (g_sview[sid]) return (__bridge id<MTLSamplerState>)g_sview[sid];
    samp_par p = g_sreg[sid];
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.magFilter = p.magf ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.minFilter = (p.minf == 1 || p.minf == 3 || p.minf == 5) ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.mipFilter = p.minf <= 1 ? MTLSamplerMipFilterNotMipmapped : (p.minf == 2 || p.minf == 3) ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
    d.sAddressMode = wrapm(p.ws); d.tAddressMode = wrapm(p.wt); d.rAddressMode = wrapm(p.wr);
    d.lodMinClamp = p.lod_min < 0 ? 0 : p.lod_min; d.lodMaxClamp = p.lod_max;
    d.maxAnisotropy = p.aniso < 1 ? 1 : p.aniso > 16 ? 16 : (NSUInteger)p.aniso;
    if (p.cmp_mode) d.compareFunction = (MTLCompareFunction)(p.cmp_func - 1);
    d.borderColor = p.border[3] > 0.5f ? (p.border[0] > 0.5f ? MTLSamplerBorderColorOpaqueWhite : MTLSamplerBorderColorOpaqueBlack) : MTLSamplerBorderColorTransparentBlack;
    d.supportArgumentBuffers = YES;
    id<MTLSamplerState> s = [g_dev newSamplerStateWithDescriptor:d];
    if (s) { g_sview[sid] = (void *)CFBridgingRetain(s); g_srid[sid] = s.gpuResourceID._impl; }
    return s;
}
static inline __attribute__((always_inline)) void *samp_fast(int sid) {
    if (sid > 0 && sid < g_nsreg && g_sreg_ok[sid] && g_sview[sid]) return g_sview[sid];
    return (__bridge void *)samp_by_id(sid);
}


// ---------- Argument buffer blocks: one block per submission, reclaimed after the GPU completes ----------
@interface NvArena : NSObject { @public id<MTLBuffer> buf; uint64_t gpu, fence; size_t used; } @end
@implementation NvArena @end
static NSMutableArray<NvArena *> *g_arenas; static NvArena *g_cur_arena; static NSMutableArray<NvArena *> *g_sub_arenas;
static NvArena *arena_new(void) {

    // The for-in loop variable does not retain the object: take a strong reference first and then remove it from the array, otherwise what is returned after the array releases it is a dangling pointer
    for (NvArena *it in g_arenas) if (it->fence && it->fence <= g_done) { NvArena *r = it; [g_arenas removeObject:r]; r->fence = 0; r->used = 0; return r; }
    NvArena *a = [NvArena new]; a->buf = [g_dev newBufferWithLength:4 << 20 options:MTLResourceStorageModeShared]; a->gpu = a->buf.gpuAddress; res_add(a->buf); return a;
}
void arena_begin(void) { if (!g_arenas) { g_arenas = [NSMutableArray array]; g_sub_arenas = [NSMutableArray array]; } g_cur_arena = nil; }
void arena_commit(uint64_t fence) { for (NvArena *a in g_sub_arenas) { a->fence = fence; [g_arenas addObject:a]; } [g_sub_arenas removeAllObjects]; g_cur_arena = nil; }
static size_t arena_alloc(size_t n, id<MTLBuffer> *buf) {
    n = (n + 255) & ~(size_t)255;
    if (!g_cur_arena || g_cur_arena->used + n > g_cur_arena->buf.length) {
        pthread_mutex_lock(&g_mu); g_cur_arena = arena_new(); pthread_mutex_unlock(&g_mu);
        [g_sub_arenas addObject:g_cur_arena];
    }
    size_t off = g_cur_arena->used; g_cur_arena->used += n; *buf = g_cur_arena->buf; return off;
}


// ---------- Draw state ----------
typedef struct { const prog_t *program; uint8_t data_index; } program_stage_binding;
static struct {
    const prog_t *prog;
    program_stage_binding program_stage[6];
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream;
    NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint64_t tex[6][2 * NVN_SEPTEX_SLOT0], samp[6][128], img[6][32];   // texture slots: 0-127 combined handles (nvnCommandBufferBindTexture), 128-255 separate binds (nvnCommandBufferBindSeparateTextures, NVN_SEPTEX_SLOT0)
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    p_poly_offset po; float vp[16][4]; float dr[2]; int has_vp;
    float blend_color[4]; int64_t copy_stride, copy_image_stride;
} S;
static NSMutableDictionary<NSData *, NvBoundProgram *> *g_bound_program_cache;
static _Atomic uint64_t g_bound_program_count;
static void program_binding_apply(program_stage_binding bound[6], const p_bind_program *binding) {
    if (!binding) return;
    uint32_t mask = (uint32_t)binding->stages & 0x3fu;
    // NVN's BindProgram(program, stage bits): the stages in the bits take that program's shaders
    // (a null program unbinds them), the other stages stay. prog_get resolves a program's shaders into
    // per-stage slots via the offline shader map, so each bound stage points at its program whole; the
    // compose step picks the stage's shader out of the program's resolution.
    for (unsigned stage = 0; stage < 6; stage++)
        if (mask & (1u << stage)) bound[stage] = (program_stage_binding){binding->prog, 0};
}
// The effective program of a draw combines the per-stage binds (program_binding_apply): resolve each
// bound stage through prog_get and pick that stage's shader; identical combinations share one host-made
// prog_t carrying only a serial, whose NvProg in the serial table holds the combined stages (upstream
// bind_stages semantics, keyed on the resolved shader keys instead of a fixed ring).
static const prog_t *program_binding_compose(const program_stage_binding bound[6]) {
    __unsafe_unretained NvProg *resolved[6] = {nil};
    unsigned count = 0;
    for (unsigned stage = 0; stage < 6; stage++) {
        if (!bound[stage].program) continue;
        __unsafe_unretained NvProg *src = prog_get(bound[stage].program);
        if (src && src->sh[stage]) { resolved[stage] = src; count++; }
    }
    if (!count) return NULL;
    char key[6][80] = {{0}};
    for (unsigned stage = 0; stage < 6; stage++) if (resolved[stage]) strlcpy(key[stage], resolved[stage]->sh[stage]->key, sizeof key[stage]);
    if (!g_bound_program_cache) g_bound_program_cache = [NSMutableDictionary dictionary];
    NSData *cache_key = [NSData dataWithBytes:key length:sizeof key];
    NvBoundProgram *entry = g_bound_program_cache[cache_key];
    if (!entry) {
        entry = [NvBoundProgram new];
        entry->program.magic = MAGIC('P', 'R', 'O', 'G');
        entry->program.serial = nvn_new_serial();
        entry->prog = [NvProg new];
        for (unsigned stage = 0; stage < 6; stage++) if (resolved[stage]) entry->prog->sh[stage] = resolved[stage]->sh[stage];
        prog_fold_gs(entry->prog);
        serial_put(entry->program.serial, entry->prog);
        g_bound_program_cache[cache_key] = entry;
        atomic_fetch_add_explicit(&g_bound_program_count, 1, memory_order_relaxed);
        { static int once; if (!once++) MLOG("per-stage program binding: first combined program (vertex %s fragment %s)", entry->prog->sh[0] ? entry->prog->sh[0]->key : "-", entry->prog->sh[1] ? entry->prog->sh[1]->key : "-"); }
    }
    return &entry->program;
}
static void udc_note_draw(NvProg *pr) {

    // Same target, all-zero set generation unchanged: it was already removed last time, so no need to scan the set again (the set has up to 4096 entries, scanned once per color attachment per draw)
    static uint64_t del_ser[8], del_gen;
    if (R.has_rt) for (int i = 0; i < R.rt.n && i < 8; i++) if (R.rt.color[i]) { uint64_t sr = R.rt.color[i]->serial;
        if (!g_la_cache || del_gen != g_zskip_gen || del_ser[i] != sr) { zskip_del(sr); del_ser[i] = sr; }
        g_zskip_cand_ok[i] = 0; }   // Zero-source set: once really drawn it is no longer all zeros
    del_gen = g_zskip_gen;

    // Whether each stage of this program has image slots / writable image slots, and which colors the fragment stage writes: depends only on sh[], computed once and attached to the program (recomputed when sh[] changes)
    if (pr && g_la_cache) {
        int ok = pr->fl_ok; for (int st = 0; ok && st < 6; st++) if (pr->fl_sh[st] != pr->sh[st]) ok = 0;
        if (!ok) { pr->fl_img = pr->fl_img_w = 0; pr->fl_cmask = 0;
            for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; pr->fl_sh[st] = sh; if (!sh) continue; if (st == 1) pr->fl_cmask = sh->rf.color_mask;
                for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].kind == NVN_RB_IMG) { pr->fl_img = 1; if (sh->rf.b[i].rw) pr->fl_img_w = 1; } }
            pr->fl_ok = 1; }
    }
    if (pr && g_la_cache) { if (pr->fl_img) zskip_clear(); }
    else if (pr) for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue; for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].kind == NVN_RB_IMG) { zskip_clear(); break; } }   // Image slots are writable: conservatively clear
    int image_write = 0;
    uint32_t shader_color_mask = 0;
    if (pr && g_la_cache) { image_write = pr->fl_img_w; shader_color_mask = pr->fl_cmask; }
    else if (pr) for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;
        if (st == 1) shader_color_mask = sh->rf.color_mask;
        for (int i = 0; i < sh->rf.nb; i++) if (sh->rf.b[i].kind == NVN_RB_IMG && sh->rf.b[i].rw) { image_write = 1; break; }
        if (image_write) break;
    }
    for (int i = 0; i < g_nsd; i++) {
        sd_att *t = &g_sd[i];
        if (image_write) { t->wrote = 1; t->wwhy = 1; }
        else if (t->depth == 0 && t->idx >= 0 && (((S.chmask.mask >> (4 * t->idx)) & 0xf) || (shader_color_mask >> t->idx & 1))) { t->wrote = 1; t->wwhy = (S.chmask.mask >> (4 * t->idx)) & 0xf ? 2 : 3; }
        else if (t->depth == 1 && S.ds.dw) { t->wrote = 1; t->wwhy = 4; }
        else if (t->depth == 2 && S.ds.st && (S.st_wmask[0] || S.st_wmask[1])
                && (S.ds.fail_f != NVN_STENCIL_OP_KEEP || S.ds.zfail_f != NVN_STENCIL_OP_KEEP || S.ds.zpass_f != NVN_STENCIL_OP_KEEP
                    || S.ds.fail_b != NVN_STENCIL_OP_KEEP || S.ds.zfail_b != NVN_STENCIL_OP_KEEP || S.ds.zpass_b != NVN_STENCIL_OP_KEEP)) { t->wrote = 1; t->wwhy = 5; }
    }
}
static void state_reset(void) {
    memset(&S, 0, sizeof S);
    g_pso_dirty = g_dss_dirty = 1;
    for (int i = 0; i < 8; i++) S.blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    S.color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; S.chmask.mask = 0xffffffffu;
    S.poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    S.ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    S.st_vmask[0] = S.st_vmask[1] = S.st_wmask[0] = S.st_wmask[1] = 0xff; S.dr[1] = 1;
}

static MTLVertexFormat vfmt(int f) {
    switch (f) {
        case 0x01: return MTLVertexFormatUCharNormalized; case 0x02: return MTLVertexFormatCharNormalized; case 0x03: return MTLVertexFormatUChar; case 0x04: return MTLVertexFormatChar;
        case 0x05: return MTLVertexFormatHalf; case 0x06: return MTLVertexFormatUShortNormalized; case 0x07: return MTLVertexFormatShortNormalized;
        case 0x08: return MTLVertexFormatUShort; case 0x09: return MTLVertexFormatShort; case 0x0A: return MTLVertexFormatFloat; case 0x0B: return MTLVertexFormatUInt; case 0x0C: return MTLVertexFormatInt;
        case 0x0D: return MTLVertexFormatUChar2Normalized; case 0x0E: return MTLVertexFormatChar2Normalized; case 0x0F: return MTLVertexFormatUChar2; case 0x10: return MTLVertexFormatChar2;
        case 0x11: return MTLVertexFormatHalf2; case 0x12: return MTLVertexFormatUShort2Normalized; case 0x13: return MTLVertexFormatShort2Normalized; case 0x14: return MTLVertexFormatUShort2;
        case 0x15: return MTLVertexFormatShort2; case 0x16: return MTLVertexFormatFloat2; case 0x17: return MTLVertexFormatUInt2; case 0x18: return MTLVertexFormatInt2;
        case 0x19: return MTLVertexFormatUChar3Normalized; case 0x1A: return MTLVertexFormatChar3Normalized; case 0x1B: return MTLVertexFormatUChar3; case 0x1C: return MTLVertexFormatChar3;
        case 0x1D: return MTLVertexFormatHalf3; case 0x1E: return MTLVertexFormatUShort3Normalized; case 0x1F: return MTLVertexFormatShort3Normalized; case 0x20: return MTLVertexFormatUShort3;
        case 0x21: return MTLVertexFormatShort3; case 0x22: return MTLVertexFormatFloat3; case 0x23: return MTLVertexFormatUInt3; case 0x24: return MTLVertexFormatInt3;
        case 0x25: return MTLVertexFormatUChar4Normalized; case 0x26: return MTLVertexFormatChar4Normalized; case 0x27: return MTLVertexFormatUChar4; case 0x28: return MTLVertexFormatChar4;
        case 0x29: return MTLVertexFormatHalf4; case 0x2A: return MTLVertexFormatUShort4Normalized; case 0x2B: return MTLVertexFormatShort4Normalized; case 0x2C: return MTLVertexFormatUShort4;
        case 0x2D: return MTLVertexFormatShort4; case 0x2E: return MTLVertexFormatFloat4; case 0x2F: return MTLVertexFormatUInt4; case 0x30: return MTLVertexFormatInt4;
        case 0x3D: return MTLVertexFormatUInt1010102Normalized; case 0x3F: return MTLVertexFormatFloatRG11B10; case 0x9A: return MTLVertexFormatUChar4Normalized_BGRA;
    }
    return MTLVertexFormatInvalid;
}
static MTLBlendFactor bfac(int f) {
    switch (f) { case 1: return MTLBlendFactorZero; case 2: return MTLBlendFactorOne; case 3: return MTLBlendFactorSourceColor; case 4: return MTLBlendFactorOneMinusSourceColor;
        case 5: return MTLBlendFactorSourceAlpha; case 6: return MTLBlendFactorOneMinusSourceAlpha; case 7: return MTLBlendFactorDestinationAlpha; case 8: return MTLBlendFactorOneMinusDestinationAlpha;
        case 9: return MTLBlendFactorDestinationColor; case 10: return MTLBlendFactorOneMinusDestinationColor; case 11: return MTLBlendFactorSourceAlphaSaturated;
        case 0x10: return MTLBlendFactorSource1Color; case 0x11: return MTLBlendFactorOneMinusSource1Color; case 0x12: return MTLBlendFactorSource1Alpha; case 0x13: return MTLBlendFactorOneMinusSource1Alpha;
        case 0x61: return MTLBlendFactorBlendColor; case 0x62: return MTLBlendFactorOneMinusBlendColor; case 0x63: return MTLBlendFactorBlendAlpha; case 0x64: return MTLBlendFactorOneMinusBlendAlpha; }
    return MTLBlendFactorOne;
}
static MTLBlendOperation bop(int e) { switch (e) { case 2: return MTLBlendOperationSubtract; case 3: return MTLBlendOperationReverseSubtract; case 4: return MTLBlendOperationMin; case 5: return MTLBlendOperationMax; } return MTLBlendOperationAdd; }


// Pipeline: cached by (program, target formats, blend / write mask, vertex layout, sample count)
static const uint8_t *g_pso_lmap; static id<MTLFunction> g_pso_fsfn; static pso_fz_desc g_pso_fz, g_fz_desc;
static os_unfair_lock g_fz_fn_lock = OS_UNFAIR_LOCK_INIT;
static NSMutableDictionary<NSString *, id<MTLFunction>> *g_bf_fn_cache;
static NSMutableDictionary<NSString *, id<MTLLibrary>> *g_bf_lib_cache;
// nxnative@0f99c35: preserve integer bit patterns at integer color attachments.
static int pf_int_kind(MTLPixelFormat f) {
    switch (f) {
        case MTLPixelFormatR8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatR32Uint: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRG32Uint:
        case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGB10A2Uint: case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA32Uint: return 1;
        case MTLPixelFormatR8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatR32Sint: case MTLPixelFormatRG8Sint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRG32Sint:
        case MTLPixelFormatRGBA8Sint: case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA32Sint: return 2;
        default: return 0;
    }
}
static int pso_all_uint(const pso_key *k, NvShader *fs, int merged) {
    int n = 0;
    for (int i = 0; i < 8; i++) {
        if (!k->cfmt[i]) continue;
        int l = merged ? k->lmap[i] : i;
        if (l == 0xff || (fs->color_known && !(fs->rf.color_mask >> l & 1))) continue;
        if (pf_int_kind((MTLPixelFormat)k->cfmt[i]) != 1) return 0;
        n++;
    }
    return n > 0;
}
static int pso_mixed_uint(const pso_key *k, NvShader *fs, int merged) {
    int ints = 0, floats = 0;
    for (int i = 0; i < 8; i++) {
        if (!k->cfmt[i]) continue;
        int l = merged ? k->lmap[i] : i;
        if (l == 0xff || (fs->color_known && !(fs->rf.color_mask >> l & 1))) continue;
        int u = pf_int_kind((MTLPixelFormat)k->cfmt[i]) == 1;
        if (u != (l < 8 && (fs->mask_m >> l & 1))) return 0;
        if (u) ints++; else floats++;
    }
    return ints && floats;
}
static const char *pso_fout_hint(NvShader *fs, const uint32_t cfmt[8], char *b, size_t n) {
    if (!fs || strlen(fs->key) < 16) return "";
    int o = snprintf(b, n, "; if its outputs are integers, add to shader_facts_manual.tsv: %.16s\t{\"fout\":{", fs->key), any = 0;
    for (int c = 0; c < 8; c++) {
        int kind = !fs->color_known || fs->rf.color_mask >> c & 1 ? pf_int_kind((MTLPixelFormat)cfmt[c]) : 0;
        if (kind && o < (int)n) o += snprintf(b + o, n - o, "%s\"%d\":\"%s\"", any++ ? "," : "", c, kind == 1 ? "uint" : "sint");
    }
    if (!any) return "";
    if (o < (int)n) snprintf(b + o, n - o, "}}");
    return b;
}
static MTLRenderPipelineDescriptor *pso_desc(const pso_key *k, NvShader *vs, NvShader *fs, id<MTLFunction> fsfn, int merged) {
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction = vs->fn; d.fragmentFunction = fsfn ? fsfn : fs ? fs->fn : nil;

    // Diagnostic DM_NVN_HILITE=fragment shader key,...: these draws use a solid-color fragment instead (keys 1–6 in order magenta / green / cyan / yellow / blue / white), to see where the geometry lands
    if (fs) { static char keys[6][24]; static int nk = -1; static id<MTLLibrary> hl;
        if (nk < 0) { nk = 0; const char *e = getenv("DM_NVN_HILITE");
            for (const char *q = e; q && *q && nk < 6; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) { size_t n = strcspn(q, ","); if (n > 23) n = 23; memcpy(keys[nk], q, n); keys[nk++][n] = 0; } }
        for (int i = 0; i < nk; i++) if (!strncmp(fs->key, keys[i], strlen(keys[i]))) {
            if (!hl) { NSError *e = nil; hl = [g_dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nstruct O { float4 c0 [[color(0)]]; float4 c1 [[color(1)]]; float4 c2 [[color(2)]]; };\n"
                "fragment O hl0() { return {float4(1,0,1,1),float4(1,0,1,1),float4(1,0,1,1)}; } fragment O hl1() { return {float4(0,1,0,1),float4(0,1,0,1),float4(0,1,0,1)}; }\n"
                "fragment O hl2() { return {float4(0,1,1,1),float4(0,1,1,1),float4(0,1,1,1)}; } fragment O hl3() { return {float4(1,1,0,1),float4(1,1,0,1),float4(1,1,0,1)}; }\n"
                "fragment O hl4() { return {float4(0,0,1,1),float4(0,0,1,1),float4(0,0,1,1)}; } fragment O hl5() { return {float4(1,1,1,1),float4(1,1,1,1),float4(1,1,1,1)}; }\n" options:nil error:&e];
                if (!hl) MLOG("[RED] HILITE shader compilation failed %s", e.localizedDescription.UTF8String); }
            if (hl) d.fragmentFunction = [hl newFunctionWithName:[NSString stringWithFormat:@"hl%d", i]];
            break; } }
    if (fs && !fsfn && d.fragmentFunction == fs->fn) {
        if (fs->fn_u && pso_all_uint(k, fs, merged)) d.fragmentFunction = fs->fn_u;
        else if (fs->fn_m && pso_mixed_uint(k, fs, merged)) d.fragmentFunction = fs->fn_m;
    }
    if (vs->layer_out) d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    d.rasterSampleCount = k->samples ? k->samples : 1;
    d.alphaToCoverageEnabled = k->a2c;
    for (int i = 0; i < 8; i++) {
        if (!k->cfmt[i]) continue;
        MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[i];
        c.pixelFormat = (MTLPixelFormat)k->cfmt[i];
        int l = merged ? k->lmap[i] : i;
        uint32_t m = l == 0xff ? 0 : (k->chmask >> (4 * l)) & 0xf;
        int unwritten = fs && fs->color_known && !fsfn && l != 0xff && !(fs->rf.color_mask >> l & 1);
        if (unwritten) m = 0;
        c.writeMask = (m & 1 ? MTLColorWriteMaskRed : 0) | (m & 2 ? MTLColorWriteMaskGreen : 0) | (m & 4 ? MTLColorWriteMaskBlue : 0) | (m & 8 ? MTLColorWriteMaskAlpha : 0);
        if (l != 0xff && k->blend_en >> l & 1 && !unwritten && !pf_int_kind(c.pixelFormat)) {
            c.blendingEnabled = YES;
            c.sourceRGBBlendFactor = bfac(k->blend[l].src_rgb); c.destinationRGBBlendFactor = bfac(k->blend[l].dst_rgb);
            c.sourceAlphaBlendFactor = bfac(k->blend[l].src_a); c.destinationAlphaBlendFactor = bfac(k->blend[l].dst_a);
            c.rgbBlendOperation = bop(k->blend[l].eq_rgb); c.alphaBlendOperation = bop(k->blend[l].eq_a);
        }
    }
    d.depthAttachmentPixelFormat = (MTLPixelFormat)k->dfmt; d.stencilAttachmentPixelFormat = (MTLPixelFormat)k->sfmt;
    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
    const nvn_refl *rf = &vs->rf;
    for (int v = 0; v < rf->nv; v++) {
        int loc = rf->v[v].loc, a = rf->v[v].nvn;   // Metal slot loc ← NVN attribute a
        MTLVertexFormat vf = a < 16 ? vfmt(k->attr[a].fmt) : MTLVertexFormatInvalid;

        // When a >= S.nattr, k.attr[a] keeps the 0 from memset, vfmt(0) is Invalid, the same fallback branch is taken and the RED_ONCE condition of fmt being non-0 does not hold.
        if (a < 16 && k->attr[a].fmt && vf == MTLVertexFormatInvalid) RED_ONCE(15, "Vertex attribute format %#x is not mapped", k->attr[a].fmt);
        if (vf == MTLVertexFormatInvalid) {
            int ct = rf->v[v].ct;
            vd.attributes[loc].format = ct == NVN_CT_UINT ? MTLVertexFormatUInt4 : ct == NVN_CT_INT ? MTLVertexFormatInt4 : MTLVertexFormatFloat4;
            vd.attributes[loc].offset = 0; vd.attributes[loc].bufferIndex = 30;
            vd.layouts[30].stride = 16; vd.layouts[30].stepFunction = MTLVertexStepFunctionConstant; vd.layouts[30].stepRate = 0;
            continue;
        }
        int st = k->attr[a].stream;
        vd.attributes[loc].format = vf; vd.attributes[loc].offset = k->attr[a].offset; vd.attributes[loc].bufferIndex = 16 + st;
        vd.layouts[16 + st].stride = k->stream[st].stride ? k->stream[st].stride : 4;
        vd.layouts[16 + st].stepFunction = k->stream[st].stride == 0 ? MTLVertexStepFunctionConstant : k->stream[st].divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
        vd.layouts[16 + st].stepRate = k->stream[st].stride == 0 ? 0 : k->stream[st].divisor ? k->stream[st].divisor : 1;
    }
    d.vertexDescriptor = vd;
    return d;
}

// Pipeline key: current state S + pass formats (normal pass; a merged pass additionally carries lmap / specialized function, filled in by pso_get)
static void pso_key_fill(pso_key *k, NvShader *vs, NvShader *fs, const uint32_t cfmt[8], uint32_t dfmt, uint32_t sfmt, uint32_t samples, int lmap) {
    memset(k, 0, sizeof *k);
    k->vs = (__bridge void *)vs; k->fs = (__bridge void *)fs;
    memcpy(k->cfmt, cfmt, sizeof k->cfmt); k->dfmt = dfmt; k->sfmt = sfmt; k->samples = samples;
    k->blend_en = S.color.blend_en; k->chmask = S.chmask.mask; k->a2c = S.ms.a2c;
    for (int i = 0; i < 8; i++) if (k->cfmt[i] || lmap) k->blend[i] = S.blend[i];
    const nvn_refl *rf = &vs->rf;
    for (int v = 0; v < rf->nv; v++) { int a = rf->v[v].nvn; if (a < 16 && a < S.nattr) { k->attr[a] = S.attr[a]; int st = S.attr[a].stream; if (st < 16) k->stream[st] = S.stream[st]; } }
}
// ---- Parallel precompilation: previously, newly encountered pipelines were compiled one by one synchronously on the submission thread; 9 consecutive misses in one frame = 284 ms (heavy scene on a real device).
// Now: on the first pipeline miss in a submission, scan the remaining commands once (replay state only, no drawing) and send any later missing ordinary-pass pipelines to the background for parallel compilation;

// ---- Parallel precompile: pipelines met for the first time used to be compiled synchronously one by one on the submit thread; 9 misses in a row in one frame = 284 ms (heavy scene on a real device).
// Now: on the first pipeline miss in a submission, scan ahead once along the remaining commands (replaying state only, no drawing) and kick all the normal-pass pipelines that will also miss later to the background to compile in parallel;
// when the submit thread reaches that draw it only waits for 'the remaining time not yet compiled'. The compiled pipelines are identical to synchronously compiled ones (same descriptor). Pipeline keys of merged passes / specialized fragments depend on the planning result at replay time, are not predicted, and are compiled synchronously as before.
#define PSO_PEND_MAX 384
static nvn_hmap g_pso = {.klen = sizeof(pso_key)}, g_dss;
static struct { pso_key k; id<MTLRenderPipelineState> ps; int used, done; } g_pp[PSO_PEND_MAX];
static pthread_mutex_t g_pp_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_pp_cv = PTHREAD_COND_INITIALIZER;
static uint64_t g_pp_submit_mark, g_pp_kicked, g_pp_taken;   // Whether this submission has already been predicted (by submission serial); cumulative kicked / taken counts
static void pso_predict_rest(void);
static int pso_pend_find(const pso_key *k) { for (int i = 0; i < PSO_PEND_MAX; i++) if (g_pp[i].used && !memcmp(&g_pp[i].k, k, sizeof *k)) return i; return -1; }
static void pso_pend_kick(const pso_key *k, NvShader *vs, NvShader *fs) {   // Called by the submit thread: kick only if it is neither in the cache nor in the table
    if (hm_get_id(&g_pso, k)) return;
    pthread_mutex_lock(&g_pp_mu);
    int at = pso_pend_find(k), slot = -1;
    if (at < 0) for (int i = 0; i < PSO_PEND_MAX; i++) if (!g_pp[i].used) { slot = i; break; }
    if (at < 0 && slot >= 0) { g_pp[slot].k = *k; g_pp[slot].ps = nil; g_pp[slot].done = 0; g_pp[slot].used = 1; }
    pthread_mutex_unlock(&g_pp_mu);
    if (at >= 0 || slot < 0) return;
    MTLRenderPipelineDescriptor *d = pso_desc(k, vs, fs, nil, 0); g_pp_kicked++;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSError *e = nil; id<MTLRenderPipelineState> ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
        pthread_mutex_lock(&g_pp_mu); g_pp[slot].ps = ps; g_pp[slot].done = 1; pthread_cond_broadcast(&g_pp_cv); pthread_mutex_unlock(&g_pp_mu);
    });
}
static id<MTLRenderPipelineState> pso_pend_take(const pso_key *k) {   // Called by the submit thread: if it is in the table, wait for it to finish compiling and take it; otherwise return nil
    pthread_mutex_lock(&g_pp_mu);
    int at = pso_pend_find(k); id<MTLRenderPipelineState> ps = nil;
    if (at >= 0) { while (!g_pp[at].done) pthread_cond_wait(&g_pp_cv, &g_pp_mu); ps = g_pp[at].ps; g_pp[at].ps = nil; g_pp[at].used = 0; }
    pthread_mutex_unlock(&g_pp_mu);
    return ps;
}
static __unsafe_unretained id<MTLRenderPipelineState> pso_get(NvProg *pr, MTLRenderPassDescriptor *unused) {
    (void)unused;
    __unsafe_unretained NvShader *vs = pr->sh[0], *fs = pr->sh[1];
    if (!vs) { if (g_draw_trace) g_draw_trace->pipeline = "missing-vs"; RED_ONCE(13, "Program has no vertex shader (or it is not mapped)"); return nil; }
    pso_key k; pso_key_fill(&k, vs, fs, g_pass.cfmt, g_pass.dfmt, g_pass.sfmt, g_pass.samples, g_pso_lmap != NULL);
    if (g_pso_lmap) { memcpy(k.lmap, g_pso_lmap, 8); k.fsfn = (__bridge void *)g_pso_fsfn; }
    __unsafe_unretained id<MTLRenderPipelineState> cached = hm_get_id(&g_pso, &k);
    if (cached) { if (g_draw_trace) g_draw_trace->pipeline = "cache-hit"; return cached; }
    id<MTLRenderPipelineState> ps;
    NSError *e = nil; uint64_t bt = bp_now();
    ps = g_pso_lmap ? nil : pso_pend_take(&k);
    if (ps) { g_pp_taken++; if (g_draw_trace) g_draw_trace->pipeline = "prefetched"; }
    else {
        if (g_pp_submit_mark != g_sub_serial) { g_pp_submit_mark = g_sub_serial; pso_predict_rest(); }   // First miss in this submission: first kick the later ones out to compile in parallel, then synchronously compile the one at hand
        MTLRenderPipelineDescriptor *d = pso_desc(&k, vs, fs, g_pso_fsfn, g_pso_lmap != NULL);
        ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
        if (ps && g_draw_trace) g_draw_trace->pipeline = "built";
    }
    g_bp_pso_n++; g_bp_pso_ns += bp_now() - bt;
    if (!ps) { char hint[320]; if (g_draw_trace) g_draw_trace->pipeline = "build-failed"; MLOG("[RED] Failed to create render pipeline FS=%s: %s%s", fs ? fs->key : "-", e.localizedDescription.UTF8String, pso_fout_hint(fs, k.cfmt, hint, sizeof hint)); return nil; }
    pso_skey sk; char shex[sizeof sk * 2 + 1]; pso_skey_from(&k, vs, fs, g_pso_lmap != NULL, g_pso_lmap ? &g_pso_fz : NULL, &sk); pso_skey_hex(&sk, shex);
    pso_record("R\t%s\n", shex);
    pso_user_record("R\t%s\n", shex);
    hm_put_id(&g_pso, &k, ps);
    return ps;
}
static NSString *warm_key(const char *s, size_t n) {
    size_t z = strnlen(s, n);
    return [[NSString alloc] initWithBytes:s length:z encoding:NSUTF8StringEncoding];
}
static NvShader *warm_shader(NSMutableDictionary<NSString *, NvShader *> *dict, NSString *key, int stage) {
    NvShader *sh = dict[key];
    if (!sh) { sh = shader_load(key, stage, 0); if (sh) dict[key] = sh; }
    return sh;
}
static void warm_enqueue(WarmItem *item) {
    os_unfair_lock_lock(&g_warm_lock);
    [g_warm_items addObject:item];
    atomic_store_explicit(&g_warm_ready, 1, memory_order_release);
    os_unfair_lock_unlock(&g_warm_lock);
}
static void warm_worker(NSArray<NSString *> *paths) {
    @autoreleasepool {
        NSMutableDictionary<NSString *, NvShader *> *dict = [NSMutableDictionary dictionary];
        NSMutableSet<NSString *> *seen = [NSMutableSet set];
        for (NSString *path in paths) {
            NSString *txt = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            if (!txt) { MLOG("[RED] Failed to read pipeline warmup manifest: %s", path.UTF8String); continue; }
            for (NSString *line in [txt componentsSeparatedByString:@"\n"]) {
                if (!line.length || [line hasPrefix:@"#"] || [seen containsObject:line]) continue;
                [seen addObject:line];
                NSArray<NSString *> *c = [line componentsSeparatedByString:@"\t"];
                if (!c.count) continue;
                if ([c[0] isEqualToString:@"S"] && c.count >= 3) {
                    NSString *key = c[1]; NvShader *sh = warm_shader(dict, key, c[2].intValue);
                    if (!sh) continue;
                    WarmItem *item = [WarmItem new]; item->kind = 0; item->key = key; item->shader = sh;
                    warm_enqueue(item); g_warm_shaders++;
                } else if ([c[0] isEqualToString:@"R"] && c.count >= 2) {
                    pso_skey sk; if (pso_skey_parse(c[1].UTF8String, &sk)) continue;
                    NSString *vkey = warm_key(sk.vs, sizeof sk.vs), *fkey = warm_key(sk.fs, sizeof sk.fs);
                    NvShader *vs = vkey.length ? warm_shader(dict, vkey, sk.vs_stage) : nil;
                    NvShader *fs = fkey.length ? warm_shader(dict, fkey, sk.fs_stage) : nil;
                    if (!vs || (fkey.length && !fs)) continue;
                    id<MTLFunction> fsfn = nil;
                    if (sk.fz) {
                        if (sk.fz_desc.kind == 1 && fs) { int fx[32]; for (int i = 0; i < 32; i++) fx[i] = sk.fz_desc.fx[i]; fsfn = bf_fn(fs, fx, sk.fz_desc.fbmask, sk.k.lmap); }
                                                if (!fsfn) { g_warm_skip_fz++; continue; }
                    }
                    pso_key k = sk.k; k.vs = (__bridge void *)vs; k.fs = (__bridge void *)fs; k.fsfn = (__bridge void *)fsfn;
                    NSError *e = nil; id ps = [g_dev newRenderPipelineStateWithDescriptor:pso_desc(&k, vs, fs, fsfn, sk.merged) error:&e];
                    if (!ps) { MLOG("[RED] Failed to create render pipeline in background: %s", e.localizedDescription.UTF8String); continue; }
                    WarmItem *item = [WarmItem new]; item->kind = 1; item->vs_key = vkey; item->fs_key = fkey; item->vs = vs; item->fs = fs; item->fsfn = fsfn; item->pso = k; item->object = ps;
                    warm_enqueue(item); g_warm_rps++;
                } else if ([c[0] isEqualToString:@"C"] && c.count >= 2) {
                    NSString *key = c[1]; if (!key.length) continue;
                    NvShader *cs = warm_shader(dict, key, 5);
                    if (!cs) continue;
                    NSError *e = nil; id ps = [g_dev newComputePipelineStateWithFunction:cs->fn error:&e];
                    if (!ps) { MLOG("[RED] Failed to create compute pipeline in background: %s", e.localizedDescription.UTF8String); continue; }
                    WarmItem *item = [WarmItem new]; item->kind = 2; item->key = key; item->shader = cs; item->object = ps;
                    warm_enqueue(item); g_warm_cps++;
                }
            }
        }
        MLOG("Pipeline warmup complete: %llu shaders / %llu render pipelines (skipped %llu specialized fragment shaders) / %llu compute pipelines, background time %.1f ms",
             g_warm_shaders, g_warm_rps, g_warm_skip_fz, g_warm_cps, (bp_now() - g_warm_t0) / 1e6);
    }
    atomic_store(&g_warm_running, 0);
}
// Wait for background warmup to finish (up to timeout_ms): return the actual time waited in milliseconds; return 0 if none is running. On a fresh install, the system shader cache is cold; if warmup is slower than the player's entry into the game,

// Wait for the background warm-up to finish (at most timeout_ms): returns how many milliseconds were actually waited, 0 if it is not running. On first install the system shader cache is fully cold; if warm-up is slower than the player entering a run,
// pipelines not yet warmed are compiled on the spot on the submit thread = in-run stutter; the loading screen on entering a run blocks the player anyway, so let it load a little longer (called where hle_nvn.c presents).
int mtl_warm_wait(int timeout_ms) {
    if (!atomic_load(&g_warm_running)) return 0;
    uint64_t t0 = bp_now();
    while (atomic_load(&g_warm_running) && (bp_now() - t0) / 1000000ull < (uint64_t)timeout_ms) usleep(2000);
    return (int)((bp_now() - t0) / 1000000ull) + 1;
}
static void warm_start(void) {
    if (g_warm_started++) return;
    const char *e = getenv("DM_PSO_MANIFEST");
    if (e && !strcmp(e, "off")) return;
    NSArray<NSString *> *paths;
    if (e && *e) {
        paths = @[@(e)];
    } else {
        NSString *seed = [g_shmap_path.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"pso_manifest.tsv"];
        NSMutableArray<NSString *> *all = [NSMutableArray arrayWithObject:seed];
        NSString *user = pso_user_path();
        if (user && [NSFileManager.defaultManager fileExistsAtPath:user]) [all addObject:user];
        if (![NSFileManager.defaultManager fileExistsAtPath:seed]) return;
        paths = all;
    }
    if (!g_warm_items) g_warm_items = [NSMutableArray array];
    g_warm_t0 = bp_now();
    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);   // Not optional background work: it decides whether the pipelines are ready before the player enters a run
    dispatch_queue_t q = dispatch_queue_create("dm.nvn.pso-warm", attr);
    atomic_store(&g_warm_running, 1);
    dispatch_async(q, ^{ warm_worker(paths); });
}
static id<MTLDepthStencilState> dss_get(int has_depth, int has_stencil) {
    struct { ds_st ds; uint32_t vm[2], wm[2]; int hd, hs; } k;
    memset(&k, 0, sizeof k); k.ds = S.ds; memcpy(k.vm, S.st_vmask, 8); memcpy(k.wm, S.st_wmask, 8); k.hd = has_depth; k.hs = has_stencil;
    if (!g_dss.klen) g_dss.klen = sizeof k;
    id<MTLDepthStencilState> s = hm_get_id(&g_dss, &k);
    if (s) return s;
    MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
    if (has_depth && S.ds.dt) { d.depthCompareFunction = (MTLCompareFunction)(S.ds.df - 1); d.depthWriteEnabled = S.ds.dw; }
    if (has_stencil && S.ds.st) {
        MTLStencilDescriptor *f = [MTLStencilDescriptor new], *b = [MTLStencilDescriptor new];
        f.stencilCompareFunction = (MTLCompareFunction)(S.ds.sf_f - 1); f.stencilFailureOperation = (MTLStencilOperation)(S.ds.fail_f - 1);
        f.depthFailureOperation = (MTLStencilOperation)(S.ds.zfail_f - 1); f.depthStencilPassOperation = (MTLStencilOperation)(S.ds.zpass_f - 1);
        f.readMask = S.st_vmask[0]; f.writeMask = S.st_wmask[0];
        b.stencilCompareFunction = (MTLCompareFunction)(S.ds.sf_b - 1); b.stencilFailureOperation = (MTLStencilOperation)(S.ds.fail_b - 1);
        b.depthFailureOperation = (MTLStencilOperation)(S.ds.zfail_b - 1); b.depthStencilPassOperation = (MTLStencilOperation)(S.ds.zpass_b - 1);
        b.readMask = S.st_vmask[1]; b.writeMask = S.st_wmask[1];
        d.frontFaceStencil = f; d.backFaceStencil = b;
    }
    s = [g_dev newDepthStencilStateWithDescriptor:d];
    hm_put_id(&g_dss, &k, s);
    return s;
}
// Parameter buffer: populate each field from B reflection (uniform slots r, separate texture / sampler slots r, storage buffer slots r, driver sections c0 / c2, c1, support_buffer)

// Argument buffer: filled item by item per the B reflection (uniform slot r, separate texture / sampler slot r, storage buffer slot r, driver segments c0 / c2, c1, support_buffer)
// Unbound resources read 0: give all-zero dummy resources (NVN descriptors 0..255 are reserved for the driver; the game binding handle 0 means 'this slot is empty')
static uint64_t g_dummy_tex_rid[8][4], g_dummy_img_rid[8][4], g_dummy_tbuf_rid[4];
static id<MTLTexture> dummy_tex(int dtt, int dpf) {
    static __strong id<MTLTexture> cache[8][4];
    MTLTextureType tt = (MTLTextureType)dtt;
    MTLPixelFormat pf = dpf == NVN_DPF_DEPTH ? MTLPixelFormatDepth32Float : dpf == NVN_DPF_UINT ? MTLPixelFormatRGBA32Uint : dpf == NVN_DPF_SINT ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA8Unorm;
    id<MTLTexture> t = cache[dtt & 7][dpf & 3];
    if (t) return t;
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = tt; d.pixelFormat = pf; d.width = d.height = d.depth = 1; d.storageMode = MTLStorageModeShared; d.usage = MTLTextureUsageShaderRead;
    t = [g_dev newTextureWithDescriptor:d];   // Shared storage is all zeros when newly created
    if (pf != MTLPixelFormatDepth32Float) { uint8_t z[64] = {0}; NSUInteger faces = tt == MTLTextureTypeCube || tt == MTLTextureTypeCubeArray ? 6 : 1;
        for (NSUInteger f = 0; f < faces; f++) [t replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:f withBytes:z bytesPerRow:16 bytesPerImage:16]; }
    cache[dtt & 7][dpf & 3] = t; g_dummy_tex_rid[dtt & 7][dpf & 3] = t.gpuResourceID._impl; res_add(t);
    return t;
}

// Write sink for unbound image slots: kept separate from the dummy textures that read 0 (writes must not pollute them); one is shared per same type and same pixel class
static id<MTLTexture> dummy_img(int dtt, int dpf) {
    static __strong id<MTLTexture> cache[8][4];
    id<MTLTexture> t = cache[dtt & 7][dpf & 3]; if (t) return t;
    MTLTextureDescriptor *d = [MTLTextureDescriptor new];
    d.textureType = (MTLTextureType)dtt; d.pixelFormat = dpf == NVN_DPF_UINT ? MTLPixelFormatRGBA32Uint : dpf == NVN_DPF_SINT ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA16Float;
    d.width = d.height = d.depth = 1; d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    t = [g_dev newTextureWithDescriptor:d]; cache[dtt & 7][dpf & 3] = t; g_dummy_img_rid[dtt & 7][dpf & 3] = t.gpuResourceID._impl; res_add(t); return t;
}
static id<MTLBuffer> dummy_buf(void);
static id<MTLTexture> dummy_tbuf(int dpf) {   // Unbound typed buffer: all zeros (pixel class matches the shader's texture_buffer<T>)
    static __strong id<MTLTexture> cache[4];
    id<MTLTexture> t = cache[dpf & 3]; if (t) return t;
    MTLPixelFormat pf = dpf == NVN_DPF_UINT ? MTLPixelFormatRGBA32Uint : dpf == NVN_DPF_SINT ? MTLPixelFormatRGBA32Sint : MTLPixelFormatRGBA32Float;
    id<MTLBuffer> b = dummy_buf();
    MTLTextureDescriptor *d = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pf width:b.length / 16 resourceOptions:b.resourceOptions usage:MTLTextureUsageShaderRead];
    t = [b newTextureWithDescriptor:d offset:0 bytesPerRow:b.length];
    cache[dpf & 3] = t; g_dummy_tbuf_rid[dpf & 3] = t.gpuResourceID._impl; return t;
}
static id<MTLSamplerState> dummy_samp(void) {
    static id<MTLSamplerState> s; if (!s) { MTLSamplerDescriptor *d = [MTLSamplerDescriptor new]; d.supportArgumentBuffers = YES; s = [g_dev newSamplerStateWithDescriptor:d]; }
    return s;
}
static uint64_t dummy_samp_rid(void) { static uint64_t rid; if (!rid) rid = dummy_samp().gpuResourceID._impl; return rid; }
static id<MTLBuffer> dummy_buf(void) { static id<MTLBuffer> b; if (!b) { b = [g_dev newBufferWithLength:64 << 10 options:MTLResourceStorageModeShared]; res_add(b); } return b; }
static uint64_t dummy_buf_gpu(void) { static uint64_t gpu; if (!gpu) gpu = dummy_buf().gpuAddress; return gpu; }

// useResource is idempotent for the same (resource, usage, stage) within one encoder: remember what this render encoder has already declared and skip repeats (a compute encoder is created anew for each dispatch, no dedup)
static uint8_t g_args_dirty[6] = {1, 1, 1, 1, 1, 1};
static uint64_t g_args_reg_seen, g_args_dirty_seen;
static inline void args_dirty_all(void) { memset(g_args_dirty, 1, sizeof g_args_dirty); }
static inline void args_sync_dirty(void) {
    uint64_t reg = g_reg_gen, dirty = nvn_dirty_now();
    if (reg != g_args_reg_seen || dirty != g_args_dirty_seen) {
        g_args_reg_seen = reg; g_args_dirty_seen = dirty; args_dirty_all();
    }
}
static void bind_args(id<MTLCommandEncoder> en, int nvn_stage, int metal_vertex, id<MTLBuffer> ab, uint64_t gpu, size_t off, NSUInteger ai) {
    if (nvn_stage == 5) enc_c_args((id<MTLComputeCommandEncoder>)en, gpu, ai);
    else enc_r_args((id<MTLRenderCommandEncoder>)en, metal_vertex ? MTLRenderStageVertex : MTLRenderStageFragment, gpu, ai);
}
// Static constant buffer for the B path: Ryujinx support_buffer (layout in Ryujinx.Graphics.Shader/SupportBuffer.cs: alpha_test u32, is_bgra u32[8],

// Static constant buffers of route B: the Ryujinx support_buffer (layout in Ryujinx.Graphics.Shader/SupportBuffer.cs: alpha_test u32, is_bgra u32[8],
// viewport_inverse f4, viewport_size f4, frag_scale_count i32, render_scale f32[73]..., each field aligned to 16 bytes) and the all-0 driver constant segment (64 KB = the upper bound declared by shaders)
static id<MTLBuffer> support_buf(void) {
    static __strong id<MTLBuffer> b; if (b) return b;
    b = [g_dev newBufferWithLength:65536 options:MTLResourceStorageModeShared]; res_add(b);   // All 0 when newly created
    const int render_scale_off = 192;   // Measured MSL layout: alpha_test 0, is_bgra[8] 16, viewport_inverse 144, viewport_size 160, frag_scale_count 176, render_scale float4[73] 192 (take .x)
    for (int i = 0; i < 73; i++) *(float *)((uint8_t *)b.contents + render_scale_off + 16 * i) = 1.0f;
    return b;
}
static uint64_t support_gpu(void) { static uint64_t gpu; if (!gpu) gpu = support_buf().gpuAddress; return gpu; }
static id<MTLBuffer> zero_bank(void) { static __strong id<MTLBuffer> b; if (!b) { b = [g_dev newBufferWithLength:65536 options:MTLResourceStorageModeShared]; res_add(b); } return b; }
static uint64_t zero_bank_gpu(void) { static uint64_t gpu; if (!gpu) gpu = zero_bank().gpuAddress; return gpu; }
// Shift the entire UI draw downward (true full-screen on iPad, with UI rectangles not flush with the top of the image): add a vertical clip-space offset to the projection matrix at the start of vertex-stage cb0, then bind a modified copy.
// Why not move the viewport: the viewport's top edge is the clip volume boundary, so host-raised top-aligned controls (UI coordinates with negative y) would be clipped entirely.

// Shift all UI draws down as a whole (iPad true fullscreen, the UI rect does not touch the top edge of the picture): add a clip-space vertical offset to the projection matrix at the start of the vertex stage's cb0, and bind a modified copy instead.
// Moving the viewport is not used: the viewport's top edge is the edge of the clip volume, so top-anchored widgets that the host lifts back to the real top of the picture (UI coordinate y negative) would be clipped away entirely.
// g_ui_moff is set per draw by draw() (0 = no change); the copies live in a ring buffer, and the same address in the same pass is copied only once.
static double g_ui_moff;
static int ui_mat_ok(void) { const p_bind_buf *u = &S.ubo[0][0]; void *mb; uint64_t mo; return u->addr && u->size >= 64 && u->size <= 4096 && !pend_hits(u->addr, u->size) && mtl_addr(u->addr, u->size, &mb, &mo); }
static uint64_t ui_mat_patch(const p_bind_buf *u, void *mb, uint64_t mo) {
    enum { RING = 4 << 20 }; static __strong id<MTLBuffer> ring; static uint64_t ring_gpu, la, ls, lp, lg; static size_t at; static double lo;
    if (!ring) { ring = [g_dev newBufferWithLength:RING options:MTLResourceStorageModeShared]; res_add(ring); ring_gpu = ring.gpuAddress; }
    if (la == u->addr && ls == u->size && lp == g_pass.serial && lo == g_ui_moff) return lg;
    const uint8_t *src = [(__bridge id<MTLBuffer>)mb contents]; if (!src || u->size < 64 || u->size > 4096) return 0;
    size_t n = (u->size + 255) & ~(size_t)255; if (at + n > RING) at = 0;
    uint8_t *dst = (uint8_t *)ring.contents + at; memcpy(dst, src + mo, u->size);
    float m[16], o = (float)g_ui_moff; memcpy(m, dst, 64); m[1] += o * m[3]; m[5] += o * m[7]; m[9] += o * m[11]; m[13] += o * m[15]; memcpy(dst, m, 64);
    la = u->addr; ls = u->size; lp = g_pass.serial; lo = g_ui_moff; lg = ring_gpu + at; at += n;
    return lg;
}
#if DM_NVN_M3
// Deduplicate resource declarations in the legacy interface: repeated useResource calls for the same resource within one render encoder are idempotent, but every new parameter block declares all resources it references—
// Field test at camp: 9.7 declarations per draw, 9.2 of them already declared in the current encoder, about 227 instructions each. Track which resources have been declared with which usages in this pass (g_pass.serial, one-to-one with render encoders:
// Increment the serial only before opening a render encoder) and skip those already covered. If the table is full (more distinct resources in one pass than its length), do not track them; declare as usual.

// De-duplicating resource declarations on the older API: repeated useResource for the same resource within one render encoder is idempotent, yet every newly encoded argument block declares all the resources it references once more -
// measured in the camp at 9.7 declarations per draw, of which 9.2 were already declared in this encoder, about 227 instructions per draw. Here it is recorded which resources have been declared with which usages in this pass (g_pass.serial, one-to-one with render encoders:
// the serial only increments before a render encoder is opened), and those already covered are skipped. When the table is full (more distinct resources in one pass than the table holds) nothing is recorded and declarations proceed as usual.
// Table entries do not retain resources: a declared resource is held by the encoder until the submission completes, so within one pass its address is not reused by another resource. Used only on the submission thread
enum { M3_USE_N = 1024 };
static struct { void *p; uint64_t pass; uint32_t u; } g_m3_use[M3_USE_N];
static inline int m3_use_seen(void *p, uint32_t u) {
    for (uint32_t h = (uint32_t)(((uintptr_t)p >> 4) * 0x9E3779B1u) >> 22, k = 0; k < M3_USE_N; k++, h = (h + 1) & (M3_USE_N - 1)) {
        if (g_m3_use[h].pass != g_pass.serial) { g_m3_use[h].p = p; g_m3_use[h].pass = g_pass.serial; g_m3_use[h].u = u; return 0; }
        if (g_m3_use[h].p == p) { if ((g_m3_use[h].u & u) == u) return 1; g_m3_use[h].u |= u; return 0; }
    }
    return 0;
}
#endif
static int encode_args(id<MTLCommandEncoder> en, NvShader *sh, int nvn_stage, int metal_vertex) {
    if (!sh->ae) return 1;
    uint64_t slots[NVN_REFL_MAXARG] = {0};
    args_sync_dirty();
    if (!passdump() && !g_args_dirty[nvn_stage] && sh->ab_buf && sh->ab_pass == g_pass.serial && nvn_stage != 5) {
        bind_args(en, nvn_stage, metal_vertex, sh->ab_buf, sh->ab_gpu, sh->ab_off, sh->rf.abi);
        g_stat_args_reuse++;
        return 1;
    }
#if DM_NVN_M3
    // Legacy interface: every resource accessed indirectly by a shader through a parameter block must have its usage declared on the current encoder, so the system knows its ordering dependencies with other encoders (upload copies, other passes);

    // Older API: every resource a shader accesses indirectly through an argument block must have its usage declared on this encoder, so that the system knows its ordering dependency against other encoders (upload copies, other passes);
    // without the declaration a read-only texture gets out of order with the copy that writes it (measured before the move to Metal 4). Repeated declarations within one encoder are idempotent
    #define USE(r, u) do { id<MTLResource> r_ = (id<MTLResource>)(r); if (r_) { if (nvn_stage == 5) [(id<MTLComputeCommandEncoder>)en useResource:r_ usage:(u)]; else if (!m3_use_seen((__bridge void *)r_, (uint32_t)(u))) [(id<MTLRenderCommandEncoder>)en useResource:r_ usage:(u) stages:MTLRenderStageVertex | MTLRenderStageFragment]; } \
        if (__builtin_expect(g_res_check_on, 0)) res_check((r), sh->key, reg); } while (0)
    #define USE_SAMP(r, u) do { (void)(u); if (__builtin_expect(g_res_check_on, 0)) res_check((r), sh->key, reg); } while (0)   // samplers are not resources and need no declaration
#else
    #define USE(r, u) do { (void)(u); if (__builtin_expect(g_res_check_on, 0)) res_check((r), sh->key, reg); } while (0)
    #define USE_SAMP(r, u) USE(r, u)
#endif
    // DM_NVN_PASSDUMP: on the first new encode per pass per stage, record the shader key (K) and the actual source of each binding (B: UBO address / length / first 4 floats, texture ID and serial, or “dummy”)
    // DM_NVN_PASSDUMP: on the first new encode per pass per stage, records the shader key (K) and the actual source of each binding (B: UBO address / length / first 4 floats, texture ID and serial, or dummy(dummy))
    static uint64_t bl_pass[6]; static int bl_all = -1; if (bl_all < 0) bl_all = getenv("DM_NVN_PASSDUMP_ALL") != NULL;   // Record on every new encode (otherwise only the first time per pass per stage)
    int bl = passdump() && (bl_all || bl_pass[nvn_stage] != g_stat_pass);
    if (bl) { bl_pass[nvn_stage] = g_stat_pass; PD("K %llu st%d key=%s nb=%d", g_stat_pass, nvn_stage, sh->key[0] ? sh->key : sh->builtin ? sh->builtin : "?", sh->rf.nb); }
    pthread_mutex_lock(&g_rmu);   // The texture / sampler registration tables may be being grown or rewritten by the game thread
    for (int i = 0; i < sh->rf.nb; i++) {
        const nvn_rbind *b = &sh->rf.b[i];
        int reg = b->reg, aid = b->aid;
        if (b->kind == NVN_RB_CBUF) {
            const p_bind_buf *u = reg < 16 ? &S.ubo[nvn_stage][reg] : NULL;
            void *mb; uint64_t mo, gpu;
            if (!u || !u->addr || !mtl_gpu_addr(u->addr, 1, &mb, &mo, &gpu)) { RED_ONCE(16, "Stage %d cb%d has no uniform buffer bound", nvn_stage, reg); continue; }
            use_rec("buf", u->addr, u->size, 0, NULL);
            if (nvn_stage == 0 && reg == 0 && g_ui_moff != 0) { uint64_t pg = ui_mat_patch(u, mb, mo); if (pg) gpu = pg; }
            if (bl) {
                PD("B %llu st%d cb reg%d addr=%#llx size=%llu pool=%#llx+%#llx f=%g,%g,%g,%g", g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr, (unsigned long long)u->size,
                   (unsigned long long)(u->addr - mo), (unsigned long long)mo, ((float *)u->addr)[0], ((float *)u->addr)[1], ((float *)u->addr)[2], ((float *)u->addr)[3]);

                // DM_NVN_CBDUMP=directory: also saves the whole uniform block to disk as-is (F<frame>_p<pass>_st<stage>_cb<slot>_<address>.bin), for the ground-truth side to compare byte by byte by guest address
                static const char *cbd; static int cbd_init; if (!cbd_init) { cbd = getenv("DM_NVN_CBDUMP"); cbd_init = 1; }
                if (cbd) {
                    char p[1024]; snprintf(p, sizeof p, "%s/F%llu_p%llu_st%d_cb%d_%llx.bin", cbd, (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass, nvn_stage, reg, (unsigned long long)u->addr);
                    FILE *f = fopen(p, "wb"); if (f) { fwrite((const void *)u->addr, 1, u->size, f); fclose(f); } else RED_ONCE(41, "CBDUMP cannot write %s", p);
                }
            }
            slots[aid] = gpu;
            USE((__bridge id<MTLBuffer>)mb, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_SAMP || b->kind == NVN_RB_SAMPT) {   // SAMPT: sampler part of the combined handle (texture ID | sampler ID << 20) | SAMPT: the sampler part of a combined handle (texture ID | sampler ID << 20)
            int sid = reg < 128 ? (int)((b->kind == NVN_RB_SAMPT ? S.tex[nvn_stage][reg] : S.samp[nvn_stage][reg]) >> 20) : 0;
            __unsafe_unretained id<MTLSamplerState> ss = sid ? (__bridge id<MTLSamplerState>)samp_fast(sid) : nil;
            slots[aid] = sid && sid < g_nsreg && g_srid[sid] ? g_srid[sid] : dummy_samp_rid();
            USE_SAMP(ss ? ss : dummy_samp(), MTLResourceUsageRead);
            if (bl && sid > 0) { int ok = sid < g_nsreg && g_sreg_ok[sid]; samp_par sp = ok ? g_sreg[sid] : (samp_par){0};   // g_rmu is already held in this section
                if (ok) PD("B %llu st%d samp reg%d sid=%d mag=%d min=%d (0 nearest / 1 bilinear; 2–5 with mipmaps)", g_stat_pass, nvn_stage, reg, sid, (int)sp.magf, (int)sp.minf); }
        } else if (b->kind == NVN_RB_TEX) {
            int treg = reg;
            int tid = treg >= 0 && treg < 128 ? (int)(S.tex[nvn_stage][treg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (t && tid < g_ntreg) use_rec("tex", (uint64_t)tex_storage(&g_treg[tid].t), g_treg[tid].t.L.size, tid, &g_treg[tid].t);
            if (t && passdump() && tid < g_ntreg) PD("R %llu st%d ser=%llu", g_stat_pass, nvn_stage, g_treg[tid].t.serial);
            if (bl) PD("B %llu st%d tex reg%d tid=%d %s ser=%llu fmt=%lu %lux%lux%lu view_lv=%lu+%lu view_sl=%lu array=%lu target=%lu", g_stat_pass, nvn_stage, reg, tid, t ? "Real" : "Dummy", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull,
                       (unsigned long)t.pixelFormat, (unsigned long)t.width, (unsigned long)t.height, (unsigned long)t.depth,
                       (unsigned long)(t.parentTexture ? t.parentRelativeLevel : 0), (unsigned long)t.mipmapLevelCount,
                       (unsigned long)(t.parentTexture ? t.parentRelativeSlice : 0), (unsigned long)t.arrayLength, (unsigned long)t.textureType);
            if (!t) { t = dummy_tex(b->dtt, b->dpf); trid = g_dummy_tex_rid[b->dtt & 7][b->dpf & 3]; }
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_IMG) {   // Writable texture: NVN image slot r (BindImages) ↔ D3D u_r
            int tid = reg < 32 ? (int)(S.img[nvn_stage][reg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (passdump()) PD("W %llu st%d reg%d tid=%d %s ser=%llu", g_stat_pass, nvn_stage, reg, tid, t ? "Real" : "Dummy", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull);
            if (t && !(t.usage & MTLTextureUsageShaderWrite)) { RED_ONCE(32, "Texture bound to image slot has no writable usage (texture ID %d)", tid); t = nil; }
            if (!t) { t = dummy_img(b->dtt, b->dpf); trid = g_dummy_img_rid[b->dtt & 7][b->dpf & 3]; }   // Unbound: D3D semantics discard the write - write into a dedicated dummy texture so the set that reads 0 is not polluted
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead | MTLResourceUsageWrite);
        } else if (b->kind == NVN_RB_TBUF) {   // Typed buffer: the buffer texture (target BUFFER) in texture slot r
            int treg = reg;
            int tid = treg >= 0 && treg < 128 ? (int)(S.tex[nvn_stage][treg] & 0xfffff) : 0;
            void *tv = NULL; uint64_t trid = tid ? tex_fast_rid(tid, &tv) : 0; __unsafe_unretained id<MTLTexture> t = (__bridge id<MTLTexture>)tv;
            if (t && t.textureType != MTLTextureTypeTextureBuffer) { RED_ONCE(40, "Typed buffer slot is not bound to a buffer texture (texture ID %d type %lu)", tid, (unsigned long)t.textureType); t = nil; }
            if (bl) PD("B %llu st%d tbuf reg%d tid=%d %s ser=%llu", g_stat_pass, nvn_stage, reg, tid, t ? "Real" : "Dummy", t && tid < g_ntreg ? g_treg[tid].t.serial : 0ull);
            if (!t) { t = dummy_tbuf(b->dpf); trid = g_dummy_tbuf_rid[b->dpf & 3]; }
            slots[aid] = trid;
            USE(t, MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_SBUFP || b->kind == NVN_RB_UBOP) {   // B: storage buffers / compute-stage uniforms are device pointers; the address is taken by NVN slot
            const p_bind_buf *u = reg < 16 ? (b->kind == NVN_RB_UBOP ? &S.ubo[nvn_stage][reg] : &S.ssbo[nvn_stage][reg]) : NULL;
            void *mb; uint64_t mo, gpu;
            if (u && u->addr && mtl_gpu_addr(u->addr, u->size ? u->size : 1, &mb, &mo, &gpu)) { use_rec("buf", u->addr, u->size, 0, NULL); slots[aid] = gpu; USE((__bridge id<MTLBuffer>)mb, b->rw ? MTLResourceUsageRead | MTLResourceUsageWrite : MTLResourceUsageRead); }
            else { slots[aid] = dummy_buf_gpu(); USE(dummy_buf(), MTLResourceUsageRead); }
            if (bl) PD("B %llu st%d sbufp reg%d %s addr=%#llx size=%llu %s", g_stat_pass, nvn_stage, reg, u && u->addr ? "Real" : "Dummy", u ? (unsigned long long)u->addr : 0ull, u ? (unsigned long long)u->size : 0ull, b->rw ? "rw" : "ro");
        } else if (b->kind == NVN_RB_C1) {      // B: the shader's own constant segment, built when the shader is fetched
            id<MTLBuffer> cb = sh->c1buf ? sh->c1buf : zero_bank();   // c1 has length 0 (600 of them in the whole library) yet the shader still declares it: reads 0
            slots[aid] = (sh->c1buf ? sh->c1_gpu : zero_bank_gpu()) + (sh->c1buf ? sh->c1off : 0);
            USE(cb, MTLResourceUsageRead);
            // OT2_NVN_5FFD_DIAG (off by default): touches only 5ffd fragments. Bit 1 records numeric summaries of translated rf.c1 and real-time Data suffix c1 (holds g_rmu);
            // Bit 4 rebinds to real-time Data suffix c1 when the translated variant lacks c1 (rf.c1_len=0) (game's own bytes, not synthetic; on failure falls back to zero_bank and records one red).
        } else if (b->kind == NVN_RB_SUPPORT) {   // support_buffer: scale 1.0, BGRA swap 0, viewport transform not disabled (viewport fields not read)
            slots[aid] = support_gpu();
            USE(support_buf(), MTLResourceUsageRead);
        } else if (b->kind == NVN_RB_DRV0 || b->kind == NVN_RB_DRV2) {
            // NVN driver constant section: statically proven unread values are set to 0 at package time; read values are filled in using the current binding — c0 = storage buffer table (base address + 0x10 r: 64-bit address, 32-bit size),

            // NVN driver constant segments: those proven unread by static analysis at packaging time are supplied as 0; those read are filled on the spot from the current bindings - c0 = storage buffer table (base + 0x10 r: address 64 bits, size 32 bits),
            // c2 = texture handle table (0x168 + 8 r) and sampler handle table (0x568 + 8 r), the values being the NVN handles the game gave at bind time; positions read outside the tables are registered at packaging time and reported a failure at load time
            int bank = b->kind == NVN_RB_DRV2 ? 2 : 0;
            if (!sh->rf.drv_read[bank]) { slots[aid] = zero_bank_gpu(); USE(zero_bank(), MTLResourceUsageRead); }
            else {
                id<MTLBuffer> db; size_t doff = arena_alloc(0xA00, &db); uint8_t *d = (uint8_t *)db.contents + doff; memset(d, 0, 0xA00);
                if (bank == 0) {
                    int base = nvn_stage == 0 ? 0x110 : nvn_stage == 5 ? 0x310 : nvn_stage == 1 ? 0x510 : -1;
                    if (base < 0) RED_ONCE(45, "Shader in stage %d reads driver segment c0; storage buffer table base address has not been verified", nvn_stage);
                    else for (int r = 0; r < 16; r++) { uint64_t a = S.ssbo[nvn_stage][r].addr; uint32_t n = (uint32_t)S.ssbo[nvn_stage][r].size; memcpy(d + base + 16 * r, &a, 8); memcpy(d + base + 16 * r + 8, &n, 4); }
                } else {
                    for (int r = 0; r < 128; r++) { uint32_t th = (uint32_t)S.tex[nvn_stage][r], sh_ = (uint32_t)S.samp[nvn_stage][r]; memcpy(d + 0x168 + 8 * r, &th, 4); memcpy(d + 0x568 + 8 * r, &sh_, 4); }
                }
                slots[aid] = g_cur_arena->gpu + doff;
                USE(db, MTLResourceUsageRead);
            }
        } else RED_ONCE(46, "Binding type %d is not implemented", b->kind);
    }
    #undef USE
    #undef USE_SAMP
    pthread_mutex_unlock(&g_rmu);
    NSUInteger arg_len = sh->arg_len ? sh->arg_len : sh->ae.encodedLength;
    id<MTLBuffer> ab; size_t off = arena_alloc(arg_len, &ab);
    memcpy((uint8_t *)ab.contents + off, slots, arg_len);
    bind_args(en, nvn_stage, metal_vertex, ab, g_cur_arena->gpu + off, off, sh->rf.abi);
    sh->ab_buf = ab; sh->ab_gpu = g_cur_arena->gpu + off; sh->ab_off = off; sh->ab_pass = g_pass.serial; g_args_dirty[nvn_stage] = 0;
    g_stat_args_enc++;
    return 1;
}
static id<MTLBuffer> g_quad_ib; static uint64_t g_quad_gpu;
static RENC rp_begin(void);
// ---------- Channel lookahead: whether this channel uses depth / stencil ----------
// When opening a channel, scan forward from the next command after the current draw, simulating depth/stencil state changes until a point where the channel must end (target reset, copy, compute dispatch—
// corresponding exactly to the operation that ends the channel during replay). If no draw enables depth testing / depth writes / stencil testing during this interval, the channel can omit the depth attachment:
// Depth contents remain completely untouched, saving the full depth read and write on TBDR (measured on real hardware: a single-draw fullscreen channel uses half the GPU time, mostly due to reading and writing D32S8).

// ---------- Pass lookahead: does this pass use depth / stencil ----------
// When opening a pass, scan forward from the op after the current draw, simulating the changes of depth-stencil state, up to where this pass must end (target reset, copy, compute dispatch -
// matching one-to-one the ops that end a pass in the replay). If no draw in between enables depth test / depth write / stencil test, this pass may leave the depth attachment unattached:
// not one byte of the depth content is touched, and on TBDR the load and store of the whole depth texture are saved (measured on a real device: a single-draw fullscreen pass takes half the GPU, with an extra D32S8 load and store attached).
// The scan follows real replay order across sub-list returns and across the remaining handles of this submission; end of submission = the pass must end. Undecidable cases such as a mid-pass clear or too-deep nesting always count as 'used'.
static const uint8_t *g_exec_next;   // The op following the current draw in the replay
static const uint8_t *g_ret[16]; static int g_ret_n;   // Call return stack of the replay (at OP_CALL the caller's next op is pushed)
static const uint64_t *g_rest; static int g_nrest;      // Handles in this submission still to be replayed after the current handle
void mtl_exec_rest(const uint64_t *h, int n) { if (g_sub_sync_failed) { g_rest = NULL; g_nrest = 0; return; } g_rest = h; g_nrest = n; }
static uint64_t g_stat_ds_omit;      // Number of passes that attach less depth-stencil because the lookahead proved it unused (including stencil-only / depth-only)
static uint64_t g_stat_s_only, g_stat_d_only;   // Of those, the number of passes that attach only stencil / only depth
// Pure-clear channel precondition (hoist_try): operations already executed early and to be skipped when replay reaches their original positions (each group's SetRenderTargets and its clears).

// Clear-only pass hoist (hoist_try): ops that were already executed early and must be skipped when replay reaches their original position (each group's SetRenderTargets and the clears in it).
// The exec loop, the store lookahead (sdx index) and the depth-stencil lookahead (ds_scan) all skip by it - otherwise the lookahead would take the clear at the original position as 'cleared again' and wrongly judge the early-cleared content redundant
#define HOIST_MAX 40   // At most 4 groups × (1 set-target op + 8 clears)
static const uint8_t *g_hoist_skip[HOIST_MAX]; static int g_hoist_n; static uint64_t g_hoist_groups;
static int hoist_skipped(const uint8_t *p) { for (int i = 0; i < g_hoist_n; i++) if (g_hoist_skip[i] == p) return 1; return 0; }
typedef struct { const uint8_t *p, *ret[8]; int ret_n, rest_i, bad; } fz_cursor;
static int fz_next(fz_cursor *c, const uint8_t **out, op_hdr *oh, const void **oa) {
    for (int guard = 0; guard < 1000000; guard++) {
        if (!c->p) return 0;
        op_hdr h; memcpy(&h, c->p, sizeof h); const void *a = c->p + 8;
        switch (h.op) {
            case OP_END:
                if (c->ret_n) c->p = c->ret[--c->ret_n];
                else if (c->rest_i < g_nrest) c->p = (const uint8_t *)g_rest[c->rest_i++];
                else { c->p = NULL; return 0; }
                continue;
            case OP_JUMP: memcpy(&c->p, a, 8); continue;
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (c->ret_n >= 8) { c->bad = 1; return 0; }
                c->ret[c->ret_n++] = c->p + h.size; c->p = (const uint8_t *)t; continue; }
            case OP_WAIT_SYNC: c->bad = 1; return 0;
            default: break;
        }
        const uint8_t *p = c->p; c->p += h.size;
        if (g_hoist_n && hoist_skipped(p)) continue;
        if (out) *out = p; if (oh) *oh = h; if (oa) *oa = a; return 1;
    }
    c->bad = 1; return 0;
}

// State of the readable-tile pass merge (merge B step 2): the earlier pass (phase 1) and the later pass (phase 2) share one render pass, and the attachments are the union
typedef struct {
    int active, phase, n1, nskip;
    int nreuse;
    const uint8_t *rt2_op, *reuse[8], *skip[12];
    p_set_rt rt1, rt2;                 // Logical targets of each of the two passes (R.rt is the union during the merge)
    int8_t p2map[8];                   // Logical color number of the later pass → physical attachment
    uint8_t lmap1[8], lmap2[8];        // Physical attachment → logical color number of this phase's draws (0xff = not written)
    uint8_t clr2[8], cd2f, cs2f; MTLClearColor cc2[8]; double cd2; uint32_t cs2;   // Clears at the start of the later pass (used as-is when the in-run dual render redraws)
    uint8_t wrote[8];                   // Physical attachments written by the later pass
    uint8_t cov2[8];                     // Physical attachments covered by the first draw of the later pass
} fuse_t;
static fuse_t g_fz; static uint64_t g_fz_n, g_fz_draws, g_fz_fallback;
typedef struct { int idx; p_draw dw; p_draw_idx di; p_scissor sc; int has_sc; void *S_; uint8_t pm[8]; } defer_snap;   // pm: group P target index → logical color number of this draw (identity for its own draws; a composed mapping for absorbed upstream groups)
typedef struct {
    int active, at_q, in_p, n, reason, reason_op, logged, reset_between; uint64_t born;   // born: creation serial (replay is ordered by it, not by slot)
    p_set_rt p_rt, q_rt;
    const uint8_t *p_end_op, *q_op;
    uint8_t clear_c[8], lmap[8];
    MTLClearColor cc[8];
    uint8_t clear_d, clear_s;
    double cd; uint32_t cs;
    uint64_t p_reads[512]; int nreads, p_dw, p_st;
    defer_snap snap[64];
} defer_t;
static defer_t g_defer_slots[4];
static defer_t *g_defer_cur = &g_defer_slots[0];
#define g_defer (*g_defer_cur)
static int sub_eq(const p_set_rt *a, int ka, const p_set_rt *b, int kb);
static uint64_t g_defer_n, g_defer_draws, g_defer_fallbacks;
static const uint8_t *g_defer_pend;   // The P-range end op of the group that is in its P range (NULL if none): the exec loop compares each op against just this one pointer
static int g_defer_nact, g_defer_natq;   // Number of active groups, number of groups that reached Q and await replay (the hot path only looks at the counts, no per-slot scan)
static int g_defer_ab = -1, g_defer_ab_on;
static void draw(const p_draw *dw, const p_draw_idx *di);
static int defer_has_in_p(void) { return g_defer_pend != NULL; }
static int defer_has_active(void) { return g_defer_nact > 0; }
static int defer_slot_free(void) { for (int i = 0; i < 4; i++) if (!g_defer_slots[i].active) return i; return -1; }
static int defer_same_q_depth(const defer_t *a, const p_set_rt *q) {
    if (!a->q_rt.depth || !q->depth) return 1;
    return sub_eq(&a->q_rt, 8, q, 8);
}
static int defer_next(int at_q_only, uint64_t after) {   // Slot number of the next active group (smallest creation serial that is > after); returns −1 if none
    int best = -1; for (int i = 0; i < 4; i++) { const defer_t *d = &g_defer_slots[i]; if (!d->active || (at_q_only && !d->at_q) || d->born <= after) continue; if (best < 0 || d->born < g_defer_slots[best].born) best = i; }
    return best;
}
static void defer_release_one(defer_t *d) { if (d->active) { g_defer_nact--; if (d->at_q) g_defer_natq--; if (d->in_p) g_defer_pend = NULL; } for (int i = 0; i < d->n; i++) free(d->snap[i].S_); memset(d, 0, sizeof *d); }
static void defer_clear_all(void) { for (int i = 0; i < 4; i++) defer_release_one(&g_defer_slots[i]); g_defer_nact = g_defer_natq = 0; g_defer_pend = NULL; }
static int hoist_take(const uint8_t *p) { for (int i = 0; i < g_hoist_n; i++) if (g_hoist_skip[i] == p) { g_hoist_skip[i] = g_hoist_skip[--g_hoist_n]; return 1; } return 0; }
enum { DS_FREE = 0, DS_USED = 1, DS_RET = 2 };
static int g_ds_why; static int g_ds_scanned; static int g_ds_need;   // g_ds_need: need accumulated during the scan, 1 depth 2 stencil   // Diagnostics: the opcode that triggered the 'used' verdict (for a draw, records 1000 + dt*100 + dw*10 + st), and the number of ops scanned
static int ds_scan(const uint8_t *p, ds_st *ds, int depth) {
    for (int guard = 0; guard < 1000000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const void *a = p + 8; g_ds_scanned++;
        if (g_hoist_n && hoist_skipped(p)) { p += h.size; continue; }   // Clear pass already executed early
        switch (h.op) {
            case OP_END: return DS_RET;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth >= 8) return DS_USED;
                int r = ds_scan((const uint8_t *)t, ds, depth + 1); if (r != DS_RET) return r; break; }
            case OP_WAIT_SYNC: g_ds_why = h.op; return DS_USED;
            case OP_SET_RT: case OP_COPY_B2B: case OP_CLEAR_BUF: case OP_UPDATE_UB: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: case OP_DISPATCH: return DS_FREE;
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: g_ds_why = h.op; return DS_USED;
            case OP_BIND_STATE: { const p_bind_state *b = a; if (b->kind == 5) memcpy(ds, b->bytes, sizeof *ds); break; }
            case OP_DRAW: case OP_DRAW_IDX:   // Depth test / write → depth needed; stencil test → stencil needed; can stop once both are needed
                g_ds_need |= (ds->dt || ds->dw ? 1 : 0) | (ds->st ? 2 : 0);
                if (g_ds_need == 3) { g_ds_why = 1000 + ds->dt * 100 + ds->dw * 10 + ds->st; return DS_USED; } break;
            default: break;
        }
        p += h.size;
    }
    return DS_USED;
}
// Writeback lookahead criteria (command by command, in actual replay order):
// Unnecessary: after being set as an attachment again, it is fully cleared before any draw (color: all four channels at that index are written; depth: depth clear; stencil: all 8 stencil bits cleared. Depth and stencil are checked and writeback-set separately)
// Needed: during a draw / dispatch, it is bound to a texture or image slot at any stage (including if already bound when scanning starts); used as a copy source / copy target / texture-clear object; or, after being set as an attachment again, drawn to before being cleared (read in)

// Store lookahead decision (command by command, in real replay order):
//   Redundant: after being set as an attachment again and before any draw, it is cleared in full (color: all four channels of that index written; depth: depth clear; stencil: all 8 stencil bits cleared. Depth and stencil are decided separately and their write-back is set separately)
//   Needed: at a draw / dispatch it is bound to a texture or image slot of any stage (including ones already bound when the scan starts); used as copy source / copy destination / texture-clear target; drawn before cleared after becoming an attachment again (load)
// Different levels / layers of the same texture are decided separately (a downsample chain writes level k and reads level k−1); a sampler binding counts as a read of the whole texture (conservative)
static const uint8_t *g_exec_cur;   // The op currently being replayed (inclusive)
static void sd_need(sd_att *t, int why) { if (!t->state) { t->state = 1; t->why = (uint8_t)why; } }   // why: 1 a draw reads it as a texture 2 as an attachment, drawn before cleared 3 copy 4 texture clear 5 scanned to the end undecided 6 no start point / index unavailable
static int sdx_decide(uint8_t st[9]);   // Store lookahead decision (for the index see mtl_sub_index)
static void sd_finish(void) {
    if (!g_nsd) return;
    if (!tbdr_on()) {   // All optimizations off: always write back
        for (int i = 0; i < g_nsd; i++) { sd_att *t = &g_sd[i];
            if (t->depth == 1) [g_enc setDepthStoreAction:MTLStoreActionStore]; else if (t->depth == 2) [g_enc setStencilStoreAction:MTLStoreActionStore]; else [g_enc setColorStoreAction:MTLStoreActionStore atIndex:(NSUInteger)t->idx]; }
        g_nsd = 0; return; }
    uint8_t nst[9];
    if (sdx_decide(nst)) { for (int i = 0; i < g_nsd; i++) if (nst[i] == 2) g_sd[i].state = 2; else sd_need(&g_sd[i], nst[i] >> 2); }
    else for (int i = 0; i < g_nsd; i++) sd_need(&g_sd[i], 6);   // Pass ended outside the list / index unavailable: always write back
    for (int i = 0; i < g_nsd; i++) {
        sd_att *t = &g_sd[i]; int udc_dead = udc_on() && t->load && !t->wrote;
        // Color attachments written in the same channel as the window texture are not tracked: they are part of the output chain, not transient attachments that are "reset then used" every frame.
        // The in-game "upscale" channel writes two textures at once—the scene texture and the window texture—and UI composition is in the same channel, so nothing in-game reads the scene texture back; when the game stops drawing 3D (reads the screen)
        // it switches to sampling it from a separate composition channel. The need to "read the screen" is only known after the previous frame is submitted, so lookahead cannot prevent it: on a real full-screen iPad, the first display-mode switch after each install always hits this once

        // A color attachment written in the same pass as the window texture does not enter the ledger: it is a link in the present chain, not a transient attachment that is 'reset every frame before use'.
        // The in-run 'upscale' pass writes two textures at once - scene texture + window texture - and the UI composite was merged into the same pass, so in-run nobody reads back the scene texture; as soon as the game stops drawing 3D (loading screen)
        // a separate composite pass samples it afterwards instead. 'About to enter the loading screen' is only known after the previous frame was submitted, so prediction cannot stop it: on iPad true fullscreen every install hits it once on the first act change
        // (the 2732×2048 attachment of frame N was not written back and is sampled for 4 consecutive frames starting at frame N+1; the key is permanently disabled). At the iPhone aspect ratio the same-size key already sees a reader during the trial period and is silently disabled.
        int xv_win = 0; for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k] && xv_present(R.rt.color[k]->serial)) xv_win = 1;
        uint64_t xtag = 0; int xvcand = xv_on() && !xv_win && t->depth == 0 && t->state != 2 && t->why == 5 && xv_first_good(t->ser, t->lv, t->sl) && !xv_present(t->ser)
                                        && !xv_disabled(xtag = xv_tag_pass(R.rt.color[t->idx], t->lv, t->idx, &R.rt));
        int xvdead = xvcand && xv_trial_get(xtag, 0) >= XV_TRIAL;   // Only after the trial period is the write-back really skipped
        if (xvcand && !xvdead) xv_set(t->ser, t->lv, t->sl, XV_PROBE, xtag);   // Trial period: write back as usual, only track the next touch
        int dead = t->state == 2 || udc_dead || xvdead; MTLStoreAction sa = dead ? MTLStoreActionDontCare : MTLStoreActionStore;
        if (xvdead) { xv_set(t->ser, t->lv, t->sl, XV_INVALID, xtag); g_xv_dc++; }
        if (t->depth == 1) [g_enc setDepthStoreAction:sa]; else if (t->depth == 2) [g_enc setStencilStoreAction:sa]; else [g_enc setColorStoreAction:sa atIndex:(NSUInteger)t->idx];
        g_sd_n_all++; if (dead) { g_sd_n_dead++; g_sd_dead_bytes += t->bytes; g_bw_store -= t->bytes; if (udc_dead) g_udc_saved_bytes += t->bytes; }
        if (!t->state) t->why = 5;
        PD("Z %llu %s ser=%llu lv%u sl%u %s why=%d by=%s slot=%d", g_stat_pass, t->depth == 1 ? "d" : t->depth == 2 ? "s" : "c", (unsigned long long)t->ser, t->lv, t->sl, xvdead ? "dontcare-xv" : dead ? "dontcare" : "store", dead ? 0 : t->why, t->why == 1 && t->wkey ? t->wkey : "-", t->wslot);
        if (udc_on()) PD("U %llu %s ser=%llu lv%u sl%u runtime=%s reason=%s state=%d load=%d", g_stat_pass, t->depth == 1 ? "d" : t->depth == 2 ? "s" : "c", (unsigned long long)t->ser, t->lv, t->sl, t->wrote ? "Written" : "Not written", t->wwhy == 1 ? "Image write" : t->wwhy == 2 ? "Color write mask" : t->wwhy == 3 ? "Fragment output" : t->wwhy == 4 ? "Depth write" : t->wwhy == 5 ? "Stencil write" : "None", t->state, t->load);
    }
    g_nsd = 0;
}
// Continue scanning in actual replay order: current list → callers at each level (return stack, top to bottom) → remaining handles in this submission; the channel must end at submission end (rp_flush before commit)
// Coverage proof (all conditions must hold; if any cannot be recognized, treat as not proven):
// Shader: vertex position xyz is copied directly from a vertex attribute, and w is 1 or the same attribute's .w (dmsh_build raster_facts); fragment shader does not discard, does not write sample_mask, and writes that color index
// State: attachment has no blending, full write mask, no alpha-to-coverage, fill mode, no depth testing (or ALWAYS), no stencil testing, single-sampled, non-3D
// Vertices: non-indexed draw or 16 / 32-bit indexed draw; both indices and positions are read from CPU-visible pool memory; clipping / viewport / geometry / facing / depth-clipping checks are in nvn_cover.h (pure functions, unit-tested)
// Read attribute a for all vertices in this draw (non-indexed or 16 / 32-bit indexed; vertex number = first + k or index value + base_vertex); missing components are filled with 0 / 1 according to fetch rules.

// Continue scanning in real replay order: current list -> callers at each level (return stack from the top down) -> remaining handles of this submit; the render pass necessarily ends when the submit ends (rp_flush before commit)
// Coverage proof (holds only if all are satisfied; anything unrecognized is treated as not holding):
//   Shader: the vertex position xyz is taken as-is from one vertex attribute, w is 1 or the same attribute's .w (dmsh_build raster_facts); the fragment does not discard, does not write sample_mask, and writes this color index
//   State: this attachment has no blending, a full write mask, no alpha-to-coverage, fill mode, no depth test (or ALWAYS), no stencil test, single sample, not 3D
//   Vertices: non-indexed or 16 / 32-bit indexed draws, with indices and positions both read from CPU-visible pool memory; the scissor / viewport / geometry / winding / depth clipping decisions are in nvn_cover.h (pure functions, with unit tests)
// Read out attribute a of all vertices of this draw (non-indexed or 16 / 32-bit indexed; vertex number = first + k or index value + base_vertex), with missing components filled with 0 / 1 per the fetch rules.
// Returns 0 on success, a reason code on failure (6 vertex count / index type, 7 attribute index, 8 format / stream, 9 out of bounds, 10 address)
static int attr_read(int a, const p_draw *dw, const p_draw_idx *di, float P[64][4]) {
    int n = dw ? dw->count : di->count;
    if (n < 1 || n > 64 || (dw && dw->first < 0) || (di && (di->itype < 1 || di->itype > 2))) return 6;
    if (a < 0 || a >= S.nattr || S.attr[a].stream >= 16) return 7;
    va_st at = S.attr[a]; vs_st sm = S.stream[at.stream]; NVNbufferRange vb = S.vbuf[at.stream];
    MTLVertexFormat vf = vfmt(at.fmt); int nc = vf == MTLVertexFormatFloat2 ? 2 : vf == MTLVertexFormatFloat3 ? 3 : vf == MTLVertexFormatFloat4 ? 4 : 0;
    if (!nc || !sm.stride || sm.divisor || !vb.address) return 8;
    const uint8_t *ib = NULL; int isz = di ? (di->itype == 1 ? 2 : 4) : 0;
    if (di) { void *m; uint64_t o; if (!mtl_addr(di->ibuf, (uint64_t)n * isz, &m, &o) || !(ib = [(__bridge id<MTLBuffer>)m contents])) return 10; ib += o; }
    for (int k = 0; k < n; k++) {
        int64_t vid = dw ? (int64_t)dw->first + k : (isz == 2 ? (int64_t)((const uint16_t *)ib)[k] : (int64_t)((const uint32_t *)ib)[k]) + di->base_vertex;
        uint64_t off = (uint64_t)vid * sm.stride + at.offset;
        if (vid < 0 || off + 4 * nc > vb.size) return 9;
        void *mb; uint64_t mo; const uint8_t *base;
        if (!mtl_addr(vb.address + off, 4 * nc, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) return 10;
        float v[4] = {0, 0, 0, 1}; memcpy(v, base + mo, 4 * nc); memcpy(P[k], v, sizeof v);
    }
    return 0;
}
static int first_draw_covers(int i, id<MTLTexture> t, NSUInteger level) {
    const p_draw *dw = g_fd_dw; const p_draw_idx *di = g_fd_di;
    int n = dw ? dw->count : di->count, prim = dw ? dw->prim : di->prim;
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr) return g_fd_why = 1, 0;
    NvShader *vs = pr->sh[0], *fs = pr->sh[1];
    if (!vs || !fs || vs->rf.pos_attr < 0 || fs->rf.may_discard || fs->rf.writes_smask || !(fs->rf.color_mask >> i & 1)) return g_fd_why = 2, 0;
    if (!t || t.sampleCount != 1 || t.textureType == MTLTextureType3D) return g_fd_why = 3, 0;
    if ((S.color.blend_en >> i & 1) || ((S.chmask.mask >> (4 * i)) & 0xf) != 0xf || S.ms.a2c || S.poly.mode != NVN_POLYGON_MODE_FILL) return g_fd_why = 4, 0;
    if (R.rt.depth && ((S.ds.dt && S.ds.df != NVN_DEPTH_FUNC_ALWAYS) || S.ds.st)) return g_fd_why = 5, 0;
    if (n < 3 || n > 64) return g_fd_why = 6, 0;
    float P[64][4]; int why = attr_read(vs->rf.pos_attr, dw, di, P);
    if (why) return g_fd_why = why, 0;
    if (vs->rf.pos_w1) for (int k = 0; k < n; k++) P[k][3] = 1.0f;   // Position w is the constant 1 (the attribute's .w is not taken)
    int sc[4] = {R.sc.x, R.sc.y, R.sc.w, R.sc.h}; float vp[4] = {S.vp[0][0], S.vp[0][1], S.vp[0][2], S.vp[0][3]};
    g_fd_why = 11;
    return nvn_cover((const float (*)[4])P, n, prim, (int)S.poly.cull, S.poly.front == NVN_FRONT_FACE_CCW,
                     S.has_vp, vp, R.has_sc, sc, (long)MAX(t.width >> level, 1), (long)MAX(t.height >> level, 1));
}

// The handles of BindTexturesDeferred are read from the record's src at execution time (h at recording time is only a fallback)
static inline void bh_fresh(const p_bind_handles *b) {
    if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);
}
static int state_op(uint32_t op, const void *a) {
    switch (op) {
        case OP_SCISSOR: R.sc = *(const p_scissor *)a; R.has_sc = 1; break;
        case OP_BIND_PROGRAM: {
            const p_bind_program *binding = a;
            const prog_t *np = binding->prog;
            static _Atomic uint64_t seq;
            uint64_t n = nvn_bind_trace_on()
                ? atomic_fetch_add_explicit(&seq, 1, memory_order_relaxed) : UINT64_MAX;
            if (n < 16 || (n != UINT64_MAX && n % 10000 == 0))
                MLOG("[DEBUG-RA-BIND] replay seq=%llu program=%p magic=%#x serial=%llu nstage=%u stages=%#x previous=%p",
                     (unsigned long long)n, (const void *)np,
                     (unsigned)(np ? np->magic : 0),
                     (unsigned long long)(np ? np->serial : 0),
                     np ? np->nstage : 0, (unsigned)binding->stages, (const void *)S.prog);
            program_binding_apply(S.program_stage, binding);
            const prog_t *composite = program_binding_compose(S.program_stage);
            if (composite != S.prog) { S.prog = composite; g_pso_dirty = 1; args_dirty_all(); }
            break;
        }
        case OP_BIND_STATE: {
            const p_bind_state *b = a; void *dst = NULL; size_t n = 0;
            blend_st bs;
            switch (b->kind) {
                case 0: memcpy(&bs, b->bytes, sizeof bs); dst = &S.blend[bs.target]; n = sizeof bs; break;
                case 1: dst = &S.color; n = sizeof S.color; break;
                case 2: dst = &S.chmask; n = sizeof S.chmask; break;
                case 3: dst = &S.ms; n = sizeof S.ms; break;
                case 4: dst = &S.poly; n = sizeof S.poly; break;
                case 5: dst = &S.ds; n = sizeof S.ds; break;
            }
            if (dst && memcmp(dst, b->bytes, n)) { memcpy(dst, b->bytes, n); if (b->kind == 5) g_dss_dirty = 1; else g_pso_dirty = 1; }
            break;
        }
        case OP_BIND_VATTR: { const p_bind_vattr *b = a; if (S.nattr != b->n || memcmp(S.attr, b->st, 4 * b->n)) { memcpy(S.attr, b->st, 4 * b->n); S.nattr = b->n; g_pso_dirty = 1; } break; }
        case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; if (S.nstream != b->n || memcmp(S.stream, b->st, 8 * b->n)) { memcpy(S.stream, b->st, 8 * b->n); S.nstream = b->n; g_pso_dirty = 1; } break; }
        case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf tmp; const p_bind_vbuf *b = vbuf_rec(op, a, &tmp); memcpy(&S.vbuf[b->first], b->r, sizeof *b->r * b->n); break; }
        case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf tmp; const p_bind_buf *b = bind_buf_rec(op, a, &tmp); if (b->stage < 6 && b->index < 16 && memcmp(&S.ubo[b->stage][b->index], b, sizeof *b)) { S.ubo[b->stage][b->index] = *b; g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage < 6 && b->index < 16 && memcmp(&S.ssbo[b->stage][b->index], b, sizeof *b)) { S.ssbo[b->stage][b->index] = *b; g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_TEX: case OP_BIND_TEX1: { p_bind_handles t1; const p_bind_handles *b = bind_rec(op, a, &t1); bh_fresh(b); if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 2 * NVN_SEPTEX_SLOT0 && memcmp(&S.tex[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.tex[b->stage][b->first], b->h, 8 * b->n); g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_SAMP: { const p_bind_handles *b = a; bh_fresh(b); if (b->stage < 6 && b->first + b->n <= 128 && memcmp(&S.samp[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.samp[b->stage][b->first], b->h, 8 * b->n); g_args_dirty[b->stage] = 1; } break; }
        case OP_BIND_IMG: { const p_bind_handles *b = a; if (b->stage < 6 && b->first >= 0 && b->first + b->n <= 32) { if (memcmp(&S.img[b->stage][b->first], b->h, 8 * b->n)) { memcpy(&S.img[b->stage][b->first], b->h, 8 * b->n); g_args_dirty[b->stage] = 1; } } else RED_ONCE(31, "Image slot out of bounds: stage %d start %d count %d", b->stage, b->first, b->n); break; }
        case OP_STENCIL: {
            const p_stencil *b = a;
            if (b->which != 2) g_dss_dirty = 1;
            for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S.st_vmask[f] = b->value_mask; else if (b->which == 1) S.st_wmask[f] = b->mask; else S.st_ref[f] = b->ref; }
            break;
        }
        case OP_POLY_OFFSET: S.po = *(const p_poly_offset *)a; break;
        case OP_VIEWPORTS: { const p_viewports *b = a; for (int i = 0; i < b->n; i++) memcpy(S.vp[b->first + i], b->v[i], 16); S.has_vp = 1; break; }
        case OP_DEPTH_RANGE: { const p_depth_range *b = a; S.dr[0] = b->n; S.dr[1] = b->f; break; }
        case OP_BLEND_COLOR: memcpy(S.blend_color, ((const p_blend_color *)a)->rgba, 16); break;
        case OP_COPY_STRIDE: { const p_copy_stride *cs = a; if (cs->image) S.copy_image_stride = cs->stride; else S.copy_stride = cs->stride; break; }
        default: return 0;
    }
    return 1;
}

static void rt_sub(const p_set_rt *r, int k, uint64_t *ser, int *lv, int *sl);

// Parallel precompilation prediction (see note before pso_get): scan from the operation after the current one to the end of this submission, replaying state only; for each draw, compute the pipeline key using "the state at that point + that channel's attachment formats" and queue unseen ones in the background.

// Prediction for parallel precompilation (see the note before pso_get): scan from after the current operation to the end of this submit, replaying only state; at a draw, compute the pipeline key from "the state at that time + that render pass's attachment formats" and dispatch those not yet built to the background.
// Whether a render pass actually attaches depth / stencil is only decided at replay (depth-only / stencil-only / neither attached are omitted), so one key is computed for each possible attach mode -- guessing too many only means wasted background compiles and does not affect the result.
static void pso_predict_rest(void) {
    if (!g_exec_next) return;
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S);
    p_scissor sc = R.sc; int has_sc = R.has_sc;
    int pso_dirty = g_pso_dirty, dss_dirty = g_dss_dirty;
    uint32_t cf[8]; memcpy(cf, g_pass.cfmt, sizeof cf); uint32_t df = g_pass.dfmt, sf = g_pass.sfmt, samples = g_pass.samples, dfull = df, sfull = sf; int kicked0 = (int)g_pp_kicked, draws = 0;
    fz_cursor cur = {.p = g_exec_next};

    // Collect them all first, then dispatch: keys for "attachments attached as-is" go first (most likely to be used), the other attach modes after -- background concurrency is limited, and what is dispatched first is compiled first
    enum { QN = 160 }; static struct { pso_key k; void *vs, *fs; } qa[QN], qb[QN]; int na = 0, nb = 0;
    for (int guard = 0; guard < 200000 && na < QN; guard++) {
        const uint8_t *q; op_hdr hq; const void *aq;
        if (!fz_next(&cur, &q, &hq, &aq)) break;
        if (hq.op == OP_SET_RT) {
            const p_set_rt *r = aq; memset(cf, 0, sizeof cf); dfull = sfull = 0; samples = 1;
            for (int i = 0; i < r->n && i < 8; i++) { id<MTLTexture> t = r->color[i] ? tex_mtl(r->color[i]) : nil; if (t) { cf[i] = (uint32_t)rt_color_fmt(t, &r->cview[i], r->has_cview[i], 0); samples = (uint32_t)t.sampleCount; } }
            id<MTLTexture> dt = r->depth ? tex_mtl(r->depth) : nil;
            if (dt) { MTLPixelFormat f = dt.pixelFormat; samples = (uint32_t)dt.sampleCount;
                dfull = f != MTLPixelFormatStencil8 ? (uint32_t)f : 0; sfull = (f == MTLPixelFormatStencil8 || f == MTLPixelFormatDepth32Float_Stencil8) ? (uint32_t)f : 0; }
            df = dfull; sf = sfull; continue;
        }
        if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX) {
            draws++;
            __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; if (!pr || !pr->sh[0]) continue;
            __unsafe_unretained NvShader *vs = pr->sh[0], *fs = pr->sh[1];
            const uint32_t dv[4] = {dfull, 0, 0, dfull}, sv[4] = {sfull, 0, sfull, 0};
            for (int v = 0; v < 4; v++) {
                if (v && dv[v] == dv[0] && sv[v] == sv[0]) continue;
                if (v == 2 && !sfull) continue; if (v == 3 && (!dfull || !sfull)) continue;
                pso_key k; pso_key_fill(&k, vs, fs, cf, dv[v], sv[v], samples, 0);
                if (hm_get_id(&g_pso, &k)) continue;
                int dup = 0; for (int i = 0; i < (v ? nb : na) && !dup; i++) dup = !memcmp(&(v ? qb : qa)[i].k, &k, sizeof k);
                if (dup) continue;
                if (!v && na < QN) { qa[na].k = k; qa[na].vs = (__bridge void *)vs; qa[na].fs = (__bridge void *)fs; na++; }
                else if (v && nb < QN) { qb[nb].k = k; qb[nb].vs = (__bridge void *)vs; qb[nb].fs = (__bridge void *)fs; nb++; }
            }
            continue;
        }
        if (hq.op == OP_CLEAR_COLOR || hq.op == OP_CLEAR_DS) continue;
        state_op(hq.op, aq);
    }
    (void)df; (void)sf;
    for (int i = 0; i < na; i++) pso_pend_kick(&qa[i].k, (__bridge NvShader *)qa[i].vs, (__bridge NvShader *)qa[i].fs);
    for (int i = 0; i < nb; i++) pso_pend_kick(&qb[i].k, (__bridge NvShader *)qb[i].vs, (__bridge NvShader *)qb[i].fs);
    memcpy(&S, &Ssave, sizeof S); R.sc = sc; R.has_sc = has_sc; args_dirty_all(); g_pso_dirty = pso_dirty; g_dss_dirty = dss_dirty;
    static int logged; if (g_pp_kicked > (uint64_t)kicked0 && logged++ < 20) MLOG("Parallel precompilation: %d draws remaining in this submission, sending %d pipelines to the background", draws, (int)g_pp_kicked - kicked0);
}

static void fz_probe_cov2(const p_set_rt *r2, const uint8_t *first, const int8_t *p2map, const uint8_t *clr, int n1, int n) {
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S); typeof(R) Rsave = R;
    const p_draw *fdw = g_fd_dw; const p_draw_idx *fdi = g_fd_di; int fdwhy = g_fd_why;
    int pso_dirty = g_pso_dirty, dss_dirty = g_dss_dirty;
    op_hdr fh; memcpy(&fh, first, sizeof fh);
    g_fd_dw = fh.op == OP_DRAW ? (const p_draw *)(first + 8) : NULL;
    g_fd_di = fh.op == OP_DRAW_IDX ? (const p_draw_idx *)(first + 8) : NULL;
    R.rt = *r2; R.has_rt = 1;
    fz_cursor cur = {.p = g_exec_next};
    for (;;) {
        const uint8_t *q; op_hdr hq; const void *aq;
        if (!fz_next(&cur, &q, &hq, &aq)) break;
        if (q == first) break;
        if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX || hq.op == OP_SET_RT || hq.op == OP_CLEAR_COLOR || hq.op == OP_CLEAR_DS) continue;
        state_op(hq.op, aq);
    }
    for (int j = n1; j < n && j < 8; j++) {
        int k = -1;
        for (int q = 0; q < 8; q++) if (p2map[q] == j) { k = q; break; }
        if (k < 0 || clr[k] || !r2->color[k]) continue;
        uint64_t ser; int level, slice;
        rt_sub(r2, k, &ser, &level, &slice);
        (void)ser; (void)slice;
        g_fz.cov2[j] = (uint8_t)first_draw_covers(k, tex_mtl(r2->color[k]), (NSUInteger)level);
    }
    memcpy(&S, &Ssave, sizeof S); args_dirty_all(); R = Rsave;
    g_fd_dw = fdw; g_fd_di = fdi; g_fd_why = fdwhy;
    g_pso_dirty = pso_dirty; g_dss_dirty = dss_dirty;
}

static int fz_load_covers(int i, id<MTLTexture> t, NSUInteger level) {
    if (g_fz.active && g_fz.phase == 1 && i >= g_fz.n1) return g_fz.cov2[i];
    return first_draw_covers(i, t, level);
}


// Probe wrap-up: for every attachment changed to clear to the sentinel color, read back that level / layer, compare pixel by pixel with a same-format 1x1 sentinel pixel, and count what remains (> 0 means the coverage proof is wrong, report a failure)
static void lp_finish(void) {
    id<MTLCommandBuffer> dcb = q_cb();
    NSArray *ps = g_lp; g_lp = nil;
    for (NSArray *e in ps) {
        id<MTLTexture> t = e[0]; NSUInteger lv = [e[1] unsignedIntegerValue], sl = [e[2] unsignedIntegerValue];
        NSUInteger W = MAX(t.width >> lv, 1), H = MAX(t.height >> lv, 1), bpp = (NSUInteger)(att_bytes(t, lv) / (W * H));
        MTLTextureDescriptor *rd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:1 height:1 mipmapped:NO];
        rd.usage = MTLTextureUsageRenderTarget; rd.storageMode = MTLStorageModePrivate;
        id<MTLTexture> ref = [g_dev newTextureWithDescriptor:rd];
        MTLRenderPassDescriptor *pd = [MTLRenderPassDescriptor renderPassDescriptor];
        pd.colorAttachments[0].texture = ref; pd.colorAttachments[0].loadAction = MTLLoadActionClear; pd.colorAttachments[0].storeAction = MTLStoreActionStore;
        pd.colorAttachments[0].clearColor = [e[6] intValue] == 3 ? MTLClearColorMake(NAN, NAN, NAN, NAN) : [e[6] intValue] == 2 ? MTLClearColorMake(0.777, 0.222, 0.555, 0.888) : MTLClearColorMake(0.123, 0.456, 0.789, 0.321);
        [[dcb renderCommandEncoderWithDescriptor:pd] endEncoding];
        id<MTLBuffer> bt = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared], br = [g_dev newBufferWithLength:bpp options:MTLResourceStorageModeShared];
        id<MTLBlitCommandEncoder> b = [dcb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:sl sourceLevel:lv sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1) toBuffer:bt destinationOffset:0 destinationBytesPerRow:W * bpp destinationBytesPerImage:W * H * bpp];
        [b copyFromTexture:ref sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(1, 1, 1) toBuffer:br destinationOffset:0 destinationBytesPerRow:bpp destinationBytesPerImage:bpp];
    enc_b_end(b);
        unsigned long long pass = [e[3] unsignedLongLongValue], ser = [e[5] unsignedLongLongValue]; int idx = [e[4] intValue];
        [dcb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c;
            const uint8_t *p = bt.contents, *r = br.contents; uint64_t left = 0; char at[96] = ""; size_t ao = 0;
            for (NSUInteger k = 0; k < W * H; k++) if (!memcmp(p + k * bpp, r, bpp)) { if (left++ < 4) ao += snprintf(at + ao, sizeof at - ao, " (%lu,%lu)", (unsigned long)(k % W), (unsigned long)(k / W)); }
            if (left) MLOG("[RED] Readback-elision probe: channel %llu c%d ser=%llu lv%lu sl%lu %lux%lu sentinel pixels remaining %llu, first few positions%s", pass, idx, ser, (unsigned long)lv, (unsigned long)sl, (unsigned long)W, (unsigned long)H, (unsigned long long)left, at);
            else MLOG("Readback-elision probe: channel %llu c%d ser=%llu lv%lu sl%lu %lux%lu fully covered (0 sentinels remaining)", pass, idx, ser, (unsigned long)lv, (unsigned long)sl, (unsigned long)W, (unsigned long)H);
        }];
    }
    [dcb commit];
}
static int rt_same(const p_set_rt *a, const p_set_rt *b) {   // Same set of targets: compare by texture serial and view (not by object pointer: the game may copy texture objects by value)
    if (a->n != b->n || !a->depth != !b->depth || (a->depth && a->depth->serial != b->depth->serial) || a->has_dview != b->has_dview) return 0;
    if (a->has_dview && memcmp(&a->dview, &b->dview, sizeof a->dview)) return 0;
    for (int k = 0; k < a->n && k < 8; k++) {
        if (!a->color[k] != !b->color[k] || (a->color[k] && a->color[k]->serial != b->color[k]->serial) || a->has_cview[k] != b->has_cview[k]) return 0;
        if (a->has_cview[k] && memcmp(&a->cview[k], &b->cview[k], sizeof a->cview[k])) return 0;
    }
    return 1;
}

// Experimental RT range candidate: only ignore positive nlevels differences not used by the actual render attachments; other fields stay word-for-word identical.
#define NVN_RT_RANGE_IMPL 1
static int rt_range_shape(const p_set_rt *a, const p_set_rt *b) {
    if (a->n != 1 || b->n != 1 || !a->color[0] || !b->color[0] || !a->has_cview[0] || !b->has_cview[0] ||
        !a->depth != !b->depth || a->has_dview != b->has_dview ||
        (a->has_dview && memcmp(&a->dview,&b->dview,sizeof a->dview))) return 0;
    const tex_t *x=a->color[0], *y=b->color[0];
    if (!x->serial || x->serial != y->serial || x->d.target != NVN_TEXTURE_TARGET_2D || y->d.target != NVN_TEXTURE_TARGET_2D ||
        x->d.samples > 1 || y->d.samples > 1 || (tex_is_packaged(x) && tex_storage(x)) || (tex_is_packaged(y) && tex_storage(y)) ||
        memcmp(&x->d,&y->d,sizeof x->d) || tex_storage(x) != tex_storage(y) || x->pool != y->pool || x->off != y->off) return 0;
    const view_t *v=&a->cview[0], *w=&b->cview[0]; int levels=x->d.levels;
    if (!v->set_levels || !w->set_levels || v->base_level < 0 || v->base_level >= levels ||
        v->nlevels <= 0 || v->nlevels > levels-v->base_level || w->nlevels <= 0 || w->nlevels > levels-v->base_level ||
        (v->set_layers && (v->min_layer || v->nlayers != 1)) || (v->set_format && v->format != x->d.format) ||
        (v->set_target && v->target != NVN_TEXTURE_TARGET_2D)) return 0;
    view_t cv=*w; cv.nlevels=v->nlevels;
    if (memcmp(v,&cv,sizeof cv)) return 0;
    if (a->depth) { const tex_t *d=a->depth,*e=b->depth;
        if (!d->serial || d->serial != e->serial || d->d.target != NVN_TEXTURE_TARGET_2D || e->d.target != NVN_TEXTURE_TARGET_2D ||
            d->d.samples > 1 || e->d.samples > 1 || (tex_is_packaged(d) && tex_storage(d)) || (tex_is_packaged(e) && tex_storage(e)) ||
            memcmp(&d->d,&e->d,sizeof d->d) || tex_storage(d) != tex_storage(e) || d->pool != e->pool || d->off != e->off) return 0;
        int dl=a->has_dview && a->dview.set_levels ? a->dview.base_level : 0;
        if (dl < 0 || dl >= d->d.levels || mip_extent(d->d.width,dl) < mip_extent(x->d.width,v->base_level) ||
            mip_extent(d->d.height,dl) < mip_extent(x->d.height,v->base_level)) return 0; }

    return v->nlevels != w->nlevels;
}
#define NVN_RT_RANGE_DEPTH_IMPL 1
static void rt_range_note_depth(void) { if (rt_range_on() && S.ds.dw) g_pass.range_depth_nonro=1; }
static int rt_range_same(const p_set_rt *a, const p_set_rt *b) {
    if (!rt_range_on() || !g_enc || g_fz.active || g_pass.range_external || g_pass.range_depth_nonro || !S.ds.dt || S.ds.dw || S.ds.st || g_enc_mode != 4 || !g_pass.hd || g_pass.hs ||
        (g_vis_on && !g_enc_has_vis) || g_pass.reg_gen != g_reg_gen || g_pass.ser_gen != g_ser_gen || g_pass.range_dirty != nvn_dirty_now() || !rt_range_shape(a,b) || !b->depth) return 0;
    id<MTLTexture> color=tex_mtl(b->color[0]), depth=tex_mtl(b->depth);
    int lv=b->cview[0].base_level;
    return mip_extent(color.width,lv) == g_pass.tw && mip_extent(color.height,lv) == g_pass.th &&
        color == g_pass.range_color && depth == g_pass.range_depth && color.textureType == MTLTextureType2D &&
        depth.textureType == MTLTextureType2D && color.sampleCount == 1 && depth.sampleCount == 1 &&
        g_pass.reg_gen == g_reg_gen && g_pass.ser_gen == g_ser_gen && g_pass.range_dirty == nvn_dirty_now();
}
static int draw_samples_rt(NvProg *pr) {   // Sampled attachments keep the original conservative ser-level guard, not split by level/layer
    uint64_t rs[9]; int nr = 0;
    for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k]) rs[nr++] = R.rt.color[k]->serial;
    if (R.rt.depth) rs[nr++] = R.rt.depth->serial;
    if (!pr) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 && sh->stage < 6 ? S.img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 && sh->stage < 6 ? S.tex[sh->stage][b->reg] : 0; else continue;
            int tid = (int)(h & 0xfffff); if (!tid) continue;
            pthread_mutex_lock(&g_rmu); uint64_t ser = tid < g_ntreg && g_treg[tid].ok ? g_treg[tid].t.serial : 0; pthread_mutex_unlock(&g_rmu);
            for (int j = 0; j < nr; j++) if (ser && ser == rs[j]) return 1; } }
    return 0;
}
// Whole-channel depth / stencil requirements (bits: 1 depth, 2 stencil; 3 = both needed or cannot prove otherwise). Starting from this draw that opens the channel, scan in actual replay order until the channel ends:

// The whole render pass's need for depth / stencil (bits: 1 depth 2 stencil; 3 = both needed or cannot be proven). Starting from the draw that opens the render pass, scan in real replay order to the end of the render pass:
// the attached clears and this draw's state are counted first; later draws accumulate by their depth-stencil state; anything unprovable such as a mid-pass clear or nesting too deep is treated as both needed
static int pass_ds_need(void) {
    int need = (R.clear_d || S.ds.dt || S.ds.dw ? 1 : 0) | (R.clear_s || S.ds.st ? 2 : 0);
    if (!R.rt.depth || need == 3 || !g_exec_next || g_ret_n > 16) {
        PD("S %llu no lookahead depth=%d clear_d=%d clear_s=%d dt=%d dw=%d st=%d next=%d ret=%d", g_stat_pass + 1, R.rt.depth != NULL, R.clear_d, R.clear_s, (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.st, g_exec_next != NULL, g_ret_n);
        return 3;
    }
    ds_st ds = S.ds; g_ds_why = -1; g_ds_scanned = 0; g_ds_need = need;
    int r = ds_scan(g_exec_next, &ds, 0);
    for (int k = g_ret_n - 1; r == DS_RET && k >= 0; k--) r = ds_scan(g_ret[k], &ds, 0);
    for (int k = 0; r == DS_RET && k < g_nrest; k++) r = ds_scan((const uint8_t *)g_rest[k], &ds, 0);
    int out = r == DS_USED ? 3 : g_ds_need;
    PD("S %llu lookahead result=%d requirement=%d (1 depth 2 stencil) reason=%d scanned %d commands return stack %d remaining handles %d", g_stat_pass + 1, r, out, g_ds_why, g_ds_scanned, g_ret_n, g_nrest);
    return out;
}
// Pure-clear channel precondition: when opening a channel, scan forward for a pattern like "several draws in this channel → several groups (SetRenderTargets + clears only) → return to the same target as this channel",
// then execute those clears before this channel and skip them at their original positions; when returning to the same target, continue with that target—this avoids writing back and reading in the channel's attachments solely for clears interleaved between draws
// (Character selection screen: both grunge and particles draw level 0 of 28 / 27 + depth; only levels 1 and 2 are cleared in between.)
// Equivalent conditions (all must hold; otherwise leave unchanged): the scan range contains only draws, state changes, and these clears—no copies / dispatches / texture clears / sublists; the cleared child resource (texture + mip + layer)

// Pure-clear render pass hoisting: when a render pass is opened, scan forward, and if the shape is "several draws of this render pass -> several groups (SetRenderTargets + only clears) -> back to the same targets as this render pass",
// then hoist those groups of clears to execute before this render pass and skip them at their original positions, and on returning to the same targets the same-target continuation picks up -- this render pass's attachments are no longer stored whole and loaded again for the clears sandwiched in between
// (character select page: grunge and particles both draw 28 / 27 level 0 + depth, with only levels 1 and 2 cleared in between).
// Equivalence conditions (hoist only if all hold, otherwise leave as-is): the scan range has only draws, state and these clears, no copies / dispatches / texture clears / sub-lists; the cleared subresources (texture + level + layer)
// are disjoint from this render pass's attachments; the cleared textures are neither on any current texture / image slot nor bound within the scan range (conservative: regardless of whether the program samples them); the scissor at the time of the clear covers the whole block (same requirement as executing at the original position)
static uint64_t h_ser(uint64_t h) { int tid = (int)(h & 0xfffff); return tid ? atomic_load_explicit(&g_tser[tid], memory_order_relaxed) : 0; }
// ---------- Writeback lookahead index: at commit start, scan once linearly in replay order and record events by texture; at channel end, binary-search each attachment for the first valid event afterward to decide ----------
// Replaces the original “scan the entire remainder at each channel end” (channels × commands; 49 channels per frame in camp). Compared against the original scan attachment by attachment: screen gate 17 pages + camp about 340,000 attachment checks (about 46,000 extra), 0 mismatches; then removed the original scan.
// Events (seq = replay sequence): READ: draw / dispatch reads this texture, as determined by program reflection; ATT: it is the current attachment during a draw; CLEAR: attachment cleared as a whole; COPY / CLRTEX: either side of a copy / texture being cleared.
// Decision (first valid event decides): READ / COPY / CLRTEX → needed; ATT and its SetRenderTargets is after channel end → needed (read); CLEAR and likewise → redundant; no event → needed.

// ---------- Store lookahead index: at the start of a submit do one linear scan in replay order and record events per texture; when a render pass ends, binary-search each attachment for "the first valid event afterwards" to decide ----------
// Replaces the original "scan the whole remainder forward at every render pass end" (pass count x command count, 49 render passes per frame in camp). Cross-checked per attachment against the original scan: after 0 mismatches over the 17 pages of the screenshot gate + a camp run with about 340,000 attachment decisions (about 46,000 redundant), the original scan was removed.
// Events (seq = replay sequence number): READ a draw / dispatch reads this texture per program reflection; ATT it is a current attachment at a draw; CLEAR cleared as a whole block while an attachment; COPY / CLRTEX either end of a copy / the cleared texture.
// Decision (the first valid event decides): READ / COPY / CLRTEX -> needed; ATT whose SetRenderTargets is after the render pass end -> needed (load); CLEAR with the same -> redundant; no event -> needed.
// On any uncertainty (replay sequence number does not match the index, nesting too deep) the index is invalidated and all remaining render passes of this submit store (storing is always correct, it just saves less bandwidth).
typedef struct { uint32_t seq, rtseq; uint64_t ser; const uint8_t *op; uint16_t lv, sl; uint8_t kind, plane, anyls, why; uint16_t rep; } sdx_ev;   // rep: number of later same-pass ATT entries merged into this one (defer_pre step counts unchanged)   // plane 0 color 8 depth-stencil (ATT) 1 depth 2 stencil (CLEAR)
enum { SDX_READ = 1, SDX_NEED, SDX_ATT, SDX_CLEAR };
#define SDX_NONE 0xffffffffu
static sdx_ev *g_sdx, *g_sdx_tmp; static uint32_t g_sdx_n, g_sdx_cap; static int g_sdx_ok;   // Index valid

// Candidate textures (those that have been attachments) -> buckets: open addressing; a texture that has not yet been an attachment when an event occurs is irrelevant to any render pass that ended before then, so it is not recorded
#define SDX_HB 2048
static uint64_t g_sdx_hk[SDX_HB]; static uint16_t g_sdx_hb[SDX_HB]; static uint32_t g_sdx_nb, g_sdx_bcnt[SDX_HB + 1], g_sdx_bst[SDX_HB + 1];
static int sdx_bucket(uint64_t ser, int add) {   // Returns the bucket number; returns -1 if not in the set (and not adding); invalidates the index when full
    if (!ser) return -1;
    for (uint32_t i = (uint32_t)(ser * 0x9E3779B97F4A7C15ull >> 53), n = 0; n < SDX_HB; i = (i + 1) & (SDX_HB - 1), n++) {
        if (g_sdx_hk[i] == ser) return g_sdx_hb[i];
        if (!g_sdx_hk[i]) { if (!add) return -1; if (g_sdx_nb >= SDX_HB / 2) { g_sdx_ok = 0; return -1; } g_sdx_hk[i] = ser; g_sdx_hb[i] = (uint16_t)g_sdx_nb; return (int)g_sdx_nb++; }
    }
    return -1;
}
static const uint8_t **g_sdx_op; static uint32_t g_sdx_nop, g_sdx_opcap;
static uint32_t g_exec_seq;   // Sequence number of the current operation during replay
static uint64_t g_pp_slot[6][160]; static program_stage_binding g_pp_program_stage[6]; static const prog_t *g_pp_prog; static const p_set_rt *g_pp_rt; static uint32_t g_pp_rtseq;
static uint32_t *g_rtx_seq, *g_drx_seq; static uint32_t g_rtx_n, g_rtx_cap, g_drx_n, g_drx_cap;
// Within the same pass (same SetRenderTargets sequence number), every draw's "as attachment" events have identical content (same texture, same plane, same level/layer), and the decision only looks at the first:
// only the entries of the first draw in the pass are recorded, and later draws just increment rep on them (g_sdx_attdedup=0 returns to recording every draw, for in-process alternating measurement)
int g_sdx_attdedup = 1; static uint32_t g_att_rtseq = 0xffffffffu, g_att_idx[9]; static int g_att_n;   // This submit's SET_RT / draw sequence number tables (used by the pre-screen of the readable tile merge)
static void seq_push(uint32_t **v, uint32_t *n, uint32_t *cap, uint32_t x) { if (*n == *cap) { *cap = *cap ? *cap * 2 : 1024; *v = realloc(*v, *cap * sizeof **v); } (*v)[(*n)++] = x; }
static void sdx_ev_add(sdx_ev e) { if (sdx_bucket(e.ser, 0) < 0) return; if (g_sdx_n == g_sdx_cap) { g_sdx_cap = g_sdx_cap ? g_sdx_cap * 2 : 4096; g_sdx = realloc(g_sdx, g_sdx_cap * sizeof *g_sdx); g_sdx_tmp = realloc(g_sdx_tmp, g_sdx_cap * sizeof *g_sdx); } g_sdx[g_sdx_n++] = e; }
static void sdx_lvsl(const view_t *v, int hv, uint16_t *lv, uint16_t *sl) { *lv = (uint16_t)(hv && v->set_levels ? v->base_level : 0); *sl = (uint16_t)(hv && v->set_layers ? v->min_layer : 0); }
static void sdx_walk(const uint8_t *p, int depth) {
    if (depth > 64) { g_sdx_ok = 0; return; }
    for (uint32_t guard = 0; guard < 4000000 && p && g_sdx_ok; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (g_sdx_nop == g_sdx_opcap) { g_sdx_opcap = g_sdx_opcap ? g_sdx_opcap * 2 : 16384; g_sdx_op = realloc(g_sdx_op, g_sdx_opcap * sizeof *g_sdx_op); }
        uint32_t seq = g_sdx_nop; g_sdx_op[g_sdx_nop++] = p;
        switch (h.op) {
            case OP_END: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); sdx_walk((const uint8_t *)t, depth + 1); break; }
            case OP_SET_RT: { const p_set_rt *r = a; g_pp_rt = r; g_pp_rtseq = seq; seq_push(&g_rtx_seq, &g_rtx_n, &g_rtx_cap, seq);
                for (int k = 0; k < r->n && k < 8; k++) if (r->color[k]) sdx_bucket(r->color[k]->serial, 1);
                if (r->depth) sdx_bucket(r->depth->serial, 1);
                break; }
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; const p_set_rt *r = g_pp_rt;
                if (!r || (c->mask & 0xf) != 0xf || c->index < 0 || c->index >= r->n || c->index >= 8 || !r->color[c->index]) break;
                uint16_t lv, sl; sdx_lvsl(&r->cview[c->index], r->has_cview[c->index], &lv, &sl); int last = -1;   // When the same subresource is attached at several indices, the last one counts
                for (int k = 0; k < r->n && k < 8; k++) { uint16_t l2, s2; if (!r->color[k] || r->color[k]->serial != r->color[c->index]->serial) continue; sdx_lvsl(&r->cview[k], r->has_cview[k], &l2, &s2); if (l2 == lv && s2 == sl) last = k; }
                if (last == c->index) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->color[c->index]->serial, p, lv, sl, SDX_CLEAR, 0, 0, 0});
                break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = a; const p_set_rt *r = g_pp_rt; if (!r || !r->depth) break;
                uint16_t lv, sl; sdx_lvsl(&r->dview, r->has_dview, &lv, &sl);
                if (c->depth_mask) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_CLEAR, 1, !r->has_dview, 0});
                if ((c->stencil_mask & 0xff) == 0xff) sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_CLEAR, 2, !r->has_dview, 0});
                break; }
            case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_IMG: { p_bind_handles t1; const p_bind_handles *b = h.op == OP_BIND_IMG ? a : bind_rec(h.op, a, &t1); bh_fresh(b); int base = h.op == OP_BIND_IMG ? 2 * NVN_SEPTEX_SLOT0 : 0, lim = h.op == OP_BIND_IMG ? 32 : 2 * NVN_SEPTEX_SLOT0;
                if (b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) g_pp_slot[b->stage][base + b->first + k] = h_ser(b->h[k]);
                break; }
            case OP_BIND_PROGRAM: {
                program_binding_apply(g_pp_program_stage, (const p_bind_program *)a);
                g_pp_prog = program_binding_compose(g_pp_program_stage);
                break;
            }
            case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: {
                if (h.op != OP_DISPATCH) seq_push(&g_drx_seq, &g_drx_n, &g_drx_cap, seq);
                __unsafe_unretained NvProg *pr = g_pp_prog ? prog_get(g_pp_prog) : nil;
                if (!pr) { for (int st = 0; st < 6; st++) for (int k = 0; k < 160; k++) if (g_pp_slot[st][k]) sdx_ev_add((sdx_ev){seq, 0, g_pp_slot[st][k], p, 0, 0, SDX_READ, 0, 0, 1}); }
                else for (int st = 0; st < 6; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || (h.op == OP_DISPATCH) != (st == 5)) continue;
                    for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; int k = b->kind == NVN_RB_IMG ? 128 + b->reg : (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) ? b->reg : -1;
                        if (k < 0 || k >= 160 || sh->stage < 0 || sh->stage >= 6) continue; uint64_t ser = g_pp_slot[sh->stage][k];
                        if (ser) sdx_ev_add((sdx_ev){seq, 0, ser, p, 0, 0, SDX_READ, 0, 0, 1}); } }
                const p_set_rt *r = g_pp_rt;
                if (h.op != OP_DISPATCH && r && g_sdx_attdedup && g_att_rtseq == g_pp_rtseq) {
                    for (int q = 0; q < g_att_n; q++) if (g_sdx[g_att_idx[q]].rep < 0xffff) g_sdx[g_att_idx[q]].rep++;
                } else if (h.op != OP_DISPATCH && r) {
                    uint32_t n0 = g_sdx_n; g_att_rtseq = g_pp_rtseq; g_att_n = 0;
                    for (int k = 0; k < r->n && k < 8; k++) if (r->color[k]) { uint16_t lv, sl; sdx_lvsl(&r->cview[k], r->has_cview[k], &lv, &sl); sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->color[k]->serial, p, lv, sl, SDX_ATT, 0, 0, 2}); }
                    if (r->depth) { uint16_t lv, sl; sdx_lvsl(&r->dview, r->has_dview, &lv, &sl); sdx_ev_add((sdx_ev){seq, g_pp_rtseq, r->depth->serial, p, lv, sl, SDX_ATT, 8, !r->has_dview, 2}); }
                    for (uint32_t q = n0; q < g_sdx_n && g_att_n < 9; q++) if (g_sdx[q].kind == SDX_ATT) g_att_idx[g_att_n++] = q;
                }
                break; }
            case OP_COPY_T2T: { const p_copy_t2t *c = a; if (c->src) sdx_ev_add((sdx_ev){seq, 0, c->src->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); if (c->dst) sdx_ev_add((sdx_ev){seq, 0, c->dst->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); break; }
            case OP_COPY_T2B: case OP_COPY_B2T: { const p_copy_b2t *c = a; if (c->tex) sdx_ev_add((sdx_ev){seq, 0, c->tex->serial, p, 0, 0, SDX_NEED, 0, 0, 3}); break; }
            case OP_CLEAR_TEX: { const p_clear_tex *c = a; if (c->tex) sdx_ev_add((sdx_ev){seq, 0, c->tex->serial, p, 0, 0, SDX_NEED, 0, 0, 4}); break; }
            default: break;
        }
        p += h.size;
    }
}
void mtl_sub_index(const uint64_t *h, int n) {   // Start of submit (hle_nvn.c QueueSubmitCommands): build this submit's store lookahead index
    g_xv_ntouch = 0; g_sdx_n = g_sdx_nop = 0; g_rtx_n = g_drx_n = 0; g_exec_seq = 0; g_sdx_ok = tb_sub(TB_SDX); g_att_rtseq = SDX_NONE; g_att_n = 0;
    if (!g_sdx_ok) { g_xv_n = 0; g_xv_block = 1; return; }   // Index unavailable: clear the ledger and add nothing in this submit (the next touch cannot be verified)
    g_xv_block = 0;
    memset(g_sdx_hk, 0, sizeof g_sdx_hk); g_sdx_nb = 0;
    for (int i = 0; i < g_xv_n; i++) sdx_bucket(g_xv[i].ser, 1);   // Textures in the ledger enter the candidate set: otherwise, when one is only sampled and not used as an attachment in this submit, its read events are not indexed and the fallback cannot see them
    if (R.has_rt) { for (int k = 0; k < R.rt.n && k < 8; k++) if (R.rt.color[k]) sdx_bucket(R.rt.color[k]->serial, 1); if (R.rt.depth) sdx_bucket(R.rt.depth->serial, 1); }   // The targets at the start of the submit (the first draw may reopen the render pass without going through SetRenderTargets)
    memcpy(g_pp_program_stage, S.program_stage, sizeof g_pp_program_stage); g_pp_prog = S.prog; g_pp_rt = NULL; g_pp_rtseq = SDX_NONE;
    for (int st = 0; st < 6; st++) { for (int k = 0; k < 128; k++) g_pp_slot[st][k] = S.tex[st][k] ? h_ser(S.tex[st][k]) : 0;
                                     for (int k = 0; k < 32; k++) g_pp_slot[st][128 + k] = S.img[st][k] ? h_ser(S.img[st][k]) : 0; }
    for (int i = 0; i < n && g_sdx_ok; i++) sdx_walk((const uint8_t *)h[i], 0);
    if (!g_sdx_ok) return;

    // Counting sort into buckets: events are appended in sequence-number order, and within the same sequence number READ precedes ATT (the same draw is judged for reads first, then attachments), so after a stable scatter each bucket is ordered
    memset(g_sdx_bcnt, 0, (g_sdx_nb + 1) * sizeof *g_sdx_bcnt);
    for (uint32_t i = 0; i < g_sdx_n; i++) g_sdx_bcnt[sdx_bucket(g_sdx[i].ser, 0)]++;
    for (uint32_t b = 0, acc = 0; b <= g_sdx_nb; b++) { g_sdx_bst[b] = acc; acc += b < g_sdx_nb ? g_sdx_bcnt[b] : 0; }
    for (uint32_t b = 0; b < g_sdx_nb; b++) g_sdx_bcnt[b] = g_sdx_bst[b];
    for (uint32_t i = 0; i < g_sdx_n; i++) g_sdx_tmp[g_sdx_bcnt[sdx_bucket(g_sdx[i].ser, 0)]++] = g_sdx[i];
    sdx_ev *t = g_sdx; g_sdx = g_sdx_tmp; g_sdx_tmp = t;
    xv_forward();
}
static void sdx_seq(const uint8_t *p) {   // Every operation in mtl_exec: check the sequence number against the index (on mismatch invalidate, and store everything for the rest)
    if (!g_sdx_ok) return;
    if (g_exec_seq >= g_sdx_nop || g_sdx_op[g_exec_seq] != p) { RED_ONCE(40, "Writeback lookahead indices do not match replay order (sequence %u / %u); write back all remaining channels in this submission", g_exec_seq, g_sdx_nop); g_sdx_ok = 0; return; }
    g_exec_seq++;
}
static int sdx_decide(uint8_t st[9]) {   // st[i]: 2 redundant; otherwise why << 2 (the reason it is needed, same as sd_need). Returns 0 = index unavailable
    if (!g_sdx_ok || !g_exec_cur || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 0;
    uint32_t P = g_exec_seq - 1;
    for (int i = 0; i < g_nsd; i++) { const sd_att *t = &g_sd[i]; st[i] = 5 << 2;
        int b = sdx_bucket(t->ser, 0); if (b < 0) continue;   // Not in the candidate set: no related event at all in this submit -> undecided after scanning to the end, needed
        uint32_t lo = g_sdx_bst[b], end = g_sdx_bst[b + 1], hi = end;   // First seq >= P in the bucket
        while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_sdx[m].seq < P) lo = m + 1; else hi = m; }
        for (uint32_t j = lo; j < end; j++) { const sdx_ev *e = &g_sdx[j];
            if (g_hoist_n && hoist_skipped(e->op)) continue;
            if (e->kind == SDX_READ || e->kind == SDX_NEED) { st[i] = (uint8_t)(e->why << 2); break; }
            if (e->rtseq == SDX_NONE || e->rtseq < P || (g_hoist_n && hoist_skipped(g_sdx_op[e->rtseq]))) continue;   // This target was set before the render pass ended: not re-set as an attachment after the render pass ended, so it does not count
            int hit = t->depth ? (e->kind == SDX_ATT ? e->plane == 8 : e->plane == t->depth) && (e->anyls || (e->lv == t->lv && e->sl == t->sl))
                               : e->plane == 0 && e->lv == t->lv && e->sl == t->sl;
            if (!hit) continue;
            st[i] = e->kind == SDX_CLEAR ? 2 : (uint8_t)(2 << 2); break; }
    }
    return 1;
}
static void xv_forward(void) {
    if (!g_xv_n) return;
    if (!g_sdx_ok) { g_xv_n = 0; g_xv_block = 1; return; }
    for (int i = 0; i < g_xv_n;) {
        xv_ent *e = &g_xv[i]; int b = sdx_bucket(e->ser, 0), action = 0;
        if (b >= 0) for (uint32_t j = g_sdx_bst[b]; j < g_sdx_bst[b + 1]; j++) { sdx_ev *v = &g_sdx[j];
            if (v->kind == SDX_READ || v->kind == SDX_NEED) { xv_ent copy = *e; xv_violation(&copy, "Submission lookahead read/requirement"); action = 2; break; }
            if (v->lv != e->lv || v->sl != e->sl) continue;
            if (v->kind == SDX_CLEAR) { xv_good(e); action = 1; break; }
            if (v->kind == SDX_ATT) break;
        }
        if (!action) i++;
    }
}
static void rt_sub(const p_set_rt *r, int k, uint64_t *ser, int *lv, int *sl) {   // Subresource of attachment k (0–7 color, 8 depth)
    const tex_t *t = k == 8 ? r->depth : r->color[k]; const view_t *v = k == 8 ? &r->dview : &r->cview[k]; int hv = k == 8 ? r->has_dview : r->has_cview[k];
    *ser = t ? t->serial : 0; *lv = hv && v->set_levels ? v->base_level : 0; *sl = hv && v->set_layers ? v->min_layer : 0;
}
static void hoist_try(void) {
    if (g_hoist_n || !g_exec_next) return;
    typedef struct { const uint8_t *rt_op; const p_set_rt *rt; uint8_t clear_c[8]; MTLClearColor cc[8]; uint8_t clear_d, clear_s; double cd; uint32_t cs;
                     p_scissor sc; int has_sc; const uint8_t *ops[8]; int nops; } grp;
    grp G[4]; int ng = 0, in_grp = 0; p_scissor sc = R.sc; int has_sc = R.has_sc;
    // The binding set needed for “cleared texture is not in any slot”: look it up only after actually finding a clear group that can be skipped and returning to this channel’s target (most channels never reach that point).

    // The binding set needed for "the cleared texture is not on any slot": the table is looked up only when a hoistable clear group is really found and we are back to this render pass's targets (the vast majority of render passes never get that far).
    // Handles bound within the scan range are noted first and converted to texture serials together with the current slots at lookup time; the scan only reads commands and the current slots S do not change meanwhile, so the result is the same as checking at the start
    uint64_t bh[512]; int nbh = 0;
    const uint8_t *p = g_exec_next;
    for (int guard = 0; guard < 200000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        switch (h.op) {
            case OP_END: case OP_CALL: case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: case OP_WAIT_SYNC: return;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_DRAW: case OP_DRAW_IDX: if (in_grp) return; break;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_IMG: { p_bind_handles t1; const p_bind_handles *b = h.op == OP_BIND_IMG ? a : bind_rec(h.op, a, &t1); bh_fresh(b); for (int k = 0; k < b->n && k < 32 && nbh < 512; k++) if (b->h[k]) bh[nbh++] = b->h[k]; break; }
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: {
                if (!in_grp || G[ng - 1].nops >= 8) return;
                grp *g = &G[ng - 1];
                if (h.op == OP_CLEAR_COLOR) { const p_clear_color *c = a;
                    if (c->index < 0 || c->index >= g->rt->n || !g->rt->color[c->index] || c->mask != 0xf) return;
                    const tex_t *t = g->rt->color[c->index];
                    if (has_sc && !(sc.x <= 0 && sc.y <= 0 && sc.x + sc.w >= (int)t->d.width && sc.y + sc.h >= (int)t->d.height)) return;
                    g->clear_c[c->index] = 1; g->cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]); }
                else { const p_clear_ds *c = a; if (!g->rt->depth) return;
                    if (c->depth_mask) { g->clear_d = 1; g->cd = c->depth; } if (c->stencil_mask) { g->clear_s = 1; g->cs = (uint32_t)c->stencil; } }
                g->sc = sc; g->has_sc = has_sc; g->ops[g->nops++] = p; break; }
            case OP_SET_RT: { const p_set_rt *r = a;
                if (rt_same(&R.rt, r)) {   // Back to this render pass's targets: execute if hoisting holds
                    if (!ng) return;
                    uint64_t bound[512]; int nb = 0;   // The order of the original "check at the start": the current slots (stage x textures 128 + images 32), followed by those bound within the scan range
                    for (int st = 0; st < 6; st++) { for (int k = 0; k < 128 && nb < 512; k++) if (S.tex[st][k]) bound[nb++] = h_ser(S.tex[st][k]);
                                                     for (int k = 0; k < 32 && nb < 512; k++) if (S.img[st][k]) bound[nb++] = h_ser(S.img[st][k]); }
                    for (int k = 0; k < nbh && nb < 512; k++) bound[nb++] = h_ser(bh[k]);
                    for (int gi = 0; gi < ng; gi++) {   // Disjoint and not bound
                        grp *g = &G[gi]; if (!g->nops) return;
                        for (int k = 0; k < 9; k++) { if (k < 8 ? !g->clear_c[k] : !(g->clear_d || g->clear_s)) continue;
                            uint64_t ser; int lv, sl; rt_sub(g->rt, k, &ser, &lv, &sl); if (!ser) return;
                            for (int j = 0; j < nb; j++) if (bound[j] == ser) return;
                            for (int m = 0; m < 9; m++) { uint64_t s2; int l2, sl2; if (m < 8 && m >= R.rt.n) continue; rt_sub(&R.rt, m, &s2, &l2, &sl2);
                                if (s2 == ser && l2 == lv && sl2 == sl) return; } } }

                    // Register all skip entries first, then execute: when a hoisted render pass ends the store lookahead scans forward, and the clears at their original positions must already be in the skip set
                    for (int gi = 0; gi < ng; gi++) { g_hoist_skip[g_hoist_n++] = G[gi].rt_op; for (int k = 0; k < G[gi].nops; k++) g_hoist_skip[g_hoist_n++] = G[gi].ops[k]; }
                    p_set_rt rt0 = R.rt; p_scissor sc0 = R.sc; int hs0 = R.has_sc;   // This render pass's targets, scissor and pending clears: restored as they were after execution
                    uint8_t cc0f[8]; MTLClearColor cc0[8]; uint8_t cd0f = R.clear_d, cs0f = R.clear_s; double cd0 = R.cd; uint32_t cs0 = R.cs;
                    memcpy(cc0f, R.clear_c, 8); memcpy(cc0, R.cc, sizeof cc0);
                    for (int gi = 0; gi < ng; gi++) {   // Execute: temporarily swap in that group's targets and clears and open a clear-only render pass
                        grp *g = &G[gi];
                        R.rt = *g->rt; memcpy(R.clear_c, g->clear_c, 8); memcpy(R.cc, g->cc, sizeof R.cc); R.clear_d = g->clear_d; R.clear_s = g->clear_s; R.cd = g->cd; R.cs = g->cs;
                        R.sc = g->sc; R.has_sc = g->has_sc;
                        PD("H %llu preceding pure-clear channel (original position is after this channel)", g_stat_pass + 1);
                        rp_begin_ex(1); rp_end();
                    }
                    R.rt = rt0; R.sc = sc0; R.has_sc = hs0; memcpy(R.clear_c, cc0f, 8); memcpy(R.cc, cc0, sizeof cc0); R.clear_d = cd0f; R.clear_s = cs0f; R.cd = cd0; R.cs = cs0;
                    if (!g_hoist_groups) MLOG("Pure-clear channel prerequisite: first case %d groups (before channel %llu)", ng, g_stat_pass + 1);
                    g_hoist_groups += (uint64_t)ng;
                    return;
                }
                if (in_grp && !G[ng - 1].nops) return;   // The previous group has no clear (only set targets): not recognized
                if (ng >= 4) return;
                memset(&G[ng], 0, sizeof G[ng]); G[ng].rt_op = p; G[ng].rt = r; ng++; in_grp = 1; break; }
            default: break;
        }
        p += h.size;
    }
}
// ---------- Readable-tile channel fusion (Fusion B, step 2) ----------
// When a later channel’s draw samples, per pixel, a color attachment written by the previous channel, merge the two channels into one render channel: the previous channel’s contents remain in the tile,
// and the later channel uses a readable-tile variant (<key>_bf, function-constant-specialized) to read directly, without first spilling to main memory and reading back for sampling.
// Planning (fuse_try, at the start of the previous channel) performs only structural checks; the closed-form proof of per-pixel addressing is performed on each draw in the later channel (fz_p2_draw). If it cannot be proven, end the fused channel,

// ---------- Readable tile render pass merge (merge B step 2) ----------
// When a draw of the later pass samples, at the same pixel, a color attachment written by the earlier pass, the two passes are merged into one render pass: what the earlier pass wrote stays in the tile,
// and the later pass reads it directly with the readable tile version (<key>_bf, function-constant specialization), instead of first landing in main memory and being read back just to be sampled.
// Planning (fuse_try, at the start of the earlier pass) does only structural checks; the closed-form proof of same-pixel addressing is done at every draw of the later pass (fz_p2_draw), and if it cannot be proven the merged render pass is ended and
// a normal render pass is reopened with the later pass's original targets (the earlier pass's result is stored as usual, equivalent to not merging).
static void fz_unskip(void) { for (int i = 0; i < g_fz.nskip; i++) for (int j = 0; j < g_hoist_n; j++) if (g_hoist_skip[j] == g_fz.skip[i]) { g_hoist_skip[j] = g_hoist_skip[--g_hoist_n]; break; } }
static int fz_reuse_op(const uint8_t *p) { for (int i = 0; i < g_fz.nreuse; i++) if (g_fz.reuse[i] == p) return 1; return 0; }
// Diagnostic DM_NVN_FUSE_CHECK=1: dual rendering within the frame. Save a state snapshot for each draw in the later channel that passes the proof; after the fused channel ends, copy the attachments written by the later channel (the fused-path result),

// Diagnostic DM_NVN_FUSE_CHECK=1: in-run double rendering. For the later pass's draws that can be proven, keep a state snapshot per draw; after the merged render pass ends, make a copy of the attachments the later pass wrote (the merge path's result),
// then turn off all lookahead and redraw one by one through the normal path with the later pass's original targets and original clears, and compare the two byte by byte (same frame, same input, no dependence on cross-run determinism). The redraw result is left on the attachment (no effect if identical)
static int fz_check_on(void) { static int v = -1; if (v < 0) v = getenv("DM_NVN_FUSE_CHECK") != NULL; return v; }
typedef struct { int idx; p_draw dw; p_draw_idx di; p_scissor sc; int has_sc; void *S_; } fz_snap;
static fz_snap g_fzs[8]; static int g_nfzs, g_fzs_bad, g_fz_checking; static uint64_t g_fzc_same, g_fzc_diff;
static void draw(const p_draw *dw, const p_draw_idx *di);
static void fz_snap_take(const p_draw *dw, const p_draw_idx *di) {
    if (g_nfzs >= 8) { g_fzs_bad = 1; return; }
    fz_snap *z = &g_fzs[g_nfzs++]; z->idx = di != NULL; if (dw) z->dw = *dw; if (di) z->di = *di; z->sc = R.sc; z->has_sc = R.has_sc;
    z->S_ = malloc(sizeof S); memcpy(z->S_, &S, sizeof S);
}
static void fz_check_run(const fuse_t *fz) {
    if (!g_nfzs || g_fzs_bad) { if (g_fzs_bad) MLOG("[RED] In-frame dual rendering: later channel drew more than 8 times; not compared"); goto out; }
    {
    id<MTLTexture> att[8], scr[8]; uint64_t sers[8]; int na = 0;

    // fz_end has already restored R.rt from the physical union to the later pass's logical targets; wrote is still registered by physical slot and must be aligned through p2map.
    for (int j = 0; j < fz->rt2.n && j < 8; j++) if (fz->p2map[j] >= 0 && fz->wrote[fz->p2map[j]] && fz->rt2.color[j]) {
        id<MTLTexture> t = tex_mtl(fz->rt2.color[j]); if (!t) continue;
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:t.width height:t.height mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> c = [g_dev newTextureWithDescriptor:d]; if (!c) continue;
        enc_b_copy_texture(blit(), t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(t.width, t.height, 1), c, 0, 0, MTLOriginMake(0, 0, 0));
        att[na] = t; scr[na] = c; sers[na] = fz->rt2.color[j]->serial; na++; }
    if (!na) { MLOG("[RED] In-frame dual rendering: retained %d draw snapshots, but no comparable attachment", g_nfzs); goto out; }
    static typeof(S) Ssave; memcpy(&Ssave, &S, sizeof S); typeof(R) Rsave = R; const uint8_t *en0 = g_exec_next;
    g_fz_checking = 1; g_exec_next = NULL;   // Turn off all lookahead during the redraw (continuation / hoisting / merging / depth-stencil on demand): normal path
    R.rt = fz->rt2; memcpy(R.clear_c, fz->clr2, 8); memcpy(R.cc, fz->cc2, sizeof R.cc); R.clear_d = fz->cd2f; R.clear_s = fz->cs2f; R.cd = fz->cd2; R.cs = fz->cs2;
    for (int i = 0; i < g_nfzs; i++) { fz_snap *z = &g_fzs[i]; memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1;
        draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
    rp_end();
    memcpy(&S, &Ssave, sizeof S); args_dirty_all(); R = Rsave; g_exec_next = en0; g_fz_checking = 0; g_pso_dirty = g_dss_dirty = 1;
    for (int a = 0; a < na; a++) {
        id<MTLTexture> t = att[a], c = scr[a]; NSUInteger W = t.width, H = t.height, bpp = (NSUInteger)(att_bytes(t, 0) / (W * H));
        id<MTLBuffer> b1 = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared], b2 = [g_dev newBufferWithLength:W * H * bpp options:MTLResourceStorageModeShared];
        BENC b = blit();
        enc_b_copy_texture_buffer(b, c, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), b1, 0, W * bpp, W * H * bpp);
        enc_b_copy_texture_buffer(b, t, 0, 0, MTLOriginMake(0, 0, 0), MTLSizeMake(W, H, 1), b2, 0, W * bpp, W * H * bpp);
        unsigned long long ser = sers[a], pass = g_stat_pass, frame = atomic_load(&g_cur_frame); NSUInteger nb = W * H * bpp;
        [g_commit_blocks addObject:^{
            const uint8_t *x = b1.contents, *y = b2.contents; uint64_t dpx = 0, maxd = 0;
            for (NSUInteger k = 0; k < nb; k += bpp) if (memcmp(x + k, y + k, bpp)) { dpx++; for (NSUInteger q = 0; q < bpp; q++) { uint64_t dd = (uint64_t)abs((int)x[k + q] - (int)y[k + q]); if (dd > maxd) maxd = dd; } }
            if (dpx) { g_fzc_diff++; MLOG("[RED] In-frame dual rendering: frame %llu channel %llu attachment ser=%llu merged vs. normal: %llu / %lu pixels differ (maximum bytewise difference %llu)", frame, pass, ser, (unsigned long long)dpx, (unsigned long)(nb / bpp), (unsigned long long)maxd); }
            else { g_fzc_same++; MLOG("In-frame dual rendering: frame %llu channel %llu attachment ser=%llu merged vs. normal are byte-for-byte identical (%lu pixels)", frame, pass, ser, (unsigned long)(nb / bpp)); }
            if ((g_fzc_same + g_fzc_diff) % 300 == 0) MLOG("In-frame dual-rendering totals: byte-for-byte identical %llu times, different %llu times", (unsigned long long)g_fzc_same, (unsigned long long)g_fzc_diff);
        }];
    }
    }
out:
    for (int i = 0; i < g_nfzs; i++) free(g_fzs[i].S_);
    g_nfzs = 0; g_fzs_bad = 0;
}
static void fz_end(void) {   // End of a merged render pass (called by rp_end): interrupted before entering the later pass -> retract the skip entries and restore the targets to the earlier pass's; otherwise restore to the later pass's
    if (!g_fz.active) return;
    if (g_fz.phase == 1) { fz_unskip(); R.rt = g_fz.rt1; } else R.rt = g_fz.rt2;
    g_fz.active = 0;
    if (g_fz.phase == 2 && fz_check_on() && !g_fz_checking) { fuse_t fz = g_fz; R.rt = fz.rt2; fz_check_run(&fz); }
    else { for (int i = 0; i < g_nfzs; i++) free(g_fzs[i].S_); g_nfzs = 0; g_fzs_bad = 0; }
}
static int sub_eq(const p_set_rt *a, int ka, const p_set_rt *b, int kb) { uint64_t s1, s2; int l1, l2, x1, x2; rt_sub(a, ka, &s1, &l1, &x1); rt_sub(b, kb, &s2, &l2, &x2); return s1 && s1 == s2 && l1 == l2 && x1 == x2; }
static int prog_samples_ser(const prog_t *pg, uint64_t tex[6][2 * NVN_SEPTEX_SLOT0], uint64_t img[6][32], uint64_t ser) {   // Whether this texture is bound in any texture / image / typed buffer slot declared by the program's reflection
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0; else continue;
            if (h && h_ser(h) == ser) return 1; } }
    return 0;
}
static int g_defer_env_ab, g_defer_env_off;   // Environment switches: read once at the start of each submit (defer_reset); the hot path does not call getenv
static void defer_env(void) { const char *x = getenv("DM_NVN_DEFER_AB"), *e = getenv("DM_NVN_DEFER"); g_defer_env_ab = x && !strcmp(x, "1"); g_defer_env_off = e && !strcmp(e, "0"); }
static int defer_on(void) {
    if (!g_defer_env_ab) return tb_sub(TB_DEFER) && !g_defer_env_off;
    if (g_defer_ab < 0) { g_defer_ab = 0; g_defer_ab_on = 0; }
    return tb_sub(TB_DEFER) && (g_defer_ab || g_defer_ab_on); }
static int defer_handle_sub(uint64_t h, const p_set_rt *r, int k, int cover) {
    int tid = (int)(h & 0xfffff); if (!tid || tid >= g_ntreg || !g_treg[tid].ok) return 0;
    uint64_t ser; int lv, sl; rt_sub(r, k, &ser, &lv, &sl); if (!ser || g_treg[tid].t.serial != ser) return 0;
    if (!g_treg[tid].has_v || !g_treg[tid].v.set_levels) return cover;
    view_t v = g_treg[tid].v; int n = v.nlevels > 0 ? v.nlevels : 1;
    return lv >= v.base_level && lv < v.base_level + n && (!v.set_layers || (sl >= v.min_layer && sl < v.min_layer + (v.nlayers > 0 ? v.nlayers : 1)));
}
static int defer_prog_hits(const prog_t *pg, uint64_t tex[6][2 * NVN_SEPTEX_SLOT0], uint64_t img[6][32], const p_set_rt *r, int color_only, int cover) {
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr) return 1;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage < 0 || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h = 0;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0;
            else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0;
            else continue;
            if (!h) continue;
            for (int k = 0; k < (color_only ? r->n : 9); k++) if ((color_only ? r->color[k] : (k < 8 ? r->color[k] : r->depth)) && defer_handle_sub(h, r, k, cover)) return 1;
        }
    }
    return 0;
}
static void defer_collect_reads(defer_t *d, const prog_t *pg, uint64_t tex[6][2 * NVN_SEPTEX_SLOT0], uint64_t img[6][32]) {
    __unsafe_unretained NvProg *pr = pg ? prog_get(pg) : nil; if (!pr) { d->nreads = 513; return; }
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage < 0 || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h = 0;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? img[sh->stage][b->reg] : 0;
            else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? tex[sh->stage][b->reg] : 0;
            else continue;
            if (!h) continue; int seen = 0; for (int j = 0; j < d->nreads; j++) seen |= d->p_reads[j] == h;
            if (!seen && d->nreads < 512) d->p_reads[d->nreads++] = h;
        }
    }
}
static int defer_rt_hits_reads(const p_set_rt *r, int idx) {
    if (idx < 0 || idx >= r->n || !r->color[idx]) return 0;
    for (int i = 0; i < g_defer.nreads; i++) if (defer_handle_sub(g_defer.p_reads[i], r, idx, 1)) return 1;
    return 0;
}
static int defer_q_superset(const p_set_rt *q) {
    memset(g_defer.lmap, 0xff, sizeof g_defer.lmap);
    for (int i = 0; i < g_defer.p_rt.n; i++) if (g_defer.p_rt.color[i]) { int found = 0;
        // Indices must match: color(n) output from a regular fragment function is written to physical attachment n; lmap changes only the write mask / blending and cannot move outputs (only the _bf variant with output-mapping constants can move them).

        // The indices must be the same: a normal fragment function's color(n) output is written to physical attachment n, and lmap only changes the write mask / blend and cannot move the output (only the _bf version with output mapping constants can).
        // With different indices, deferral would write P's output into a different attachment of Q.
        if (i < q->n && q->color[i] && sub_eq(&g_defer.p_rt, i, q, i)) { g_defer.lmap[i] = (uint8_t)i; found = 1; }
        if (!found) return 0;
    }
    return 1;
}
static int defer_take(const p_draw *dw, const p_draw_idx *di) {
    if (g_defer.n >= 64) { g_defer.reason = 20; return 0; }
    defer_snap *z = &g_defer.snap[g_defer.n++]; z->idx = di != NULL; if (dw) z->dw = *dw; if (di) z->di = *di;
    z->sc = R.sc; z->has_sc = R.has_sc; z->S_ = malloc(sizeof S); memcpy(z->S_, &S, sizeof S); for (int k = 0; k < 8; k++) z->pm[k] = (uint8_t)k; g_defer_draws++; return 1;
}
static void defer_release(void) { defer_release_one(&g_defer); }
static void defer_env(void);
static void defer_reset(void) { defer_clear_all(); g_defer_replaying = 0; defer_env(); }
static void defer_apply_p_clear(void) {
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    for (int j = 0; j < g_defer.q_rt.n; j++) { int i = g_defer.lmap[j]; if (i >= 0 && g_defer.clear_c[i]) { R.clear_c[j] = 1; R.cc[j] = g_defer.cc[i]; } }
    if (g_defer.clear_d) { R.clear_d = 1; R.cd = g_defer.cd; } if (g_defer.clear_s) { R.clear_s = 1; R.cs = g_defer.cs; }
}
static int defer_begin_q(const uint8_t *op) {
    int found = 0, first = -1;
    for (int i = 0; i < 4; i++) if (g_defer_slots[i].active && !g_defer_slots[i].at_q && g_defer_slots[i].q_op == op) { found = 1; if (first < 0) first = i; }
    if (!found) return 0;
    rp_flush();
    g_defer_cur = &g_defer_slots[first]; R.rt = g_defer.q_rt; R.has_rt = 1;
    for (int i = 0; i < 4 && !R.rt.depth; i++) { const defer_t *d = &g_defer_slots[i]; if (d->active && !d->at_q && d->q_op == op && d->q_rt.depth) { R.rt.depth = d->q_rt.depth; R.rt.dview = d->q_rt.dview; R.rt.has_dview = d->q_rt.has_dview; } }   // Attach the depth-stencil needed by any group (the depths of groups with the same Q have been checked compatible)
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0;
    for (int i = 0; i < 4; i++) { defer_t *d = &g_defer_slots[i]; if (!d->active || d->at_q || d->q_op != op) continue;
        for (int j = 0; j < d->q_rt.n; j++) { int k = d->lmap[j]; if (k >= 0 && d->clear_c[k]) { R.clear_c[j] = 1; R.cc[j] = d->cc[k]; } }
        if (d->clear_d) { R.clear_d = 1; R.cd = d->cd; } if (d->clear_s) { R.clear_s = 1; R.cs = d->cs; }
        d->at_q = 1; g_defer_natq++;
    }
    return 1;
}
static void defer_replay(void) {
    if (g_defer_replaying) return;
    int any = 0; for (int i = 0; i < 4; i++) if (g_defer_slots[i].active && g_defer_slots[i].at_q) any = 1;
    if (!any) return;
    typeof(S) save; memcpy(&save, &S, sizeof S); p_scissor sc = R.sc; int has_sc = R.has_sc; int vis = g_vis_counting;
    const uint8_t *en0 = g_exec_next; g_defer_replaying = 1; g_exec_next = NULL;
    for (int gi, after = 0; (gi = defer_next(1, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born;   // In creation order: the groups' P ranges do not overlap and are arranged in sequence, so the group order is the original draw order
        g_defer_cur = d; if (d->reset_between) g_vis_counting = 0;
        { static int once; if (!once++) MLOG("Deferred-merge channel: first case channel %llu, group %d replayed %d draws", g_stat_pass, gi, d->n); }   // With multiple groups, several groups may already have been created before the first replay, so it cannot be judged by g_defer_n == 1
        for (int i = 0; i < d->n; i++) { defer_snap *z = &d->snap[i]; static uint8_t lm[8]; for (int j = 0; j < 8; j++) lm[j] = d->lmap[j] == 0xff || d->lmap[j] >= 8 ? 0xff : z->pm[d->lmap[j]];
            memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1; g_pso_lmap = lm; draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
        defer_release_one(d);
    }
    g_vis_counting = vis; memcpy(&S, &save, sizeof S); args_dirty_all(); R.sc = sc; R.has_sc = has_sc; g_pso_lmap = NULL; g_pso_fsfn = nil; memset(&g_pso_fz, 0, sizeof g_pso_fz); g_pso_dirty = g_dss_dirty = 1; g_exec_next = en0; g_defer_replaying = 0; g_defer_cur = &g_defer_slots[0];
}
static void defer_fallback(void) {
    if (g_defer_replaying || !defer_has_active()) return;
    g_defer_fallbacks++; MLOG("[RED] Deferred-merge channel: unreplayed groups remain at submission end; fall back to each group's original P target");
    if (g_enc) { enc_r_end(g_enc); g_enc = nil; }
    typeof(S) save; memcpy(&save, &S, sizeof S); p_set_rt rt = R.rt; g_defer_replaying = 1;
    for (int gi, after = 0; (gi = defer_next(0, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born; g_defer_cur = d; R.rt = d->p_rt; R.has_rt = 1; memcpy(R.clear_c, d->clear_c, 8); memcpy(R.cc, d->cc, sizeof R.cc);
        R.clear_d = d->clear_d; R.clear_s = d->clear_s; R.cd = d->cd; R.cs = d->cs;
        for (int i = 0; i < d->n; i++) { defer_snap *z = &d->snap[i]; memcpy(&S, z->S_, sizeof S); args_dirty_all(); R.sc = z->sc; R.has_sc = z->has_sc; g_pso_dirty = g_dss_dirty = 1; g_pso_lmap = z->pm; draw(z->idx ? NULL : &z->dw, z->idx ? &z->di : NULL); }
        rp_end(); defer_release_one(d);
    }
    memcpy(&S, &save, sizeof S); args_dirty_all(); R.rt = rt; g_defer_replaying = 0; g_defer_cur = &g_defer_slots[0];
}
static const uint8_t *defer_cont(int *ci) {   // The next segment after a list ends (within the same submit): first the caller return stack, then the remaining handles of this submit; returns NULL when there are none left
    for (;;) { int k = (*ci)++;
        if (k < g_ret_n) { if (g_ret_n > 16) return NULL; return g_ret[g_ret_n - 1 - k]; }
        k -= g_ret_n; if (k < g_nrest) return (const uint8_t *)g_rest[k];
        return NULL; }
}
// Deferred prefilter (binary-search writeback lookahead index, zero scans): for each color subresource P, after this draw the first “as attachment / cleared” event must be “as attachment under another SetRenderTargets,”

// Deferral pre-screen (binary search on the store lookahead index, zero scanning): for every color subresource of P, the first "used as attachment / cleared" event after this draw must be "used as attachment under a different SetRenderTargets,
// whose targets cover all of P's colors"; cleared first / never used as attachment again in this submit -> deferral is impossible, return directly. Read events carry no level; handed to the full lookahead to check by level.
static int defer_pre(void) {
    if (!g_sdx_ok || !g_exec_cur || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 0;
    uint32_t P = g_exec_seq - 1;
    for (int k = 0; k < R.rt.n; k++) { if (!R.rt.color[k]) continue;
        uint64_t ser; int lv, sl; rt_sub(&R.rt, k, &ser, &lv, &sl);
        int b = sdx_bucket(ser, 0); if (b < 0) return 0;
        uint32_t lo = g_sdx_bst[b], end = g_sdx_bst[b + 1], hi = end;
        while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_sdx[m].seq <= P) lo = m + 1; else hi = m; }

        // SetRenderTargets sequence number of P's own pass: previously taken from this draw's as-attachment event; after as-attachment events were deduplicated per pass, it is taken from the SET_RT sequence table as the last one before P (same value)
        uint32_t prt = SDX_NONE;
        if (!g_sdx_attdedup) prt = lo > g_sdx_bst[b] && g_sdx[lo - 1].seq == P ? g_sdx[lo - 1].rtseq : SDX_NONE;
        else { uint32_t a0 = 0, a1 = g_rtx_n; while (a0 < a1) { uint32_t m = (a0 + a1) / 2; if (g_rtx_seq[m] <= P) a0 = m + 1; else a1 = m; } if (a0 > 0) prt = g_rtx_seq[a0 - 1]; }
        int ok = 0, same = 0; uint32_t walk = 0;
        for (uint32_t j = lo; j < end; j++) { const sdx_ev *e = &g_sdx[j];
            walk += 1u + e->rep; if (walk > 64) return 0;   // Too many events (heavily sampled texture): not a candidate; conservative and avoids long walks (merged same-pass as-attachment events still count steps as before)
            if (g_hoist_n && hoist_skipped(e->op)) continue;
            if (e->kind == SDX_READ || e->kind == SDX_NEED) continue;   // Read events carry no level (reading another level is not a conflict); handed to the full lookahead to check by level
            if (e->plane != 0 || e->lv != lv || e->sl != sl) continue;
            if (e->kind == SDX_CLEAR) return 0;
            if (e->rtseq == SDX_NONE || e->rtseq >= g_sdx_nop) return 0;
            const uint8_t *rop = g_sdx_op[e->rtseq]; const p_set_rt *r = (const p_set_rt *)(rop + 8);
            if (rt_same(r, &R.rt)) { if (e->rtseq != prt) same = 1; continue; }   // Same target: one immediately following is a continuation segment of P, one separated by other render passes is Q; the pre-screen does not distinguish them, and if there is still no other candidate at the end it is handed to the full lookahead
            for (int i = 0; i < R.rt.n; i++) { if (!R.rt.color[i]) continue; int f = 0;
                for (int q = 0; q < r->n && !f; q++) if (r->color[q] && sub_eq(&R.rt, i, r, q)) f = 1;
                if (!f) return 0; }
            ok = 1; break; }
        if (!ok && !same) return 0; }
    return 1;
}
static void defer_try(const p_draw *dw, const p_draw_idx *di) {
    if (!defer_on() || defer_has_in_p() || g_defer_replaying || g_fz.active || g_fz_checking || !g_exec_next || !R.rt.n) return;
    int slot = defer_slot_free(); if (slot < 0) return;
    g_defer_cur = &g_defer_slots[slot];
    int ncol = 0; for (int i = 0; i < R.rt.n; i++) ncol += R.rt.color[i] != NULL; if (!ncol) return;   // Render passes with only depth-stencil attached (shadow maps etc.): Q is defined as covering P's colors, so a P without color can never be deferred -- previously the pre-screen let it through idly and the full lookahead scanned dozens of draws for nothing
    int has_clear = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) has_clear |= R.clear_c[i]; if (!has_clear) return;
    if (!defer_pre()) return;
    memset(&g_defer, 0, sizeof g_defer); g_defer.p_rt = R.rt; memcpy(g_defer.clear_c, R.clear_c, 8); memcpy(g_defer.cc, R.cc, sizeof R.cc);
    g_defer.clear_d = R.clear_d; g_defer.clear_s = R.clear_s; g_defer.cd = R.cd; g_defer.cs = R.cs;
    uint64_t tex[6][2 * NVN_SEPTEX_SLOT0], img[6][32]; memcpy(tex, S.tex, sizeof tex); memcpy(img, S.img, sizeof img);
    program_stage_binding pg_stage[6]; memcpy(pg_stage, S.program_stage, sizeof pg_stage);
    const prog_t *pg = S.prog; ds_st ds = S.ds;
    const uint8_t *p = g_exec_next, *p_end = NULL, *qop = NULL;
    const p_set_rt *q = NULL; p_set_rt scan_rt = R.rt;
    int srt_hit = -1, ci = 0, draws = 1, p_has_draw = 1;
    defer_collect_reads(&g_defer, pg, tex, img);
    for (int guard = 0; guard < 200000; guard++) {
        op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (g_hoist_n && hoist_skipped(p)) { p += h.size; continue; }
        switch (h.op) {
            case OP_END: { const uint8_t *nx = defer_cont(&ci); if (!nx) { g_defer.reason = 1; goto fail; } p = nx; continue; }
            case OP_CALL: g_defer.reason = 2; goto fail;
            case OP_JUMP: memcpy(&p, a, 8); continue;
            case OP_WAIT_SYNC: g_defer.reason = 3; g_defer.reason_op = h.op; goto fail;
            case OP_BIND_PROGRAM:
                program_binding_apply(pg_stage, (const p_bind_program *)a);
                pg = program_binding_compose(pg_stage);
                break;
            case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_IMG: { p_bind_handles t1; const p_bind_handles *b = h.op == OP_BIND_IMG ? a : bind_rec(h.op, a, &t1); bh_fresh(b); int lim = h.op == OP_BIND_IMG ? 32 : 2 * NVN_SEPTEX_SLOT0;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (h.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } break; }
            case OP_BIND_STATE: { const p_bind_state *b = a; if (b->kind == 5) memcpy(&ds, b->bytes, sizeof ds); break; }
            case OP_DRAW: case OP_DRAW_IDX:
                if (defer_prog_hits(pg, tex, img, &g_defer.p_rt, 1, 1)) { /* Suggestion: here, after identifying that an output mapping is needed but _bf is missing, call bf_want_note. */ g_defer.reason = 7; goto fail; }
                if (p_end) {   // Draws in the intermediate range: must not write textures P reads, must not write P's depth-stencil (and must not test stencil when P writes stencil)
                    if (srt_hit < 0) { srt_hit = 0; for (int k = 0; k < scan_rt.n && !srt_hit; k++) if (scan_rt.color[k] && defer_rt_hits_reads(&scan_rt, k)) srt_hit = 1; }   // Attachments change only at SetRenderTargets: computed once per segment
                    if (srt_hit) { g_defer.reason = 12; goto fail; }
                    if (scan_rt.depth && g_defer.p_rt.depth && scan_rt.depth->serial == g_defer.p_rt.depth->serial && (ds.dw || (g_defer.p_st && ds.st))) { g_defer.reason = 13; goto fail; }
                    break; }
                draws++; if (draws > 64) { g_defer.reason = 20; goto fail; } p_has_draw = 1; defer_collect_reads(&g_defer, pg, tex, img); g_defer.p_dw |= ds.dw; g_defer.p_st |= ds.st && ((S.st_wmask[0] | S.st_wmask[1]) != 0) && (ds.fail_f != NVN_STENCIL_OP_KEEP || ds.zfail_f != NVN_STENCIL_OP_KEEP || ds.zpass_f != NVN_STENCIL_OP_KEEP || ds.fail_b != NVN_STENCIL_OP_KEEP || ds.zfail_b != NVN_STENCIL_OP_KEEP || ds.zpass_b != NVN_STENCIL_OP_KEEP); if (g_defer.p_dw) { g_defer.reason = 8; goto fail; } break;
            case OP_RESET_COUNTER: g_defer.reset_between = 1; break;
            case OP_COPY_T2T: { const p_copy_t2t *c = a;   // Texture copy: allowed only if neither end touches P's attachments and the destination is not a texture P samples
                uint64_t ss = c->src ? c->src->serial : 0, ds_ = c->dst ? c->dst->serial : 0; int bad = !ss || !ds_;
                for (int k = 0; k < g_defer.p_rt.n && !bad; k++) if (g_defer.p_rt.color[k] && (g_defer.p_rt.color[k]->serial == ss || g_defer.p_rt.color[k]->serial == ds_)) bad = 1;
                if (g_defer.p_rt.depth && (g_defer.p_rt.depth->serial == ss || g_defer.p_rt.depth->serial == ds_)) bad = 1;
                for (int k = 0; k < g_defer.nreads && !bad; k++) if (h_ser(g_defer.p_reads[k]) == ds_) bad = 1;
                if (g_defer.nreads > 512) bad = 1;
                if (bad) { g_defer.reason = 3; g_defer.reason_op = h.op; goto fail; } break; }
            case OP_REPORT_COUNTER: case OP_UPDATE_UB: case OP_COPY_B2B: case OP_CLEAR_BUF: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: g_defer.reason = 3; g_defer.reason_op = h.op; goto fail;
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; if (!p_end) { g_defer.reason = 16; goto fail; }   // A clear inside P's continuation segment: do not defer
                if (c->index >= 0 && c->index < scan_rt.n && (defer_rt_hits_reads(&scan_rt, c->index) || (scan_rt.color[c->index] && defer_rt_hits_reads(&g_defer.p_rt, c->index)))) { g_defer.reason = 5; goto fail; } break; }
            case OP_CLEAR_DS: if (!p_end) { g_defer.reason = 16; goto fail; } break;
            case OP_SET_RT: { const p_set_rt *r = a;
                if (!p_end && rt_same(r, &g_defer.p_rt)) break;   // The same target set once more (the game often sends it repeatedly): still a continuation segment of P, and its draws are deferred together
                if (!p_end) p_end = p; int shared = 0; for (int i = 0; i < g_defer.p_rt.n; i++) for (int j = 0; j < r->n; j++) if (g_defer.p_rt.color[i] && r->color[j] && sub_eq(&g_defer.p_rt, i, r, j)) shared = 1;
                if (shared) {   // An empty render pass (targets set, then changed again without drawing or clearing) touches no attachment: it is neither taken as Q nor counted as a conflict
                    const uint8_t *e = p + h.size; int ec = ci, empty = 0;
                    for (int g2 = 0; g2 < 200000; g2++) { op_hdr he; memcpy(&he, e, sizeof he);
                        if (he.op == OP_JUMP) { memcpy(&e, e + 8, 8); continue; }
                        if (he.op == OP_END) { const uint8_t *nx = defer_cont(&ec); if (!nx) break; e = nx; continue; }
                        if (he.op == OP_SET_RT) { empty = 1; break; }
                        if (he.op == OP_WAIT_SYNC) { g_defer.reason = 3; g_defer.reason_op = he.op; goto fail; }
                        if (he.op == OP_DRAW || he.op == OP_DRAW_IDX || he.op == OP_CLEAR_COLOR || he.op == OP_CLEAR_DS || he.op == OP_CALL || he.op == OP_DISPATCH || he.op == OP_CLEAR_TEX || he.op == OP_COPY_B2B || he.op == OP_CLEAR_BUF || he.op == OP_COPY_T2T || he.op == OP_COPY_B2T || he.op == OP_COPY_T2B) break;
                        e += he.size; }
                    if (empty) break; }
                if (!shared) break; if (!p_has_draw || !defer_q_superset(r)) { g_defer.reason = 6; goto fail; } q = r; qop = p; goto q_found; }
            default: break;
        }
        if (h.op == OP_SET_RT) { scan_rt = *(const p_set_rt *)a; srt_hit = -1; }
        p += h.size;
    }
fail:
    if (g_defer.reason && passdump()) PD("G %llu not established reason=%d op=%d", g_stat_pass + 1, g_defer.reason, g_defer.reason_op); memset(&g_defer, 0, sizeof g_defer); return;
q_found:
    p = qop; op_hdr qh; memcpy(&qh, p, sizeof qh); p += qh.size; int q_body = 0, q_clears = 0;   // Binding tracking continues from the main scan's state (does not go back to S at the start of P)
    for (int guard = 0; guard < 200000; guard++) { op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8; if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        if (h.op == OP_WAIT_SYNC) { g_defer.reason = 3; g_defer.reason_op = h.op; goto fail; }

        // The whole Q segment (up to the next SetRenderTargets / end of list): no draw may sample P's color attachments through a view covering P's level -- after deferral P's result is still in the tile and has not landed in main memory, so sampling would read old content
        if (h.op == OP_DRAW || h.op == OP_DRAW_IDX) { if (defer_prog_hits(pg, tex, img, &g_defer.p_rt, 1, 1)) { g_defer.reason = 14; goto fail; } q_body = 1; p += h.size; continue; }
        if (q_body && (h.op == OP_SET_RT || h.op == OP_END)) break;
        if (!q_body && h.op == OP_END) { const uint8_t *nx = defer_cont(&ci); if (!nx) { g_defer.reason = 9; goto fail; } p = nx; continue; }
        if (h.op == OP_CALL) { g_defer.reason = 9; g_defer.reason_op = h.op; goto fail; }

        // After Q's SetRenderTargets and before its first draw and any clear, another SetRenderTargets arrives (the game sets targets repeatedly, the previous one being an empty render pass): if the new targets still cover P take it as Q, otherwise do not defer
        if (!q_body && h.op == OP_SET_RT && !q_clears) { const p_set_rt *r2 = a; if (!defer_q_superset(r2)) { g_defer.reason = 15; goto fail; } q = r2; qop = p; p += h.size; continue; }
        if (h.op == OP_CLEAR_COLOR || h.op == OP_CLEAR_DS) q_clears = 1;
        if (!q_body && (h.op == OP_END || h.op == OP_SET_RT || h.op == OP_UPDATE_UB || h.op == OP_COPY_B2B || h.op == OP_CLEAR_BUF || h.op == OP_DISPATCH)) { g_defer.reason = 9; g_defer.reason_op = h.op; goto fail; }
        if (h.op == OP_BIND_PROGRAM) {
            program_binding_apply(pg_stage, (const p_bind_program *)a);
            pg = program_binding_compose(pg_stage);
        } else if (h.op == OP_BIND_TEX || h.op == OP_BIND_IMG) { const p_bind_handles *b = a; int lim = h.op == OP_BIND_IMG ? 32 : 128; if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (h.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } }
        else if (h.op == OP_CLEAR_COLOR) { const p_clear_color *c = a; if (c->index >= 0 && c->index < q->n && q->color[c->index]) for (int i = 0; i < g_defer.p_rt.n; i++) if (g_defer.p_rt.color[i] && sub_eq(q, c->index, &g_defer.p_rt, i)) { g_defer.reason = 10; goto fail; } }
        p += h.size;
    }
    if (!q || !defer_q_superset(q) || g_defer.n >= 64) { g_defer.reason = 11; goto fail; }
    for (int i = 0; i < 4; i++) if (&g_defer_slots[i] != &g_defer && g_defer_slots[i].active && g_defer_slots[i].q_op == qop && !defer_same_q_depth(&g_defer_slots[i], q)) { g_defer.reason = 17; goto fail; }
    g_defer.q_rt = *q; if (!q->depth && g_defer.p_rt.depth) { g_defer.q_rt.depth = g_defer.p_rt.depth; g_defer.q_rt.dview = g_defer.p_rt.dview; g_defer.q_rt.has_dview = g_defer.p_rt.has_dview; } { int na = 0; for (int i = 0; i < 4; i++) { const defer_t *d = &g_defer_slots[i]; if (d != &g_defer && d->active && d->at_q) na += d->n; }
      if (na + 1 > 64) { g_defer.reason = 20; goto fail; } }
    g_defer.q_op = qop; g_defer.p_end_op = p_end; g_defer.in_p = 1; g_defer.active = 1; g_defer_nact++; g_defer_pend = p_end; g_defer_n++; g_defer.born = g_defer_n;
    // Chained: this render pass is exactly the Q of upstream groups (they are at_q, waiting to be replayed when this render pass opens), and this render pass is itself deferred as a whole -- the upstream groups are merged into the new group in creation order:
    // the target space of an upstream group = this render pass's targets = the new group's P targets, and the snapshot's mapping is composed as pm' = upstream lmap o pm; the upstream clears are already in R and are carried along with the new group below
    for (int gi, after = 0; (gi = defer_next(1, after)) >= 0;) { defer_t *d = &g_defer_slots[gi]; after = d->born; if (d == &g_defer) continue;
        for (int k = 0; k < 8; k++) if (d->lmap[k] != 0xff && d->lmap[k] != k) { g_defer.reason = 18; goto fail; }   // Only merge in upstream groups with an identity mapping (see defer_q_superset)
        for (int i = 0; i < d->n; i++) { defer_snap z = d->snap[i]; uint8_t pm[8]; for (int k = 0; k < 8; k++) pm[k] = d->lmap[k] == 0xff || d->lmap[k] >= 8 ? 0xff : z.pm[d->lmap[k]];
            memcpy(z.pm, pm, 8); g_defer.snap[g_defer.n++] = z; }
        g_defer.reset_between |= d->reset_between; d->n = 0; defer_release_one(d); } if (!defer_take(dw, di)) { defer_release_one(&g_defer); return; }
    memset(R.clear_c, 0, sizeof R.clear_c); R.clear_d = R.clear_s = 0; PD("G %llu established group %d P→Q draws=%d", g_stat_pass + 1, slot, draws);
}
// Pre-screen of the necessary condition for merging (binary search on the submit index, zero scanning on failure): merging requires that "the first draw of the later pass (the next SET_RT with a different target that has not been hoisted)
// samples some color attachment of the earlier pass", i.e. there is an SDX_READ event for that attachment at that draw's sequence number. If not satisfied there is certainly no merge, saving the per-operation scan below;
// only if satisfied does it go through the original planning (the decision is exactly the same as without the pre-screen). When the index is unavailable, plan as before.
static int fz_prefilter(void) {
    if (!g_sdx_ok || !g_exec_seq || g_sdx_op[g_exec_seq - 1] != g_exec_cur) return 1;
    uint32_t P = g_exec_seq - 1, lo = 0, hi = g_rtx_n;
    while (lo < hi) { uint32_t m = (lo + hi) / 2; if (g_rtx_seq[m] <= P) lo = m + 1; else hi = m; }
    for (; lo < g_rtx_n; lo++) { const uint8_t *op = g_sdx_op[g_rtx_seq[lo]]; if (g_hoist_n && hoist_skipped(op)) continue; if (!rt_same(&R.rt, (const p_set_rt *)(op + 8))) break; }
    if (lo >= g_rtx_n) return 0;
    uint32_t r2 = g_rtx_seq[lo], a = 0, b = g_drx_n;
    while (a < b) { uint32_t m = (a + b) / 2; if (g_drx_seq[m] <= r2) a = m + 1; else b = m; }
    if (a >= g_drx_n) return 0;
    uint32_t fd = g_drx_seq[a];
    for (uint32_t k = lo + 1; k < g_rtx_n && g_rtx_seq[k] < fd; k++) if (!(g_hoist_n && hoist_skipped(g_sdx_op[g_rtx_seq[k]]))) return 0;   // The target changes again before the first draw: step two of the original planning gives up as well
    for (int j = 0; j < R.rt.n && j < 8; j++) { if (!R.rt.color[j]) continue; int bk = sdx_bucket(R.rt.color[j]->serial, 0); if (bk < 0) continue;
        uint32_t e0 = g_sdx_bst[bk], e1 = g_sdx_bst[bk + 1];
        while (e0 < e1) { uint32_t m = (e0 + e1) / 2; if (g_sdx[m].seq < fd) e0 = m + 1; else e1 = m; }
        for (; e0 < g_sdx_bst[bk + 1] && g_sdx[e0].seq == fd; e0++) if (g_sdx[e0].kind == SDX_READ) return 1; }
    return 0;
}
static void fuse_try(void) {
    if (g_fz.active || !g_exec_next || g_hoist_n + 12 > HOIST_MAX || !R.rt.n) return;
    if (!fz_prefilter()) return;

    // One: from the earlier pass to the next SetRenderTargets, only draws and state are allowed
    const uint8_t *p = NULL, *r2op = NULL, *reuse[8]; int nreuse = 0; const p_set_rt *r2 = NULL; p_scissor sc = R.sc; int has_sc = R.has_sc;
    fz_cursor cur = {.p = g_exec_next}; op_hdr h; const void *a;
    for (int guard = 0; guard < 200000 && !r2; guard++) {
        if (!fz_next(&cur, &p, &h, &a)) return;
        switch (h.op) {
            case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH:
            case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_UPDATE_UB: case OP_COPY_B2B: case OP_CLEAR_BUF: return;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_SET_RT: {
                int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                if (rt_same(&R.rt, a)) {
                    if (g_enc_mode == 1 || pend || !R.has_rt || !tbdr_on() || nreuse >= 8) return;
                    reuse[nreuse++] = p; continue;
                }
                r2 = a; r2op = p; continue;
            }
            default: break;
        }
    }
    if (!r2 || rt_same(&R.rt, r2)) return;

    // Two: from the start of the later pass to its first draw, only clears (not clearing shared attachments) and state are allowed
    const uint8_t *cl[8]; int ncl = 0; uint8_t clr[8] = {0}; MTLClearColor cc[8]; int cd_f = 0, cs_f = 0; double cdv = 0; uint32_t csv = 0;
    if (!fz_next(&cur, &p, &h, &a)) return;   // Continue from the cursor of step one (already past the later pass's SET_RT, keeping the cross-handle / return stack position)
    for (int guard = 0; guard < 200000; guard++) {
        if (h.op == OP_DRAW || h.op == OP_DRAW_IDX) break;
        switch (h.op) {
            case OP_COPY_T2T: case OP_COPY_B2T: case OP_COPY_T2B: case OP_CLEAR_TEX: case OP_DISPATCH: case OP_SET_RT: case OP_UPDATE_UB: case OP_COPY_B2B: case OP_CLEAR_BUF: return;
            case OP_SCISSOR: sc = *(const p_scissor *)a; has_sc = 1; break;
            case OP_CLEAR_COLOR: { const p_clear_color *c = a; if (ncl >= 8 || c->index < 0 || c->index >= r2->n || !r2->color[c->index] || c->mask != 0xf) return;
                const tex_t *t = r2->color[c->index];
                if (has_sc && !(sc.x <= 0 && sc.y <= 0 && sc.x + sc.w >= (int)t->d.width && sc.y + sc.h >= (int)t->d.height)) return;
                clr[c->index] = 1; cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]); cl[ncl++] = p; break; }
            case OP_CLEAR_DS: { const p_clear_ds *c = a; if (ncl >= 8 || !r2->depth) return;
                if (c->depth_mask) { cd_f = 1; cdv = c->depth; } if (c->stencil_mask) { cs_f = 1; csv = (uint32_t)c->stencil; } cl[ncl++] = p; break; }
            default: break;
        }
        if (!fz_next(&cur, &p, &h, &a)) return;
    }

    // Three: union of attachments (the earlier pass's attachments keep their original indices, those unique to the later pass are appended), consistent sizes, compatible depth
    p_set_rt U = R.rt; int n = R.rt.n; int8_t p2map[8]; memset(p2map, -1, sizeof p2map);
    for (int k = 0; k < r2->n && k < 8; k++) { if (!r2->color[k]) continue;
        int j = -1; for (int m = 0; m < R.rt.n; m++) if (R.rt.color[m] && sub_eq(&R.rt, m, r2, k)) { j = m; break; }
        if (j < 0) { if (n >= 8) return; j = n++; U.color[j] = r2->color[k]; U.cview[j] = r2->cview[k]; U.has_cview[j] = r2->has_cview[k]; }
        else if (clr[k]) return;   // The start of the later pass wants to clear an attachment the earlier pass is drawing to: after merging this clear has nowhere to go
        p2map[k] = (int8_t)j; }
    U.n = n;
    { uint64_t s0; int l0, x0; rt_sub(&U, 0, &s0, &l0, &x0); if (!U.color[0]) return;
      uint32_t w0 = MAX(U.color[0]->d.width >> l0, 1u), h0 = MAX(U.color[0]->d.height >> l0, 1u);
      for (int j = 1; j < n; j++) { uint64_t s_; int l_, x_; rt_sub(&U, j, &s_, &l_, &x_); if (!U.color[j] || MAX(U.color[j]->d.width >> l_, 1u) != w0 || MAX(U.color[j]->d.height >> l_, 1u) != h0) return; } }
    int depth_from_r2 = 0;
    if (R.rt.depth && r2->depth && !sub_eq(&R.rt, 8, r2, 8)) return;
    if (!R.rt.depth && r2->depth) { U.depth = r2->depth; U.dview = r2->dview; U.has_dview = r2->has_dview; depth_from_r2 = 1; }
    if ((cd_f || cs_f) && !depth_from_r2) return;   // The depth is already in use in the earlier pass: the depth clear at the start of the later pass has nowhere to go

    // Four: draws of the earlier pass must not sample attachments brought in by the later pass; when the depth comes from the later pass, depth-stencil testing must not be enabled
    static uint64_t tex[6][2 * NVN_SEPTEX_SLOT0], img[6][32];
    memcpy(tex, S.tex, sizeof tex); memcpy(img, S.img, sizeof img);
    program_stage_binding pg_stage[6]; memcpy(pg_stage, S.program_stage, sizeof pg_stage);
    const prog_t *pg = S.prog; ds_st ds = S.ds;
    cur = (fz_cursor){.p = g_exec_next};
    for (;;) {   // The first loop iteration checks the draw that opens the render pass (current state), then checks the draws in the range one by one
        if (depth_from_r2 && (ds.dt || ds.dw || ds.st)) return;
        for (int j = R.rt.n; j < n; j++) if (prog_samples_ser(pg, tex, img, U.color[j]->serial)) return;
        if (depth_from_r2 && prog_samples_ser(pg, tex, img, U.depth->serial)) return;
        for (;;) {
            const uint8_t *q; op_hdr hq; const void *aq;
            if (!fz_next(&cur, &q, &hq, &aq)) return;
            if (q == r2op) goto planned;
            if (hq.op == OP_BIND_PROGRAM) {
                program_binding_apply(pg_stage, (const p_bind_program *)aq);
                pg = program_binding_compose(pg_stage);
            }
            else if (hq.op == OP_BIND_STATE) { const p_bind_state *b = aq; if (b->kind == 5) memcpy(&ds, b->bytes, sizeof ds); }
            else if (hq.op == OP_BIND_TEX || hq.op == OP_BIND_TEX1 || hq.op == OP_BIND_IMG) { p_bind_handles t1; const p_bind_handles *b = hq.op == OP_BIND_IMG ? aq : bind_rec(hq.op, aq, &t1); bh_fresh(b); int lim = hq.op == OP_BIND_IMG ? 32 : 2 * NVN_SEPTEX_SLOT0;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (hq.op == OP_BIND_IMG) img[b->stage][b->first + k] = b->h[k]; else tex[b->stage][b->first + k] = b->h[k]; } }
            else if (hq.op == OP_DRAW || hq.op == OP_DRAW_IDX) break;
        }
    }
planned:
    {   // Five: pre-screen the first draw of the later pass (static conditions), to avoid attaching the union of attachments for nothing: track the program and texture bindings up to that draw
        program_stage_binding pg2_stage[6]; memcpy(pg2_stage, S.program_stage, sizeof pg2_stage);
        const prog_t *pg2 = S.prog; static uint64_t t2[6][128], i2[6][32]; memcpy(t2, S.tex, sizeof t2); memcpy(i2, S.img, sizeof i2);
        cur = (fz_cursor){.p = g_exec_next};
        for (;;) { const uint8_t *q; op_hdr hq; const void *aq; if (!fz_next(&cur, &q, &hq, &aq)) return; if (q == p) break;
            if (hq.op == OP_BIND_PROGRAM) {
                program_binding_apply(pg2_stage, (const p_bind_program *)aq);
                pg2 = program_binding_compose(pg2_stage);
            }
            else if (hq.op == OP_BIND_TEX || hq.op == OP_BIND_TEX1 || hq.op == OP_BIND_IMG) { p_bind_handles t1; const p_bind_handles *b = hq.op == OP_BIND_IMG ? aq : bind_rec(hq.op, aq, &t1); bh_fresh(b); int lim = hq.op == OP_BIND_IMG ? 32 : 2 * NVN_SEPTEX_SLOT0;
                if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= lim) for (int k = 0; k < b->n; k++) { if (hq.op == OP_BIND_IMG) i2[b->stage][b->first + k] = b->h[k]; else t2[b->stage][b->first + k] = b->h[k]; } }
        }
        NvProg *pr2 = pg2 ? prog_get(pg2) : nil; NvShader *vs2 = pr2 ? pr2->sh[0] : nil, *fs2 = pr2 ? pr2->sh[1] : nil;
        if (!vs2 || !fs2) return;
        int touches = 0;
        for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr2->sh[st]; if (!sh || sh->stage >= 6) continue;
            for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *bb = &sh->rf.b[i]; uint64_t h;
                if (bb->kind == NVN_RB_IMG) h = bb->reg < 32 ? i2[sh->stage][bb->reg] : 0; else if (bb->kind == NVN_RB_TEX || bb->kind == NVN_RB_TBUF) h = bb->reg < 128 ? t2[sh->stage][bb->reg] : 0; else continue;
                uint64_t ser = h ? h_ser(h) : 0; if (!ser) continue;
                int j = -1; for (int m = 0; m < R.rt.n; m++) if (R.rt.color[m] && R.rt.color[m]->serial == ser) { j = m; break; }
                if (j < 0) continue;
                if (sh != fs2 || bb->kind != NVN_RB_TEX || (fs2->rf.fc_mixed[bb->reg >> 6] >> (bb->reg & 63) & 1)) return;
                int any = 0;
                for (int c = 0; c < fs2->rf.nfc; c++) if (fs2->rf.fc_reg[c] == bb->reg) { int l = fs2->rf.fc_locn[c];
                    if (l >= 32 || (vs2->rf.uv_attr[l] < 0 && vs2->rf.uv_aff_attr[l] < 0)) return; any = 1; }
                if (!any) return;
                touches = 1; } }
        if (!touches) return;   // The later pass does not read the earlier pass's attachments: merging has no benefit
        int ident = 1; for (int c = 0; c < 8; c++) if (fs2->rf.color_mask >> c & 1) { if (p2map[c] < 0) return; if (p2map[c] != c) ident = 0; }
        if (!ident && !fs2->rf.nfo) return;
    }
    memset(&g_fz, 0, sizeof g_fz);
    g_fz.active = 1; g_fz.phase = 1; g_fz.n1 = R.rt.n; g_fz.nreuse = nreuse; g_fz.rt1 = R.rt; g_fz.rt2 = *r2; g_fz.rt2_op = r2op; memcpy(g_fz.reuse, reuse, sizeof reuse); memcpy(g_fz.p2map, p2map, 8);
    for (int j = 0; j < 8; j++) g_fz.lmap1[j] = j < R.rt.n ? (uint8_t)j : 0xff;
    fz_probe_cov2(r2, p, p2map, clr, g_fz.n1, n);
    g_fz.skip[g_fz.nskip++] = r2op; for (int i = 0; i < ncl; i++) g_fz.skip[g_fz.nskip++] = cl[i];
    for (int i = 0; i < g_fz.nskip; i++) g_hoist_skip[g_hoist_n++] = g_fz.skip[i];
    R.rt = U;
    memcpy(g_fz.clr2, clr, 8); memcpy(g_fz.cc2, cc, sizeof cc); g_fz.cd2f = (uint8_t)cd_f; g_fz.cs2f = (uint8_t)cs_f; g_fz.cd2 = cdv; g_fz.cs2 = csv;
    for (int k = 0; k < 8; k++) if (clr[k]) { R.clear_c[p2map[k]] = 1; R.cc[p2map[k]] = cc[k]; }
    if (cd_f) { R.clear_d = 1; R.cd = cdv; } if (cs_f) { R.clear_s = 1; R.cs = csv; }
    if (!g_fz_n++) MLOG("Readable-tile channel merging: first case channel %llu (%d physical attachments)", g_stat_pass + 1, n);
    PD("F %llu merge next channel (readable tile): %d physical attachments, previous channel %d", g_stat_pass + 1, n, R.rt.n == n ? g_fz.n1 : g_fz.n1);
}
static id<MTLFunction> bf_fn(NvShader *fs, const int *fx, uint8_t fbmask, const uint8_t *lmap) {   // The readable tile version is specialized by constants (cached)
    os_unfair_lock_lock(&g_fz_fn_lock);
    if (!g_bf_fn_cache) { g_bf_fn_cache = [NSMutableDictionary dictionary]; g_bf_lib_cache = [NSMutableDictionary dictionary]; }
    size_t kl = strlen(fs->key); NSString *base = [NSString stringWithFormat:@"%.*s", (int)(kl > 2 ? kl - 2 : kl), fs->key];
    id<MTLLibrary> lib = g_bf_lib_cache[base];
    if (!lib) {
        NSError *e = nil; NSString *path = [NSString stringWithFormat:@"%@/%@_bf.metallib", g_msl_dir, base];
        lib = lib_from_file(path, &e);
        if (!lib) MLOG("[RED] Failed to load readable-tile variant %s: %s", path.UTF8String, e.localizedDescription.UTF8String);
        else g_bf_lib_cache[base] = lib;
    }
    if (!lib) { os_unfair_lock_unlock(&g_fz_fn_lock); return nil; }
    NSMutableString *k = [NSMutableString stringWithFormat:@"%02x", fbmask];
    for (int i = 0; i < fs->rf.nfc; i++) [k appendFormat:@",%d", fx[i]];
    for (int j = 0; j < 8; j++) [k appendFormat:@"/%02x", lmap[j]];
    NSString *cacheKey = [NSString stringWithFormat:@"%@|%@", base, k];
    id<MTLFunction> f = g_bf_fn_cache[cacheKey];
    if (f) { os_unfair_lock_unlock(&g_fz_fn_lock); return f; }
    MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
    for (int i = 0; i < fs->rf.nfc; i++) { int v = fx[i]; [cv setConstantValue:&v type:MTLDataTypeInt atIndex:(NSUInteger)i]; }
    for (int q = 0; q < 8; q++) { bool b = fbmask >> q & 1; [cv setConstantValue:&b type:MTLDataTypeBool atIndex:(NSUInteger)(32 + q)]; }
    for (int j = 0; j < fs->rf.nfo; j++) for (int q = 0; q < 8; q++) { bool b = lmap[q] == fs->rf.fo_color[j]; [cv setConstantValue:&b type:MTLDataTypeBool atIndex:(NSUInteger)(64 + 8 * j + q)]; }
    NSError *e = nil; f = [lib newFunctionWithName:@"main0" constantValues:cv error:&e];
    if (!f) { MLOG("[RED] Failed to specialize readable-tile variant %s (%s): %s", fs->key, k.UTF8String, e.localizedDescription.UTF8String); os_unfair_lock_unlock(&g_fz_fn_lock); return nil; }
    g_bf_fn_cache[cacheKey] = f; os_unfair_lock_unlock(&g_fz_fn_lock); return f;
}
static int ubo_float(int slot, uint16_t ref, float *out) {   // Component cmp of vector vec of the uniform slot (read on the CPU side; not read if it falls within a buffer write pending execution in this submit)
    int sl = ref >> 8, vec = ref >> 2 & 63, cmp = ref & 3;
    if (slot < 0 || slot >= 6 || sl < 0 || sl >= 16) return 0;
    const p_bind_buf *u = &S.ubo[slot][sl]; uint64_t off = (uint64_t)vec * 16 + (uint64_t)cmp * 4;
    if (!u->addr || off + 4 > u->size || pend_hits(u->addr + off, 4)) return 0;
    void *mb; uint64_t mo; const uint8_t *base;
    if (!mtl_addr(u->addr + off, 4, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) return 0;
    memcpy(out, base + mo, 4); return 1;
}
static int zskip_fact_for(const char *key, zskip_fact *out) {
    if (!key) return 0;
    for (int i = 0; i < g_zskip_nf; i++) if (!strncmp(key, g_zskip_facts[i].key, strlen(g_zskip_facts[i].key))) { *out = g_zskip_facts[i]; return 1; }   // Runtime keys carry the _b suffix
    return 0;
}
static int zskip_view_zero(int tid) {
    if (!tid || tid >= g_ntreg || !g_treg[tid].ok) return 0;
    treg_t *r = &g_treg[tid]; const tex_t *t = &r->t; int lv0 = r->has_v && r->v.set_levels ? r->v.base_level : 0;
    int nl = r->has_v && r->v.set_levels && r->v.nlevels > 0 ? r->v.nlevels : 1;
    int sl0 = r->has_v && r->v.set_layers ? r->v.min_layer : 0;
    int ns = r->has_v && r->v.set_layers && r->v.nlayers > 0 ? r->v.nlayers : 1;
    for (int l = 0; l < nl; l++) for (int s = 0; s < ns; s++) if (!zskip_has(t->serial, lv0 + l, sl0 + s)) return 0;
    return 1;
}
// When counting occlusion queries, a skipped draw would otherwise contribute to the passing sample count: it matters only if there is no report before the next reset.

// While an occlusion query is counting, a skipped draw should have contributed passed samples: it only does not matter if "there is no report before the next reset".
// Looking forward in replay order from after this draw: sample count reset encountered first -> does not matter (return 1); sample count report encountered first, or the end cannot be seen (submit ends) -> matters (return 0)
static int zskip_vis_dead(void) {
    if (!g_vis_counting) return 1;
    const uint8_t *p = g_exec_next; int ci = 0; if (!p) return 0;
    for (int guard = 0; guard < 200000; guard++) { op_hdr h; memcpy(&h, p, sizeof h); const void *a = p + 8;
        if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        if (h.op == OP_END) { p = defer_cont(&ci); if (!p) return 0; continue; }
        if (h.op == OP_CALL) return 0;
        if (h.op == OP_RESET_COUNTER || h.op == OP_REPORT_COUNTER) { uint64_t t; memcpy(&t, a, 8); if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) return h.op == OP_RESET_COUNTER; }
        p += h.size; }
    return 0;
}
static int zskip_can_draw(NvProg *pr) {
    // Skip conditions (all must hold): the fragment key is in the fact table; the views bound to every texture slot declared by its reflection cover only known-zero subresources; coefficients are finite;
    // no depth writes or stencil operations; no occlusion query is being counted; every color attachment to be written is already 0 (a candidate cleared to 0 across the whole channel, or known zero)—writing 0 over 0.

    // Skip conditions (all must hold): the fragment key is in the fact table, the view bound to every texture slot declared by its reflection covers only subresources known to be all zero, the coefficient is finite,
    // no depth write and no stencil, not counting an occlusion query, and every color attachment to be written is already 0 (a candidate cleared entirely to 0 in this render pass, or known all zero) -- writing 0 over 0.
    // Reasons for not holding (the N line of PASSDUMP): 1 depth / stencil 2 occlusion query counting with a report before the reset 3 coefficient unreadable or non-finite 4 image slot 5 sampled view not known all zero 6 no target 7 target is not 0
    if (!zskip_on() || !pr) return 0;
    NvShader *fs = nil; for (int st = 0; st < 5; st++) if (pr->sh[st] && pr->sh[st]->stage == 1) fs = pr->sh[st];
    zskip_fact f; float coeff; int why = 0;
    if (!fs) return 0;
    if (g_la_cache) { if (!fs->zf_state || fs->zf_nf != g_zskip_nf) { fs->zf_state = zskip_fact_for(fs->key, &fs->zf) ? 2 : 1; fs->zf_nf = (int8_t)g_zskip_nf; } if (fs->zf_state == 1) return 0; f = fs->zf; }   // Key unchanged; one lookup is enough
    else if (!zskip_fact_for(fs->key, &f)) return 0;
    if (S.ds.dw || S.ds.st) why = 1;
    else if (!zskip_vis_dead()) why = 2;
    else if (f.ubo_slot >= 0 && (!ubo_float(1, (uint16_t)((f.ubo_slot << 8) | (f.vec << 2) | f.comp), &coeff) || !isfinite(coeff))) why = 3;   // ubo_slot < 0: this shader has no coefficient
    for (int i = 0; !why && i < fs->rf.nb; i++) { nvn_rbind *b = &fs->rf.b[i];
        if (b->kind == NVN_RB_IMG) { why = 4; break; }
        if (b->kind != NVN_RB_TEX && b->kind != NVN_RB_TBUF) continue;
        int tid = b->reg < 128 ? (int)(S.tex[1][b->reg] & 0xfffff) : 0; if (!zskip_view_zero(tid)) why = 5; }
    if (!why && (!R.has_rt || !R.rt.n)) why = 6;
    for (int i = 0; !why && i < R.rt.n && i < 8; i++) { if (!R.rt.color[i]) continue;
        int pending = g_enc ? g_zskip_cand_ok[i] : zskip_zero_clear(i);   // Render pass already open: look at the candidates noted when the render pass was opened (not yet drawn to); not open: look at the pending clears
        if (!pending && (g_enc || R.clear_c[i] || !zskip_target_zero(R.rt.color[i], &R.rt.cview[i], R.rt.has_cview[i]))) why = 7; }
    if (why) PD("N %llu zero source not skipped reason=%d set %d", g_stat_pass + 1, why, g_zskip_n);
    return !why;
}
static int uv_proof(NvShader *vs, int locn, const p_draw *dw, const p_draw_idx *di) {   // The interpolated output locn at every pixel center is exactly this pixel's texel center: u = (x + 1) / 2, v = (1 - y) / 2, w = 1, viewport and scissor cover the whole block
    if (locn < 0 || locn >= 32 || vs->rf.pos_attr < 0) return 0;
    int direct = vs->rf.uv_attr[locn] >= 0, aff = vs->rf.uv_aff_attr[locn] >= 0;
    if (!direct && !aff) return 0;
    float sc[4] = {1, 0, 1, 0};   // Pass-through as-is = scale 1, offset 0
    if (!direct) for (int q = 0; q < 4; q++) if (!ubo_float(0, vs->rf.uv_aff[locn][q], &sc[q])) return 0;
    NSUInteger W = g_pass.tw, H = g_pass.th;
    if (S.has_vp && !(S.vp[0][0] == 0 && S.vp[0][1] == 0 && S.vp[0][2] == (float)W && S.vp[0][3] == (float)H)) return 0;
    if (R.has_sc && !(R.sc.x <= 0 && R.sc.y <= 0 && R.sc.x + R.sc.w >= (int)W && R.sc.y + R.sc.h >= (int)H)) return 0;
    float P[64][4], Q[64][4]; int n = dw ? dw->count : di->count;
    if (attr_read(vs->rf.pos_attr, dw, di, P) || attr_read(direct ? vs->rf.uv_attr[locn] : vs->rf.uv_aff_attr[locn], dw, di, Q)) return 0;
    for (int k = 0; k < n; k++) { float w = vs->rf.pos_w1 ? 1.0f : P[k][3];
        float u = direct ? Q[k][0] : fmaf(Q[k][0], sc[0], sc[1]), v = direct ? Q[k][1] : fmaf(Q[k][1], sc[2], sc[3]);   // The same fma as the shader
        if (w != 1.0f || u != (P[k][0] + 1.0f) * 0.5f || v != (1.0f - P[k][1]) * 0.5f) { PD("F %llu UV proof: vertex %d position %g,%g UV %g,%g (scale / offset %g,%g,%g,%g)", g_stat_pass, k, P[k][0], P[k][1], u, v, sc[0], sc[1], sc[2], sc[3]); return 0; } }
    return 1;
}

// One draw of the later pass: verify and produce the specialization. Returns 0 = cannot be proven (the caller ends the merged render pass and reopens with the original targets)
static int fz_p2_draw(NvProg *pr, const p_draw *dw, const p_draw_idx *di, uint8_t lmap[8], __strong id<MTLFunction> *fn) {
    NvShader *vs = pr->sh[0], *fs = pr->sh[1]; *fn = nil;
    if (!vs || !fs) { PD("F %llu cannot prove: no vertex / fragment shader", g_stat_pass); return 0; }
    if (!g_fz.rt2.depth && R.rt.depth && (S.ds.dt || S.ds.dw || S.ds.st)) { PD("F %llu cannot prove: depth comes from the previous channel and this draw has depth-stencil testing enabled", g_stat_pass); return 0; }
    int fx[32]; for (int i = 0; i < 32; i++) fx[i] = -1; uint8_t fbmask = 0;
    for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (!sh || sh->stage >= 6) continue;
        for (int i = 0; i < sh->rf.nb; i++) { const nvn_rbind *b = &sh->rf.b[i]; uint64_t h;
            if (b->kind == NVN_RB_IMG) h = b->reg < 32 ? S.img[sh->stage][b->reg] : 0; else if (b->kind == NVN_RB_TEX || b->kind == NVN_RB_TBUF) h = b->reg < 128 ? S.tex[sh->stage][b->reg] : 0; else continue;
            uint64_t ser = h ? h_ser(h) : 0; if (!ser) continue;
            int j = -1; for (int m = 0; m < g_fz.n1; m++) if (R.rt.color[m] && R.rt.color[m]->serial == ser) { j = m; break; }
            if (j < 0) continue;   // What is sampled is not an attachment of the earlier pass
            if (sh != fs || b->kind != NVN_RB_TEX) { PD("F %llu cannot prove: non-fragment stage or non-texture slot reads an attachment from the previous channel", g_stat_pass); return 0; }
            if (fs->rf.fc_mixed[b->reg >> 6] >> (b->reg & 63) & 1) { PD("F %llu cannot prove: slot %d is mixed", g_stat_pass, b->reg); return 0; }
            uint64_t s_; int lv, sl; rt_sub(&R.rt, j, &s_, &lv, &sl);
            if (lv || sl || MAX(R.rt.color[j]->d.width, 1u) != g_pass.tw || MAX(R.rt.color[j]->d.height, 1u) != g_pass.th) { PD("F %llu cannot prove: attachment is not mip level 0, layer 0, or dimensions do not match", g_stat_pass); return 0; }
            {   // The sampled view must also be exactly the attachment itself (previously only the attachment being level 0 layer 0 was checked, not the view bound by the shader -- when the same texture is sampled through a view of level k / layer k / a changed format,

                // the proof would still hold, the sampling would be changed to read on-tile level 0 / the original format, and the picture would be wrong). The view's starting level / layer must be 0 and its format must equal the attachment's actual format, otherwise do not merge
                int tid_ = (int)(h & 0xfffff); void *tv_ = NULL; if (tid_) tex_fast_rid(tid_, &tv_); __unsafe_unretained id<MTLTexture> tv = (__bridge id<MTLTexture>)tv_;
                uint32_t apf_ = tv.parentTexture ? (uint32_t)tv.parentTexture.pixelFormat : (uint32_t)tv.pixelFormat; uint8_t swz_[4];
                if (R.rt.has_cview[j] && R.rt.cview[j].set_format && !(nvn_fmt_to_mtl(R.rt.cview[j].format, &apf_, swz_) && apf_)) apf_ = 0;   // The attachment is attached through a format-changing view: take that format; if unrecognized, do not merge
                MTLPixelFormat apf = (MTLPixelFormat)apf_;
                if (!tv || (tv.parentTexture && (tv.parentRelativeLevel || tv.parentRelativeSlice)) || tv.pixelFormat != apf) {
                    static int lg; if (lg++ < 8) LOG("[nvn-mtl] on-chip read merge: slot %d sampled view (base level %lu layer %lu format %lu) does not match attachment (level 0 layer 0 format %lu); not merging (frame %llu channel %llu)", b->reg,
                        tv && tv.parentTexture ? (unsigned long)tv.parentRelativeLevel : 0ul, tv && tv.parentTexture ? (unsigned long)tv.parentRelativeSlice : 0ul, (unsigned long)tv.pixelFormat, (unsigned long)apf,
                        (unsigned long long)atomic_load(&g_cur_frame), g_stat_pass);
                    PD("F %llu cannot prove: slot %d sampled view is not the same image as the attachment (level / layer / format mismatch)", g_stat_pass, b->reg); return 0; } }
            int any = 0;
            for (int c = 0; c < fs->rf.nfc; c++) if (fs->rf.fc_reg[c] == b->reg) {
                if (!uv_proof(vs, fs->rf.fc_locn[c], dw, di)) { PD("F %llu cannot prove: slot %d sampling coordinates are not this pixel (interpolated output %d)", g_stat_pass, b->reg, fs->rf.fc_locn[c]); return 0; }
                fx[c] = j; fbmask |= (uint8_t)(1u << j); any = 1; }
            if (!any) { PD("F %llu cannot prove: sampling from slot %d is not in the editable table", g_stat_pass, b->reg); return 0; } } }
    memset(lmap, 0xff, 8);
    for (int c = 0; c < 8; c++) if (fs->rf.color_mask >> c & 1) { if (g_fz.p2map[c] < 0) { PD("F %llu cannot prove: no corresponding attachment for output color %d", g_stat_pass, c); return 0; } lmap[g_fz.p2map[c]] = (uint8_t)c; }
    int ident = !fbmask; for (int c = 0; c < 8; c++) if ((fs->rf.color_mask >> c & 1) && g_fz.p2map[c] != c) ident = 0;
    if (ident) { for (int q = 0; q < 8; q++) if (lmap[q] != 0xff) g_fz.wrote[q] = 1; if (fz_check_on()) fz_snap_take(dw, di); return 1; }   // Does not read the tile and the output is not shifted: use the original version (the write mask still follows the mapping)
    if (!fs->rf.nfo) { bf_want_note(fs->key); PD("F %llu cannot prove: no readable tile version (output must be shifted)", g_stat_pass); return 0; }
    *fn = bf_fn(fs, fx, fbmask, lmap);
    if (!*fn) return 0;
    memset(&g_fz_desc, 0, sizeof g_fz_desc); g_fz_desc.kind = 1; g_fz_desc.fbmask = fbmask; for (int i = 0; i < 32; i++) g_fz_desc.fx[i] = (int8_t)fx[i];
    for (int q = 0; q < 8; q++) if (lmap[q] != 0xff) g_fz.wrote[q] = 1;
    if (fz_check_on()) fz_snap_take(dw, di);
    g_fz_draws++; return 1;
}
static void bloomq_reset(void) { memset(&g_bloomq, 0, sizeof g_bloomq); }
static int bloomq_enabled(void) {
    if (g_bloom_q < 0) { const char *e = getenv("DM_NVN_BLOOM_Q"); g_bloom_q = !e || strcmp(e, "0"); }
    return g_bloom_q;
}
static int bloomq_target(uint64_t *ser, int *lv, int *sl) {
    if (!R.has_rt || R.rt.n != 1 || !R.rt.color[0]) return 0;
    rt_sub(&R.rt, 0, ser, lv, sl); return *ser != 0;
}
static int bloomq_make_pipeline(MTLPixelFormat fmt) {
    if (g_bloomq_pso && g_bloomq_fmt == fmt) return 1;
    static const char *src =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "struct V { float4 p [[position]]; float2 uv; };\n"
        "vertex V vs(uint id [[vertex_id]]) { float2 uv=float2((id<<1)&2,id&2); V o; o.p=float4(uv*float2(2,-2)+float2(-1,1),0,1); o.uv=uv; return o; }\n"
        "struct P { float threshold; float coeff_g; float coeff_b; float dm; float sh; };\n"
        "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant P& p [[buffer(0)]]) { float2 d=float2(1.0/p.dm,1.0/p.sh); float3 c=0; for (int y=-1;y<=1;y+=2) for (int x=-1;x<=1;x+=2) { float3 v=t.sample(s,in.uv+float2(x,y)*d).rgb; float l=v.r*0.2126+v.g*p.coeff_g+v.b*p.coeff_b; if(l>=p.threshold) c+=v/(1.0+l); } return float4(c*0.25,1); }\n";
    NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src) options:nil error:&e];
    if (!lib) { MLOG("[RED] Bloom 1/4 shader compilation failed: %s", e.localizedDescription.UTF8String); return 0; }
    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new]; pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"]; pd.colorAttachments[0].pixelFormat = fmt;
    g_bloomq_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!g_bloomq_pso) { MLOG("[RED] Bloom 1/4 pipeline failed: %s", e.localizedDescription.UTF8String); return 0; }
    g_bloomq_fmt = fmt;
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear; sd.sAddressMode = sd.tAddressMode = MTLSamplerAddressModeClampToEdge; sd.supportArgumentBuffers = YES;
    g_bloomq_samp = [g_dev newSamplerStateWithDescriptor:sd]; return g_bloomq_samp != nil;
}
static int bloomq_capture(void) {
    uint64_t ser; int lv, sl; if (!bloomq_target(&ser, &lv, &sl) || !g_bloomq.s_tex) return 0;
    id<MTLTexture> base = tex_mtl(R.rt.color[0]), src = tex_mtl(g_bloomq.s_tex); if (!base || !src) return 0;
    id<MTLTexture> dst = base; if (lv || sl) { static id<MTLTexture> vb; static int vl, vs; if (vb != base || vl != lv || vs != sl) { defer_obj_release(g_bloomq_view);   /* retire the old view (it used to be overwritten directly: it stayed in the residency set and pinned the old base texture with it) */ g_bloomq_view = [base newTextureViewWithPixelFormat:base.pixelFormat textureType:base.textureType levels:NSMakeRange(lv, 1) slices:NSMakeRange(sl, 1)]; if (g_bloomq_view) res_add(g_bloomq_view); vb = base; vl = lv; vs = sl; } dst = g_bloomq_view; }
    if (!dst || !bloomq_make_pipeline(dst.pixelFormat)) return 0;
    rp_end();
    RPD *rp = [RPD new]; rp.colorAttachments[0].texture = dst; rp.colorAttachments[0].loadAction = MTLLoadActionDontCare; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    RENC en = [g_cb renderCommandEncoderWithDescriptor:rp]; if (!en) return 0;
#if DM_NVN_M3
    [en setFragmentTexture:src atIndex:0]; [en setFragmentSamplerState:g_bloomq_samp atIndex:0];
    float *p = g_bloomq_const[g_bloomq_i].contents; p[0] = g_bloomq.threshold; p[1] = g_bloomq.coeff_g; p[2] = g_bloomq.coeff_b; p[3] = src.width; p[4] = src.height; [en setFragmentBuffer:g_bloomq_const[g_bloomq_i] offset:0 atIndex:0];
#else
    [en barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice]; [en setArgumentTable:g_bloomq_args atStages:MTLRenderStageFragment];
    [g_bloomq_args setTexture:src.gpuResourceID atIndex:0]; [g_bloomq_args setSamplerState:g_bloomq_samp.gpuResourceID atIndex:0];
    float *p = g_bloomq_const[g_bloomq_i].contents; p[0] = g_bloomq.threshold; p[1] = g_bloomq.coeff_g; p[2] = g_bloomq.coeff_b; p[3] = src.width; p[4] = src.height; [g_bloomq_args setAddress:g_bloomq_const[g_bloomq_i].gpuAddress atIndex:0];
#endif
    [en setViewport:(MTLViewport){0,0,dst.width,dst.height,0,1}]; [en setRenderPipelineState:g_bloomq_pso]; [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; m4_end(en);
    g_bloomq_i = (g_bloomq_i + 1) % 3; args_dirty_all(); g_pso_dirty = g_dss_dirty = 1; return 1;
}
// ---------- Dragged cargo follows the finger: goods on the merchant’s shelf have no “picked up” state, so during dragging the replay layer moves “the draw that renders this item’s icon” to the finger and draws it again at the end of the same pass ----------
// Each rectangle in the UI pass is a 4-vertex draw (20 bytes per vertex: 3 position floats + 2 texture-coordinate floats; position is transformed to clip coordinates by the matrix in vertex constant 0).
// The frontend (front_ios.m) provides the finger-down point, shelf grid, and dragged-icon center (normalized, origin at top left); here, in the UI pass, find the smallest rectangle that covers that point and has all four edges on grid lines—that is the item
// (Using only the point can misidentify it: about 0.1 seconds after the press, the game draws a one-cell-sized rectangle centered on the press point, not aligned to grid lines—it was mistaken for the item twice on the device)
// (Does not depend on the game’s selected state: selection takes about 0.1 seconds for the controller cursor to land on the item, which is too long for a fast drag starting on press). The same rectangle is drawn twice: once for the cell background and once for the icon:
// The grid base uses a shared texture (many rectangles use it this run); each icon has its own texture—select the texture used the fewest times this run. Report recognized rectangles to the frontend (to darken the original shelf positions).
// Vertical position and height of the UI rectangle in the actual display (both as a fraction of screen height: y = 0 is top-aligned, h = 1 is the entire screen); registered by the frontend and read at each draw (changes between console / handheld modes).

// ---------- Drag-follow image for shop items: items on a vendor's shelf have no "picked up" state, so during a drag the replay layer takes "the draw that draws this item's icon", translates it to the finger at the end of the same pass and draws it once more ----------
// In the UI pass each rectangle is one 4-vertex draw (20-byte vertices: position 3 floats + texture coordinates 2 floats; the position goes to clip coordinates through the matrix in vertex constant 0).
// The front end (front_ios.m) gives the point where the finger pressed, the shelf grid and the center of the drag-follow image (normalized, top-left origin); here, in the UI pass, find the rectangle that "covers this point, has all four edges on grid lines, and has the smallest area" = that item
// (identifying by the point alone gets it wrong: about 0.1 seconds after the press the game draws a one-cell-sized rectangle centered on the press point, not on grid lines -- on a real device it was twice taken for the item)
// (does not depend on the game's selected state: selection has to wait about 0.1 seconds for the controller cursor to land on the item, and a quick drag that starts on press cannot wait for that). The same rectangle has two draws, cell background and icon:
// the cell background uses a common texture (many rectangles in this pass use it), while each item has its own icon texture -- take the draw with the texture "used the fewest times in this pass". The recognized rectangle is reported back to the front end (used to cover the original shelf position with a dark color).
// Vertical position and height of the UI rectangle in the real picture (both as a fraction of the picture height: y = 0 touches the top, h = 1 is the whole picture); registered by the front end and read fresh on every draw (changes when switching between docked / handheld mode).
// The game draws the UI into the upper h portion of the UI texture (the viewport is the whole texture, and the projection matrix carries the scaling).
static void (*g_ui_rect_fn)(double *y, double *h);
void mtl_set_ui_rect_fn(void (*fn)(double *y, double *h)) { g_ui_rect_fn = fn; }
static double ui_hfrac(void) { double y = 0, h = 1; if (g_ui_rect_fn) g_ui_rect_fn(&y, &h); return h > 0 && h < 1 ? h : 1; }
static _Atomic int g_ghost_on; static _Atomic uint64_t g_ghost_pt, g_ghost_at, g_ghost_found, g_ghost_grid;
void mtl_ui_ghost(int on, float px, float py, const float grid[4], float cx, float cy) {
    if (on) { uint64_t gv = 0; for (int i = 0; i < 4; i++) gv |= (uint64_t)(uint16_t)(fminf(fmaxf(grid[i], 0), 1) * 65535 + .5f) << (16 * i); atomic_store(&g_ghost_grid, gv);
              atomic_store(&g_ghost_pt, (uint64_t)(uint32_t)lroundf(fminf(fmaxf(px, 0), 1) * 65535) | (uint64_t)(uint32_t)lroundf(fminf(fmaxf(py, 0), 1) * 65535) << 32);
              atomic_store(&g_ghost_at, (uint64_t)(uint32_t)(int32_t)lroundf(cx * 65535) | (uint64_t)(uint32_t)(int32_t)lroundf(cy * 65535) << 32); }
    else atomic_store(&g_ghost_found, 0);
    atomic_store(&g_ghost_on, on);
}
int mtl_ui_ghost_rect(float r[4]) { uint64_t v = atomic_load(&g_ghost_found); if (!v) return 0; for (int i = 0; i < 4; i++) r[i] = (float)((v >> (16 * i)) & 0xffff) / 65535.f; return 1; }
static struct { uint64_t pass; int n, nt, busy; float area, r[4]; struct { void *S_; p_draw dw; uint64_t tex; } c[4]; uint64_t tid[128]; uint16_t tn[128]; } g_gh;
static uint64_t ghost_tex(void) {   // The texture actually sampled by this draw's fragment shader (slot taken from shader reflection; the other slots in S.tex are left over from earlier draws)
    __unsafe_unretained NvProg *pr = S.prog ? prog_get(S.prog) : nil; NvShader *f = pr ? pr->sh[1] : nil; if (!f) return 0;
    for (int i = 0; i < f->rf.nb; i++) if ((f->rf.b[i].kind == NVN_RB_TEX || f->rf.b[i].kind == NVN_RB_SAMP) && f->rf.b[i].reg < 128 && S.tex[1][f->rf.b[i].reg]) return S.tex[1][f->rf.b[i].reg];
    return 0; }

// Whether the current UI draw is "untextured and exactly covering the whole UI rectangle" (the UI rectangle occupies the upper hf of the texture)
static int ui_cover_all(double hf) {
    if (S.vbuf[0].size != 80 || ghost_tex()) return 0;
    const p_bind_buf *u = &S.ubo[0][0]; void *vb, *mb; uint64_t vo, mo; const uint8_t *vp_, *mp_;
    if (!u->addr || u->size < 64 || pend_hits(u->addr, 64) || pend_hits(S.vbuf[0].address, 80)) return 0;
    if (!mtl_addr(S.vbuf[0].address, 80, &vb, &vo) || !(vp_ = [(__bridge id<MTLBuffer>)vb contents]) || !mtl_addr(u->addr, 64, &mb, &mo) || !(mp_ = [(__bridge id<MTLBuffer>)mb contents])) return 0;
    float m[16], x0 = 9, y0 = 9, x1 = -9, y1 = -9; memcpy(m, mp_ + mo, 64);
    for (int i = 0; i < 4; i++) { float f[2]; memcpy(f, vp_ + vo + 20 * i, 8); float nx = (m[0] * f[0] + m[4] * f[1] + m[12] + 1) / 2, ny = (m[1] * f[0] + m[5] * f[1] + m[13] + 1) / 2;
        x0 = fminf(x0, nx); x1 = fmaxf(x1, nx); y0 = fminf(y0, ny); y1 = fmaxf(y1, ny); }
    return x0 < 0.002f && x1 > 0.998f && y0 < 0.002f && fabsf(y1 - (float)hf) < 0.002f;
}
static void ghost_note(const p_draw *dw) {
    if (g_gh.busy || dw->count != 4 || S.nattr != 2 || S.stream[0].stride != 20 || S.vbuf[0].size != 80) return;
    if (g_gh.pass != g_stat_pass) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = g_gh.nt = 0; g_gh.pass = g_stat_pass; }
    uint64_t t = ghost_tex(); int k = 0; for (; k < g_gh.nt; k++) if (g_gh.tid[k] == t) break;
    if (k == g_gh.nt && g_gh.nt < 128) { g_gh.tid[k] = t; g_gh.tn[k] = 0; g_gh.nt++; } if (k < g_gh.nt) g_gh.tn[k]++;
    const p_bind_buf *u = &S.ubo[0][0]; void *vb, *mb; uint64_t vo, mo; const uint8_t *vp_, *mp_;
    if (!u->addr || u->size < 64 || pend_hits(u->addr, 64) || pend_hits(S.vbuf[0].address, 80)) return;
    if (!mtl_addr(S.vbuf[0].address, 80, &vb, &vo) || !(vp_ = [(__bridge id<MTLBuffer>)vb contents]) || !mtl_addr(u->addr, 64, &mb, &mo) || !(mp_ = [(__bridge id<MTLBuffer>)mb contents])) return;
    float m[16], x0 = 9, y0 = 9, x1 = -9, y1 = -9; memcpy(m, mp_ + mo, 64);
    for (int i = 0; i < 4; i++) { float f[2]; memcpy(f, vp_ + vo + 20 * i, 8); float nx = (m[0] * f[0] + m[4] * f[1] + m[12] + 1) / 2, ny = (m[1] * f[0] + m[5] * f[1] + m[13] + 1) / 2;
        x0 = fminf(x0, nx); x1 = fmaxf(x1, nx); y0 = fminf(y0, ny); y1 = fmaxf(y1, ny); }
    { float hf = (float)ui_hfrac(); y0 /= hf; y1 /= hf; }   // The points given by the front end are normalized coordinates within the UI rectangle; when the UI occupies only the upper hf portion of the texture, convert to the same convention (unchanged when hf = 1)
    uint64_t pv = atomic_load(&g_ghost_pt); float qx = (float)(uint32_t)pv / 65535.f, qy = (float)(pv >> 32) / 65535.f;
    if (qx < x0 || qx > x1 || qy < y0 || qy > y1) return;
    uint64_t gv = atomic_load(&g_ghost_grid); float gx = (float)(gv & 0xffff) / 65535.f, gy = (float)((gv >> 16) & 0xffff) / 65535.f, gw = (float)((gv >> 32) & 0xffff) / 65535.f, gh = (float)(gv >> 48) / 65535.f;
    if (gw <= 0 || gh <= 0) return;
    float w = x1 - x0, h = y1 - y0, area = w * h, c0 = (x0 - gx) / gw, r0 = (y0 - gy) / gh, cn = w / gw, rn = h / gh;   // Which cell the top-left corner is in, and how many cells it occupies
    if (fabsf(c0 - roundf(c0)) > 0.15f || fabsf(r0 - roundf(r0)) > 0.15f || fabsf(cn - roundf(cn)) > 0.15f || fabsf(rn - roundf(rn)) > 0.15f || cn < 0.85f || rn < 0.85f || cn > 2.15f || rn > 4.15f) return;   // One item is at most 2x4 cells
    if (g_gh.n && area > g_gh.area * 1.03f) return;                       // A smaller one already exists
    if (g_gh.n && area < g_gh.area * 0.97f) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0; }   // Found a smaller one: the earlier ones are invalidated
    if (g_gh.n == 4) { free(g_gh.c[0].S_); memmove(&g_gh.c[0], &g_gh.c[1], 3 * sizeof g_gh.c[0]); g_gh.n = 3; }
    g_gh.area = area; g_gh.r[0] = x0; g_gh.r[1] = y0; g_gh.r[2] = x1; g_gh.r[3] = y1;
    typeof(g_gh.c[0]) *c = &g_gh.c[g_gh.n++]; c->S_ = malloc(sizeof S); memcpy(c->S_, &S, sizeof S); c->dw = *dw; c->tex = t;
}
static void ghost_flush(void) {   // Before the render pass ends: translate the recognized icon draw to the finger and draw it once more (moves the viewport, does not change vertices; the scissor is opened up)
    if (g_gh.busy || !g_gh.n) return;
    if (g_gh.pass != g_stat_pass || !atomic_load(&g_ghost_on)) { for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0; return; }
    int best = -1, bn = 1 << 30; for (int i = 0; i < g_gh.n; i++) { int cnt = 0; for (int k = 0; k < g_gh.nt; k++) if (g_gh.tid[k] == g_gh.c[i].tex) cnt = g_gh.tn[k]; if (cnt < bn) { bn = cnt; best = i; } }
    typeof(g_gh.c[0]) *c = &g_gh.c[best]; uint64_t at = atomic_load(&g_ghost_at); float ax = (float)(int32_t)(uint32_t)at / 65535.f, ay = (float)(int32_t)(uint32_t)(at >> 32) / 65535.f;
    float W = (float)g_pass.tw, H = (float)g_pass.th, ccx = (g_gh.r[0] + g_gh.r[2]) / 2, ccy = (g_gh.r[1] + g_gh.r[3]) / 2;
    { uint64_t v = 0; for (int i = 0; i < 4; i++) v |= (uint64_t)(uint16_t)(fminf(fmaxf(g_gh.r[i], 0), 1) * 65535 + .5f) << (16 * i); atomic_store(&g_ghost_found, v ? v : 1); }
    g_gh.busy = 1; typeof(S) save; memcpy(&save, &S, sizeof S); p_scissor sc = R.sc; int has_sc = R.has_sc;
    memcpy(&S, c->S_, sizeof S); if (!S.has_vp) { S.vp[0][0] = S.vp[0][1] = 0; S.vp[0][2] = W; S.vp[0][3] = H; S.has_vp = 1; }
    S.vp[0][0] += (ax - ccx) * W; S.vp[0][1] += (ay - ccy) * H * (float)ui_hfrac(); R.has_sc = 0;
    args_dirty_all(); g_pso_dirty = g_dss_dirty = 1; draw(&c->dw, NULL);
    { static int once; if (!once++) MLOG("Cargo-following hand icon: first case channel %llu, %d candidates, choosing the texture used %d times this run", g_stat_pass, g_gh.n, bn); }
    memcpy(&S, &save, sizeof S); args_dirty_all(); R.sc = sc; R.has_sc = has_sc; g_pso_dirty = g_dss_dirty = 1; g_gh.busy = 0;
    for (int i = 0; i < g_gh.n; i++) free(g_gh.c[i].S_); g_gh.n = 0;
}
static void draw(const p_draw *dw, const p_draw_idx *di) {
    draw_trace_ctx trace __attribute__((cleanup(draw_trace_finish))) = {
        .prev = g_draw_trace, .nvn_program = S.prog,
        .pipeline = "not-requested", .result = "not-encoded",
        .enabled = nvn_draw_trace_on()
    };
    if (trace.enabled) {
        trace.seq = atomic_fetch_add(&g_draw_trace_seq, 1) + 1;
        trace.frame = atomic_load(&g_cur_frame);
        g_draw_trace = &trace;
    }
    if (!S.prog) {
        static _Atomic uint64_t null_draw_seq;
        uint64_t n = nvn_bind_trace_on()
            ? atomic_fetch_add_explicit(&null_draw_seq, 1, memory_order_relaxed) : UINT64_MAX;
        if (n < 16)
            MLOG("[DEBUG-RA-BIND] null-draw seq=%llu frame=%llu traceSeq=%llu",
                 (unsigned long long)n, (unsigned long long)atomic_load(&g_cur_frame),
                 (unsigned long long)trace.seq);
        trace.result = "no-program";
        RED_ONCE(18, "No program bound before draw");
        return;
    }
    __unsafe_unretained NvProg *pr = prog_get(S.prog);
    trace.program = pr;
    if (!pr) { trace.result = "program-unresolved"; return; }
    if (pr->gs_unresolved || pr->sh[2]) {
        trace.result = "unsupported-geometry";
        static int n; if (n++ < 4) MLOG("[RED] Unfoldable geometry; skipping draw (fail-closed)");
        return;
    }
    if (g_ruler_truth == 0) {   // Ruler negative control (DM_NVN_RULER_NEG=<fragment key>): in the mode under test drop this shader's draws, and the ruler must report non-equivalence
        static const char *neg; static int ng = -1; if (ng < 0) { neg = getenv("DM_NVN_RULER_NEG"); ng = neg && *neg; }
        if (ng && pr) for (int st = 0; st < 5; st++) { __unsafe_unretained NvShader *sh = pr->sh[st]; if (sh && sh->stage == 1 && !strncmp(sh->key, neg, strlen(neg))) { static int once; if (!once++) MLOG("DM_NVN_RULER_NEG: tested mode discards draw for shader %s", sh->key); return; } }
    }
    if (zskip_can_draw(pr)) {   // Zero-source skip: the output is always 0, writing 0 over 0, so this draw changes no result (see zskip_can_draw)
        g_zskip_count++; { static int once; if (!once++) MLOG("Zero source skip: first case channel %llu", g_stat_pass + 1); }
        PD("N %llu zero source skip", g_stat_pass + 1); return; }
    if (!g_defer_replaying) {
        if (g_defer.active && g_defer.in_p && !g_defer.at_q) { if (defer_take(dw, di)) return; }
    }
    // Return before opening a pass for draws that produce nothing (both faces culled, or empty scissor): otherwise the pass is open and the entire attachment is needlessly read and written on TBDR
    // (Mac character selection page: all 3 passes open only for draws of this kind; all reads, no draws)

    // Draws that cannot draw anything (both faces culled, empty scissor) return before the render pass is opened: otherwise the render pass is already open and on TBDR the whole attachment is loaded and stored for nothing
    // (Mac character select page: 3 render passes are opened only because of such draws, all loaded, zero draws)
    // Diagnostic switch for measuring an upper bound DM_NVN_SKIP_FS=<fragment key>[,<fragment key>...] (none by default): drops all draws of these fragment shaders (the picture will be wrong; only used in real-device ABA to measure "how much GPU time this class of draws is worth in total")
    { static char sk[16][20]; static int nsk = -1;
      if (nsk < 0) { nsk = 0; for (const char *q = getenv("DM_NVN_SKIP_FS"); q && *q && nsk < 16; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) { size_t n = strcspn(q, ","); if (n > 16) n = 16; memcpy(sk[nsk], q, n); sk[nsk++][n] = 0; }
                     if (nsk) MLOG("DM_NVN_SKIP_FS: discarding draws for %d fragment shaders (diagnostic, incorrect image)", nsk); }
      if (g_ub_n > 0 && pr->sh[1] && !g_defer_replaying && ub_skip(pr->sh[1]->key)) return;   // In-process alternating measurement: this phase drops this class
      if (nsk && pr->sh[1] && !g_defer_replaying) for (int i = 0; i < nsk; i++) if (!strncmp(pr->sh[1]->key, sk[i], strlen(sk[i]))) return; }

    // Similar diagnostic DM_NVN_SKIP_SMALL=<width>: drop all draws whose target width (by the attached level) is <= this value -- measures how much "dropping tiny render passes in bulk" is worth under the parallel pipeline (fixed per-pass overhead)
    { static int small = -1; if (small < 0) { const char *e = getenv("DM_NVN_SKIP_SMALL"); small = e ? atoi(e) : 0; if (small) MLOG("DM_NVN_SKIP_SMALL: discarding draws with target width ≤ %d (diagnostic, incorrect image)", small); }
      if (small > 0 && R.has_rt && !g_defer_replaying) { int k = R.rt.n && R.rt.color[0] ? 0 : R.rt.depth ? 8 : -1;
        if (k >= 0) { uint64_t s_; int lv, sl; rt_sub(&R.rt, k, &s_, &lv, &sl); const tex_t *t = k == 8 ? R.rt.depth : R.rt.color[0]; if ((int)MAX(t->d.width >> lv, 1u) <= small) return; } } }

    // Experimental switch DM_NVN_NOFXAA=1 (off by default; keep or drop decided by the visual ruler + looking at the picture on a real machine): skips the FXAA full-screen draw, and the later sharpening / upscale reads FXAA's input instead (the tone mapping result)
    { static int nofxaa = -1; static uint64_t fx_from, fx_to; if (nofxaa < 0) { const char *e = getenv("DM_NVN_NOFXAA"); nofxaa = e && !strcmp(e, "1"); if (nofxaa) MLOG("DM_NVN_NOFXAA=1: skipping FXAA"); }
      if (nofxaa && tbdr_on() && !g_defer_replaying && pr->sh[1]) { NvShader *f = pr->sh[1];
        if (!strncmp(f->key, "ebcfcd1fa0a77dbc", 16) && !g_enc && R.rt.n == 1 && R.rt.color[0] && !R.clear_c[0]) {
            uint64_t h = 0; for (int i = 0; i < f->rf.nb && !h; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128) h = S.tex[1][f->rf.b[i].reg];
            if (h) { fx_from = R.rt.color[0]->serial; fx_to = h; PD("N %llu skipping FXAA: reader of ser=%llu now reads ser=%llu", g_stat_pass, (unsigned long long)fx_from, (unsigned long long)h_ser(h)); return; } }
        else if (fx_from && !strncmp(f->key, "52aac1f2306b8fc5", 16)) { for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == fx_from) { S.tex[1][r] = fx_to; g_args_dirty[1] = 1; } fx_from = 0; } } }
    { static uint64_t bloomq_frame; if (bloomq_frame != atomic_load(&g_cur_frame)) { bloomq_frame = atomic_load(&g_cur_frame); bloomq_reset(); }
      if (bloomq_enabled() && tbdr_on() && !g_defer_replaying && pr->sh[1]) {
        NvShader *f = pr->sh[1]; uint64_t tgt = 0; int lv, sl;
        if (!strncmp(f->key, "2938a5b71b8e21b2", 16)) {
            if (g_bloomq.state) bloomq_reset();
            if (bloomq_target(&tgt, &lv, &sl) && R.rt.color[0]) {
                uint64_t h = 0; const tex_t *st = NULL; int cbuf = 0;
                for (int i = 0; i < f->rf.nb; i++) { nvn_rbind *b = &f->rf.b[i];
                    if (b->kind == NVN_RB_TEX && b->reg < 128 && !h) { h = S.tex[1][b->reg]; int tid = (int)(h & 0xfffff); if (tid > 0 && tid < g_ntreg && g_treg[tid].ok) st = &g_treg[tid].t; }
                    if (b->kind == NVN_RB_SAMPT && b->reg < 128) g_bloomq.samp_h = S.tex[1][b->reg] >> 20;
                    if (b->kind == NVN_RB_SAMP && b->reg < 128) g_bloomq.samp_h = S.samp[1][b->reg] >> 20;
                    if (b->kind == NVN_RB_CBUF && !cbuf) cbuf = ubo_float(1, (uint16_t)(b->reg << 8), &g_bloomq.threshold);   // c7[0].x
                }
                if (h && st && cbuf && f->rf.c1_len >= sizeof(float) * 2) {
                    float c1[2]; memcpy(c1, f->rf.c1, sizeof c1); g_bloomq.s_tex = st; g_bloomq.s_h = h; g_bloomq.s_ser = st->serial; g_bloomq.b0_ser = tgt; g_bloomq.coeff_g = c1[0]; g_bloomq.coeff_b = c1[1]; g_bloomq.state = 1;
                    { static int once; if (!once++) MLOG("Bloom 1/4 start: active (S ser=%llu B0 ser=%llu)", (unsigned long long)g_bloomq.s_ser, (unsigned long long)g_bloomq.b0_ser); } PD("A %llu bloom 1/4 start S=%llu B0=%llu", g_stat_pass, (unsigned long long)g_bloomq.s_ser, (unsigned long long)g_bloomq.b0_ser); R.clear_c[0] = 0;   // The whole B0 level is no longer needed: drop the pending clear too, otherwise a clear-only render pass would be opened just for it
                    return;
                }
            }
        } else if (g_bloomq.state == 1 && !strncmp(f->key, "019ce2f82edbda84", 16)) {
            uint64_t src = 0; for (int i = 0; i < f->rf.nb; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128) { src = h_ser(S.tex[1][f->rf.b[i].reg]); break; }
            if (src == g_bloomq.b0_ser && bloomq_target(&tgt, &lv, &sl) && bloomq_capture()) { R.clear_c[0] = 0; g_bloomq.b1_ser = tgt; g_bloomq.state = 2; PD("B %llu bloom 1/4 native capture B0=%llu B1=%llu", g_stat_pass, (unsigned long long)g_bloomq.b0_ser, (unsigned long long)g_bloomq.b1_ser); return; }
            RED_ONCE(62, "Bloom 1/4 start: capture was skipped, but native capture did not render; bloom is missing this frame"); bloomq_reset();
        } else if (g_bloomq.state == 2 && !strncmp(f->key, "019ce2f82edbda84", 16)) {
            for (int i = 0; i < f->rf.nb; i++) if (f->rf.b[i].kind == NVN_RB_TEX && f->rf.b[i].reg < 128 && h_ser(S.tex[1][f->rf.b[i].reg]) == g_bloomq.b1_ser) { g_bloomq.b1_h = S.tex[1][f->rf.b[i].reg]; break; }
        } else if (g_bloomq.state == 2 && !strncmp(f->key, "cc264583f2000c81", 16)) {
            if (bloomq_target(&tgt, &lv, &sl) && tgt == g_bloomq.b0_ser) { g_bloomq.state = 3; PD("C %llu bloom 1/4 skip B0 upsampling", g_stat_pass); return; }
        } else if (g_bloomq.state == 3 && !strncmp(f->key, "2e2d6700a2569795", 16)) {
            if (g_bloomq.b1_h) { for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == g_bloomq.b0_ser) { S.tex[1][r] = g_bloomq.b1_h; g_args_dirty[1] = 1; } PD("D %llu bloom 1/4 composite now reads B1", g_stat_pass); g_bloomq_last_b0 = g_bloomq.b0_ser; g_bloomq_last_b1h = g_bloomq.b1_h; g_bloomq_last_b1s = g_bloomq.b1_ser; bloomq_reset(); }
            else { RED_ONCE(63, "Bloom 1/4 state is missing the B1 handle"); bloomq_reset(); }
        } else if (!g_bloomq.state && g_bloomq_last_b1h && h_ser(g_bloomq_last_b1h) == g_bloomq_last_b1s && !strncmp(f->key, "2e2d6700a2569795", 16)) {   // The B1 handle still points to the same texture (not recognized after a rebuild on resolution change)
            // If the bloom chain did not run this frame (runtime extrapolation skipped it on an extrapolated frame: outside the 469 input regions is the previous frame's LDR, so it cannot run), we reach bloom compositing: read B1 as in the latest change and composite the previous frame's bloom.

            // The bloom overlay is reached without the bloom chain having run this frame (the extrapolated frame of runtime extrapolation skipped it: outside the region the input 469 is the previous frame's LDR, so it cannot run): redirect the read to B1 as last time, overlaying the previous frame's bloom.
            // Without the redirect it would read B0 -- with bloom 1/4 on, B0's upsample is always skipped and never written, so what is read is uninitialized video memory (on M4 Pro the units in the region turn entirely green; elsewhere it is mostly 0, i.e. missing bloom)
            int hit = 0; for (int r = 0; r < 128; r++) if (S.tex[1][r] && h_ser(S.tex[1][r]) == g_bloomq_last_b0) { S.tex[1][r] = g_bloomq_last_b1h; hit = 1; g_args_dirty[1] = 1; }
            if (hit) { PD("D %llu bloom chain did not run: composite reuses previous frame's B1", g_stat_pass); static uint64_t n; if (n++ % 600 == 0) MLOG("Bloom 1/4: composite reuses previous frame's B1 when the bloom chain did not run (occurrence %llu)", (unsigned long long)n); }
        } else if (g_bloomq.state) { RED_ONCE(61, "Bloom 1/4 start: another draw occurred mid-chain; bloom is incomplete this frame"); bloomq_reset(); }
      }
    }
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK) return;

    // The continued render pass may have been opened with fewer depth / stencil attachments (open modes 2 / 3 / 4): if the plane this draw needs is not attached, split before it (on reopen, attach per the new lookahead)
    if (g_rt_merged && g_enc_mode != 0 && (((S.ds.dt || S.ds.dw) && !g_pass.hd) || (S.ds.st && !g_pass.hs))) {
        static int once2; if (!once2++) MLOG("Same-target reuse: required depth/stencil planes are not attached for drawing within the reused channel; split here (first occurrence, channel %llu)", g_stat_pass);
        PD("M %llu required depth/stencil planes are not attached for drawing within the reused channel; split here", g_stat_pass); rp_end(); }
    if (g_rt_range_active && (!rt_range_pr_ro(pr) || !S.ds.dt || S.ds.dw || S.ds.st)) rp_end();   // The external write visibility at the original encoder boundary is not removed
    if (g_rt_merged && draw_samples_rt(pr)) { static int once; if (!once++) MLOG("Same-target reuse: required sampled attachments are not attached for drawing within the reused channel; split here (first occurrence, channel %llu)", g_stat_pass);
        PD("M %llu required sampled attachments are not attached for drawing within the reused channel; split here", g_stat_pass); rp_end(); }
    static id<MTLFunction> fz_fn;   // Specialized fragment function for this draw of the later pass in a merged render pass (nil = use the original version)
    if (g_fz.active && g_enc && g_fz.phase == 2 && !fz_p2_draw(pr, dw, di, g_fz.lmap2, &fz_fn)) {   // Cannot be proven: end the merged render pass (the earlier pass's result is stored as usual) and reopen with the later pass's original targets
        static int once3; if (!once3++) MLOG("Readable tile channel merge: drawing in the subsequent channel cannot be verified; fall back to separate channels (first occurrence, channel %llu)", g_stat_pass);
        g_fz_fallback++; rp_end(); }
    if (R.has_sc && !g_enc) {
        NSUInteger tw0 = 0, th0 = 0; rt_dims(&tw0, &th0);
        long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y < 0 ? 0 : R.sc.y, x1 = (long)R.sc.x + R.sc.w, y1 = (long)R.sc.y + R.sc.h;
        if (x1 > (long)tw0) x1 = tw0; if (y1 > (long)th0) y1 = th0;
        if (x1 <= x0 || y1 <= y0) return;
    }
#ifdef NVN_RT_RANGE_TEST
    if (!g_enc) rr_begin();
#endif
    int defer_q_draw = !g_enc && !g_defer_replaying && g_defer_natq > 0;   // The draw that opens Q's render pass: after the render pass is opened as usual through hoist_try / fuse_try, replay P first and then draw this one
    if (!g_enc && tb_sub(TB_HOIST)) hoist_try();   // First hoist and execute the pure-clear render passes sandwiched between this render pass and its continuation (before rp_begin_ex, not occupying this render pass)

    // The deferral decision comes after hoist_try: hoisted pure-clear render passes are skipped in the lookahead, and only then can P's same-target continuation segment be recognized
    if (!g_enc && tbdr_on() && !defer_has_in_p() && !g_defer_replaying && !g_fz.active && !g_fz_checking && g_exec_next) { defer_try(dw, di); if (g_defer_cur->active && g_defer_cur->in_p) return; }
    if (!g_enc && tb_sub(TB_FUSE) && !g_fz_checking) fuse_try();   // Readable tile render pass merge is on by default; when the structure / same-pixel proof does not hold it still falls back to a normal render pass, and it is not re-entered during the CHECK redraw
    int need = g_enc || !tb_sub(TB_DS) ? 3 : pass_ds_need(), mode = defer_q_draw ? 0 : need == 0 ? 2 : need == 2 ? 3 : need == 1 ? 4 : 0;   // 2 attach neither, 3 attach stencil only, 4 attach depth only
    int skip_d = mode != 0;
    if (!g_enc) { g_fd_dw = dw; g_fd_di = di; }
    g_rp_layered = pr->sh[0] && pr->sh[0]->layer_out;
    if (g_enc && g_pass.layered != g_rp_layered) rp_end();
    id<MTLRenderCommandEncoder> en = rp_begin_ex(mode);
    g_rp_layered = 0;
    g_fd_dw = NULL; g_fd_di = NULL;
    if (!en) { trace.result = "no-render-encoder"; return; }
    if (defer_q_draw) defer_replay();
    if (skip_d) { g_stat_ds_omit++; if (mode == 3) g_stat_s_only++; else if (mode == 4) g_stat_d_only++; }

    // The pipeline depends only on (program, render pass targets, blend / color / write mask / multisample state, vertex layout): if these did not change, reuse the previous one
    static id<MTLRenderPipelineState> cur_ps; static NvProg *cur_pr; static NvShader *cur_vs, *cur_fs; static uint64_t cur_ps_pass;
    if (g_fz.active && !g_defer_replaying) { g_pso_lmap = g_fz.phase == 1 ? g_fz.lmap1 : g_fz.lmap2; g_pso_fsfn = g_fz.phase == 1 ? nil : fz_fn; g_pso_fz = g_fz.phase == 1 ? (pso_fz_desc){0} : g_fz_desc; g_pso_dirty = 1; }   // Readable tile merge: fetch the pipeline by the physical -> logical mapping and the specialization
    NvShader *draw_vs = pr->sh[0], *draw_fs = pr->sh[1];
    if (g_pso_dirty || pr != cur_pr || draw_vs != cur_vs || draw_fs != cur_fs || cur_ps_pass != g_pass.serial) {
        cur_ps = pso_get(pr, nil); cur_pr = pr; cur_vs = draw_vs; cur_fs = draw_fs;
        cur_ps_pass = g_pass.serial; g_pso_dirty = !cur_ps;
    }
    else if (g_draw_trace) g_draw_trace->pipeline = "encoder-reuse";
    if (g_pso_lmap) { g_pso_lmap = NULL; g_pso_fsfn = nil; memset(&g_pso_fz, 0, sizeof g_pso_fz); g_pso_dirty = 1; }   // The next draw fetches again according to its own situation
    __unsafe_unretained id<MTLRenderPipelineState> ps = cur_ps;
    if (!ps) { trace.result = "pipeline-unavailable"; return; }
    if (ps != g_enc_pso) { enc_r_pso(en, ps); g_enc_pso = ps; }
    static id<MTLDepthStencilState> cur_ds; static uint64_t cur_ds_pass;
    if (g_dss_dirty || cur_ds_pass != g_pass.serial) { cur_ds = dss_get(g_pass.hd, g_pass.hs); cur_ds_pass = g_pass.serial; g_dss_dirty = 0; }
    if (cur_ds != g_enc_dss) { enc_r_dss(en, cur_ds); g_enc_dss = cur_ds; }
    #define FX_SET(bit, same, assign, call) do { if (!g_fx_dedup || !(g_enc_fx.valid & (1u << (bit))) || !(same)) { assign; g_enc_fx.valid |= 1u << (bit); call; } } while (0)
    { uint32_t r = S.st_ref[0]; FX_SET(7, g_enc_fx.sref == r, g_enc_fx.sref = r, enc_r_stencil_ref(en, r, 0)); }

    // Viewport / scissor (clamped within the target size)
    NSUInteger tw = g_pass.tw, th = g_pass.th;
    // When the UI rectangle is not flush with the top of the screen (true fullscreen on iPad: UI is laid out in 16:9 at the bottom of the screen; see dm/host/dm_uiscreen.c), shift all UI draws down: add a vertical offset to the projection matrix (ui_mat_patch), and shift the scissor down too.
    // UI draw = 4 vertices, 2 attributes, 20 bytes per vertex; a dedicated pass, composited at frame end (measured: startup loading, character selection, character creation, in-game loading, gameplay, opening a panel, and full-screen map have no other draws mixed in).
    // If the matrix cannot be read this time (there are writes in this commit that have not landed yet), fall back to moving the viewport: same position, but controls shifted up to the top will be clipped by the viewport's top edge this time.
    // Do not shift scissors that start at the top edge (y ≤ 0): when the host moves top-aligned controls (minimap, place names, etc.) back up to the actual top of the screen, they are drawn above the UI rectangle and must not be clipped.
    // Exception: the untextured draw that exactly covers the entire UI rectangle (panel dimming or transition fade to black) instead covers the entire screen—scale the viewport vertically by 1/h and do not shift it; otherwise a brightness boundary remains where the 3D outside the UI rectangle shows through.

    // When the UI rectangle does not touch the top edge of the picture (iPad true full screen: the UI is laid out at 16:9 at the bottom of the picture, see dm/host/dm_uiscreen.c), UI draws are shifted down as a whole: a vertical offset is added to the projection matrix (ui_mat_patch), and the scissor moves down with it.
    // A UI draw = 4 vertices, 2 attributes, 20-byte vertices, in a pass of its own and composited at end of frame (measured: startup loading, character select, character creation, entering-game loading, in-game, open panel, and full-screen map never mix in other draws).
    // When the matrix cannot be read this time (there are still unlanded writes in this submit), fall back to moving the viewport: same position, except that widgets lifted to the top get clipped by the viewport's top edge this one time.
    // A scissor starting from the top edge (y <= 0) does not have its top edge moved down: when the host lifts top-anchored widgets (minimap, area name etc.) back to the real top of the picture, they are drawn above the UI rectangle and must not be clipped.
    // Exception: the untextured block that exactly covers the whole UI rectangle (the dimming when a panel opens, the black curtain of a transition) is changed to cover the whole picture -- the viewport is scaled vertically by 1/h and not moved down, otherwise the 3D outside the UI rectangle is left with a light/dark dividing line.
    // When nobody registered the entry or the UI rectangle is the whole picture (iPhone, Mac default), vo = so = 0, vs = 1, the matrix is not changed, and what follows is word-for-word the same as before.
    long vo = 0, so = 0; double vs = 1, moff = 0;
    if (g_ui_rect_fn && dw && dw->count == 4 && S.nattr == 2 && S.stream[0].stride == 20) { double uy = 0, uh = 1; g_ui_rect_fn(&uy, &uh);
        if (uh > 0 && uh < 1) { if (ui_cover_all(uh)) vs = 1 / uh; else if (uy > 0) { so = lround(uy * th); if (ui_mat_ok()) moff = 2 * uy; else { vo = so; static int n; if (n++ < 4) MLOG("UI shift down: projection matrix could not be read this time; fall back to moving the viewport (channel %llu)", g_stat_pass); } } } }
    if (moff != g_ui_moff) { g_ui_moff = moff; g_args_dirty[0] = 1; }
    { MTLViewport v = S.has_vp ? (MTLViewport){S.vp[0][0], S.vp[0][1] + vo, S.vp[0][2], S.vp[0][3] * vs, S.dr[0], S.dr[1]} : (MTLViewport){0, vo, tw, th * vs, 0, 1};
      if (atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) { v.originY += v.height; v.height = -v.height; }   // lower-left origin (see g_origin_ll)
      FX_SET(0, !memcmp(&g_enc_fx.vp, &v, sizeof v), g_enc_fx.vp = v, enc_r_viewport(en, v)); }
    MTLScissorRect sc = {0, 0, tw, th};
    if (R.has_sc) {
        long x0 = R.sc.x < 0 ? 0 : R.sc.x, y0 = R.sc.y <= 0 ? 0 : R.sc.y + so, x1 = (long)R.sc.x + R.sc.w, y1 = vs != 1 ? (long)th : (long)R.sc.y + R.sc.h + so;
        if (x1 > (long)tw) x1 = tw; if (y1 > (long)th) y1 = th;
        if (x1 <= x0 || y1 <= y0) return;
        sc = (MTLScissorRect){(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)};
    }
    FX_SET(1, !memcmp(&g_enc_fx.sc, &sc, sizeof sc), g_enc_fx.sc = sc, enc_r_scissor(en, sc));
    if (S.poly.cull == NVN_FACE_FRONT_AND_BACK) return;
    { int c = S.poly.cull == NVN_FACE_FRONT ? MTLCullModeFront : S.poly.cull == NVN_FACE_BACK ? MTLCullModeBack : MTLCullModeNone; FX_SET(2, g_enc_fx.cull == c, g_enc_fx.cull = c, enc_r_cull(en, (MTLCullMode)c)); }
    { int w = (S.poly.front == NVN_FRONT_FACE_CCW) != atomic_load_explicit(&g_origin_ll, memory_order_relaxed) ? MTLWindingCounterClockwise : MTLWindingClockwise; FX_SET(3, g_enc_fx.wind == w, g_enc_fx.wind = w, enc_r_winding(en, (MTLWinding)w)); }   // the y flip of lower-left mode reverses the winding
    { int m = S.poly.mode == NVN_POLYGON_MODE_LINE ? MTLTriangleFillModeLines : MTLTriangleFillModeFill; FX_SET(4, g_enc_fx.fill == m, g_enc_fx.fill = m, enc_r_fill(en, (MTLTriangleFillMode)m)); }
    { float d[3] = {0, 0, 0}; if (S.poly.offset_en & 4) { d[0] = S.po.units; d[1] = S.po.factor; d[2] = S.po.clamp; }
      FX_SET(5, !memcmp(g_enc_fx.db, d, sizeof d), memcpy(g_enc_fx.db, d, sizeof d), enc_r_depth_bias(en, d[0], d[1], d[2])); }
    FX_SET(6, !memcmp(g_enc_fx.bc, S.blend_color, sizeof g_enc_fx.bc), memcpy(g_enc_fx.bc, S.blend_color, sizeof g_enc_fx.bc), enc_r_blend_color(en, S.blend_color[0], S.blend_color[1], S.blend_color[2], S.blend_color[3]));
    #undef FX_SET
    for (int s_ = 0; s_ < 16; s_++) {
        if (!S.vbuf[s_].address) continue;
        void *mb; uint64_t mo;
        uint64_t gpu;
        if (mtl_gpu_addr(S.vbuf[s_].address, 1, &mb, &mo, &gpu)) { enc_r_vbuf(en, 16 + s_, gpu); use_rec("buf", S.vbuf[s_].address, S.vbuf[s_].size, 0, NULL); }
    }
    static id<MTLBuffer> zero; if (!zero) { zero = [g_dev newBufferWithLength:64 options:MTLResourceStorageModeShared]; res_add(zero); }
    static uint64_t zero_gpu; if (!zero_gpu) zero_gpu = zero.gpuAddress;
    enc_r_vbuf(en, 30, zero_gpu);
    if (pr->sh[0]) encode_args(en, pr->sh[0], 0, 1);
    if (pr->sh[1]) encode_args(en, pr->sh[1], 1, 0);

    // Parameter binding captures the actual resources under the registry lock; if the version changed between the eligibility check and the binding, split and retry before any GPU draw.
    if (g_rt_range_active && (g_pass.reg_gen != g_reg_gen || g_pass.ser_gen != g_ser_gen || g_pass.range_dirty != nvn_dirty_now())) { rp_end(); draw(dw, di); return; }
    int prim = dw ? dw->prim : di->prim;
    MTLPrimitiveType mp;
    switch (prim) {
        case 0: mp = MTLPrimitiveTypePoint; break; case 1: mp = MTLPrimitiveTypeLine; break; case 3: mp = MTLPrimitiveTypeLineStrip; break;
        case 4: case 7: mp = MTLPrimitiveTypeTriangle; break; case 5: mp = MTLPrimitiveTypeTriangleStrip; break;
        default: RED_ONCE(19, "Primitive type %d not implemented", prim); return;
    }
    if (g_vis_counting) {   // Occlusion query counting: a new slot for this render pass (also switch to a new slot after a report / reset)
        if (!g_enc_has_vis) RED_ONCE(34, "No visibility buffer was attached to the channel when occlusion query counting began (the first reset occurred mid-channel); samples from this channel are not counted");
        else {
            if (g_vis_cur < 0 || g_vis_cur_pass != g_pass.serial) {
                if (g_vis_n >= VIS_SLOTS) RED_ONCE(35, "Visibility slots exhausted");
                else { g_vis_cur = (int)g_vis_n++; g_vis_cur_pass = g_pass.serial;
                    if (g_nvs == g_vscap) { g_vscap = g_vscap ? g_vscap * 2 : 64; g_vs = realloc(g_vs, g_vscap * sizeof *g_vs); }
                    g_vs[g_nvs++] = (vis_slot){(void *)CFBridgingRetain(g_vis), (uint32_t)g_vis_cur}; }
            }
            if (g_vis_cur >= 0 && g_enc_vis != g_vis_cur + 1) { enc_r_visibility(en, MTLVisibilityResultModeCounting, (NSUInteger)g_vis_cur * 8); g_enc_vis = g_vis_cur + 1; }
        }
    }
    g_stat_draw++;
    { static int tg; if (!tg) { tg = 1; const char *e = getenv("DM_TEST_GHOST"); float v[8];   // Test switch DM_TEST_GHOST=px,py,gx,gy,gw,gh,cx,cy (normalized; press point, grid, center of the drag-follow image): turns on the drag-follow image directly without going through the front end (for Mac host screenshot verification)
        if (e && sscanf(e, "%f,%f,%f,%f,%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) == 8) mtl_ui_ghost(1, v[0], v[1], &v[2], v[6], v[7]); } }
    if (dw && atomic_load(&g_ghost_on)) ghost_note(dw);
    if (enc_label_on() && !g_el_draws++ && pr->sh[1]) { memcpy(g_el_key, pr->sh[1]->key, 8); g_el_key[8] = 0; }
    if (passdump()) {   // Characteristics of a render pass's first draw (to judge whether the load can be omitted: full screen, no blending, writes all channels, does not depend on depth); with DM_NVN_PASSDUMP_ALL every draw is recorded
        static uint64_t last_pass = ~0ull; static int all = -1; if (all < 0) all = getenv("DM_NVN_PASSDUMP_ALL") != NULL;
        if (all) PD("d %llu bv=%d bi=%d first=%d vs=%s fs=%s vc=%d inst=%d blend=%#x b0=%#llx dt=%d dw=%d df=%d cull=%d st=%d sf=%d,%d sop=%d,%d,%d/%d,%d,%d smask=%#x,%#x/%#x,%#x sref=%#x,%#x ib=%#llx it=%d cmask=%#x", g_stat_pass, di ? di->base_vertex : 0, di ? di->base_instance : 0, dw ? dw->first : 0, pr->sh[0] ? pr->sh[0]->key : "-", pr->sh[1] ? pr->sh[1]->key : "-",
                    dw ? dw->count : di->count, dw ? 1 : di->instances, S.color.blend_en, (unsigned long long)*(uint64_t *)&S.blend[0], (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.df, (int)S.poly.cull,
                    (int)S.ds.st, (int)S.ds.sf_f, (int)S.ds.sf_b, (int)S.ds.fail_f, (int)S.ds.zfail_f, (int)S.ds.zpass_f,
                    (int)S.ds.fail_b, (int)S.ds.zfail_b, (int)S.ds.zpass_b, S.st_vmask[0], S.st_vmask[1], S.st_wmask[0], S.st_wmask[1], S.st_ref[0], S.st_ref[1],
                    di ? (unsigned long long)di->ibuf : 0ull, di ? di->itype : 0, S.chmask.mask);
        if (all && pr->sh[0]) {   // Content fingerprint of each constant buffer of the vertex stage (to judge whether a draw is completely identical across frames: used by the shadow cache etc.); when unreadable (a write pending execution in this submit) record P
            char ub[400]; size_t o = 0; NvShader *v = pr->sh[0];
            for (int i = 0; i < v->rf.nb && o < sizeof ub - 40; i++) if (v->rf.b[i].kind == NVN_RB_CBUF && v->rf.b[i].reg < 16) {
                const p_bind_buf *u = &S.ubo[0][v->rf.b[i].reg]; void *mb; uint64_t mo; const uint8_t *base; uint64_t h = 1469598103934665603ull;
                if (!u->addr || !u->size) { o += snprintf(ub + o, sizeof ub - o, " r%d=-", v->rf.b[i].reg); continue; }
                if (pend_hits(u->addr, u->size) || !mtl_addr(u->addr, u->size, &mb, &mo) || !(base = [(__bridge id<MTLBuffer>)mb contents])) { o += snprintf(ub + o, sizeof ub - o, " r%d=P/%llu", v->rf.b[i].reg, (unsigned long long)u->size); continue; }
                uint64_t h64 = 0; for (uint64_t q = 0; q < u->size; q++) { h = (h ^ base[mo + q]) * 1099511628211ull; if (q == 63) h64 = h; }   // h64: the first 64 bytes (world matrix vec0–3 of the object buffer)
                { static uint64_t hp = ~0ull; static int hn; if (hp != g_stat_pass) { hp = g_stat_pass; hn = 0; }   // For the first 6 draws of each render pass, additionally print buffers of <= 608 bytes in hex (to compare byte by byte which fields changed across frames)
                  if (hn < 12 && u->size <= 608 && v->rf.b[i].reg == 3) { hn++; char hx[1300]; for (uint64_t q = 0; q < u->size; q++) snprintf(hx + 2 * q, 3, "%02x", base[mo + q]); PD("x %llu r%d %s", g_stat_pass, v->rf.b[i].reg, hx); } }
                o += snprintf(ub + o, sizeof ub - o, " r%d=%016llx/%llu/%016llx", v->rf.b[i].reg, (unsigned long long)h, (unsigned long long)u->size, (unsigned long long)h64); }
            PD("u %llu%s", g_stat_pass, ub); }
        if (all) { char vb[512]; size_t o = 0; o += snprintf(vb + o, sizeof vb - o, "nattr=%d", S.nattr);
            for (int i = 0; i < S.nattr && i < 6; i++) { int st = S.attr[i].stream;
                o += snprintf(vb + o, sizeof vb - o, " a%d{fmt=%#x off=%d s=%d stride=%u div=%u vb=%#llx/%llu}", i, S.attr[i].fmt, S.attr[i].offset, st,
                              st < 16 ? S.stream[st].stride : 0, st < 16 ? S.stream[st].divisor : 0, st < 16 ? (unsigned long long)S.vbuf[st].address : 0ull, st < 16 ? (unsigned long long)S.vbuf[st].size : 0ull); }
            PD("v %llu %s", g_stat_pass, vb); }
        if (last_pass != g_stat_pass) {
            last_pass = g_stat_pass;
            PD("D %llu prim=%d vc=%d inst=%d blend=%#x wm=%#x dt=%d dw=%d df=%d st=%d sc=%lu,%lu,%lu,%lu tgt=%lux%lu vp=%.0f,%.0f,%.0f,%.0f", g_stat_pass, prim,
               dw ? dw->count : di->count, dw ? 1 : di->instances, S.color.blend_en, S.chmask.mask, (int)S.ds.dt, (int)S.ds.dw, (int)S.ds.df, (int)S.ds.st,
               (unsigned long)sc.x, (unsigned long)sc.y, (unsigned long)sc.width, (unsigned long)sc.height, (unsigned long)tw, (unsigned long)th,
               S.has_vp ? S.vp[0][0] : 0, S.has_vp ? S.vp[0][1] : 0, S.has_vp ? S.vp[0][2] : (float)tw, S.has_vp ? S.vp[0][3] : (float)th);
        }
    }
    if (rt_range_on() && !rt_range_pr_ro(pr)) g_pass.range_external=1;
    rt_range_note_depth();
#ifdef NVN_RT_RANGE_TEST
    rr_snap(dw,di);
#endif
    udc_note_draw(pr);

    // A draw with 0 primitives (the game does send count=0) draws no pixels, so skip encoding; Metal requires the count to be non-zero, otherwise the validation layer aborts on the spot (measured on the camp / wilderness / combat pages)
    if (dw && prim == 7) {   // Quads: shared indices 0,1,2,0,2,3
        if (!g_quad_ib) {
            uint32_t n = 65536; uint32_t *ix = malloc(n * 6 * 4);
            for (uint32_t q = 0; q < n; q++) { uint32_t b = q * 4; uint32_t v[6] = {b, b + 1, b + 2, b, b + 2, b + 3}; memcpy(ix + q * 6, v, 24); }
            g_quad_ib = [g_dev newBufferWithBytes:ix length:n * 24 options:MTLResourceStorageModeShared]; g_quad_gpu = g_quad_ib.gpuAddress; res_add(g_quad_ib); free(ix);
        }
        if (dw->count >= 4) enc_r_draw_indexed(en, mp, (dw->count / 4) * 6, MTLIndexTypeUInt32, g_quad_gpu, g_quad_ib.length,
                dw->instances ? dw->instances : 1, dw->first, dw->base_instance);
    } else if (dw) {
        if (dw->count) enc_r_draw_instanced(en, mp, dw->first, dw->count, dw->instances ? dw->instances : 1, dw->base_instance);
    } else {
        if (prim == 7) { RED_ONCE(20, "Indexed quads not implemented"); return; }
        if (di->itype == 0) { RED_ONCE(21, "8-bit indices not implemented"); return; }
        void *mb; uint64_t mo;
        uint64_t gpu;
        if (!mtl_gpu_addr(di->ibuf, 1, &mb, &mo, &gpu)) { RED_ONCE(22, "Index buffer is not in a memory pool"); return; }
        use_rec("buf", di->ibuf, (uint64_t)di->count * (di->itype == 1 ? 2 : 4), 0, NULL);
        if (di->count) enc_r_draw_indexed(en, mp, di->count, di->itype == 1 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32, gpu, ((__bridge id<MTLBuffer>)mb).length - mo,
                di->instances ? di->instances : 1, di->base_vertex, di->base_instance);
    }
    if (g_rt_range_pending) { PD("j rt_level_range pass=%llu vc=%u", g_stat_pass, (unsigned)(dw ? dw->count : di->count)); g_rt_range_pending=0; }
}
static nvn_hmap g_cps = {.klen = sizeof(void *)};
static void warm_claim(void) {
    if (!atomic_exchange_explicit(&g_warm_ready, 0, memory_order_acq_rel)) return;
    os_unfair_lock_lock(&g_warm_lock);
    NSArray<WarmItem *> *items = [g_warm_items copy];
    [g_warm_items removeAllObjects];
    os_unfair_lock_unlock(&g_warm_lock);
    for (WarmItem *item in items) {
        if (item->kind == 0) {
            if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
            if (!g_shcache[item->key]) { g_shcache[item->key] = item->shader; atomic_fetch_add(&g_shcache_n, 1); g_warm_claim_shaders++; }
        } else if (item->kind == 1) {
            if (!g_shcache) g_shcache = [NSMutableDictionary dictionary];
            NvShader *vs = g_shcache[item->vs_key], *fs = item->fs_key.length ? g_shcache[item->fs_key] : nil;
            if (!vs) { vs = item->vs; g_shcache[item->vs_key] = vs; atomic_fetch_add(&g_shcache_n, 1); }
            if (item->fs_key.length && !fs) { fs = item->fs; g_shcache[item->fs_key] = fs; atomic_fetch_add(&g_shcache_n, 1); }
            item->pso.vs = (__bridge void *)vs; item->pso.fs = (__bridge void *)fs;
            if (!hm_get_id(&g_pso, &item->pso)) { hm_put_id(&g_pso, &item->pso, item->object); g_warm_claim_rps++; }
        } else {
            NvShader *cs = g_shcache[item->key];
            if (!cs) { cs = item->shader; if (!g_shcache) g_shcache = [NSMutableDictionary dictionary]; g_shcache[item->key] = cs; atomic_fetch_add(&g_shcache_n, 1); }
            const void *key = (__bridge const void *)cs;
            if (!hm_get_id(&g_cps, &key)) { hm_put_id(&g_cps, &key, item->object); g_warm_claim_cps++; }
        }
    }
}
// ---------- Magenta flash probe ----------
// A16 video: under heavy load, the entire 3D image turns magenta (255,0,254), while the UI draws normally on top, in bursts every 0.35 seconds. Enabled by default (set DM_MAGENTA_PROBE_OFF=1 to disable); check logs only in internal test builds:
// For each frame, append copies of 5 points from the game's final image (center and four quarter points) and 4 palette entries to a shared buffer at the end of the present command buffer; check them in the callback when the command buffer completes, without waiting for the GPU or using a frame.
// If ≥3 of the 5 image points are magenta (R≥200, G≤60, B≥200), log 【red】 along with palette readings, brightness multiplier, and display-vsync callback stall count; log palette entries that are NaN / infinite / magenta separately as 【red】; write a summary line every 3600 frames (range).

// ---------- Magenta probe ----------
// A user video on an A16 device: under heavy load the whole 3D picture turns magenta (255,0,254) while the UI still draws on top, in bursts of about 0.35 s. On by default (DM_MAGENTA_PROBE_OFF=1 turns it off); read the logs from test builds:
// for each present, at the end of the present command buffer, copy 5 points of the game's final picture (the center and the four quarter points) and 4 cells of the color table into a shared buffer, and judge them in the completion callback — no GPU wait, no frame cost.
// If ≥3 of the 5 picture points are magenta (R≥200, G≤60, B≥200), log a red line with the color-table readings, the brightening factor and the number of display-link callback stalls; a color-table cell that is NaN / infinite / magenta gets its own red line; log a summary every 3600 presents (coverage).
// Color table = the 32³ table looked up by tone mapping (see the 3D brightening section below), recorded at dispatch 552f45c348a61b86. Only on the Metal 4 present path (iOS / Mac release builds)
static os_unfair_lock g_mp_lk = OS_UNFAIR_LOCK_INIT; static id<MTLTexture> g_mp_lut;
static MTLPixelFormat g_mp_fmt[3]; static int g_mp_haslut[3];
static _Atomic uint64_t g_mp_n, g_mp_mag, g_mp_lutbad;
static uint64_t g_stall_t0, g_stall_n;   // Defined in the display-link section
float dm_bright_k(void);   // Defined below in the 3D brightening section (weak definition; the host dm_bright.c provides the strong one)
static int mp_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("DM_MAGENTA_PROBE_OFF"); v = !(e && *e && strcmp(e, "0")); } return v; }
static int mp_fmt_ok(MTLPixelFormat f) { return f == MTLPixelFormatRGBA8Unorm || f == MTLPixelFormatRGBA8Unorm_sRGB || f == MTLPixelFormatBGRA8Unorm || f == MTLPixelFormatBGRA8Unorm_sRGB || f == MTLPixelFormatRGB10A2Unorm || f == MTLPixelFormatRGBA16Float; }
static void mp_encode(CBUF cb, id<MTLTexture> src, int i) {
#if !DM_NVN_M3
    uint64_t t_start = bp_now();   // Cost measurement: CPU time start
    g_mp_fmt[i] = 0; g_mp_haslut[i] = 0;
    if (!mp_on() || !g_mp_buf[i] || !src || src.textureType != MTLTextureType2D || !mp_fmt_ok(src.pixelFormat)) return;
    os_unfair_lock_lock(&g_mp_lk); id<MTLTexture> lut = g_mp_lut; os_unfair_lock_unlock(&g_mp_lk);
    if (lut && (lut.pixelFormat != MTLPixelFormatRGBA16Float || lut.width < 2 || lut.height < 2 || lut.depth < 2)) lut = nil;
    NSUInteger w = src.width, h = src.height, bpp = src.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : 4;
    const NSUInteger px[5][2] = {{w / 2, h / 2}, {w / 4, h / 4}, {w * 3 / 4, h / 4}, {w / 4, h * 3 / 4}, {w * 3 / 4, h * 3 / 4}};
    CENC e = [cb computeCommandEncoder];
    [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    for (int k = 0; k < 5; k++) enc_b_copy_texture_buffer(e, src, 0, 0, MTLOriginMake(px[k][0], px[k][1], 0), MTLSizeMake(1, 1, 1), g_mp_buf[i], k * 256, bpp, bpp);
    if (lut) { NSUInteger lw = lut.width - 1, lh = lut.height - 1, ld = lut.depth - 1; const NSUInteger q[4][3] = {{0, 0, 0}, {lw / 2, lh / 2, ld / 2}, {lw, lh, ld}, {lw / 2, lh / 4, ld * 3 / 4}};
        for (int k = 0; k < 4; k++) enc_b_copy_texture_buffer(e, lut, 0, 0, MTLOriginMake(q[k][0], q[k][1], q[k][2]), MTLSizeMake(1, 1, 1), g_mp_buf[i], (5 + k) * 256, 8, 8);
        g_mp_haslut[i] = 1; }
    m4_end(e);
    g_mp_fmt[i] = src.pixelFormat;

    // Cost measurement: record CPU time (ns) per call, summarize every 600 calls
    static uint64_t cost_sum, cost_n, cost_max; uint64_t cost_ns = bp_now() - t_start;
    cost_sum += cost_ns; cost_n++; if (cost_ns > cost_max) cost_max = cost_ns;
    if (cost_n % 600 == 0) { MLOG("Purple flash probe CPU cost: first %llu runs, average %.3f μs / peak %.3f μs", (unsigned long long)cost_n, cost_sum / (double)cost_n / 1000, cost_max / 1000.0); }
#else
    (void)cb; (void)src; (void)i;
#endif
}
static void mp_px(MTLPixelFormat f, const uint8_t *p, int *r, int *g, int *b) {
    if (f == MTLPixelFormatBGRA8Unorm || f == MTLPixelFormatBGRA8Unorm_sRGB) { *r = p[2]; *g = p[1]; *b = p[0]; }
    else if (f == MTLPixelFormatRGB10A2Unorm) { uint32_t v; memcpy(&v, p, 4); *r = (v & 1023) >> 2; *g = ((v >> 10) & 1023) >> 2; *b = ((v >> 20) & 1023) >> 2; }
    else if (f == MTLPixelFormatRGBA16Float) { __fp16 hv[4]; memcpy(hv, p, 8); float c[3] = {hv[0], hv[1], hv[2]}; int *o[3] = {r, g, b};
        for (int k = 0; k < 3; k++) { float x = c[k] != c[k] ? 0 : c[k] < 0 ? 0 : c[k] > 1 ? 1 : c[k]; *o[k] = (int)(x * 255 + 0.5f); } }
    else { *r = p[0]; *g = p[1]; *b = p[2]; }
}
static void mp_check(int i, uint64_t ticket) {
    MTLPixelFormat f = g_mp_fmt[i]; if (!f) return;
    const uint8_t *m = g_mp_buf[i].contents; char pt[96] = "", lt[160] = "None"; int o = 0, mag = 0, lutbad = 0;
    for (int k = 0; k < 5; k++) { int r, g, b; mp_px(f, m + k * 256, &r, &g, &b); mag += r >= 200 && g <= 60 && b >= 200; o += snprintf(pt + o, sizeof pt - o, " %02x%02x%02x", r, g, b); }
    if (g_mp_haslut[i]) { o = 0; for (int k = 0; k < 4; k++) { __fp16 hv[4]; memcpy(hv, m + (5 + k) * 256, 8); float c[3] = {hv[0], hv[1], hv[2]};
        int bad = 0; for (int j = 0; j < 3; j++) bad |= c[j] != c[j] || isinf(c[j]); bad |= c[0] > 0.8f && c[1] < 0.2f && c[2] > 0.8f; lutbad |= bad;
        o += snprintf(lt + o, sizeof lt - o, "%s(%.3f,%.3f,%.3f)", k ? " " : "", c[0], c[1], c[2]); } }
    uint64_t n = atomic_fetch_add(&g_mp_n, 1) + 1, nm = mag >= 3 ? atomic_fetch_add(&g_mp_mag, 1) + 1 : atomic_load(&g_mp_mag), nl = lutbad ? atomic_fetch_add(&g_mp_lutbad, 1) + 1 : atomic_load(&g_mp_lutbad);
    if (mag >= 3 && (nm <= 30 || nm % 50 == 0)) MLOG("[RED] Purple flash probe: rendered %llu magenta frames (5 points%s, %d magenta points); palette 4 cells %s; brightness k=%.2f; display pacing stop callbacks accumulated %llu segments, currently%s paused; cumulative magenta %llu times",
        (unsigned long long)ticket, pt, mag, lt, dm_bright_k(), (unsigned long long)g_stall_n, g_stall_t0 ? "in" : "not in", (unsigned long long)nm);
    if (lutbad && (nl <= 30 || nl % 50 == 0)) MLOG("[RED] Purple flash probe: draw %llu palette grid anomaly (non-number / infinity / magenta) %s; 5 screen points %s; total %llu times", (unsigned long long)ticket, lt, pt, (unsigned long long)nl);
    if (n % 3600 == 0 || n == 600) MLOG("Purple flash probe through draw %llu: sampled %llu times, screen magenta %llu times, palette anomaly %llu times (this time 5 points %s, palette %s)", (unsigned long long)ticket, (unsigned long long)n, (unsigned long long)nm, (unsigned long long)nl, pt, lt);
}
// ---------- 3D image brightness ----------
// Tone mapping (fragment a1b1aab66933fa8f) = PQ-encode the composited HDR ×0.008, then look up the 32³ LUT (recomputed regionally each frame by compute shader 552f45c348a61b86, read only by tone mapping).
// Multiplying HDR by exposure k is equivalent to replacing the LUT with L'(u) = L(PQ(k · PQ⁻¹(u))): after the game's dispatch that computes the LUT, copy it and rewrite it in place accordingly (one small 32³ dispatch); the LUT is recomputed each frame, so brightness does not compound.
// Affects only the 3D world: UI, text, and blood orbs are composited after tone mapping; highlights are compressed by PQ, so they will not wash out into a solid white like multiplying the final screen by a factor would. The regular / readable-tile-merge / frozen paths all read this LUT, so the change applies everywhere.

// ---------- 3D picture brightening ----------
// Tone mapping (fragment a1b1aab66933fa8f) = PQ-encode the composited HDR x0.008, then look up a 32^3 color lookup table (recomputed per region every frame by compute shader 552f45c348a61b86, read only by tone mapping).
// Multiplying HDR by exposure k first is equivalent to replacing the table with L'(u) = L(PQ(k * PQ^-1(u))): after the dispatch in which the game finishes computing the table, make a copy and rewrite it in place accordingly (one small 32^3 dispatch); the table is recomputed every frame, so it does not get brighter with each multiplication.
// Only the 3D world is touched: UI, text and the health orbs are composited after tone mapping; highlights are compressed by PQ and do not blur into a sheet of white as a final whole-screen multiply would. The normal / readable tile merge / freeze paths all just read this table, so it takes effect in one place.
// k is given every frame by the host's dm_bright_k() (dm/host/dm_bright.c: 1 out of game, 4 for maze-type areas, 4 where the original game's picture does not follow day and night, the rest derived from the original game's ambient brightness, with smooth transitions); no rewrite when k ~= 1.
__attribute__((weak)) float dm_bright_k(void) { return 1.0f; }   // The host's dm_bright.c provides the strong definition; no brightening when graphics unit tests do not link the host
static const char k_lut_msl[] =
"#include <metal_stdlib>\nusing namespace metal;\n"
"constant float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;\n"   // Bit-identical to the PQ constants in the tone mapping shader
"static float3 pq(float3 v) { float3 p = pow(abs(v), m1); return pow((c1 + c2 * p) / (1.0 + c3 * p), m2); }\n"
"static float3 pq_inv(float3 u) { float3 q = pow(u, 1.0 / m2); return pow(max(q - c1, 0.0) / max(c2 - c3 * q, 1e-6), 1.0 / m1); }\n"
"kernel void lutk(texture3d<float> src [[texture(0)]], texture3d<float, access::write> dst [[texture(1)]], constant float &k [[buffer(0)]], uint3 g [[thread_position_in_grid]]) {\n"
"  uint3 n = uint3(dst.get_width(), dst.get_height(), dst.get_depth()); if (any(g >= n)) return;\n"
"  constexpr sampler ls(filter::linear, address::clamp_to_edge, coord::normalized);\n"
"  float3 u = (float3(g) + 0.5) / float3(n); float3 u2 = clamp(pq(k * pq_inv(u)), 0.0, 1.0);\n"
"  float4 b = src.sample(ls, u2), o = src.read(g);\n"   // b: output after brightening by exposure k; o: this cell in the original game
"  float s = smoothstep(0.65, 1.0, dot(b.rgb, float3(0.2126, 0.7152, 0.0722)));\n"   // Highlight protection: after brightening, the closer to white, the more it is pulled back to the original value (snow and skylight do not blur into a sheet of white); in dark areas s = 0 and brightening proceeds as usual
"  dst.write(float4(max(mix(b.rgb, o.rgb, s), o.rgb), b.a), g); }\n";
static void dispatch(const p_dispatch *dp) {
    zskip_clear();
    if (!S.prog) {
        RED_ONCE(27, "No program bound before dispatch");
        return;
    }
    __unsafe_unretained NvProg *pr = prog_get(S.prog);
    if (!pr) return;
    NvShader *cs = pr->sh[5];
    if (!cs) { RED_ONCE(28, "The dispatched program has no compute shader (or is unmapped)"); return; }
    const void *k = (__bridge const void *)cs;
    id<MTLComputePipelineState> ps = hm_get_id(&g_cps, &k);
    if (!ps) {
        NSError *e = nil; uint64_t bt = bp_now(); ps = [g_dev newComputePipelineStateWithFunction:cs->fn error:&e];
        g_bp_pso_n++; g_bp_pso_ns += bp_now() - bt;
        if (!ps) { MLOG("[RED] Failed to create compute pipeline: %s", e.localizedDescription.UTF8String); return; }
        pso_record("C\t%s\n", cs->key);
        if (cs->key[0]) pso_user_record("C\t%s\n", cs->key);
        hm_put_id(&g_cps, &k, ps);
    }
    if (cs->builtin) {   // Built-in compute kernel: n = uniform buffer 0, src = storage buffer 0, dst = storage buffer 1
        const p_bind_buf *u = &S.ubo[5][0], *a0 = &S.ssbo[5][0], *a1 = &S.ssbo[5][1];
        void *ub, *sb, *db; uint64_t uo, so, dof;
        if (!mtl_addr(u->addr, 4, &ub, &uo) || !mtl_addr(a0->addr, 8, &sb, &so) || !mtl_addr(a1->addr, 8, &db, &dof)) { RED_ONCE(30, "Not all buffers are bound for embedded compute kernel %s", cs->builtin); return; }
        if (!cs->logged++) MLOG("First dispatch of embedded compute kernel %s: call site %#llx, thread groups %dx%dx%d, element count (current CPU-side value) %u",
                                cs->builtin, (unsigned long long)dp->caller, dp->x, dp->y, dp->z, *(uint32_t *)u->addr);
        rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
        id<MTLComputeCommandEncoder> ce = pg_compute();
        enc_c_pso(ce, ps);
        enc_c_args(ce, ((__bridge id<MTLBuffer>)ub).gpuAddress + uo, 0); enc_c_args(ce, ((__bridge id<MTLBuffer>)sb).gpuAddress + so, 1); enc_c_args(ce, ((__bridge id<MTLBuffer>)db).gpuAddress + dof, 2);   // Built-in compute kernel: cold path, take the addresses directly
        enc_c_dispatch(ce, MTLSizeMake(dp->x, dp->y, dp->z), MTLSizeMake(32, 1, 1));
        pg_compute_end(ce); enc_c_end(ce);
        return;
    }
    const uint16_t *tg = cs->rf.tg;
    if (!tg[0]) { RED_ONCE(29, "Compute shader is missing thread group size reflection"); return; }
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    id<MTLComputeCommandEncoder> ce = pg_compute();
    enc_c_pso(ce, ps);
    encode_args(ce, cs, 5, 0);
    enc_c_dispatch(ce, MTLSizeMake(dp->x, dp->y, dp->z), MTLSizeMake(tg[0], tg[1], tg[2]));
    pg_compute_end(ce); enc_c_end(ce);
}

// GPU: texture -> buffer (row stride taken from SetCopyRowStride, 0 = tightly packed)
static void copy_t2b(const p_copy_b2t *c) {
    if (c->tex) xv_check(c->tex->serial, c->has_view && c->view.set_levels ? c->view.base_level : 0, c->has_view && c->view.set_layers ? c->view.min_layer : 0, "Texture read buffer");
    id<MTLTexture> t = tex_mtl(c->tex);
    if (nvn_draw_trace_detail_on()) MLOG("[DEBUG-RA-DRAW] event=copy phase=begin kind=t2b frame=%llu src=#%llu(%dx%d/fmt=%#x) buffer=%#llx",
         (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)(c->tex ? c->tex->serial : 0),
         c->r.width, c->r.height, c->tex ? (unsigned)c->tex->d.format : 0u, (unsigned long long)c->src);
    if (!t) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh);
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row, img, extent;
    if (!nvn_copy_pitch(c->r.width, c->r.height, c->r.depth, bw, bh, bpb,
                        S.copy_stride, S.copy_image_stride, &row, &img, &extent)) {
        MLOG("Invalid texture copy pitch"); return;
    }
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    void *mb; uint64_t mo;
    if (!mtl_addr(c->src, extent, &mb, &mo)) { RED_ONCE(31, "Destination of texture→buffer copy is not in the memory pool"); return; }
    BENC b = blit();
    if (is3d)
        enc_b_copy_texture_buffer(b, t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset), MTLSizeMake(c->r.width, c->r.height, nz), (__bridge id<MTLBuffer>)mb, mo, row, img);
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_texture_buffer(b, t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0), MTLSizeMake(c->r.width, c->r.height, 1), (__bridge id<MTLBuffer>)mb, mo + z * img, row, img);
    if (nvn_draw_trace_detail_on()) MLOG("[DEBUG-RA-DRAW] event=copy phase=queued kind=t2b frame=%llu src=#%llu buffer=%#llx",
         (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)c->tex->serial, (unsigned long long)c->src);
}

// CPU texture read (ReadTexels): for packaged textures the pool memory is the ground truth, de-layout and read directly; the rest are read back from the MTLTexture by blit (the caller has already waited for the GPU per NVN rules)
void mtl_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    if (!on_sub_thread()) while (atomic_load(&g_up_pending)) usleep(200);   // Translation runs on the translation thread: the caller has already drained the queue (batches were flushed with the submit), so here we only wait for uploads to finish on the GPU (on the submit thread q_cb flushes and waits)
    xv_check(t->serial, level, layer0, "CPU readback");
    PD("U cpu_read ser=%llu lv%u sl%u xyz=%d,%d,%d size=%d,%d,%d dst=%p row=%lu img=%lu", t->serial, level, layer0, r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth, dst, (unsigned long)row, (unsigned long)img);
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (r->height + bh - 1) / bh, x0 = (size_t)(r->xoffset / bw) * bpb, y0 = r->yoffset / bh;
    int nz = r->depth ? r->depth : 1, is3d = t->d.target == 2;
    uint8_t *mem = tex_storage(t);
    if (tex_is_packaged(t) && mem) {
        uint32_t lw = MAX(t->d.width >> level, 1u), lh = MAX(t->d.height >> level, 1u), lz = is3d ? MAX(t->d.depth >> level, 1u) : 1;
        size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
        uint8_t *lin = malloc(lp * lr * lz);
        for (int z = 0; z < nz; z++) {
            unsigned layer = is3d ? 0 : layer0 + r->zoffset + z;
            if ((is3d && z == 0) || !is3d) if (nvn_deswizzle(&t->d, level, layer, mem, lin, lp)) { RED_ONCE(32, "ReadTexels layout decoding failed"); free(lin); return; }
            const uint8_t *src = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0);
            for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, src + (y0 + y) * lp + x0, wb);
        }
        free(lin);
        return;
    }
    id<MTLTexture> mt = tex_mtl(t);
    if (!mt) { memset(dst, 0, img * nz); return; }
    mtl_finish_gpu();
    id<MTLBuffer> buf = [g_dev newBufferWithLength:wb * hb * nz options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = enc_b_begin(cb);
    for (int z = 0; z < nz; z++)
        [b copyFromTexture:mt sourceSlice:is3d ? 0 : layer0 + r->zoffset + z sourceLevel:level sourceOrigin:MTLOriginMake(r->xoffset, r->yoffset, is3d ? r->zoffset + z : 0)
                sourceSize:MTLSizeMake(r->width, r->height, 1) toBuffer:buf destinationOffset:z * wb * hb destinationBytesPerRow:wb destinationBytesPerImage:wb * hb];
    enc_b_end(b); [cb commit]; [cb waitUntilCompleted];
    for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, (uint8_t *)buf.contents + z * wb * hb + y * wb, wb);
}
static void copy_b2t(const p_copy_b2t *c) {
    if (c->tex) zskip_del(c->tex->serial);
    id<MTLTexture> t = tex_mtl(c->tex);
    PD("C ser=%llu", c->tex->serial);
    if (nvn_draw_trace_detail_on()) MLOG("[DEBUG-RA-DRAW] event=copy phase=begin kind=b2t frame=%llu dst=#%llu(%dx%d/fmt=%#x) buffer=%#llx",
         (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)(c->tex ? c->tex->serial : 0),
         c->r.width, c->r.height, c->tex ? (unsigned)c->tex->d.format : 0u, (unsigned long long)c->src);
    if (!t) return;
    int bw, bh, bpb = nvn_fmt_bpb(c->tex->d.format, &bw, &bh);
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    size_t row, img, extent;
    if (!nvn_copy_pitch(c->r.width, c->r.height, c->r.depth, bw, bh, bpb,
                        S.copy_stride, S.copy_image_stride, &row, &img, &extent)) {
        MLOG("Invalid texture copy pitch"); return;
    }
    int is3d = c->tex->d.target == 2, nz = c->r.depth ? c->r.depth : 1;
    void *mb; uint64_t mo;
    if (!mtl_addr(c->src, extent, &mb, &mo)) { RED_ONCE(23, "Source of buffer→texture copy is not in the memory pool"); return; }
    BENC b = blit();
    if (is3d)
        enc_b_copy_buffer_texture(b, (__bridge id<MTLBuffer>)mb, mo, row, img, MTLSizeMake(c->r.width, c->r.height, nz), t, 0, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, c->r.zoffset));
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_buffer_texture(b, (__bridge id<MTLBuffer>)mb, mo + z * img, row, img, MTLSizeMake(c->r.width, c->r.height, 1), t, layer0 + c->r.zoffset + z, level, MTLOriginMake(c->r.xoffset, c->r.yoffset, 0));
    if (nvn_draw_trace_detail_on()) MLOG("[DEBUG-RA-DRAW] event=copy phase=queued kind=b2t frame=%llu dst=#%llu buffer=%#llx",
         (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)c->tex->serial, (unsigned long long)c->src);
}


// GPU: texture -> texture (previously mislisted as a no-op, 3 times per frame on the character select page). Same size, not mirrored, same format -> blit layer by layer; the other semantics (scaling with filtering, mirroring, format change) cannot be expressed by blit: report a failure, to be implemented
static void copy_t2t(const p_copy_t2t *c) {
    if (c->src) xv_check(c->src->serial, c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, "Texture copy source");
    rp_flush();   // Land the pending pure-clear render pass first (zero-source candidates only enter the set when the render pass ends); the blit() below would end the render pass anyway
    if (c->dst) zskip_del(c->dst->serial);
    if (c->src && c->dst && c->sr.xoffset == 0 && c->sr.yoffset == 0 && c->dr.xoffset == 0 && c->dr.yoffset == 0
        && c->sr.width == (int)c->src->d.width && c->sr.height == (int)c->src->d.height && c->dr.width == (int)c->dst->d.width && c->dr.height == (int)c->dst->d.height) {
        int sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
        int ss = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, ds = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
        if (zskip_has(c->src->serial, sl, ss)) zskip_add(c->dst->serial, dl, ds);
    }
    id<MTLTexture> s = tex_mtl(c->src), d = tex_mtl(c->dst);
    PD("T src=%llu dst=%llu %dx%dx%d→%dx%dx%d flags=%#x", c->src->serial, c->dst->serial, c->sr.width, c->sr.height, c->sr.depth, c->dr.width, c->dr.height, c->dr.depth, c->flags);
    if (nvn_draw_trace_detail_on()) MLOG("[DEBUG-RA-DRAW] event=copy phase=begin kind=t2t frame=%llu src=#%llu dst=#%llu region=%dx%dx%d fmt=%lu->%lu",
         (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)c->src->serial,
         (unsigned long long)c->dst->serial, c->sr.width, c->sr.height, c->sr.depth,
         s ? (unsigned long)s.pixelFormat : 0ul, d ? (unsigned long)d.pixelFormat : 0ul);
    if (!s || !d) return;
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    PD("t src=%llu lv%u sl%u xyz=%d,%d,%d dst=%llu lv%u sl%u xyz=%d,%d,%d size=%d,%d,%d", c->src->serial, sl, sz0, c->sr.xoffset, c->sr.yoffset, c->sr.zoffset,
       c->dst->serial, dl, dz0, c->dr.xoffset, c->dr.yoffset, c->dr.zoffset, c->sr.width, c->sr.height, c->sr.depth);
    int s3 = c->src->d.target == 2, d3 = c->dst->d.target == 2, nz = c->sr.depth ? c->sr.depth : 1;
    if (c->sr.width != c->dr.width || c->sr.height != c->dr.height || (c->sr.depth ? c->sr.depth : 1) != (c->dr.depth ? c->dr.depth : 1) || (c->flags & ~NVN_COPY_FLAGS_ENGINE_2D_BIT & ~NVN_COPY_FLAGS_LINEAR_FILTER_BIT)
        || s.pixelFormat != d.pixelFormat || s3 != d3) {
        RED_ONCE(33, "Unimplemented texture→texture copy form: %dx%dx%d→%dx%dx%d flags %#x formats %lu→%lu 3D %d→%d", c->sr.width, c->sr.height, c->sr.depth, c->dr.width, c->dr.height, c->dr.depth,
                 c->flags, (unsigned long)s.pixelFormat, (unsigned long)d.pixelFormat, s3, d3);
        return;
    }
    BENC b = blit();
    if (s3)
        enc_b_copy_texture(b, s, 0, sl, MTLOriginMake(c->sr.xoffset, c->sr.yoffset, c->sr.zoffset), MTLSizeMake(c->sr.width, c->sr.height, nz), d, 0, dl, MTLOriginMake(c->dr.xoffset, c->dr.yoffset, c->dr.zoffset));
    else
        for (int z = 0; z < nz; z++)
            enc_b_copy_texture(b, s, sz0 + c->sr.zoffset + z, sl, MTLOriginMake(c->sr.xoffset, c->sr.yoffset, 0), MTLSizeMake(c->sr.width, c->sr.height, 1), d, dz0 + c->dr.zoffset + z, dl, MTLOriginMake(c->dr.xoffset, c->dr.yoffset, 0));
    if (nvn_draw_trace_detail_on()) MLOG("[DEBUG-RA-DRAW] event=copy phase=queued kind=t2t backend=blit frame=%llu src=#%llu dst=#%llu",
         (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)c->src->serial, (unsigned long long)c->dst->serial);
}

// GPU: clear a texture region to the given value (previously incorrectly listed as a no-op; called 5 times during initialization; initial contents of Metal private textures are undefined, and the game relies on the cleared initial values).
// Entire mip level and face, all four channels written → open a clear-only pass for each layer / depth slice (hardware converts the clear value according to the texture format, including integer and depth formats);

// GPU: clear a texture region with the given value (previously mislisted as a no-op, called 5 times at initialization; the initial value of a Metal private texture is undefined, and the game relies on the cleared initial value).
// Whole level and whole face, all four channels written -> open one clear-only render pass per layer / depth slice (the hardware converts the clear value by the texture format, including integer and depth);
// view with changed format: clear on a view of that format; partial region, partial channels, compressed / packaged textures (no render target usage): report a failure, to be implemented
static void clear_tex(const p_clear_tex *c) {
    if (c->tex) zskip_del(c->tex->serial);
    id<MTLTexture> t = tex_mtl(c->tex);
    PD("X ser=%llu view format %#x region %d,%d,%d %dx%dx%d mask %#x %s %g,%g,%g,%g", c->tex->serial, c->has_view && c->view.set_format ? c->view.format : -1, c->r.xoffset, c->r.yoffset, c->r.zoffset, c->r.width, c->r.height, c->r.depth, c->mask,
       c->is_uint ? "ui" : "f", c->is_uint ? (double)c->c.u[0] : c->c.f[0], c->is_uint ? (double)c->c.u[1] : c->c.f[1], c->is_uint ? (double)c->c.u[2] : c->c.f[2], c->is_uint ? (double)c->c.u[3] : c->c.f[3]);
    if (!t) return;
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u);
    int is3d = t.textureType == MTLTextureType3D, nz = c->r.depth ? c->r.depth : 1;
    if (c->has_view && c->view.set_format) {   // View with changed format: the clear value is interpreted in the view format (all 5 of the game's calls are the ui version with an integer view), and the clear is done on a view of the same format
        uint32_t vpf = 0; uint8_t swz[4];
        if (!nvn_fmt_to_mtl(c->view.format, &vpf, swz) || !vpf) { RED_ONCE(42, "No matching view for clear texture format %#x", c->view.format); return; }
        if (tex_need_pfv(t.pixelFormat, (MTLPixelFormat)vpf) && !(t.usage & MTLTextureUsagePixelFormatView)) t = tex_upgrade_pfv(c->tex, t, (MTLPixelFormat)vpf);
        if (tex_need_pfv(t.pixelFormat, (MTLPixelFormat)vpf)) {
            id<MTLTexture> v = [t newTextureViewWithPixelFormat:(MTLPixelFormat)vpf];
            if (!v) { RED_ONCE(43, "Cannot create view for clear texture: format %lu → %u", (unsigned long)t.pixelFormat, vpf); return; }
            t = v;
        }
    }
    MTLPixelFormat pf = t.pixelFormat;
    int depth = pf == MTLPixelFormatDepth32Float || pf == MTLPixelFormatDepth16Unorm || pf == MTLPixelFormatDepth32Float_Stencil8, stencil = pf == MTLPixelFormatStencil8 || pf == MTLPixelFormatDepth32Float_Stencil8;
    if (c->r.xoffset || c->r.yoffset || (NSUInteger)c->r.width != w || (NSUInteger)c->r.height != h || (c->mask & 0xf) != 0xf
        || !(t.usage & MTLTextureUsageRenderTarget)) {
        RED_ONCE(44, "Unimplemented clear texture form: region %d,%d %dx%d (mip level %lux%lu) mask %#x renderable target %d", c->r.xoffset, c->r.yoffset, c->r.width, c->r.height,
                 (unsigned long)w, (unsigned long)h, c->mask, !!(t.usage & MTLTextureUsageRenderTarget));
        return;
    }
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    for (int z = 0; z < nz; z++) {
        RPD *rp = [RPD new];
        MTLRenderPassAttachmentDescriptor *a = depth ? rp.depthAttachment : stencil ? rp.stencilAttachment : rp.colorAttachments[0];
        a.texture = t; a.level = lv; a.loadAction = MTLLoadActionClear; a.storeAction = MTLStoreActionStore;
        if (is3d) a.depthPlane = c->r.zoffset + z; else a.slice = l0 + c->r.zoffset + z;
        if (depth) { rp.depthAttachment.clearDepth = c->c.f[0];
            if (stencil) { rp.stencilAttachment.texture = t; rp.stencilAttachment.level = lv; rp.stencilAttachment.slice = a.slice; rp.stencilAttachment.loadAction = MTLLoadActionClear;
                           rp.stencilAttachment.storeAction = MTLStoreActionStore; rp.stencilAttachment.clearStencil = c->c.u[1]; } }
        else if (stencil) rp.stencilAttachment.clearStencil = c->c.u[0];
        else rp.colorAttachments[0].clearColor = c->is_uint ? MTLClearColorMake(c->c.u[0], c->c.u[1], c->c.u[2], c->c.u[3]) : MTLClearColorMake(c->c.f[0], c->c.f[1], c->c.f[2], c->c.f[3]);
        ser_gate(); enc_r_end(enc_r_begin(rp));
    }
}


// Context description of one submit (mtl_sub_commit takes one per submit and prints it into the failure line when the command buffer errors): for the GPU memory fault on the first frame entering Travincal, the failure line had only the sequence number and error code, and it was impossible to tell what was in that submit
static const char *cctx_snap(void) {
    static char b[320]; static uint64_t p_pass, p_up, p_reup, p_ser; uint64_t up = 0, reup = 0; mtl_pkg_stats(&up, &reup);
    pthread_mutex_lock(&g_deferred_mu); unsigned long nd = (unsigned long)g_deferred.count; pthread_mutex_unlock(&g_deferred_mu);
    snprintf(b, sizeof b, "Frame %llu; since last submission: channels +%llu, first texture upload +%llu, retransmissions +%llu, sequence table changes +%llu; %lu items in deferred release queue",
             (unsigned long long)atomic_load(&g_cur_frame), (unsigned long long)(g_stat_pass - p_pass), (unsigned long long)(up - p_up), (unsigned long long)(reup - p_reup),
             (unsigned long long)(g_ser_gen - p_ser), nd);
    p_pass = g_stat_pass; p_up = up; p_reup = reup; p_ser = g_ser_gen; return b;
}

#if DM_NVN_M3
// g_sub_mu serializes replay onto one Metal queue. Queue order satisfies only prior fences; a future same-queue wait could deadlock.
static int command_buffer_wait_is_ordered(const sync_t *s) {
    if (!s || s->magic != MAGIC('S', 'Y', 'N', 'C') ||
        (s->fenced != SYNC_FENCE_QUEUE && s->fenced != SYNC_FENCE_ACQUIRE &&
         s->fenced != SYNC_FENCE_COMMAND_BUFFER)) return 0;
    uint64_t committed = atomic_load_explicit(&g_fence, memory_order_acquire);
    if (s->fence <= committed) return !mtl_fence_failed(s->fence);
    return s->fenced == SYNC_FENCE_COMMAND_BUFFER && s->fence == committed + 1;
}
static void command_buffer_wait_fail(const sync_t *s) {
    g_sub_sync_failed = 1;
    g_exec_cur = g_exec_next = NULL;
    g_rest = NULL; g_nrest = 0;
    if (!s || s->magic != MAGIC('S', 'Y', 'N', 'C'))
        MLOG("[RED] CommandBufferWaitSync received an uninitialized sync; the current submission failed");
    else if (!s->fenced)
        MLOG("[RED] CommandBufferWaitSync cannot wait on an unfenced sync; the current submission failed");
    else if (s->fenced != SYNC_FENCE_QUEUE && s->fenced != SYNC_FENCE_ACQUIRE && s->fenced != SYNC_FENCE_COMMAND_BUFFER)
        MLOG("[RED] CommandBufferWaitSync received an invalid fence state; the current submission failed");
    else if (s->fence <= g_fence && mtl_fence_failed(s->fence))
        MLOG("[RED] CommandBufferWaitSync references a failed fence; the current submission failed");
    else
        MLOG("[RED] CommandBufferWaitSync references a future fence not ordered before this point; the current submission failed");
}
#endif
void mtl_exec(const uint8_t *p) {
    static int inited; if (!inited) { inited = 1; state_reset(); }
    if (g_sub_sync_failed) { g_exec_next = g_exec_cur = NULL; return; }
    static int dump_all = -1; if (dump_all < 0) dump_all = getenv("DM_NVN_PASSDUMP_ALL") != NULL;
    int trace = dump_all && passdump();   // Normal mode does not check the frame range per operation; the full command position is only used for the designated diagnostic frames
    for (;;) {
        if (g_sub_sync_failed) { g_exec_next = g_exec_cur = NULL; return; }
        op_hdr h; memcpy(&h, p, sizeof h);
        g_exec_cur = p;   // Scan start point of the store lookahead: the operation that ends the render pass (inclusive)
        if (p == g_defer_pend) { for (int gi = 0; gi < 4; gi++) if (g_defer_slots[gi].active && g_defer_slots[gi].p_end_op == p) g_defer_slots[gi].in_p = 0; g_defer_pend = NULL; }
        sdx_seq(p);
        if (trace) PD("o cmd=%p pass=%llu op=%u size=%u", p, g_stat_pass, (unsigned)h.op, (unsigned)h.size);
        if (g_hoist_n && hoist_take(p)) { if (g_fz.active && p == g_fz.rt2_op) { g_fz.phase = 2; g_rt_merged = 0; PD("F %llu merged channel enters the draw of the subsequent channel", g_stat_pass); } p += h.size; continue; }   // A pure-clear render pass already executed early / the start of a later pass already merged in   // Entering the later pass: the same-target continuation of the earlier pass ends here; the later pass reading the earlier pass's attachment is proven to be same-pixel by fz_p2_draw, and is no longer split by the continuation rule
        const void *a = p + 8;
        switch (h.op) {
            case OP_END: g_exec_cur = NULL;
                if (g_hoist_n && !g_ret_n && !g_nrest) {
                    if (g_fz.active) fz_unskip();
                    if (g_hoist_n) { MLOG("[RED] Pure-clear channel precondition: %d preconditioned operations remain unprocessed at submission end; clearing (should not happen)", g_hoist_n); g_hoist_n = 0; }
                }
                return;   // An end-of-pass triggered outside a list (present / CPU readback etc.) must not scan expired command memory: a null start point means store everything
            case OP_BEGIN: break;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8);
                if (g_ret_n < 16) g_ret[g_ret_n] = p + h.size; g_ret_n++;   // Return stack for lookahead: after a sub-list ends, continue looking at the caller
                mtl_exec((const uint8_t *)t); g_ret_n--; break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, (const uint8_t *)a + 8, 4); memcpy(&v, (const uint8_t *)a + 12, 4);
                PD("Y signal_event cmd=%p dst=%p mode=%u value=%u", p, val, m, v); mtl_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, (const uint8_t *)a + 8, 8); g_cnt_report[t & 31]++;
                PD("Q report type=%llu cmd=%p dst=%#llx", (unsigned long long)t, p, (unsigned long long)ad);
                if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) { post((post_op){3, NULL, 0, 0, ad, (uint32_t)t, vis_snapshot()}); g_vis_cur = -1; }
                else mtl_sub_counter(ad, (uint32_t)t);
                break; }
            case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); g_cnt_reset[t & 31]++; PD("Q reset type=%llu cmd=%p", (unsigned long long)t, p);
                if (t == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
#ifdef NVN_RT_RANGE_TEST
                    rr_reset();
#endif
                    g_vis_on = 1; g_vis_counting = 1; vis_clear(); }
                break; }
            case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, (const uint8_t *)a + 8, 8); memcpy(&n, (const uint8_t *)a + 16, 8);
                PD("Y copy_b2b cmd=%p src=%#llx dst=%#llx bytes=%llu", p, (unsigned long long)s_, (unsigned long long)d, (unsigned long long)n); mtl_sub_copy(s_, d, n); break; }
            case OP_CLEAR_BUF: { const p_clear_buf *c = a;
                PD("Y clear_buffer cmd=%p dst=%#llx bytes=%llu value=%#x", p, (unsigned long long)c->addr, (unsigned long long)c->size, c->value);
                mtl_sub_fill(c->addr, c->size, c->value); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, (const uint8_t *)a + 8, 8);
                PD("Y update_ub cmd=%p dst=%#llx bytes=%llu", p, (unsigned long long)d, (unsigned long long)n); mtl_sub_update((const uint8_t *)a + 16, d, n); break; }
            case OP_SET_RT: {
                const p_set_rt *r = a;
                if (defer_begin_q(p)) break;
                int pend = R.clear_d || R.clear_s; for (int i = 0; i < 8; i++) pend |= R.clear_c[i];
                if (g_enc && !pend && R.has_rt && tbdr_on() && rt_range_same(&R.rt,r)) {
                    g_rt_merged = g_rt_range_active = g_rt_range_pending = 1; R.rt = *r; break; }
                if (g_enc && g_fz.active && g_fz.phase == 1 && fz_reuse_op(p) && !pend && g_enc_mode != 1 && R.has_rt && tbdr_on() && rt_same(&g_fz.rt1, r)) {
                    g_rt_merged = 1; if (!g_rt_merge_n++) MLOG("Same-target reuse: first instance, channel %llu", g_stat_pass); PD("M %llu same-target reuse channel", g_stat_pass); break; }

                // Do not continue a render pass opened only for a clear; one with fewer depth / stencil attachments can be continued (check the planes before drawing)
                if (g_enc && g_enc_mode != 1 && !pend && R.has_rt && tbdr_on() && rt_same(&R.rt, r)) { g_rt_merged = 1; if (!g_rt_merge_n++) MLOG("Same-target reuse: first instance, channel %llu", g_stat_pass); PD("M %llu same-target reuse channel", g_stat_pass); R.rt = *r; break; }
                rp_flush(); R.rt = *r; R.has_rt = 1; break;
            }
            case OP_CLEAR_COLOR: {
                const p_clear_color *c = a;
                if (g_rt_merged) rp_end();   // A clear arrived in a continued render pass: end it here, and the clear is attached to the start of the next render pass as usual
                if (c->index < 0 || c->index >= R.rt.n || !R.rt.color[c->index]) { RED_ONCE(5, "Clear color target %d is invalid", c->index); break; }
                if (g_enc || c->mask != 0xf || !scissor_full(R.rt.color[c->index], &R.rt.cview[c->index], R.rt.has_cview[c->index])) { RED_ONCE(6, "Mid-channel clear of color with partial mask / partial scissor is not implemented (first mask %#x)", c->mask); break; }
                R.clear_c[c->index] = 1; R.cc[c->index] = MTLClearColorMake(c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]);
                break;
            }
            case OP_CLEAR_DS: {
                const p_clear_ds *c = a;
                if (g_rt_merged) rp_end();
                if (g_enc) { RED_ONCE(7, "Mid-channel depth-stencil clear is not implemented"); break; }
                if (c->depth_mask) { R.clear_d = 1; R.cd = c->depth; }
                if (c->stencil_mask) { R.clear_s = 1; R.cs = (uint32_t)c->stencil; }
                break;
            }
            case OP_DRAW: g_exec_next = p + h.size; draw(a, NULL); g_exec_next = NULL; break;   // Lookahead start point
            case OP_DRAW_IDX: g_exec_next = p + h.size; draw(NULL, a); g_exec_next = NULL; break;

            // Do not mark dirty if the state did not change (the game often re-sends the same program / state block / vertex format before each draw; the dirty flag only decides whether to re-fetch the pipeline / depth-stencil object, and equal values mean the same object)
            case OP_SCISSOR: case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF: case OP_BIND_VBUF1:
            case OP_BIND_UBO: case OP_BIND_UBO_SMALL: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_TEX1: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_STENCIL:
            case OP_POLY_OFFSET: case OP_VIEWPORTS: case OP_DEPTH_RANGE: case OP_BLEND_COLOR: case OP_COPY_STRIDE:
                state_op(h.op, a); break;
            case OP_COPY_B2T: copy_b2t(a); break;
            case OP_COPY_T2B: copy_t2b(a); break;
            case OP_COPY_T2T: copy_t2t(a); break;
            case OP_CLEAR_TEX: clear_tex(a); break;
            case OP_DISPATCH: dispatch(a); break;
            case OP_FENCE_SYNC: {
                sync_t *s; memcpy(&s, a, 8);
                if (!s || s->magic != MAGIC('S', 'Y', 'N', 'C')) {
                    MLOG("[RED] CommandBufferFenceSync received an uninitialized sync; the current submission failed");
                    g_sub_sync_failed = 1; g_exec_next = g_exec_cur = NULL; g_rest = NULL; g_nrest = 0; return;
                }
                s->fence = atomic_load_explicit(&g_fence, memory_order_relaxed) + 1; s->fenced = SYNC_FENCE_COMMAND_BUFFER;   // Distinguish this recorded fence from a future queue-fence snapshot.
                break;
            }
            // Ordered waits need no Metal event; unresolved or future waits fail closed instead of succeeding as no-ops.
            case OP_WAIT_SYNC: {
                const sync_t *s; memcpy(&s, a, 8);
                if (!command_buffer_wait_is_ordered(s)) { command_buffer_wait_fail(s); return; }
                break;
            }
            case OP_BARRIER: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: break;
            default: break;
        }
        p += h.size;
    }
}


// ---------- Present: draw the swapchain texture to the CAMetalLayer; optional screenshot ----------
static id<MTLRenderPipelineState> g_blit_pso; static MTLPixelFormat g_blit_fmt;
static id<MTLSamplerState> g_blit_samp;
static uint64_t g_present_n;
uint64_t mtl_present_count(void) { return g_present_n; }   // Actual present count (heat attribution DM_HEAT computes the present frame rate)
void mtl_res_snapshot(mtl_res_count *out) {
    memset(out, 0, sizeof *out);
    out->pso = g_pso.n;
    out->cps = g_cps.n;
    out->dss = g_dss.n;
    out->shaders = atomic_load(&g_shcache_n);   // Resource counts are taken on the host thread while replay changes the tables on the translation thread
    os_unfair_lock_lock(&g_c1_lock);
    out->c1_bufs = (uint32_t)(g_c1_banks.count + g_c1_exclusive); out->c1_alloc = g_c1_alloc_bytes; out->c1_data = g_c1_data_bytes;
    os_unfair_lock_unlock(&g_c1_lock);
    pthread_mutex_lock(&g_smu);
    out->live_tex = g_by_serial.n;
    pthread_mutex_unlock(&g_smu);
    pthread_mutex_lock(&g_mu);
    out->pools = (uint32_t)g_npool;
    out->pool_adds = g_pool_adds;
    pthread_mutex_unlock(&g_mu);
    pthread_mutex_lock(&g_rmu);
    out->view_slots = (uint32_t)g_ntreg;
    for (int i = 0; i < g_ntreg; i++) if (g_treg[i].view) out->view_nonnull++;
    for (int i = 0; i < g_nsreg; i++) if (g_sview[i]) out->sview_nonnull++;
    pthread_mutex_unlock(&g_rmu);
}
// Resource counts (long-duration stability, read by L3): the host calls on game ticks / events (every 250 ticks, after loading-screen dismissal, on exit), not sampled by wall-clock time / frame count.

// Resource counts (read by long-run stability L3): the host calls it per game tick / event (every 250 ticks, loading screen dismissed, exit), not sampled by wall clock / present count.
// Line format: [nvn] resource counts tick <tick number> <trigger>: <name> <value> ... ([nvn] resource counts tick <tick number> <trigger>: <name> <value> ...) (names contain no spaces, single-space separated); new fields are appended to the end of this line
void mtl_res_counts(uint32_t tick, const char *why) {
    mtl_res_count c;
    mtl_res_snapshot(&c);
    int dl, du; uint64_t da; nvn_dirty_slot_stats(&dl, &du, &da);
    if (g_flag.dbg) {
        unsigned programs = 0, packaged = 0, textures = 0;
        uint64_t serial_min = UINT64_MAX, serial_max = 0;
        pthread_mutex_lock(&g_smu);
        for (uint32_t i = 0; i < g_by_serial.cap; i++) if (g_by_serial.vals[i]) {
            uint64_t serial; memcpy(&serial, g_by_serial.keys + (size_t)i * g_by_serial.klen, 8);
            if (serial < serial_min) serial_min = serial;
            if (serial > serial_max) serial_max = serial;
            id object = (__bridge id)g_by_serial.vals[i];
            if ([object isKindOfClass:[NvProg class]]) programs++;
            else if ([object isKindOfClass:[NvPkg class]]) packaged++;
            else textures++;
        }
        pthread_mutex_unlock(&g_smu);
        LOG("[DEBUG-UE-MEM] serial_programs=%u serial_packaged=%u serial_textures=%u serial_min=%#llx serial_max=%#llx",
            programs, packaged, textures, (unsigned long long)serial_min, (unsigned long long)serial_max);
    }
    LOG("[nvn] Resource counts tick %u %s: render pipelines %u compute pipelines %u shaders %u depth-stencil %u non-empty views %u view slots %u live textures %u live memory pools %u cumulative memory pools %llu non-empty sampler views %u live dirty-page ranges %d cumulative dirty-page ranges %llu c1 buffers %u c1 bytes %llu c1 actual %llu combined programs %llu",
        tick, why, c.pso, c.cps, c.shaders, c.dss, c.view_nonnull, c.view_slots, c.live_tex,
        c.pools, (unsigned long long)c.pool_adds, c.sview_nonnull, dl, (unsigned long long)da, c.c1_bufs, (unsigned long long)c.c1_alloc, (unsigned long long)c.c1_data,
        (unsigned long long)atomic_load_explicit(&g_bound_program_count, memory_order_relaxed));
}

// Screenshot on request (used by DM_NVN_SHOT_UI): dm_gfx_shot_at(k, path) = take a screenshot at the kth frame after the current present frame; due requests are checked at every frame's present
#define SHOTQ_MAX 8
static struct { uint64_t frame; char path[1024]; } g_shotq[SHOTQ_MAX]; static int g_nshotq;
static pthread_mutex_t g_shotq_mu = PTHREAD_MUTEX_INITIALIZER;
static int present_encode(id<MTLCommandBuffer> cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4]);
static int present_encode_m4(CBUF cb, id<MTLTexture> src, id<MTLTexture> dst, const int crop[4], int uv_i);
// The game image needed for a full-window screenshot (frontend window_shot; no longer use drawViewHierarchy on the Metal layer with display pacing enabled): on the next present, the render thread copies the same frame

// Game picture needed by the whole-window screenshot (front end window_shot; no longer does drawViewHierarchy on a Metal layer with the display link attached): at the next present, the present thread draws the same frame
// with the same encoding as the present (present_encode) into an offscreen texture of the same size as the swapchain, and after the GPU finishes hands out a CGImage in the completion callback (the callback does not block the present). Guarded by g_shotq_mu
static void (^g_grab_cb)(CGImageRef img);
void mtl_grab_next_present(void (^cb)(CGImageRef img)) { pthread_mutex_lock(&g_shotq_mu); g_grab_cb = [cb copy]; pthread_mutex_unlock(&g_shotq_mu); }
static void grab_present(id<MTLTexture> src, const int crop[4], CGSize sz) {
    pthread_mutex_lock(&g_shotq_mu); void (^cb)(CGImageRef) = g_grab_cb; g_grab_cb = nil; pthread_mutex_unlock(&g_shotq_mu);
    if (!cb) return;
    size_t w = (size_t)sz.width, h = (size_t)sz.height, row = w * 4;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:w height:h mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> out = w && h ? [g_dev newTextureWithDescriptor:d] : nil; id<MTLBuffer> buf = out ? [g_dev newBufferWithLength:row * h options:MTLResourceStorageModeShared] : nil;
    id<MTLCommandBuffer> cb2 = buf ? q_cb() : nil;
    if (!cb2 || !present_encode(cb2, src, out, crop)) { MLOG("[RED] Full-window screenshot: failed to encode game image offscreen (%zux%zu)", w, h); cb(NULL); return; }
    id<MTLBlitCommandEncoder> b = [cb2 blitCommandEncoder];
    [b copyFromTexture:out sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:row * h];
    [b endEncoding];
    [cb2 addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c;
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(buf.contents, w, h, 8, row, cs, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little);
        CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : NULL; cb(img);
        CGImageRelease(img); CGContextRelease(ctx); CGColorSpaceRelease(cs); }];
    [cb2 commit];
}
// ---------- Test: burst capture (for measuring image continuity; disabled by default; triggered by touch-script step “<time> burst <frame count> <directory name>”, directory created under Documents) ----------
// For N consecutive frames (maximum BURST_MAX), copy a block of the game image before presentation (screen center, square with side = 0.6 × image height, raw pixels): the captured frame incurs only one extra VRAM copy and does not wait for the GPU,
// Frame pacing is unaffected (16.5–16.9 ms measured on Mac); a background thread writes to disk after all captures: f000.raw… (width × height × 4 bytes) + meta.txt (per-frame render frame number, wall clock, guest clock, local player logical position).

// ---------- Test: burst capture (for measuring picture continuity, inactive by default; triggered by the touch-script step "<time> burst <frames> <directory name>", directory created under Documents) ----------
// For N consecutive frames (limit BURST_MAX) copy a block of the game picture before presentation (a square at the screen centre with side = picture height x 0.6, raw pixels): a captured frame costs only one more GPU memory copy and does not wait for the GPU,
// so the frame interval is unaffected (16.5–16.9 ms measured on a Mac); after all frames are captured a background thread writes to disk: f000.raw... (width x height x 4 bytes) + meta.txt (presented frame number, wall clock, guest clock and logical position of the local player for each frame).
// Analysis: dm/ios/tools/burst_sad.py <directory> (per-frame picture displacement compared with logical displacement)
__attribute__((weak)) int dm_diag_player_fixed(uint32_t *x, uint32_t *y) { (void)x; (void)y; return 0; }   // the strong definition is provided by the host's diag.c
#define BURST_MAX 512   /* 512 frames is about 415 MB (450x450x4 per frame), this much only on a Mac; do not exceed 128 on a device */
static int g_burst_left, g_burst_n; static char g_burst_dir[700];
static id<MTLBuffer> g_burst_buf[BURST_MAX]; static id<MTLCommandBuffer> g_burst_cb[BURST_MAX]; static uint64_t g_burst_fr[BURST_MAX], g_burst_ns[BURST_MAX], g_burst_gns[BURST_MAX]; static uint32_t g_burst_px[BURST_MAX], g_burst_py[BURST_MAX];
static size_t g_burst_w, g_burst_h; static unsigned long g_burst_fmt;
void dm_gfx_burst(int n, const char *dir) { if (g_burst_left > 0 || n <= 0) return; if (n > BURST_MAX) n = BURST_MAX; strlcpy(g_burst_dir, dir, sizeof g_burst_dir); g_burst_n = n; g_burst_left = n; }
static void burst_grab(id<MTLTexture> src) {
    uint64_t guest_ns(void);
    int i = g_burst_n - g_burst_left; size_t dm_ = src.width, sh_ = src.height, c = sh_ * 6 / 10; if (c > dm_) c = dm_;
    g_burst_w = c; g_burst_h = c; g_burst_fmt = (unsigned long)src.pixelFormat;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:c * c * 4 options:MTLResourceStorageModeShared]; id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake((dm_ - c) / 2, (sh_ - c) / 2, 0) sourceSize:MTLSizeMake(c, c, 1) toBuffer:buf destinationOffset:0 destinationBytesPerRow:c * 4 destinationBytesPerImage:c * c * 4];
    [b endEncoding]; [cb commit];
    g_burst_buf[i] = buf; g_burst_cb[i] = cb; g_burst_fr[i] = g_present_n; g_burst_ns[i] = now_ns(); g_burst_gns[i] = guest_ns(); g_burst_px[i] = g_burst_py[i] = 0; dm_diag_player_fixed(&g_burst_px[i], &g_burst_py[i]);
    if (--g_burst_left == 0) { int n = g_burst_n; size_t w = g_burst_w, h = g_burst_h; unsigned long fmt = g_burst_fmt; NSString *dir = [NSString stringWithUTF8String:g_burst_dir];
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
            [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
            NSMutableString *meta = [NSMutableString stringWithFormat:@"w %zu h %zu fmt %lu n %d\n", w, h, fmt, n];
            for (int k = 0; k < n; k++) { [g_burst_cb[k] waitUntilCompleted];
                [[NSData dataWithBytes:g_burst_buf[k].contents length:w * h * 4] writeToFile:[dir stringByAppendingPathComponent:[NSString stringWithFormat:@"f%03d.raw", k]] atomically:NO];
                [meta appendFormat:@"%d frame %llu ns %llu guest_ns %llu px %.4f py %.4f\n", k, (unsigned long long)g_burst_fr[k], (unsigned long long)g_burst_ns[k], (unsigned long long)g_burst_gns[k], g_burst_px[k] / 65536.0, g_burst_py[k] / 65536.0];
                g_burst_buf[k] = nil; g_burst_cb[k] = nil; }
            [meta writeToFile:[dir stringByAppendingPathComponent:@"meta.txt"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
            MLOG("Test: wrote burst of %d frames to %s (%zux%zu format %lu)", n, dir.UTF8String, w, h, fmt); });
    }
}
void dm_gfx_shot_at(long k, const char *path) {
    pthread_mutex_lock(&g_shotq_mu);
    if (g_nshotq < SHOTQ_MAX) { g_shotq[g_nshotq].frame = g_present_n + (uint64_t)(k > 0 ? k : 0); strlcpy(g_shotq[g_nshotq].path, path, sizeof g_shotq[0].path); g_nshotq++; }
    else MLOG("[RED] Screenshot requests exceeded %d; dropping %s", SHOTQ_MAX, path);
    pthread_mutex_unlock(&g_shotq_mu);
}

// DM_NVN_SHOT_OUT=WIDTHxHEIGHT (for picture quality metrics): the present-frame screenshot instead captures the result "upscaled to this resolution by present_encode", i.e. the picture the player sees on screen; screen=0 (texture dump) is unaffected
static void shot(id<MTLTexture> src, const char *path, int screen, const int crop[4], uint64_t present_ticket) {
    static int out_w = -1, out_h;
    if (out_w < 0) {
        out_w = 0;
        const char *e = getenv("DM_NVN_SHOT_OUT"); int w = 0, h = 0; char sep = 0, extra = 0;
        if (e && sscanf(e, "%d%c%d%c", &w, &sep, &h, &extra) == 3 && sep == 'x' && w > 0 && h > 0) { out_w = w; out_h = h; }
    }
    id<MTLTexture> read = src;
    if (out_w && screen) {
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:out_w height:out_h mipmapped:NO];
        d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate;
        id<MTLTexture> out = [g_dev newTextureWithDescriptor:d];
        if (!out) { MLOG("[RED] Failed to create screenshot output texture %dx%d", out_w, out_h); return; }
        id<MTLCommandBuffer> cb = q_cb();
        if (!present_encode(cb, src, out, crop)) return;
        [cb commit]; [cb waitUntilCompleted];
        read = out;
    }
    if (read.pixelFormat != MTLPixelFormatRGBA8Unorm && read.pixelFormat != MTLPixelFormatRGBA8Unorm_sRGB &&
        read.pixelFormat != MTLPixelFormatBGRA8Unorm && read.pixelFormat != MTLPixelFormatBGRA8Unorm_sRGB) {
        if (present_ticket) MLOG("[nvn-present-readback] ticket=%llu size=%zux%zu format=%lu unsupported; PNG not written", (unsigned long long)present_ticket, read.width, read.height, (unsigned long)read.pixelFormat);
        else MLOG("[RED] Screenshot format %lu is unsupported", (unsigned long)read.pixelFormat);
        return;
    }
    size_t w = read.width, h = read.height, row = (w * 4 + 255) & ~(size_t)255;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:row * h options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = q_cb();
    id<MTLBlitCommandEncoder> b = cb ? [cb blitCommandEncoder] : nil;
    if (!buf || !b) { MLOG("[RED] Screenshot Metal readback resource allocation failed: buffer=%d encoder=%d", buf != nil, b != nil); return; }
    [b copyFromTexture:read sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, 1)
              toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:row * h];
    [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
    if (cb.error || !buf.contents) { MLOG("[RED] Screenshot Metal readback failed: %s", cb.error.localizedDescription.UTF8String ?: "Shared buffer is not readable"); return; }
    int bgra = read.pixelFormat == MTLPixelFormatBGRA8Unorm || read.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB;
    uint64_t frame_nonzero = 0, frame_magenta = 0, frame_nonmagenta_nonzero = 0, frame_nonmagenta_pixels = 0;
    uint64_t frame_sum[3] = {0, 0, 0}, frame_sum_sq[3] = {0, 0, 0};
    uint64_t frame_pixels = (uint64_t)w * h;
    if (present_ticket) {
        const uint8_t *pixels = buf.contents;
        for (size_t y = 0; y < h; y++) for (size_t x = 0; x < w; x++) {
            const uint8_t *p = pixels + y * row + x * 4;
            uint8_t r = p[bgra ? 2 : 0], g = p[1], bl = p[bgra ? 0 : 2];
            frame_sum[0] += r; frame_sum[1] += g; frame_sum[2] += bl;
            frame_sum_sq[0] += (uint64_t)r * r; frame_sum_sq[1] += (uint64_t)g * g; frame_sum_sq[2] += (uint64_t)bl * bl;
            frame_nonzero += (r | g | bl) != 0;
            if (r >= 200 && g <= 60 && bl >= 200) frame_magenta++;
            else { frame_nonmagenta_pixels++; frame_nonmagenta_nonzero += (r | g | bl) != 0; }
        }
    }
    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    CGBitmapInfo bi = bgra ? (kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little) : (CGBitmapInfo)kCGImageAlphaNoneSkipLast;
    CGContextRef ctx = CGBitmapContextCreate(buf.contents, w, h, 8, row, cs, bi);
    CGImageRef img = ctx ? CGBitmapContextCreateImage(ctx) : NULL;
    CFURLRef url = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)path, strlen(path), false);
    CGImageDestinationRef dst = url ? CGImageDestinationCreateWithURL(url, (__bridge CFStringRef)UTTypePNG.identifier, 1, NULL) : NULL;

    // If any step yields null (directory does not exist / not writable, bitmap context cannot be created), report a failure and return; previously it went straight to CFRelease(NULL) and crashed in CFRelease.cold (hit during a 1952-wide TEXDUMP)
    int ok = 0;
    if (!ctx || !img || !dst) MLOG("[RED] Screenshot %s failed: %s", path, !ctx ? "Failed to create bitmap context" : !img ? "Failed to capture image" : "Failed to write (directory does not exist or is not writable)");
    else { CGImageDestinationAddImage(dst, img, NULL); ok = CGImageDestinationFinalize(dst); }
    if (dst) CFRelease(dst);
    if (url) CFRelease(url);
    if (img) CGImageRelease(img);
    if (ctx) CGContextRelease(ctx);
    if (cs) CGColorSpaceRelease(cs);
    if (present_ticket) {
        double mean_r = frame_pixels ? (double)frame_sum[0] / frame_pixels : 0.0;
        double mean_g = frame_pixels ? (double)frame_sum[1] / frame_pixels : 0.0;
        double mean_b = frame_pixels ? (double)frame_sum[2] / frame_pixels : 0.0;
        double variance_rgb = frame_pixels ? (((double)frame_sum_sq[0] / frame_pixels - mean_r * mean_r) +
                                               ((double)frame_sum_sq[1] / frame_pixels - mean_g * mean_g) +
                                               ((double)frame_sum_sq[2] / frame_pixels - mean_b * mean_b)) / 3.0 : 0.0;
        if (variance_rgb < 0.0) variance_rgb = 0.0;
        MLOG("[nvn-present-readback] ticket=%llu size=%zux%zu format=%lu nonzero_rgb=%llu/%llu (%.2f%%) mean_rgb8=(%.2f,%.2f,%.2f) variance_rgb8=%.2f magenta=%llu/%llu nonmagenta_nonzero=%llu/%llu (%.2f%%) PNG=%s %s",
             (unsigned long long)present_ticket, w, h, (unsigned long)read.pixelFormat,
             (unsigned long long)frame_nonzero, (unsigned long long)frame_pixels,
             frame_pixels ? 100.0 * frame_nonzero / frame_pixels : 0.0,
             mean_r, mean_g, mean_b, variance_rgb, (unsigned long long)frame_magenta,
             (unsigned long long)frame_pixels, (unsigned long long)frame_nonmagenta_nonzero,
             (unsigned long long)frame_nonmagenta_pixels,
             frame_nonmagenta_pixels ? 100.0 * frame_nonmagenta_nonzero / frame_nonmagenta_pixels : 0.0,
             path, ok ? "Success" : "Failure");
    } else {
        dm_shot_written(path, ok);
        MLOG("Screenshot %zux%zu format %lu → %s %s", w, h, (unsigned long)read.pixelFormat, path, ok ? "Success" : "Failure");
    }
    if (!present_ticket && g_ruler_truth && getenv("DM_NVN_RULER") && !strcmp(getenv("DM_NVN_RULER"), "1") && ok) { static int shots; if (++shots == 2) g_ruler_pending = 1; }
    if (!present_ticket && g_defer_ab >= 0 && getenv("DM_NVN_DEFER_AB") && !strcmp(getenv("DM_NVN_DEFER_AB"), "1") && ok) { static int shots; if (++shots == 2) { g_defer_ab_on = 1; MLOG("DM_NVN_DEFER_AB: second screenshot finished writing to disk; enabling deferred merge channel"); } }
    if (!present_ticket && g_udc_ab >= 0 && getenv("DM_NVN_UDC_AB") && !strcmp(getenv("DM_NVN_UDC_AB"), "1") && ok) {
        static int shots;
        if (++shots == 2) { g_udc_ab_on = 1; MLOG("DM_NVN_UDC_AB: second screenshot finished writing to disk; enabling DM_NVN_UNWRITTEN_DC"); }
    }
}
// ---------- DM_NVN_TEXDUMP=frame number:path (replace %d with texture serial*100 + mip level): save each color render target used by that frame as an image ----------
// Diagnostic: before the first render pass of the next frame is opened, wait for the GPU to finish, then draw each one (the mip level used by that frame) by full-screen sampling into RGBA8 and save as PNG (float formats are clamped to [0,1])
static long g_td_frame = -2; static const char *g_td_path; static NSMutableArray *g_td_list; static int g_td_done;
static void td_dump(void);
void td_note(RPD *pd, NSUInteger nc) {
    if (g_td_frame == -2) { const char *e = getenv("DM_NVN_TEXDUMP"), *c = e ? strchr(e, ':') : NULL; g_td_frame = c ? atol(e) : -1; g_td_path = c ? c + 1 : NULL; g_td_list = [NSMutableArray array]; }
    if (g_td_frame < 0 || g_td_done) return;
    long f = (long)atomic_load(&g_cur_frame);
    if (f > g_td_frame) { td_dump(); return; }
    if (f != g_td_frame) return;
    for (NSUInteger i = 0; i < nc && i < 8; i++) {
        id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        uint64_t ser = i < (NSUInteger)R.rt.n && R.rt.color[i] ? R.rt.color[i]->serial : 0;
        NSUInteger lv = pd.colorAttachments[i].level;
        BOOL seen = NO; for (NSArray *e in g_td_list) if ([e[1] unsignedLongLongValue] == ser && [e[2] unsignedIntegerValue] == lv) seen = YES;
        if (!seen) [g_td_list addObject:@[t, @(ser), @(lv)]];
    }
}
static void td_dump(void) {
    g_td_done = 1;
    rp_end(); if (g_blit) { enc_b_end(g_blit); g_blit = nil; }
    mtl_finish_gpu();
    static id<MTLRenderPipelineState> ps; static id<MTLSamplerState> sp;
    if (!ps) {
        static const char *src_msl =
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "struct V { float4 p [[position]]; float2 uv; };\n"
            "vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uv; return o; }\n"
            "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]]) { return t.sample(s, in.uv); }\n";
        NSError *e = nil; id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src_msl) options:nil error:&e];
        MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = [lib newFunctionWithName:@"vs"]; d.fragmentFunction = [lib newFunctionWithName:@"fs"];
        d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        ps = [g_dev newRenderPipelineStateWithDescriptor:d error:&e];
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sp = [g_dev newSamplerStateWithDescriptor:sd];
        if (!ps) { MLOG("[RED] TEXDUMP pipeline failed %s", e.localizedDescription.UTF8String); return; }
    }
    MLOG("[texdump] Frame %ld has %lu color targets", g_td_frame, (unsigned long)g_td_list.count);

    // DM_NVN_TEXRAW=index,...: save raw bytes by texture index (including 3D; file: %d in the path is replaced with the index, suffix .raw, first-line text header: width height depth format bytes-per-row)
    for (const char *q = getenv("DM_NVN_TEXRAW"); q && *q; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) {
        uint64_t ser = (uint64_t)atoll(q); id o = serial_get(ser);
        id<MTLTexture> t = [o isKindOfClass:[NvPkg class]] ? ((NvPkg *)o)->tex : o;
        if (!t) { MLOG("[texraw] ser=%llu not in sequence table", ser); continue; }
        int bw, bh; NSUInteger w = t.width, h = t.height, dz = t.depth;
        size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : 4; (void)bw; (void)bh;
        size_t row = w * bpp, img = row * h;
        id<MTLBuffer> buf = [g_dev newBufferWithLength:img * dz options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, dz) toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:img];
        [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
        char path[1024]; snprintf(path, sizeof path, g_td_path, (int)ser); strlcat(path, ".raw", sizeof path);
        FILE *f = fopen(path, "wb"); if (f) { fprintf(f, "%lu %lu %lu %lu %zu\n", (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, row); fwrite(buf.contents, 1, img * dz, f); fclose(f); }
        MLOG("[texraw] ser=%llu %lux%lux%lu format %lu → %s", ser, (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, path);
    }
    // DM_NVN_TEXPNG=serial,...: sample any registered 2D texture and output it as PNG
    for (const char *q = getenv("DM_NVN_TEXPNG"); q && *q; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) {
        uint64_t ser = (uint64_t)atoll(q); id o = serial_get(ser);
        id<MTLTexture> t = [o isKindOfClass:[NvPkg class]] ? ((NvPkg *)o)->tex : o;
        if (!t) { MLOG("[texpng] ser=%llu not in serial table", ser); continue; }
        if (t.textureType != MTLTextureType2D) { MLOG("[texpng] ser=%llu type %lu skipped", ser, (unsigned long)t.textureType); continue; }
        BOOL seen = NO; for (NSArray *e in g_td_list) if ([e[1] unsignedLongLongValue] == ser && [e[2] unsignedIntegerValue] == 0) seen = YES;
        if (!seen) [g_td_list addObject:@[t, @(ser), @(0)]];
    }
    for (NSArray *e in g_td_list) {
        id<MTLTexture> t = e[0]; uint64_t ser = [e[1] unsignedLongLongValue]; NSUInteger lv = [e[2] unsignedIntegerValue];
        if (t.textureType != MTLTextureType2D) { MLOG("[texdump] ser=%llu type %lu skipped", ser, (unsigned long)t.textureType); continue; }
        if (getenv("DM_NVN_SNAPRAW")) {   // Also save the raw bytes (converting an HDR target to an 8-bit image truncates; first-line text header: width height format bytes-per-row)
            NSUInteger rw = MAX(t.width >> lv, 1u), rh = MAX(t.height >> lv, 1u);
            size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : t.pixelFormat == MTLPixelFormatR16Float || t.pixelFormat == MTLPixelFormatR16Unorm ? 2 : 4;
            id<MTLBuffer> rb = [g_dev newBufferWithLength:rw * rh * bpp options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> rc = q_cb(); id<MTLBlitCommandEncoder> bb = [rc blitCommandEncoder];
            [bb copyFromTexture:t sourceSlice:0 sourceLevel:lv sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(rw, rh, 1) toBuffer:rb destinationOffset:0 destinationBytesPerRow:rw * bpp destinationBytesPerImage:rw * rh * bpp];
            [bb endEncoding]; [rc commit]; [rc waitUntilCompleted];
            char rp_[1024]; snprintf(rp_, sizeof rp_, g_td_path, (int)(ser * 100 + lv)); strlcat(rp_, ".raw", sizeof rp_);
            FILE *f = fopen(rp_, "wb"); if (f) { fprintf(f, "%lu %lu %lu %zu\n", (unsigned long)rw, (unsigned long)rh, (unsigned long)t.pixelFormat, rw * bpp); fwrite(rb.contents, 1, rw * rh * bpp, f); fclose(f); }
        }
        id<MTLTexture> v = [t newTextureViewWithPixelFormat:t.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(lv, 1) slices:NSMakeRange(0, 1)];
        NSUInteger w = MAX(t.width >> lv, 1u), h = MAX(t.height >> lv, 1u);
        MTLTextureDescriptor *od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:w height:h mipmapped:NO];
        od.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; od.storageMode = MTLStorageModePrivate;
        id<MTLTexture> o = [g_dev newTextureWithDescriptor:od];
        id<MTLCommandBuffer> cb = q_cb();
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = o; rp.colorAttachments[0].loadAction = MTLLoadActionDontCare; rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
        [en setRenderPipelineState:ps]; [en setFragmentTexture:v atIndex:0]; [en setFragmentSamplerState:sp atIndex:0];
        [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [en endEncoding]; [cb commit]; [cb waitUntilCompleted];
        char path[1024]; snprintf(path, sizeof path, g_td_path, (int)(ser * 100 + lv));
        MLOG("[texdump] ser=%llu lv=%lu format %lu %lux%lu", ser, (unsigned long)lv, (unsigned long)t.pixelFormat, (unsigned long)w, (unsigned long)h);
        shot(o, path, 0, NULL, 0);
    }
    [g_td_list removeAllObjects];
}
// DM_NVN_TEXSNAP=zero-based in-frame pass index,... (used with DM_NVN_TEXDUMP): copy those passes' color attachments when they finish, preserving intermediate states.
// The output number is pass index*100000 + texture serial*100 + mip level; a target may be cleared and reused later in the same frame.
static int g_ts_idx[64], g_ts_n = -1, g_ts_on; static uint64_t g_ts_first, g_ts_curf = ~0ull;
static __strong id<MTLTexture> g_ts_att[8]; static uint64_t g_ts_ser[8]; static NSUInteger g_ts_lv[8]; static int g_ts_na;
void ts_note(RPD *pd, NSUInteger nc) {
    if (g_ts_n < 0) { g_ts_n = 0; const char *e = getenv("DM_NVN_TEXSNAP");
        for (const char *q = e; q && *q && g_ts_n < 64; q = strchr(q, ',') ? strchr(q, ',') + 1 : q + strlen(q)) g_ts_idx[g_ts_n++] = atoi(q); }
    g_ts_on = 0; if (!g_ts_n || g_td_frame < 0 || g_td_done) return;
    uint64_t f = atomic_load(&g_cur_frame); if (f != g_ts_curf) { g_ts_curf = f; g_ts_first = g_stat_pass; }
    if ((long)f != g_td_frame) return;
    int idx = (int)(g_stat_pass - g_ts_first); for (int i = 0; i < g_ts_n; i++) if (g_ts_idx[i] == idx) g_ts_on = idx + 1;
    g_ts_na = 0; if (!g_ts_on) return;
    for (NSUInteger i = 0; i < nc && i < 8; i++) { id<MTLTexture> t = pd.colorAttachments[i].texture; if (!t) continue;
        g_ts_att[g_ts_na] = t; g_ts_ser[g_ts_na] = R.rt.color[i] ? R.rt.color[i]->serial : 0; g_ts_lv[g_ts_na] = pd.colorAttachments[i].level; g_ts_na++; }
}
void ts_after(void) {
    if (!g_ts_on) return;
    id<MTLCommandBuffer> dcb = q_cb();
    for (int i = 0; i < g_ts_na; i++) {
        id<MTLTexture> t = g_ts_att[i]; if (t.textureType != MTLTextureType2D) continue;
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:t.pixelFormat width:t.width height:t.height mipmapped:t.mipmapLevelCount > 1];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModePrivate; d.mipmapLevelCount = t.mipmapLevelCount;
        id<MTLTexture> c = [g_dev newTextureWithDescriptor:d];
        id<MTLBlitCommandEncoder> b = [dcb blitCommandEncoder];
        [b copyFromTexture:t sourceSlice:0 sourceLevel:g_ts_lv[i] toTexture:c destinationSlice:0 destinationLevel:g_ts_lv[i] sliceCount:1 levelCount:1]; [b endEncoding];
        [g_td_list addObject:@[c, @((uint64_t)(g_ts_on - 1) * 1000 + g_ts_ser[i]), @(g_ts_lv[i])]];
    }
    g_ts_on = 0;
    [dcb commit];
}

// Ring of actual presentation times: present sequence number -> presentedTime (seconds; -1 = dropped by the system, not presented)
#define PRING 1024
typedef struct { uint64_t n; double t; } pres_ent;
static pres_ent g_pres[PRING];

// Get the actual presentation time of the nth present (returns 0 if not ready, -1 if dropped)
double mtl_presented_time(uint64_t n) { pres_ent e = g_pres[n % PRING]; return e.n == n ? e.t : 0; }
static _Atomic uint64_t g_drawable_ns;   // Accumulated wall clock waiting for nextDrawable in present (timing instrumentation)
uint64_t mtl_drawable_ns_take(void) { return atomic_exchange(&g_drawable_ns, 0); }

// Refresh rate and name of the screen the window is on: the Mac front end reads AppKit on the main thread and stores a copy (front_mac.m screen_note); front ends without this copy return 0 and take the fallback below
__attribute__((weak)) double front_display_refresh(char *name, size_t n) { if (name && n) *name = 0; return 0; }

// Screen refresh rate when there is a window (Hz, not rounded); returns 0 with no window. Fallback: Mac uses the CoreGraphics display mode of the main display, iOS uses the UIScreen max frame rate
double mtl_display_refresh(void) {
    if (!front_metal_layer()) return 0;
    double hz = front_display_refresh(NULL, 0); if (hz > 0) return hz;
#if TARGET_OS_OSX
    CGDisplayModeRef m = CGDisplayCopyDisplayMode(CGMainDisplayID());
    hz = m ? CGDisplayModeGetRefreshRate(m) : 0; if (m) CGDisplayModeRelease(m);
    return hz;
#else
    static double cached;   // The iOS screen max frame rate never changes: UIScreen is read only the first time (at present setup); later per-frame calls do not touch UIKit
    if (cached <= 0) { Class c = NSClassFromString(@"UIScreen"); id s = c ? [c valueForKey:@"mainScreen"] : nil; cached = s ? [[s valueForKey:@"maximumFramesPerSecond"] doubleValue] : 0; }
    return cached;
#endif
}
int mtl_display_hz(void) { return (int)(mtl_display_refresh() + 0.5); }

// Previously fixed at 60: 60 does not evenly divide a 144 Hz screen, so the system can only alternate 2 / 3 refreshes (13.9 / 20.8 ms), which looks stuttery.
// Source: d2r-mobile/dm/gfx/nvn/nvn_pace.h (nvn_pace_hz); commit 9393b00ea9aca52d9254214da93738f26582562f; MIT.
// 60 / 120 Hz screen → 60; 144 → 72; 165 → 82.5; 100 → 100; 240 → 60; power saving 30: 120 → 30, 144 → 36. When the refresh rate cannot be read (0), the target is used.
// The 0.02 margin lets NTSC rates like 119.88 / 59.94 count as 2 / 1 times
static inline double nvn_pace_hz(double screen_hz, double floor_hz) {
    if (screen_hz <= 0 || floor_hz <= 0) return floor_hz;
    double k = (double)(long)(screen_hz / floor_hz + 0.02); return k >= 1 ? screen_hz / k : screen_hz;
}
double dm_pace_hz(double screen_hz, double floor_hz) { return nvn_pace_hz(screen_hz, floor_hz); }
int mtl_fps_target(void) {   // nvn_obj.h: 30 / 60 target (DM_FPS, the game itself asking for 30); the classic layer has no power-saving display-link mode of its own
    static int v = -1;
    if (v < 0) { const char *e = getenv("DM_FPS"); v = e && atoi(e) == 30 ? 30 : 60; }
    return v;
}
double mtl_pace_hz(void) { return dm_pace_hz(mtl_display_refresh(), mtl_fps_target()); }
static _Atomic uint64_t g_present_stall_ns;   // For unit tests: an extra pause before each present on the present thread (simulating a drawable that stays unavailable)
#if !DM_NVN_M3
static CAMetalLayer *g_present_layer_resident;
#endif
void mtl_present_test_stall(uint64_t ns) { g_present_stall_ns = ns; }
static void present_pipeline(MTLPixelFormat format) {
    if (!g_blit_pso || g_blit_fmt != format) {
        static const char *src_msl =
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "struct V { float4 p [[position]]; float2 uv; };\n"
            "vertex V vs(uint id [[vertex_id]]) { float2 uv = float2((id << 1) & 2, id & 2); V o; o.p = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1); o.uv = uv; return o; }\n"
            "struct Crop { float4 rect; float4 edge; };\n"
            "fragment float4 fs(V in [[stage_in]], texture2d<float> t [[texture(0)]], sampler s [[sampler(0)]], constant Crop& c [[buffer(0)]]) { return t.sample(s, clamp(in.uv * c.rect.zw + c.rect.xy, c.edge.xy, c.edge.zw)); }\n";
        NSError *e = nil;
        id<MTLLibrary> lib = [g_dev newLibraryWithSource:@(src_msl) options:nil error:&e];
        if (!lib) MDIE("Render shader compilation failed %s", e.localizedDescription.UTF8String);
        MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
        pd.vertexFunction = [lib newFunctionWithName:@"vs"]; pd.fragmentFunction = [lib newFunctionWithName:@"fs"];
        pd.colorAttachments[0].pixelFormat = format;
        g_blit_pso = [g_dev newRenderPipelineStateWithDescriptor:pd error:&e];
        if (!g_blit_pso) MDIE("Render pipeline failed %s", e.localizedDescription.UTF8String);
        g_blit_fmt = format;
        MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear; sd.supportArgumentBuffers = YES;
        g_blit_samp = [g_dev newSamplerStateWithDescriptor:sd];
    }
}
// Source for presentation sampling: when the swapchain texture is sRGB (the game creates the swapchain as RGBA8_SRGB), sample through its non-sRGB view. The stored bytes are displayed onscreen, matching Switch scanout;
// Direct sampling first decodes to linear, and the presentation layer (BGRA8Unorm) does not encode again, making the whole image too dark (public repository #36). Screenshot reads raw bytes and cannot reveal this.

// Source for presentation sampling: when the swapchain texture has an sRGB format (the game created the swapchain as RGBA8_SRGB), sample through its non-sRGB view so the screen shows the stored bytes, as the Switch's scan-out does;
// sampling it directly decodes to linear first, and the present layer (BGRA8Unorm) does not encode again, so the whole picture is too dark (public repository #36). Screenshots read the raw bytes and cannot see this.
// Views are cached per source texture, 4 of them, replacing the least recently used: presentation runs on its own queue and cannot use the deferred release of the main submission; at most 3 presents are in flight, so the replaced one is already done
static id<MTLTexture> present_src(id<MTLTexture> src) {
    MTLPixelFormat raw = MTLPixelFormatInvalid;
    for (size_t i = 0; i < sizeof g_pfv_srgb / sizeof *g_pfv_srgb; i++) if (src.pixelFormat == g_pfv_srgb[i].b) raw = g_pfv_srgb[i].a;
    if (raw == MTLPixelFormatInvalid) return src;
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static id<MTLTexture> cs[4], cv[4]; static uint64_t used[4], tick;
    pthread_mutex_lock(&mu);
    int k = -1, old = 0;
    for (int i = 0; i < 4; i++) { if (cs[i] == src) k = i; if (used[i] < used[old]) old = i; }
    if (k < 0) { k = old; cs[k] = src; cv[k] = [src newTextureViewWithPixelFormat:raw];
        if (!cv[k]) RED_ONCE(50, "Render: failed to create a non-sRGB view of the sRGB swapchain texture (format %lu); sampling directly (image will appear dark, first occurrence)", (unsigned long)src.pixelFormat);
        else MLOG("Render: swapchain texture is sRGB format %lu, sampled through a format %lu view", (unsigned long)src.pixelFormat, (unsigned long)raw); }
    used[k] = ++tick; id<MTLTexture> v = cv[k] ?: src;
    pthread_mutex_unlock(&mu);
    return v;
}

// Present samples only the valid source rectangle declared by NVN; kept separate from the raw swapchain readback screenshot. Edges are clamped to the texel centers of the valid area, so bilinear does not read the black texels outside.
static int present_encode(id<MTLCommandBuffer> cb, id<MTLTexture> src0, id<MTLTexture> dst, const int crop[4]) {
    id<MTLTexture> src = present_src(src0);
    NSUInteger cx = 0, cy = 0, cw = src.width, ch = src.height;
    if (crop && (crop[2] || crop[3])) {
        if (crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 ||
            (uint64_t)crop[0] + crop[2] > src.width || (uint64_t)crop[1] + crop[3] > src.height) {
            MLOG("[RED] Render crop out of bounds: %d,%d %dx%d / original texture %lux%lu", crop[0], crop[1], crop[2], crop[3], (unsigned long)src.width, (unsigned long)src.height); return 0;
        }
        cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3];
    }
    static uint64_t last_crop[8]; uint64_t meta[8] = {src.width, src.height, cx, cy, cw, ch, dst.width, dst.height};
    if (memcmp(meta, last_crop, sizeof meta)) { memcpy(last_crop, meta, sizeof meta);
        MLOG("Render crop: original texture %lux%lu, valid %lu,%lu %lux%lu → target %lux%lu", (unsigned long)src.width, (unsigned long)src.height,
             (unsigned long)cx, (unsigned long)cy, (unsigned long)cw, (unsigned long)ch, (unsigned long)dst.width, (unsigned long)dst.height); }
    present_pipeline(dst.pixelFormat);
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    // Center the game image while preserving its aspect ratio, leaving the remainder black (do not stretch when screen and game aspect ratios differ, e.g. iPhone at about 19.5:9 versus a 16:9 game);

    // Keep the game picture's aspect ratio and center it, leaving black bars in the rest (no stretching when the screen ratio differs from the game's, e.g. iPhone about 19.5:9 versus the game's 16:9);
    // when the ratio difference is < 1% (the remainder from the game width being a multiple of 8 in the full-screen setup, e.g. 1952x900 versus 2868x1320 differs by 0.17%) fill directly, leaving no few-pixel border
    double dw = dst.width, dh = dst.height, s = fmin(dw / cw, dh / ch);
    MTLViewport vp = {floor((dw - cw * s) / 2), floor((dh - ch * s) / 2), round(cw * s), round(ch * s), 0, 1};
    if (fabs((dw / dh) / ((double)cw / ch) - 1) < 0.01) vp = (MTLViewport){0, 0, dw, dh, 0, 1};
    int full = vp.originX == 0 && vp.originY == 0 && vp.width == dw && vp.height == dh;
    static double last_vp[4];   // Log one line when the viewport changes (used by the real-device screen-level acceptance screen_check.py to check the aspect ratio)
    if (memcmp(last_vp, &vp, sizeof last_vp)) { memcpy(last_vp, &vp, sizeof last_vp);
        MLOG("Render viewport %.0f,%.0f %.0fx%.0f / canvas %.0fx%.0f (game image %lux%lu)", vp.originX, vp.originY, vp.width, vp.height, dw, dh, (unsigned long)src.width, (unsigned long)src.height); }
    rp.colorAttachments[0].texture = dst; rp.colorAttachments[0].loadAction = full ? MTLLoadActionDontCare : MTLLoadActionClear;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
    [en setViewport:vp];
    [en setRenderPipelineState:g_blit_pso]; [en setFragmentTexture:src atIndex:0]; [en setFragmentSamplerState:g_blit_samp atIndex:0];
    float uv[8] = {(float)cx / src.width, (float)cy / src.height, (float)cw / src.width, (float)ch / src.height,
                   (cx + 0.5f) / src.width, (cy + 0.5f) / src.height, (cx + cw - 0.5f) / src.width, (cy + ch - 0.5f) / src.height};
    if (atomic_load_explicit(&g_origin_ll, memory_order_relaxed)) { uv[1] += uv[3]; uv[3] = -uv[3]; }   // lower-left origin: the window image is stored bottom-up (see g_origin_ll)
    [en setFragmentBytes:uv length:sizeof uv atIndex:0];
    [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    [en endEncoding];
    return 1;
}
static int present_encode_m4(CBUF cb, id<MTLTexture> src0, id<MTLTexture> dst, const int crop[4], int uv_i) {
    id<MTLTexture> src = present_src(src0);
    NSUInteger cx = 0, cy = 0, cw = src.width, ch = src.height;
    if (crop && (crop[2] || crop[3])) {
        if (crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0 ||
            (uint64_t)crop[0] + crop[2] > src.width || (uint64_t)crop[1] + crop[3] > src.height) {
            MLOG("[RED] Render crop out of bounds: %d,%d %dx%d / original texture %lux%lu", crop[0], crop[1], crop[2], crop[3], (unsigned long)src.width, (unsigned long)src.height); return 0;
        }
        cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3];
    }
    present_pipeline(dst.pixelFormat);
    double dw = dst.width, dh = dst.height, s = fmin(dw / cw, dh / ch);
    MTLViewport vp = {floor((dw - cw * s) / 2), floor((dh - ch * s) / 2), round(cw * s), round(ch * s), 0, 1};
    if (fabs((dw / dh) / ((double)cw / ch) - 1) < 0.01) vp = (MTLViewport){0, 0, dw, dh, 0, 1};
    int full = vp.originX == 0 && vp.originY == 0 && vp.width == dw && vp.height == dh;
    RPD *rp = [RPD new];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = full ? MTLLoadActionDontCare : MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    RENC en = [cb renderCommandEncoderWithDescriptor:rp];
    if (!en) { MLOG("[RED] M4 render encoder creation failed"); return 0; }
#if DM_NVN_M3
    [en setFragmentTexture:src atIndex:0];
    [en setFragmentSamplerState:g_blit_samp atIndex:0];
#else
    [en barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
    [en setArgumentTable:g_present_args atStages:MTLRenderStageFragment];
    [g_present_args setTexture:src.gpuResourceID atIndex:0];
    [g_present_args setSamplerState:g_blit_samp.gpuResourceID atIndex:0];
#endif
    float *uv = g_present_uv[uv_i].contents;
    uv[0] = (float)cx / src.width; uv[1] = (float)cy / src.height; uv[2] = (float)cw / src.width; uv[3] = (float)ch / src.height;
    uv[4] = (cx + 0.5f) / src.width; uv[5] = (cy + 0.5f) / src.height; uv[6] = (cx + cw - 0.5f) / src.width; uv[7] = (cy + ch - 0.5f) / src.height;
#if DM_NVN_M3
    [en setFragmentBuffer:g_present_uv[uv_i] offset:0 atIndex:0];
#else
    [g_present_args setAddress:g_present_uv[uv_i].gpuAddress atIndex:0];
#endif
    [en setViewport:vp]; [en setRenderPipelineState:g_blit_pso];
    [en drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
    m4_end(en);
    return 1;
}
// Test DM_TEST_PRESENT_RGB=<N> (test only, to investigate “full-screen magenta for one or two frames on cold start”): for the process's first N presentations, after each frame is rendered and before handing it to the system, read back 5 points on the drawable
// (center and four quarter points) and their colors; log a line only when the color combination changes. If any point is magenta (R≥200, G≤60, B≥200), log a “magenta” line; summarize on the Nth presentation.

// Test DM_TEST_PRESENT_RGB=<N> (test only, for "one or two full-screen magenta frames at cold start"): for the process's first N presents, wait for each frame to finish and, before handing it to the system, read back the colors of 5 points on the drawable
// (the center and the four quarter points); log a line only when the color combination changes, log a "magenta" line if any point is magenta (R≥200, G≤60, B≥200), and log a summary at the Nth present.
// Coverage: only the drawable we hand over; colors added after the system compositor are not visible. Each present waits for the GPU synchronously, so enabling it drops frames
static long present_rgb_n(void) { static long n = -1; if (n < 0) { const char *e = getenv("DM_TEST_PRESENT_RGB"); n = e ? atol(e) : 0; if (n) MLOG("DM_TEST_PRESENT_RGB: read back drawable colors for the first %ld onscreen presentations", n); } return n; }
static void present_rgb(id<MTLTexture> dt, uint64_t ticket) {
    static long k, magenta; static uint32_t last[5]; static uint64_t t0;
    if (k >= present_rgb_n()) return;
    pthread_mutex_lock(&g_pq_mu); while (g_pq_done < ticket) pthread_cond_wait(&g_pq_cv, &g_pq_mu); pthread_mutex_unlock(&g_pq_mu);
    NSUInteger w = dt.width, h = dt.height; const NSUInteger px[5][2] = {{w / 2, h / 2}, {w / 4, h / 4}, {w * 3 / 4, h / 4}, {w / 4, h * 3 / 4}, {w * 3 / 4, h * 3 / 4}};
    id<MTLBuffer> buf = [g_dev newBufferWithLength:5 * 256 options:MTLResourceStorageModeShared]; id<MTLCommandBuffer> c2 = q_cb(); id<MTLBlitCommandEncoder> b = [c2 blitCommandEncoder];
    for (int i = 0; i < 5; i++) [b copyFromTexture:dt sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(px[i][0], px[i][1], 0) sourceSize:MTLSizeMake(1, 1, 1) toBuffer:buf destinationOffset:i * 256 destinationBytesPerRow:4 destinationBytesPerImage:4];
    [b endEncoding]; [c2 commit]; [c2 waitUntilCompleted];
    uint64_t now = now_ns(); if (!t0) t0 = now; k++;
    uint32_t c[5]; int mag = 0, chg = 0; char txt[160]; int o = 0;
    for (int i = 0; i < 5; i++) { const uint8_t *q = (const uint8_t *)buf.contents + i * 256;   // BGRA8
        int r = q[2], g = q[1], bl = q[0]; c[i] = (uint32_t)(r << 16 | g << 8 | bl); chg |= c[i] != last[i]; mag |= r >= 200 && g <= 60 && bl >= 200;
        o += snprintf(txt + o, sizeof txt - o, " %02x%02x%02x", r, g, bl); }
    if (dt.pixelFormat != MTLPixelFormatBGRA8Unorm && k == 1) MLOG("DM_TEST_PRESENT_RGB: drawable format %lu is not BGRA8; color reading as BGRA may be inaccurate", (unsigned long)dt.pixelFormat);
    if (mag) magenta++;
    if (mag) MLOG("Onscreen color readback, occurrence %ld (%.3f s), magenta: %s", k, (now - t0) / 1e9, txt);
    else if (chg || k == 1) MLOG("Onscreen color readback, occurrence %ld (%.3f s), changed to: %s", k, (now - t0) / 1e9, txt);
    memcpy(last, c, sizeof c);
    if (k == present_rgb_n()) MLOG("Onscreen color readback summary: first %ld presentations, magenta %ld times (drawable %lux%lu)", k, magenta, (unsigned long)w, (unsigned long)h);
}
static double g_link_it_tin;   // Enqueue time of the frame the display-link callback is presenting (read and written only on the display-link thread)
static int present_now(const tex_t *t, double period, uint64_t frame, uint64_t ticket, const int crop[4], id<CAMetalDrawable> supplied, double target_ts) {
    if (g_present_stall_ns) { struct timespec ts = {0, (long)g_present_stall_ns}; nanosleep(&ts, NULL); }
    g_present_n++;
    if (g_use != USE_OFF && g_use != USE_DONE) use_tick();
    if (bootprof() && (g_present_n == 1 || g_present_n % 120 == 0)) {
        if (g_present_n == 1) g_bp_first_present = bp_now() - g_bp_t0;
        MLOG("[bootprof] t=%.3f s, render occurrence %llu: loaded %llu metallibs / %.1f ms; created %llu pipelines / %.1f ms (all synchronous on the submission thread); warmup allocations: shaders %llu / render pipelines %llu / compute pipelines %llu;"
             "Texture upload %.1f MB / %llu first uploads + %llu reuploads / %.1f ms; first-frame render t=%.3f s", (bp_now() - g_bp_t0) / 1e9, g_present_n,
             g_bp_lib_n, g_bp_lib_ns / 1e6, g_bp_pso_n, g_bp_pso_ns / 1e6, g_warm_claim_shaders, g_warm_claim_rps, g_warm_claim_cps,
             g_bp_up_bytes / 1048576.0, g_pkg_uploads, g_pkg_reuploads,
             g_up_total_ns / 1e6, g_bp_first_present / 1e9);
    }
    PD("X ser=%llu", t->serial);
    id<MTLTexture> src = tex_mtl(t);
    if (nvn_draw_trace_on()) MLOG("[DEBUG-RA-DRAW] event=present frame=%llu ticket=%llu present=%llu src=#%llu size=%lux%lu nvnfmt=%#x mtlfmt=%lu",
         (unsigned long long)frame, (unsigned long long)ticket, (unsigned long long)g_present_n,
         (unsigned long long)t->serial, (unsigned long)src.width, (unsigned long)src.height,
         (unsigned)t->d.format, (unsigned long)src.pixelFormat);
    if (!src) { RED_ONCE(8, "Failed to create MTLTexture for render texture"); return 0; }
    // DM_NVN_REGLUM=x0,y0,x1,y1:file (normalized coordinates): read back this region on every presentation and log “presentation serial, presentation time ns, average region brightness, subsampled grayscale thumbnail (hex)” (diagnostic: verify animation speed is independent of frame rate).

    // DM_NVN_REGLUM=x0,y0,x1,y1:file (normalized coordinates): read back this region on every present and record "present sequence number, present time ns, region mean luminance, subsampled grayscale thumbnail (hex)" (diagnostic: verify that animation speed is independent of frame rate).
    // A separate command buffer is queued after this frame's rendering (the same queue executes in submit order); does not wait for the GPU and does not disturb frame pacing
    static int rl = -1; static float rr[4]; static FILE *rlf; static pthread_mutex_t rlmu = PTHREAD_MUTEX_INITIALIZER;
    if (rl < 0) { const char *e = getenv("DM_NVN_REGLUM"), *c = e ? strchr(e, ':') : NULL;
        rl = c && sscanf(e, "%f,%f,%f,%f", &rr[0], &rr[1], &rr[2], &rr[3]) == 4 && (rlf = fopen(c + 1, "w")) != NULL;
        if (e && !rl) MLOG("[RED] Invalid DM_NVN_REGLUM parameter or file: %s", e); }
    if (rl && (src.pixelFormat == MTLPixelFormatRGBA8Unorm || src.pixelFormat == MTLPixelFormatBGRA8Unorm || src.pixelFormat == MTLPixelFormatRGBA8Unorm_sRGB || src.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB)) {
        NSUInteger x0 = (NSUInteger)(rr[0] * src.width), y0 = (NSUInteger)(rr[1] * src.height), w = (NSUInteger)(rr[2] * src.width) - x0, h = (NSUInteger)(rr[3] * src.height) - y0;
        id<MTLBuffer> rb = [g_dev newBufferWithLength:w * h * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
        [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(x0, y0, 0) sourceSize:MTLSizeMake(w, h, 1) toBuffer:rb destinationOffset:0 destinationBytesPerRow:w * 4 destinationBytesPerImage:w * h * 4];
        [b endEncoding];
        unsigned long long n = g_present_n, tns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { (void)c; const uint8_t *q = rb.contents; uint64_t sum = 0;
            uint8_t g[4096]; NSUInteger ng = 0;   // Subsampled grayscale thumbnail (every 4th row / column, at most 4096 points), recorded along with the mean, for per-frame image correlation analysis
            for (NSUInteger y = 0; y < h; y += 4) for (NSUInteger x = 0; x < w && ng < sizeof g; x += 4) { const uint8_t *px = q + 4 * (y * w + x); g[ng++] = (uint8_t)((px[0] + px[1] + px[2]) / 3); }
            for (NSUInteger k = 0; k < w * h; k++) sum += q[4 * k] + q[4 * k + 1] + q[4 * k + 2];
            pthread_mutex_lock(&rlmu); fprintf(rlf, "%llu\t%llu\t%.4f\t", n, tns, sum / (3.0 * w * h));
            for (NSUInteger k = 0; k < ng; k++) fprintf(rlf, "%02x", g[k]);
            fputc('\n', rlf); fflush(rlf); pthread_mutex_unlock(&rlmu); }];
        [cb commit];
    }

    // DM_NVN_SHOT=frame number[,frame number...]:path (%d in the path is replaced with the frame number)
    static long shots[16]; static int nshot = -1; static const char *shot_path;
    if (nshot < 0) {
        nshot = 0; const char *e = getenv("DM_NVN_SHOT"), *c = e ? strchr(e, ':') : NULL;
        if (c) { shot_path = c + 1; for (const char *q = e; q < c && nshot < 16; q = strchr(q, ',') ? strchr(q, ',') + 1 : c) shots[nshot++] = atol(q); }
    }
    for (int i = 0; i < nshot; i++) if ((long)g_present_n == shots[i]) { char path[1024]; snprintf(path, sizeof path, shot_path, (int)g_present_n); mtl_finish_gpu(); shot(src, path, 1, crop, 0); }
    pthread_mutex_lock(&g_shotq_mu);
    for (int i = 0; i < g_nshotq;) if (g_present_n >= g_shotq[i].frame) { mtl_finish_gpu(); shot(src, g_shotq[i].path, 1, crop, 0); g_shotq[i] = g_shotq[--g_nshotq]; } else i++;
    pthread_mutex_unlock(&g_shotq_mu);

    if (g_burst_left > 0) burst_grab(src);   // test: burst capture
    CAMetalLayer *layer = (__bridge CAMetalLayer *)front_metal_layer();
    if (!layer) return 0;
#if !DM_NVN_M3
    if (g_present_layer_resident != layer) { [g_q4 addResidencySet:layer.residencySet]; g_present_layer_resident = layer; }
#endif
#if TARGET_OS_OSX
    if (!layer.displaySyncEnabled) layer.displaySyncEnabled = YES;   // Windowed mode follows the display's vsync to prevent tearing
#endif
    if (present_rgb_n() && layer.framebufferOnly) layer.framebufferOnly = NO;   // Sampling colors for the test requires reading back the drawable
    present_pipeline(layer.pixelFormat);   // Same as the original path: the first frame compiles the pipeline before taking the drawable
    if (g_grab_cb) grab_present(src, crop, layer.drawableSize);   // Game picture needed by the whole-window screenshot (peek without the lock first; the real take locks inside the function)
    uint64_t dt0 = now_ns();
    id<CAMetalDrawable> dr = supplied ? supplied : [layer nextDrawable];
    if (!supplied) g_drawable_ns += now_ns() - dt0;
    if (!dr) return 0;
    int alloc_i = (int)(ticket % 3);
    if (g_present_alloc_fence[alloc_i]) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_done < g_present_alloc_fence[alloc_i]) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        pthread_mutex_unlock(&g_pq_mu);
    }
#if DM_NVN_M3
    CBUF cb = [g_q commandBuffer];   // older API: presentation is recorded on the command buffer (presentDrawable), so the frame reaches the screen only after its drawing is done
    if (!present_encode_m4(cb, src, dr.texture, crop, alloc_i)) return 0;
#else
    [g_present_alloc4[alloc_i] reset];
    [g_q4 waitForDrawable:dr];
    CBUF cb = g_present_cb[alloc_i];
    [cb beginCommandBufferWithAllocator:g_present_alloc4[alloc_i]];
    if (!present_encode_m4(cb, src, dr.texture, crop, alloc_i)) { [cb endCommandBuffer]; [g_q4 signalDrawable:dr]; return 0; }
    mp_encode(cb, src, alloc_i);
    [cb endCommandBuffer];
#endif
#if !DM_NVN_M3
    MTL4CommitOptions *o = [MTL4CommitOptions new];
#endif
    [FB_ON(o, cb) FB_SEL:^(FB_ARG f) {
        if (f.error) MLOG("[RED] Onscreen command buffer error: %s", f.error.description.UTF8String);
#if !DM_NVN_M3
        else mp_check(alloc_i, ticket);
#endif
        pthread_mutex_lock(&g_pq_mu); if (ticket > g_pq_done) g_pq_done = ticket; g_pq_inflight--; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
    }];
#if !DM_NVN_M3
    CBUF buffers[] = {cb};
#endif
    pthread_mutex_lock(&g_pq_mu); g_pq_inflight++; pthread_mutex_unlock(&g_pq_mu);
#if !DM_NVN_M3
    [g_q4 commit:buffers count:1 options:o];
#endif
    if (target_ts > 0) {
        static double leads[300], e2e[300];   // e2e: game enqueue -> system-scheduled present (including the time waiting in the queue for the display-link callback), the full span of input latency that our side can read
        static uint64_t lead_n;
        leads[lead_n % 300] = (target_ts - CACurrentMediaTime()) * 1000.0; e2e[lead_n % 300] = g_link_it_tin > 0 ? (target_ts - g_link_it_tin) * 1000.0 : 0;
        lead_n++;
        if (lead_n % 300 == 0) {
            double a[300]; memcpy(a, leads, sizeof a);
            for (int i = 1; i < 300; i++) { double v = a[i]; int j = i - 1; while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; } a[j + 1] = v; }
            double b[300]; memcpy(b, e2e, sizeof b);
            for (int i = 1; i < 300; i++) { double v = b[i]; int j = i - 1; while (j >= 0 && b[j] > v) { b[j + 1] = b[j]; j--; } b[j + 1] = v; }
            MLOG("Display cadence through %llu: ticks 300, renders %llu, idle ticks %llu, merged drops of old frames %llu; submission lead over target presentation p50 %.3f ms / minimum %.3f ms; game enqueue to target presentation p50 %.3f ms / minimum %.3f / maximum %.3f ms",
                 (unsigned long long)g_link_tick_n, (unsigned long long)lead_n, (unsigned long long)(g_link_tick_n - lead_n),   // Count only frames presented via the display link: the period of acquiring drawables ourselves after a stall does not count (previously the total present count was used, which went negative after a rescue)
                 (unsigned long long)g_link_dropped, a[149], a[0], b[149], b[0], b[299]);
        }
    }
#if !DM_NVN_M3
    if (present_rgb_n()) present_rgb(dr.texture, ticket);
    [g_q4 signalDrawable:dr];
#endif
    g_present_alloc_fence[alloc_i] = ticket;
#if TARGET_OS_SIMULATOR   // The iOS simulator SDK has no presentation-time callback and no minimum display duration (used only for picture acceptance, display-link ticks are not counted)
    (void)period; M3_PRESENT(cb, dr);
#else
    uint64_t pn = ticket;   // Record the present time by the game's present sequence number (ticket number): hle's present statistics are taken by "the game's nth present"; in the display link path the actual present count != the game's present count (old frames get merged and dropped)
    [dr addPresentedHandler:^(id<MTLDrawable> d) {   // Actual presentation time (the time this frame is really put on screen, CACurrentMediaTime base)
        CFTimeInterval tt = d.presentedTime;
        g_pres[pn % PRING] = (pres_ent){pn, tt > 0 ? tt : -1};
    }];

    // Each frame is displayed for at least one present period: when the GPU completion time jitters around a 120 Hz refresh boundary, the paired "25 ms + 8.33 ms" stutter no longer appears
#if DM_NVN_M3
    if (target_ts <= 0 && period > 0) [cb presentDrawable:dr afterMinimumDuration:period]; else [cb presentDrawable:dr];
#else
    if (target_ts > 0) [dr present];
    else if (period > 0) [dr presentAfterMinimumDuration:period]; else [dr present];
#endif
#endif
#if DM_NVN_M3
    [cb commit];
#endif
    (void)frame;
    return 1;
}
// ---------- Presentation thread ----------
// NVN semantics: QueuePresentTexture only queues the presentation and returns immediately (the game often calls it while holding a lock; on a real iOS device, synchronously waiting for nextDrawable inside it
// takes about 30 ms, holding the lock about 90% of the time and starving the loading thread). Drawable acquisition, blit, and present are all performed on this thread in enqueue order.
// Ticket number = enqueue number (starting at 1); g_pq_done = ticket number processed (blit submitted to the GPU queue).
// Swapchain texture return: before AcquireTexture obtains a texture, wait for the ticket for its previous presentation to finish (mtl_present_wait); commands the game submits to it afterward

// ---------- Present thread ----------
// NVN semantics: QueuePresentTexture only queues the present and returns immediately (the game often calls it while holding a lock; iOS real device: synchronously waiting for nextDrawable inside it
// takes about 30 ms, the lock is held about 90% of the time, and the loading thread starves). Taking the drawable, blit and present are all done on this thread in enqueue order.
// Ticket number = which enqueue it is (starting from 1); g_pq_done = the ticket number that has been processed (blit committed to the GPU queue).
// Swapchain texture return: before AcquireTexture gets a texture it waits for the ticket of that texture's previous present to be processed (mtl_present_wait), after which the commands the game draws into it
// are necessarily ordered after that blit on the GPU queue.
#define PQ 8
typedef struct { tex_t t; double period; uint64_t frame; int crop[4]; double t_in; } pq_item;   // t_in: the time the game handed this frame into the present queue (CACurrentMediaTime)
static pq_item g_pq[PQ];
static uint64_t g_pq_in;
static int present_now(const tex_t *, double, uint64_t, uint64_t, const int[4], id<CAMetalDrawable>, double);

@interface NvMetalDisplayLinkDelegate : NSObject <CAMetalDisplayLinkDelegate>
@end

// Fallback when the foreground display link stops (two regressions caused hangs: while foregrounded, callbacks stopped; queued frames were never acquired, and the game waited forever for the previous presentation to finish before “acquiring the next swapchain texture”):
// If mtl_present_wait exceeds STALL_NS and there has been no display-link callback for that long, drop the waiting frame and all earlier queued frames, as when entering the background (accounting for dropped frames), and let the game continue; normal operation resumes automatically when callbacks return.

// Fallback for the display link going silent while in the foreground (two smoke-test hangs before merge: the display link stopped calling back while the app was in the foreground, nobody takes the queued frames, and the game waits forever for its previous present to be processed before "acquiring the next swapchain texture"):
// when mtl_present_wait has waited longer than STALL_NS and the display link has not called back for that long, drop the waiting frame and earlier queued frames as when entering the background (accounted as dropped frames) and let the game keep running; when the display link comes back things return to normal automatically.
// One stall logs one hint line, and one line on recovery (neither is a failure); only when the same stall lasts longer than STALL_RED_NS is it a failure. All under g_pq_mu
#define STALL_NS 100000000ull
#define STALL_RED_NS 2000000000ull
static uint64_t g_stall_t0, g_stall_drop, g_stall_n; static int g_stall_red, g_stall_logged;
// If the display link has stopped for LINK_RESCUE_NS while active, stop waiting indefinitely for it to return: have the presentation thread acquire drawables and present them itself (mode 0, independent of display-link callbacks), so animation continues;
// After g_rescue_back_ns, try switching back to the display link; if it stops again, double the interval (up to 60 seconds). Two log examples: the display link stopped on an active iPhone and never returned,

// When the display link stalls for LINK_RESCUE_NS while active, stop waiting for it: switch to the present thread acquiring a drawable itself (mode 0, no display-link callback), so the picture keeps moving;
// after g_rescue_back_ns, switch back to the display link once more, and if it stalls again double the interval (cap 60 s). Two logs: on iPhone the display link stalled while active and never came back,
// and the old fallback only dropped frames, freezing the picture for 80 s (players saw a "black screen / lockup"). All changes are made under g_pq_mu; mode switching is done by link_sync on the game present thread
#define LINK_RESCUE_NS 500000000ull
static int g_rescue_mode; static uint64_t g_rescue_back_at, g_rescue_back_ns = 3000000000ull, g_rescue_n;
// The display-link watchdog only considers the “active” state: when the app is inactive (Control Center / notifications pulled down, call, system alert) or in the background, the system does not call back anyway and the screen is obscured—dropped-frame fallback remains unchanged,

// The display link watchdog only recognizes the "active" state: when the App loses active status (pulling down Control Center / notifications, incoming call, system popup) and in the background the system does not call back anyway, and the picture is covered too -- the frame-drop fallback stays as before,
// but it is not counted as stuck and no failure is reported; the stall is measured from the later of "the most recent callback / the most recent activation" (A16 logs: 8–17 seconds inactive and 17 seconds in the background were both reported as “stuck in foreground”)
__attribute__((weak)) int dm_focus_now(void) { return 1; }   // The host's hle_oe.c provides the strong definition; when graphics unit tests do not link the host it is treated as always active
__attribute__((weak)) uint64_t dm_focus_in_since(void) { return 0; }
@implementation NvMetalDisplayLinkDelegate
- (void)metalDisplayLink:(CAMetalDisplayLink *)link needsUpdate:(CAMetalDisplayLinkUpdate *)update {
    (void)link;
    pthread_mutex_lock(&g_pq_mu);
    if (g_stall_t0) { uint64_t since = dm_focus_in_since(), t = now_ns();
        MLOG("Display cadence recovered: paused for %.0f ms during this interval (%.0f ms active); fallback dropped %llu frames (interval %llu)", (t - g_stall_t0) / 1e6, (t - (g_stall_t0 > since ? g_stall_t0 : since)) / 1e6, (unsigned long long)g_stall_drop, (unsigned long long)g_stall_n); g_stall_t0 = 0; }
    pthread_mutex_unlock(&g_pq_mu);
    pthread_mutex_lock(&g_link_mu);
    g_link_tick_ns = now_ns();
    g_link_tick_n++;
    pthread_cond_broadcast(&g_link_cv);
    pthread_mutex_unlock(&g_link_mu);
    pthread_mutex_lock(&g_pq_mu);
    if (!g_dm_present_link || g_pq_read == g_pq_in) { pthread_mutex_unlock(&g_pq_mu); return; }
    uint64_t first = g_pq_read + 1, ticket = g_pq_in;
    pq_item it = g_pq[(g_pq_in - 1) % PQ]; g_link_it_tin = it.t_in;
    g_pq_read = g_pq_in;
    if (ticket > first) { g_link_dropped += ticket - first; for (uint64_t k = first; k < ticket; k++) g_pres[k % PRING] = (pres_ent){k, -1}; }   // Old frames merged and dropped are recorded as "not presented", so the present statistics honestly count them as dropped frames
    if (ticket > 1 && g_pq_done < ticket - 1) g_pq_done = ticket - 1;
    pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    int submitted;
    @autoreleasepool { submitted = present_now(&it.t, it.period, it.frame, ticket, it.crop, update.drawable, update.targetPresentationTimestamp); }
    if (!submitted) {
        pthread_mutex_lock(&g_pq_mu);
        if (ticket > g_pq_done) g_pq_done = ticket;
        pthread_cond_broadcast(&g_pq_cv);
        pthread_mutex_unlock(&g_pq_mu);
    }
}
@end

static CAMetalDisplayLink *g_link; static CFRunLoopRef g_link_rl; static int g_link_started;
static int g_link_owns_layer, g_classic_busy;   // Under g_pq_mu: nextDrawable throws while the layer has the display link object attached, so the present thread that takes its own drawable and the display link object are mutually exclusive
static int link_on(void) { return g_dm_present_link && g_link != nil; }
static CAFrameRateRange link_range(int mode) {   // 1 / 3 pick the evenly dividing tier by screen refresh rate (mtl_pace_hz); 2 (120 Hz pacing, test tier) unchanged
    if (mode == 2) return CAFrameRateRangeMake(60, 120, 120);
    float r = (float)mtl_pace_hz(); return CAFrameRateRangeMake(r, r, r); }
static int g_link_mode = 1;   // Current mode of the display link object: 1 / 2 same as g_dm_present_link, 3 = power-saving mode 30 Hz
static void *link_main(void *arg) {   // Display link thread: started when switching to display-link mode, exits when switching back to 0
    (void)arg;
    pthread_setname_np("nvn-present-link");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    CAMetalLayer *layer = (__bridge CAMetalLayer *)front_metal_layer();
    pthread_mutex_lock(&g_pq_mu); g_link_owns_layer = 1; while (g_classic_busy) pthread_cond_wait(&g_pq_cv, &g_pq_mu); pthread_mutex_unlock(&g_pq_mu);   // Wait for the present thread to finish handing over the frame it holds
    NvMetalDisplayLinkDelegate *delegate = [NvMetalDisplayLinkDelegate new];
    CAMetalDisplayLink *link = [[CAMetalDisplayLink alloc] initWithMetalLayer:layer];
    link.preferredFrameRateRange = link_range(g_link_mode);
    link.preferredFrameLatency = g_present_latency;
    link.delegate = delegate;
    [link addToRunLoop:[NSRunLoop currentRunLoop] forMode:NSDefaultRunLoopMode];
    g_link_rl = CFRunLoopGetCurrent(); g_link = link;
    CFRunLoopRun();   // When switching back to mode 0 the display link object is invalidated and this thread exits (nextDrawable cannot get a drawable while the layer still has the display link object attached, so pausing alone is not enough)
    g_link_started = 0;
    return NULL;
}
static void link_sync(void) {   // Called every frame by the game present thread: aligns the pacing object state with g_dm_present_link, power saving mode and screen refresh rate (changeable at runtime)
    static int applied; static double applied_hz;
    if (g_rescue_mode && now_ns() >= g_rescue_back_at) {   // Rescue period expired: switch back to the display link and retry (stall timing restarts, so we do not rescue again right after switching back based on the old stall)
        pthread_mutex_lock(&g_pq_mu); g_dm_present_link = g_rescue_mode; g_rescue_mode = 0; g_stall_t0 = 0; g_link_tick_ns = now_ns(); pthread_mutex_unlock(&g_pq_mu);
        MLOG("Display cadence: recovery period expired, switching back to cadence mode %d to retry (frame %llu)", g_dm_present_link, (unsigned long long)atomic_load(&g_cur_frame)); }
    int want = g_dm_present_link;
    if (want < 0 || want > 2) { g_dm_present_link = applied == 3 ? 1 : applied; return; }
    if (want && mtl_fps_target() == 30) want = 3;   // Power saving mode: display pacing 30 Hz (evenly dividing tier by screen refresh rate)
    if (want == applied && want && want != 2 && g_link) {   // Window moved to another screen / screen refresh rate changed: pacing switches to the new evenly dividing tier
        double hz = mtl_pace_hz();
        if (hz != applied_hz) { CAMetalDisplayLink *link = g_link; link.preferredFrameRateRange = link_range(want);
            if (applied_hz) MLOG("Frame pacing changed %.2f → %.2f Hz (screen refresh rate %.2f Hz, frame %llu)", applied_hz, hz, mtl_display_refresh(), (unsigned long long)atomic_load(&g_cur_frame));
            applied_hz = hz; }
        return;
    }
    if (want == applied) return;
    g_link_mode = want;
    if (!g_link_started) {
        if (!want) { applied = 0; return; }
        if (!front_metal_layer()) return;
        g_link_started = 1; pthread_t th; if (pthread_create(&th, NULL, link_main, NULL)) MDIE("Failed to start display pacing thread"); pthread_detach(th);
        return;   // The display link object is created on a new thread; align again next frame
    }
    CAMetalDisplayLink *link = g_link; if (!link) return;
    if (want) link.preferredFrameRateRange = link_range(want);
    else { CFRunLoopRef rl = g_link_rl; g_link = nil; CFRunLoopPerformBlock(rl, kCFRunLoopDefaultMode, ^{ [link invalidate]; pthread_mutex_lock(&g_pq_mu); g_link_owns_layer = 0; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu); CFRunLoopStop(rl); }); CFRunLoopWakeUp(rl); }
    MLOG("Frame pacing mode %d → %d (frame %llu)", applied, want, (unsigned long long)atomic_load(&g_cur_frame));
    applied = want;
    pthread_mutex_lock(&g_pq_mu); pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
}
// Before entering the background (mtl_drain): when presentation is driven by the display link, queued frames that have not yet been presented can only be consumed by display-link callbacks, which stop in the background—
// waiting for the “presentation queue to empty” as-is would wait forever. The background-entry callback would never return, and the system would kill the process (the process disappeared within 22 seconds after switching to another app).

// Before going to the background (mtl_drain): when the display link drives presents, frames queued but not yet presented can only be taken by the display-link callback, and in the background the system no longer calls back --
// waiting as-is for "the present queue to empty" would wait forever, the enter-background callback would not return, and the system would kill the process (the process disappears within 22 seconds after switching to another App).
// These frames will not be seen anyway: discard them directly (accounted as dropped frames) and only wait for the one present already submitted to the GPU to finish.
static void present_drop_for_suspend(void) {
    pthread_mutex_lock(&g_pq_mu);
    if (g_link_owns_layer) {
        uint64_t n = g_pq_in - g_pq_read;
        for (uint64_t k = g_pq_read + 1; k <= g_pq_in; k++) g_pres[k % PRING] = (pres_ent){k, -1};
        g_link_dropped += n; g_pq_read = g_pq_in;
        while (g_pq_inflight) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        if (g_pq_done < g_pq_in) g_pq_done = g_pq_in;
        pthread_cond_broadcast(&g_pq_cv);
        if (n) MLOG("Entering background: discarding %llu queued frames not yet displayed (display pacing does not callback in the background)", (unsigned long long)n);
    }
    pthread_mutex_unlock(&g_pq_mu);
}
static void *present_main(void *arg) {
    (void)arg;
    pthread_setname_np("nvn-present");
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    qos_class_t qc = QOS_CLASS_UNSPECIFIED; int rp = 0; pthread_get_qos_class_np(pthread_self(), &qc, &rp); atomic_store(&g_present_qos, (uint32_t)qc);
    for (;;) {
        pthread_mutex_lock(&g_pq_mu);
        while (g_pq_read == g_pq_in || g_link_owns_layer) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // In display-link mode the display-link callback takes them; an empty queue is judged by "taken"; "done" has to wait for the feedback of the present submit and lags behind taken
        pq_item it = g_pq[g_pq_read % PQ];
        uint64_t ticket = ++g_pq_read; g_classic_busy = 1;
        pthread_mutex_unlock(&g_pq_mu);
        int submitted;
        @autoreleasepool { submitted = present_now(&it.t, it.period, it.frame, ticket, it.crop, nil, 0); }
        pthread_mutex_lock(&g_pq_mu); g_classic_busy = 0; pthread_cond_broadcast(&g_pq_cv); pthread_mutex_unlock(&g_pq_mu);
        if (!submitted) {
            pthread_mutex_lock(&g_pq_mu);
            if (ticket > g_pq_done) g_pq_done = ticket;
            pthread_cond_broadcast(&g_pq_cv);
            pthread_mutex_unlock(&g_pq_mu);
        }
    }
    return NULL;
}
uint32_t mtl_present_qos(void) { return atomic_load(&g_present_qos); }
static void present_start(void) { pthread_t th; if (pthread_create(&th, NULL, present_main, NULL)) MDIE("Failed to start rendering thread"); pthread_detach(th); }
static _Atomic int g_present_discard;
void mtl_link_pause(int on);
// While the frontend enters the background and the game saves: drop presentations directly (the display link does not callback in the background, and queuing would block the game thread; no one is watching the screen).
// Also clear frames already queued when opening: the system first stops display-link callbacks and only then notifies the app of background entry. During this gap, the game has filled the swapchain and is stuck at “waiting for the previous presentation to finish before acquiring the next texture”

// The short period when the front end goes to the background and waits for the game to save: presents are dropped directly (the display link does not call back in the background, enqueueing would block the game thread; nobody sees the picture anyway).
// When turned on, also clear the frames already queued: the system first stops the display-link callback and only afterwards notifies about going to the background, and in that gap the game has already filled the swapchain and is stuck at "before acquiring the next texture, wait for its previous present to be processed"
// (hle_nvn.c nvnWindowAcquireTexture -> mtl_present_wait). Without clearing, the game cannot advance a single tick during those 2.5 seconds of waiting for the save, and the save is not written (the same pattern three times in player logs: 3 frames queued, 0 submits)
void mtl_present_discard(int on) { atomic_store(&g_present_discard, on != 0); if (on) present_drop_for_suspend(); }
uint64_t mtl_present(const tex_t *t, double period, const int crop[4]) {
    if (atomic_load(&g_present_discard)) return 0;   // Ticket number 0 = no pending present; the swapchain texture acquire does not wait
    if (t) { xv_check(t->serial, 0, 0, "Render"); if (g_xv_npresent < 128) g_xv_present[g_xv_npresent++] = t->serial; }
    draw_trace_present_source(t, atomic_load(&g_cur_frame));
    static pthread_once_t once = PTHREAD_ONCE_INIT; pthread_once(&once, present_start);
    link_sync();
    {   // Test switch DM_TEST_LINK_STALL=start seconds,duration seconds: counting from the first present, pause the display link at that point (imitating the display link not calling back while in the foreground) and release it when the duration is up. For verifying the fallback; not set in the product
        static double t0, at = -1, dur; static int st; double now = CACurrentMediaTime();
        if (at < 0) { at = 0; const char *e = getenv("DM_TEST_LINK_STALL"); if (e && sscanf(e, "%lf,%lf", &at, &dur) == 2 && at > 0 && dur > 0) { t0 = now; st = 1; } else at = 0; }
        if (st == 1 && link_on() && now - t0 >= at) { st = 2; mtl_link_pause(1); MLOG("DM_TEST_LINK_STALL: pausing display pacing for %.1f seconds (frame %llu)", dur, (unsigned long long)atomic_load(&g_cur_frame)); }
        else if (st == 2 && now - t0 >= at + dur) { st = 3; mtl_link_pause(0); MLOG("DM_TEST_LINK_STALL: resuming display pacing (frame %llu)", (unsigned long long)atomic_load(&g_cur_frame)); }
    }
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_in - g_pq_done >= PQ) pthread_cond_wait(&g_pq_cv, &g_pq_mu);   // Wait only when the queue is full; normally AcquireTexture blocks first
    pq_item *it = &g_pq[g_pq_in % PQ]; *it = (pq_item){*t, period, atomic_load(&g_cur_frame), {0}, CACurrentMediaTime()};
    if (crop) memcpy(it->crop, crop, sizeof it->crop);   // The current frame's state is enqueued with the ticket number, rather than reading the mutable window late on the present thread
    uint64_t tk = ++g_pq_in;
    pthread_cond_broadcast(&g_pq_cv);
    pthread_mutex_unlock(&g_pq_mu);
    return tk;
}
// A frame queued only after the flag is set during discard (the render thread happened to pass the flag check) won't be picked up; waiting for it would hang again. Leave it for the subsequent mtl_drain to clear.

// Do not wait during discard: the frame enqueued only after the flag was set (the present thread had just passed the flag check) has nobody to take it, and waiting for it would stall again; it is left for the following mtl_drain to clear
// Do not wait indefinitely in display-link mode: see the note at STALL_NS. Only frames "not yet taken by the display link" are dropped; the one already on its way to the screen is waited for until handed to the GPU (same dropping as when entering the background)
void mtl_present_wait(uint64_t tk) {
    int trace = nvn_sync_trace_on();
    uint64_t done_before = 0, in_before = 0, read_before = 0, inflight_before = 0;
    if (trace) {
        pthread_mutex_lock(&g_pq_mu);
        done_before = g_pq_done; in_before = g_pq_in; read_before = g_pq_read; inflight_before = (uint64_t)g_pq_inflight;
        pthread_mutex_unlock(&g_pq_mu);
        nvn_sync_trace("present-wait-enter", "mtl_present_wait", tk, done_before, in_before, read_before, inflight_before);
    }
    pthread_mutex_lock(&g_pq_mu);
    while (g_pq_done < tk && !atomic_load(&g_present_discard)) {
        if (!g_link_owns_layer) { pthread_cond_wait(&g_pq_cv, &g_pq_mu); continue; }
        if (g_pq_read >= tk) { pthread_cond_wait(&g_pq_cv, &g_pq_mu); continue; }   // This one has already been taken by the display link and is being presented: wait for it to be handed to the GPU (the GPU completion callback will wake us)
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += (long)STALL_NS; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }

        // Already determined to be in a stall (when the display link comes back the callback first clears g_stall_t0, then records the display-link tick time): no longer wait 100 ms in vain every frame, drop directly, and the game proceeds by its own display-link wait timeout (about two ticks)
        if (!g_stall_t0 && (pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &ts) != ETIMEDOUT || g_pq_done >= tk)) continue;
        uint64_t now = now_ns(), last = __atomic_load_n(&g_link_tick_ns, __ATOMIC_RELAXED);
        if (g_pq_read >= tk) continue;
        if (now - last < STALL_NS) { pthread_cond_timedwait(&g_pq_cv, &g_pq_mu, &ts); continue; }   // The display link is still running, it is just not this one's turn yet: keep waiting (bounded, no busy spinning)
        uint64_t n = tk - g_pq_read;
        for (uint64_t k = g_pq_read + 1; k <= tk; k++) g_pres[k % PRING] = (pres_ent){k, -1};
        g_link_dropped += n; g_pq_read = tk;
        while (g_pq_inflight) pthread_cond_wait(&g_pq_cv, &g_pq_mu);
        if (g_pq_done < tk) g_pq_done = tk;
        pthread_cond_broadcast(&g_pq_cv);
        if (!g_stall_t0) { g_stall_t0 = last ? last : now; g_stall_drop = 0; g_stall_red = 0; g_stall_logged = 0; g_stall_n++; }
        g_stall_drop += n;
        int active = dm_focus_now() == 1; uint64_t since = dm_focus_in_since(), from = g_stall_t0 > since ? g_stall_t0 : since;
        if (active && !g_stall_logged && now - from >= STALL_NS) { g_stall_logged = 1;
            MLOG("No display pacing callback for %.0f ms (active): discarding %llu queued frames not yet displayed; game continues (segment %llu; pacing resumes normally when callbacks return)", (now - from) / 1e6, (unsigned long long)n, (unsigned long long)g_stall_n); }
        if (active && !g_rescue_mode && g_dm_present_link && now - from >= LINK_RESCUE_NS) {
            g_rescue_mode = g_dm_present_link; g_dm_present_link = 0; g_rescue_n++; g_rescue_back_at = now + g_rescue_back_ns;
            MLOG("Display pacing stopped for %.0f ms while active: fetching drawable and rendering directly (attempt %llu); will try switching back to pacing in %.0f seconds", (now - from) / 1e6, (unsigned long long)g_rescue_n, g_rescue_back_ns / 1e9);
            if (g_rescue_back_ns < 60000000000ull) g_rescue_back_ns *= 2;
        }
        if (active && !g_stall_red && now - from >= STALL_RED_NS) { g_stall_red = 1; MLOG("[RED] Display pacing has had no callback for %.1f consecutive seconds while active; all rendering relies on fallback frame drops, and the image is frozen", (now - from) / 1e9); }
    }
    uint64_t done_after = g_pq_done, in_after = g_pq_in, read_after = g_pq_read;
    uint64_t inflight_after = (uint64_t)g_pq_inflight, discarded = atomic_load(&g_present_discard);
    pthread_mutex_unlock(&g_pq_mu);
    if (trace) nvn_sync_trace(discarded ? "present-wait-exit-discard" : "present-wait-exit",
                              "mtl_present_wait", tk, done_after, in_after, read_after, inflight_after);
}
// Wait until all queued rendering is complete (e.g. mtl_finish = nvnQueueFinish): use mtl_present_wait; there is also a fallback if the timing stops (on hardware: the game called QueueFinish once while timing was stopped,

// Wait until all queued presents have been processed (used by mtl_finish = nvnQueueFinish etc.): goes through mtl_present_wait, so there is still a fallback when the display link stops (real device: the game called QueueFinish once while the display link was stopped,
// and this used to wait here without a timeout; nobody took the 3 queued frames and the game stalled; those two runs on Mac did not hit this call)
void mtl_present_flush(void) { pthread_mutex_lock(&g_pq_mu); uint64_t h = g_pq_in; pthread_mutex_unlock(&g_pq_mu); mtl_present_wait(h); }
void mtl_stats(uint64_t *passes, uint64_t *draws) { *passes = g_stat_pass; *draws = g_stat_draw; }
void mtl_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = g_stat_args_enc; *reuse = g_stat_args_reuse; }
void mtl_use_stats(uint64_t *call, uint64_t *skip) { *call = 0; *skip = 0; }
uint64_t mtl_ds_omit(void) { return g_stat_ds_omit; }
void mtl_bw(uint64_t *load, uint64_t *store, uint64_t *full) { *load = g_bw_load; *store = g_bw_store; *full = g_pass_full_rw; }
uint64_t mtl_udc_saved_bytes(void) { return g_udc_saved_bytes; }
int mtl_has_window(void) { return front_metal_layer() != NULL; }
int mtl_link_active(void) { return link_on(); }
double mtl_link_tick_age_ms(void) { uint64_t t = __atomic_load_n(&g_link_tick_ns, __ATOMIC_RELAXED); return t ? (now_ns() - t) / 1e6 : -1; }
void mtl_link_pause(int on) { CAMetalDisplayLink *l = g_link; if (l) l.paused = on != 0; }   // For testing (DM_TEST_BG): imitates the system stopping the display-link callback in the background
static int link_wait_step(uint64_t step, uint64_t timeout_ns) {
    static uint64_t started_tick;   // Only the game's present thread calls the display-link wait functions.
    pthread_mutex_lock(&g_link_mu);
    if (g_link_tick_n >= started_tick + step) {
        started_tick = g_link_tick_n;
        pthread_mutex_unlock(&g_link_mu);
        return 1;
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ns / 1000000000ull;
    ts.tv_nsec += timeout_ns % 1000000000ull;
    if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
    while (g_link_tick_n < started_tick + step) {
        if (pthread_cond_timedwait(&g_link_cv, &g_link_mu, &ts) == ETIMEDOUT) {
            pthread_mutex_unlock(&g_link_mu);
            return 0;
        }
    }
    started_tick = g_link_tick_n;
    pthread_mutex_unlock(&g_link_mu);
    return 1;
}
int mtl_link_wait_tick(int vsyncs, uint64_t timeout_ns) {
    if (!mtl_link_active()) return 0;
    if (vsyncs < 1) vsyncs = 1;
    uint64_t step = g_link_mode == 3 ? (uint64_t)(vsyncs + 1) / 2 : (uint64_t)vsyncs * (g_link_mode == 2 ? 2 : 1);
    return link_wait_step(step, timeout_ns);
}
int mtl_link_rate(void) { return g_link_mode == 3 ? 30 : g_link_mode == 2 ? 120 : 60; }
int mtl_link_wait_ticks(int ticks, uint64_t timeout_ns) {
    if (!mtl_link_active()) return 0;
    return link_wait_step(ticks < 1 ? 1 : (uint64_t)ticks, timeout_ns);
}

static id<CAMetalDrawable> g_prefetched_drawable; // Used for compositing straight onto the drawable
id<CAMetalDrawable> mtl_present_take_prefetched(void) { return g_prefetched_drawable; }

#ifdef NVN_RT_RANGE_TEST
#include "nvn/rt_range_check.inc"
#endif


// Called by the watchdog (no progress): writes the state of all blocking wait points on the graphics side to the log -- who is waiting for whom is visible at a glance (after switching to the Metal 4 submit model there was once a case of "ran to 900 frames then stopped, no failure line")
void mtl_wait_state_dump(void) {
    MLOG("Wait point status: main fence submitted %llu / completed %llu; main allocator fences [%llu %llu %llu]; uploads incomplete %d, allocator occupancy [%d %d %d], sequence %u, has unflushed batches %d; render queue enqueued %llu / completed %llu; main command buffer recording %d, submit thread set %d",
         (unsigned long long)g_fence, (unsigned long long)g_done, (unsigned long long)g_alloc_fence[0], (unsigned long long)g_alloc_fence[1], (unsigned long long)g_alloc_fence[2],
         atomic_load(&g_up_pending), atomic_load(&g_up_alloc_busy[0]), atomic_load(&g_up_alloc_busy[1]), atomic_load(&g_up_alloc_busy[2]), g_up_seq, g_up_cb != nil,
         (unsigned long long)g_pq_in, (unsigned long long)g_pq_done, g_cb != nil, atomic_load(&g_sub_thread_set));
}
void mtl_alloc_stats(uint64_t *waits, uint64_t *grown, int *n) { *waits = g_alloc_waits; *grown = g_alloc_grown; *n = g_alloc_n; }   // Main submit allocator pool: how many times it waited, how many times it grew, how many exist now

// ---------- Current upstream contract surface (dm/gfx/nvn/nvn_obj.h, nvn_mtl.h) ----------
// Everything the shared HLE / host layer calls beyond the classic lineage's own exports.
void mtl_tex_init(const tex_t *t) {   // TextureInitialize: the serial belongs to a live texture object until mtl_tex_release (un-retire it)
    if (!t->serial) return;
    pthread_mutex_lock(&g_rmu);
    nvn_hm_del(&g_texture_retired, &t->serial);
    pthread_mutex_unlock(&g_rmu);
}
void mtl_tex_cpu_written(const tex_t *t) {   // the CPU wrote the storage: drop the cached GPU-only texture so the next use rebuilds and uploads it from guest memory
    if (t && t->serial) serial_del(t->serial);
}
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) {   // WindowAcquireTexture's wait: same GPU-completion guarantee as mtl_wait, its own consumer
    return mtl_wait(fence, timeout_ns);
}
void mtl_set_origin_lower_left(int on) {   // NVN window origin (GL lower left, NVN's default): flip the viewport vertically, reverse the winding, and store the window image bottom-up (the present flips its UVs back)
    if (atomic_exchange(&g_origin_ll, on) != on) MLOG("window origin: %s", on ? "lower left (GL: viewport flipped, winding reversed, present flipped)" : "upper left");
}
void mtl_alias_drop_pool(const void *pool) {   // the classic layer has no storage-sharing alias table (textures are owned by their serials); pool finalization flows through mtl_tex_release
    (void)pool;
}
void mtl_vbuf_stats(uint64_t *resolve, uint64_t *reuse, uint64_t *set, uint64_t *skip) {   // vertex-buffer resolve/reuse is a Metal-4 submission-path optimization; the classic path binds buffers per slot and has no such counters
    *resolve = 0; *reuse = 0; *set = 0; *skip = 0;
}
void *mtl_device(void) { mtl_init(); return (__bridge void *)g_dev; }
void *mtl_device_classic(void) { return mtl_device(); }   // the classic layer is the classic device: the GL / Vulkan layers get a usable device in this mode too
uint64_t mtl_ext_tex(void *tex) {   // an external producer's texture (the Vulkan layer's swapchain images): a serial the present path resolves, never collected by the game side
    static _Atomic uint64_t n;
    id<MTLTexture> t = (__bridge id<MTLTexture>)tex;
    uint64_t ser = (1ull << 62) + atomic_fetch_add(&n, 1) + 1;
    serial_put(ser, t); mtl_xv_window_tex(ser); return ser;
}
uint64_t mtl_present_ext(uint64_t serial, double period, void *ev, uint64_t ev_val, const int crop[4]) {   // present an external texture once the producer's event reaches ev_val
    if (ev) {   // the classic present queue has no external-event slot: order by waiting here, before enqueueing
        id<MTLSharedEvent> e = (__bridge id<MTLSharedEvent>)ev;
        if (![e waitUntilSignaledValue:ev_val timeoutMS:5000]) {
            MLOG("[RED] external producer event timed out (serial=%llu, value=%llu); frame not enqueued",
                 (unsigned long long)serial, (unsigned long long)ev_val);
            return MTL_PRESENT_ERROR;
        }
    }
    tex_t t; memset(&t, 0, sizeof t); t.magic = MAGIC('T', 'E', 'X', 0); t.serial = serial;
    return mtl_present(&t, period, crop);
}
int mtl_link_stalled(void) { return __atomic_load_n(&g_stall_t0, __ATOMIC_RELAXED) != 0; }   // the display link is silent and queued frames are dropped
int mtl_snap_on(void) {
    const char *e = getenv("DM_PRESENT_SNAP");
    int requested = e ? atoi(e) != 0 : DM_GAME_PRESENT_SNAP != 0;
    static int warned;
    if (requested && !warned++) MLOG("[RED] present snapshots are unavailable on classic Metal; using direct presentation");
    return 0;
}
uint64_t mtl_snap_acquire_wait(void) { return 0; }
void mtl_res_stats(char *out, size_t n) {   // one resource line for the timing summary (hle_nvn.c), from this layer's own snapshot
    mtl_res_count c; mtl_res_snapshot(&c);
    snprintf(out, n, "pso=%u cps=%u shaders=%u dss=%u view=%u/%u tex=%u sview=%u pools=%u+%llu c1=%u/%llu/%llu",
             c.pso, c.cps, c.shaders, c.dss, c.view_nonnull, c.view_slots, c.live_tex, c.sview_nonnull,
             c.pools, (unsigned long long)c.pool_adds, c.c1_bufs, (unsigned long long)c.c1_alloc, (unsigned long long)c.c1_data);
}
// Counters of Metal-4 submission-path subsystems the classic layer does not have (manual barriers, blit/cut splits,
// uniform-buffer deferral / patching, pipeline merging, load-store hoisting, c-shader hoisting): the truthful value is 0,
// the same as upstream's own classic-device mode reports for them.
void mtl_barrier_stats(uint64_t *skip, uint64_t *full) { *skip = 0; *full = 0; }
uint64_t mtl_blit_splits(void) { return 0; }
uint64_t mtl_copy_cut_count(void) { return 0; }
uint64_t mtl_cut_count(void) { return 0; }
void mtl_ubd_stats(uint64_t *deferred, uint64_t *conflicts) { *deferred = 0; *conflicts = 0; }
void mtl_ubp_stats(uint64_t *made, uint64_t *reused, uint64_t *refused) { *made = 0; *reused = 0; *refused = 0; }
void mtl_ubp_why(uint64_t w[5]) { memset(w, 0, 5 * sizeof *w); }
void mtl_lc_stats(uint64_t *taken, uint64_t *landed) { *taken = 0; *landed = 0; }
void mtl_sm_stats(uint64_t *np, uint64_t *nc, uint64_t *nw) { *np = 0; *nc = 0; *nw = 0; }
void mtl_cshoist_stats(uint64_t *chm, uint64_t *chd, uint64_t chs[8]) { *chm = 0; *chd = 0; memset(chs, 0, 8 * sizeof *chs); }
void mtl_hybrid_stats(uint64_t *draws, uint64_t *made, uint64_t *none) { *draws = 0; *made = 0; *none = 0; }
uint64_t mtl_pkg_edge_skips(void) { return g_pkg_edge_skips; }   // packaged-texture checks that found only writes beside the texture (pkg_edges_clean)
void mtl_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) {   // DM_AB metrics are not collected by the classic backend
    (void)f; (void)gpu_ms; (void)frame_ms;
    const char *e = getenv("DM_AB");
    static int warned;
    if (e && atol(e) > 0 && !warned++) MLOG("[RED] DM_AB GPU phase metrics are unavailable on classic Metal; no measurements were recorded");
}
int mtl_pso_prebuild_tool(void) {   // DM_PSO_PREBUILD=tool: build the pipeline archive alone and exit, no game (main.c). The classic layer has no Metal 4 pipeline archive; the tool mode cannot run.
    const char *e = getenv("DM_PSO_PREBUILD");
    if (!e || strcmp(e, "tool")) return 0;
    MDIE("DM_PSO_PREBUILD=tool is the Metal 4 pipeline-archive tool; the classic NVN layer (DM_NVN_M3=1) has no pipeline archive - build without DM_NVN_M3=1");
}
void mtl_pso_line(char *o, size_t n) {   // pipeline time line of the timing summary: this layer's own counters
    static uint64_t p[8];
    uint64_t v[8] = {g_bp_pso_ns, g_bp_pso_n, g_bp_lib_ns, g_bp_lib_n, g_pp_kicked, g_warm_rps, g_shcache_n, 0};
    uint64_t d[8]; for (int i = 0; i < 8; i++) d[i] = v[i] - p[i];
    snprintf(o, n, "pso +%.1f ms (%llu) / shader lib +%.1f ms (%llu) / predict kick %llu / warm rps %llu / shaders %llu",
             d[0] / 1e6, (unsigned long long)d[1], d[2] / 1e6, (unsigned long long)d[3],
             (unsigned long long)d[4], (unsigned long long)d[5], (unsigned long long)d[6]);
    memcpy(p, v, sizeof p);
}
// Developer-control hooks (DM_CTL channel, hle_hid.c). texdump/texraw/texlist drive this layer's own machinery directly.
void dm_gfx_texdump_next(const char *path) {   // DM_CTL "texdump <path with %d>": save the colour targets of the next frame, driving this layer's DM_NVN_TEXDUMP machinery
    if (!path || !*path) { MLOG("[RED] texdump: empty path"); return; }
    static char buf[1024];   // the ctl line buffer is reused by the reader; keep our own copy
    snprintf(buf, sizeof buf, "%s", path);
    if (!g_td_list) g_td_list = [NSMutableArray array];
    [g_td_list removeAllObjects];
    g_td_path = buf; g_td_done = 0;
    g_td_frame = (long)atomic_load(&g_cur_frame) + 1;
    MLOG("[texdump] colour targets of frame %ld -> %s", g_td_frame, buf);
}
void dm_gfx_texraw(uint64_t serial, const char *path) {
    pthread_mutex_lock(&g_rmu);
    id o = serial_get(serial);
    id<MTLTexture> t = [o isKindOfClass:[NvPkg class]] ? ((NvPkg *)o)->tex : o;
    pthread_mutex_unlock(&g_rmu);
    if (!t) { MLOG("[texraw] ser=%llu not in sequence table", (unsigned long long)serial); return; }
    NSUInteger w = t.width, h = t.height, dz = t.depth;
    size_t bpp = t.pixelFormat == MTLPixelFormatRGBA16Float ? 8 : t.pixelFormat == MTLPixelFormatRGBA32Float ? 16 : 4;
    size_t row = w * bpp, img = row * h;
    id<MTLBuffer> buf = [g_dev newBufferWithLength:img * dz options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> cb = q_cb(); id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:t sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(w, h, dz) toBuffer:buf destinationOffset:0 destinationBytesPerRow:row destinationBytesPerImage:img];
    [b endEncoding]; [cb commit]; [cb waitUntilCompleted];
    char p[1024]; snprintf(p, sizeof p, "%s.raw", path);
    FILE *f = fopen(p, "wb"); if (f) { fprintf(f, "%lu %lu %lu %lu %zu\n", (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, row); fwrite(buf.contents, 1, img * dz, f); fclose(f); }
    MLOG("[texraw] ser=%llu %lux%lux%lu format %lu → %s", (unsigned long long)serial, (unsigned long)w, (unsigned long)h, (unsigned long)dz, (unsigned long)t.pixelFormat, p);
}
void dm_gfx_texlist(int target) {   // registered textures of that target (2 = 3D) with storage address and block-linear layout
    pthread_mutex_lock(&g_rmu);
    unsigned n = 0;
    for (int i = 0; i < g_ntreg; i++) if (g_treg[i].ok && (target == 2) == (g_treg[i].t.d.target == NVN_TEXTURE_TARGET_3D)) n++;
    MLOG("[texlist] target=%d registered=%u", target, n);
    for (int i = 0; i < g_ntreg && n; i++) if (g_treg[i].ok && (target == 2) == (g_treg[i].t.d.target == NVN_TEXTURE_TARGET_3D)) {
        const tex_t *t = &g_treg[i].t;
        MLOG("[texlist] id=%d ser=%llu %ux%ux%u fmt=%#x lv=%u target=%d storage=%p+%#llx layout=%u,%u", i, (unsigned long long)t->serial,
             t->d.width, t->d.height, t->d.depth, t->d.format, t->d.levels, t->d.target, (void *)tex_storage(t),
             t->pool ? (unsigned long long)t->off : 0ull, t->has_layout ? t->layout[4] : 0, t->has_layout ? t->layout[5] : 0);
    }
    pthread_mutex_unlock(&g_rmu);
}
#pragma clang attribute pop
