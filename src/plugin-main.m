/*
OBS macOS Background Removal
Copyright (C) 2023-2024 Sebastian Beckmann

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>
*/

#include <obs-module.h>
#include <plugin-support.h>
#include <graphics/vec2.h>
#include <util/platform.h>
#include <util/threading.h>
#include <math.h>
#include <stdatomic.h>
#include <CoreML/CoreML.h>
#include <Vision/Vision.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

/* Can be overridden at build time to install a build next to another version of the plugin. */
#ifndef VISION_FILTER_ID
#define VISION_FILTER_ID "mac_vision_filter"
#endif

#define MAX_SYNC_DELAY 4
#define FRAME_RING (MAX_SYNC_DELAY + 1)
#define MASK_RING 4
#define GUIDE_RING 8
/* Segmentation requests that may be running or waiting at once. Two lets a request that takes a
 * little longer than one frame be followed directly by the next frame instead of skipping it. */
#define MAX_IN_FLIGHT 2
#define PENDING_MAX 4
/* Masks to ignore for timing after a start: the first requests load the model and are slow. */
#define SYNC_WARMUP 30
#define STATS_INTERVAL 600

enum vision_processor {
    VISION_PROCESSOR_AUTO = 0,
    VISION_PROCESSOR_NEURAL_ENGINE = 1,
    VISION_PROCESSOR_GPU = 2,
    VISION_PROCESSOR_CPU = 3,
};

struct vision_data {
    obs_source_t *context;

    gs_effect_t *effect;
    gs_eparam_t *image_param;
    gs_eparam_t *image_texel_param;
    gs_eparam_t *image_linear_param;
    gs_eparam_t *mask_param;
    gs_eparam_t *mask_prev_param;
    gs_eparam_t *mask_texel_param;
    gs_eparam_t *guide_param;
    gs_eparam_t *guide_prev_param;
    gs_eparam_t *coeff_param;
    gs_eparam_t *coeff_next_param;
    gs_eparam_t *coeff_mix_param;
    gs_eparam_t *blur_radius_param;
    gs_eparam_t *smoothing_param;
    gs_eparam_t *guide_eps_param;
    gs_eparam_t *edge_lo_param;
    gs_eparam_t *edge_hi_param;
    gs_eparam_t *defringe_px_param;

    /* Settings */
    float threshold;
    float softness;
    float refinement;
    float outline;
    float smoothing;
    float defringe;
    bool sync_auto;
    int sync_delay;
    VNGeneratePersonSegmentationRequestQualityLevel quality;
    enum vision_processor processor;

    /* Recent source frames, kept on the GPU so that a frame can be shown together with the mask
     * that was computed from it. A frame id of 0 marks an empty slot. */
    gs_texrender_t *frames[FRAME_RING];
    uint64_t frame_ids[FRAME_RING];
    uint64_t frame_counter;
    bool needs_capture;
    bool stale;

    /* Small copies of the frames that were sent for segmentation. They are what is read back for
     * Vision, and later tell the mask passes what the picture looked like. */
    gs_texrender_t *guides[GUIDE_RING];
    uint64_t guide_ids[GUIDE_RING];
    gs_stagesurf_t *stagesurf;

    /* Recent masks after stabilisation, and the guided filter coefficients made from each.
     * A mask id is the id of the frame it belongs to. */
    gs_texture_t *mask_raw;
    gs_texrender_t *masks[MASK_RING];
    gs_texrender_t *coeffs[MASK_RING];
    uint64_t mask_ids[MASK_RING];
    uint32_t mask_width;
    uint32_t mask_height;
    int mask_head;

    /* Only touched on the mask queue */
    VNGeneratePersonSegmentationRequest *request;
    VNSequenceRequestHandler *handler;
    size_t handler_width;
    size_t handler_height;
    enum vision_processor handler_processor;
    bool logged_failure;

    dispatch_queue_t mask_queue;
    atomic_int in_flight;

    /* Guards everything below */
    pthread_mutex_t mutex;
    CVPixelBufferRef pending_masks[PENDING_MAX];
    uint64_t pending_ids[PENDING_MAX];
    int pending_count;
    uint64_t stats_inference_ns;
    uint32_t stats_inferences;

    /* How many frames after its frame a mask becomes available (rises quickly, falls slowly) and
     * how many frames apart masks are. Together they drive the automatic sync delay. */
    float latency_estimate;
    float interval_average;
    uint64_t last_mask_id;
    int delay_votes;
    int warmup;

    uint64_t stats_latency_frames;
    uint32_t stats_uploads;
};

static const char *vision_get_name(void *)
{
    return obs_module_text("Name");
}

static void vision_reset_history(struct vision_data *filter)
{
    memset(filter->frame_ids, 0, sizeof(filter->frame_ids));
    memset(filter->guide_ids, 0, sizeof(filter->guide_ids));
    memset(filter->mask_ids, 0, sizeof(filter->mask_ids));
    filter->mask_head = -1;
    filter->last_mask_id = 0;
    filter->warmup = SYNC_WARMUP;
}

/* Longest side, in pixels, of the image handed to Vision. The models work at a fixed internal
 * resolution, so reading back more than this only costs time. */
static uint32_t vision_analysis_limit(VNGeneratePersonSegmentationRequestQualityLevel quality)
{
    switch (quality) {
    case VNGeneratePersonSegmentationRequestQualityLevelFast:
        return 256;
    case VNGeneratePersonSegmentationRequestQualityLevelBalanced:
        return 512;
    default:
        return 2016;
    }
}

/* Asks Vision to run the model on a particular processor. Segmentation on the GPU competes with
 * OBS' own rendering and encoding for GPU time; the Neural Engine does not. */
static void vision_apply_processor(VNGeneratePersonSegmentationRequest *request, enum vision_processor processor)
{
    if (processor == VISION_PROCESSOR_AUTO)
        return;
    if (@available(macOS 14.0, *)) {
        Class wanted = Nil;
        switch (processor) {
        case VISION_PROCESSOR_NEURAL_ENGINE:
            wanted = [MLNeuralEngineComputeDevice class];
            break;
        case VISION_PROCESSOR_GPU:
            wanted = [MLGPUComputeDevice class];
            break;
        default:
            wanted = [MLCPUComputeDevice class];
            break;
        }

        NSError *error = nil;
        NSDictionary<VNComputeStage, NSArray<id<MLComputeDeviceProtocol>> *> *supported =
            [request supportedComputeStageDevicesAndReturnError:&error];
        bool applied = false;
        for (VNComputeStage stage in supported) {
            for (id<MLComputeDeviceProtocol> device in supported[stage]) {
                if ([device isKindOfClass:wanted]) {
                    [request setComputeDevice:device forComputeStage:stage];
                    applied = true;
                    break;
                }
            }
        }
        if (!applied)
            obs_log(LOG_WARNING, "The selected processor is not available for this model, using the default");
    }
}

/* Runs on the mask queue. Takes ownership of `input`. */
static void vision_segment(struct vision_data *filter, CVPixelBufferRef input, uint64_t frame_id,
                           VNGeneratePersonSegmentationRequestQualityLevel quality, enum vision_processor processor)
{
    @autoreleasepool {
        const size_t width = CVPixelBufferGetWidth(input);
        const size_t height = CVPixelBufferGetHeight(input);

        /* The request keeps state between frames, so start over whenever the input changes shape. */
        if (!filter->handler || filter->request.qualityLevel != quality || filter->handler_width != width ||
            filter->handler_height != height || filter->handler_processor != processor) {
            [filter->request release];
            [filter->handler release];
            filter->request = [[VNGeneratePersonSegmentationRequest alloc] init];
            filter->request.qualityLevel = quality;
            filter->request.outputPixelFormat = kCVPixelFormatType_OneComponent8;
            vision_apply_processor(filter->request, processor);
            filter->handler = [[VNSequenceRequestHandler alloc] init];
            filter->handler_width = width;
            filter->handler_height = height;
            filter->handler_processor = processor;
        }

        const uint64_t start = os_gettime_ns();
        NSError *error = nil;
        const BOOL success = [filter->handler performRequests:@[filter->request] onCVPixelBuffer:input error:&error];
        const uint64_t elapsed = os_gettime_ns() - start;

        CVPixelBufferRef output = success ? filter->request.results.firstObject.pixelBuffer : NULL;
        if (output) {
            CVPixelBufferRetain(output);
            pthread_mutex_lock(&filter->mutex);
            if (filter->pending_count == PENDING_MAX) {
                /* The render thread is not picking masks up; drop the oldest. */
                CVPixelBufferRelease(filter->pending_masks[0]);
                for (int i = 1; i < PENDING_MAX; i++) {
                    filter->pending_masks[i - 1] = filter->pending_masks[i];
                    filter->pending_ids[i - 1] = filter->pending_ids[i];
                }
                filter->pending_count--;
            }
            filter->pending_masks[filter->pending_count] = output;
            filter->pending_ids[filter->pending_count] = frame_id;
            filter->pending_count++;
            filter->stats_inference_ns += elapsed;
            filter->stats_inferences++;
            pthread_mutex_unlock(&filter->mutex);
        } else if (!filter->logged_failure) {
            filter->logged_failure = true;
            obs_log(LOG_ERROR, "Person segmentation failed: %s",
                    error ? error.localizedDescription.UTF8String : "no result");
        }

        CVPixelBufferRelease(input);
    }
    atomic_fetch_sub(&filter->in_flight, 1);
}

static gs_texture_t *vision_find_guide(struct vision_data *filter, uint64_t frame_id)
{
    if (!frame_id)
        return NULL;
    for (int i = 0; i < GUIDE_RING; i++) {
        if (filter->guide_ids[i] == frame_id)
            return gs_texrender_get_texture(filter->guides[i]);
    }
    return NULL;
}

/* Makes the small copy of the frame, reads it back from the GPU and hands it to Vision. */
static void vision_dispatch_frame(struct vision_data *filter, gs_texture_t *frame, uint32_t width, uint32_t height,
                                  uint64_t frame_id)
{
    const uint32_t limit = vision_analysis_limit(filter->quality);
    const uint32_t longest = width > height ? width : height;
    uint32_t guide_width = width;
    uint32_t guide_height = height;
    if (longest > limit) {
        guide_width = (uint32_t) ((uint64_t) width * limit / longest);
        guide_height = (uint32_t) ((uint64_t) height * limit / longest);
        if (guide_width < 2)
            guide_width = 2;
        if (guide_height < 2)
            guide_height = 2;
    }

    const int slot = (int) (frame_id % GUIDE_RING);
    gs_texrender_t *texrender = filter->guides[slot];
    filter->guide_ids[slot] = 0;

    const bool previous_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(false);
    gs_texrender_reset(texrender);
    bool rendered = false;
    if (gs_texrender_begin(texrender, guide_width, guide_height)) {
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0, 0);
        gs_ortho(0, guide_width, 0, guide_height, -100, 100);
        gs_effect_t *scale_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture(gs_effect_get_param_by_name(scale_effect, "image"), frame);
        while (gs_effect_loop(scale_effect, "Draw"))
            gs_draw_sprite(frame, 0, guide_width, guide_height);
        gs_texrender_end(texrender);
        rendered = true;
    }
    gs_enable_framebuffer_srgb(previous_srgb);

    gs_texture_t *guide = rendered ? gs_texrender_get_texture(texrender) : NULL;
    if (!guide)
        return;
    filter->guide_ids[slot] = frame_id;

    if (filter->stagesurf && (guide_width != gs_stagesurface_get_width(filter->stagesurf) ||
                              guide_height != gs_stagesurface_get_height(filter->stagesurf))) {
        gs_stagesurface_destroy(filter->stagesurf);
        filter->stagesurf = NULL;
    }
    if (!filter->stagesurf)
        filter->stagesurf = gs_stagesurface_create(guide_width, guide_height, GS_BGRA);
    if (!filter->stagesurf)
        return;

    gs_stage_texture(filter->stagesurf, guide);
    uint8_t *data;
    uint32_t linesize;
    if (!gs_stagesurface_map(filter->stagesurf, &data, &linesize))
        return;

    /* Copy into a buffer of our own: the staging surface is reused for the next frame while Vision
     * is still reading this one. */
    CVPixelBufferRef input = NULL;
    NSDictionary *attributes = @{(NSString *) kCVPixelBufferIOSurfacePropertiesKey: @ {}};
    CVPixelBufferCreate(kCFAllocatorDefault, guide_width, guide_height, kCVPixelFormatType_32BGRA,
                        (CFDictionaryRef) attributes, &input);
    if (input) {
        CVPixelBufferLockBaseAddress(input, 0);
        uint8_t *dest = CVPixelBufferGetBaseAddress(input);
        const size_t dest_linesize = CVPixelBufferGetBytesPerRow(input);
        const size_t row_bytes = (size_t) guide_width * 4;
        for (uint32_t y = 0; y < guide_height; y++)
            memcpy(dest + y * dest_linesize, data + (size_t) y * linesize, row_bytes);
        CVPixelBufferUnlockBaseAddress(input, 0);
    }
    gs_stagesurface_unmap(filter->stagesurf);
    if (!input)
        return;

    const VNGeneratePersonSegmentationRequestQualityLevel quality = filter->quality;
    const enum vision_processor processor = filter->processor;
    atomic_fetch_add(&filter->in_flight, 1);
    dispatch_async(filter->mask_queue, ^{
        vision_segment(filter, input, frame_id, quality, processor);
    });
}

/* Renders the filter target into the next slot of the frame ring. */
static void vision_capture_frame(struct vision_data *filter, uint32_t width, uint32_t height)
{
    obs_source_t *target = obs_filter_get_target(filter->context);
    const obs_source_t *parent = obs_filter_get_parent(filter->context);

    const uint32_t target_flags = obs_source_get_output_flags(target);
    const bool custom_draw = (target_flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
    const bool async_source = (target_flags & OBS_SOURCE_ASYNC) != 0;

    const uint64_t frame_id = ++filter->frame_counter;
    const int slot = (int) (frame_id % FRAME_RING);
    gs_texrender_t *texrender = filter->frames[slot];
    filter->frame_ids[slot] = 0;

    gs_texrender_reset(texrender);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    if (gs_texrender_begin(texrender, width, height)) {
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0, 0);
        gs_ortho(0, width, 0, height, -100, 100);
        if (target == parent && !custom_draw && !async_source) {
            obs_source_default_render(target);
        } else {
            obs_source_video_render(target);
        }
        gs_texrender_end(texrender);
        filter->frame_ids[slot] = frame_id;

        /* Only when there is room, so that requests can never pile up. */
        gs_texture_t *frame = gs_texrender_get_texture(texrender);
        if (frame && atomic_load(&filter->in_flight) < MAX_IN_FLIGHT)
            vision_dispatch_frame(filter, frame, width, height, frame_id);
    }
    gs_blend_state_pop();
}

/* Follows how late masks arrive and holds the video back by just enough frames to cover it. */
static void vision_update_sync(struct vision_data *filter, uint64_t mask_id)
{
    const bool warming_up = filter->warmup > 0;
    if (warming_up)
        filter->warmup--;

    /* Rise faster than it falls: the delay has to cover the slower requests, not the average one.
     * Still slowly enough that a single stall does not move it. */
    float latency = (float) (filter->frame_counter - mask_id);
    if (latency > MAX_SYNC_DELAY + 1)
        latency = MAX_SYNC_DELAY + 1;
    if (warming_up || filter->latency_estimate <= 0.0f)
        filter->latency_estimate = warming_up ? 0.0f : latency;
    else if (latency > filter->latency_estimate)
        filter->latency_estimate += (latency - filter->latency_estimate) * 0.05f;
    else
        filter->latency_estimate += (latency - filter->latency_estimate) * 0.01f;

    if (filter->last_mask_id && mask_id > filter->last_mask_id) {
        const float interval = (float) (mask_id - filter->last_mask_id);
        filter->interval_average = filter->interval_average > 0.0f
                                       ? filter->interval_average * 0.95f + interval * 0.05f
                                       : interval;
    }
    filter->last_mask_id = mask_id;

    if (!filter->sync_auto || warming_up || filter->latency_estimate <= 0.0f)
        return;

    int wanted = (int) ceilf(filter->latency_estimate - 0.25f);
    /* When segmentation is slower than the frame rate, some frames have no mask of their own and
     * get one blended from the masks either side. Wait one more frame for the later one. */
    if (filter->interval_average > 1.3f)
        wanted++;
    if (wanted < 1)
        wanted = 1;
    if (wanted > MAX_SYNC_DELAY)
        wanted = MAX_SYNC_DELAY;

    /* Every change repeats or drops a frame, so do not hunt: go up after half a second of needing
     * it, come down only after half a minute of not needing it. */
    if (wanted == filter->sync_delay) {
        filter->delay_votes = 0;
    } else if (wanted > filter->sync_delay) {
        if (filter->delay_votes < 0)
            filter->delay_votes = 0;
        if (++filter->delay_votes >= 15) {
            filter->sync_delay = wanted;
            filter->delay_votes = 0;
        }
    } else {
        if (filter->delay_votes > 0)
            filter->delay_votes = 0;
        if (--filter->delay_votes <= -900) {
            filter->sync_delay--;
            filter->delay_votes = 0;
        }
    }
}

/* Stabilises one finished mask against the previous one, works out its guided filter coefficients
 * and stores both in the mask ring. Takes ownership of `mask`. */
static void vision_process_mask(struct vision_data *filter, CVPixelBufferRef mask, uint64_t mask_id)
{
    const uint32_t width = (uint32_t) CVPixelBufferGetWidth(mask);
    const uint32_t height = (uint32_t) CVPixelBufferGetHeight(mask);
    if (CVPixelBufferGetPixelFormatType(mask) != kCVPixelFormatType_OneComponent8 || !width || !height) {
        CVPixelBufferRelease(mask);
        return;
    }

    if (filter->mask_raw &&
        (width != gs_texture_get_width(filter->mask_raw) || height != gs_texture_get_height(filter->mask_raw))) {
        gs_texture_destroy(filter->mask_raw);
        filter->mask_raw = NULL;
    }
    if (!filter->mask_raw)
        filter->mask_raw = gs_texture_create(width, height, GS_R8, 1, NULL, GS_DYNAMIC);

    if (filter->mask_raw) {
        CVPixelBufferLockBaseAddress(mask, kCVPixelBufferLock_ReadOnly);
        gs_texture_set_image(filter->mask_raw, CVPixelBufferGetBaseAddress(mask),
                             (uint32_t) CVPixelBufferGetBytesPerRow(mask), false);
        CVPixelBufferUnlockBaseAddress(mask, kCVPixelBufferLock_ReadOnly);
    }
    CVPixelBufferRelease(mask);
    if (!filter->mask_raw)
        return;

    gs_texture_t *previous = NULL;
    uint64_t previous_id = 0;
    if (filter->mask_head >= 0 && filter->mask_ids[filter->mask_head]) {
        previous = gs_texrender_get_texture(filter->masks[filter->mask_head]);
        previous_id = filter->mask_ids[filter->mask_head];
    }
    gs_texture_t *guide = vision_find_guide(filter, mask_id);
    gs_texture_t *guide_previous = vision_find_guide(filter, previous_id);
    const bool can_smooth = previous && guide && guide_previous;

    const int slot = (filter->mask_head + 1) % MASK_RING;
    filter->mask_ids[slot] = 0;

    struct vec2 mask_texel;
    vec2_set(&mask_texel, 1.0f / (float) width, 1.0f / (float) height);

    const bool previous_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(false);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

    bool done = false;
    gs_texrender_reset(filter->masks[slot]);
    if (gs_texrender_begin(filter->masks[slot], width, height)) {
        gs_ortho(0, width, 0, height, -100, 100);
        gs_effect_set_texture(filter->mask_param, filter->mask_raw);
        gs_effect_set_texture(filter->mask_prev_param, previous ? previous : filter->mask_raw);
        gs_effect_set_texture(filter->guide_param, can_smooth ? guide : filter->mask_raw);
        gs_effect_set_texture(filter->guide_prev_param, can_smooth ? guide_previous : filter->mask_raw);
        gs_effect_set_vec2(filter->mask_texel_param, &mask_texel);
        gs_effect_set_float(filter->blur_radius_param, filter->outline * 1.5f);
        gs_effect_set_float(filter->smoothing_param, can_smooth ? filter->smoothing * 0.95f : 0.0f);
        while (gs_effect_loop(filter->effect, "Stabilize"))
            gs_draw_sprite(filter->mask_raw, 0, width, height);
        gs_texrender_end(filter->masks[slot]);
        done = true;
    }

    gs_texture_t *stabilized = done ? gs_texrender_get_texture(filter->masks[slot]) : NULL;
    done = false;
    gs_texrender_reset(filter->coeffs[slot]);
    if (stabilized && gs_texrender_begin(filter->coeffs[slot], width, height)) {
        gs_ortho(0, width, 0, height, -100, 100);
        /* A larger epsilon makes the result follow the picture less. Without a guide the picture
         * is ignored altogether, which leaves the smooth local average of the mask. */
        const float guide_eps = guide ? 0.0004f * powf(100.0f, 1.0f - filter->refinement) : 1000.0f;
        gs_effect_set_texture(filter->mask_param, stabilized);
        gs_effect_set_texture(filter->guide_param, guide ? guide : stabilized);
        gs_effect_set_vec2(filter->mask_texel_param, &mask_texel);
        gs_effect_set_float(filter->guide_eps_param, guide_eps);
        while (gs_effect_loop(filter->effect, "Guide"))
            gs_draw_sprite(stabilized, 0, width, height);
        gs_texrender_end(filter->coeffs[slot]);
        done = true;
    }

    gs_blend_state_pop();
    gs_enable_framebuffer_srgb(previous_srgb);
    if (!done)
        return;

    filter->mask_ids[slot] = mask_id;
    filter->mask_head = slot;
    filter->mask_width = width;
    filter->mask_height = height;

    vision_update_sync(filter, mask_id);

    /* Occasionally report how the mask is keeping up. */
    filter->stats_latency_frames += filter->frame_counter - mask_id;
    if (++filter->stats_uploads >= STATS_INTERVAL) {
        pthread_mutex_lock(&filter->mutex);
        const double inference_ms = filter->stats_inferences
                                        ? (double) filter->stats_inference_ns / filter->stats_inferences / 1000000.0
                                        : 0.0;
        filter->stats_inference_ns = 0;
        filter->stats_inferences = 0;
        pthread_mutex_unlock(&filter->mutex);
        obs_log(LOG_INFO,
                "Mask %ux%u, segmentation %.1f ms, mask ready %.2f frames after its frame, one mask per %.2f frames "
                "(sync delay %d)",
                width, height, inference_ms, (double) filter->stats_latency_frames / filter->stats_uploads,
                (double) filter->interval_average, filter->sync_delay);
        filter->stats_latency_frames = 0;
        filter->stats_uploads = 0;
    }
}

/* Picks up the masks that the mask queue has finished since the last frame. */
static void vision_collect_masks(struct vision_data *filter)
{
    CVPixelBufferRef masks[PENDING_MAX];
    uint64_t ids[PENDING_MAX];

    pthread_mutex_lock(&filter->mutex);
    const int count = filter->pending_count;
    memcpy(masks, filter->pending_masks, sizeof(masks));
    memcpy(ids, filter->pending_ids, sizeof(ids));
    filter->pending_count = 0;
    pthread_mutex_unlock(&filter->mutex);

    for (int i = 0; i < count; i++)
        vision_process_mask(filter, masks[i], ids[i]);
}

static void vision_tick(void *filter_ptr, float)
{
    struct vision_data *filter = filter_ptr;

    /* The last tick was never rendered, so the source was hidden. Whatever is in the rings is old. */
    if (filter->needs_capture)
        filter->stale = true;
    filter->needs_capture = true;
}

static void vision_render(void *filter_ptr, gs_effect_t *)
{
    struct vision_data *filter = filter_ptr;

    obs_source_t *target = obs_filter_get_target(filter->context);
    const uint32_t width = obs_source_get_base_width(target);
    const uint32_t height = obs_source_get_base_height(target);
    if (!width || !height || !filter->effect)
        return;

    if (filter->stale) {
        vision_reset_history(filter);
        filter->stale = false;
    }

    /* A source can be rendered several times per frame (preview, program, filter dialog).
     * Capture once per tick and draw the same result every time. */
    if (filter->needs_capture) {
        filter->needs_capture = false;
        vision_capture_frame(filter, width, height);
    }
    vision_collect_masks(filter);

    /* Pick the frame to show: the newest one when the delay is 0, otherwise one held back so that
     * its own mask has had time to arrive. */
    if (filter->frame_counter <= (uint64_t) filter->sync_delay)
        return;
    const uint64_t shown_id = filter->frame_counter - (uint64_t) filter->sync_delay;
    gs_texture_t *frame = NULL;
    for (int i = 0; i < FRAME_RING; i++) {
        if (filter->frame_ids[i] == shown_id)
            frame = gs_texrender_get_texture(filter->frames[i]);
    }

    /* And the mask computed from that frame. If that frame was skipped because segmentation was
     * busy, blend the masks of the frames before and after it; failing that, use the closest. */
    gs_texture_t *coeff = NULL;
    gs_texture_t *coeff_next = NULL;
    float coeff_mix = 0.0f;
    uint64_t before_id = 0;
    uint64_t after_id = UINT64_MAX;
    for (int i = 0; i < MASK_RING; i++) {
        const uint64_t id = filter->mask_ids[i];
        if (!id)
            continue;
        gs_texture_t *texture = gs_texrender_get_texture(filter->coeffs[i]);
        if (id <= shown_id && id >= before_id) {
            before_id = id;
            coeff = texture;
        }
        if (id > shown_id && id < after_id) {
            after_id = id;
            coeff_next = texture;
        }
    }
    if (coeff && coeff_next && before_id != shown_id) {
        coeff_mix = (float) (shown_id - before_id) / (float) (after_id - before_id);
    } else if (!coeff) {
        coeff = coeff_next;
    }
    if (coeff_mix == 0.0f)
        coeff_next = coeff;

    /* Draw nothing rather than the unmasked source while there is no frame or mask yet: showing
     * the background, even for a few frames, is the one thing this filter must not do. */
    if (!frame || !coeff || !filter->mask_width || !filter->mask_height)
        return;

    float half_band = (filter->softness > 0.02f ? filter->softness : 0.02f) / 2.0f;
    float edge_lo = filter->threshold - half_band;
    float edge_hi = filter->threshold + half_band;
    if (edge_lo < 0.0f)
        edge_lo = 0.0f;
    if (edge_lo > 0.98f)
        edge_lo = 0.98f;
    if (edge_hi > 1.0f)
        edge_hi = 1.0f;
    if (edge_hi < edge_lo + 0.02f)
        edge_hi = edge_lo + 0.02f;

    struct vec2 image_texel;
    vec2_set(&image_texel, 1.0f / (float) width, 1.0f / (float) height);
    struct vec2 mask_texel;
    vec2_set(&mask_texel, 1.0f / (float) filter->mask_width, 1.0f / (float) filter->mask_height);

    const bool linear_srgb = gs_get_linear_srgb();
    const bool previous_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(linear_srgb);

    if (linear_srgb)
        gs_effect_set_texture_srgb(filter->image_param, frame);
    else
        gs_effect_set_texture(filter->image_param, frame);
    gs_effect_set_float(filter->image_linear_param, linear_srgb ? 1.0f : 0.0f);
    gs_effect_set_vec2(filter->image_texel_param, &image_texel);
    gs_effect_set_texture(filter->coeff_param, coeff);
    gs_effect_set_texture(filter->coeff_next_param, coeff_next);
    gs_effect_set_float(filter->coeff_mix_param, coeff_mix);
    gs_effect_set_vec2(filter->mask_texel_param, &mask_texel);
    gs_effect_set_float(filter->edge_lo_param, edge_lo);
    gs_effect_set_float(filter->edge_hi_param, edge_hi);
    gs_effect_set_float(filter->defringe_px_param, filter->defringe * 6.0f);

    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_INVSRCALPHA);
    while (gs_effect_loop(filter->effect, "Draw"))
        gs_draw_sprite(frame, 0, width, height);
    gs_blend_state_pop();

    gs_enable_framebuffer_srgb(previous_srgb);
}

static void vision_add_slider(obs_properties_t *props, const char *name, const char *text_key,
                              const char *description_key)
{
    obs_property_t *property = obs_properties_add_float_slider(props, name, obs_module_text(text_key), 0, 1, 0.01);
    obs_property_set_long_description(property, obs_module_text(description_key));
}

static bool vision_sync_auto_modified(obs_properties_t *props, obs_property_t *, obs_data_t *settings)
{
    obs_property_set_enabled(obs_properties_get(props, "sync_delay"), !obs_data_get_bool(settings, "sync_auto"));
    return true;
}

static obs_properties_t *vision_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_property_t *list = obs_properties_add_list(props, "quality", obs_module_text("Quality"), OBS_COMBO_TYPE_LIST,
                                                   OBS_COMBO_FORMAT_INT);
    obs_property_list_add_int(list, obs_module_text("Quality.Accurate"),
                              VNGeneratePersonSegmentationRequestQualityLevelAccurate);
    obs_property_list_add_int(list, obs_module_text("Quality.Balanced"),
                              VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_property_list_add_int(list, obs_module_text("Quality.Fast"),
                              VNGeneratePersonSegmentationRequestQualityLevelFast);

    if (@available(macOS 14.0, *)) {
        list = obs_properties_add_list(props, "processor", obs_module_text("Processor"), OBS_COMBO_TYPE_LIST,
                                       OBS_COMBO_FORMAT_INT);
        obs_property_set_long_description(list, obs_module_text("Processor.Description"));
        obs_property_list_add_int(list, obs_module_text("Processor.Auto"), VISION_PROCESSOR_AUTO);
        obs_property_list_add_int(list, obs_module_text("Processor.NeuralEngine"), VISION_PROCESSOR_NEURAL_ENGINE);
        obs_property_list_add_int(list, obs_module_text("Processor.GPU"), VISION_PROCESSOR_GPU);
        obs_property_list_add_int(list, obs_module_text("Processor.CPU"), VISION_PROCESSOR_CPU);
    }

    vision_add_slider(props, "threshold", "Threshold", "Threshold.Description");
    vision_add_slider(props, "softness", "Softness", "Softness.Description");
    vision_add_slider(props, "outline", "Outline", "Outline.Description");
    vision_add_slider(props, "refinement", "Refinement", "Refinement.Description");
    vision_add_slider(props, "smoothing", "Smoothing", "Smoothing.Description");
    vision_add_slider(props, "defringe", "Defringe", "Defringe.Description");

    obs_property_t *sync_auto = obs_properties_add_bool(props, "sync_auto", obs_module_text("SyncAuto"));
    obs_property_set_long_description(sync_auto, obs_module_text("SyncAuto.Description"));
    obs_property_set_modified_callback(sync_auto, vision_sync_auto_modified);
    obs_property_t *delay =
        obs_properties_add_int_slider(props, "sync_delay", obs_module_text("SyncDelay"), 0, MAX_SYNC_DELAY, 1);
    obs_property_set_long_description(delay, obs_module_text("SyncDelay.Description"));
    return props;
}

static void vision_defaults(obs_data_t *settings)
{
    obs_data_set_default_int(settings, "quality", VNGeneratePersonSegmentationRequestQualityLevelBalanced);
    obs_data_set_default_int(settings, "processor", VISION_PROCESSOR_AUTO);
    obs_data_set_default_double(settings, "threshold", 0.5);
    obs_data_set_default_double(settings, "softness", 0.25);
    obs_data_set_default_double(settings, "outline", 0.5);
    obs_data_set_default_double(settings, "refinement", 0.4);
    obs_data_set_default_double(settings, "smoothing", 0.9);
    obs_data_set_default_double(settings, "defringe", 0.3);
    obs_data_set_default_bool(settings, "sync_auto", true);
    obs_data_set_default_int(settings, "sync_delay", 2);
}

static void vision_update(void *filter_ptr, obs_data_t *settings)
{
    struct vision_data *filter = filter_ptr;

    filter->quality = obs_data_get_int(settings, "quality");
    filter->processor = (enum vision_processor) obs_data_get_int(settings, "processor");
    filter->threshold = (float) obs_data_get_double(settings, "threshold");
    filter->softness = (float) obs_data_get_double(settings, "softness");
    filter->outline = (float) obs_data_get_double(settings, "outline");
    filter->refinement = (float) obs_data_get_double(settings, "refinement");
    filter->smoothing = (float) obs_data_get_double(settings, "smoothing");
    filter->defringe = (float) obs_data_get_double(settings, "defringe");

    int sync_delay = (int) obs_data_get_int(settings, "sync_delay");
    if (sync_delay < 0)
        sync_delay = 0;
    if (sync_delay > MAX_SYNC_DELAY)
        sync_delay = MAX_SYNC_DELAY;
    filter->sync_auto = obs_data_get_bool(settings, "sync_auto");
    if (!filter->sync_auto)
        filter->sync_delay = sync_delay;
    else if (filter->sync_delay < 1)
        filter->sync_delay = 1;
    filter->delay_votes = 0;
}

static void *vision_create(obs_data_t *settings, struct obs_source *source)
{
    struct vision_data *filter = bzalloc(sizeof(struct vision_data));
    filter->context = source;
    filter->mask_head = -1;
    filter->warmup = SYNC_WARMUP;
    atomic_init(&filter->in_flight, 0);
    pthread_mutex_init(&filter->mutex, NULL);

    /* Performing the segmentation in realtime takes too long and will lag OBS, especially at higher
     * quality modes. As such, defer it to a different thread. The mask arrives a frame or so late,
     * which the sync delay makes up for by holding the video back to match. */
    filter->mask_queue = dispatch_queue_create("Filter mask dispatch queue", NULL);

    obs_enter_graphics();
    for (int i = 0; i < FRAME_RING; i++)
        filter->frames[i] = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    for (int i = 0; i < GUIDE_RING; i++)
        filter->guides[i] = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    for (int i = 0; i < MASK_RING; i++) {
        filter->masks[i] = gs_texrender_create(GS_R8, GS_ZS_NONE);
        filter->coeffs[i] = gs_texrender_create(GS_RGBA16F, GS_ZS_NONE);
    }

    char *file = obs_module_file("alpha_mask.effect");
    filter->effect = gs_effect_create_from_file(file, NULL);
    bfree(file);
    if (filter->effect) {
        filter->image_param = gs_effect_get_param_by_name(filter->effect, "image");
        filter->image_texel_param = gs_effect_get_param_by_name(filter->effect, "image_texel");
        filter->image_linear_param = gs_effect_get_param_by_name(filter->effect, "image_linear");
        filter->mask_param = gs_effect_get_param_by_name(filter->effect, "mask");
        filter->mask_prev_param = gs_effect_get_param_by_name(filter->effect, "mask_prev");
        filter->mask_texel_param = gs_effect_get_param_by_name(filter->effect, "mask_texel");
        filter->guide_param = gs_effect_get_param_by_name(filter->effect, "guide");
        filter->guide_prev_param = gs_effect_get_param_by_name(filter->effect, "guide_prev");
        filter->coeff_param = gs_effect_get_param_by_name(filter->effect, "coeff");
        filter->coeff_next_param = gs_effect_get_param_by_name(filter->effect, "coeff_next");
        filter->coeff_mix_param = gs_effect_get_param_by_name(filter->effect, "coeff_mix");
        filter->blur_radius_param = gs_effect_get_param_by_name(filter->effect, "blur_radius");
        filter->smoothing_param = gs_effect_get_param_by_name(filter->effect, "smoothing");
        filter->guide_eps_param = gs_effect_get_param_by_name(filter->effect, "guide_eps");
        filter->edge_lo_param = gs_effect_get_param_by_name(filter->effect, "edge_lo");
        filter->edge_hi_param = gs_effect_get_param_by_name(filter->effect, "edge_hi");
        filter->defringe_px_param = gs_effect_get_param_by_name(filter->effect, "defringe_px");
    } else {
        obs_log(LOG_ERROR, "Could not load alpha_mask.effect");
    }
    obs_leave_graphics();

    vision_update(filter, settings);
    return filter;
}

static void vision_destroy(void *filter_ptr)
{
    struct vision_data *filter = filter_ptr;

    /* Wait for requests that are still running; they use the filter data. */
    dispatch_sync(filter->mask_queue, ^{
    });
    dispatch_release(filter->mask_queue);
    [filter->request release];
    [filter->handler release];
    for (int i = 0; i < filter->pending_count; i++)
        CVPixelBufferRelease(filter->pending_masks[i]);
    pthread_mutex_destroy(&filter->mutex);

    obs_enter_graphics();
    for (int i = 0; i < FRAME_RING; i++)
        gs_texrender_destroy(filter->frames[i]);
    for (int i = 0; i < GUIDE_RING; i++)
        gs_texrender_destroy(filter->guides[i]);
    for (int i = 0; i < MASK_RING; i++) {
        gs_texrender_destroy(filter->masks[i]);
        gs_texrender_destroy(filter->coeffs[i]);
    }
    if (filter->stagesurf) {
        gs_stagesurface_destroy(filter->stagesurf);
    }
    if (filter->mask_raw) {
        gs_texture_destroy(filter->mask_raw);
    }
    gs_effect_destroy(filter->effect);
    obs_leave_graphics();
    bfree(filter);
};

bool obs_module_load(void)
{
    struct obs_source_info info = {
        .id = VISION_FILTER_ID,
        .type = OBS_SOURCE_TYPE_FILTER,
        .output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB,
        .get_name = vision_get_name,
        .create = vision_create,
        .destroy = vision_destroy,
        .video_tick = vision_tick,
        .video_render = vision_render,
        .get_defaults = vision_defaults,
        .get_properties = vision_properties,
        .update = vision_update,
    };
    obs_register_source(&info);
    obs_log(LOG_INFO, "Loaded successfully (version %s)", PLUGIN_VERSION);
    return true;
}
