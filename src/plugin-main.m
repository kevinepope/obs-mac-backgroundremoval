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
#define STATS_INTERVAL 600

struct vision_data {
    obs_source_t *context;

    gs_effect_t *effect;
    gs_eparam_t *image_param;
    gs_eparam_t *mask_param;
    gs_eparam_t *mask_prev_param;
    gs_eparam_t *mask_next_param;
    gs_eparam_t *mask_mix_param;
    gs_eparam_t *tap_step_param;
    gs_eparam_t *image_texel_param;
    gs_eparam_t *edge_lo_param;
    gs_eparam_t *edge_hi_param;
    gs_eparam_t *range_scale_param;
    gs_eparam_t *defringe_px_param;
    gs_eparam_t *smoothing_param;

    /* Settings */
    float threshold;
    float softness;
    float refinement;
    float smoothing;
    float defringe;
    bool sync_auto;
    int sync_delay;
    VNGeneratePersonSegmentationRequestQualityLevel quality;

    /* Recent source frames, kept on the GPU so that a frame can be shown together with the mask
     * that was computed from it. A frame id of 0 marks an empty slot. */
    gs_texrender_t *frames[FRAME_RING];
    uint64_t frame_ids[FRAME_RING];
    uint64_t frame_counter;
    bool needs_capture;
    bool stale;

    /* Downscaled copy of the frame that is read back for Vision */
    gs_texrender_t *analysis;
    gs_stagesurf_t *stagesurf;

    /* Recent masks, after temporal stabilisation. A mask id is the id of the frame it belongs to. */
    gs_texture_t *mask_raw;
    gs_texrender_t *masks[MASK_RING];
    uint64_t mask_ids[MASK_RING];
    uint32_t mask_width;
    uint32_t mask_height;
    int mask_head;

    /* Only touched on the mask queue */
    VNGeneratePersonSegmentationRequest *request;
    VNSequenceRequestHandler *handler;
    size_t handler_width;
    size_t handler_height;
    bool logged_failure;

    dispatch_queue_t mask_queue;
    atomic_bool busy;

    /* Guards everything below */
    pthread_mutex_t mutex;
    CVPixelBufferRef pending_mask;
    uint64_t pending_id;
    uint64_t stats_inference_ns;
    uint32_t stats_inferences;

    /* How many frames after its frame a mask becomes available, smoothed. Drives the automatic
     * sync delay. */
    float latency_average;
    float interval_average;
    uint64_t last_mask_id;
    int delay_votes;

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
    memset(filter->mask_ids, 0, sizeof(filter->mask_ids));
    filter->mask_head = -1;
    filter->last_mask_id = 0;
}

/* Longest side, in pixels, of the image handed to Vision. The models work at a fixed internal
 * resolution, so reading back more than this only costs time. */
static uint32_t vision_analysis_limit(VNGeneratePersonSegmentationRequestQualityLevel quality)
{
    switch (quality) {
    case VNGeneratePersonSegmentationRequestQualityLevelFast:
        return 512;
    case VNGeneratePersonSegmentationRequestQualityLevelBalanced:
        return 1024;
    default:
        return 2048;
    }
}

/* Runs on the mask queue. Takes ownership of `input`. */
static void vision_segment(struct vision_data *filter, CVPixelBufferRef input, uint64_t frame_id,
                           VNGeneratePersonSegmentationRequestQualityLevel quality)
{
    @autoreleasepool {
        const size_t width = CVPixelBufferGetWidth(input);
        const size_t height = CVPixelBufferGetHeight(input);

        /* The request keeps state between frames, so start over whenever the input changes shape. */
        if (!filter->handler || filter->request.qualityLevel != quality || filter->handler_width != width ||
            filter->handler_height != height) {
            [filter->request release];
            [filter->handler release];
            filter->request = [[VNGeneratePersonSegmentationRequest alloc] init];
            filter->request.qualityLevel = quality;
            filter->request.outputPixelFormat = kCVPixelFormatType_OneComponent8;
            filter->handler = [[VNSequenceRequestHandler alloc] init];
            filter->handler_width = width;
            filter->handler_height = height;
        }

        const uint64_t start = os_gettime_ns();
        NSError *error = nil;
        const BOOL success = [filter->handler performRequests:@[filter->request] onCVPixelBuffer:input error:&error];
        const uint64_t elapsed = os_gettime_ns() - start;

        CVPixelBufferRef output = success ? filter->request.results.firstObject.pixelBuffer : NULL;
        if (output) {
            CVPixelBufferRetain(output);
            pthread_mutex_lock(&filter->mutex);
            if (filter->pending_mask)
                CVPixelBufferRelease(filter->pending_mask);
            filter->pending_mask = output;
            filter->pending_id = frame_id;
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
    atomic_store(&filter->busy, false);
}

/* Reads the frame back from the GPU (downscaled to what the model can use) and hands it to Vision.
 * Only called when the previous request has finished, so requests can never queue up. */
static void vision_dispatch_frame(struct vision_data *filter, gs_texture_t *frame, uint32_t width, uint32_t height,
                                  uint64_t frame_id)
{
    const uint32_t limit = vision_analysis_limit(filter->quality);
    const uint32_t longest = width > height ? width : height;
    uint32_t analysis_width = width;
    uint32_t analysis_height = height;
    gs_texture_t *analysis_texture = frame;

    if (longest > limit) {
        analysis_width = (uint32_t) ((uint64_t) width * limit / longest);
        analysis_height = (uint32_t) ((uint64_t) height * limit / longest);
        if (analysis_width < 2)
            analysis_width = 2;
        if (analysis_height < 2)
            analysis_height = 2;

        gs_texrender_reset(filter->analysis);
        if (!gs_texrender_begin(filter->analysis, analysis_width, analysis_height))
            return;
        struct vec4 clear_color;
        vec4_zero(&clear_color);
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0, 0);
        gs_ortho(0, analysis_width, 0, analysis_height, -100, 100);
        gs_effect_t *scale_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
        gs_effect_set_texture(gs_effect_get_param_by_name(scale_effect, "image"), frame);
        while (gs_effect_loop(scale_effect, "Draw"))
            gs_draw_sprite(frame, 0, analysis_width, analysis_height);
        gs_texrender_end(filter->analysis);
        analysis_texture = gs_texrender_get_texture(filter->analysis);
        if (!analysis_texture)
            return;
    }

    if (filter->stagesurf && (analysis_width != gs_stagesurface_get_width(filter->stagesurf) ||
                              analysis_height != gs_stagesurface_get_height(filter->stagesurf))) {
        gs_stagesurface_destroy(filter->stagesurf);
        filter->stagesurf = NULL;
    }
    if (!filter->stagesurf)
        filter->stagesurf = gs_stagesurface_create(analysis_width, analysis_height, GS_BGRA);
    if (!filter->stagesurf)
        return;

    gs_stage_texture(filter->stagesurf, analysis_texture);
    uint8_t *data;
    uint32_t linesize;
    if (!gs_stagesurface_map(filter->stagesurf, &data, &linesize))
        return;

    /* Copy into a buffer of our own: the staging surface is reused for the next frame while Vision
     * is still reading this one. */
    CVPixelBufferRef input = NULL;
    NSDictionary *attributes = @{(NSString *) kCVPixelBufferIOSurfacePropertiesKey: @ {}};
    CVPixelBufferCreate(kCFAllocatorDefault, analysis_width, analysis_height, kCVPixelFormatType_32BGRA,
                        (CFDictionaryRef) attributes, &input);
    if (input) {
        CVPixelBufferLockBaseAddress(input, 0);
        uint8_t *dest = CVPixelBufferGetBaseAddress(input);
        const size_t dest_linesize = CVPixelBufferGetBytesPerRow(input);
        const size_t row_bytes = (size_t) analysis_width * 4;
        for (uint32_t y = 0; y < analysis_height; y++)
            memcpy(dest + y * dest_linesize, data + (size_t) y * linesize, row_bytes);
        CVPixelBufferUnlockBaseAddress(input, 0);
    }
    gs_stagesurface_unmap(filter->stagesurf);
    if (!input)
        return;

    const VNGeneratePersonSegmentationRequestQualityLevel quality = filter->quality;
    atomic_store(&filter->busy, true);
    dispatch_async(filter->mask_queue, ^{
        vision_segment(filter, input, frame_id, quality);
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

        gs_texture_t *frame = gs_texrender_get_texture(texrender);
        if (frame && !atomic_load(&filter->busy))
            vision_dispatch_frame(filter, frame, width, height, frame_id);
    }
    gs_blend_state_pop();
}

/* Picks up a finished mask from the mask queue, stabilises it against the previous mask and stores
 * it in the mask ring. */
static void vision_upload_mask(struct vision_data *filter)
{
    pthread_mutex_lock(&filter->mutex);
    CVPixelBufferRef mask = filter->pending_mask;
    const uint64_t mask_id = filter->pending_id;
    filter->pending_mask = NULL;
    pthread_mutex_unlock(&filter->mutex);
    if (!mask)
        return;

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
    if (filter->mask_head >= 0 && filter->mask_ids[filter->mask_head])
        previous = gs_texrender_get_texture(filter->masks[filter->mask_head]);

    const int slot = (filter->mask_head + 1) % MASK_RING;
    gs_texrender_t *texrender = filter->masks[slot];
    filter->mask_ids[slot] = 0;

    const bool previous_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(false);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
    gs_texrender_reset(texrender);
    if (gs_texrender_begin(texrender, width, height)) {
        gs_ortho(0, width, 0, height, -100, 100);
        gs_effect_set_texture(filter->mask_param, filter->mask_raw);
        gs_effect_set_texture(filter->mask_prev_param, previous ? previous : filter->mask_raw);
        gs_effect_set_float(filter->smoothing_param, previous ? filter->smoothing : 0.0f);
        while (gs_effect_loop(filter->effect, "Stabilize"))
            gs_draw_sprite(filter->mask_raw, 0, width, height);
        gs_texrender_end(texrender);

        filter->mask_ids[slot] = mask_id;
        filter->mask_head = slot;
        filter->mask_width = width;
        filter->mask_height = height;
    }
    gs_blend_state_pop();
    gs_enable_framebuffer_srgb(previous_srgb);

    /* Follow how late masks arrive and hold the video back by just enough frames to cover it.
     * The delay only moves after the new value has been wanted for a while, so that it does not
     * hunt back and forth (every change repeats or drops a frame). */
    const float latency = (float) (filter->frame_counter - mask_id);
    filter->latency_average = filter->latency_average > 0.0f ? filter->latency_average * 0.95f + latency * 0.05f
                                                             : latency;
    if (filter->last_mask_id && mask_id > filter->last_mask_id) {
        const float interval = (float) (mask_id - filter->last_mask_id);
        filter->interval_average = filter->interval_average > 0.0f
                                       ? filter->interval_average * 0.95f + interval * 0.05f
                                       : interval;
    }
    filter->last_mask_id = mask_id;
    if (filter->sync_auto) {
        int wanted = (int) ceilf(filter->latency_average - 0.25f);
        /* When segmentation is slower than the frame rate, some frames have no mask of their own
         * and get one blended from the masks either side. Wait one more frame for the later one. */
        if (filter->interval_average > 1.3f)
            wanted++;
        if (wanted < 1)
            wanted = 1;
        if (wanted > MAX_SYNC_DELAY)
            wanted = MAX_SYNC_DELAY;
        if (wanted == filter->sync_delay) {
            filter->delay_votes = 0;
        } else if (++filter->delay_votes >= 30) {
            filter->sync_delay = wanted;
            filter->delay_votes = 0;
        }
    }

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
        obs_log(LOG_INFO, "Mask %ux%u, segmentation %.1f ms, mask ready %.2f frames after its frame (sync delay %d)",
                width, height, inference_ms, (double) filter->stats_latency_frames / filter->stats_uploads,
                filter->sync_delay);
        filter->stats_latency_frames = 0;
        filter->stats_uploads = 0;
    }
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
    vision_upload_mask(filter);

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
    gs_texture_t *mask = NULL;
    gs_texture_t *mask_next = NULL;
    float mask_mix = 0.0f;
    uint64_t before_id = 0;
    uint64_t after_id = UINT64_MAX;
    for (int i = 0; i < MASK_RING; i++) {
        const uint64_t id = filter->mask_ids[i];
        if (!id)
            continue;
        gs_texture_t *texture = gs_texrender_get_texture(filter->masks[i]);
        if (id <= shown_id && id >= before_id) {
            before_id = id;
            mask = texture;
        }
        if (id > shown_id && id < after_id) {
            after_id = id;
            mask_next = texture;
        }
    }
    if (mask && mask_next && before_id != shown_id) {
        mask_mix = (float) (shown_id - before_id) / (float) (after_id - before_id);
    } else if (!mask) {
        mask = mask_next;
    }
    if (mask_mix == 0.0f)
        mask_next = mask;

    /* Draw nothing rather than the unmasked source while there is no frame or mask yet: showing
     * the background, even for a few frames, is the one thing this filter must not do. */
    if (!frame || !mask || !filter->mask_width || !filter->mask_height)
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

    /* Refinement taps are one mask pixel apart, but never closer than 1.5 image pixels so that the
     * kernel still spans an edge when the mask is as large as the image. */
    struct vec2 image_texel;
    vec2_set(&image_texel, 1.0f / (float) width, 1.0f / (float) height);
    struct vec2 tap_step;
    vec2_set(&tap_step, 1.0f / (float) filter->mask_width, 1.0f / (float) filter->mask_height);
    if (tap_step.x < image_texel.x * 1.5f)
        tap_step.x = image_texel.x * 1.5f;
    if (tap_step.y < image_texel.y * 1.5f)
        tap_step.y = image_texel.y * 1.5f;

    const bool linear_srgb = gs_get_linear_srgb();
    const bool previous_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(linear_srgb);

    if (linear_srgb)
        gs_effect_set_texture_srgb(filter->image_param, frame);
    else
        gs_effect_set_texture(filter->image_param, frame);
    gs_effect_set_texture(filter->mask_param, mask);
    gs_effect_set_texture(filter->mask_next_param, mask_next);
    gs_effect_set_float(filter->mask_mix_param, mask_mix);
    gs_effect_set_vec2(filter->tap_step_param, &tap_step);
    gs_effect_set_vec2(filter->image_texel_param, &image_texel);
    gs_effect_set_float(filter->edge_lo_param, edge_lo);
    gs_effect_set_float(filter->edge_hi_param, edge_hi);
    gs_effect_set_float(filter->range_scale_param, filter->refinement * filter->refinement * 200.0f);
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

    vision_add_slider(props, "threshold", "Threshold", "Threshold.Description");
    vision_add_slider(props, "softness", "Softness", "Softness.Description");
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
    obs_data_set_default_double(settings, "threshold", 0.5);
    obs_data_set_default_double(settings, "softness", 0.25);
    obs_data_set_default_double(settings, "refinement", 0.5);
    obs_data_set_default_double(settings, "smoothing", 0.5);
    obs_data_set_default_double(settings, "defringe", 0.3);
    obs_data_set_default_bool(settings, "sync_auto", true);
    obs_data_set_default_int(settings, "sync_delay", 1);
}

static void vision_update(void *filter_ptr, obs_data_t *settings)
{
    struct vision_data *filter = filter_ptr;

    filter->quality = obs_data_get_int(settings, "quality");
    filter->threshold = (float) obs_data_get_double(settings, "threshold");
    filter->softness = (float) obs_data_get_double(settings, "softness");
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
    atomic_init(&filter->busy, false);
    pthread_mutex_init(&filter->mutex, NULL);

    /* Performing the segmentation in realtime takes too long and will lag OBS, especially at higher
     * quality modes. As such, defer it to a different thread. The mask arrives a frame or so late,
     * which the sync delay makes up for by holding the video back to match. */
    filter->mask_queue = dispatch_queue_create("Filter mask dispatch queue", NULL);

    obs_enter_graphics();
    for (int i = 0; i < FRAME_RING; i++)
        filter->frames[i] = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    for (int i = 0; i < MASK_RING; i++)
        filter->masks[i] = gs_texrender_create(GS_R8, GS_ZS_NONE);
    filter->analysis = gs_texrender_create(GS_BGRA, GS_ZS_NONE);

    char *file = obs_module_file("alpha_mask.effect");
    filter->effect = gs_effect_create_from_file(file, NULL);
    bfree(file);
    if (filter->effect) {
        filter->image_param = gs_effect_get_param_by_name(filter->effect, "image");
        filter->mask_param = gs_effect_get_param_by_name(filter->effect, "mask");
        filter->mask_prev_param = gs_effect_get_param_by_name(filter->effect, "mask_prev");
        filter->mask_next_param = gs_effect_get_param_by_name(filter->effect, "mask_next");
        filter->mask_mix_param = gs_effect_get_param_by_name(filter->effect, "mask_mix");
        filter->tap_step_param = gs_effect_get_param_by_name(filter->effect, "tap_step");
        filter->image_texel_param = gs_effect_get_param_by_name(filter->effect, "image_texel");
        filter->edge_lo_param = gs_effect_get_param_by_name(filter->effect, "edge_lo");
        filter->edge_hi_param = gs_effect_get_param_by_name(filter->effect, "edge_hi");
        filter->range_scale_param = gs_effect_get_param_by_name(filter->effect, "range_scale");
        filter->defringe_px_param = gs_effect_get_param_by_name(filter->effect, "defringe_px");
        filter->smoothing_param = gs_effect_get_param_by_name(filter->effect, "smoothing");
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

    /* Wait for a request that is still running; it uses the filter data. */
    dispatch_sync(filter->mask_queue, ^{
    });
    dispatch_release(filter->mask_queue);
    [filter->request release];
    [filter->handler release];
    if (filter->pending_mask)
        CVPixelBufferRelease(filter->pending_mask);
    pthread_mutex_destroy(&filter->mutex);

    obs_enter_graphics();
    for (int i = 0; i < FRAME_RING; i++)
        gs_texrender_destroy(filter->frames[i]);
    for (int i = 0; i < MASK_RING; i++)
        gs_texrender_destroy(filter->masks[i]);
    gs_texrender_destroy(filter->analysis);
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
