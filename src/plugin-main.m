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
#include <graphics/vec4.h>
#include <util/platform.h>
#include <util/threading.h>
#include <math.h>
#include <stdatomic.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreML/CoreML.h>
#include <ImageIO/ImageIO.h>
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

/* The optional matting model. It is not part of the plugin; when the file is present in the plugin's
 * data it can be chosen instead of Vision. Its input size is fixed when the model is exported. */
#define MATTING_MODEL_FILE "rvm_mobilenetv3_1280x720_s0.375_fp16.mlmodel"
#define MATTING_WIDTH 1280
#define MATTING_HEIGHT 720
#define BLUR_DIVISOR 4
#define BLUR_ROUNDS 3

enum vision_model {
    VISION_MODEL_APPLE = 0,
    VISION_MODEL_MATTING = 1,
};

enum vision_background {
    VISION_BACKGROUND_NONE = 0,
    VISION_BACKGROUND_BLUR = 1,
    VISION_BACKGROUND_COLOR = 2,
    VISION_BACKGROUND_IMAGE = 3,
};

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
    gs_eparam_t *edge_linear_param;
    gs_eparam_t *backdrop_param;
    gs_eparam_t *backdrop_soft_param;
    gs_eparam_t *backdrop_on_param;
    gs_eparam_t *light_wrap_param;
    gs_eparam_t *wrap_px_param;
    gs_eparam_t *uv_scale_param;
    gs_eparam_t *uv_offset_param;
    gs_eparam_t *prepare_weighted_param;
    gs_eparam_t *blur_step_param;

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
    enum vision_model model;
    enum vision_background background;
    float background_blur;
    uint32_t background_color;
    float light_wrap;

    /* The background drawn behind the subject: a picture or colour at full size, and the blurred
     * quarter size copy that is the background itself in blur mode and the light wrap colour in
     * the others. */
    char *background_path;
    gs_texture_t *background_image;
    gs_texrender_t *backdrop_sharp;
    gs_texrender_t *blur[2];
    bool backdrop_dirty;
    uint32_t backdrop_width;
    uint32_t backdrop_height;
    uint64_t backdrop_frame_id;

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
    MLModel *matting;
    enum vision_processor matting_processor;
    MLFeatureValue *matting_state[4];
    /* Set by the mask queue when the matting model cannot be used, so that Vision takes over, and
     * by the render thread when the model's memory of earlier frames no longer applies. */
    atomic_bool matting_failed;
    atomic_bool matting_reset;

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
    filter->backdrop_frame_id = 0;
    atomic_store(&filter->matting_reset, true);
}

/* Reads a picture file into a texture. Call inside the graphics context. */
static gs_texture_t *vision_load_image(const char *path)
{
    gs_texture_t *texture = NULL;
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        CGImageSourceRef source = CGImageSourceCreateWithURL((CFURLRef) url, NULL);
        CGImageRef image = source ? CGImageSourceCreateImageAtIndex(source, 0, NULL) : NULL;
        if (image) {
            /* Nothing is gained from a picture larger than the video, so keep textures modest. */
            size_t width = CGImageGetWidth(image);
            size_t height = CGImageGetHeight(image);
            const size_t longest = width > height ? width : height;
            if (longest > 4096) {
                width = width * 4096 / longest;
                height = height * 4096 / longest;
            }
            uint8_t *pixels = (width && height) ? bzalloc(width * height * 4) : NULL;
            CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            CGContextRef context = pixels ? CGBitmapContextCreate(pixels, width, height, 8, width * 4, colorspace,
                                                                  kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little)
                                          : NULL;
            if (context) {
                CGContextDrawImage(context, CGRectMake(0, 0, width, height), image);
                CGContextRelease(context);
                texture = gs_texture_create((uint32_t) width, (uint32_t) height, GS_BGRA, 1,
                                            (const uint8_t **) &pixels, 0);
            }
            CGColorSpaceRelease(colorspace);
            bfree(pixels);
            CGImageRelease(image);
        }
        if (source)
            CFRelease(source);
    }
    return texture;
}

static bool vision_uses_matting(struct vision_data *filter)
{
    return filter->model == VISION_MODEL_MATTING && !atomic_load(&filter->matting_failed);
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

/* Runs on the mask queue. Returns the mask, retained, or NULL. */
static CVPixelBufferRef vision_run_vision(struct vision_data *filter, CVPixelBufferRef input,
                                          VNGeneratePersonSegmentationRequestQualityLevel quality,
                                          enum vision_processor processor, NSError **error)
{
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

    if (![filter->handler performRequests:@[filter->request] onCVPixelBuffer:input error:error])
        return NULL;
    CVPixelBufferRef output = filter->request.results.firstObject.pixelBuffer;
    if (output)
        CVPixelBufferRetain(output);
    return output;
}

static void vision_clear_matting_state(struct vision_data *filter)
{
    for (int i = 0; i < 4; i++) {
        [filter->matting_state[i] release];
        filter->matting_state[i] = nil;
    }
}

/* Runs on the mask queue. Loads the matting model, compiling it the first time it is used and
 * keeping the compiled copy in the plugin's configuration folder. */
static bool vision_load_matting(struct vision_data *filter, enum vision_processor processor, NSError **error)
{
    [filter->matting release];
    filter->matting = nil;
    vision_clear_matting_state(filter);
    filter->matting_processor = processor;

    char *source = obs_module_file("models/" MATTING_MODEL_FILE);
    if (!source)
        return false;
    NSURL *source_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:source]];
    bfree(source);

    NSURL *compiled_url = nil;
    char *cache = obs_module_config_path(MATTING_MODEL_FILE "c");
    if (cache) {
        compiled_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:cache]];
        bfree(cache);
    }
    if (!compiled_url || !os_file_exists(compiled_url.path.UTF8String)) {
        NSURL *fresh = [MLModel compileModelAtURL:source_url error:error];
        if (!fresh)
            return false;
        if (compiled_url) {
            NSFileManager *files = [NSFileManager defaultManager];
            [files createDirectoryAtURL:[compiled_url URLByDeletingLastPathComponent] withIntermediateDirectories:YES
                             attributes:nil
                                  error:NULL];
            if (![files moveItemAtURL:fresh toURL:compiled_url error:NULL])
                compiled_url = fresh;
        } else {
            compiled_url = fresh;
        }
    }

    MLModelConfiguration *configuration = [[[MLModelConfiguration alloc] init] autorelease];
    configuration.computeUnits = MLComputeUnitsAll;
    if (processor == VISION_PROCESSOR_GPU) {
        configuration.computeUnits = MLComputeUnitsCPUAndGPU;
    } else if (processor == VISION_PROCESSOR_CPU) {
        configuration.computeUnits = MLComputeUnitsCPUOnly;
    } else if (processor == VISION_PROCESSOR_NEURAL_ENGINE) {
        if (@available(macOS 13.0, *))
            configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    }

    MLModel *model = [MLModel modelWithContentsOfURL:compiled_url configuration:configuration error:error];
    if (!model)
        return false;
    MLImageConstraint *constraint = model.modelDescription.inputDescriptionsByName[@"src"].imageConstraint;
    if (constraint.pixelsWide != MATTING_WIDTH || constraint.pixelsHigh != MATTING_HEIGHT ||
        !model.modelDescription.outputDescriptionsByName[@"pha"])
        return false;
    filter->matting = [model retain];
    return true;
}

/* Runs on the mask queue. Returns the alpha matte, retained, or NULL. The model carries what it
 * saw in earlier frames along in four state tensors, which is what keeps its matte steady. */
static CVPixelBufferRef vision_run_matting(struct vision_data *filter, CVPixelBufferRef input,
                                           enum vision_processor processor, NSError **error)
{
    if (!filter->matting || filter->matting_processor != processor) {
        if (!vision_load_matting(filter, processor, error))
            return NULL;
    }
    if (atomic_exchange(&filter->matting_reset, false))
        vision_clear_matting_state(filter);

    NSMutableDictionary<NSString *, MLFeatureValue *> *inputs = [NSMutableDictionary dictionary];
    inputs[@"src"] = [MLFeatureValue featureValueWithPixelBuffer:input];
    for (int i = 0; i < 4; i++) {
        if (filter->matting_state[i])
            inputs[[NSString stringWithFormat:@"r%di", i + 1]] = filter->matting_state[i];
    }
    MLDictionaryFeatureProvider *provider = [[[MLDictionaryFeatureProvider alloc] initWithDictionary:inputs
                                                                                               error:error] autorelease];
    id<MLFeatureProvider> result = provider ? [filter->matting predictionFromFeatures:provider error:error] : nil;
    if (!result)
        return NULL;

    for (int i = 0; i < 4; i++) {
        MLFeatureValue *state = [result featureValueForName:[NSString stringWithFormat:@"r%do", i + 1]];
        [state retain];
        [filter->matting_state[i] release];
        filter->matting_state[i] = state;
    }
    CVPixelBufferRef output = [result featureValueForName:@"pha"].imageBufferValue;
    if (output)
        CVPixelBufferRetain(output);
    return output;
}

/* Runs on the mask queue. Takes ownership of `input`. */
static void vision_segment(struct vision_data *filter, CVPixelBufferRef input, uint64_t frame_id, bool matting,
                           VNGeneratePersonSegmentationRequestQualityLevel quality, enum vision_processor processor)
{
    @autoreleasepool {
        const uint64_t start = os_gettime_ns();
        NSError *error = nil;
        CVPixelBufferRef output = matting ? vision_run_matting(filter, input, processor, &error)
                                          : vision_run_vision(filter, input, quality, processor, &error);
        const uint64_t elapsed = os_gettime_ns() - start;

        if (output) {
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
        } else if (matting) {
            atomic_store(&filter->matting_failed, true);
            obs_log(LOG_ERROR, "The matting model could not be used, falling back to Vision: %s",
                    error ? error.localizedDescription.UTF8String : "model missing or not the expected shape");
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
    const bool matting = vision_uses_matting(filter);
    if (matting) {
        guide_width = MATTING_WIDTH;
        guide_height = MATTING_HEIGHT;
    } else if (longest > limit) {
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
        vision_segment(filter, input, frame_id, matting, quality, processor);
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

/* Draws `source` into `target` with one of the effect's techniques. */
static bool vision_pass(struct vision_data *filter, gs_texrender_t *target, uint32_t width, uint32_t height,
                        const char *technique, gs_texture_t *source)
{
    gs_texrender_reset(target);
    if (!gs_texrender_begin(target, width, height))
        return false;
    gs_ortho(0, width, 0, height, -100, 100);
    gs_effect_set_texture(filter->image_param, source);
    while (gs_effect_loop(filter->effect, technique))
        gs_draw_sprite(source, 0, width, height);
    gs_texrender_end(target);
    return true;
}

/* Blurs `source` into blur[0] at a quarter of the output size. When `weighted`, the subject is
 * left out of the blur using the mask coefficients. */
static bool vision_blur(struct vision_data *filter, gs_texture_t *source, uint32_t width, uint32_t height,
                        bool weighted, gs_texture_t *coeff, gs_texture_t *coeff_next, float coeff_mix)
{
    uint32_t blur_width = width / BLUR_DIVISOR;
    uint32_t blur_height = height / BLUR_DIVISOR;
    if (blur_width < 2)
        blur_width = 2;
    if (blur_height < 2)
        blur_height = 2;

    struct vec2 texel;
    vec2_set(&texel, 1.0f / (float) gs_texture_get_width(source), 1.0f / (float) gs_texture_get_height(source));
    struct vec2 scale;
    vec2_set(&scale, 1.0f, 1.0f);
    struct vec2 offset;
    vec2_zero(&offset);
    gs_effect_set_vec2(filter->image_texel_param, &texel);
    gs_effect_set_vec2(filter->uv_scale_param, &scale);
    gs_effect_set_vec2(filter->uv_offset_param, &offset);
    gs_effect_set_float(filter->prepare_weighted_param, weighted ? 1.0f : 0.0f);
    gs_effect_set_texture(filter->coeff_param, weighted ? coeff : source);
    gs_effect_set_texture(filter->coeff_next_param, weighted ? coeff_next : source);
    gs_effect_set_float(filter->coeff_mix_param, weighted ? coeff_mix : 0.0f);
    if (!vision_pass(filter, filter->blur[0], blur_width, blur_height, "Backdrop", source))
        return false;

    /* The step is in quarter size pixels of a 720 line picture, so the blur looks the same at
     * any resolution. Each round is wider than the last, which keeps a strong blur smooth. */
    const float base = (0.25f + 1.75f * filter->background_blur) * (float) blur_height / 180.0f;
    for (int round = 1; round <= BLUR_ROUNDS; round++) {
        struct vec2 step;
        vec2_set(&step, base * (float) round / (float) blur_width, 0.0f);
        gs_effect_set_vec2(filter->blur_step_param, &step);
        if (!vision_pass(filter, filter->blur[1], blur_width, blur_height, "Blur",
                         gs_texrender_get_texture(filter->blur[0])))
            return false;
        vec2_set(&step, 0.0f, base * (float) round / (float) blur_height);
        gs_effect_set_vec2(filter->blur_step_param, &step);
        if (!vision_pass(filter, filter->blur[0], blur_width, blur_height, "Blur",
                         gs_texrender_get_texture(filter->blur[1])))
            return false;
    }
    return true;
}

/* Brings the background textures up to date for the frame about to be shown. On success
 * `backdrop` is what goes behind the subject and `soft` what the light wrap takes its colour from. */
static bool vision_update_backdrop(struct vision_data *filter, gs_texture_t *frame, uint64_t frame_id, uint32_t width,
                                   uint32_t height, gs_texture_t *coeff, gs_texture_t *coeff_next, float coeff_mix,
                                   gs_texture_t **backdrop, gs_texture_t **soft)
{
    enum vision_background mode = filter->background;
    if (mode == VISION_BACKGROUND_IMAGE && !filter->background_image)
        mode = VISION_BACKGROUND_NONE;
    if (mode == VISION_BACKGROUND_NONE)
        return false;

    if (width != filter->backdrop_width || height != filter->backdrop_height) {
        filter->backdrop_width = width;
        filter->backdrop_height = height;
        filter->backdrop_dirty = true;
    }

    const bool previous_srgb = gs_framebuffer_srgb_enabled();
    gs_enable_framebuffer_srgb(false);
    gs_blend_state_push();
    gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

    bool ready = true;
    if (mode == VISION_BACKGROUND_BLUR) {
        /* Follows the picture, so it is redone for every frame shown. */
        if (filter->backdrop_frame_id != frame_id) {
            ready = vision_blur(filter, frame, width, height, true, coeff, coeff_next, coeff_mix);
            filter->backdrop_frame_id = ready ? frame_id : 0;
        }
        filter->backdrop_dirty = true;
    } else if (filter->backdrop_dirty) {
        if (mode == VISION_BACKGROUND_COLOR) {
            gs_texrender_reset(filter->backdrop_sharp);
            ready = gs_texrender_begin(filter->backdrop_sharp, 4, 4);
            if (ready) {
                struct vec4 color;
                vec4_from_rgba(&color, filter->background_color);
                color.w = 1.0f;
                gs_clear(GS_CLEAR_COLOR, &color, 0, 0);
                gs_texrender_end(filter->backdrop_sharp);
            }
        } else {
            /* Scale the picture to fill the frame, cropping whichever side is too long. */
            gs_texture_t *picture = filter->background_image;
            const float picture_aspect = (float) gs_texture_get_width(picture) / (float) gs_texture_get_height(picture);
            const float frame_aspect = (float) width / (float) height;
            struct vec2 scale;
            vec2_set(&scale, 1.0f, 1.0f);
            if (picture_aspect > frame_aspect)
                scale.x = frame_aspect / picture_aspect;
            else
                scale.y = picture_aspect / frame_aspect;
            struct vec2 offset;
            vec2_set(&offset, (1.0f - scale.x) / 2.0f, (1.0f - scale.y) / 2.0f);
            struct vec2 texel;
            vec2_zero(&texel);
            gs_effect_set_vec2(filter->image_texel_param, &texel);
            gs_effect_set_vec2(filter->uv_scale_param, &scale);
            gs_effect_set_vec2(filter->uv_offset_param, &offset);
            gs_effect_set_float(filter->prepare_weighted_param, 0.0f);
            gs_effect_set_texture(filter->coeff_param, picture);
            gs_effect_set_texture(filter->coeff_next_param, picture);
            gs_effect_set_float(filter->coeff_mix_param, 0.0f);
            ready = vision_pass(filter, filter->backdrop_sharp, width, height, "Backdrop", picture);
            if (ready)
                ready = vision_blur(filter, gs_texrender_get_texture(filter->backdrop_sharp), width, height, false,
                                    NULL, NULL, 0.0f);
        }
        filter->backdrop_dirty = !ready;
    }

    gs_blend_state_pop();
    gs_enable_framebuffer_srgb(previous_srgb);
    if (!ready)
        return false;

    gs_texture_t *sharp = gs_texrender_get_texture(filter->backdrop_sharp);
    gs_texture_t *blurred = gs_texrender_get_texture(filter->blur[0]);
    *backdrop = mode == VISION_BACKGROUND_BLUR ? blurred : sharp;
    *soft = mode == VISION_BACKGROUND_COLOR ? sharp : blurred;
    return *backdrop && *soft;
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

    gs_texture_t *backdrop = NULL;
    gs_texture_t *backdrop_soft = NULL;
    const bool composite = vision_update_backdrop(filter, frame, shown_id, width, height, coeff, coeff_next, coeff_mix,
                                                  &backdrop, &backdrop_soft);

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
    /* A matting model gives real transparency (hair, motion blur), so its matte is used as it is.
     * The threshold only trims it: by default just the faint haze at the bottom of the range. */
    const bool matte = vision_uses_matting(filter);
    if (matte) {
        const float shift = (filter->threshold - 0.5f) * 0.8f;
        edge_lo = 0.03f + shift;
        edge_hi = 0.97f + shift;
        if (edge_lo < 0.0f)
            edge_lo = 0.0f;
        if (edge_hi > 1.0f)
            edge_hi = 1.0f;
        if (edge_hi < edge_lo + 0.05f)
            edge_hi = edge_lo + 0.05f;
    }

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
    gs_effect_set_float(filter->edge_linear_param, matte ? 1.0f : 0.0f);
    gs_effect_set_texture(filter->backdrop_param, composite ? backdrop : frame);
    gs_effect_set_texture(filter->backdrop_soft_param, composite ? backdrop_soft : frame);
    gs_effect_set_float(filter->backdrop_on_param, composite ? 1.0f : 0.0f);
    gs_effect_set_float(filter->light_wrap_param, filter->light_wrap * 0.8f);
    gs_effect_set_float(filter->wrap_px_param, (float) height / 90.0f);

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

static bool vision_model_modified(obs_properties_t *props, obs_property_t *, obs_data_t *settings)
{
    const bool matting = obs_data_get_int(settings, "model") == VISION_MODEL_MATTING;
    obs_property_set_visible(obs_properties_get(props, "quality"), !matting);
    obs_property_set_visible(obs_properties_get(props, "softness"), !matting);
    return true;
}

static bool vision_background_modified(obs_properties_t *props, obs_property_t *, obs_data_t *settings)
{
    const long long mode = obs_data_get_int(settings, "background");
    obs_property_set_visible(obs_properties_get(props, "background_blur"), mode != VISION_BACKGROUND_NONE);
    obs_property_set_visible(obs_properties_get(props, "background_color"), mode == VISION_BACKGROUND_COLOR);
    obs_property_set_visible(obs_properties_get(props, "background_image"), mode == VISION_BACKGROUND_IMAGE);
    obs_property_set_visible(obs_properties_get(props, "light_wrap"), mode != VISION_BACKGROUND_NONE);
    return true;
}

static obs_properties_t *vision_properties(void *)
{
    obs_properties_t *props = obs_properties_create();
    obs_property_t *list;

    /* Only offered when the model file has been added to the plugin. */
    char *matting_model = obs_module_file("models/" MATTING_MODEL_FILE);
    if (matting_model) {
        bfree(matting_model);
        list = obs_properties_add_list(props, "model", obs_module_text("Model"), OBS_COMBO_TYPE_LIST,
                                       OBS_COMBO_FORMAT_INT);
        obs_property_set_long_description(list, obs_module_text("Model.Description"));
        obs_property_list_add_int(list, obs_module_text("Model.Vision"), VISION_MODEL_APPLE);
        obs_property_list_add_int(list, obs_module_text("Model.Matting"), VISION_MODEL_MATTING);
        obs_property_set_modified_callback(list, vision_model_modified);
    }

    list = obs_properties_add_list(props, "quality", obs_module_text("Quality"), OBS_COMBO_TYPE_LIST,
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

    list = obs_properties_add_list(props, "background", obs_module_text("Background"), OBS_COMBO_TYPE_LIST,
                                   OBS_COMBO_FORMAT_INT);
    obs_property_set_long_description(list, obs_module_text("Background.Description"));
    obs_property_list_add_int(list, obs_module_text("Background.None"), VISION_BACKGROUND_NONE);
    obs_property_list_add_int(list, obs_module_text("Background.Blur"), VISION_BACKGROUND_BLUR);
    obs_property_list_add_int(list, obs_module_text("Background.Color"), VISION_BACKGROUND_COLOR);
    obs_property_list_add_int(list, obs_module_text("Background.Image"), VISION_BACKGROUND_IMAGE);
    obs_property_set_modified_callback(list, vision_background_modified);
    obs_properties_add_color(props, "background_color", obs_module_text("BackgroundColor"));
    obs_properties_add_path(props, "background_image", obs_module_text("BackgroundImage"), OBS_PATH_FILE,
                            "Images (*.png *.jpg *.jpeg *.bmp *.tga *.gif)", NULL);
    vision_add_slider(props, "background_blur", "BackgroundBlur", "BackgroundBlur.Description");
    vision_add_slider(props, "light_wrap", "LightWrap", "LightWrap.Description");

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
    obs_data_set_default_int(settings, "model", VISION_MODEL_APPLE);
    obs_data_set_default_int(settings, "background", VISION_BACKGROUND_NONE);
    obs_data_set_default_double(settings, "background_blur", 0.5);
    obs_data_set_default_int(settings, "background_color", 0xFF202020);
    obs_data_set_default_double(settings, "light_wrap", 0.3);
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
    filter->background_blur = (float) obs_data_get_double(settings, "background_blur");
    filter->background_color = (uint32_t) obs_data_get_int(settings, "background_color");
    filter->light_wrap = (float) obs_data_get_double(settings, "light_wrap");
    filter->background = (enum vision_background) obs_data_get_int(settings, "background");
    filter->backdrop_dirty = true;

    const enum vision_model model = (enum vision_model) obs_data_get_int(settings, "model");
    if (model != filter->model) {
        /* Give the matting model another chance whenever it is chosen again. */
        filter->model = model;
        atomic_store(&filter->matting_failed, false);
        filter->warmup = SYNC_WARMUP;
    }

    const char *path = obs_data_get_string(settings, "background_image");
    if (filter->background != VISION_BACKGROUND_IMAGE)
        path = "";
    if (!filter->background_path || strcmp(path, filter->background_path) != 0) {
        bfree(filter->background_path);
        filter->background_path = bstrdup(path);
        obs_enter_graphics();
        gs_texture_destroy(filter->background_image);
        filter->background_image = *path ? vision_load_image(path) : NULL;
        obs_leave_graphics();
        if (*path && !filter->background_image)
            obs_log(LOG_WARNING, "Could not load the background image '%s'", path);
    }

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
    atomic_init(&filter->matting_failed, false);
    atomic_init(&filter->matting_reset, false);
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
    filter->backdrop_sharp = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
    filter->blur[0] = gs_texrender_create(GS_RGBA16F, GS_ZS_NONE);
    filter->blur[1] = gs_texrender_create(GS_RGBA16F, GS_ZS_NONE);

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
        filter->edge_linear_param = gs_effect_get_param_by_name(filter->effect, "edge_linear");
        filter->backdrop_param = gs_effect_get_param_by_name(filter->effect, "backdrop");
        filter->backdrop_soft_param = gs_effect_get_param_by_name(filter->effect, "backdrop_soft");
        filter->backdrop_on_param = gs_effect_get_param_by_name(filter->effect, "backdrop_on");
        filter->light_wrap_param = gs_effect_get_param_by_name(filter->effect, "light_wrap");
        filter->wrap_px_param = gs_effect_get_param_by_name(filter->effect, "wrap_px");
        filter->uv_scale_param = gs_effect_get_param_by_name(filter->effect, "uv_scale");
        filter->uv_offset_param = gs_effect_get_param_by_name(filter->effect, "uv_offset");
        filter->prepare_weighted_param = gs_effect_get_param_by_name(filter->effect, "prepare_weighted");
        filter->blur_step_param = gs_effect_get_param_by_name(filter->effect, "blur_step");
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
    [filter->matting release];
    vision_clear_matting_state(filter);
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
    gs_texrender_destroy(filter->backdrop_sharp);
    gs_texrender_destroy(filter->blur[0]);
    gs_texrender_destroy(filter->blur[1]);
    gs_texture_destroy(filter->background_image);
    if (filter->stagesurf) {
        gs_stagesurface_destroy(filter->stagesurf);
    }
    if (filter->mask_raw) {
        gs_texture_destroy(filter->mask_raw);
    }
    gs_effect_destroy(filter->effect);
    obs_leave_graphics();
    bfree(filter->background_path);
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
