# macOS Background Removal for OBS

## Introduction

This plugin provides an OBS filter to remove a persons background using macOS' built-in [Vision](https://developer.apple.com/documentation/vision) API.

This plugin requires OBS 31.1 or later.

## Getting started

To get started, download and run the installer from the [releases](https://github.com/gxalpha/obs-mac-backgroundremoval/releases) page.
It's currently not signed or notarized.
Because of that, follow these installation instructions:
- Open the `.pkg` file that you have downloaded. You will get prompted with a warning, select "Done" (do not move to trash).
- Open System Settings. Navigate to "Privacy & Security" → "Security". Near the bottom, you will find an information that `obs-mac-backgroundremoval-[...].pkg` was blocked.
- Click "Open Anyway" and confirm using your administrator login.

The filter can be found in the filters window under "Effect Filters".
It can be used on any type of source, not just Video Capture Devices!

## Settings

- **Quality**: which Vision model to use. Balanced produces a 512x384 mask in roughly 10-30 ms on Apple Silicon; Accurate produces a larger mask but takes longer than one frame at 30 fps, so some frames get a mask blended from their neighbours.
- **Processor** (macOS 14 and later): where the model runs. On the GPU it competes with OBS for rendering and encoding time.
- **Threshold** / **Edge softness**: where the cut is made and how wide the transition is.
- **Outline smoothing**: rounds off the outline of the low resolution mask before it is cut.
- **Edge refinement**: a guided filter that pulls the mask edge onto real edges in the picture.
- **Temporal smoothing**: holds the mask steady wherever the picture itself has not changed, which removes flicker without making moving edges trail.
- **Defringe**: replaces leftover background colour on the edge with colour from just inside the subject.
- **Automatic mask sync** / **Mask sync delay**: segmentation runs on another thread and finishes a frame or more after the frame it was started for. The filter holds the video back by that many frames so that every frame is shown with its own mask, which removes the halo that otherwise trails a moving subject. Automatic picks the smallest delay that works; add the same delay to your audio if lip sync matters.

The OBS log reports the mask size, segmentation time and how late masks arrive every 600 masks.

## License and Thanks
This plugin is licensed under the terms of the General Public License, Version 2. You can find the full text in the `LICENSE` file in this repository.

Huge credits also go to pkv from OBS who implemented a similar filter in OBS for NVIDIA GPUs. His code was referenced heavily in the creation of this plugin, especially during the early stages.

The build system is based on the [OBS Plugin Template](https://github.com/obsproject/obs-plugintemplate). Its build instructions also apply to this plugin.
