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
- **Background**: what the filter puts behind the subject. None leaves it transparent for the layers underneath. Blur, Colour and Image make the filter draw the background itself, which lets it do what a separate layer cannot: blur the room without smearing the subject into it, and blend the rim of the subject into the new background.
- **Background blur**: strength of the blur.
- **Light wrap**: lets the colour of the background spill a little over the rim of the subject, so the cut-out sits in its background and leftover fringes disappear. Only available with a background.
- **Model** (only shown when the matting model is installed, see below): Apple Vision or Robust Video Matting.
- **Automatic mask sync** / **Mask sync delay**: segmentation runs on another thread and finishes a frame or more after the frame it was started for. The filter holds the video back by that many frames so that every frame is shown with its own mask, which removes the halo that otherwise trails a moving subject. Automatic picks the smallest delay that works; add the same delay to your audio if lip sync matters.

The OBS log reports the mask size, segmentation time and how late masks arrive every 600 masks.

## Optional matting model

Vision answers "person or not" per pixel. [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) instead estimates real transparency, so hair and motion blur come out partly see-through rather than cut, and it remembers earlier frames, which keeps the edge steadier than any smoothing after the fact. It takes about 20 ms per frame on an M3 Pro (Neural Engine), against about 11 ms for Vision Balanced.

The model is not part of this plugin. To use it, download `rvm_mobilenetv3_1280x720_s0.375_fp16.mlmodel` from the Robust Video Matting [releases](https://github.com/PeterL1n/RobustVideoMatting/releases/tag/v1.0.0) and put it in a `models` folder inside the plugin's resources (`obs-mac-backgroundremoval.plugin/Contents/Resources/models/`, or `data/models/` before building). A **Model** setting then appears in the filter. The model is compiled the first time it is used. Robust Video Matting's code is GPL-3.0; its authors have not stated a licence for the trained weights, so check that before redistributing them.

The model works on a 1280x720 picture, so sources with another shape are stretched to it for the analysis.

## License and Thanks
This plugin is licensed under the terms of the General Public License, Version 2. You can find the full text in the `LICENSE` file in this repository.

Huge credits also go to pkv from OBS who implemented a similar filter in OBS for NVIDIA GPUs. His code was referenced heavily in the creation of this plugin, especially during the early stages.

The build system is based on the [OBS Plugin Template](https://github.com/obsproject/obs-plugintemplate). Its build instructions also apply to this plugin.
