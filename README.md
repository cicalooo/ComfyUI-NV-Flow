# ComfyUI NV Flow

Self-contained NVIDIA GPU frame interpolation and image upscaling for ComfyUI.
The RIFE 4.9 weights ship in the repository and the pack uses only the CUDA
PyTorch runtime already supplied by ComfyUI. There are no model downloads,
install scripts, TensorRT engine builds, or extra Python dependencies.

## Nodes

- **Load RIFE (NV Flow)** loads the bundled RIFE 4.9 model and selects fp16 or
  fp32 compute.
- **RIFE Interpolate (NV Flow)** interpolates an image sequence in GPU batches,
  supports 2x-16x frame multiplication, optional bidirectional ensemble, scene
  cut protection, and automatic batch reduction after a CUDA out-of-memory.
- **RIFE Video FPS (NV Flow)** reads the input video's frame rate in the backend
  and returns a video at either that rate multiplied by 1x-16x or a custom target
  rate. Audio and video metadata are preserved.
- **CUDA Detail Upscale (NV Flow)** performs bicubic CUDA resizing followed by
  restrained luminance-aware detail recovery. It supports scale or exact-size
  modes and preserves alpha.
- **Long Video Process (NV Flow)** streams a `VIDEO` through RIFE, upscaling, or
  both without holding the complete clip in RAM. It writes resumable temporary
  chunks, uses H.264 NVENC by default, preserves audio, and returns a file-backed
  `VIDEO` for ComfyUI's Save Video node.

The loader is separate from interpolation so one loaded model can be reused by
multiple branches, matching normal ComfyUI model-loader design.

## Small and long videos

NV Flow has two processing paths. The IMAGE-based RIFE and upscale nodes are
the small-video side: they return frame tensors that can be connected to any
IMAGE node, but the complete input and output must fit in system RAM. Use them
for images and short clips where that composability is useful.

Use **Long Video Process (NV Flow)** for clips measured in minutes, high output
resolutions or frame rates, and any workflow where full-frame tensors would use
too much RAM. It decodes, processes, and encodes bounded batches. In combined
mode it runs RIFE before upscaling so interpolation stays at source resolution.

The long-video node supports source-FPS multiplication or a custom target FPS.
It uses source timestamps when scheduling frames, including variable-rate input,
and retains the first and last frames. H.264 NVENC is the default encoder;
HEVC NVENC and software H.264 are also available. Quality and encoder speed are
configured on the node.

Work is encoded in temporary chunks. If execution is cancelled or fails,
completed chunks are reused when the same job is queued again during the same
ComfyUI session. ComfyUI clears its temporary directory when it restarts, so
resume does not survive a restart. A completed result is returned as `VIDEO`;
connect it to the standard **Save Video** node to choose its permanent filename.
Keep at least 2 GiB free on the ComfyUI temporary drive in addition to space for
the encoded result. MP4 output does not preserve alpha; use the IMAGE nodes when
an alpha channel is required.

Long-video jobs report verbose progress in the ComfyUI console. Processing,
resume scanning, video assembly, and audio assembly each show completed work,
rate, elapsed time, and estimated time remaining. Estimates settle after the
first few batches and adjust continuously to the measured speed.

## Install

Clone this folder into `ComfyUI/custom_nodes` and restart ComfyUI. No additional
commands or downloads are required. An NVIDIA GPU and a CUDA-enabled ComfyUI
PyTorch build are required.

## Example workflows

The `example_workflows` folder contains importable examples for every path:

- `nv_flow_small_upscale.json` — tensor-based CUDA upscale with source audio and FPS.
- `nv_flow_small_rife_video.json` — VIDEO-based RIFE with backend FPS detection and a custom target FPS.
- `nv_flow_small_upscale_then_rife.json` — tensor-based upscale followed by frame-batch RIFE; set Create Video FPS to source FPS multiplied by the RIFE multiplier.
- `nv_flow_long_upscale.json` — bounded-memory streaming upscale.
- `nv_flow_long_rife.json` — bounded-memory streaming RIFE with automatic source FPS detection.
- `nv_flow_long_rife_then_upscale.json` — bounded-memory RIFE followed by upscale, using a custom target FPS.

Select the input video and adjust output dimensions before queueing. The long
examples default to H.264 NVENC for NVIDIA GPUs and return a temporary `VIDEO`
to the standard Save Video node.

## Output frame count

For `N` inputs and multiplier `M`, interpolation returns
`(N - 1) * M + 1` frames. This preserves both endpoints without duplicating the
last frame.

## Credits

RIFE architecture and weights are credited in `THIRD_PARTY_NOTICES.md`.

## License

NV Flow is licensed under the Apache License 2.0. Bundled RIFE-derived code and
weights retain their upstream MIT notices in `THIRD_PARTY_NOTICES.md`.
