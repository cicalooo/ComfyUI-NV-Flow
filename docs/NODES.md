# Nodes and controls

## Load Video From Path (NV Flow)

Creates a streamable `VIDEO` directly from an absolute path on the machine
running ComfyUI. This bypasses browser upload and avoids copying a large source
into ComfyUI's input directory. It is the recommended loader for the long-video
examples.

The path must be absolute and must identify an existing file. Both Windows paths
such as `D:\Videos\source.mp4` and absolute POSIX paths are accepted on their
respective platforms. The node tracks file size and modification time so replacing
or modifying the source invalidates ComfyUI's cached result.

This node reads files with the permissions of the ComfyUI server process. Only
load trusted paths, and remove private local paths before sharing a workflow.

## Load RIFE (NV Flow)

Loads the bundled RIFE 4.9 weights. Use `fp16` for lower VRAM use and faster
processing on supported NVIDIA GPUs; use `fp32` when maximum numerical precision
is more important. A single loaded model can feed multiple interpolation nodes.

## RIFE Interpolate (NV Flow)

Interpolates an IMAGE sequence in GPU batches.

- Supports 2x-16x frame multiplication.
- Optional bidirectional ensemble can improve consistency at additional cost.
- Motion scaling adjusts the interpolation timing curve.
- Scene-cut protection avoids inventing transition frames across detected cuts.
- Batch size is reduced automatically after a CUDA out-of-memory error.

For `N` input frames and multiplier `M`, output contains
`(N - 1) * M + 1` frames. Both endpoints are retained without duplicating the
last frame.

## RIFE Video FPS (NV Flow)

Accepts a `VIDEO`, detects its frame rate in the backend, and preserves its audio
and metadata. Choose either:

- `multiplier` to multiply the detected input rate by 1x-16x; or
- `target_fps` to request a custom output rate.

Timestamp-aware scheduling handles variable-rate sources and retains the first
and last frames.

## CUDA Detail Upscale (NV Flow)

Upscales IMAGE tensors by scale factor or exact dimensions. It supports automatic
source analysis, four quality modes, alpha preservation, and an optional standard
ComfyUI `UPSCALE_MODEL`. See [Upscaling](UPSCALING.md).

## Long Video Process (NV Flow)

Streams a `VIDEO` through RIFE, upscaling, or RIFE followed by upscaling without
holding the complete clip in memory. It preserves audio and returns a file-backed
`VIDEO` for ComfyUI's Save Video node. See
[Short and long video processing](VIDEO_PROCESSING.md).
