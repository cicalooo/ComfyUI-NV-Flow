# Short and long video processing

## Short-video path

Use the IMAGE-based RIFE and upscale nodes for images and short clips. Their
outputs can pass through normal ComfyUI IMAGE nodes, which makes this path the
most composable. The tradeoff is that the complete input and output tensors must
fit in system RAM, while active batches and models must fit in VRAM.

## Long-video path

Use **Long Video Process (NV Flow)** for clips measured in minutes, high output
resolutions or frame rates, or whenever full-frame tensors would consume too
much RAM. It decodes, processes, and encodes bounded batches. Combined mode runs
RIFE before upscaling so interpolation occurs at the smaller source resolution.

Start with `batch_size: 1` for demanding model upscales. Larger batches may
increase throughput when VRAM permits.

## Frame rate and timing

The long-video node can multiply the detected source FPS or produce a custom
target FPS. It schedules output from source timestamps, including variable-rate
input, and retains the first and last frames.

## Encoding and audio

H.264 NVENC is the default encoder. HEVC NVENC and software H.264 are also
available, with quality and speed controls on the node. Audio is preserved and
assembled into the completed output. MP4 does not preserve alpha; use the IMAGE
path when alpha is required.

The result is a temporary file-backed `VIDEO`. Connect it to ComfyUI's standard
Save Video node to choose a permanent filename.

## Chunks, resume, and disk space

Long jobs write temporary encoded chunks. If a run is cancelled or fails,
completed chunks are reused when the identical job is queued again in the same
ComfyUI session. Resume data does not survive a ComfyUI restart because ComfyUI
clears its temporary directory.

Keep at least 2 GiB free on the ComfyUI temporary drive in addition to the space
needed for the encoded result.

## Progress and time remaining

The ComfyUI console reports verbose progress for processing, resume scanning,
video assembly, and audio assembly. Each phase shows completed work, rate,
elapsed time, and estimated time remaining. Estimates settle after the first few
batches and continuously adjust to measured speed.
