# ComfyUI NV Flow

CUDA-accelerated RIFE frame interpolation and quality-aware upscaling for
ComfyUI, with separate paths for short, composable workflows and bounded-memory
long-video processing.

## Highlights

- Bundled RIFE 4.9 weights with 2x-16x interpolation or a custom target FPS.
- Automatic input-FPS detection and timestamp-aware frame scheduling.
- Quality-aware upscaling with `auto`, `faithful`, `restore`, and `enhance` modes.
- Optional use of any ComfyUI-supported `UPSCALE_MODEL`.
- Streaming processing for videos measured in minutes, with audio preservation,
  resumable chunks, NVENC output, and verbose time-remaining estimates.
- Direct loading from an absolute local path for large videos that should not be
  uploaded or copied into ComfyUI's input directory.
- No install scripts, model downloads, TensorRT builds, or extra Python packages.

## Install

Place this folder in `ComfyUI/custom_nodes` and restart ComfyUI. An NVIDIA GPU
and a CUDA-enabled ComfyUI PyTorch build are required.

## Choose a processing path

| Path | Best for | Memory behavior |
| --- | --- | --- |
| Short-video nodes | Images, short clips, and workflows that need IMAGE-node composability | Complete frame tensors remain in RAM |
| Long Video Process | Clips measured in minutes, high resolution/FPS, and constrained RAM or VRAM | Decodes, processes, and encodes bounded batches |

Start with an importable graph from [`example_workflows`](example_workflows),
then consult the focused guides below.

## Documentation

- [Documentation index](docs/README.md)
- [Nodes and controls](docs/NODES.md)
- [Short and long video processing](docs/VIDEO_PROCESSING.md)
- [Upscaling and source-quality tuning](docs/UPSCALING.md)
- [Example workflows](docs/WORKFLOWS.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## License

NV Flow is licensed under the [Apache License 2.0](LICENSE). Bundled
RIFE-derived code and weights retain their upstream MIT notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
