# Example workflows

Import the JSON files from [`example_workflows`](../example_workflows) into
ComfyUI. Select the input video and review output dimensions before queueing.

## Short-video examples

- [`nv_flow_small_upscale.json`](../example_workflows/nv_flow_small_upscale.json)
  uses tensor-based quality-aware CUDA upscale while preserving source audio and FPS.
- [`nv_flow_small_model_upscale.json`](../example_workflows/nv_flow_small_model_upscale.json)
  demonstrates model-assisted `restore` mode.
- [`nv_flow_small_rife_video.json`](../example_workflows/nv_flow_small_rife_video.json)
  uses backend FPS detection and a custom RIFE target FPS.
- [`nv_flow_small_upscale_then_rife.json`](../example_workflows/nv_flow_small_upscale_then_rife.json)
  upscales tensors before frame-batch RIFE. Set Create Video FPS to source FPS
  multiplied by the selected RIFE multiplier.

## Long-video examples

- [`nv_flow_long_upscale.json`](../example_workflows/nv_flow_long_upscale.json)
  performs bounded-memory streaming upscale.
- [`nv_flow_long_model_upscale.json`](../example_workflows/nv_flow_long_model_upscale.json)
  performs bounded-memory model-assisted `restore` processing.
- [`nv_flow_long_rife.json`](../example_workflows/nv_flow_long_rife.json)
  performs streaming RIFE with automatic source-FPS detection.
- [`nv_flow_long_rife_then_upscale.json`](../example_workflows/nv_flow_long_rife_then_upscale.json)
  performs streaming RIFE followed by upscale using a custom target FPS.

All long examples use **Load Video From Path (NV Flow)**. Replace the placeholder
with an absolute path that is accessible to the machine running ComfyUI. They
default to H.264 NVENC and pass their temporary `VIDEO` result to ComfyUI's
standard Save Video node.

## Model-assisted examples

Before queueing either model-assisted graph, choose a model already installed in
`ComfyUI/models/upscale_models` in its Load Upscale Model node. The other examples
remain self-contained and do not require external weights.
