# Upscaling and source-quality tuning

Both the short and long upscale paths provide `upscale_quality`.

| Mode | Intended result |
| --- | --- |
| `auto` | Samples the source, classifies it as clean, compressed, noisy, or soft, and selects stable tuning |
| `faithful` | Favors the bicubic result with conservative thresholded sharpening |
| `restore` | Suppresses defects in flat regions and applies restrained sharpening |
| `enhance` | Favors stronger detail and the full output of a connected upscale model |

For video, automatic analysis is performed once and the selected tuning remains
fixed across the clip to reduce flicker. The `detail` control is still applied,
but its strength is adjusted by the selected quality mode. Gaussian detail
estimation and thresholding reduce amplification of noise and compression residue.

## External upscale models

Connect **Load Upscale Model (NV Flow)** or ComfyUI's native **Load Upscale
Model** to the optional `upscale_model` input. The NV Flow loader also handles
training checkpoints wrapped in `model_state_dict` and weights saved with the
PyTorch Compile `_orig_mod.` prefix. Any architecture supported by Spandrel can
be used; NV Flow does not download models or introduce another model format.

Place models in `ComfyUI/models/upscale_models`, restart or refresh ComfyUI if
needed, and select the model in the loader.

Model inference uses ComfyUI's tiled execution and automatically reduces tile
size after a VRAM exhaustion error. If the model's native scale differs from the
requested output, its result is resized to the exact requested dimensions.

Approximate model-result blending is:

- `faithful`: 50%
- `restore`: 80%
- `enhance`: 100%

`auto` chooses behavior from its source classification. When no external model
is connected, NV Flow uses its built-in CUDA resize, cleanup, and sharpening path.
