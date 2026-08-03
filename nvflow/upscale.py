import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.utils
from comfy.utils import ProgressBar


def _gaussian_blur(image):
    channels = image.shape[1]
    kernel = torch.tensor([1, 4, 6, 4, 1], device=image.device, dtype=image.dtype)
    kernel = (kernel[:, None] * kernel[None, :]).div_(256).expand(channels, 1, 5, 5)
    return F.conv2d(F.pad(image, (2, 2, 2, 2), mode="reflect"), kernel, groups=channels)


def analyze_source(images):
    sample = images[: min(images.shape[0], 8), :, :, :3].movedim(-1, 1).float()
    if max(sample.shape[-2:]) > 256:
        scale = 256 / max(sample.shape[-2:])
        sample = F.interpolate(sample, scale_factor=scale, mode="area")
    luma = sample[:, 0:1] * 0.2126 + sample[:, 1:2] * 0.7152 + sample[:, 2:3] * 0.0722
    smooth = _gaussian_blur(luma)
    residual = (luma - smooth).abs()
    gradient_x = F.pad((luma[:, :, :, 1:] - luma[:, :, :, :-1]).abs(), (0, 1))
    gradient_y = F.pad((luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs(), (0, 0, 0, 1))
    gradient = gradient_x + gradient_y
    flat_threshold = torch.quantile(gradient.flatten(), 0.3)
    noise = residual[gradient <= flat_threshold].mean().item()
    laplacian = (4 * luma - F.pad(luma, (1, 1, 1, 1), mode="reflect")[:, :, 1:-1, :-2] - F.pad(luma, (1, 1, 1, 1), mode="reflect")[:, :, 1:-1, 2:] - F.pad(luma, (1, 1, 1, 1), mode="reflect")[:, :, :-2, 1:-1] - F.pad(luma, (1, 1, 1, 1), mode="reflect")[:, :, 2:, 1:-1]).abs().mean().item()

    vertical = (luma[:, :, :, 1:] - luma[:, :, :, :-1]).abs()
    horizontal = (luma[:, :, 1:, :] - luma[:, :, :-1, :]).abs()
    boundary = []
    interior = []
    if vertical.shape[-1] > 8:
        boundary.append(vertical[:, :, :, 7::8].mean())
        interior.append(vertical[:, :, :, 3::8].mean())
    if horizontal.shape[-2] > 8:
        boundary.append(horizontal[:, :, 7::8, :].mean())
        interior.append(horizontal[:, :, 3::8, :].mean())
    block_ratio = (torch.stack(boundary).mean() / torch.stack(interior).mean().clamp_min(1e-6)).item() if boundary else 1.0

    if block_ratio > 1.15 and noise > 0.003:
        profile = "compressed"
    elif noise > 0.012:
        profile = "noisy"
    elif laplacian < 0.025:
        profile = "soft"
    else:
        profile = "clean"
    return {"profile": profile, "noise": noise, "sharpness": laplacian, "block_ratio": block_ratio}


def _cleanup(image, strength):
    if strength == 0:
        return image
    smooth = _gaussian_blur(image)
    luma = image[:, 0:1] * 0.2126 + image[:, 1:2] * 0.7152 + image[:, 2:3] * 0.0722
    smooth_luma = smooth[:, 0:1] * 0.2126 + smooth[:, 1:2] * 0.7152 + smooth[:, 2:3] * 0.0722
    flat = (1 - (luma - smooth_luma).abs() / 0.04).clamp_(0, 1)
    return torch.lerp(image, smooth, flat * strength)


def _enhance_luminance(image, amount, threshold):
    if amount == 0:
        return image
    luma = image[:, 0:1] * 0.2126 + image[:, 1:2] * 0.7152 + image[:, 2:3] * 0.0722
    detail = luma - _gaussian_blur(luma)
    if threshold > 0:
        detail = detail.sign() * (detail.abs() - threshold).clamp_min_(0)
    return image.add(detail * amount).clamp_(0, 1)


def _quality_settings(quality, analysis, detail, has_model):
    profile = analysis["profile"] if quality == "auto" else quality
    if profile == "faithful":
        return 0.0, detail * 0.75, 0.004, 0.5 if has_model else 0.0
    if profile in ("restore", "compressed"):
        return 0.2 if not has_model else 0.0, detail * 0.5, 0.008, 0.8 if has_model else 0.0
    if profile == "noisy":
        return 0.25 if not has_model else 0.0, detail * 0.25, 0.012, 0.8 if has_model else 0.0
    if profile in ("enhance", "soft"):
        return 0.0, detail * 1.5, 0.003, 1.0 if has_model else 0.0
    return 0.0, detail, 0.005, 0.8 if has_model else 0.0


def _model_upscale(images, upscale_model):
    device = upscale_model.patcher.load_device
    memory_required = (512 * 512 * 3) * images.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += images.nelement() * images.element_size()
    mm.load_models_gpu([upscale_model.patcher], memory_required=memory_required)
    source = images.movedim(-1, 1).to(device)
    tile = 512
    while True:
        try:
            steps = source.shape[0] * comfy.utils.get_tiled_scale_steps(source.shape[3], source.shape[2], tile_x=tile, tile_y=tile, overlap=32)
            result = comfy.utils.tiled_scale(source, lambda tile_image: upscale_model(tile_image.float()), tile_x=tile, tile_y=tile, overlap=32, upscale_amount=upscale_model.scale, pbar=ProgressBar(steps), output_device=mm.intermediate_device())
            return result.clamp(0, 1).movedim(1, -1).float().cpu()
        except torch.OutOfMemoryError:
            mm.soft_empty_cache()
            tile //= 2
            if tile < 128:
                raise


def cuda_upscale(images, width, height, detail, batch_size, upscale_quality="auto", upscale_model=None, analysis=None):
    device = mm.get_torch_device()
    progress = ProgressBar(images.shape[0])
    output = torch.empty((images.shape[0], height, width, images.shape[-1]), dtype=images.dtype, device="cpu")
    analysis = analysis or analyze_source(images)
    cleanup, sharpen, threshold, model_strength = _quality_settings(upscale_quality, analysis, detail, upscale_model is not None)
    start = 0
    chunk = min(batch_size, images.shape[0])
    while start < images.shape[0]:
        stop = min(start + chunk, images.shape[0])
        source = images[start:stop]
        try:
            base = source.movedim(-1, 1).to(device=device, dtype=torch.float32)
            base = F.interpolate(base, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
            rgb = _cleanup(base[:, :3], cleanup)
            if upscale_model is not None:
                model_output = _model_upscale(source[:, :, :, :3], upscale_model).movedim(-1, 1).to(device)
                if model_output.shape[-2:] != (height, width):
                    model_output = F.interpolate(model_output, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
                rgb = torch.lerp(rgb, model_output, model_strength)
            rgb = _enhance_luminance(rgb, sharpen, threshold)
            if base.shape[1] > 3:
                rgb = torch.cat((rgb, base[:, 3:4].clamp_(0, 1)), dim=1)
        except torch.OutOfMemoryError:
            del base
            mm.soft_empty_cache()
            if chunk == 1:
                raise
            chunk = max(1, chunk // 2)
            continue
        output[start:stop] = rgb.movedim(1, -1).to(device="cpu", dtype=images.dtype)
        progress.update(stop - start)
        start = stop
    mm.soft_empty_cache()
    return output
