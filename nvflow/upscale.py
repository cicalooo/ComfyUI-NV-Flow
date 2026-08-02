import torch
import torch.nn.functional as F

import comfy.model_management as mm
from comfy.utils import ProgressBar


def _enhance_luminance(image: torch.Tensor, amount: float) -> torch.Tensor:
    if amount == 0:
        return image
    luminance = image[:, 0:1] * 0.2126 + image[:, 1:2] * 0.7152 + image[:, 2:3] * 0.0722
    blur = F.avg_pool2d(F.pad(luminance, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
    detail = luminance - blur
    return (image + detail * amount).clamp_(0, 1)


def cuda_upscale(images, width, height, detail, batch_size):
    device = mm.get_torch_device()
    progress = ProgressBar(images.shape[0])
    output = torch.empty((images.shape[0], height, width, images.shape[-1]), dtype=images.dtype, device="cpu")
    start = 0
    chunk = min(batch_size, images.shape[0])
    while start < images.shape[0]:
        stop = min(start + chunk, images.shape[0])
        batch = images[start:stop].movedim(-1, 1).to(device=device, dtype=torch.float32)
        try:
            resized = F.interpolate(batch, size=(height, width), mode="bicubic", align_corners=False, antialias=True)
            rgb = _enhance_luminance(resized[:, :3], detail)
            if resized.shape[1] > 3:
                rgb = torch.cat((rgb, resized[:, 3:4].clamp_(0, 1)), dim=1)
        except torch.OutOfMemoryError:
            del batch
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
