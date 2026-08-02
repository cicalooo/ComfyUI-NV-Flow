import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar


def _scene_cuts(first: torch.Tensor, second: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0.0 or threshold >= 1.0:
        return torch.zeros(first.shape[0], dtype=torch.bool)
    sample_a = torch.nn.functional.interpolate(first, size=(32, 32), mode="area").float()
    sample_b = torch.nn.functional.interpolate(second, size=(32, 32), mode="area").float()
    return (sample_a - sample_b).abs().mean(dim=(1, 2, 3)).cpu() >= threshold


def _output_plan(frame_count, rate):
    output_count = round((frame_count - 1) * rate) + 1
    plan = []
    source = []
    for output_index in range(output_count):
        if output_index == output_count - 1:
            source.append((output_index, frame_count - 1))
            continue
        position = min(output_index / rate, frame_count - 1)
        first_index = min(int(position), frame_count - 2)
        timestep = position - first_index
        if timestep < 1e-7:
            source.append((output_index, first_index))
            continue
        if 1.0 - timestep < 1e-7:
            source.append((output_index, first_index + 1))
            continue
        plan.append((output_index, first_index, timestep))
    return output_count, source, plan


def interpolate_frames(frames, rife, multiplier, ensemble, scale, batch_size, scene_cut, target_fps=None, source_fps=None):
    if frames.shape[0] < 2:
        return frames

    rate = multiplier if target_fps is None else target_fps / source_fps
    if rate <= 1.0:
        return frames

    output_count, source, plan = _output_plan(frames.shape[0], rate)
    output = torch.empty((output_count, *frames.shape[1:3], 3), dtype=frames.dtype, device="cpu")
    output[[item[0] for item in source]] = frames[[item[1] for item in source], :, :, :3].cpu()

    device = mm.get_torch_device()
    model = rife.load(device)
    scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
    progress = ProgressBar(len(plan))

    try:
        start = 0
        chunk = min(batch_size, len(plan))
        while start < len(plan):
            stop = min(start + chunk, len(plan))
            batch_items = plan[start:stop]
            frame_indices = [item[1] for item in batch_items]
            timesteps = torch.tensor([item[2] for item in batch_items], device=device, dtype=rife.dtype).reshape(-1, 1, 1, 1)
            image0 = frames[frame_indices, :, :, :3].movedim(-1, 1).to(device=device, dtype=rife.dtype)
            image1 = frames[[index + 1 for index in frame_indices], :, :, :3].movedim(-1, 1).to(device=device, dtype=rife.dtype)
            try:
                interpolated = model(
                    image0,
                    image1,
                    timestep=timesteps,
                    scale_list=scale_list,
                    training=False,
                    fastmode=True,
                    ensemble=ensemble,
                )
            except torch.OutOfMemoryError:
                del image0, image1, timesteps
                mm.soft_empty_cache()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
                continue

            cuts = _scene_cuts(image0, image1, scene_cut)
            interpolated = interpolated.clamp_(0, 1).to(device="cpu", dtype=frames.dtype).movedim(1, -1)
            if cuts.any():
                nearest_indices = [item[1] if item[2] < 0.5 else item[1] + 1 for item in batch_items]
                interpolated[cuts] = frames[nearest_indices, :, :, :3][cuts]
            output[[item[0] for item in batch_items]] = interpolated
            progress.update(stop - start)
            start = stop
    finally:
        rife.offload()
        mm.soft_empty_cache()

    return output
