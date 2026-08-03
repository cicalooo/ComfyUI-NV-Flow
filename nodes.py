from pathlib import Path
from fractions import Fraction

import torch
import folder_paths
from comfy_api.latest import InputImpl, Types

from .nvflow.interpolation import interpolate_frames
from .nvflow.model import RIFEModel
from .nvflow.upscale import cuda_upscale
from .nvflow.streaming import process_long_video


ROOT = Path(__file__).resolve().parent
CATEGORY = "image/NV Flow"


class NVFlowRIFELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"precision": (["fp16", "fp32"], {"default": "fp16"})}}

    RETURN_TYPES = ("NVFLOW_RIFE",)
    RETURN_NAMES = ("rife_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = "Load the bundled RIFE 4.9 model for NVIDIA GPU interpolation."

    def load(self, precision):
        weights = ROOT / "models" / "rife49.pth"
        if not weights.is_file():
            raise FileNotFoundError(f"Bundled RIFE weights are missing: {weights}")
        dtype = torch.float16 if precision == "fp16" else torch.float32
        return (RIFEModel(weights, dtype),)


class NVFlowRIFEInterpolate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "rife_model": ("NVFLOW_RIFE",),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 16}),
                "ensemble": ("BOOLEAN", {"default": False}),
                "motion_scale": ([1.0, 0.5, 0.25], {"default": 1.0}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "interpolate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Interpolate frames with batched CUDA RIFE and scene-cut protection."

    def interpolate(self, frames, rife_model, multiplier, ensemble, motion_scale, batch_size, scene_cut_threshold):
        return (interpolate_frames(frames, rife_model, multiplier, ensemble, motion_scale, batch_size, scene_cut_threshold),)


class NVFlowRIFEVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "rife_model": ("NVFLOW_RIFE",),
                "fps_mode": (["multiplier", "target fps"], {"default": "multiplier"}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 16}),
                "target_fps": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 1000.0, "step": 0.001}),
                "ensemble": ("BOOLEAN", {"default": False}),
                "motion_scale": ([1.0, 0.5, 0.25], {"default": 1.0}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "interpolate"
    CATEGORY = CATEGORY
    DESCRIPTION = "Interpolate a video to a multiplied or custom frame rate detected from its source metadata."

    def interpolate(self, video, rife_model, fps_mode, multiplier, target_fps, ensemble, motion_scale, batch_size, scene_cut_threshold):
        source_fps = video.get_frame_rate()
        output_fps = source_fps * multiplier if fps_mode == "multiplier" else Fraction(round(target_fps * 1000), 1000)
        if output_fps < source_fps:
            raise ValueError(f"Target FPS must be at least the input rate ({float(source_fps):.3f} FPS).")

        components = video.get_components()
        images = interpolate_frames(
            components.images,
            rife_model,
            multiplier,
            ensemble,
            motion_scale,
            batch_size,
            scene_cut_threshold,
            float(output_fps),
            float(source_fps),
        )
        return (InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images, audio=components.audio, frame_rate=output_fps, metadata=components.metadata),
            bit_depth=video.get_bit_depth(),
        ),)


class NVFlowCUDAUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "resize_mode": (["scale", "exact"], {"default": "scale"}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "width": ("INT", {"default": 1920, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1080, "min": 8, "max": 16384, "step": 8}),
                "detail": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "upscale_quality": (["auto", "faithful", "restore", "enhance"], {"default": "auto"}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64}),
            },
            "optional": {"upscale_model": ("UPSCALE_MODEL",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "Quality-aware CUDA upscale with source analysis and optional ComfyUI upscale models."

    def upscale(self, images, resize_mode, scale, width, height, detail, batch_size, upscale_quality="auto", upscale_model=None):
        if resize_mode == "scale":
            height = round(images.shape[1] * scale)
            width = round(images.shape[2] * scale)
        width = max(8, round(width / 8) * 8)
        height = max(8, round(height / 8) * 8)
        return (cuda_upscale(images, width, height, detail, batch_size, upscale_quality, upscale_model),)


class NVFlowLongVideoProcess:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "operation": (["rife", "upscale", "rife + upscale"], {"default": "rife"}),
                "fps_mode": (["multiplier", "target fps"], {"default": "multiplier"}),
                "multiplier": ("INT", {"default": 2, "min": 1, "max": 16}),
                "target_fps": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 1000.0, "step": 0.001}),
                "ensemble": ("BOOLEAN", {"default": False}),
                "motion_scale": ([1.0, 0.5, 0.25], {"default": 1.0}),
                "scene_cut_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "resize_mode": (["scale", "exact"], {"default": "scale"}),
                "upscale_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "width": ("INT", {"default": 3840, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 2160, "min": 8, "max": 16384, "step": 8}),
                "detail": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "upscale_quality": (["auto", "faithful", "restore", "enhance"], {"default": "auto"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "encoder": (["h264_nvenc", "hevc_nvenc", "libx264"], {"default": "h264_nvenc"}),
                "quality": ("INT", {"default": 75, "min": 0, "max": 100}),
                "speed": (["quality", "balanced", "fast"], {"default": "balanced"}),
                "chunk_seconds": ("INT", {"default": 30, "min": 5, "max": 300}),
            },
            "optional": {"rife_model": ("NVFLOW_RIFE",), "upscale_model": ("UPSCALE_MODEL",)},
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "process"
    CATEGORY = CATEGORY
    DESCRIPTION = "Stream long videos through RIFE and quality-aware upscaling with bounded memory and resumable chunks."

    def process(self, video, operation, fps_mode, multiplier, target_fps, ensemble, motion_scale, scene_cut_threshold, resize_mode, upscale_scale, width, height, detail, upscale_quality, batch_size, encoder, quality, speed, chunk_seconds, rife_model=None, upscale_model=None):
        settings = {
            "operation": operation,
            "fps_mode": fps_mode,
            "multiplier": multiplier,
            "target_fps": target_fps,
            "ensemble": ensemble,
            "motion_scale": motion_scale,
            "scene_cut_threshold": scene_cut_threshold,
            "resize_mode": resize_mode,
            "upscale_scale": upscale_scale,
            "width": width,
            "height": height,
            "detail": detail,
            "upscale_quality": upscale_quality,
            "batch_size": batch_size,
            "encoder": encoder,
            "quality": quality,
            "speed": speed,
            "chunk_seconds": chunk_seconds,
        }
        output = process_long_video(video, rife_model, upscale_model, settings, folder_paths.get_temp_directory())
        return (InputImpl.VideoFromFile(str(output)),)


NODE_CLASS_MAPPINGS = {
    "NVFlowRIFELoader": NVFlowRIFELoader,
    "NVFlowRIFEInterpolate": NVFlowRIFEInterpolate,
    "NVFlowRIFEVideo": NVFlowRIFEVideo,
    "NVFlowCUDAUpscale": NVFlowCUDAUpscale,
    "NVFlowLongVideoProcess": NVFlowLongVideoProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NVFlowRIFELoader": "Load RIFE (NV Flow)",
    "NVFlowRIFEInterpolate": "RIFE Interpolate (NV Flow)",
    "NVFlowRIFEVideo": "RIFE Video FPS (NV Flow)",
    "NVFlowCUDAUpscale": "CUDA Detail Upscale (NV Flow)",
    "NVFlowLongVideoProcess": "Long Video Process (NV Flow)",
}
