from fractions import Fraction
from pathlib import Path

import torch
import folder_paths
import comfy.model_management as mm
import comfy.model_patcher
import comfy.utils
from comfy_api.latest import InputImpl, Types, io
from spandrel import ImageModelDescriptor, ModelLoader

from .nvflow.interpolation import interpolate_frames, interpolate_timeline
from .nvflow.model import RIFEModel
from .nvflow.streaming import process_long_video
from .nvflow.upscale import cuda_upscale


ROOT = Path(__file__).resolve().parent
CATEGORY = "image/NV Flow"
RIFE = io.Custom("NVFLOW_RIFE")


class _VideoFromComponents(InputImpl.VideoFromComponents):
    def __init__(self, components, bit_depth=8):
        super().__init__(components, bit_depth=bit_depth)
        self.components = components

    def get_components(self):
        return self.components


class NVFlowLoadVideoPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVFlowLoadVideoPath",
            display_name="Load Video From Path (NV Flow)",
            category=CATEGORY,
            description="Load a server-local video directly from an absolute path without uploading or copying it.",
            inputs=[io.String.Input("video_path", default="C:\\path\\to\\video.mp4")],
            outputs=[io.Video.Output(display_name="video")],
        )

    @staticmethod
    def _resolve(video_path):
        raw = str(video_path).strip().strip('"')
        if not raw:
            raise ValueError("Video path cannot be empty.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise ValueError("Video path must be absolute.")
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Video path is not a file: {path}")
        return path

    @classmethod
    def validate_inputs(cls, video_path):
        try:
            cls._resolve(video_path)
        except (OSError, ValueError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, video_path):
        try:
            path = cls._resolve(video_path)
            stat = path.stat()
            return str(path), stat.st_size, stat.st_mtime_ns
        except (OSError, ValueError):
            return float("nan")

    @classmethod
    def execute(cls, video_path):
        return io.NodeOutput(InputImpl.VideoFromFile(str(cls._resolve(video_path))))


class NVFlowRIFELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVFlowRIFELoader",
            display_name="Load RIFE (NV Flow)",
            category=CATEGORY,
            description="Load the bundled RIFE 4.9 model for NVIDIA GPU interpolation.",
            inputs=[io.Combo.Input("precision", options=["fp16", "fp32"], default="fp16")],
            outputs=[RIFE.Output(display_name="rife_model")],
        )

    @classmethod
    def execute(cls, precision):
        weights = ROOT / "models" / "rife49.pth"
        if not weights.is_file():
            raise FileNotFoundError(f"Bundled RIFE weights are missing: {weights}")
        dtype = torch.float16 if precision == "fp16" else torch.float32
        return io.NodeOutput(RIFEModel(weights, dtype))


class NVFlowUpscaleModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVFlowUpscaleModelLoader",
            display_name="Load Upscale Model (NV Flow)",
            category=CATEGORY,
            description="Load a ComfyUI/Spandrel upscale model, including wrapped training and torch.compile checkpoints.",
            inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("upscale_models"))],
            outputs=[io.UpscaleModel.Output(display_name="upscale_model")],
        )

    @classmethod
    def execute(cls, model_name):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        state = comfy.utils.load_torch_file(model_path, safe_load=True)
        if isinstance(state, dict) and isinstance(state.get("model_state_dict"), dict):
            state = state["model_state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Upscale checkpoint does not contain a state dictionary: {model_name}")
        if state and all(key.startswith("_orig_mod.") for key in state):
            state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in state:
            state = {key.removeprefix("module."): value for key, value in state.items()}

        model = ModelLoader().load_from_state_dict(state).eval()
        if not isinstance(model, ImageModelDescriptor):
            raise ValueError("Upscale model must be a single-image model.")
        model.patcher = comfy.model_patcher.CoreModelPatcher(
            model.model,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )
        return io.NodeOutput(model)


class NVFlowRIFEInterpolate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVFlowRIFEInterpolate",
            display_name="RIFE Interpolate (NV Flow)",
            category=CATEGORY,
            description="Interpolate frames with batched CUDA RIFE and scene-cut protection.",
            inputs=[
                io.Image.Input("frames"),
                RIFE.Input("rife_model"),
                io.Int.Input("multiplier", default=2, min=1, max=16),
                io.Boolean.Input("ensemble", default=False),
                io.Combo.Input("motion_scale", options=[1.0, 0.5, 0.25], default=1.0),
                io.Int.Input("batch_size", default=8, min=1, max=64),
                io.Float.Input("scene_cut_threshold", default=0.3, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, frames, rife_model, multiplier, ensemble, motion_scale, batch_size, scene_cut_threshold):
        return io.NodeOutput(interpolate_frames(frames, rife_model, multiplier, ensemble, motion_scale, batch_size, scene_cut_threshold))


class NVFlowRIFEVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVFlowRIFEVideo",
            display_name="RIFE Video FPS (NV Flow)",
            category=CATEGORY,
            description="Interpolate a video to a multiplied or custom frame rate detected from its source metadata.",
            inputs=[
                io.Video.Input("video"),
                RIFE.Input("rife_model"),
                io.Combo.Input("fps_mode", options=["multiplier", "target fps"], default="multiplier"),
                io.Int.Input("multiplier", default=2, min=1, max=16),
                io.Float.Input("target_fps", default=60.0, min=1.0, max=1000.0, step=0.001),
                io.Boolean.Input("ensemble", default=False),
                io.Combo.Input("motion_scale", options=[1.0, 0.5, 0.25], default=1.0),
                io.Int.Input("batch_size", default=8, min=1, max=64),
                io.Float.Input("scene_cut_threshold", default=0.3, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, video, rife_model, fps_mode, multiplier, target_fps, ensemble, motion_scale, batch_size, scene_cut_threshold):
        source_fps = video.get_frame_rate()
        output_fps = source_fps * multiplier if fps_mode == "multiplier" else Fraction(round(target_fps * 1000), 1000)
        if output_fps < source_fps:
            raise ValueError(f"Target FPS must be at least the input rate ({float(source_fps):.3f} FPS).")

        components = video.get_components()
        rate = float(output_fps / source_fps)
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
        alpha = interpolate_timeline(components.alpha, rate) if components.alpha is not None else None
        return io.NodeOutput(_VideoFromComponents(
            Types.VideoComponents(images=images, alpha=alpha, audio=components.audio, frame_rate=output_fps, metadata=components.metadata),
            bit_depth=video.get_bit_depth(),
        ))


class NVFlowCUDAUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVFlowCUDAUpscale",
            display_name="CUDA Detail Upscale (NV Flow)",
            category=CATEGORY,
            description="Quality-aware CUDA upscale with source analysis and optional ComfyUI upscale models.",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("resize_mode", options=["scale", "exact"], default="scale"),
                io.Float.Input("scale", default=2.0, min=1.0, max=8.0, step=0.05),
                io.Int.Input("width", default=1920, min=8, max=16384, step=8),
                io.Int.Input("height", default=1080, min=8, max=16384, step=8),
                io.Float.Input("detail", default=0.2, min=0.0, max=1.0, step=0.05),
                io.Combo.Input("upscale_quality", options=["auto", "faithful", "restore", "enhance"], default="auto"),
                io.Int.Input("batch_size", default=4, min=1, max=64),
                io.UpscaleModel.Input("upscale_model", optional=True),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, resize_mode, scale, width, height, detail, upscale_quality, batch_size, upscale_model=None):
        if resize_mode == "scale":
            height = round(images.shape[1] * scale)
            width = round(images.shape[2] * scale)
        width = max(8, round(width / 8) * 8)
        height = max(8, round(height / 8) * 8)
        return io.NodeOutput(cuda_upscale(images, width, height, detail, batch_size, upscale_quality, upscale_model))


class NVFlowLongVideoProcess(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NVFlowLongVideoProcess",
            display_name="Long Video Process (NV Flow)",
            category=CATEGORY,
            description="Stream long videos through RIFE and quality-aware upscaling with bounded memory and resumable chunks.",
            inputs=[
                io.Video.Input("video"),
                io.Combo.Input("operation", options=["rife", "upscale", "rife + upscale"], default="rife"),
                io.Combo.Input("fps_mode", options=["multiplier", "target fps"], default="multiplier"),
                io.Int.Input("multiplier", default=2, min=1, max=16),
                io.Float.Input("target_fps", default=60.0, min=1.0, max=1000.0, step=0.001),
                io.Boolean.Input("ensemble", default=False),
                io.Combo.Input("motion_scale", options=[1.0, 0.5, 0.25], default=1.0),
                io.Float.Input("scene_cut_threshold", default=0.3, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("resize_mode", options=["scale", "exact"], default="scale"),
                io.Float.Input("upscale_scale", default=2.0, min=1.0, max=8.0, step=0.05),
                io.Int.Input("width", default=3840, min=8, max=16384, step=8),
                io.Int.Input("height", default=2160, min=8, max=16384, step=8),
                io.Float.Input("detail", default=0.2, min=0.0, max=1.0, step=0.05),
                io.Combo.Input("upscale_quality", options=["auto", "faithful", "restore", "enhance"], default="auto"),
                io.Int.Input("batch_size", default=1, min=1, max=16),
                io.Combo.Input("encoder", options=["h264_nvenc", "hevc_nvenc", "libx264"], default="h264_nvenc"),
                io.Int.Input("quality", default=75, min=0, max=100),
                io.Combo.Input("speed", options=["quality", "balanced", "fast"], default="balanced"),
                io.Int.Input("chunk_seconds", default=30, min=5, max=300),
                RIFE.Input("rife_model", optional=True),
                io.UpscaleModel.Input("upscale_model", optional=True),
            ],
            outputs=[io.Video.Output(display_name="video")],
        )

    @classmethod
    def execute(cls, video, operation, fps_mode, multiplier, target_fps, ensemble, motion_scale, scene_cut_threshold, resize_mode, upscale_scale, width, height, detail, upscale_quality, batch_size, encoder, quality, speed, chunk_seconds, rife_model=None, upscale_model=None):
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
        return io.NodeOutput(InputImpl.VideoFromFile(str(output)))


NODE_CLASSES = [
    NVFlowLoadVideoPath,
    NVFlowRIFELoader,
    NVFlowUpscaleModelLoader,
    NVFlowRIFEInterpolate,
    NVFlowRIFEVideo,
    NVFlowCUDAUpscale,
    NVFlowLongVideoProcess,
]
