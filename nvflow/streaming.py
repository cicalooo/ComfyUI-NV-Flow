import hashlib
import json
import logging
import math
import os
import shutil
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar

from .interpolation import _scene_cuts
from .upscale import analyze_source, cuda_upscale


PIPELINE_VERSION = 1
MIN_FREE_SPACE = 2 * 1024 ** 3
log = logging.getLogger(__name__)


def _duration(seconds):
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ETAReporter:
    def __init__(self, label, total, interval=5.0):
        self.label = label
        self.total = max(1, total)
        self.current = 0
        self.started = time.perf_counter()
        self.last_report = self.started
        self.last_reported_current = -1
        self.interval = interval

    def update(self, amount=1, force=False):
        self.current += amount
        now = time.perf_counter()
        elapsed = now - self.started
        if force and self.current == self.last_reported_current:
            return
        if not force and self.current < self.total and now - self.last_report < self.interval:
            return
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.current) / rate if rate > 0 else 0
        percent = min(100.0, self.current * 100 / self.total)
        log.info("NV Flow %s: %.0f/%.0f (%.1f%%), %.2f/sec, elapsed %s, remaining %s", self.label, self.current, self.total, percent, rate, _duration(elapsed), _duration(remaining))
        self.last_report = now
        self.last_reported_current = self.current


def _source_key(video, source):
    if isinstance(source, str):
        path = Path(source).resolve()
        stat = path.stat()
        return [str(path), stat.st_size, stat.st_mtime_ns]
    source.seek(0)
    digest = hashlib.sha256(source.getbuffer()).hexdigest()
    source.seek(0)
    return ["buffer", digest]


def _job_key(video, source, settings, rife, upscale_model):
    model_key = None
    if rife is not None:
        stat = rife.weights.stat()
        model_key = [str(rife.weights.resolve()), stat.st_size, stat.st_mtime_ns, str(rife.dtype)]
    upscale_key = None
    if upscale_model is not None:
        upscale_key = [upscale_model.__class__.__module__, upscale_model.__class__.__name__, upscale_model.scale, id(upscale_model)]
    payload = {
        "version": PIPELINE_VERSION,
        "source": _source_key(video, source),
        "trim": video.get_active_trim_window(),
        "settings": settings,
        "model": model_key,
        "upscale_model": upscale_key,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _encoder_options(encoder, quality, speed):
    value = str(round(35 - quality * 21 / 100))
    if encoder.endswith("_nvenc"):
        preset = {"quality": "p7", "balanced": "p4", "fast": "p1"}[speed]
        return {"preset": preset, "cq": value, "rc": "vbr"}
    preset = {"quality": "slow", "balanced": "medium", "fast": "veryfast"}[speed]
    return {"preset": preset, "crf": value}


def _check_encoder(encoder):
    try:
        av.codec.Codec(encoder, "w")
    except av.error.FFmpegError as error:
        raise ValueError(f"Video encoder '{encoder}' is unavailable. Select another encoder.") from error


def _video_info(source, start_time):
    if not isinstance(source, str):
        source.seek(0)
    with av.open(source, mode="r") as container:
        stream = container.streams.video[0]
        if start_time:
            container.seek(int(start_time / stream.time_base), stream=stream)
        rate = Fraction(stream.average_rate) if stream.average_rate else Fraction(1)
        bit_depth = max((component.bits for component in stream.format.components), default=8)
        metadata = dict(container.metadata)
        color = {
            name: getattr(stream.codec_context, name, None)
            for name in ("color_range", "colorspace", "color_primaries", "color_trc")
        }
        has_alpha = any(component.is_alpha for component in stream.format.components)
        sampled = []
        first_frame = None
        for frame in container.decode(stream):
            frame_time = float(frame.pts * stream.time_base) if frame.pts is not None else start_time
            if frame_time < start_time:
                continue
            first_frame = first_frame or frame
            image = _frame_tensor(frame)
            if max(image.shape[:2]) > 256:
                scale = 256 / max(image.shape[:2])
                image = torch.nn.functional.interpolate(image.movedim(-1, 0).unsqueeze(0), scale_factor=scale, mode="area").squeeze(0).movedim(0, -1)
            sampled.append(image)
            if len(sampled) == 8:
                break
        rotation = first_frame.rotation if first_frame is not None else 0
        if rotation % 180:
            width, height = stream.height, stream.width
        else:
            width, height = stream.width, stream.height
        analysis = analyze_source(torch.stack(sampled)) if sampled else None
        return rate, bit_depth, metadata, color, has_alpha, width, height, analysis


def _valid_segment(path, expected_frames, rate, width, height):
    try:
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            if stream.width != width or stream.height != height or Fraction(stream.average_rate) != rate:
                return False
            return sum(1 for _ in container.decode(stream)) == expected_frames
    except (av.error.FFmpegError, IndexError, TypeError, ValueError):
        return False


def _frame_tensor(frame):
    image = frame.to_ndarray(format="rgb24")
    if frame.rotation:
        image = np.rot90(image, k=int(round(frame.rotation / 90)), axes=(0, 1)).copy()
    return torch.from_numpy(image).float().div_(255.0)


class _ChunkWriter:
    def __init__(self, directory, rate, width, height, encoder, quality, speed, bit_depth, color, manifest, completed):
        self.directory = directory
        self.rate = rate
        self.width = width
        self.height = height
        self.encoder = encoder
        self.options = _encoder_options(encoder, quality, speed)
        self.bit_depth = bit_depth
        self.color = color
        self.manifest = manifest
        self.completed = completed
        self.segment = None
        self.container = None
        self.stream = None
        self.partial = None
        self.frame_count = 0

    def is_complete(self, segment):
        return segment in self.completed

    def _open(self, segment):
        if shutil.disk_usage(self.directory).free < MIN_FREE_SPACE:
            raise OSError("Less than 2 GiB of free space remains in the ComfyUI temporary directory.")
        self.segment = segment
        self.partial = self.directory / f"segment_{segment:06d}.partial.mp4"
        self.container = av.open(str(self.partial), mode="w")
        self.stream = self.container.add_stream(self.encoder, rate=self.rate, options=self.options)
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "p010le" if self.bit_depth > 8 else "yuv420p"
        for name, value in self.color.items():
            if value is not None:
                setattr(self.stream.codec_context, name, value)
        self.frame_count = 0

    def write(self, segment, image):
        if self.segment != segment:
            self.close()
            self._open(segment)
        if self.bit_depth > 8:
            array = image.mul(65535).clamp_(0, 65535).to(dtype=torch.uint16).cpu().numpy()
            frame = av.VideoFrame.from_ndarray(array, format="rgb48le")
        else:
            array = image.mul(255).clamp_(0, 255).to(dtype=torch.uint8).cpu().numpy()
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame.pts = self.frame_count
        try:
            packets = self.stream.encode(frame)
        except av.error.FFmpegError as error:
            raise ValueError(f"Encoder '{self.encoder}' could not encode {self.width}x{self.height} video. Select another encoder or output size.") from error
        for packet in packets:
            self.container.mux(packet)
        self.frame_count += 1

    def close(self):
        if self.container is None:
            return
        for packet in self.stream.encode(None):
            self.container.mux(packet)
        self.container.close()
        final = self.directory / f"segment_{self.segment:06d}.mp4"
        os.replace(self.partial, final)
        self.completed[self.segment] = self.frame_count
        self.manifest["completed"] = {str(key): value for key, value in self.completed.items()}
        temporary = self.directory / "manifest.partial.json"
        temporary.write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")
        os.replace(temporary, self.directory / "manifest.json")
        self.container = None
        self.stream = None
        self.partial = None

    def abort(self):
        if self.container is not None:
            self.container.close()
        if self.partial is not None:
            self.partial.unlink(missing_ok=True)
        self.container = None


def _rife_outputs(entries, rife, model, device, ensemble, scale, scene_cut):
    first = torch.stack([entry[1] for entry in entries]).movedim(-1, 1).to(device=device, dtype=rife.dtype)
    second = torch.stack([entry[2] for entry in entries]).movedim(-1, 1).to(device=device, dtype=rife.dtype)
    timesteps = torch.tensor([entry[3] for entry in entries], device=device, dtype=rife.dtype).reshape(-1, 1, 1, 1)
    try:
        result = model(first, second, timestep=timesteps, scale_list=[8 / scale, 4 / scale, 2 / scale, 1 / scale], training=False, fastmode=True, ensemble=ensemble)
    except torch.OutOfMemoryError:
        del first, second, timesteps
        mm.soft_empty_cache()
        if len(entries) == 1:
            raise
        middle = len(entries) // 2
        return _rife_outputs(entries[:middle], rife, model, device, ensemble, scale, scene_cut) + _rife_outputs(entries[middle:], rife, model, device, ensemble, scale, scene_cut)
    cuts = _scene_cuts(first, second, scene_cut)
    result = result.clamp_(0, 1).float().cpu().movedim(1, -1)
    if cuts.any():
        nearest = torch.stack([entry[1] if entry[3] < 0.5 else entry[2] for entry in entries])
        result[cuts] = nearest[cuts]
    return list(result)


def _assemble(directory, segments, output, source, start_time, duration, metadata):
    output_partial = output.with_suffix(".partial.mp4")
    output_partial.unlink(missing_ok=True)
    if not isinstance(source, str):
        source.seek(0)
    with av.open(source, mode="r") as source_container, av.open(str(output_partial), mode="w") as destination:
        for key, value in metadata.items():
            destination.metadata[key] = value

        input_audio = source_container.streams.audio[0] if source_container.streams.audio else None
        output_audio = None
        if input_audio is not None:
            sample_rate = input_audio.codec_context.sample_rate or 48000
            layout = input_audio.codec_context.layout.name if input_audio.codec_context.layout else "stereo"
            output_audio = destination.add_stream("aac", rate=sample_rate, layout=layout)

        video_eta = _ETAReporter("video assembly", segments)
        output_video = None
        video_offset = 0
        for segment in range(segments):
            with av.open(str(directory / f"segment_{segment:06d}.mp4"), mode="r") as chunk:
                input_video = chunk.streams.video[0]
                if output_video is None:
                    output_video = destination.add_stream_from_template(input_video, opaque=True)
                last_end = video_offset
                for packet in chunk.demux(input_video):
                    if packet.dts is None:
                        continue
                    packet.pts += video_offset
                    packet.dts += video_offset
                    packet.stream = output_video
                    destination.mux(packet)
                    last_end = max(last_end, packet.pts + (packet.duration or 0))
                video_offset = last_end
            video_eta.update(force=True)

        if input_audio is not None:
            audio_eta = _ETAReporter("audio assembly", duration)
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
            cursor = 0
            end_time = start_time + duration if duration else math.inf
            audio_done = False
            for frame in source_container.decode(input_audio):
                for resampled in resampler.resample(frame):
                    frame_start = float(resampled.time or 0)
                    frame_end = frame_start + resampled.samples / sample_rate
                    if frame_end <= start_time:
                        continue
                    if frame_start >= end_time:
                        audio_done = True
                        break
                    array = resampled.to_ndarray()
                    begin = max(0, round((start_time - frame_start) * sample_rate))
                    end = resampled.samples if math.isinf(end_time) else min(resampled.samples, round((end_time - frame_start) * sample_rate))
                    if end <= begin:
                        continue
                    audio_frame = av.AudioFrame.from_ndarray(array[..., begin:end], format="fltp", layout=layout)
                    audio_frame.sample_rate = sample_rate
                    audio_frame.pts = cursor
                    audio_frame.time_base = Fraction(1, sample_rate)
                    cursor += audio_frame.samples
                    audio_eta.update(audio_frame.samples / sample_rate)
                    for packet in output_audio.encode(audio_frame):
                        destination.mux(packet)
                if audio_done:
                    break
            for packet in output_audio.encode(None):
                destination.mux(packet)
            audio_eta.update(0, force=True)
    os.replace(output_partial, output)


def process_long_video(video, rife, upscale_model, settings, temp_root):
    job_started = time.perf_counter()
    source = video.get_stream_source()
    start_time, duration = video.get_active_trim_window()
    source_rate, source_bit_depth, metadata, color, has_alpha, source_width, source_height, source_analysis = _video_info(source, start_time)
    if has_alpha:
        raise ValueError("Long-video MP4 processing does not support alpha video. Use the IMAGE nodes when alpha must be preserved.")
    operation = settings["operation"]
    use_rife = operation != "upscale"
    use_upscale = operation != "rife"
    if use_rife and rife is None:
        raise ValueError("Connect a RIFE model for the selected long-video operation.")

    output_rate = source_rate
    if use_rife:
        output_rate = source_rate * settings["multiplier"] if settings["fps_mode"] == "multiplier" else Fraction(round(settings["target_fps"] * 1000), 1000)
        if output_rate < source_rate:
            raise ValueError(f"Target FPS must be at least the input rate ({float(source_rate):.3f} FPS).")

    if use_upscale:
        if settings["resize_mode"] == "scale":
            width = max(8, round(source_width * settings["upscale_scale"] / 8) * 8)
            height = max(8, round(source_height * settings["upscale_scale"] / 8) * 8)
        else:
            width = round(settings["width"] / 8) * 8
            height = round(settings["height"] / 8) * 8
    else:
        width, height = source_width, source_height

    _check_encoder(settings["encoder"])
    key = _job_key(video, source, settings, rife, upscale_model if use_upscale else None)
    directory = Path(temp_root) / "nvflow_long" / key
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "result.mp4"
    if output.is_file():
        log.info("NV Flow long video: reusing completed temporary result %s", output)
        return output

    manifest_path = directory / "manifest.json"
    manifest = {"version": PIPELINE_VERSION, "job": key, "completed": {}}
    if manifest_path.is_file():
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if loaded.get("version") == PIPELINE_VERSION and loaded.get("job") == key:
            manifest = loaded
    completed = {}
    for segment, count in manifest["completed"].items():
        path = directory / f"segment_{int(segment):06d}.mp4"
        if path.is_file() and _valid_segment(path, count, output_rate, width, height):
            completed[int(segment)] = count

    frames_per_segment = max(1, round(settings["chunk_seconds"] * float(output_rate)))
    writer = _ChunkWriter(directory, output_rate, width, height, settings["encoder"], settings["quality"], settings["speed"], source_bit_depth, color, manifest, completed)
    end_time = start_time + duration if duration else math.inf
    total_frames = video.get_frame_count()
    estimated = max(1, round(total_frames * float(output_rate / source_rate)))
    progress = ProgressBar(estimated)
    completed_frames = sum(completed.values())
    processing_eta = _ETAReporter(f"{operation} processing", max(1, estimated - completed_frames))
    resume_eta = _ETAReporter("resume scan", completed_frames) if completed_frames else None
    log.info(
        "NV Flow long video started: %s, %dx%d at %.3f FPS, approximately %d output frames, %d resumable frames already complete",
        operation,
        width,
        height,
        float(output_rate),
        estimated,
        completed_frames,
    )
    device = mm.get_torch_device()
    model = rife.load(device) if use_rife else None
    queue = []
    output_index = 0
    upscale_analysis = source_analysis
    if use_upscale and upscale_analysis is not None:
        log.info("NV Flow source analysis: %s", upscale_analysis)

    def flush():
        nonlocal queue, upscale_analysis
        if not queue:
            return
        interpolation_entries = [entry for entry in queue if entry[0] == "rife"]
        generated = iter(_rife_outputs(interpolation_entries, rife, model, device, settings["ensemble"], settings["motion_scale"], settings["scene_cut_threshold"])) if interpolation_entries else iter(())
        images = [next(generated) if entry[0] == "rife" else entry[1] for entry in queue]
        batch = torch.stack(images)
        if use_upscale:
            if upscale_analysis is None:
                upscale_analysis = analyze_source(torch.stack([entry[1] for entry in queue]))
                log.info("NV Flow source analysis: %s", upscale_analysis)
            batch = cuda_upscale(batch, width, height, settings["detail"], settings["batch_size"], settings["upscale_quality"], upscale_model, upscale_analysis)
        for entry, image in zip(queue, batch):
            segment = entry[-1] // frames_per_segment
            if not writer.is_complete(segment):
                writer.write(segment, image)
            progress.update(1)
            processing_eta.update()
        queue = []

    try:
        if not isinstance(source, str):
            source.seek(0)
        with av.open(source, mode="r") as container:
            stream = container.streams.video[0]
            seek_pts = int(start_time / stream.time_base)
            if seek_pts:
                container.seek(seek_pts, stream=stream)
            previous = None
            previous_time = None
            last_emitted_time = None
            next_time = start_time
            for frame in container.decode(stream):
                mm.throw_exception_if_processing_interrupted()
                frame_time = float(frame.pts * stream.time_base) if frame.pts is not None else (previous_time + 1 / float(source_rate) if previous_time is not None else start_time)
                if frame_time < start_time:
                    previous = _frame_tensor(frame)
                    previous_time = frame_time
                    continue
                if frame_time > end_time:
                    break
                current = _frame_tensor(frame)
                if not use_rife:
                    segment = output_index // frames_per_segment
                    if not writer.is_complete(segment):
                        queue.append(("source", current, output_index))
                    else:
                        progress.update(1)
                        resume_eta.update()
                    output_index += 1
                    if len(queue) >= settings["batch_size"]:
                        flush()
                    continue
                if previous is None:
                    previous = current
                    previous_time = frame_time
                while next_time <= frame_time + 1e-9:
                    segment = output_index // frames_per_segment
                    if not writer.is_complete(segment):
                        if next_time <= previous_time + 1e-9 or frame_time == previous_time:
                            queue.append(("source", previous, output_index))
                        elif next_time >= frame_time - 1e-9:
                            queue.append(("source", current, output_index))
                        else:
                            timestep = (next_time - previous_time) / (frame_time - previous_time)
                            queue.append(("rife", previous, current, timestep, output_index))
                    else:
                        progress.update(1)
                        resume_eta.update()
                    last_emitted_time = next_time
                    output_index += 1
                    next_time = start_time + output_index / float(output_rate)
                    if sum(entry[0] == "rife" for entry in queue) >= settings["batch_size"]:
                        flush()
                previous = current
                previous_time = frame_time

            if previous is not None and (last_emitted_time is None or previous_time - last_emitted_time > 1e-7):
                queue.append(("source", previous, output_index))
                output_index += 1
            flush()
        writer.close()
        segments = math.ceil(output_index / frames_per_segment)
        processing_eta.update(0, force=True)
        log.info("NV Flow frame processing complete; assembling %d video chunks and audio", segments)
        _assemble(directory, segments, output, source, start_time, duration or video.get_duration(), metadata)
        for segment in range(segments):
            (directory / f"segment_{segment:06d}.mp4").unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        log.info("NV Flow long video complete: %d output frames in %s; total elapsed %s", output_index, output, _duration(time.perf_counter() - job_started))
    except BaseException:
        writer.abort()
        output.with_suffix(".partial.mp4").unlink(missing_ok=True)
        raise
    finally:
        if rife is not None and use_rife:
            rife.offload()
        mm.soft_empty_cache()
    return output
