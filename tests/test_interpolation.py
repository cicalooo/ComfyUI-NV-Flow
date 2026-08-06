from fractions import Fraction
import unittest
from unittest import mock

import torch

import _plugin
from nv_flow_plugin import nodes
from nv_flow_plugin.nvflow.interpolation import interpolate_frames, interpolate_timeline


class FakeRIFE:
    dtype = torch.float32

    def load(self, device):
        def model(first, second, timestep, **kwargs):
            return torch.lerp(first, second, timestep)

        return model

    def offload(self):
        pass


class InterpolationTests(unittest.TestCase):
    def test_interpolate_timeline_integer_rate(self):
        values = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1)
        result = interpolate_timeline(values, 2.0)
        self.assertTrue(torch.equal(result.flatten(), torch.tensor([0.0, 0.5, 1.0])))

    def test_interpolate_timeline_fractional_rate(self):
        values = torch.tensor([0.0, 1.0, 2.0]).reshape(3, 1, 1, 1)
        result = interpolate_timeline(values, 1.5)
        self.assertTrue(torch.allclose(result.flatten(), torch.tensor([0.0, 2 / 3, 4 / 3, 2.0])))

    def test_interpolate_frames_preserves_embedded_alpha(self):
        frames = torch.zeros((2, 4, 4, 4))
        frames[1, ..., :3] = 1.0
        frames[1, ..., 3] = 1.0
        with mock.patch("nv_flow_plugin.nvflow.interpolation.mm.get_torch_device", return_value=torch.device("cpu")), mock.patch("nv_flow_plugin.nvflow.interpolation.mm.soft_empty_cache"):
            result = interpolate_frames(frames, FakeRIFE(), 2, False, 1.0, 1, 0.0)
        self.assertEqual(result.shape, (3, 4, 4, 4))
        self.assertTrue(torch.equal(result[:, 0, 0, 3], torch.tensor([0.0, 0.5, 1.0])))

    def test_video_node_preserves_components(self):
        images = torch.zeros((2, 2, 2, 3))
        alpha = torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1).expand(2, 2, 2, 1)
        audio = {"waveform": torch.zeros((1, 2, 8)), "sample_rate": 8}
        metadata = {"title": "alpha test"}
        components = nodes.Types.VideoComponents(images=images, alpha=alpha, audio=audio, frame_rate=Fraction(24), metadata=metadata)

        class Video:
            def get_frame_rate(self):
                return Fraction(24)

            def get_components(self):
                return components

            def get_bit_depth(self):
                return 8

        with mock.patch.object(nodes, "interpolate_frames", side_effect=lambda frames, *args, **kwargs: interpolate_timeline(frames, 2.0)):
            output = nodes.NVFlowRIFEVideo.execute(Video(), object(), "multiplier", 2, 60.0, False, 1.0, 1, 0.3).result[0]
        result = output.get_components()
        self.assertEqual(result.images.shape[0], 3)
        self.assertTrue(torch.equal(result.alpha[:, 0, 0, 0], torch.tensor([0.0, 0.5, 1.0])))
        self.assertIs(result.audio, audio)
        self.assertIs(result.metadata, metadata)
        self.assertEqual(result.frame_rate, 48)
