import unittest
from unittest import mock

import torch

import _plugin
from nv_flow_plugin.nvflow import upscale


class UpscaleTests(unittest.TestCase):
    def test_initial_transfer_oom_is_not_masked(self):
        images = torch.zeros((2, 8, 8, 3))
        original_to = torch.Tensor.to

        def fail_transfer(self, *args, **kwargs):
            if kwargs.get("dtype") == torch.float32 and kwargs.get("device") == torch.device("cpu"):
                raise torch.OutOfMemoryError("test")
            return original_to(self, *args, **kwargs)

        analysis = {"profile": "clean", "noise": 0.0, "sharpness": 1.0, "block_ratio": 1.0}
        with mock.patch.object(upscale, "analyze_source", return_value=analysis), mock.patch.object(upscale.mm, "get_torch_device", return_value=torch.device("cpu")), mock.patch.object(upscale.mm, "soft_empty_cache"), mock.patch.object(torch.Tensor, "to", fail_transfer):
            with self.assertRaises(torch.OutOfMemoryError):
                upscale.cuda_upscale(images, 16, 16, 0.2, 2)
