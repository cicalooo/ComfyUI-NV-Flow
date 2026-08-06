import unittest

import torch

import _plugin
from nv_flow_plugin.nvflow.streaming import _upscale_model_key


class FakeUpscaleModel:
    scale = 2

    def __init__(self, value):
        self.model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(value)


class StreamingTests(unittest.TestCase):
    def test_upscale_model_key_is_stable_and_weight_sensitive(self):
        self.assertEqual(_upscale_model_key(FakeUpscaleModel(1.0)), _upscale_model_key(FakeUpscaleModel(1.0)))
        self.assertNotEqual(_upscale_model_key(FakeUpscaleModel(1.0)), _upscale_model_key(FakeUpscaleModel(2.0)))
