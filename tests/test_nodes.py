import asyncio
import unittest

import _plugin
import nv_flow_plugin


EXPECTED_INPUTS = {
    "NVFlowLoadVideoPath": ["video_path"],
    "NVFlowRIFELoader": ["precision"],
    "NVFlowUpscaleModelLoader": ["model_name"],
    "NVFlowRIFEInterpolate": ["frames", "rife_model", "multiplier", "ensemble", "motion_scale", "batch_size", "scene_cut_threshold"],
    "NVFlowRIFEVideo": ["video", "rife_model", "fps_mode", "multiplier", "target_fps", "ensemble", "motion_scale", "batch_size", "scene_cut_threshold"],
    "NVFlowCUDAUpscale": ["images", "resize_mode", "scale", "width", "height", "detail", "upscale_quality", "batch_size", "upscale_model"],
    "NVFlowLongVideoProcess": ["video", "operation", "fps_mode", "multiplier", "target_fps", "ensemble", "motion_scale", "scene_cut_threshold", "resize_mode", "upscale_scale", "width", "height", "detail", "upscale_quality", "batch_size", "encoder", "quality", "speed", "chunk_seconds", "rife_model", "upscale_model"],
}


class NodeSchemaTests(unittest.TestCase):
    def test_extension_preserves_node_ids_and_input_order(self):
        extension = asyncio.run(nv_flow_plugin.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        schemas = {node.define_schema().node_id: node.define_schema() for node in node_classes}
        self.assertEqual(set(schemas), set(EXPECTED_INPUTS))
        for node_id, expected in EXPECTED_INPUTS.items():
            self.assertEqual([item.id for item in schemas[node_id].inputs], expected)
