from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import NODE_CLASSES


class NVFlowExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODE_CLASSES


async def comfy_entrypoint() -> NVFlowExtension:
    return NVFlowExtension()


__all__ = ["comfy_entrypoint"]
