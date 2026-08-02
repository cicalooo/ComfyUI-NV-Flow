from dataclasses import dataclass
from pathlib import Path

import torch

from ..rife_arch import IFNet


@dataclass
class RIFEModel:
    weights: Path
    dtype: torch.dtype
    model: IFNet | None = None

    def load(self, device: torch.device) -> IFNet:
        if self.model is None:
            state = torch.load(self.weights, map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            model = IFNet(arch_ver="4.7")
            model.load_state_dict(state)
            self.model = model.eval()
        return self.model.to(device=device, dtype=self.dtype)

    def offload(self):
        if self.model is not None:
            self.model.to(device="cpu")
