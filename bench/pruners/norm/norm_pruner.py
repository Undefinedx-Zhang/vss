from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from ..base import Pruner, PruningConfig


class NormPruner(Pruner):
    def __init__(self, model: nn.Module, config: PruningConfig, norm_type: str = "l2"):
        super().__init__(model, config)
        assert norm_type in ("l1", "l2")
        self.norm_type = norm_type

    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        model = self.model
        device = example_inputs.device
        model.eval()

        scores: Dict[nn.Module, torch.Tensor] = {}
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                w = m.weight.detach().view(m.out_channels, -1)
                if self.norm_type == "l1":
                    s = torch.norm(w, p=1, dim=1)
                else:
                    s = torch.norm(w, p=2, dim=1)
                scores[m] = s.to(device)
        return scores


class RandomPruner(Pruner):
    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        model = self.model
        device = example_inputs.device
        model.eval()
        scores: Dict[nn.Module, torch.Tensor] = {}
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                s = torch.rand(m.out_channels, device=device)
                scores[m] = s
        return scores


