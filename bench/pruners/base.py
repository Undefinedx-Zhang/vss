from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch_pruning as tp


@dataclass
class PruningConfig:
    global_ratio: float = 0.3
    layer_wise: bool = False
    norm_type: str = "l2"
    slimming_threshold: Optional[float] = None
    slimming_lambda: float = 0.0
    taylor_samples: int = 64
    fgp_samples: int = 64
    sirfp_samples: int = 32
    sirfp_corr_threshold: float = 0.8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    input_size: tuple = (1, 3, 32, 32)
    ignore_layers: Optional[List[str]] = None


class Pruner:
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        # Try to infer number of classes for segmentation/classification heads
        self.num_classes = getattr(model, "num_classes", None)

    def importance_scores(self, example_inputs: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        raise NotImplementedError

    def _build_dependency(self, example_inputs: torch.Tensor) -> tp.DependencyGraph:
        self.model.eval()
        DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=example_inputs)
        return DG

    def _collect_prunable(self) -> List[nn.Module]:
        prunable = []
        for name, m in self.model.named_modules():
            # prune only Conv2d with out_channels > 1
            if isinstance(m, nn.Conv2d) and m.out_channels > 1:
                # Optionally ignore specified layers by name
                if self.config.ignore_layers and any(ig in name for ig in self.config.ignore_layers):
                    continue
                # Do not prune final classification head (e.g., logits layer)
                if self.num_classes is not None and m.out_channels == self.num_classes:
                    continue
                prunable.append(m)
        return prunable

    def prune(self, example_inputs: torch.Tensor) -> nn.Module:
        scores = self.importance_scores(example_inputs)
        DG = self._build_dependency(example_inputs)
        prunable = self._collect_prunable()

        groups = []
        for m in prunable:
            group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=[])
            if group is not None:
                groups.append(group)

        # gather global threshold (only from modules that have valid scores)
        all_scores = []
        mapping: Dict[nn.Module, torch.Tensor] = {}
        for m in prunable:
            s = scores.get(m)
            if s is None or s.numel() == 0:
                # skip modules without valid scores (e.g., non-BN-following convs for Taylor)
                continue
            mapping[m] = s
            all_scores.append(s.detach().view(-1))
        if len(all_scores) == 0:
            return self.model
        all_scores = torch.cat(all_scores)
        k = int(self.config.global_ratio * all_scores.numel())
        if k <= 0:
            return self.model
        threshold = torch.topk(all_scores, k, largest=False).values.max()

        for m in prunable:
            s = mapping.get(m)
            if s is None or s.numel() == 0:
                continue
            idxs = torch.nonzero(s <= threshold, as_tuple=False).view(-1).tolist()
            if len(idxs) == 0:
                continue
            # Ensure at least 1 output channel remains
            max_prunable = max(0, m.out_channels - 1)
            if len(idxs) > max_prunable:
                # Keep channels with highest importance; prune the least important up to max_prunable
                # Sort selected idxs by their scores ascending (least important first)
                scores_subset = [(idx, float(s[idx].item())) for idx in idxs]
                scores_subset.sort(key=lambda t: t[1])
                idxs = [idx for idx, _ in scores_subset[:max_prunable]]
            if len(idxs) == 0:
                continue
            group = DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs=idxs)
            if group is None:
                continue
            group.prune()

        return self.model


