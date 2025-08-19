from .base import PruningConfig, Pruner
from .taylor import TaylorBNPruner
from .slimming import SlimmingPruner
from .fpgm import FPGMPruner
from .norm import NormPruner, RandomPruner
from .dmcp import DMCPPruner
from .fgp import FGPPruner
from .sirfp import SIRFPPruner

PRUNER_REGISTRY = {
    "taylor": TaylorBNPruner,
    "slimming": SlimmingPruner,
    "fpgm": FPGMPruner,
    "l1": lambda m, c: NormPruner(m, c, norm_type="l1"),
    "l2": lambda m, c: NormPruner(m, c, norm_type="l2"),
    "random": RandomPruner,
    # Integrated methods
    "dmcp": DMCPPruner,
    "fgp": FGPPruner,
    "sirfp": SIRFPPruner,
}

def build_pruner(name: str, model, config: PruningConfig) -> Pruner:
    name = name.lower()
    if name not in PRUNER_REGISTRY:
        raise ValueError(f"Unsupported pruner: {name}")
    ctor = PRUNER_REGISTRY[name]
    if isinstance(ctor, type):
        return ctor(model, config)
    return ctor(model, config)


