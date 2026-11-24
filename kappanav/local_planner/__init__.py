"""CurveNav: conditional curve Flow Matching for local navigation.

Model symbols are imported lazily so the CPU-only expert manifest builder can
run without importing PyTorch or initializing a CUDA runtime.
"""

__all__ = [
    "CurveNavConfig",
    "PolicyCondition",
    "build_policy",
]


def __getattr__(name: str):
    if name == "CurveNavConfig":
        from kappanav.local_planner.model.config import CurveNavConfig

        return CurveNavConfig
    if name == "build_policy":
        from kappanav.local_planner.model.factory import build_policy

        return build_policy
    if name == "PolicyCondition":
        from kappanav.local_planner.model.types import PolicyCondition

        return PolicyCondition
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
