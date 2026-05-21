"""
HM3D data root resolution. Call resolve_hm3d_root() at startup (e.g. from main.py) to set
HM3D_DATA_ROOT before any env/ppo imports.

Set the HM3D_DATA_ROOT environment variable to the directory containing the HM3D dataset.
See README.md for the expected directory layout.
"""
import os

_RESOLVED: str | None = None


def resolve_hm3d_root() -> str:
    """
    Resolve HM3D data root and set HM3D_DATA_ROOT in env. Call early (e.g. from main.py).
    Returns the resolved path.
    """
    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED

    _RESOLVED = os.environ.get("HM3D_DATA_ROOT", "/workspace/data")
    os.environ["HM3D_DATA_ROOT"] = _RESOLVED
    return _RESOLVED
