"""Seam Carving implementation for content-aware image resizing."""

from .__version__ import __version__
from .seam_carving import (
    calculate_energy,
    seam_carve,
    visualize_seams,
)
from .utils import load_image, save_image

__all__ = [
    "calculate_energy",
    "seam_carve",
    "visualize_seams",
    "load_image",
    "save_image",
    "__version__",
]
