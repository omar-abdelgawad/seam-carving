from typing import Tuple

import numpy as np
from PIL import Image


def load_image(image_path) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(image_path)
    # get original image dimensions
    orig_dims = img.size
    max_dim = max(img.size)
    prefered_size = 800
    if max_dim > prefered_size:
        img = img.resize(
            (
                int(img.width * prefered_size / max_dim),
                int(img.height * prefered_size / max_dim),
            )
        )
    return np.array(img), orig_dims


def save_image(image: np.ndarray, output_path: str) -> None:
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    img = Image.fromarray(image)
    img.save(output_path)
