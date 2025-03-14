"""
Command-line interface for the seam carving algorithm.
"""

import argparse
import os
import time
from typing import Tuple

import numpy as np

from seam_carving import load_image, save_image, seam_carve


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Seam Carving for Content-Aware Image Resizing"
    )

    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument("output_image", type=str, help="Path to save the resized image")
    parser.add_argument(
        "--vis_output", type=str, help="Path to save the visualization of removed seams"
    )

    # Specify either width or height
    size_group = parser.add_argument_group("Size options (specify one or both)")
    size_group.add_argument(
        "--width", type=int, help="Target width for the output image"
    )
    size_group.add_argument(
        "--height", type=int, help="Target height for the output image"
    )
    size_group.add_argument(
        "--scale", type=float, help="Scale factor (0.1-1.0) for both dimensions"
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate that at least one of --width, --height, or --scale is specified
    if args.width is None and args.height is None and args.scale is None:
        parser.error("At least one of --width, --height, or --scale must be specified")

    return args


def get_target_dimensions(
    image: np.ndarray,
    width: "int | None" = None,
    height: "int | None" = None,
    scale: "float | None" = None,
) -> Tuple[int, int]:
    """
    Calculate target dimensions based on command-line arguments.

    Args:
        image: Input image
        width: Target width (if specified)
        height: Target height (if specified)
        scale: Scale factor (if specified)

    Returns:
        Tuple of (target_width, target_height)
    """
    img_height, img_width = image.shape[:2]

    # If scale is specified, calculate width and height from it
    if scale is not None:
        if not (0.1 <= scale <= 1.0):
            raise ValueError("Scale must be between 0.1 and 1.0")
        target_width = int(img_width * scale)
        target_height = int(img_height * scale)
    else:
        # Use specified width/height or keep original
        target_width = width if width is not None else img_width
        target_height = height if height is not None else img_height

    # Validate target dimensions
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target dimensions must be positive")
    if target_width > img_width:
        raise ValueError(
            f"Target width ({target_width}) is larger than original width ({img_width})"
        )
    if target_height > img_height:
        raise ValueError(
            f"Target height ({target_height}) is larger than original height ({img_height})"
        )

    return target_width, target_height


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()

    # Load the input image
    print(f"Loading image: {args.input_image}")
    image, orig_dims = load_image(args.input_image)

    # Get target dimensions
    target_width, target_height = get_target_dimensions(
        image, args.width, args.height, args.scale
    )

    # Print dimensions
    original_width, original_height = orig_dims
    resized_height, resized_width = image.shape[:2]
    print(f"Original dimensions: {original_width}x{original_height}")
    print(f"Resized dimensions: {resized_width}x{resized_height}")
    print(f"Target dimensions: {target_width}x{target_height}")

    # Perform seam carving
    print("Performing seam carving...")
    start_time = time.time()
    carved_image, seams_visualization = seam_carve(image, target_width, target_height)
    end_time = time.time()

    # Print timing information
    print(f"Seam carving completed in {end_time - start_time:.2f} seconds")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_image)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the resized image
    print(f"Saving resized image to: {args.output_image}")
    save_image(carved_image, args.output_image)

    # Save the visualization if requested
    if args.vis_output:
        vis_dir = os.path.dirname(args.vis_output)
        if vis_dir and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        print(f"Saving seam visualization to: {args.vis_output}")
        save_image(seams_visualization, args.vis_output)

    print("Done!")


if __name__ == "__main__":
    main()
