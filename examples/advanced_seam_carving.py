#!/usr/bin/env python3
"""
Advanced Seam Carving Example

This script demonstrates more advanced usages of the seam carving algorithm:
1. Vertical seam carving only (reducing width)
2. Horizontal seam carving only (reducing height)
3. Combined seam carving (reducing both dimensions)
4. Performance measurement
"""

import argparse
import os
import sys
import time

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.seam_carving.seam_carving import (
    load_image,
    save_image,
    seam_carve,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Seam Carving Example")

    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--width_scale",
        type=float,
        default=0.5,
        help="Scale factor for width (default: 0.5)",
    )
    parser.add_argument(
        "--height_scale",
        type=float,
        default=0.5,
        help="Scale factor for height (default: 0.5)",
    )
    parser.add_argument(
        "--vertical_only",
        action="store_true",
        help="Perform vertical seam carving only (reduce width)",
    )
    parser.add_argument(
        "--horizontal_only",
        action="store_true",
        help="Perform horizontal seam carving only (reduce height)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the advanced seam carving examples."""
    args = parse_args()

    # Load the input image
    print(f"Loading image: {args.input_image}")
    image = load_image(args.input_image)
    height, width = image.shape[:2]
    print(f"Original image dimensions: {width}x{height}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output paths
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]

    # Determine target dimensions based on args
    if args.vertical_only:
        # Only reduce width
        target_width = int(width * args.width_scale)
        target_height = None
        output_suffix = f"w{target_width}"
    elif args.horizontal_only:
        # Only reduce height
        target_width = None
        target_height = int(height * args.height_scale)
        output_suffix = f"h{target_height}"
    else:
        # Reduce both dimensions
        target_width = int(width * args.width_scale)
        target_height = int(height * args.height_scale)
        output_suffix = f"w{target_width}_h{target_height}"

    # Print target dimensions
    print(
        f"Target dimensions: width={target_width or 'unchanged'}, height={target_height or 'unchanged'}"
    )

    # Define output paths
    resized_path = os.path.join(args.output_dir, f"{base_name}_{output_suffix}.jpg")
    seams_path = os.path.join(args.output_dir, f"{base_name}_{output_suffix}_seams.jpg")

    # Measure performance
    start_time = time.time()

    # Perform seam carving
    print("Performing seam carving...")
    resized_image, seams_visualization = seam_carve(
        image, target_width=target_width, target_height=target_height
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Seam carving completed in {elapsed_time:.2f} seconds")

    # Get final dimensions
    final_height, final_width = resized_image.shape[:2]
    print(f"Final image dimensions: {final_width}x{final_height}")

    # Save the output images
    print(f"Saving resized image to: {resized_path}")
    save_image(resized_image, resized_path)

    print(f"Saving seams visualization to: {seams_path}")
    save_image(seams_visualization, seams_path)

    print("Done!")


if __name__ == "__main__":
    main()
