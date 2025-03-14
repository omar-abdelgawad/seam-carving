#!/usr/bin/env python3
"""
Test the seam carving algorithm on a sample image.
"""

import argparse
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from seam_carving import load_image, save_image, seam_carve


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the seam carving algorithm on a sample image"
    )

    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Scale factor for width and height (default: 0.5)",
    )

    return parser.parse_args()


def main() -> None:
    """Run the seam carving algorithm on the input image."""
    args = parse_args()

    # Load the input image
    print(f"Loading image: {args.input_image}")
    image, _ = load_image(args.input_image)
    height, width = image.shape[:2]
    print(f"Image dimensions: {width}x{height}")

    # Calculate target dimensions
    target_width = int(width * args.scale)
    target_height = int(height * args.scale)
    print(f"Target dimensions: {target_width}x{target_height}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output paths
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    resized_path = os.path.join(args.output_dir, f"{base_name}_resized.jpg")
    seams_path = os.path.join(args.output_dir, f"{base_name}_seams.jpg")

    # Perform seam carving
    print("Performing seam carving...")
    resized_image, seams_visualization = seam_carve(
        image, target_width=target_width, target_height=target_height
    )

    # Save the output images
    print(f"Saving resized image to: {resized_path}")
    save_image(resized_image, resized_path)

    print(f"Saving seams visualization to: {seams_path}")
    save_image(seams_visualization, seams_path)

    print("Done!")


if __name__ == "__main__":
    main()
