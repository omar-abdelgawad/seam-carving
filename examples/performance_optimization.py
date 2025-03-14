#!/usr/bin/env python3
"""
Performance Optimization Examples for Seam Carving

This script demonstrates performance optimizations for the seam carving algorithm.
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.seam_carving.seam_carving import (
    calculate_energy,
    compute_cumulative_energy_map,
    find_optimal_seam,
    load_image,
    remove_seam,
    save_image,
    seam_carve,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Performance Optimization Examples for Seam Carving"
    )

    parser.add_argument("input_image", type=str, help="Path to the input image")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=0.5,
        help="Scale factor for width and height (default: 0.5)",
    )

    return parser.parse_args()


def optimized_seam_carving(
    image: np.ndarray,
    target_width: "int | None" = None,
    target_height: "int | None " = None,
) -> tuple:
    """
    Optimized seam carving function.

    Args:
        image: A numpy array representing the image (height, width, channels)
        target_width: The desired width of the output image
        target_height: The desired height of the output image

    Returns:
        A tuple containing:
        - The resized image
        - A visualization of the removed seams
    """
    if target_width is None and target_height is None:
        raise ValueError("Either target_width or target_height must be specified")

    # Create copies of the image to work with
    result = np.copy(image)
    visualization = np.copy(image)

    original_height, original_width = image.shape[:2]

    # Calculate the number of seams to remove
    width_seams_to_remove = (
        (original_width - target_width) if target_width is not None else 0
    )
    height_seams_to_remove = (
        (original_height - target_height) if target_height is not None else 0
    )

    # Ensure we're not trying to remove more seams than we have
    width_seams_to_remove = max(0, min(width_seams_to_remove, original_width - 1))
    height_seams_to_remove = max(0, min(height_seams_to_remove, original_height - 1))

    # Process vertical seams first
    if width_seams_to_remove > 0:
        # Optimization: Split the task into smaller batches
        batch_size = min(width_seams_to_remove, 20)
        for batch_start in range(0, width_seams_to_remove, batch_size):
            batch_end = min(batch_start + batch_size, width_seams_to_remove)

            # Find and remove vertical seams one by one for this batch
            for _ in range(batch_start, batch_end):
                # Calculate energy for the current image
                energy = calculate_energy(result)

                # Compute cumulative energy map
                cumulative_energy = compute_cumulative_energy_map(energy, "vertical")

                # Find the optimal seam
                seam = find_optimal_seam(cumulative_energy, "vertical")

                # Find a corresponding seam in the visualization image
                vis_energy = calculate_energy(visualization)
                vis_cumulative_energy = compute_cumulative_energy_map(
                    vis_energy, "vertical"
                )
                vis_seam = find_optimal_seam(vis_cumulative_energy, "vertical")

                # Visualize the seam
                visualization = visualize_seams(visualization, [vis_seam], "vertical")

                # Remove the seam from the result image
                result = remove_seam(result, seam, "vertical")

    # Process horizontal seams
    if height_seams_to_remove > 0:
        # Optimization: Split the task into smaller batches
        batch_size = min(height_seams_to_remove, 20)
        for batch_start in range(0, height_seams_to_remove, batch_size):
            batch_end = min(batch_start + batch_size, height_seams_to_remove)

            # Find and remove horizontal seams one by one for this batch
            for _ in range(batch_start, batch_end):
                # Calculate energy for the current image
                energy = calculate_energy(result)

                # Compute cumulative energy map
                cumulative_energy = compute_cumulative_energy_map(energy, "horizontal")

                # Find the optimal seam
                seam = find_optimal_seam(cumulative_energy, "horizontal")

                # Find a corresponding seam in the visualization image
                vis_energy = calculate_energy(visualization)
                vis_cumulative_energy = compute_cumulative_energy_map(
                    vis_energy, "horizontal"
                )
                vis_seam = find_optimal_seam(vis_cumulative_energy, "horizontal")

                # Visualize the seam
                visualization = visualize_seams(visualization, [vis_seam], "horizontal")

                # Remove the seam from the result image
                result = remove_seam(result, seam, "horizontal")

    return result, visualization


def visualize_seams(
    image: np.ndarray, seams: list, direction: str = "vertical"
) -> np.ndarray:
    """
    Create a visualization of the image with the seams highlighted.

    Args:
        image: A numpy array representing the image (height, width, channels)
        seams: A list of seams to visualize
        direction: 'vertical' for vertical seams, 'horizontal' for horizontal seams

    Returns:
        A numpy array representing the image with seams highlighted in red
    """
    if direction not in ["vertical", "horizontal"]:
        raise ValueError("Direction must be either 'vertical' or 'horizontal'")

    # Create a copy of the image to draw seams on
    visualization = np.copy(image)
    height, width = image.shape[:2]

    # Draw each seam in red
    for seam in seams:
        if direction == "vertical":
            for i in range(min(len(seam), height)):
                if 0 <= seam[i] < width:  # Check bounds
                    if len(image.shape) == 3:
                        visualization[i, seam[i]] = [255, 0, 0]  # Red color
                    else:
                        visualization[i, seam[i]] = 255  # White for grayscale
        else:  # horizontal
            for j in range(min(len(seam), width)):
                if 0 <= seam[j] < height:  # Check bounds
                    if len(image.shape) == 3:
                        visualization[seam[j], j] = [255, 0, 0]  # Red color
                    else:
                        visualization[seam[j], j] = 255  # White for grayscale

    return visualization


def downscale_image(image: np.ndarray, max_size: int = 400) -> np.ndarray:
    """
    Downscale an image to have maximum dimension of max_size.

    Args:
        image: Input image as numpy array
        max_size: Maximum dimension (width or height) of the output image

    Returns:
        Downscaled image
    """
    height, width = image.shape[:2]

    # If the image is already small enough, return it as is
    if width <= max_size and height <= max_size:
        return image

    # Calculate the scaling factor
    scale = min(max_size / width, max_size / height)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize using PIL
    if len(image.shape) == 3:
        # Color image
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((new_width, new_height))
        return np.array(resized_image)
    else:
        # Grayscale image
        pil_image = Image.fromarray(image)
        resized_image = pil_image.resize((new_width, new_height))
        return np.array(resized_image)


def main() -> None:
    """Run the performance optimization examples."""
    args = parse_args()

    # Load the input image
    print(f"Loading image: {args.input_image}")
    original_image = load_image(args.input_image)
    height, width = original_image.shape[:2]
    print(f"Original image dimensions: {width}x{height}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output paths
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]

    # Calculate target dimensions
    target_width = int(width * args.scale_factor)
    target_height = int(height * args.scale_factor)
    print(f"Target dimensions: {target_width}x{target_height}")

    # 1. Standard seam carving for baseline
    print("\n1. Running standard seam carving as baseline...")
    start_time = time.time()
    standard_result, standard_vis = seam_carve(
        original_image, target_width=target_width, target_height=target_height
    )
    standard_time = time.time() - start_time
    print(f"   Standard seam carving completed in {standard_time:.2f} seconds")

    # Save results
    standard_path = os.path.join(args.output_dir, f"{base_name}_standard.jpg")
    standard_vis_path = os.path.join(args.output_dir, f"{base_name}_standard_seams.jpg")
    save_image(standard_result, standard_path)
    save_image(standard_vis, standard_vis_path)

    # 2. Optimized seam carving
    print("\n2. Running optimized seam carving...")
    start_time = time.time()
    optimized_result, optimized_vis = optimized_seam_carving(
        original_image, target_width=target_width, target_height=target_height
    )
    optimized_time = time.time() - start_time
    print(f"   Optimized seam carving completed in {optimized_time:.2f} seconds")

    # Save results
    optimized_path = os.path.join(args.output_dir, f"{base_name}_optimized.jpg")
    optimized_vis_path = os.path.join(
        args.output_dir, f"{base_name}_optimized_seams.jpg"
    )
    save_image(optimized_result, optimized_path)
    save_image(optimized_vis, optimized_vis_path)

    # 3. Downscaled then seam carved
    print("\n3. Running seam carving on downscaled image...")
    # Downscale the image
    max_size = min(400, min(width, height))
    downscaled_image = downscale_image(original_image, max_size)
    d_height, d_width = downscaled_image.shape[:2]
    print(f"   Downscaled image dimensions: {d_width}x{d_height}")

    # Calculate new target dimensions proportionally
    d_target_width = int(d_width * args.scale_factor)
    d_target_height = int(d_height * args.scale_factor)

    # Run seam carving on downscaled image
    start_time = time.time()
    downscaled_result, downscaled_vis = seam_carve(
        downscaled_image, target_width=d_target_width, target_height=d_target_height
    )
    downscaled_time = time.time() - start_time
    print(
        f"   Seam carving on downscaled image completed in {downscaled_time:.2f} seconds"
    )

    # Save results
    downscaled_path = os.path.join(args.output_dir, f"{base_name}_downscaled.jpg")
    downscaled_vis_path = os.path.join(
        args.output_dir, f"{base_name}_downscaled_seams.jpg"
    )
    save_image(downscaled_result, downscaled_path)
    save_image(downscaled_vis, downscaled_vis_path)

    # Performance summary
    print("\nPerformance Summary:")
    print(f"1. Standard Seam Carving: {standard_time:.2f} seconds")
    print(f"2. Optimized Seam Carving: {optimized_time:.2f} seconds")
    print(f"3. Downscaled Seam Carving: {downscaled_time:.2f} seconds")

    speedup1 = standard_time / optimized_time if optimized_time > 0 else float("inf")
    speedup2 = standard_time / downscaled_time if downscaled_time > 0 else float("inf")

    print(f"Speedup from optimization: {speedup1:.2f}x")
    print(f"Speedup from downscaling: {speedup2:.2f}x")


if __name__ == "__main__":
    main()
