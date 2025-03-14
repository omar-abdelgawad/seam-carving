"""
Implementation of the Seam Carving algorithm for content-aware image resizing.

This module provides functions to:
1. Calculate energy in an image using gradients
2. Find optimal seams to remove (vertical or horizontal)
3. Remove seams from the image
4. Visualize the seams that were removed
"""

from typing import List, Optional, Tuple

import numpy as np


def calculate_energy(image: np.ndarray) -> np.ndarray:
    """
    Calculate the energy of an image using gradient magnitude.
    Args:
        image: A numpy array representing the image (height, width, channels)
    Returns:
        A 2D numpy array representing the energy at each pixel
    """
    # Convert image to grayscale if it has multiple channels
    if len(image.shape) == 3 and image.shape[2] > 1:
        # Convert to grayscale using weighted average (same as RGB to Luminance conversion)
        gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray_image = image

    gray_image = gray_image.astype(np.int16)
    # Calculate x gradient using forward and backward differences at the edges
    # and central differences elsewhere
    grad_x = np.zeros_like(gray_image)
    grad_x[:, 1:-1] = np.abs(gray_image[:, 2:] - gray_image[:, :-2]) / 2
    grad_x[:, 0] = np.abs(gray_image[:, 1] - gray_image[:, 0])
    grad_x[:, -1] = np.abs(gray_image[:, -1] - gray_image[:, -2])

    # Calculate y gradient using forward and backward differences at the edges
    # and central differences elsewhere
    grad_y = np.zeros_like(gray_image)
    grad_y[1:-1, :] = np.abs(gray_image[2:, :] - gray_image[:-2, :]) / 2
    grad_y[0, :] = np.abs(gray_image[1, :] - gray_image[0, :])
    grad_y[-1, :] = np.abs(gray_image[-1, :] - gray_image[-2, :])
    assert grad_x.shape == grad_y.shape
    assert grad_x.shape == gray_image.shape
    # Energy is the sum of the absolute values of the x and y gradients
    energy = np.abs(grad_x) + np.abs(grad_y)

    return energy


def compute_cumulative_energy_map(
    energy: np.ndarray, direction: str = "vertical"
) -> np.ndarray:
    """
    Compute the cumulative energy map for finding optimal seams.

    Args:
        energy: A 2D numpy array of energy values
        direction: 'vertical' for vertical seams, 'horizontal' for horizontal seams

    Returns:
        A 2D numpy array of cumulative energy values
    """
    if direction not in ["vertical", "horizontal"]:
        raise ValueError("Direction must be either 'vertical' or 'horizontal'")

    # For vertical seams, we traverse from top to bottom
    # For horizontal seams, we transpose the energy map and traverse from left to right
    if direction == "horizontal":
        energy = energy.T

    height, width = energy.shape
    cumulative_energy = np.copy(energy)

    # Fill the cumulative energy map
    for i in range(1, height):
        left = np.roll(cumulative_energy[i - 1], 1)
        left[0] = 2**15 - 1
        right = np.roll(cumulative_energy[i - 1], -1)
        right[-1] = 2**15 - 1
        center = cumulative_energy[i - 1]
        cumulative_energy[i] += np.minimum(center, np.minimum(left, right))
    # Transpose back if we're finding horizontal seams
    if direction == "horizontal":
        cumulative_energy = cumulative_energy.T

    return cumulative_energy


def find_optimal_seam(
    cumulative_energy: np.ndarray, direction: str = "vertical"
) -> np.ndarray:
    """
    Find the optimal seam in the cumulative energy map.

    Args:
        cumulative_energy: A 2D numpy array of cumulative energy values
        direction: 'vertical' for vertical seams, 'horizontal' for horizontal seams

    Returns:
        A 1D numpy array with indices of the optimal seam
    """
    if direction not in ["vertical", "horizontal"]:
        raise ValueError("Direction must be either 'vertical' or 'horizontal'")

    # For horizontal seams, we transpose the energy map
    if direction == "horizontal":
        cumulative_energy = cumulative_energy.T

    height, width = cumulative_energy.shape

    # Find the index of the minimum value in the last row
    seam_idx = np.zeros(height, dtype=np.int32)
    seam_idx[-1] = np.argmin(cumulative_energy[-1])

    # Backtrack to find the path of the seam
    for i in range(height - 2, -1, -1):
        prev_idx = seam_idx[i + 1]

        # Check the three possible parents
        if prev_idx == 0:
            # Only check present and right parent
            idx_range = [prev_idx, prev_idx + 1]
        elif prev_idx == width - 1:
            # Only check left and present parent
            idx_range = [prev_idx - 1, prev_idx]
        else:
            # Check all three parents
            idx_range = [prev_idx - 1, prev_idx, prev_idx + 1]

        # Find the parent with minimum cumulative energy
        seam_idx[i] = idx_range[
            np.argmin([cumulative_energy[i, idx] for idx in idx_range])
        ]

    # If we were finding a horizontal seam, convert row indices to column indices
    if direction == "horizontal":
        # Swap the meanings of the indices
        seam_idx = np.column_stack((np.arange(height), seam_idx))[:, ::-1]
        seam_idx = seam_idx[:, 0]  # The y-coordinates

    return seam_idx


def remove_seam(
    image: np.ndarray, seam: np.ndarray, direction: str = "vertical"
) -> np.ndarray:
    """
    Remove a single seam from the image.

    Args:
        image: A numpy array representing the image (height, width, channels)
        seam: A 1D numpy array with indices of the seam to remove
        direction: 'vertical' for vertical seams, 'horizontal' for horizontal seams

    Returns:
        A numpy array representing the image with the seam removed
    """
    if direction not in ["vertical", "horizontal"]:
        raise ValueError("Direction must be either 'vertical' or 'horizontal'")

    height, width = image.shape[:2]

    if direction == "horizontal":
        # Transpose image for horizontal seams
        return (
            remove_seam(
                image.transpose(1, 0, 2) if len(image.shape) == 3 else image.T,
                seam,
                "vertical",
            ).transpose(1, 0, 2)
            if len(image.shape) == 3
            else remove_seam(image.T, seam, "vertical").T
        )

    # Create a mask to exclude the seam
    mask = np.ones((height, width), dtype=bool)

    # Make sure the seam indices are within bounds
    for i in range(height):
        if 0 <= seam[i] < width:  # Ensure index is within bounds
            mask[i, seam[i]] = False

    # Reshape the mask to include all color channels
    if len(image.shape) == 3:
        mask = np.stack([mask] * image.shape[2], axis=2)

    # Remove the seam and reshape
    if len(image.shape) == 3:
        # For color images
        result = np.zeros((height, width - 1, image.shape[2]), dtype=image.dtype)
        for i in range(height):
            # Copy all pixels except the seam pixel for each row
            new_row = np.delete(image[i], seam[i], axis=0)
            result[i] = new_row
    else:
        # For grayscale images
        result = np.zeros((height, width - 1), dtype=image.dtype)
        for i in range(height):
            # Copy all pixels except the seam pixel for each row
            result[i] = np.delete(image[i], seam[i])

    return result


def visualize_seams(
    image: np.ndarray, seams: List[np.ndarray], direction: str = "vertical"
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


def seam_carve(
    image: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform seam carving to resize an image to target dimensions.

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
        raise ValueError(
            "At least one of target_width or target_heigt must be specified"
        )

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
        # Find and remove vertical seams one by one
        for _ in range(width_seams_to_remove):
            # Calculate energy for the current image
            energy = calculate_energy(result)

            # Compute cumulative energy map
            cumulative_energy = np.array(
                compute_cumulative_energy_map(energy, "vertical")
            )

            # Find the optimal seam
            seam = find_optimal_seam(cumulative_energy, "vertical")

            # Find a corresponding seam in the original image dimensions for visualization
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
        # Find and remove horizontal seams one by one
        for _ in range(height_seams_to_remove):
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
