"""
Tests for the seam carving implementation.
"""

import numpy as np

from seam_carving import (
    calculate_energy,
    find_seams,
    remove_seams,
    seam_carve,
)


def test_calculate_energy():
    """Test the energy calculation function."""
    # Create a simple test image
    image = np.array([[100, 100, 100], [100, 0, 100], [100, 100, 100]], dtype=np.uint8)

    # Calculate energy
    energy = calculate_energy(image)
    expected_energy = np.array([[0, 100, 0], [100, 0, 100], [0, 100, 0]])
    # The middle pixel should have high energy (strong gradient)
    assert energy.shape == image.shape
    assert np.all(energy == expected_energy)


def test_seam_removal_vertical():
    """Test removal of a vertical seam."""
    # Create a simple test image
    image = np.zeros((5, 5), dtype=np.uint8)

    # Create a seam (column of indices)
    seam = np.array([2, 2, 2, 2, 2])  # Remove middle column

    # Remove the seam
    result = remove_seams(image, [seam], direction="vertical")

    # Check that the result has one fewer column
    assert result.shape == (5, 4)


def test_seam_removal_horizontal():
    """Test removal of a horizontal seam."""
    # Create a simple test image
    image = np.zeros((5, 5), dtype=np.uint8)

    # Create a seam (row of indices)
    seam = np.array([2, 2, 2, 2, 2])  # Remove middle row

    # Remove the seam
    result = remove_seams(image, [seam], direction="horizontal")

    # Check that the result has one fewer row
    assert result.shape == (4, 5)


def test_find_seams():
    """Test finding multiple seams."""
    # Create a simple test image with a gradient
    image = np.zeros((10, 10), dtype=np.uint8)
    for i in range(10):
        for j in range(10):
            image[i, j] = abs(j - 5)  # Make middle column lowest energy

    # Find 3 vertical seams
    seams = find_seams(image, 3, direction="vertical")

    # Check that we got 3 seams
    assert len(seams) == 3

    # Each seam should have 10 elements (one for each row)
    for seam in seams:
        assert len(seam) == 10


def test_seam_carve_resize():
    """Test the complete seam carving process."""
    # Create a simple test image
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    # Perform seam carving to reduce width and height
    target_width = 25
    target_height = 15

    # Carve the image
    carved_image, _ = seam_carve(image, target_width, target_height)

    # Check the dimensions of the result
    assert carved_image.shape == (target_height, target_width, 3)
