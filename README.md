[![license](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit/)
[![Tests](https://github.com/omar-abdelgawad/python-project-template/actions/workflows/tests.yml/badge.svg)](https://github.com/omar-abdelgawad/python-project-template/actions)
[![PythonVersion](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->

# Seam Carving Implementation

This project implements the seam carving algorithm for content-aware image resizing. Seam carving is a technique that allows for image resizing while preserving important visual content by removing less important seams (paths of pixels) from the image.

## Lab Requirements

The implementation satisfies the following lab requirements:

1. Energy calculation formula: e₁ = |∂/∂x I| + |∂/∂y I|

   - Custom implementation without using predefined functions
   - Calculates both x and y gradients and sums their absolute values

2. The algorithm handles images with size up to 800×800 pixels and can reduce the image to half its original size.

3. The implementation generates two outputs:
   - The resized image (after seam removal)
   - A visualization of the seams that were removed (overlaid in red on the original image)

## Requirements

- Python 3.8 or higher
- NumPy
- Pillow (PIL)
- Requests (for downloading sample images)

## Installation

```bash
# Install required packages
pip install numpy pillow requests
```

## Usage Examples

### Basic Usage

```bash
# Create output directory
mkdir -p output

# Run the seam carving algorithm
python3 seam_carve.py examples/images/landscape.jpg output/landscape_resized.jpg --width 400 --height 265 --vis_output output/landscape_seams.jpg
```

The above command will:

1. Load the image from examples/images/landscape.jpg
2. Resize it to 400×265 pixels using seam carving
3. Save the resized image to output/landscape_resized.jpg
4. Save a visualization of the removed seams to output/landscape_seams.jpg

### Profiling
```bash
python3 profile_1.py examples/images/landscape.jpg output/landscape_resized.jpg --width 600 --height 500 --vis_output output/landscape_seams.jpg
```

### Additional Options

```bash
# Resize to a specific width (maintaining original height)
python3 seam_carve.py examples/images/landscape.jpg output/width_only.jpg --width 400 --vis_output output/width_only_seams.jpg

# Resize to a specific height (maintaining original width)
python3 seam_carve.py examples/images/landscape.jpg output/height_only.jpg --height 265 --vis_output output/height_only_seams.jpg

# Scale the image to a percentage of its original size
python3 seam_carve.py examples/images/landscape.jpg output/scaled.jpg --scale 0.5 --vis_output output/scaled_seams.jpg
```

## Examples Directory

The `examples` directory contains several scripts demonstrating different aspects of the seam carving algorithm. Each example showcases unique features and capabilities:

### 1. Download Sample Image

This utility downloads a sample landscape image for testing the seam carving algorithm.

```bash
python3 examples/download_sample_image.py
```

**Features:**

- Downloads a high-resolution landscape image from Pexels.com
- Automatically resizes the image to fit within 800×800 pixels
- Saves the image to `examples/images/landscape.jpg`

### 2. Basic Seam Carving Test

A simple test of the seam carving algorithm that reduces both dimensions of an image.

```bash
python3 examples/test_seam_carving.py examples/images/landscape.jpg --scale 0.5
```

**Features:**

- Resizes an image to a specified scale (default: 0.5, or half size)
- Outputs both resized image and visualization of removed seams
- Simple command-line interface with minimal options

**Options:**

- `--output_dir`: Directory to save output images (default: "output")
- `--scale`: Scale factor for width and height (0.0-1.0)

### 3. Advanced Seam Carving

Demonstrates advanced seam carving options, including separate control of width and height.

```bash
# Vertical seam carving only (reduce width)
python3 examples/advanced_seam_carving.py examples/images/landscape.jpg --vertical_only

# Horizontal seam carving only (reduce height)
python3 examples/advanced_seam_carving.py examples/images/landscape.jpg --horizontal_only

# Both horizontal and vertical seam carving (default)
python3 examples/advanced_seam_carving.py examples/images/landscape.jpg
```

**Features:**

- Allows separate control over width and height reduction
- Provides performance timing information
- More detailed output of dimensions and process

**Options:**

- `--output_dir`: Directory to save output images (default: "output")
- `--width_scale`: Scale factor for width (0.0-1.0, default: 0.5)
- `--height_scale`: Scale factor for height (0.0-1.0, default: 0.5)
- `--vertical_only`: Only perform vertical seam carving (width reduction)
- `--horizontal_only`: Only perform horizontal seam carving (height reduction)


**Features:**

- Compares three different approaches to seam carving:
  1. Standard implementation (baseline)
  2. Optimized implementation (with batch processing)
  3. Downscaled implementation (resize first, then seam carve)
- Provides detailed performance metrics and speedup calculations
- Saves all three result versions for quality comparison

**Options:**

- `--output_dir`: Directory to save output images (default: "output")
- `--scale`: Scale factor for width and height (0.0-1.0, default: 0.5)

**Key Optimization Strategies:**

- **Batch Processing**: Processes seams in small batches for better memory efficiency
- **Downscaling**: Reduces image size before seam carving for dramatic speed improvements

## Implementation Details

The seam carving algorithm follows these steps:

1. Calculate the energy of each pixel in the image using gradient magnitude
2. Compute the cumulative energy map using dynamic programming
3. Find the optimal seam (path of least energy)
4. Remove the seam from the image
5. Repeat until the target size is reached

The algorithm can reduce both width (by removing vertical seams) and height (by removing horizontal seams).

