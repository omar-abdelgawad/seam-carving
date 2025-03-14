# Seam Carving Examples

This directory contains several scripts demonstrating different aspects of the seam carving algorithm. Each example showcases unique features and capabilities.

## 1. Download Sample Image

This utility downloads a sample landscape image for testing the seam carving algorithm.

```bash
python3 download_sample_image.py
```

**Features:**

- Downloads a high-resolution landscape image from Pexels.com
- Automatically resizes the image to fit within 800×800 pixels
- Saves the image to `images/landscape.jpg`

## 2. Basic Seam Carving Test

A simple test of the seam carving algorithm that reduces both dimensions of an image.

```bash
python3 test_seam_carving.py images/landscape.jpg --scale 0.5
```

**Features:**

- Resizes an image to a specified scale (default: 0.5, or half size)
- Outputs both resized image and visualization of removed seams
- Simple command-line interface with minimal options

**Options:**

- `--output_dir`: Directory to save output images (default: "output")
- `--scale`: Scale factor for width and height (0.0-1.0)

## 3. Advanced Seam Carving

Demonstrates advanced seam carving options, including separate control of width and height.

```bash
# Vertical seam carving only (reduce width)
python3 advanced_seam_carving.py images/landscape.jpg --vertical_only

# Horizontal seam carving only (reduce height)
python3 advanced_seam_carving.py images/landscape.jpg --horizontal_only

# Both horizontal and vertical seam carving (default)
python3 advanced_seam_carving.py images/landscape.jpg
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

## 4. Performance Optimization

Compares different performance optimization strategies for seam carving.

```bash
python3 performance_optimization.py images/landscape.jpg
```

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

- **Batch Processing**: Processes seams in small batches for better memory efficiency and potentially improved CPU cache utilization
- **Downscaling**: Reduces image size before seam carving for dramatic speed improvements at some cost to precision

## Example Output Files

When running the examples, the following output files are created:

1. **Basic Test:**

   - `output/[image_name]_resized.jpg`: The resized image
   - `output/[image_name]_seams.jpg`: Visualization of the removed seams

2. **Advanced Example:**

   - `output/[image_name]_[dimensions].jpg`: The resized image (e.g., `landscape_w400.jpg` for width 400)
   - `output/[image_name]_[dimensions]_seams.jpg`: Visualization of removed seams

3. **Performance Optimization:**
   - `output/[image_name]_standard.jpg`: Result from standard algorithm
   - `output/[image_name]_standard_seams.jpg`: Standard seams visualization
   - `output/[image_name]_optimized.jpg`: Result from optimized algorithm
   - `output/[image_name]_optimized_seams.jpg`: Optimized seams visualization
   - `output/[image_name]_downscaled.jpg`: Result from downscaled algorithm
   - `output/[image_name]_downscaled_seams.jpg`: Downscaled seams visualization

## Finding Test Images

Good candidates for seam carving:

1. Landscape images with clear content areas and less important background areas
2. Images with distinct foreground and background elements
3. Images with a variety of textures and edges

Recommended sources:

- [Unsplash](https://unsplash.com/) - Free high-resolution photos
- [Pexels](https://www.pexels.com/) - Free stock photos
- Your own personal photos

## Notes

- For best results, use images that aren't too small (at least 400×400 pixels)
- The algorithm handles images up to 800×800 pixels efficiently
- Larger images may take significantly more time to process
- The seam carving algorithm works best when there are clear areas of low importance in the image
