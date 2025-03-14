#!/usr/bin/env python3
"""
Download a sample landscape image for testing the seam carving algorithm.
"""

import os
import requests
from PIL import Image
from io import BytesIO

# URL of a sample landscape image (from Pexels.com, under free license)
sample_image_url = "https://images.pexels.com/photos/1619317/pexels-photo-1619317.jpeg"
output_dir = os.path.join(os.path.dirname(__file__), "images")
output_path = os.path.join(output_dir, "landscape.jpg")


def download_image():
    """Download the sample image and save it to the images directory."""
    print(f"Downloading sample image from {sample_image_url}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download the image
    response = requests.get(sample_image_url)
    response.raise_for_status()

    # Open the image and resize it if needed
    image = Image.open(BytesIO(response.content))

    # Save the image
    image.save(output_path)
    print(f"Sample image saved to {output_path}")


if __name__ == "__main__":
    download_image()
