#!/usr/bin/env python3
"""
Augmentation.py - Leaffliction Part 2: Data Augmentation

Applies 6 types of data augmentation to leaf images.
Saves augmented images with original filename + augmentation type.

Usage:
    ./Augmentation.py <image_path>
    ./Augmentation.py ./Apple/apple_healthy/image (1).JPG
"""

import sys
from pathlib import Path

import cv2 as cv
from plantcv import plantcv as pcv

# Disable PlantCV debug output
pcv.params.debug = None


def augment_flip(image):
    """Horizontal flip."""
    return pcv.flip(img=image, direction="horizontal")


def augment_rotate(image):
    """Rotation 90 degrees clockwise."""
    # PlantCV doesn't have a simple rotate function, use OpenCV
    return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)


def augment_brightness(image):
    """Increase brightness by 30%."""
    # Simple brightness increase using cv.convertScaleAbs
    return cv.convertScaleAbs(image, alpha=1.3, beta=0)


def augment_contrast(image):
    """Increase contrast by 50%."""
    # Convert to LAB color space for better contrast adjustment
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)
    # Apply CLAHE to L channel
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    # Merge and convert back
    lab = cv.merge([l_channel, a, b])
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)


def augment_crop(image):
    """Center crop to 80% and resize back."""
    height, width = image.shape[:2]

    # Calculate crop coordinates (center crop at 80%)
    crop_ratio = 0.8
    crop_h = int(height * crop_ratio)
    crop_w = int(width * crop_ratio)

    x = (width - crop_w) // 2
    y = (height - crop_h) // 2

    # Crop using PlantCV
    cropped = pcv.crop(img=image, x=x, y=y, h=crop_h, w=crop_w)
    # Resize back to original size
    return cv.resize(cropped, (width, height), interpolation=cv.INTER_CUBIC)


def augment_blur(image):
    """Gaussian blur with 7x7 kernel."""
    return cv.GaussianBlur(image, (7, 7), 2.0)


# Augmentation mapping
AUGMENTATIONS = {
    'Flip': augment_flip,
    'Rotate': augment_rotate,
    'Brightness': augment_brightness,
    'Contrast': augment_contrast,
    'Crop': augment_crop,
    'Blur': augment_blur
}


def apply_augmentations(image_path):
    """Apply all 6 augmentations and save results.

    Args:
        image_path: Path to the input image

    Returns:
        list: Paths to generated augmented images
    """
    # Load image
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image '{image_path}'")
        sys.exit(1)

    # Get base name and extension
    base_name = image_path.stem
    extension = image_path.suffix
    output_dir = image_path.parent

    output_paths = []

    # Apply each augmentation
    for aug_name, aug_func in AUGMENTATIONS.items():
        # Apply augmentation
        augmented = aug_func(image.copy())

        # Create output filename
        output_name = f"{base_name}_{aug_name}{extension}"
        output_path = output_dir / output_name

        # Save
        cv.imwrite(str(output_path), augmented)
        output_paths.append(output_path)
        print(f"  Created: {output_name}")

    return output_paths


def main():
    """Main function."""
    # Check arguments
    if len(sys.argv) != 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print("Augmentation.py - Data Augmentation Tool")
        print("\nUsage:")
        print("  ./Augmentation.py <image_path>")
        print("\nDescription:")
        print("  Applies 6 types of augmentation to a leaf image:")
        print("  Flip, Rotate, Brightness, Contrast, Crop, Blur")
        print("\nExample:")
        print("  ./Augmentation.py ./Apple/apple_healthy/image (1).JPG")
        print("\nOutput:")
        print("  image (1)_Flip.JPG")
        print("  image (1)_Rotate.JPG")
        print("  image (1)_Brightness.JPG")
        print("  image (1)_Contrast.JPG")
        print("  image (1)_Crop.JPG")
        print("  image (1)_Blur.JPG")
        sys.exit(0 if len(sys.argv) == 2 else 1)

    # Get image path
    image_path = Path(sys.argv[1]).resolve()

    # Validate
    if not image_path.exists():
        print(f"Error: File '{image_path}' does not exist")
        sys.exit(1)

    if not image_path.is_file():
        print(f"Error: '{image_path}' is not a file")
        sys.exit(1)

    # Check if it's an image
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if image_path.suffix.lower() not in valid_ext:
        print(f"Error: '{image_path}' is not a valid image file")
        sys.exit(1)

    # Process image
    print(f"\nProcessing: {image_path.name}")
    print("Applying 6 augmentations...\n")

    output_paths = apply_augmentations(image_path)

    print(f"\n✓ Successfully created {len(output_paths)} augmented images")
    print(f"  Saved in: {image_path.parent}")


if __name__ == "__main__":
    main()
