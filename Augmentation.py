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
import json
import shutil

import cv2 as cv
from plantcv import plantcv as pcv

# Disable PlantCV debug output
pcv.params.debug = None


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def is_image(filename):
    """Check if file is an image based on extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


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


def balance_dataset(stats_file, output_dir):
    """Balance dataset based on statistics JSON file.

    Args:
        stats_file: Path to stats JSON from Distribution.py
        output_dir: Output directory for augmented dataset
    """
    # Load statistics
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    dataset_path = Path(stats['dataset_path'])
    classes = stats['classes']
    target_count = stats['max_count']

    print(f"\n{'='*60}")
    print(f"Balancing dataset: {stats['dataset_name']}")
    print(f"{'='*60}")
    print(f"Target count per class: {target_count}")
    print(f"Balance ratio before: {stats['balance_ratio']:.2f}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_augmented = 0

    # Process each class
    for class_name, current_count in sorted(classes.items()):
        print(f"Processing class: {class_name}")
        print(f"  Current: {current_count} images")

        # Find class directory
        class_dir = dataset_path / class_name

        if not class_dir.exists():
            print("  Warning: Directory not found, skipping")
            continue

        # Create output class directory
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # Get all images in class
        images = [f for f in class_dir.iterdir()
                  if f.is_file() and is_image(f.name)]

        if not images:
            print("  Warning: No images found, skipping")
            continue

        # Copy original images
        for img in images:
            shutil.copy2(img, output_class_dir / img.name)

        # Calculate how many augmented images we need
        needed = target_count - current_count

        if needed <= 0:
            print(f"  Already balanced, copied {current_count} images")
            continue

        # Calculate augmentations needed per image
        aug_per_image = needed // current_count
        remaining = needed % current_count

        print(f"  Need: {needed} more images")
        print(f"  Strategy: {aug_per_image} augmentations per image")

        augmented_count = 0

        # Apply augmentations
        for i, img_path in enumerate(images):
            # Determine number of augmentations for this image
            n_aug = aug_per_image + (1 if i < remaining else 0)

            if n_aug == 0:
                continue

            # Load image once
            image = cv.imread(str(img_path))
            if image is None:
                continue

            # Apply augmentations cyclically
            aug_funcs = list(AUGMENTATIONS.items())
            for j in range(n_aug):
                aug_name, aug_func = aug_funcs[j % len(aug_funcs)]

                # Apply augmentation
                augmented = aug_func(image.copy())

                # Save with unique name
                base_name = img_path.stem
                ext = img_path.suffix
                output_name = f"{base_name}_aug{i}_{aug_name}{ext}"
                output_file = output_class_dir / output_name

                cv.imwrite(str(output_file), augmented)
                augmented_count += 1

        total_augmented += augmented_count
        final_count = current_count + augmented_count
        print(f"  Created: {augmented_count} augmented images")
        print(f"  Final: {final_count} images\n")

    print(f"{'='*60}")
    print("✓ Dataset balanced successfully!")
    print(f"  Total augmented images: {total_augmented}")
    print(f"  Output directory: {output_path}")
    print(f"{'='*60}\n")


def main():
    """Main function."""
    # Check arguments
    if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print("Augmentation.py - Data Augmentation Tool")
        print("\nUsage:")
        print("  ./Augmentation.py <image_path>")
        print("  ./Augmentation.py --balance <stats.json> --output <dir>")
        print("\nModes:")
        print("  Single image mode:")
        print("    Applies 6 augmentations to one image")
        print("\n  Balance dataset mode:")
        print("    Balances entire dataset using Distribution stats")
        print("\nOptions:")
        print("  --balance FILE  JSON stats file from Distribution.py")
        print("  --output DIR Output directory (default: augmented_dataset)")
        print("\nExamples:")
        print("  ./Augmentation.py ./Apple/apple_healthy/image (1).JPG")
        print("  ./Augmentation.py --balance stats.json --output augmented")
        sys.exit(0)

    # Check for balance mode
    if '--balance' in sys.argv:
        idx = sys.argv.index('--balance')
        if idx + 1 >= len(sys.argv):
            print("Error: --balance requires a stats file")
            sys.exit(1)

        stats_file = Path(sys.argv[idx + 1])
        if not stats_file.exists():
            print(f"Error: Stats file '{stats_file}' not found")
            sys.exit(1)

        # Get output directory
        output_dir = 'augmented_dataset'
        if '--output' in sys.argv:
            out_idx = sys.argv.index('--output')
            if out_idx + 1 < len(sys.argv):
                output_dir = sys.argv[out_idx + 1]

        balance_dataset(stats_file, output_dir)
        return

    # Single image mode
    if len(sys.argv) != 2:
        print("Error: Invalid arguments. Use -h for help")
        sys.exit(1)

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
