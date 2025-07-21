#!/usr/bin/env python3
"""
Augmentation.py

Leaffliction - Computer Vision Project
Data augmentation tool for balancing leaf disease datasets.

This program applies 6 types of biologically realistic data augmentation
to leaf images: Flip, Rotate, Brightness, Contrast, Crop, and Blur.

Usage:
    python Augmentation.py <image_path> [options]
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import random
from typing import Optional


def setup_logging(verbose: bool = False) -> None:
    """Configure the logging system."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return Path(filename).suffix.lower() in IMG_EXTENSIONS


def load_image(image_path: Path) -> np.ndarray:
    """Load an image using OpenCV."""
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        raise


def save_image(image: np.ndarray, output_path: Path) -> None:
    """Save an image using OpenCV."""
    try:
        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise ValueError(f"Could not save image: {output_path}")
        logging.debug(f"Saved augmented image: {output_path}")
    except Exception as e:
        logging.error(f"Error saving image {output_path}: {e}")
        raise


def augment_flip(image: np.ndarray) -> np.ndarray:
    """Apply horizontal flip augmentation."""
    return cv2.flip(image, 1)


def augment_rotate(image: np.ndarray,
                   angle: Optional[float] = None) -> np.ndarray:
    """Apply rotation augmentation."""
    if angle is None:
        angle = random.uniform(-30, 30)

    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                             borderMode=cv2.BORDER_REFLECT)
    return rotated


def augment_brightness(image: np.ndarray) -> np.ndarray:
    """Apply brightness adjustment augmentation."""
    # Random brightness factor between 0.7 and 1.3
    brightness_factor = random.uniform(0.7, 1.3)

    # Convert to float for calculations
    brightened = image.astype(np.float32) * brightness_factor

    # Clip values to valid range and convert back to uint8
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    return brightened


def augment_contrast(image: np.ndarray) -> np.ndarray:
    """Apply contrast adjustment augmentation."""
    # Random contrast factor between 0.7 and 1.4
    contrast_factor = random.uniform(0.7, 1.4)

    # Convert to float for calculations
    contrasted = image.astype(np.float32)

    # Apply contrast: new_pixel = (old_pixel - 128) * contrast + 128
    contrasted = (contrasted - 128) * contrast_factor + 128

    # Clip values to valid range and convert back to uint8
    contrasted = np.clip(contrasted, 0, 255).astype(np.uint8)
    return contrasted


def augment_crop(image: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
    """Apply random crop augmentation with resize back to original size."""
    height, width = image.shape[:2]

    # Calculate crop dimensions
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)

    # Random crop position
    start_y = random.randint(0, height - crop_height)
    start_x = random.randint(0, width - crop_width)

    # Crop the image
    cropped = image[start_y:start_y + crop_height,
                    start_x:start_x + crop_width]

    # Resize back to original dimensions
    resized = cv2.resize(cropped, (width, height))
    return resized


def augment_blur(image: np.ndarray) -> np.ndarray:
    """Apply slight blur augmentation to simulate camera conditions."""
    # Random kernel size (odd numbers only)
    kernel_size = random.choice([3, 5])

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred


def apply_augmentations(image_path: Path,
                        output_dir: Optional[Path] = None) -> List[Path]:
    """Apply all 6 augmentation techniques to an image."""
    if output_dir is None:
        output_dir = image_path.parent

    # Load original image
    image = load_image(image_path)

    # Define augmentation functions
    augmentations = {
        'Flip': augment_flip,
        'Rotate': augment_rotate,
        'Brightness': augment_brightness,
        'Contrast': augment_contrast,
        'Crop': augment_crop,
        'Blur': augment_blur
    }

    # Generate base filename without extension
    base_name = image_path.stem
    extension = image_path.suffix

    output_paths = []

    # Apply each augmentation
    for aug_name, aug_func in augmentations.items():
        try:
            logging.info(f"Applying {aug_name} augmentation...")
            augmented_image = aug_func(image)

            # Create output filename
            output_filename = f"{base_name}_{aug_name}{extension}"
            output_path = output_dir / output_filename

            # Save augmented image
            save_image(augmented_image, output_path)
            output_paths.append(output_path)

        except Exception as e:
            logging.error(f"Failed to apply {aug_name} augmentation: {e}")
            continue

    return output_paths


def process_single_image(image_path: Path, args: argparse.Namespace) -> None:
    """Process a single image with augmentations."""
    logging.info(f"Processing image: {image_path}")

    if not image_path.exists():
        logging.error(f"Image file does not exist: {image_path}")
        return

    if not is_image_file(image_path.name):
        logging.error(f"File is not a supported image format: {image_path}")
        return

    # Determine output directory
    output_dir = Path(args.output) if args.output else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Apply augmentations
    try:
        output_paths = apply_augmentations(image_path, output_dir)

        logging.info(f"Successfully created {len(output_paths)} "
                     f"augmented images:")
        for path in output_paths:
            print(f"  {path.name}")

    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")


def process_directory(directory_path: Path, args: argparse.Namespace) -> None:
    """Process all images in a directory."""
    logging.info(f"Processing directory: {directory_path}")

    if not directory_path.exists():
        logging.error(f"Directory does not exist: {directory_path}")
        return

    if not directory_path.is_dir():
        logging.error(f"Path is not a directory: {directory_path}")
        return

    # Find all image files
    image_files = []
    for file_path in directory_path.iterdir():
        if file_path.is_file() and is_image_file(file_path.name):
            image_files.append(file_path)

    if not image_files:
        logging.warning(f"No image files found in: {directory_path}")
        return

    logging.info(f"Found {len(image_files)} image files")

    # Process each image
    for image_path in image_files:
        try:
            process_single_image(image_path, args)
        except Exception as e:
            logging.error(f"Failed to process {image_path}: {e}")
            continue

    logging.info("Directory processing completed")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="""
Leaffliction - Data Augmentation Tool

Apply 6 types of biologically realistic data augmentation to leaf images:
Flip, Rotate, Brightness, Contrast, Crop, and Blur.

This tool helps balance datasets by generating augmented versions of images.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s ./Apple/apple_healthy/image (1).JPG
  %(prog)s ./Apple/apple_healthy/ --output ./augmented/
  %(prog)s ./images/leaf.jpg --verbose

Output files:
  image (1)_Flip.JPG
  image (1)_Rotate.JPG
  image (1)_Brightness.JPG
  image (1)_Contrast.JPG
  image (1)_Crop.JPG
  image (1)_Blur.JPG
        """
    )

    parser.add_argument(
        'input_path',
        help="Path to image file or directory containing images"
    )

    parser.add_argument(
        '--output', '-o',
        metavar='DIR',
        help="Output directory for augmented images (default: same as input)"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output for debugging"
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Leaffliction Data Augmentation Tool v1.0'
    )

    return parser


def main() -> None:
    """Main program function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.verbose)

    try:
        input_path = Path(args.input_path).resolve()

        logging.info("Starting data augmentation process...")

        # Process input path
        if input_path.is_file():
            process_single_image(input_path, args)
        elif input_path.is_dir():
            process_directory(input_path, args)
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        logging.info("Data augmentation completed successfully")

    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during augmentation: {e}")
        if args.verbose:
            logging.exception("Error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
