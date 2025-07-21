#!/usr/bin/env python3
"""
Transformation.py

Leaffliction - Computer Vision Project
Image transformation tool for feature extraction from leaf images.

This program implements 6 types of image transformations for analyzing
leaf characteristics: Gaussian Blur, Mask, ROI Objects, Analyze Object,
Pseudolandmarks, and Color Histogram.

Usage:
    python Transformation.py <image_path>            # Display transformations
    python Transformation.py -src <dir> -dst <dir>  # Process directory
    python Transformation.py -src <dir> -dst <dir> -mask  # Process with mask
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


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
        logging.debug(f"Saved transformed image: {output_path}")
    except Exception as e:
        logging.error(f"Error saving image {output_path}: {e}")
        raise


def transform_gaussian_blur(image: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur transformation for noise reduction."""
    # Apply Gaussian blur with kernel size 15x15
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return blurred


def transform_mask(image: np.ndarray) -> np.ndarray:
    """Create a mask to isolate the leaf from background."""
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for green colors (leaf)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    
    # Create mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result


def transform_roi_objects(image: np.ndarray) -> np.ndarray:
    """Identify and highlight regions of interest (ROI) objects."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    result = image.copy()
    
    # Draw bounding rectangles around significant objects
    min_area = 1000  # Minimum area to consider as ROI
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add area text
            cv2.putText(result, f'Area: {int(area)}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return result


def transform_analyze_object(image: np.ndarray) -> np.ndarray:
    """Analyze objects by detecting edges and features."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convert edges back to 3-channel for visualization
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine original image with edge detection
    result = cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)
    
    return result


def transform_pseudolandmarks(image: np.ndarray) -> np.ndarray:
    """Detect and mark pseudo-landmarks on the leaf."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect corners using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.01,
                                      minDistance=20, blockSize=3)
    
    result = image.copy()
    
    # Draw circles at corner points (pseudo-landmarks)
    if corners is not None:
        corners = corners.astype(np.int32)
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            x, y = int(x), int(y)
            cv2.circle(result, (x, y), 5, (255, 0, 0), -1)
            
            # Add landmark number
            cv2.putText(result, str(i+1), (x+10, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    return result


def transform_color_histogram(image: np.ndarray) -> np.ndarray:
    """Create a color histogram visualization."""
    # Calculate histograms for each color channel
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
    
    # Create a simple histogram visualization on the image
    h, w = image.shape[:2]
    hist_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize histograms
    hist_b = cv2.normalize(hist_b, hist_b, 0, h//2, cv2.NORM_MINMAX)
    hist_g = cv2.normalize(hist_g, hist_g, 0, h//2, cv2.NORM_MINMAX)
    hist_r = cv2.normalize(hist_r, hist_r, 0, h//2, cv2.NORM_MINMAX)
    
    # Draw histogram bars
    bin_w = int(w / 256)
    for i in range(256):
        pt1_b = (i * bin_w, h)
        pt2_b = (i * bin_w, h - int(hist_b[i][0]))
        cv2.line(hist_img, pt1_b, pt2_b, (255, 0, 0), 1)
        
        pt1_g = (i * bin_w, h)
        pt2_g = (i * bin_w, h - int(hist_g[i][0]))
        cv2.line(hist_img, pt1_g, pt2_g, (0, 255, 0), 1)
        
        pt1_r = (i * bin_w, h)
        pt2_r = (i * bin_w, h - int(hist_r[i][0]))
        cv2.line(hist_img, pt1_r, pt2_r, (0, 0, 255), 1)
    
    # Combine original image and histogram
    result = np.hstack((image, hist_img))
    
    return result


def apply_transformations(image_path: Path,
                          output_dir: Optional[Path] = None,
                          display_mode: bool = False) -> List[Path]:
    """Apply all 6 transformation techniques to an image."""
    if output_dir is None and not display_mode:
        output_dir = image_path.parent

    # Load original image
    image = load_image(image_path)

    # Define transformation functions
    transformations = {
        'GaussianBlur': transform_gaussian_blur,
        'Mask': transform_mask,
        'ROIObjects': transform_roi_objects,
        'AnalyzeObject': transform_analyze_object,
        'Pseudolandmarks': transform_pseudolandmarks,
        'ColorHistogram': transform_color_histogram
    }

    # Generate base filename without extension
    base_name = image_path.stem
    extension = image_path.suffix

    output_paths = []

    if display_mode:
        # Display transformations in a grid
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f'Image Transformations: {image_path.name}', fontsize=14)
        
        # Show original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Apply and display each transformation
        for idx, (trans_name, trans_func) in enumerate(
                transformations.items()):
            # Calculate subplot position
            row = (idx + 1) // 4
            col = (idx + 1) % 4
            
            try:
                logging.info(f"Applying {trans_name} transformation...")
                transformed_image = trans_func(image)
                
                # Display transformation
                if trans_name == 'ColorHistogram':
                    # Color histogram already includes the plot
                    img_rgb = cv2.cvtColor(transformed_image,
                                           cv2.COLOR_BGR2RGB)
                    axes[row, col].imshow(img_rgb)
                else:
                    img_rgb = cv2.cvtColor(transformed_image,
                                           cv2.COLOR_BGR2RGB)
                    axes[row, col].imshow(img_rgb)
                
                axes[row, col].set_title(trans_name)
                axes[row, col].axis('off')
                
            except Exception as e:
                msg = f"Failed to apply {trans_name} transformation: {e}"
                logging.error(msg)
                axes[row, col].text(0.5, 0.5, f'Error: {trans_name}',
                                    ha='center', va='center',
                                    transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        
        # Hide unused subplot
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Save transformations to files
        for trans_name, trans_func in transformations.items():
            try:
                logging.info(f"Applying {trans_name} transformation...")
                transformed_image = trans_func(image)

                # Create output filename
                output_filename = f"{base_name}_{trans_name}{extension}"
                output_path = output_dir / output_filename

                # Save transformed image
                save_image(transformed_image, output_path)
                output_paths.append(output_path)

            except Exception as e:
                msg = f"Failed to apply {trans_name} transformation: {e}"
                logging.error(msg)
                continue

    return output_paths


def process_single_image(image_path: Path, args: argparse.Namespace) -> None:
    """Process a single image with transformations."""
    logging.info(f"Processing image: {image_path}")

    if not image_path.exists():
        logging.error(f"Image file does not exist: {image_path}")
        return

    if not is_image_file(image_path.name):
        logging.error(f"File is not a supported image format: {image_path}")
        return

    # Determine if we're in display mode (no destination directory)
    display_mode = args.destination is None

    if display_mode:
        # Display transformations
        apply_transformations(image_path, display_mode=True)
    else:
        # Save transformations to destination directory
        output_dir = Path(args.destination)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = apply_transformations(image_path, output_dir)
        
        logging.info(f"Successfully created {len(output_paths)} "
                     f"transformed images:")
        for path in output_paths:
            print(f"  {path.name}")


def process_directory(src_dir: Path, dst_dir: Path,
                      args: argparse.Namespace) -> None:
    """Process all images in a source directory."""
    logging.info(f"Processing directory: {src_dir}")

    if not src_dir.exists():
        logging.error(f"Source directory does not exist: {src_dir}")
        return

    if not src_dir.is_dir():
        logging.error(f"Source path is not a directory: {src_dir}")
        return

    # Create destination directory
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = []
    for file_path in src_dir.rglob("*"):
        if file_path.is_file() and is_image_file(file_path.name):
            image_files.append(file_path)

    if not image_files:
        logging.warning(f"No image files found in: {src_dir}")
        return

    logging.info(f"Found {len(image_files)} image files")

    # Process each image
    for image_path in image_files:
        try:
            # Create subdirectory structure in destination
            relative_path = image_path.relative_to(src_dir)
            output_subdir = dst_dir / relative_path.parent
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Apply transformations
            apply_transformations(image_path, output_subdir)
            
        except Exception as e:
            logging.error(f"Failed to process {image_path}: {e}")
            continue

    logging.info("Directory processing completed")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="""
Leaffliction - Image Transformation Tool

Apply 6 types of image transformations for leaf feature analysis:
Gaussian Blur, Mask, ROI Objects, Analyze Object, Pseudolandmarks,
and Color Histogram.

This tool extracts characteristics from leaf images for disease
classification.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s ./Apple/apple_healthy/image (1).JPG        # Display transformations
  %(prog)s -src ./Apple/apple_healthy/ -dst ./output/ # Process directory
  %(prog)s -src ./images/ -dst ./output/ -mask        # Process with mask
  %(prog)s --help                                     # Show this help

Output files (when using -dst):
  image (1)_GaussianBlur.JPG
  image (1)_Mask.JPG
  image (1)_ROIObjects.JPG
  image (1)_AnalyzeObject.JPG
  image (1)_Pseudolandmarks.JPG
  image (1)_ColorHistogram.JPG
        """
    )

    # Positional argument for single image mode
    parser.add_argument(
        'image_path',
        nargs='?',
        help="Path to image file (displays transformations interactively)"
    )

    # Source directory argument
    parser.add_argument(
        '-src', '--source',
        metavar='DIR',
        help="Source directory containing images to process"
    )

    # Destination directory argument
    parser.add_argument(
        '-dst', '--destination',
        metavar='DIR',
        help="Destination directory for transformed images"
    )

    # Mask focus option
    parser.add_argument(
        '-mask',
        action='store_true',
        help="Focus on mask-based transformations"
    )

    # Verbose output
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output for debugging"
    )

    return parser


def main() -> None:
    """Main program function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.verbose)

    try:
        # Validate arguments
        if args.image_path and (args.source or args.destination):
            parser.error("Cannot use positional image_path with "
                         "-src/-dst options")
        
        if args.source and not args.destination:
            parser.error("-dst (destination) is required when using "
                         "-src (source)")
        
        if not args.image_path and not args.source:
            parser.error("Must specify either image_path or -src option")

        logging.info("Starting image transformation process...")

        if args.image_path:
            # Single image mode - display transformations
            input_path = Path(args.image_path).resolve()
            process_single_image(input_path, args)
        else:
            # Directory mode - save transformations
            src_path = Path(args.source).resolve()
            dst_path = Path(args.destination).resolve()
            process_directory(src_path, dst_path, args)

        logging.info("Image transformation completed successfully")

    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during transformation: {e}")
        if args.verbose:
            logging.exception("Error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
