#!/usr/bin/env python3
"""Transformation.py - Image Transformation Tool

Applies 6 types of image transformations for leaf feature extraction.

Usage:
  ./Transformation.py <image_path>
  ./Transformation.py -src <dir> -dst <dir> [-mask]

Transformations:
  Gaussian Blur, Mask, ROI Objects, Analyze Object,
  Pseudolandmarks, Color Histogram

Examples:
  ./Transformation.py ./Apple/image.JPG
  ./Transformation.py -src ./Apple/ -dst ./output/
"""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv

# Disable PlantCV debug output by default
pcv.params.debug = None

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


def is_image(filename):
    """Check if file is an image based on extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


class Transformation:
    """Image transformation class using PlantCV."""

    def __init__(self, image: np.ndarray):
        """Initialize with an image array."""
        self.img = image
        self.mask = self._create_mask()

    def _create_mask(self):
        """Create mask to isolate leaf from background."""
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        lower_bound = np.array([35, 40, 40])
        upper_bound = np.array([85, 255, 255])
        self.mask = cv.inRange(hsv, lower_bound, upper_bound)
        return self.mask

    def _grayscale(self, color_space, channel, thresh, img_type):
        """Convert to colorspace, extract channel and apply threshold."""
        # Convert to specified color space
        converted = cv.cvtColor(self.img, color_space)
        # Extract specified channel
        channel_img = converted[:, :, channel]
        # Apply threshold
        _, binary = cv.threshold(channel_img, thresh, 255, img_type)
        return binary

    def transform_mask(self):
        """Mask transformation - isolate leaf."""
        # Create binary mask using same method as Gaussian blur
        binary_mask = self._grayscale(cv.COLOR_BGR2HSV,
                                       channel=1,
                                       thresh=58,
                                       img_type=cv.THRESH_BINARY)
        
        # Apply binary mask to original image to keep colors
        result = cv.bitwise_and(self.img, self.img, mask=binary_mask)
        # Set background to white
        result[binary_mask == 0] = [255, 255, 255]
        return result

    def transform_gaussian_blur(self):
        """Gaussian Blur transformation."""
        # Get the mask (reuse from transform_mask logic)
        binary_mask = self._grayscale(cv.COLOR_BGR2HSV,
                                       channel=1,
                                       thresh=58,
                                       img_type=cv.THRESH_BINARY)
        
        # Verify mask is valid
        if binary_mask is None or binary_mask.size == 0:
            return np.zeros_like(self.img)
        
        # Apply Gaussian blur using PlantCV
        blurred = pcv.gaussian_blur(img=binary_mask, ksize=(7, 7),
                                    sigma_x=0, sigma_y=None)
        return blurred

    def transform_roi_objects(self):
        """ROI Objects transformation - draw contours."""
        # Use the same binary mask as mask and gaussian blur
        binary_mask = self._grayscale(cv.COLOR_BGR2HSV,
                                      channel=1,
                                      thresh=58,
                                      img_type=cv.THRESH_BINARY)
        
        # Find contours to get bounding rectangle
        contours, _ = cv.findContours(binary_mask, cv.RETR_EXTERNAL,
                                      cv.CHAIN_APPROX_SIMPLE)
        
        # Start with original image
        result = self.img.copy()
        
        if contours:
            # Fill the masked area in green
            result[binary_mask > 0] = [0, 255, 0]
            
            # Get bounding rectangle for all contours combined
            all_points = np.vstack(contours)
            x, y, w, h = cv.boundingRect(all_points)
            
            # Draw single rectangle around entire leaf
            cv.rectangle(result, (x, y), (x + w, y + h),
                         (255, 0, 0), 2)
        
        return result

    def transform_analyze_object(self):
        """Analyze Object transformation."""
        # Use the same binary mask as other transformations
        binary_mask = self._grayscale(cv.COLOR_BGR2HSV,
                                      channel=1,
                                      thresh=58,
                                      img_type=cv.THRESH_BINARY)

        # Try PlantCV v4+ method (analyze.size with labeled mask)
        shape_img = pcv.analyze.size(img=self.img,
                                        labeled_mask=binary_mask,
                                        n_labels=1)
        return shape_img

    def transform_pseudolandmarks(self):
        """Pseudolandmarks transformation - corner detection."""
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        # Mask gray image
        gray_masked = cv.bitwise_and(gray, gray, mask=self.mask)

        # Detect corners
        corners = cv.goodFeaturesToTrack(gray_masked, maxCorners=50,
                                         qualityLevel=0.01, minDistance=20)

        result = self.img.copy()

        # Draw landmarks
        if corners is not None:
            for i, corner in enumerate(corners):
                x, y = corner.ravel()
                cv.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv.putText(result, str(i + 1), (int(x) + 10, int(y) + 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return result

    def transform_color_histogram(self):
        """Color Histogram transformation."""
        # Use PlantCV analyze.color
        pcv.params.debug = None

        # Calculate color histogram for masked region
        masked = pcv.apply_mask(img=self.img, mask=self.mask,
                                mask_color='white')

        # Create histogram visualization
        hist_b = cv.calcHist([masked], [0], self.mask, [256], [0, 256])
        hist_g = cv.calcHist([masked], [1], self.mask, [256], [0, 256])
        hist_r = cv.calcHist([masked], [2], self.mask, [256], [0, 256])

        # Create histogram image
        h, w = self.img.shape[:2]
        hist_img = np.zeros((h, w, 3), dtype=np.uint8)
        hist_img.fill(255)  # White background

        # Normalize histograms
        hist_b = cv.normalize(hist_b, hist_b, 0, h - 50, cv.NORM_MINMAX)
        hist_g = cv.normalize(hist_g, hist_g, 0, h - 50, cv.NORM_MINMAX)
        hist_r = cv.normalize(hist_r, hist_r, 0, h - 50, cv.NORM_MINMAX)

        # Draw histogram
        bin_w = int(w / 256)
        for i in range(256):
            cv.line(hist_img, (i * bin_w, h),
                    (i * bin_w, h - int(hist_b[i][0])), (255, 0, 0), 1)
            cv.line(hist_img, (i * bin_w, h),
                    (i * bin_w, h - int(hist_g[i][0])), (0, 255, 0), 1)
            cv.line(hist_img, (i * bin_w, h),
                    (i * bin_w, h - int(hist_r[i][0])), (0, 0, 255), 1)

        # Combine original and histogram
        result = np.hstack((self.img, hist_img))

        return result

    def get_all_transformations(self):
        """Get all transformation functions."""
        return {
            'GaussianBlur': self.transform_gaussian_blur,
            'Mask': self.transform_mask,
            'ROIObjects': self.transform_roi_objects,
            'AnalyzeObject': self.transform_analyze_object,
            'Pseudolandmarks': self.transform_pseudolandmarks,
            'ColorHistogram': self.transform_color_histogram
        }


def display_transformations(image_path):
    """Display all transformations in a grid."""
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    transformer = Transformation(image)
    transforms = transformer.get_all_transformations()

    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Image Transformations: {image_path.name}', fontsize=14)

    # Show original
    axes[0, 0].imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # Apply and display transformations
    for idx, (name, func) in enumerate(transforms.items()):
        row = (idx + 1) // 4
        col = (idx + 1) % 4

        try:
            print(f"  Applying {name}...")
            result = func()

            axes[row, col].imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
            axes[row, col].set_title(name)
            axes[row, col].axis('off')
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            axes[row, col].text(0.5, 0.5, f'Error:\\n{name}',
                                ha='center', va='center',
                                transform=axes[row, col].transAxes)
            axes[row, col].axis('off')

    # Hide unused subplot
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()


def save_transformations(image_path, output_dir, mask_only=False):
    """Save all transformations to files."""
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return []

    transformer = Transformation(image)
    base_name = image_path.stem
    extension = image_path.suffix

    output_paths = []

    if mask_only:
        # Only save mask
        try:
            result = transformer.transform_mask()
            output_name = f"{base_name}_Mask{extension}"
            output_path = output_dir / output_name
            cv.imwrite(str(output_path), result)
            output_paths.append(output_path)
            print(f"  Created: {output_name}")
        except Exception as e:
            print(f"  ✗ Mask failed: {e}")
    else:
        # Save all transformations
        transforms = transformer.get_all_transformations()
        for name, func in transforms.items():
            try:
                result = func()
                output_name = f"{base_name}_{name}{extension}"
                output_path = output_dir / output_name
                cv.imwrite(str(output_path), result)
                output_paths.append(output_path)
                print(f"  Created: {output_name}")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

    return output_paths


def process_directory(src_dir, dst_dir, mask_only=False):
    """Process all images in source directory."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists() or not src_path.is_dir():
        print(f"Error: Invalid source directory: {src_dir}")
        sys.exit(1)

    dst_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    images = [f for f in src_path.rglob("*")
              if f.is_file() and is_image(f.name)]

    if not images:
        print(f"No images found in: {src_dir}")
        sys.exit(1)

    mode = "mask transformations" if mask_only else "all transformations"
    print(f"\nProcessing {len(images)} images ({mode})...")

    for img_path in images:
        print(f"\n{img_path.name}:")

        # Create subdirectory structure
        rel_path = img_path.relative_to(src_path)
        output_subdir = dst_path / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        save_transformations(img_path, output_subdir, mask_only)

    print(f"\n✓ All transformations saved to: {dst_path}")


def main():
    """Main function."""
    try:
        # Show help
        if len(sys.argv) < 2 or sys.argv[1] in ['-h', '--help']:
            print(__doc__)
            sys.exit(0)

        # Directory mode
        if '-src' in sys.argv:
            if '-dst' not in sys.argv:
                print("Error: -dst required when using -src")
                sys.exit(1)

            src_idx = sys.argv.index('-src')
            dst_idx = sys.argv.index('-dst')

            if src_idx + 1 >= len(sys.argv) or dst_idx + 1 >= len(sys.argv):
                print("Error: Missing directory path")
                sys.exit(1)

            src_dir = sys.argv[src_idx + 1]
            dst_dir = sys.argv[dst_idx + 1]
            mask_only = '-mask' in sys.argv

            process_directory(src_dir, dst_dir, mask_only)
            return

        # Single image mode
        image_path = Path(sys.argv[1]).resolve()

        if not image_path.exists() or not image_path.is_file():
            print(f"Error: File not found: {image_path}")
            sys.exit(1)

        if not is_image(image_path.name):
            print(f"Error: Not a valid image: {image_path}")
            sys.exit(1)

        print(f"\nProcessing: {image_path.name}")
        print("Applying 6 transformations...\n")

        display_transformations(image_path)

    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
