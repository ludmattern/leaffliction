#!/usr/bin/env python3
"""Transformation.py - Image Transformation Tool

Applies 6 types of image transformations for leaf feature extraction.

Usage:
  ./Transformation.py <image_path>
  ./Transformation.py -src <dir> -dst <dir>

Transformations:
  Gaussian Blur, Mask, ROI Objects, Analyze Object,
  Pseudolandmarks, Color Histogram

Examples:
  ./Transformation.py ./Apple/image.JPG
  ./Transformation.py -src ./Apple/ -dst ./output/
"""

import sys
import os
import tempfile
from pathlib import Path

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from plantcv import plantcv as pcv

# Disable PlantCV debug output by default
pcv.params.debug = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


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
        """Create binary mask to isolate leaf from background."""
        # Convert to HSV and extract saturation channel
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        # Apply binary threshold
        _, binary_mask = cv.threshold(saturation, 58, 255, cv.THRESH_BINARY)
        return binary_mask

    def gaussian_blur(self):
        """Gaussian Blur transformation."""
        blurred = pcv.gaussian_blur(
            img=self.mask, ksize=(7, 7), sigma_x=0, sigma_y=None
        )

        # Clean self.mask: fill holes and remove isolated pixels
        open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
        self.mask = cv.morphologyEx(self.mask, cv.MORPH_CLOSE, close_kernel)
        self.mask = cv.morphologyEx(self.mask, cv.MORPH_OPEN, open_kernel)

        return blurred

    def masked_leaf(self):
        """Mask transformation - isolate leaf."""
        # Apply binary mask to original image to keep colors
        result = cv.bitwise_and(self.img, self.img, mask=self.mask)
        # Set background to white
        result[self.mask == 0] = [255, 255, 255]

        # Remove dark pixels (0-50) from the mask for next transformations
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        dark_pixels = gray <= 30
        self.mask[dark_pixels] = 0

        # Fill the holes left by dark pixel removal
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        self.mask = cv.morphologyEx(self.mask, cv.MORPH_CLOSE, kernel)

        second_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        self.mask = cv.morphologyEx(self.mask, cv.MORPH_OPEN, second_kernel)

        return result

    def roi_contours(self):
        """ROI Objects transformation - draw contours."""
        # Find contours to get bounding rectangle
        contours, _ = cv.findContours(
            self.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )

        # Start with original image
        result = self.img.copy()

        if contours:
            # Fill the masked area in green
            result[self.mask > 0] = [0, 255, 0]

            # Get bounding rectangle for all contours combined
            all_points = np.vstack(contours)
            x, y, w, h = cv.boundingRect(all_points)

            # Draw single rectangle around entire leaf
            cv.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return result

    def analyze_shape(self):
        """Analyze shape using PlantCV."""
        # Try PlantCV v4+ method (analyze.size with labeled mask)
        shape_img = pcv.analyze.size(img=self.img,
                                     labeled_mask=self.mask, n_labels=1)
        return shape_img

    def _draw_pseudolandmarks(self, img, pseudolandmarks, color, radius: int):
        """Draw pseudolandmark circles on image using OpenCV."""
        if pseudolandmarks is None or len(pseudolandmarks) == 0:
            return img

        for plm in pseudolandmarks:
            if plm is None or len(plm) == 0:
                continue

            pt = np.squeeze(np.asarray(plm))
            if pt.size < 2:
                continue

            x = int(pt[0])  # column
            y = int(pt[1])  # row

            # OpenCV expects (x, y) = (col, row)
            cv.circle(img, (x, y), radius, color, thickness=-1)

        return img

    def pseudolandmarks(self):
        """Pseudolandmarks transformation."""
        result = self.img.copy()

        # Get pseudolandmarks from PlantCV (y-axis for horizontal variation)
        left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
            img=result, mask=self.mask, label="leaf"
        )

        # Draw landmark sets with different colors
        landmarks = [
            (left, (0, 0, 255)),  # Red - left edge
            (right, (255, 0, 255)),  # Magenta - right edge
            (center_h, (0, 255, 0)),  # Green - horizontal center
        ]
        for landmark_set, color in landmarks:
            result = self._draw_pseudolandmarks(result, landmark_set, color, 5)

        return result

    def color_histogram(self):
        """Color Histogram transformation."""
        # Convert to different colorspaces and combine channels
        # RGB channels
        rgb_b = self.img[:, :, 0]
        rgb_g = self.img[:, :, 1]
        rgb_r = self.img[:, :, 2]

        # HSV channels
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        # LAB channels
        lab = cv.cvtColor(self.img, cv.COLOR_BGR2LAB)
        lightness = lab[:, :, 0]
        green_magenta = lab[:, :, 1]  # a* channel
        blue_yellow = lab[:, :, 2]  # b* channel

        # Stack all 9 channels
        all_channels = np.dstack(
            [
                rgb_b,
                rgb_g,
                rgb_r,
                hue,
                saturation,
                value,
                lightness,
                green_magenta,
                blue_yellow,
            ]
        )

        # Generate histogram for all channels
        hist_chart, hist_data = pcv.visualize.histogram(
            img=all_channels, mask=self.mask, hist_data=True
        )

        # Update channel labels
        channel_names = [
            "blue",
            "green",
            "red",
            "hue",
            "saturation",
            "value",
            "lightness",
            "green-magenta",
            "blue-yellow",
        ]
        hist_data["color channel"] = hist_data["color channel"].replace(
            {str(i): name for i, name in enumerate(channel_names)}
        )

        # Recreate chart with proper labels
        hist_chart = (
            alt.Chart(hist_data)
            .mark_line(point=True)
            .encode(
                x="pixel intensity",
                y="proportion of pixels (%)",
                color="color channel",
                tooltip=["pixel intensity", "proportion of pixels (%)"],
            )
        )

        # Save Altair chart to temporary file and reload as image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        hist_chart.save(tmp_path)
        hist_img = cv.imread(tmp_path)
        os.unlink(tmp_path)

        return hist_img

    def get_all_transformations(self):
        """Get all transformation functions."""
        return {
            "GaussianBlur": self.gaussian_blur,
            "Mask": self.masked_leaf,
            "ROIObjects": self.roi_contours,
            "AnalyzeObject": self.analyze_shape,
            "Pseudolandmarks": self.pseudolandmarks,
            "ColorHistogram": self.color_histogram,
        }


def display_transformations(image_path):
    """Display all transformations in a grid."""
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)

    transformer = Transformation(image)
    transforms = transformer.get_all_transformations()

    # Create figure with GridSpec: 3 rows, 3 cols (layout: 3-3-1 vertical)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        3, 3, hspace=0.1, wspace=0.1, left=0.02,
        right=0.98, top=0.89, bottom=0.02
    )
    fig.suptitle(f"Image Transformations: {image_path.name}", fontsize=14)

    # Show original in top-left
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    # Define positions for transformations in 3-3-1 vertical layout
    # fig1(orig) fig4     ->  (0,0) (0,1)
    # fig2       fig5 fig7 ->  (1,0) (1,1) (1,2)
    # fig3       fig6     ->  (2,0) (2,1)
    transform_list = list(transforms.items())[:-1]  # All except ColorHistogram
    positions = [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]

    for idx, (name, func) in enumerate(transform_list):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])

        print(f"  Applying {name}...")
        result = func()
        ax.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
        ax.set_title(name)
        ax.axis("off")

    # ColorHistogram spans full height on right
    ax_hist = fig.add_subplot(gs[:, 2])
    print("  Applying ColorHistogram...")
    hist_result = transforms["ColorHistogram"]()
    ax_hist.imshow(cv.cvtColor(hist_result, cv.COLOR_BGR2RGB))
    ax_hist.set_title("ColorHistogram")
    ax_hist.axis("off")

    plt.show()


def save_transformations(image_path, output_dir, mask_only=False, silent=False):
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
            result = transformer.masked_leaf()
            output_name = f"{base_name}_Mask{extension}"
            output_path = output_dir / output_name
            cv.imwrite(str(output_path), result)
            output_paths.append(output_path)
            if not silent:
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
                if not silent:
                    print(f"  Created: {output_name}")
            except Exception as e:
                print(f"  ✗ {name} failed: {e}")

    return output_paths


def process_directory(src_dir, dst_dir, mask_only=False, silent=False):
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

        if not silent:
            print(f"\n{img_path.name}:")

        # Create subdirectory structure
        rel_path = img_path.relative_to(src_path)
        output_subdir = dst_path / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        save_transformations(img_path, output_subdir, mask_only, silent)

    print(f"\n✓ All transformations saved to: {dst_path}")


def main():
    """Main function."""
    try:
        # Silent mode
        silent = False
        if "-s" in sys.argv:
            silent = True
            sys.argv.remove("-s")

        # Show help
        if len(sys.argv) < 2 or sys.argv[1] in ["-h", "--help"]:
            print(__doc__)
            sys.exit(0)

        # Directory mode
        if "-src" in sys.argv:
            if "-dst" not in sys.argv:
                print("Error: -dst required when using -src")
                sys.exit(1)

            src_idx = sys.argv.index("-src")
            dst_idx = sys.argv.index("-dst")

            if src_idx + 1 >= len(sys.argv) or dst_idx + 1 >= len(sys.argv):
                print("Error: Missing directory path")
                sys.exit(1)

            src_dir = sys.argv[src_idx + 1]
            dst_dir = sys.argv[dst_idx + 1]
            mask_only = "-mask" in sys.argv

            process_directory(src_dir, dst_dir, mask_only, silent)
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

        display_transformations(image_path, silent)

    except KeyboardInterrupt:
        print("\n\n⚠ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
