#!/usr/bin/env python3
"""
Distribution.py - Leaffliction Part 1: Dataset Analysis

Analyzes leaf disease dataset and generates distribution charts.
Takes a directory as argument and displays pie + bar charts.

Usage:
    ./Distribution.py <directory>
    ./Distribution.py ./Apple
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Chart style
sns.set_palette("husl")


def is_image(filename):
    """Check if file is an image based on extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


def count_images(directory):
    """Count images recursively in all subdirectories.
    Handles any level of nesting and preserves path hierarchy.

    Args:
        directory: Path to root directory

    Returns:
        dict: {class_name: image_count}
    """
    counts = {}

    def scan_recursive(path, prefix=""):
        """Recursively scan directory for images."""
        if not path.is_dir():
            return

        # Count images in current directory
        images = [f for f in path.iterdir()
                  if f.is_file() and is_image(f.name)]

        if images:
            # This directory contains images - it's a class
            class_name = (prefix + path.name) if prefix else path.name
            counts[class_name] = len(images)
            print(f"  {class_name}: {len(images)} images")
            return  # Don't go deeper if we found images

        # No images here, scan subdirectories
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        if subdirs:
            # Build prefix from current path
            is_root = (path == directory)
            new_prefix = "" if is_root else (prefix + path.name + "_")
            for subdir in subdirs:
                scan_recursive(subdir, new_prefix)

    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        return counts

    # Start recursive scan
    scan_recursive(directory)

    return counts


def display_charts(counts, dataset_name):
    """Display pie chart and bar chart for the distribution.

    Args:
        counts: dict {class_name: image_count}
        dataset_name: name of the dataset
    """
    if not counts:
        print("No data to display")
        return

    # Sort by count (descending)
    sorted_data = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Distribution Analysis - {dataset_name}',
                 fontsize=16, fontweight='bold')

    # 1. Pie chart
    colors = sns.color_palette("husl", len(labels))
    ax1.pie(values, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Distribution by Class', fontsize=12, fontweight='bold')

    # 2. Bar chart
    bars = ax2.bar(labels, values, color=colors, edgecolor='black')
    ax2.set_title('Number of Images per Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)

    # Add count labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{value}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_statistics(counts, dataset_name):
    """Print dataset statistics.

    Args:
        counts: dict {class_name: image_count}
        dataset_name: name of the dataset
    """
    if not counts:
        return

    total = sum(counts.values())
    num_classes = len(counts)
    min_count = min(counts.values())
    max_count = max(counts.values())
    balance_ratio = min_count / max_count if max_count > 0 else 0

    print("\n" + "="*50)
    print(f"Dataset: {dataset_name}")
    print("="*50)
    print(f"Total images:      {total}")
    print(f"Number of classes: {num_classes}")
    print(f"Min/Max images:    {min_count}/{max_count}")
    print(f"Balance ratio:     {balance_ratio:.2f}")

    if balance_ratio < 0.5:
        print("\nImbalanced dataset - consider data augmentation")

    print("\nDistribution by class:")
    for class_name, count in sorted(counts.items()):
        percentage = (count / total) * 100
        print(f"  {class_name:30s} {count:5d} ({percentage:5.1f}%)")
    print("="*50 + "\n")


def main():
    """Main function - analyze dataset and display charts."""
    # Check arguments and show help
    if len(sys.argv) != 2 or sys.argv[1] in ['-h', '--help', 'help']:
        print("Distribution.py - Leaffliction Dataset Analysis")
        print("\nUsage:")
        print("  ./Distribution.py <directory>")
        print("\nDescription:")
        print("  Analyzes image distribution in a dataset and displays")
        print("  pie chart and bar chart for visualization.")
        print("\nSupported structures:")
        print("  - Flat:        dataset/Disease/images.jpg")
        print("  - Hierarchical: dataset/Plant/Disease/images.jpg")
        print("  - Any depth:   dataset/A/B/C/.../images.jpg")
        print("\nExamples:")
        print("  ./Distribution.py ./Apple")
        print("  ./Distribution.py ./leaves")
        print("  ./Distribution.py /path/to/dataset")
        sys.exit(0 if sys.argv[1] in ['-h', '--help', 'help'] else 1)

    # Get directory path
    directory = Path(sys.argv[1]).resolve()

    # Validate directory exists
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)

    # Get dataset name
    dataset_name = directory.name

    print(f"\nAnalyzing dataset: {dataset_name}")
    print(f"Directory: {directory}")
    print("\nScanning subdirectories...")

    # Count images in each class
    counts = count_images(directory)

    if not counts:
        print("\nNo images found in subdirectories")
        print("Expected structure: <directory>/<class>/<images>")
        sys.exit(1)

    # Display statistics
    print_statistics(counts, dataset_name)

    # Display charts
    print("Generating charts...")
    display_charts(counts, dataset_name)


if __name__ == "__main__":
    main()
