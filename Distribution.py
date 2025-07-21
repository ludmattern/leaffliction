#!/usr/bin/env python3
"""
Distribution.py

Leaffliction - Computer Vision Project
Analyzes a leaf image dataset and generates class distribution charts.

This program is part of the Leaffliction project which aims to classify leaf
diseases using computer vision. It analyzes image distribution in a structured
dataset and generates visualizations for each plant type.

Usage:
    python Distribution.py <dataset_directory> [options]
"""

import os
import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration of supported image extensions
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Configuration of chart style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def setup_logging(verbose: bool = False) -> None:
    """Configure the logging system."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Disable matplotlib debug messages (findfont, etc.)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    return Path(filename).suffix.lower() in IMG_EXTENSIONS


def count_images_per_class(root_dir: Path) -> Dict[str, int]:
    """Count images in each subfolder (class) of a directory."""
    class_counts = defaultdict(int)

    try:
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                class_name = entry.name
                try:
                    img_count = sum(
                        1 for f in os.scandir(entry.path)
                        if f.is_file() and is_image_file(f.name)
                    )
                    class_counts[class_name] = img_count
                    logging.debug(
                        f"Class '{class_name}': {img_count} images found")
                except PermissionError:
                    logging.warning(
                        f"Permission denied to access {entry.path}")
                    continue
    except OSError as e:
        logging.error(f"Error accessing directory {root_dir}: {e}")
        raise

    return dict(class_counts)


def calculate_statistics(class_counts: Dict[str, int]) -> Dict[str, float]:
    """Calculate basic statistics on class distribution."""
    if not class_counts:
        return {}

    counts = list(class_counts.values())
    total_images = sum(counts)

    stats = {
        'total_images': total_images,
        'num_classes': len(class_counts),
        'min': min(counts),
        'max': max(counts),
        'balance_ratio': min(counts) / max(counts) if max(counts) > 0 else 0
    }

    return stats


def print_statistics(class_counts: Dict[str, int], dataset_name: str) -> None:
    """Display class distribution statistics."""
    stats = calculate_statistics(class_counts)

    if not stats:
        logging.warning(f"No statistics to display for {dataset_name}")
        return

    print(f"\n=== Statistics for {dataset_name} ===")
    print(f"Total images: {stats['total_images']:.0f}")
    print(f"Number of classes: {stats['num_classes']}")
    print(f"Min/Max: {stats['min']:.0f}/{stats['max']:.0f}")
    print(f"Balance ratio: {stats['balance_ratio']:.2f}")

    # Warning if dataset is imbalanced
    if stats['balance_ratio'] < 0.5:
        print("Imbalanced dataset detected - consider data augmentation")

    print("\nDetail by class:")
    for class_name, count in sorted(class_counts.items()):
        percentage = (count / stats['total_images']) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")


def create_enhanced_plots(class_counts: Dict[str,
                                             int],
                          dataset_name: str,
                          save_dir: Optional[Path] = None,
                          display: bool = True) -> None:
    """Create enhanced charts with visual statistics and multiple
    plot types."""
    if not class_counts:
        logging.warning(f"No data to display for {dataset_name}")
        return
    # Sort data by count (descending order) for better readability
    sorted_data = sorted(class_counts.items(),
                         key=lambda x: x[1],
                         reverse=True)
    labels = [item[0] for item in sorted_data]
    counts = [item[1] for item in sorted_data]
    stats = calculate_statistics(class_counts)

    # Color palette configuration
    n_colors = len(labels)
    colors = sns.color_palette("husl", n_colors)

    # Create figure with 2x3 layout
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        f'Distribution Analysis - {dataset_name}',
        fontsize=16,
        fontweight='bold')

    # 1. Pie chart
    ax1 = plt.subplot(1, 3, 1)
    ax1.pie(
        counts, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, textprops={'fontsize': 9}
    )
    ax1.set_title('Proportional Distribution', fontsize=12, fontweight='bold')

    # 2. Bar chart with annotations
    ax2 = plt.subplot(1, 3, 2)
    bars = ax2.bar(
        labels,
        counts,
        color=colors,
        edgecolor='black',
        linewidth=0.8)
    ax2.set_title('Number of Images per Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Images')
    ax2.tick_params(axis='x', rotation=45)

    # Annotations on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() +
            bar.get_width() /
            2.,
            height +
            max(counts) *
            0.01,
            f'{count}',
            ha='center',
            va='bottom',
            fontweight='bold')

    # 3. Information panel
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')

    if stats:
        info_text = f"""
Dataset Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━
Total: {stats['total_images']:.0f} images
Classes: {stats['num_classes']}
Minimum: {stats['min']:.0f}
Maximum: {stats['max']:.0f}
Balance: {stats['balance_ratio']:.2f}

{"Balanced dataset" if stats['balance_ratio'] > 0.7 else "Imbalanced dataset"}
        """
        ax3.text(
            0.1,
            0.9,
            info_text,
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightgray",
                alpha=0.8))

    plt.tight_layout()

    # Save if requested
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{dataset_name}_analysis.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        logging.info(f"Charts saved: {output_path}")

    # Display only if requested
    if display:
        plt.show()
    else:
        plt.close(fig)  # Close figure to free memory


def plot_charts(class_counts: Dict[str, int], dataset_name: str,
                save_dir: Optional[Path] = None, display: bool = True) -> None:
    """Simplified version maintaining compatibility with old code."""
    create_enhanced_plots(
        class_counts,
        dataset_name,
        save_dir,
        display=display)


def validate_dataset_structure(dataset_path: Path) -> Tuple[bool, List[str]]:
    """Validate dataset structure and return any issues found."""
    issues = []

    if not dataset_path.exists():
        return False, [f"Directory {dataset_path} does not exist"]

    if not dataset_path.is_dir():
        return False, [f"{dataset_path} is not a directory"]

    # Check if there are subdirectories (classes)
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]

    if not subdirs:
        # Check if there are images directly in root directory
        images_in_root = [f for f in dataset_path.iterdir()
                          if f.is_file() and is_image_file(f.name)]
        if images_in_root:
            issues.append(
                f"Images found in root ({len(images_in_root)} images)")
            return True, issues
        else:
            return False, ["No subdirectories or images found"]

    # Check each subdirectory
    for subdir in subdirs:
        images = [f for f in subdir.iterdir()
                  if f.is_file() and is_image_file(f.name)]
        if not images:
            issues.append(f"No images in directory '{subdir.name}'")
        elif len(images) < 5:
            issues.append(
                f"Few images in '{subdir.name}' ({len(images)} images)")

    return True, issues


def analyze_dataset(dataset_path: Path) -> Dict[str, int]:
    """Recursively analyze dataset and return all classes found with
    automatic structure detection."""
    all_classes = {}

    def scan_directory(path: Path, prefix: str = "") -> None:
        """Recursively scan a directory to find images."""
        if not path.is_dir():
            return

        # Count images directly in this directory
        images = [f for f in path.iterdir() if f.is_file()
                  and is_image_file(f.name)]

        if images:
            # This directory contains images, so it's a class
            class_name = prefix + path.name if prefix else path.name
            all_classes[class_name] = len(images)
            logging.debug(f"Class '{class_name}': {len(images)} images")
            return

        # No images here, search in subdirectories
        subdirs = [d for d in path.iterdir() if d.is_dir()]

        if subdirs:
            # If we're at root and there are subdirectories
            if not prefix:
                # Check if it's a hierarchical structure (Plant/Disease)
                has_nested = any(
                    any(d.is_dir() for d in subdir.iterdir())
                    for subdir in subdirs
                )

                if has_nested:
                    # Hierarchical structure: add Plant_ prefix
                    for subdir in subdirs:
                        scan_directory(subdir, f"{subdir.name}_")
                else:
                    # Flat structure: no prefix
                    for subdir in subdirs:
                        scan_directory(subdir)
            else:
                # Already in a substructure
                for subdir in subdirs:
                    scan_directory(subdir, prefix)

    scan_directory(dataset_path)
    return all_classes


def analyze_single_directory(
        dataset_path: Path,
        args: argparse.Namespace) -> None:
    """Analyze a directory and generate visualizations with automatic
    structure detection."""
    dataset_name = dataset_path.name
    logging.info(f"Analyzing dataset: {dataset_name}")

    # Basic validation
    if not dataset_path.exists():
        logging.error(f"Directory {dataset_path} does not exist")
        return

    if not dataset_path.is_dir():
        logging.error(f"{dataset_path} is not a directory")
        return

    # Automatic dataset analysis
    class_counts = analyze_dataset(dataset_path)

    if not class_counts:
        logging.warning("No images found in dataset")
        return

    # Display results
    total_images = sum(class_counts.values())
    num_classes = len(class_counts)

    logging.info(
        f"Dataset analyzed: {total_images} images distributed across "
        f"{num_classes} classes")
    logging.info(f"Classes found: {list(class_counts.keys())}")

    # Always display statistics
    print_statistics(class_counts, dataset_name)

    # Generate charts
    save_dir = Path(args.save) if args.save else None
    create_enhanced_plots(class_counts, dataset_name, save_dir,
                          display=not args.no_display)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="""
Leaffliction - Leaf Image Distribution Analysis

This program automatically analyzes image distribution in a dataset
and generates visualizations. It automatically detects the dataset structure.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s ./images                    # Automatic analysis
  %(prog)s ./images --save ./output    # With saving
  %(prog)s ./images --verbose          # With verbose output
  %(prog)s ./dataset --no-display      # No display (save only)

Supported structures:
  Flat structure (Disease/images.jpg):
    dataset/
    ├── Apple_healthy/
    ├── Apple_scab/
    └── Grape_healthy/

  Hierarchical structure (Plant/Disease/images.jpg):
    dataset/
    ├── Apple/
    │   ├── healthy/     → Apple_healthy
    │   ├── scab/        → Apple_scab
    │   └── rust/        → Apple_rust
    └── Grape/
        ├── healthy/     → Grape_healthy
        └── black_rot/   → Grape_black_rot
        """
    )

    parser.add_argument(
        'dataset_dir',
        help="Root directory of the dataset to analyze"
    )

    parser.add_argument(
        '--save', '-s',
        metavar='DIR',
        help="Directory where to save generated charts"
    )

    parser.add_argument(
        '--no-display',
        action='store_true',
        help="Do not display charts (useful for batch processing)"
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output for debugging"
    )

    return parser


def main() -> None:
    """Main program function - analyze leaf image dataset and generate
    distribution visualizations."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Configure logging
    setup_logging(args.verbose)

    try:
        dataset_path = Path(args.dataset_dir).resolve()

        # Input directory validation
        if not dataset_path.exists():
            raise FileNotFoundError(f"Directory {dataset_path} does not exist")

        if not dataset_path.is_dir():
            raise NotADirectoryError(f"{dataset_path} is not a directory")

        # Configure matplotlib if --no-display
        if args.no_display:
            plt.ioff()  # Non-interactive mode
            import matplotlib
            matplotlib.use('Agg')  # Backend without display

        logging.info(f"Starting dataset analysis: {dataset_path}")
        logging.info(f"Save: {'Enabled' if args.save else 'Disabled'}")

        # Dataset analysis
        analyze_single_directory(dataset_path, args)

        logging.info("Analysis completed successfully")

    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error during analysis: {e}")
        if args.verbose:
            logging.exception("Error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
