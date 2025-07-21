# Leaffliction

Computer vision project for leaf disease classification using machine learning.

## Overview

Leaffliction provides automated dataset analysis and visualization for leaf disease datasets. The project automatically detects dataset structure and generates distribution charts to understand class balance.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic analysis:
```bash
python Distribution.py ./images
```

Save charts:
```bash
python Distribution.py ./images --save ./output
```

Batch processing:
```bash
python Distribution.py ./images --save ./output --no-display
```

## Features

- Automatic dataset structure detection (flat and hierarchical)
- Statistical analysis with class distribution
- Visual charts generation (pie charts, bar charts)
- Support for multiple image formats
- Command-line interface

## Dataset Structure

Supports both flat (`Disease/images.jpg`) and hierarchical (`Plant/Disease/images.jpg`) structures with automatic detection.

## License

Academic research project in computer vision for agricultural applications.