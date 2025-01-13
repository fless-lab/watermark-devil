# Setup

This directory contains installation and setup scripts for the watermark detection system.

## Files

- `install.py`: Main installation script that:
  - Creates virtual environment
  - Installs Python dependencies
  - Downloads required ML models
  - Builds Rust components
  - Creates necessary directories

## Usage

Run from the project root:

```bash
python -m setup.install
```

This will set up all necessary components for the watermark detection system.
