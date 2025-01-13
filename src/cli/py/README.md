# Python CLI Implementation

Alternative Python implementation of the watermark detection CLI.

## Usage

```bash
python -m src.cli.py.main [OPTIONS] IMAGE_PATH
```

### Arguments

- `IMAGE_PATH`: Path to the image file to analyze

### Options

- `--detectors DETECTORS [DETECTORS ...]`: Which detectors to use (default: all)
  - Choices: all, logo, text, pattern, transparency
  - Can specify multiple detectors
- `--format FORMAT`: Output format (default: text)
  - Choices: text, json
- `--output OUTPUT`: Output file path (optional, default: print to stdout)

### Examples

1. Basic usage (all detectors):
```bash
python -m src.cli.py.main path/to/image.jpg
```

2. Use specific detectors:
```bash
python -m src.cli.py.main path/to/image.jpg --detectors logo text
```

3. Output JSON format:
```bash
python -m src.cli.py.main path/to/image.jpg --format json
```

4. Save results to file:
```bash
python -m src.cli.py.main path/to/image.jpg --output results.txt
```

## Note

This is an alternative implementation in Python. For better performance and easier distribution, consider using the Rust implementation in the parent directory.
