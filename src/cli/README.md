# CLI Interface

Command-line interface for watermark detection.

## Usage

```bash
python -m src.cli.main image_path [--detectors DETECTORS [DETECTORS ...]] [--format FORMAT] [--output OUTPUT]
```

### Arguments

- `image_path`: Path to the image file to analyze
- `--detectors`: Which detectors to use (default: all)
  - Choices: all, logo, text, pattern, transparency
  - Can specify multiple detectors
- `--format`: Output format (default: text)
  - Choices: text, json
- `--output`: Output file path (optional, default: print to stdout)

### Examples

1. Basic usage (all detectors):
```bash
python -m src.cli.main path/to/image.jpg
```

2. Use specific detectors:
```bash
python -m src.cli.main path/to/image.jpg --detectors logo text
```

3. Output JSON format:
```bash
python -m src.cli.main path/to/image.jpg --format json
```

4. Save results to file:
```bash
python -m src.cli.main path/to/image.jpg --output results.txt
```
