# CLI Interface

Command-line interface for watermark detection.

## Rust Implementation (Default)

The main CLI implementation is in Rust for better performance and distribution.

### Usage

```bash
watermark-evil [OPTIONS] <IMAGE_PATH>
```

### Arguments
- `<IMAGE_PATH>`: Path to the image file to analyze

### Options
- `-d, --detectors <DETECTORS>`: Which detectors to use [default: all] [possible values: all, logo, text, pattern, transparency]
- `-f, --format <FORMAT>`: Output format [default: text] [possible values: text, json]
- `-o, --output <FILE>`: Output file path (optional)

### Examples

1. Basic usage (all detectors):
```bash
watermark-evil image.jpg
```

2. Use specific detectors:
```bash
watermark-evil image.jpg -d logo -d text
```

3. Output JSON format:
```bash
watermark-evil image.jpg -f json
```

4. Save results to file:
```bash
watermark-evil image.jpg -o results.txt
```

## Python Implementation (Alternative)

An alternative Python implementation is available in the `py` directory.

### Usage

```bash
python -m src.cli.py.main [OPTIONS] IMAGE_PATH
```

See the Python implementation's README in the `py` directory for more details.
