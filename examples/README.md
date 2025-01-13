# WatermarkEvil Examples

This directory contains example code demonstrating how to use WatermarkEvil in different scenarios.

## Examples List

1. `cli_usage.rs` - Example of using the Rust CLI directly
   - Shows how to use different detectors
   - Demonstrates processing options and output formats
   - Error handling examples

2. `api_usage.py` - Example of using the HTTP API
   - Complete workflow from detection to removal
   - Shows how to handle multipart form data
   - Demonstrates async/await usage
   - Error handling and response processing

## Running the Examples

### Rust CLI Example

```bash
# Build and run the CLI example
cargo run --example cli_usage
```

### Python API Example

```bash
# Install required packages
pip install aiohttp

# Run the API example
python examples/api_usage.py
```

## Additional Notes

- Make sure to replace the API key in the examples with your actual key
- The API server should be running before testing the API examples
- Image paths in the examples should be updated to point to your test images
