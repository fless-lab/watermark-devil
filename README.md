# WatermarkEvil: Advanced Watermark Detection and Removal

A state-of-the-art tool for detecting and removing watermarks from images and videos using hybrid Rust/Python architecture and adaptive learning.

## 🚀 Features

- **Advanced Detection Engine**
  - Pattern-based detection using OpenCV
  - Frequency domain analysis with FFT
  - Neural detection with modified ResNet50
  - Support for multiple watermark types (logos, text, patterns)

- **Powerful Reconstruction Engine**
  - Inpainting with edge preservation
  - Diffusion-based reconstruction
  - Frequency domain reconstruction
  - Hybrid approach for optimal results

- **Adaptive Learning System**
  - Continuous improvement through usage
  - Pattern database with performance history
  - Automatic model retraining
  - Quality assessment and feedback integration

- **High-Performance Architecture**
  - Core processing in Rust for maximum speed
  - ML components in Python for flexibility
  - GPU acceleration support
  - Async processing for better throughput

## 🛠 Installation

### Prerequisites

- Rust 1.70+
- Python 3.9+
- CUDA 11.0+ (optional, for GPU support)
- OpenCV 4.5+

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/watermark-evil.git
cd watermark-evil
```

2. Install Rust dependencies:
```bash
cargo build --release
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp config/example.env .env
# Edit .env with your settings
```

5. Run the API:
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

The system can be configured through environment variables or a `.env` file:

- `WATERMARK_EVIL_MODE`: Development/production mode
- `GPU_ENABLED`: Enable/disable GPU acceleration
- `MODEL_PATH`: Path to ML models
- `API_KEY`: API authentication key
- `LOG_LEVEL`: Logging verbosity
- `MAX_BATCH_SIZE`: Maximum batch size for processing
- `DB_URL`: Database connection URL

See `config/example.env` for all available options.

## 📚 API Documentation

The API is available at `http://localhost:8000` with the following endpoints:

### Detection

```http
POST /detect
Content-Type: multipart/form-data

file: <image_file>
options: {
  "detect_multiple": true,
  "min_confidence": 0.5
}
```

### Reconstruction

```http
POST /reconstruct
Content-Type: multipart/form-data

file: <image_file>
detection_id: <detection_id>
options: {
  "method": "hybrid",
  "quality": "high"
}
```

### Complete Pipeline

```http
POST /process
Content-Type: multipart/form-data

file: <image_file>
options: {
  "detect_multiple": true,
  "method": "hybrid"
}
```

Full API documentation is available at `http://localhost:8000/docs`.

## 🔒 Security

- API authentication using API keys
- Rate limiting per client
- Input validation and sanitization
- Secure file handling
- Access control and logging

## 📊 Monitoring

The system provides comprehensive monitoring through:

- Prometheus metrics at `/metrics`
- Detailed logging with correlation IDs
- Performance tracking
- Error reporting
- Resource usage monitoring

## 🧪 Testing

Run the test suite:

```bash
# Run Rust tests
cargo test

# Run Python tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run API tests
pytest tests/api/
```

## 📈 Performance

Benchmarks on standard hardware (Intel i7, 32GB RAM, RTX 3080):

- Detection: ~100ms per image
- Reconstruction: ~200-500ms per image
- Batch processing: 10 images/second
- Memory usage: ~2GB RAM
- GPU memory: ~4GB VRAM

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- OpenCV team for computer vision components
- PyTorch team for deep learning framework
- Rust and Python communities

## 📞 Support

- GitHub Issues for bug reports and feature requests
- Documentation Wiki for guides and examples
- Community Discord for discussions
