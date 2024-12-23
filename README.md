# Comic Splitter

An ML-powered tool for intelligently splitting long comic images into smaller WebP files with optimal quality and size.

## Features

- 🤖 Machine Learning-based split point detection
- 🖼️ Smart image preprocessing and optimization
- 📱 WebP output format for optimal quality/size ratio
- 🖥️ User-friendly GUI interface
- 📦 Batch processing support
- 🗄️ Automated file management
- 🎯 Training module for custom comic styles

## Requirements

- Python 3.10 or higher
- GPU recommended for ML training (but not required for usage)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/comic-splitter.git
cd comic-splitter
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

### GUI Application
```bash
comic-splitter
```

### Training Tool
```bash
comic-trainer
```

## Development Setup

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

## Project Structure

```
comic-splitter/
├── src/                 # Source code
├── models/             # Trained ML models
├── data/               # Training and temp data
├── tests/              # Unit tests
└── docs/               # Documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
