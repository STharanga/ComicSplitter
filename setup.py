import os
from setuptools import setup, find_packages

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Development dependencies
dev_requirements = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "black>=24.1.0",
    "isort>=5.13.0",
    "flake8>=7.0.0",
    "mypy>=1.8.0",
    "pre-commit>=3.6.0",
    "sphinx>=7.2.0",
    "sphinx-rtd-theme>=2.0.0",
]

setup(
    name="comic-splitter",
    version="0.1.0",
    author="Supun Tharanga",
    author_email="stharanga411@gmail.com",
    description="ML-powered tool for splitting long comic images into smaller WebP files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/comic-splitter",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/comic-splitter/issues",
        "Documentation": "https://github.com/yourusername/comic-splitter/docs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Framework :: Pytest",
        "Framework :: Sphinx",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": ["sphinx>=7.2.0", "sphinx-rtd-theme>=2.0.0"],
        "test": ["pytest>=8.0.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "comic-splitter=main.cli:main",
            "comic-trainer=training.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    platforms=["any"],
    keywords=[
        "comic",
        "image processing",
        "machine learning",
        "webp",
        "GUI",
        "image splitting",
        "deep learning",
    ],
    package_data={
        "comic_splitter": [
            "models/*.pkl",
            "models/*.json",
            "data/*.json",
        ],
    },
    data_files=[
        ("", ["LICENSE", "README.md"]),
    ],
)