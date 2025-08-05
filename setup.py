"""
Carbon-Aware-Trainer: Intelligent ML training scheduler for carbon reduction.
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="carbon-aware-trainer",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.com",
    description="Drop-in scheduler that aligns ML training with carbon intensity forecasts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/carbon-aware-trainer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "pydantic>=1.8.0",
        "python-dateutil>=2.8.0",
        "pytz>=2021.1",
        "aiohttp>=3.8.0",
        "asyncio-mqtt>=0.11.0",
    ],
    extras_require={
        "pytorch": ["torch>=1.11.0", "torchvision>=0.12.0"],
        "lightning": ["pytorch-lightning>=1.6.0"],
        "tensorflow": ["tensorflow>=2.8.0"],
        "jax": ["jax>=0.3.0", "flax>=0.5.0"],
        "huggingface": ["transformers>=4.18.0", "datasets>=2.0.0"],
        "monitoring": ["mlflow>=1.26.0", "wandb>=0.12.0", "tensorboard>=2.8.0"],
        "cloud": ["boto3>=1.21.0", "google-cloud-storage>=2.3.0", "azure-storage-blob>=12.11.0"],
        "cluster": ["kubernetes>=23.3.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.3.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "all": [
            "torch>=1.11.0", "torchvision>=0.12.0",
            "pytorch-lightning>=1.6.0",
            "tensorflow>=2.8.0",
            "jax>=0.3.0", "flax>=0.5.0",
            "transformers>=4.18.0", "datasets>=2.0.0",
            "mlflow>=1.26.0", "wandb>=0.12.0", "tensorboard>=2.8.0",
            "boto3>=1.21.0", "google-cloud-storage>=2.3.0", "azure-storage-blob>=12.11.0",
            "kubernetes>=23.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "carbon-trainer=carbon_aware_trainer.cli:main",
        ],
    },
)