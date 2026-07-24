"""
Setup file for CADRE package.
Allows `import src.models.vla_backbone` etc. from anywhere.
"""

from setuptools import setup, find_packages

setup(
    name="cadre",
    version="1.0.0",
    description="Continual Adaptation for Driving with Robust Evolution",
    author="CADRE Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # NOTE: torch>=2.1.0 must be installed MANUALLY with correct CUDA version BEFORE this.
        # See README.md for PyTorch installation instructions.
        "transformers>=4.40.0",
        "peft>=0.11.0",
        "accelerate>=0.30.0",
        "nuscenes-devkit>=1.1.11",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
    ],
)
