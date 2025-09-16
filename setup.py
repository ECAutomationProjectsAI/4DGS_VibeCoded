"""Setup script for 4D Gaussian Splatting package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gs4d",
    version="1.0.0",
    author="4DGS Team",
    description="4D Gaussian Splatting implementation with CUDA acceleration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ECAutomationProjectsAI/4DGS_VibeCoded",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy==1.24.3",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.5.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.5",
        "Pillow>=9.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.0",
        "pandas>=1.3.0",
        "h5py>=3.0.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "cuda": ["gsplat==0.1.11"],
    },
    entry_points={
        "console_scripts": [
            "gs4d-train=tools.train:main",
            "gs4d-render=tools.render:main",
            "gs4d-preprocess=tools.preprocess_video:main",
            "gs4d-convert=tools.convert:main",
            "gs4d-evaluate=tools.evaluate:main",
        ],
    },
)