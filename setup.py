"""Package setup for NLP Financial Signals Pipeline."""

from pathlib import Path
from setuptools import setup, find_packages

requirements = Path("requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="nlp-financial-signals",
    version="1.0.0",
    description=(
        "Unsupervised NLP pipeline for extracting financial signals from "
        "central bank communications using BERTopic and FinBERT"
    ),
    author="Vishal Joshi",
    author_email="vishal.joshi@warwick.ac.uk",
    url="https://github.com/VishalKJ-ai/nlp-financial-signals",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "nlp-signals=src.pipeline:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
