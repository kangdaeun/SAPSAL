#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 16:53:43 2025

@author: daeun
"""

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="sapsal",
    version="0.1.0",
    author='Da Eun Kang',
    description='Star And Protoplanetary disk Spectroscopic data AnaLyzer with Neural Networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kangdaeun/SAPSAL",
    packages=find_packages(),  # sapsal/ 하위의 모든 패키지를 자동 탐색
    python_requires=">=3.8",
    install_requires=[
        # "numpy>=1.21",
        # "scipy>=1.7",
        # "matplotlib>=3.4",
        # "pandas>=1.3",
        # "torch>=1.12",  # 예시로 넣은 PyTorch
        # "scikit-learn>=1.0",
        # 필요한 만큼 추가
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    license_files=["LICENSE"],
    keywords=["astronomy", "deep learning", "stellar spectra", "cINN", "SAPSAL", "MUSE"],
    project_urls={
        "Source": "https://github.com/kangdaeun/SAPSAL",
        "Bug Tracker": "https://github.com/kangdaeun/SAPSAL/issues",
    },

    
)
