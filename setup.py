# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# setup.py

from setuptools import setup, find_packages

# Leer el contenido del README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clustering-lib",
    version="0.1.0",
    author="Miguel Conde",
    author_email="miguelco2000@gmail.com",
    description="Una biblioteca de clustering en Python con múltiples algoritmos y herramientas de análisis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/miguel-conde/clustering-lib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "scikit-learn>=0.22.0",
        "matplotlib>=3.1.0",
        "ipywidgets>=7.5.1",
    ],
    python_requires='>=3.7',
)
