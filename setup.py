#!/usr/bin/env python3
"""
Setup script para PTZ Tracker
Permite instalación local del paquete
"""

from setuptools import setup, find_packages

setup(
    name="ptz_tracker",
    version="1.0.0",
    description="Sistema PTZ Profesional con Seguimiento Automático",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.0.0",
        "requests>=2.25.0",
        "numpy>=1.20.0",
        "Pillow>=8.0.0"
    ],
    python_requires=">=3.8",
    author="PTZ Tracker Team",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
