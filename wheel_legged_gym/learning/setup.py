from setuptools import find_packages
from distutils.core import setup

setup(
    name="learning",
    version="0.1.0",
    author="Wen Jiang",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="",
    description="Reinforcement learning algorithms",
    install_requires=[
        # "isaacgym",
        "torch",
        "matplotlib",
        # "tensorboard",
        "setuptools==59.5.0",
        "numpy",
        # "GitPython",
        # "onnx",
    ],
)
