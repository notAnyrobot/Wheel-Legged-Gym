#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Submodule defining the wheel_legged_gym definitions."""

from .vec_env import VecEnv
from .dummy_env import DummyEnv

__all__ = ["VecEnv", "DummyEnv"]
