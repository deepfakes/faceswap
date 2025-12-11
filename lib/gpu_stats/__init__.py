#!/usr/bin/env python3
""" Dynamically import the correct GPU Stats library based on the faceswap backend and the machine
being used. """

from lib.utils import get_backend

from ._base import GPUInfo, _GPUStats

backend = get_backend()

GPUStats: type[_GPUStats] | None
try:
    if backend == "nvidia":
        from .nvidia import NvidiaStats as GPUStats
    elif backend == "apple_silicon":
        from .apple_silicon import AppleSiliconStats as GPUStats
    elif backend == "rocm":
        from .rocm import ROCm as GPUStats
    else:
        from .cpu import CPUStats as GPUStats
except (ImportError, ModuleNotFoundError):
    GPUStats = None
