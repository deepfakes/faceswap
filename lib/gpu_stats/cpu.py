#!/usr/bin/env python3
""" Dummy functions for running faceswap on CPU. """


from typing import List

from ._base import GPUStats


class CPUStats(GPUStats):
    """ Holds information and statistics about the CPU on the currently running system.

    Notes
    -----
    The information held here is not  useful, but GPUStats is dynamically imported depending on the
    backend used, so we need to make sure this class is available for Faceswap run on the CPU
    Backend.

    The base :class:`GPUStats` handles the dummying in of information when no GPU is detected.

    Parameters
    ----------
    log: bool, optional
        Whether the class should output information to the logger. There may be occasions where the
        logger has not yet been set up when this class is queried. Attempting to log in these
        instances will raise an error. If GPU stats are being queried prior to the logger being
        available then this parameter should be set to ``False``. Otherwise set to ``True``.
        Default: ``True``
    """

    def _get_device_count(self) -> int:
        """ Detect the number of GPUs attached to the system. Always returns zero for CPU
        backends.

        Returns
        -------
        int
            The total number of GPUs connected to the PC
        """
        retval = 0
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_handles(self) -> list:
        """ Obtain the device handles for all connected GPUs.

        Returns
        -------
        list
            An empty list for CPU Backends
        """
        handles = []
        self._log("debug", f"GPU Handles found: {len(handles)}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the driver version currently in use.

        Returns
        -------
        str
            An empty string for CPU backends
        """
        driver = ""
        self._log("debug", f"GPU Driver: {driver}")
        return driver

    def _get_device_names(self) -> List[str]:
        """ Obtain the list of names of connected GPUs as identified in :attr:`_handles`.

        Returns
        -------
        list
            An empty list for CPU backends
        """
        names = []
        self._log("debug", f"GPU Devices: {names}")
        return names

    def _get_vram(self) -> List[float]:
        """ Obtain the RAM in Megabytes for the running system.

        Returns
        -------
        list
            An empty list for CPU backends
        """
        vram = []
        self._log("debug", f"GPU VRAM: {vram}")
        return vram

    def _get_free_vram(self) -> List[float]:
        """ Obtain the amount of RAM that is available, in Megabytes, for the running system.

        Returns
        -------
        list
             An empty list for CPU backends
        """
        vram = []
        self._log("debug", f"GPU VRAM free: {vram}")
        return vram
