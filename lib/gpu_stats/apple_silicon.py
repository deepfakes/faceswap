#!/usr/bin/env python3
""" Collects and returns Information on available Apple Silicon SoCs in Apple Macs. """
from typing import List

import lib.metal as metal

from lib.utils import FaceswapError

from ._base import GPUStats


class AppleSiliconStats(GPUStats):
    """ Holds information and statistics about Apple Silicon SoC(s) available on the currently
    running Apple system.

    Parameters
    ----------
    log: bool, optional
        Whether the class should output information to the logger. There may be occasions where the
        logger has not yet been set up when this class is queried. Attempting to log in these
        instances will raise an error. If GPU stats are being queried prior to the logger being
        available then this parameter should be set to ``False``. Otherwise set to ``True``.
        Default: ``True``
    """

    def _initialize(self) -> None:
        """ Initialize Metal for Apple Silicon SoC(s).

        If :attr:`_is_initialized` is ``True`` then this function just returns performing no
        action. Otherwise :attr:`is_initialized` is set to ``True`` after successfully
        initializing Metal.

        Raises
        ------
        FaceswapError
            If the Metal library could not be successfully loaded
        """
        if self._is_initialized:
            return
        self._log("debug", "Initializing Metal for Apple Silicon SoC.")
        try:
            metal.init()  # pylint:disable=no-member
        except RuntimeError as err:
            msg = ("An unhandled exception occured initializing the device via Metal"
                   f"Library. Original error: {str(err)}")
            raise FaceswapError(msg) from err
        super()._initialize()

    def _get_device_count(self) -> int:
        """ Detect the number of SoCs attached to the system.

        Returns
        -------
        int
            The total number of SoCs available
        """
        retval = metal.get_device_count()  # pylint:disable=no-member
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_handles(self) -> list:
        """ Obtain the device handles for all available Apple Silicon SoCs.

        Returns
        -------
        list
            The list of pointers for available Apple Silicon SoCs
        """
        handles = metal.get_handles()  # pylint:disable=no-member
        self._log("debug", f"GPU Handles found: {len(handles)}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the Apple Silicon driver version currently in use.

        Returns
        -------
        str
            The current SoC driver version
        """
        driver = metal.get_driver_version()  # pylint:disable=no-member
        self._log("debug", f"GPU Driver: {driver}")
        return driver

    def _get_device_names(self) -> List[str]:
        """ Obtain the list of names of available Apple Silicon SoC(s) as identified in :attr:`_handles`.

        Returns
        -------
        list
            The list of available Apple Silicon SoC names
        """
        names = metal.get_device_names()
        self._log("debug", f"GPU Devices: {names}")
        return names

    def _get_vram(self) -> List[float]:
        """ Obtain the VRAM in Megabytes for each available Apple Silicon SoC(s) as identified in
        :attr:`_handles`.

        Returns
        -------
        list
            The VRAM in Megabytes for each available Apple Silicon SoC
        """
        vram = [
            metal.get_memory_info(i) / (1024 * 1024)
            for i in range(self._device_count)]
        self._log("debug", f"GPU VRAM: {vram}")
        return vram

    def _get_free_vram(self) -> List[float]:
        """ Obtain the amount of VRAM that is available, in Megabytes, for each available Apple Silicon
        SoC.

        Returns
        -------
        list
             List of `float`s containing the amount of VRAM available, in Megabytes, for each
             available SoC as corresponding to the values in :attr:`_handles
        """
        vram = [
            metal.get_memory_info(i) / (1024 * 1024)
            for i in range(self._device_count)]
        self._log("debug", f"GPU VRAM free: {vram}")
        return vram
