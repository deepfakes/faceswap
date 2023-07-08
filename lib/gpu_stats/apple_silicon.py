#!/usr/bin/env python3
""" Collects and returns Information on available Apple Silicon SoCs in Apple Macs. """
import typing as T

import os
import psutil
import tensorflow as tf

from lib.utils import FaceswapError

from ._base import _GPUStats


_METAL_INITIALIZED: bool = False


class AppleSiliconStats(_GPUStats):
    """ Holds information and statistics about Apple Silicon SoC(s) available on the currently
    running Apple system.

    Notes
    -----
    Apple Silicon is a bit different from other backends, as it does not have a dedicated GPU with
    it's own dedicated VRAM, rather the RAM is shared with the CPU and GPU. A combination of psutil
    and Tensorflow are used to pull as much useful information as possible.

    Parameters
    ----------
    log: bool, optional
        Whether the class should output information to the logger. There may be occasions where the
        logger has not yet been set up when this class is queried. Attempting to log in these
        instances will raise an error. If GPU stats are being queried prior to the logger being
        available then this parameter should be set to ``False``. Otherwise set to ``True``.
        Default: ``True``
    """
    def __init__(self, log: bool = True) -> None:
        # Following attribute set in :func:``_initialize``
        self._tf_devices: list[T.Any] = []

        super().__init__(log=log)

    def _initialize(self) -> None:
        """ Initialize Metal for Apple Silicon SoC(s).

        If :attr:`_is_initialized` is ``True`` then this function just returns performing no
        action. Otherwise :attr:`is_initialized` is set to ``True`` after successfully
        initializing Metal.
        """
        if self._is_initialized:
            return
        self._log("debug", "Initializing Metal for Apple Silicon SoC.")
        self._initialize_metal()

        self._tf_devices = tf.config.list_physical_devices(device_type="GPU")

        super()._initialize()

    def _initialize_metal(self) -> None:
        """ Initialize Metal on first call to this class and set global
        :attr:``_METAL_INITIALIZED`` to ``True``. If Metal has already been initialized then return
        performing no action.
        """
        global _METAL_INITIALIZED  # pylint:disable=global-statement

        if _METAL_INITIALIZED:
            return

        self._log("debug", "Performing first time Apple SoC setup.")

        os.environ["DISPLAY"] = ":0"

        try:
            os.system("open -a XQuartz")
        except Exception as err:  # pylint:disable=broad-except
            self._log("debug", f"Swallowing error opening XQuartz: {str(err)}")

        self._test_tensorflow()

        _METAL_INITIALIZED = True

    def _test_tensorflow(self) -> None:
        """ Test that tensorflow can execute correctly.

        Raises
        ------
        FaceswapError
            If the Tensorflow library could not be successfully initialized
        """
        try:
            meminfo = tf.config.experimental.get_memory_info('GPU:0')
            devices = tf.config.list_logical_devices()
            self._log("debug",
                      f"Tensorflow initialization test: (mem_info: {meminfo}, devices: {devices}")
        except RuntimeError as err:
            msg = ("An unhandled exception occured initializing the device via Tensorflow "
                   f"Library. Original error: {str(err)}")
            raise FaceswapError(msg) from err

    def _get_device_count(self) -> int:
        """ Detect the number of SoCs attached to the system.

        Returns
        -------
        int
            The total number of SoCs available
        """
        retval = len(self._tf_devices)
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_handles(self) -> list:
        """ Obtain the device handles for all available Apple Silicon SoCs.

        Notes
        -----
        Apple SoC does not use handles, so return a list of indices corresponding to found
        GPU devices

        Returns
        -------
        list
            The list of indices for available Apple Silicon SoCs
        """
        handles = list(range(self._device_count))
        self._log("debug", f"GPU Handles found: {handles}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the Apple Silicon driver version currently in use.

        Notes
        -----
        As the SoC is not a discreet GPU it does not technically have a driver version, so just
        return `'Not Applicable'` as a string

        Returns
        -------
        str
            The current SoC driver version
        """
        driver = "Not Applicable"
        self._log("debug", f"GPU Driver: {driver}")
        return driver

    def _get_device_names(self) -> list[str]:
        """ Obtain the list of names of available Apple Silicon SoC(s) as identified in
        :attr:`_handles`.

        Returns
        -------
        list
            The list of available Apple Silicon SoC names
        """
        names = [d.name for d in self._tf_devices]
        self._log("debug", f"GPU Devices: {names}")
        return names

    def _get_vram(self) -> list[int]:
        """ Obtain the VRAM in Megabytes for each available Apple Silicon SoC(s) as identified in
        :attr:`_handles`.

        Notes
        -----
        `tf.config.experimental.get_memory_info('GPU:0')` does not work, so uses psutil instead.
        The total memory on the system is returned as it is shared between the CPU and the GPU.
        There is no dedicated VRAM.

        Returns
        -------
        list
            The RAM in Megabytes for each available Apple Silicon SoC
        """
        vram = [int((psutil.virtual_memory().total / self._device_count) / (1024 * 1024))
                for _ in range(self._device_count)]
        self._log("debug", f"SoC RAM: {vram}")
        return vram

    def _get_free_vram(self) -> list[int]:
        """ Obtain the amount of VRAM that is available, in Megabytes, for each available Apple
        Silicon SoC.

        Returns
        -------
        list
             List of `float`s containing the amount of RAM available, in Megabytes, for each
             available SoC as corresponding to the values in :attr:`_handles
        """
        vram = [int((psutil.virtual_memory().available / self._device_count) / (1024 * 1024))
                for _ in range(self._device_count)]
        self._log("debug", f"SoC RAM free: {vram}")
        return vram
