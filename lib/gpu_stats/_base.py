#!/usr/bin/env python3
""" Parent class for obtaining Stats for various GPU/TPU backends. All GPU Stats should inherit
from the :class:`_GPUStats` class contained here. """

import logging

from dataclasses import dataclass

from lib.utils import get_backend

_EXCLUDE_DEVICES: list[int] = []


@dataclass
class GPUInfo():
    """Dataclass for storing information about the available GPUs on the system.

    Attributes:
    ----------
    vram: list[int]
        List of integers representing the total VRAM available on each GPU, in MB.
    vram_free: list[int]
        List of integers representing the free VRAM available on each GPU, in MB.
    driver: str
        String representing the driver version being used for the GPUs.
    devices: list[str]
        List of strings representing the names of each GPU device.
    devices_active: list[int]
        List of integers representing the indices of the active GPU devices.
    """
    vram: list[int]
    vram_free: list[int]
    driver: str
    devices: list[str]
    devices_active: list[int]


@dataclass
class BiggestGPUInfo():
    """ Dataclass for holding GPU Information about the card with most available VRAM.

    Attributes
    ----------
    card_id: int
        Integer representing the index of the GPU device.
    device: str
        The name of the device
    free: float
        The amount of available VRAM on the GPU
    total: float
        the total amount of VRAM on the GPU
    """
    card_id: int
    device: str
    free: float
    total: float


def set_exclude_devices(devices: list[int]) -> None:
    """ Add any explicitly selected GPU devices to the global list of devices to be excluded
    from use by Faceswap.

    Parameters
    ----------
    devices: list[int]
        list of GPU device indices to exclude

    Example
    -------
    >>> set_exclude_devices([0, 1]) # Exclude the first two GPU devices
    """
    logger = logging.getLogger(__name__)
    logger.debug("Excluding GPU indicies: %s", devices)
    if not devices:
        return
    _EXCLUDE_DEVICES.extend(devices)


class _GPUStats():
    """ Parent class for collecting GPU device information.

    Parameters:
    -----------
    log : bool, optional
        Flag indicating whether or not to log debug messages. Default: `True`.
    """

    def __init__(self, log: bool = True) -> None:
        # Logger is held internally, as we don't want to log when obtaining system stats on crash
        # or when querying the backend for command line options
        self._logger: logging.Logger | None = logging.getLogger(__name__) if log else None
        self._log("debug", f"Initializing {self.__class__.__name__}")

        self._is_initialized = False
        self._initialize()

        self._device_count: int = self._get_device_count()
        self._active_devices: list[int] = self._get_active_devices()
        self._handles: list = self._get_handles()
        self._driver: str = self._get_driver()
        self._device_names: list[str] = self._get_device_names()
        self._vram: list[int] = self._get_vram()
        self._vram_free: list[int] = self._get_free_vram()

        if get_backend() != "cpu" and not self._active_devices:
            self._log("warning", "No GPU detected")

        self._shutdown()
        self._log("debug", f"Initialized {self.__class__.__name__}")

    @property
    def device_count(self) -> int:
        """int: The number of GPU devices discovered on the system. """
        return self._device_count

    @property
    def cli_devices(self) -> list[str]:
        """ list[str]: Formatted index: name text string for each GPU """
        return [f"{idx}: {device}" for idx, device in enumerate(self._device_names)]

    @property
    def exclude_all_devices(self) -> bool:
        """ bool: ``True`` if all GPU devices have been explicitly disabled otherwise ``False`` """
        return all(idx in _EXCLUDE_DEVICES for idx in range(self._device_count))

    @property
    def sys_info(self) -> GPUInfo:
        """ :class:`GPUInfo`: The GPU Stats that are required for system information logging """
        return GPUInfo(vram=self._vram,
                       vram_free=self._get_free_vram(),
                       driver=self._driver,
                       devices=self._device_names,
                       devices_active=self._active_devices)

    def _log(self, level: str, message: str) -> None:
        """ If the class has been initialized with :attr:`log` as `True` then log the message
        otherwise skip logging.

        Parameters
        ----------
        level: str
            The log level to log at
        message: str
            The message to log
        """
        if self._logger is None:
            return
        logger = getattr(self._logger, level.lower())
        logger(message)

    def _initialize(self) -> None:
        """ Override to initialize the GPU device handles and any other necessary resources. """
        self._is_initialized = True

    def _shutdown(self) -> None:
        """ Override to shutdown the GPU device handles and any other necessary resources. """
        self._is_initialized = False

    def _get_device_count(self) -> int:
        """ Override to obtain the number of GPU devices

        Returns
        -------
        int
            The total number of GPUs connected to the PC
        """
        raise NotImplementedError()

    def _get_active_devices(self) -> list[int]:
        """ Obtain the indices of active GPUs (those that have not been explicitly excluded in
        the command line arguments).

        Notes
        -----
        Override for GPU specific checking

        Returns
        -------
        list
            The list of device indices that are available for Faceswap to use
        """
        devices = [idx for idx in range(self._device_count) if idx not in _EXCLUDE_DEVICES]
        self._log("debug", f"Active GPU Devices: {devices}")
        return devices

    def _get_handles(self) -> list:
        """ Override to obtain GPU specific device handles for all connected devices.

        Returns
        -------
        list
            The device handle for each connected GPU
        """
        raise NotImplementedError()

    def _get_driver(self) -> str:
        """ Override to obtain the GPU specific driver version.

        Returns
        -------
        str
            The GPU driver currently in use
        """
        raise NotImplementedError()

    def _get_device_names(self) -> list[str]:
        """ Override to obtain the names of all connected GPUs. The quality of this information
        depends on the backend and OS being used, but it should be sufficient for identifying
        cards.

        Returns
        -------
        list
            List of device names for connected GPUs as corresponding to the values in
            :attr:`_handles`
        """
        raise NotImplementedError()

    def _get_vram(self) -> list[int]:
        """ Override to obtain the total VRAM in Megabytes for each connected GPU.

        Returns
        -------
        list
             List of `float`s containing the total amount of VRAM in Megabytes for each
             connected GPU as corresponding to the values in :attr:`_handles`
        """
        raise NotImplementedError()

    def _get_free_vram(self) -> list[int]:
        """ Override to obtain the amount of VRAM that is available, in Megabytes, for each
        connected GPU.

        Returns
        -------
        list
            List of `float`s containing the amount of VRAM available, in Megabytes, for each
            connected GPU as corresponding to the values in :attr:`_handles
        """
        raise NotImplementedError()

    def get_card_most_free(self) -> BiggestGPUInfo:
        """ Obtain statistics for the GPU with the most available free VRAM.

        Returns
        -------
        :class:`BiggestGpuInfo`
            If a GPU is not detected then the **card_id** is returned as ``-1`` and the amount
            of free and total RAM available is fixed to 2048 Megabytes.
        """
        if len(self._active_devices) == 0:
            retval = BiggestGPUInfo(card_id=-1,
                                    device="No GPU devices found",
                                    free=2048,
                                    total=2048)
        else:
            free_vram = [self._vram_free[i] for i in self._active_devices]
            vram_free = max(free_vram)
            card_id = self._active_devices[free_vram.index(vram_free)]
            retval = BiggestGPUInfo(card_id=card_id,
                                    device=self._device_names[card_id],
                                    free=vram_free,
                                    total=self._vram[card_id])
        self._log("debug", f"Active GPU Card with most free VRAM: {retval}")
        return retval
