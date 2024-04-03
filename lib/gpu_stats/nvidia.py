#!/usr/bin/env python3
""" Collects and returns Information on available Nvidia GPUs. """
import os

import pynvml

from lib.utils import FaceswapError

from ._base import _GPUStats


class NvidiaStats(_GPUStats):
    """ Holds information and statistics about Nvidia GPU(s) available on the currently
    running system.

    Notes
    -----
    PyNVML is used for hooking in to Nvidia's Machine Learning Library and allows for pulling
    fairly extensive statistics for Nvidia GPUs

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
        """ Initialize PyNVML for Nvidia GPUs.

        If :attr:`_is_initialized` is ``True`` then this function just returns performing no
        action. Otherwise :attr:`is_initialized` is set to ``True`` after successfully
        initializing NVML.

        Raises
        ------
        FaceswapError
            If the NVML library could not be successfully loaded
        """
        if self._is_initialized:
            return
        try:
            self._log("debug", "Initializing PyNVML for Nvidia GPU.")
            pynvml.nvmlInit()
        except (pynvml.NVMLError_LibraryNotFound,  # pylint:disable=no-member
                pynvml.NVMLError_DriverNotLoaded,  # pylint:disable=no-member
                pynvml.NVMLError_NoPermission) as err:  # pylint:disable=no-member
            msg = ("There was an error reading from the Nvidia Machine Learning Library. The most "
                   "likely cause is incorrectly installed drivers. If this is the case, Please "
                   "remove and reinstall your Nvidia drivers before reporting. Original "
                   f"Error: {str(err)}")
            raise FaceswapError(msg) from err
        except Exception as err:  # pylint:disable=broad-except
            msg = ("An unhandled exception occured reading from the Nvidia Machine Learning "
                   f"Library. Original error: {str(err)}")
            raise FaceswapError(msg) from err
        super()._initialize()

    def _shutdown(self) -> None:
        """ Cleanly close access to NVML and set :attr:`_is_initialized` back to ``False``. """
        self._log("debug", "Shutting down NVML")
        pynvml.nvmlShutdown()
        super()._shutdown()

    def _get_device_count(self) -> int:
        """ Detect the number of GPUs attached to the system.

        Returns
        -------
        int
            The total number of GPUs connected to the PC
        """
        try:
            retval = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError as err:
            self._log("debug", "Error obtaining device count. Setting to 0. "
                               f"Original error: {str(err)}")
            retval = 0
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_active_devices(self) -> list[int]:
        """ Obtain the indices of active GPUs (those that have not been explicitly excluded by
        CUDA_VISIBLE_DEVICES environment variable or explicitly excluded in the command line
        arguments).

        Returns
        -------
        list
            The list of device indices that are available for Faceswap to use
        """
        devices = super()._get_active_devices()
        env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if env_devices:
            new_devices = [int(i) for i in env_devices.split(",")]
            devices = [idx for idx in devices if idx in new_devices]
        self._log("debug", f"Active GPU Devices: {devices}")
        return devices

    def _get_handles(self) -> list:
        """ Obtain the device handles for all connected Nvidia GPUs.

        Returns
        -------
        list
            The list of pointers for connected Nvidia GPUs
        """
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                   for i in range(self._device_count)]
        self._log("debug", f"GPU Handles found: {len(handles)}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the Nvidia driver version currently in use.

        Returns
        -------
        str
            The current GPU driver version
        """
        try:
            driver = pynvml.nvmlSystemGetDriverVersion()
        except pynvml.NVMLError as err:
            self._log("debug", f"Unable to obtain driver. Original error: {str(err)}")
            driver = "No Nvidia driver found"
        self._log("debug", f"GPU Driver: {driver}")
        return driver

    def _get_device_names(self) -> list[str]:
        """ Obtain the list of names of connected Nvidia GPUs as identified in :attr:`_handles`.

        Returns
        -------
        list
            The list of connected Nvidia GPU names
        """
        names = [pynvml.nvmlDeviceGetName(handle)
                 for handle in self._handles]
        self._log("debug", f"GPU Devices: {names}")
        return names

    def _get_vram(self) -> list[int]:
        """ Obtain the VRAM in Megabytes for each connected Nvidia GPU as identified in
        :attr:`_handles`.

        Returns
        -------
        list
            The VRAM in Megabytes for each connected Nvidia GPU
        """
        vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)
                for handle in self._handles]
        self._log("debug", f"GPU VRAM: {vram}")
        return vram

    def _get_free_vram(self) -> list[int]:
        """ Obtain the amount of VRAM that is available, in Megabytes, for each connected Nvidia
        GPU.

        Returns
        -------
        list
             List of `float`s containing the amount of VRAM available, in Megabytes, for each
             connected GPU as corresponding to the values in :attr:`_handles
        """
        is_initialized = self._is_initialized
        if not is_initialized:
            self._initialize()
            self._handles = self._get_handles()

        vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 * 1024)
                for handle in self._handles]
        if not is_initialized:
            self._shutdown()

        self._log("debug", f"GPU VRAM free: {vram}")
        return vram
