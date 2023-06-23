#!/usr/bin/env python3
""" Collects and returns Information on available Nvidia GPUs connected to Apple Macs. """
import pynvx

from lib.utils import FaceswapError

from ._base import _GPUStats


class NvidiaAppleStats(_GPUStats):
    """ Holds information and statistics about Nvidia GPU(s) available on the currently
    running Apple system.

    Notes
    -----
    PyNvx is used for hooking in to Nvidia's Machine Learning Library and allows for pulling fairly
    extensive statistics for Apple based Nvidia GPUs

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
        """ Initialize PyNvx for Nvidia GPUs on Apple.

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
        self._log("debug", "Initializing Pynvx for Apple Nvidia GPU.")
        try:
            pynvx.cudaInit()  # pylint:disable=no-member
        except RuntimeError as err:
            msg = ("An unhandled exception occured reading from the Nvidia Machine Learning "
                   f"Library. Original error: {str(err)}")
            raise FaceswapError(msg) from err
        super()._initialize()

    def _shutdown(self) -> None:
        """ Set :attr:`_is_initialized` back to ``False``. """
        self._log("debug", "Shutting down NVML")
        super()._shutdown()

    def _get_device_count(self) -> int:
        """ Detect the number of GPUs attached to the system.

        Returns
        -------
        int
            The total number of GPUs connected to the PC
        """
        retval = pynvx.cudaDeviceGetCount(ignore=True)  # pylint:disable=no-member
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_handles(self) -> list:
        """ Obtain the device handles for all Apple connected Nvidia GPUs.

        Returns
        -------
        list
            The list of pointers for connected Nvidia GPUs
        """
        handles = pynvx.cudaDeviceGetHandles(ignore=True)  # pylint:disable=no-member
        self._log("debug", f"GPU Handles found: {len(handles)}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the Nvidia driver version currently in use.

        Returns
        -------
        str
            The current GPU driver version
        """
        driver = pynvx.cudaSystemGetDriverVersion(ignore=True)  # pylint:disable=no-member
        self._log("debug", f"GPU Driver: {driver}")
        return driver

    def _get_device_names(self) -> list[str]:
        """ Obtain the list of names of connected Nvidia GPUs as identified in :attr:`_handles`.

        Returns
        -------
        list
            The list of connected Nvidia GPU names
        """
        names = [pynvx.cudaGetName(handle, ignore=True)  # pylint:disable=no-member
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
        vram = [
            pynvx.cudaGetMemTotal(handle, ignore=True) / (1024 * 1024)  # pylint:disable=no-member
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
        vram = [
            pynvx.cudaGetMemFree(handle, ignore=True) / (1024 * 1024)  # pylint:disable=no-member
            for handle in self._handles]
        self._log("debug", f"GPU VRAM free: {vram}")
        return vram
