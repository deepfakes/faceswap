#!/usr/bin python3
""" Collects and returns Information on available GPUs.

The information returned from this module provides information for both Nvidia and AMD GPUs.
However, the information available for Nvidia is far more thorough than what is available for
AMD, where we need to plug into plaidML to pull stats. The quality of this data will vary
depending on the OS' particular OpenCL implementation.
"""

import logging
import os
import platform

from lib.utils import get_backend

if platform.system() == 'Darwin':
    import pynvx  # pylint: disable=import-error
    IS_MACOS = True
else:
    import pynvml
    IS_MACOS = False

# Limited PlaidML/AMD Stats
try:
    from lib.plaidml_tools import PlaidMLStats as plaidlib  # pylint:disable=ungrouped-imports
except ImportError:
    plaidlib = None


class GPUStats():
    """ Holds information and statistics about the GPU(s) available on the currently
    running system.

    Parameters
    ----------
    log: bool, optional
        Whether the class should output information to the logger. There may be occasions where the
        logger has not yet been set up when this class is queried. Attempting to log in these
        instances will raise an error. If GPU stats are being queried prior to the logger being
        available then this parameter should be set to ``False``. Otherwise set to ``True``.
        Default: ``True``
    """
    def __init__(self, log=True):
        # Logger is held internally, as we don't want to log when obtaining system stats on crash
        self._logger = logging.getLogger(__name__) if log else None
        self._log("debug", "Initializing {}".format(self.__class__.__name__))

        self._plaid = None
        self._initialized = False
        self._device_count = 0
        self._active_devices = list()
        self._handles = list()
        self._driver = None
        self._devices = list()
        self._vram = None

        self._initialize(log)

        self._driver = self._get_driver()
        self._devices = self._get_devices()
        self._vram = self._get_vram()
        if not self._active_devices:
            self._log("warning", "No GPU detected. Switching to CPU mode")
            return

        self._shutdown()
        self._log("debug", "Initialized {}".format(self.__class__.__name__))

    @property
    def device_count(self):
        """int: The number of GPU devices discovered on the system. """
        return self._device_count

    @property
    def _is_plaidml(self):
        """ bool: ``True`` if the backend is plaidML otherwise ``False``. """
        return self._plaid is not None

    @property
    def sys_info(self):
        """ dict: GPU Stats that are required for system information logging.

        The dictionary contains the following data:

            **vram** (`list`): the total amount of VRAM in Megabytes for each GPU as pertaining to
            :attr:`_handles`

            **driver** (`str`): The GPU driver version that is installed on the OS

            **devices** (`list`): The device name of each GPU on the system as pertaining
            to :attr:`_handles`

            **devices_active** (`list`): The device name of each active GPU on the system as
            pertaining to :attr:`_handles`
        """
        return dict(vram=self._vram,
                    driver=self._driver,
                    devices=self._devices,
                    devices_active=self._active_devices)

    def _log(self, level, message):
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

    def _initialize(self, log=False):
        """ Initialize the library that will be returning stats for the system's GPU(s).
        For Nvidia (on Linux and Windows) the library is `pynvml`. For Nvidia (on macOS) the
        library is `pynvx`. For AMD `plaidML` is used.

        Parameters
        ----------
        log: bool, optional
            Whether the class should output information to the logger. There may be occasions where
            the logger has not yet been set up when this class is queried. Attempting to log in
            these instances will raise an error. If GPU stats are being queried prior to the
            logger being available then this parameter should be set to ``False``. Otherwise set
            to ``True``. Default: ``False``
        """
        if not self._initialized:
            if get_backend() == "amd":
                self._log("debug", "AMD Detected. Using plaidMLStats")
                loglevel = "INFO" if self._logger is None else self._logger.getEffectiveLevel()
                self._plaid = plaidlib(loglevel=loglevel, log=log)
            elif IS_MACOS:
                self._log("debug", "macOS Detected. Using pynvx")
                try:
                    pynvx.cudaInit()
                except RuntimeError:
                    self._initialized = True
                    return
            else:
                try:
                    self._log("debug", "OS is not macOS. Trying pynvml")
                    pynvml.nvmlInit()
                except (pynvml.NVMLError_LibraryNotFound,  # pylint: disable=no-member
                        pynvml.NVMLError_DriverNotLoaded,  # pylint: disable=no-member
                        pynvml.NVMLError_NoPermission) as err:  # pylint: disable=no-member
                    if plaidlib is not None:
                        self._log("debug", "pynvml errored. Trying plaidML")
                        self._plaid = plaidlib(log=log)
                    else:
                        msg = ("There was an error reading from the Nvidia Machine Learning "
                               "Library. Either you do not have an Nvidia GPU (in which case "
                               "this warning can be ignored) or the most likely cause is "
                               "incorrectly installed drivers. If this is the case, Please remove "
                               "and reinstall your Nvidia drivers before reporting."
                               "Original Error: {}".format(str(err)))
                        self._log("warning", msg)
                        self._initialized = True
                        return
                except Exception as err:  # pylint: disable=broad-except
                    msg = ("An unhandled exception occured loading pynvml. "
                           "Original error: {}".format(str(err)))
                    if self._logger:
                        self._logger.error(msg)
                    else:
                        print(msg)
                    self._initialized = True
                    return
            self._initialized = True
            self._get_device_count()
            self._get_active_devices()
            self._get_handles()

    def _shutdown(self):
        """ Shutdown pynvml if it was the library used for obtaining stats and set
        :attr:`_initialized` back to ``False``. """
        if self._initialized:
            self._handles = list()
            if not IS_MACOS and not self._is_plaidml:
                pynvml.nvmlShutdown()
            self._initialized = False

    def _get_device_count(self):
        """ Detect the number of GPUs attached to the system and allocate to
        :attr:`_device_count`. """
        if self._is_plaidml:
            self._device_count = self._plaid.device_count
        elif IS_MACOS:
            self._device_count = pynvx.cudaDeviceGetCount(ignore=True)
        else:
            try:
                self._device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                self._device_count = 0
        self._log("debug", "GPU Device count: {}".format(self._device_count))

    def _get_active_devices(self):
        """ Obtain the indices of active GPUs (those that have not been explicitly excluded by
        CUDA_VISIBLE_DEVICES or plaidML) and allocate to :attr:`_active_devices`. """
        if self._is_plaidml:
            self._active_devices = self._plaid.active_devices
        else:
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if self._device_count == 0:
                self._active_devices = list()
            elif devices is not None:
                self._active_devices = [int(i) for i in devices.split(",") if devices]
            else:
                self._active_devices = list(range(self._device_count))
            self._log("debug", "Active GPU Devices: {}".format(self._active_devices))

    def _get_handles(self):
        """ Obtain the internal handle identifiers for the system GPUs and allocate to
        :attr:`_handles`. """
        if self._is_plaidml:
            self._handles = self._plaid.devices
        elif IS_MACOS:
            self._handles = pynvx.cudaDeviceGetHandles(ignore=True)
        else:
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(self._device_count)]
        self._log("debug", "GPU Handles found: {}".format(len(self._handles)))

    def _get_driver(self):
        """ Obtain and return the installed driver version for the system's GPUs.

        Returns
        -------
        str
            The currently installed GPU driver version
        """
        if self._is_plaidml:
            driver = self._plaid.drivers
        elif IS_MACOS:
            driver = pynvx.cudaSystemGetDriverVersion(ignore=True)
        else:
            try:
                driver = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
            except pynvml.NVMLError:
                driver = "No Nvidia driver found"
        self._log("debug", "GPU Driver: {}".format(driver))
        return driver

    def _get_devices(self):
        """ Obtain the name of the installed devices. The quality of this information depends on
        the backend and OS being used, but it should be sufficient for identifying cards.

        Returns
        -------
        list
            List of device names for connected GPUs as corresponding to the values in
            :attr:`_handles`
        """
        self._initialize()
        if self._device_count == 0:
            names = list()
        if self._is_plaidml:
            names = self._plaid.names
        elif IS_MACOS:
            names = [pynvx.cudaGetName(handle, ignore=True)
                     for handle in self._handles]
        else:
            names = [pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                     for handle in self._handles]
        self._log("debug", "GPU Devices: {}".format(names))
        return names

    def _get_vram(self):
        """ Obtain the total VRAM in Megabytes for each connected GPU.

        Returns
        -------
        list
             List of floats containing the total amount of VRAM in Megabytes for each connected GPU
             as corresponding to the values in :attr:`_handles
        """
        self._initialize()
        if self._device_count == 0:
            vram = list()
        elif self._is_plaidml:
            vram = self._plaid.vram
        elif IS_MACOS:
            vram = [pynvx.cudaGetMemTotal(handle, ignore=True) / (1024 * 1024)
                    for handle in self._handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).total /
                    (1024 * 1024)
                    for handle in self._handles]
        self._log("debug", "GPU VRAM: {}".format(vram))
        return vram

    def _get_free_vram(self):
        """ Obtain the amount of VRAM that is available, in Megabytes, for each connected GPU.

        Returns
        -------
        list
             List of floats containing the amount of VRAM available, in Megabytes, for each
             connected GPU as corresponding to the values in :attr:`_handles

        Notes
        -----
        There is no useful way to get free VRAM on PlaidML. OpenCL loads and unloads VRAM as
        required, so this returns the total memory available per card for AMD cards, which us
        not particularly useful.

        """
        self._initialize()
        if self._is_plaidml:
            vram = self._plaid.vram
        elif IS_MACOS:
            vram = [pynvx.cudaGetMemFree(handle, ignore=True) / (1024 * 1024)
                    for handle in self._handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 * 1024)
                    for handle in self._handles]
        self._shutdown()
        self._log("debug", "GPU VRAM free: {}".format(vram))
        return vram

    def get_card_most_free(self):
        """ Obtain statistics for the GPU with the most available free VRAM.

        Returns
        -------
        dict
            The dictionary contains the following data:

                **card_id** (`int`):  The index of the card as pertaining to :attr:`_handles`

                **device** (`str`): The name of the device

                **free** (`float`): The amount of available VRAM on the GPU

                **total** (`float`): the total amount of VRAM on the GPU

            If a GPU is not detected then the **card_id** is returned as ``-1`` and the amount
            of free and total RAM available is fixed to 2048 Megabytes.
        """
        if self._device_count == 0:
            return {"card_id": -1,
                    "device": "No GPU devices found",
                    "free": 2048,
                    "total": 2048}
        free_vram = [self._get_free_vram()[i] for i in self._active_devices]
        vram_free = max(free_vram)
        card_id = self._active_devices[free_vram.index(vram_free)]
        retval = {"card_id": card_id,
                  "device": self._devices[card_id],
                  "free": vram_free,
                  "total": self._vram[card_id]}
        self._log("debug", "Active GPU Card with most free VRAM: {}".format(retval))
        return retval
