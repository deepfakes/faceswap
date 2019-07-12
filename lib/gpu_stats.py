#!/usr/bin python3
""" Information on available Nvidia GPUs """

import logging
import os
import platform

from lib.utils import keras_backend_quiet

K = keras_backend_quiet()

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
    """ Holds information about system GPU(s) """
    def __init__(self, log=True):
        self.logger = None
        if log:
            # Logger is held internally, as we don't want to log
            # when obtaining system stats on crash
            self.logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
            self.logger.debug("Initializing %s", self.__class__.__name__)

        self.plaid = None
        self.initialized = False
        self.device_count = 0
        self.active_devices = list()
        self.handles = None
        self.driver = None
        self.devices = None
        self.vram = None

        self.initialize(log)

        self.driver = self.get_driver()
        self.devices = self.get_devices()
        self.vram = self.get_vram()
        if not self.active_devices:
            if self.logger:
                self.logger.warning("No GPU detected. Switching to CPU mode")
            return

        self.shutdown()
        if self.logger:
            self.logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_plaidml(self):
        """ Return whether running on plaidML backend """
        return self.plaid is not None

    def initialize(self, log=False):
        """ Initialize pynvml """
        if not self.initialized:
            if K.backend() == "plaidml.keras.backend":
                loglevel = "INFO"
                if self.logger:
                    self.logger.debug("plaidML Detected. Using plaidMLStats")
                    loglevel = self.logger.getEffectiveLevel()
                self.plaid = plaidlib(loglevel=loglevel, log=log)
            elif IS_MACOS:
                if self.logger:
                    self.logger.debug("macOS Detected. Using pynvx")
                try:
                    pynvx.cudaInit()
                except RuntimeError:
                    self.initialized = True
                    return
            else:
                try:
                    if self.logger:
                        self.logger.debug("OS is not macOS. Using pynvml")
                    pynvml.nvmlInit()
                except (pynvml.NVMLError_LibraryNotFound,  # pylint: disable=no-member
                        pynvml.NVMLError_DriverNotLoaded,  # pylint: disable=no-member
                        pynvml.NVMLError_NoPermission) as err:  # pylint: disable=no-member
                    if plaidlib is not None:
                        self.plaid = plaidlib(log=log)
                    else:
                        msg = ("There was an error reading from the Nvidia Machine Learning "
                               "Library. The most likely cause is incorrectly installed drivers. "
                               "Please remove and reinstall your Nvidia drivers before reporting."
                               "Original Error: {}".format(str(err)))
                        if self.logger:
                            self.logger.error(msg)
                        raise ValueError(msg)
            self.initialized = True
            self.get_device_count()
            self.get_active_devices()
            self.get_handles()

    def shutdown(self):
        """ Shutdown pynvml """
        if self.initialized:
            self.handles = None
            if not IS_MACOS and not self.plaid:
                pynvml.nvmlShutdown()
            self.initialized = False

    def get_device_count(self):
        """ Return count of Nvidia devices """
        if self.plaid is not None:
            self.device_count = self.plaid.device_count
        elif IS_MACOS:
            self.device_count = pynvx.cudaDeviceGetCount(ignore=True)
        else:
            try:
                self.device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                self.device_count = 0
        if self.logger:
            self.logger.debug("GPU Device count: %s", self.device_count)

    def get_active_devices(self):
        """ Return list of active Nvidia devices """
        if self.plaid is not None:
            self.active_devices = self.plaid.active_devices
        else:
            devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if self.device_count == 0:
                self.active_devices = list()
            elif devices is not None:
                self.active_devices = [int(i) for i in devices.split(",") if devices]
            else:
                self.active_devices = list(range(self.device_count))
            if self.logger:
                self.logger.debug("Active GPU Devices: %s", self.active_devices)

    def get_handles(self):
        """ Return all listed Nvidia handles """
        if self.plaid is not None:
            self.handles = self.plaid.devices
        elif IS_MACOS:
            self.handles = pynvx.cudaDeviceGetHandles(ignore=True)
        else:
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                            for i in range(self.device_count)]
        if self.logger:
            self.logger.debug("GPU Handles found: %s", len(self.handles))

    def get_driver(self):
        """ Get the driver version """
        if self.plaid is not None:
            driver = self.plaid.drivers
        elif IS_MACOS:
            driver = pynvx.cudaSystemGetDriverVersion(ignore=True)
        else:
            try:
                driver = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
            except pynvml.NVMLError:
                driver = "No Nvidia driver found"
        if self.logger:
            self.logger.debug("GPU Driver: %s", driver)
        return driver

    def get_devices(self):
        """ Return name of devices """
        self.initialize()
        if self.device_count == 0:
            names = list()
        if self.plaid is not None:
            names = self.plaid.names
        elif IS_MACOS:
            names = [pynvx.cudaGetName(handle, ignore=True)
                     for handle in self.handles]
        else:
            names = [pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                     for handle in self.handles]
        if self.logger:
            self.logger.debug("GPU Devices: %s", names)
        return names

    def get_vram(self):
        """ Return total vram in megabytes per device """
        self.initialize()
        if self.device_count == 0:
            vram = list()
        elif self.plaid:
            vram = self.plaid.vram
        elif IS_MACOS:
            vram = [pynvx.cudaGetMemTotal(handle, ignore=True) / (1024 * 1024)
                    for handle in self.handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).total /
                    (1024 * 1024)
                    for handle in self.handles]
        if self.logger:
            self.logger.debug("GPU VRAM: %s", vram)
        return vram

    def get_used(self):
        """ Return the vram in use """
        self.initialize()
        if self.plaid:
            # NB There is no useful way to get allocated VRAM on PlaidML.
            # OpenCL loads and unloads VRAM as required, so this returns 0
            # It's not particularly useful
            vram = [0 for idx in range(self.device_count)]

        elif IS_MACOS:
            vram = [pynvx.cudaGetMemUsed(handle, ignore=True) / (1024 * 1024)
                    for handle in self.handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
                    for handle in self.handles]
        self.shutdown()

        if self.logger:
            self.logger.verbose("GPU VRAM used: %s", vram)
        return vram

    def get_free(self):
        """ Return the vram available """
        self.initialize()
        if self.plaid:
            # NB There is no useful way to get free VRAM on PlaidML.
            # OpenCL loads and unloads VRAM as required, so this returns the total memory
            # It's not particularly useful
            vram = self.plaid.vram
        elif IS_MACOS:
            vram = [pynvx.cudaGetMemFree(handle, ignore=True) / (1024 * 1024)
                    for handle in self.handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 * 1024)
                    for handle in self.handles]
        self.shutdown()
        if self.logger:
            self.logger.debug("GPU VRAM free: %s", vram)
        return vram

    def get_card_most_free(self, supports_plaidml=True):
        """ Return the card and available VRAM for active card with
            most VRAM free """
        if self.device_count == 0 or (self.is_plaidml and not supports_plaidml):
            return {"card_id": -1,
                    "device": "No Nvidia devices found",
                    "free": 2048,
                    "total": 2048}
        free_vram = [self.get_free()[i] for i in self.active_devices]
        vram_free = max(free_vram)
        card_id = self.active_devices[free_vram.index(vram_free)]
        retval = {"card_id": card_id,
                  "device": self.devices[card_id],
                  "free": vram_free,
                  "total": self.vram[card_id]}
        if self.logger:
            self.logger.debug("Active GPU Card with most free VRAM: %s", retval)
        return retval
