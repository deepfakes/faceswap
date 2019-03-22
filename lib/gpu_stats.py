#!/usr/bin python3
""" Information on available Nvidia GPUs """

import logging
import platform

if platform.system() == 'Darwin':
    import pynvx  # pylint: disable=import-error
    IS_MACOS = True
else:
    import pynvml
    IS_MACOS = False


class GPUStats():
    """ Holds information about system GPU(s) """
    def __init__(self, log=True):
        self.logger = None
        if log:
            # Logger is held internally, as we don't want to log
            # when obtaining system stats on crash
            self.logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
            self.logger.debug("Initializing %s", self.__class__.__name__)

        self.initialized = False
        self.device_count = 0
        self.handles = None
        self.driver = None
        self.devices = None
        self.vram = None

        self.initialize()

        self.driver = self.get_driver()
        self.devices = self.get_devices()
        self.vram = self.get_vram()
        if self.device_count == 0:
            if self.logger:
                self.logger.warning("No GPU detected. Switching to CPU mode")
            return

        self.shutdown()
        if self.logger:
            self.logger.debug("Initialized %s", self.__class__.__name__)

    def initialize(self):
        """ Initialize pynvml """
        if not self.initialized:
            if IS_MACOS:
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
                except (pynvml.NVMLError_LibraryNotFound,
                        pynvml.NVMLError_DriverNotLoaded,
                        pynvml.NVMLError_NoPermission):
                    self.initialized = True
                    return
            self.initialized = True
            self.get_device_count()
            self.get_handles()

    def shutdown(self):
        """ Shutdown pynvml """
        if self.initialized:
            self.handles = None
            if not IS_MACOS:
                pynvml.nvmlShutdown()
            self.initialized = False

    def get_device_count(self):
        """ Return count of Nvidia devices """
        if IS_MACOS:
            self.device_count = pynvx.cudaDeviceGetCount(ignore=True)
        else:
            try:
                self.device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                self.device_count = 0
        if self.logger:
            self.logger.debug("GPU Device count: %s", self.device_count)

    def get_handles(self):
        """ Return all listed Nvidia handles """
        if IS_MACOS:
            self.handles = pynvx.cudaDeviceGetHandles(ignore=True)
        else:
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                            for i in range(self.device_count)]
        if self.logger:
            self.logger.debug("GPU Handles found: %s", len(self.handles))

    def get_driver(self):
        """ Get the driver version """
        if IS_MACOS:
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
        if IS_MACOS:
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
        if IS_MACOS:
            vram = [pynvx.cudaGetMemFree(handle, ignore=True) / (1024 * 1024)
                    for handle in self.handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 * 1024)
                    for handle in self.handles]
        self.shutdown()
        if self.logger:
            self.logger.debug("GPU VRAM free: %s", vram)
        return vram

    def get_card_most_free(self):
        """ Return the card and available VRAM for card with
            most VRAM free """
        if self.device_count == 0:
            return {"card_id": -1,
                    "device": "No Nvidia devices found",
                    "free": 2048,
                    "total": 2048}
        free_vram = self.get_free()
        vram_free = max(free_vram)
        card_id = free_vram.index(vram_free)
        retval = {"card_id": card_id,
                  "device": self.devices[card_id],
                  "free": vram_free,
                  "total": self.vram[card_id]}
        if self.logger:
            self.logger.debug("GPU Card with most free VRAM: %s", retval)
        return retval
