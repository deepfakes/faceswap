#!/usr/bin python3
""" Information on available Nvidia GPUs """

import pynvml


class GPUStats(object):
    """ Holds information about system GPU(s) """
    def __init__(self):
        self.verbose = False

        self.initialized = False
        self.device_count = 0
        self.handles = None
        self.driver = None
        self.devices = None
        self.vram = None

        self.initialize()

        if self.device_count == 0:
            return

        self.driver = self.get_driver()
        self.devices = self.get_devices()
        self.vram = self.get_vram()

        self.shutdown()

    def initialize(self):
        """ Initialize pynvml """
        if not self.initialized:
            try:
                pynvml.nvmlInit()
            except pynvml.NVMLError_LibraryNotFound:
                self.initialized = True
                return
            self.initialized = True
            self.get_device_count()
            self.get_handles()

    def shutdown(self):
        """ Shutdown pynvml """
        if self.initialized:
            self.handles = None
            pynvml.nvmlShutdown()
            self.initialized = False

    def get_device_count(self):
        """ Return count of Nvidia devices """
        try:
            self.device_count = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            self.device_count = 0

    def get_handles(self):
        """ Return all listed Nvidia handles """
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                        for i in range(self.device_count)]

    @staticmethod
    def get_driver():
        """ Get the driver version """
        try:
            driver = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
        except pynvml.NVMLError:
            driver = "No Nvidia driver found"
        return driver

    def get_devices(self):
        """ Return total vram in megabytes per device """
        vram = [pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                for handle in self.handles]
        return vram

    def get_vram(self):
        """ Return total vram in megabytes per device """
        vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)
                for handle in self.handles]
        return vram

    def get_used(self):
        """ Return the vram in use """
        self.initialize()
        vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
                for handle in self.handles]
        self.shutdown()

        if self.verbose:
            print("GPU VRAM used:    {}".format(vram))

        return vram

    def get_free(self):
        """ Return the vram available """
        self.initialize()
        vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 * 1024)
                for handle in self.handles]
        self.shutdown()
        return vram

    def print_info(self):
        """ Output GPU info in verbose mode """
        print("GPU Driver:       {}".format(self.driver))
        print("GPU Device count: {}".format(self.device_count))
        print("GPU Devices:      {}".format(self.devices))
        print("GPU VRAM:         {}".format(self.vram))
