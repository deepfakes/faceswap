#!/usr/bin python3
""" Information on available Nvidia GPUs """

import platform

if platform.system() == 'Darwin':
    import pynvx
    is_macos = True
else:
    import pynvml
    is_macos = False


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
            if is_macos:
                try:
                    pynvx.cudaInit()
                except RuntimeError:
                    self.initialized = True
                    return
            else:
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
            if not is_macos:
                pynvml.nvmlShutdown()
            self.initialized = False

    def get_device_count(self):
        """ Return count of Nvidia devices """
        if is_macos:
            self.device_count = pynvx.cudaDeviceGetCount(ignore=True)
        else:
            try:
                self.device_count = pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError:
                self.device_count = 0

    def get_handles(self):
        """ Return all listed Nvidia handles """
        if is_macos:
            self.handles = pynvx.cudaDeviceGetHandles(ignore=True)
        else:
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                            for i in range(self.device_count)]

    @staticmethod
    def get_driver():
        """ Get the driver version """
        if is_macos:
            driver = pynvx.cudaSystemGetDriverVersion(ignore=True)
        else:
            try:
                driver = pynvml.nvmlSystemGetDriverVersion().decode("utf-8")
            except pynvml.NVMLError:
                driver = "No Nvidia driver found"
        return driver

    def get_devices(self):
        """ Return name of devices """
        self.initialize()
        if is_macos:
            names = [pynvx.cudaGetName(handle, ignore=True)
                    for handle in self.handles]
        else:
            names = [pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                    for handle in self.handles]
        return names

    def get_vram(self):
        """ Return total vram in megabytes per device """
        self.initialize()
        if is_macos:
            vram = [pynvx.cudaGetMemTotal(handle, ignore=True) / (1024 * 1024)
                    for handle in self.handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024)
                    for handle in self.handles]
        return vram

    def get_used(self):
        """ Return the vram in use """
        self.initialize()
        if is_macos:
            vram = [pynvx.cudaGetMemUsed(handle, ignore=True) / (1024 * 1024)
                    for handle in self.handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)
                    for handle in self.handles]
        self.shutdown()

        if self.verbose:
            print("GPU VRAM used:    {}".format(vram))

        return vram

    def get_free(self):
        """ Return the vram available """
        self.initialize()
        if is_macos:
            vram = [pynvx.cudaGetMemFree(handle, ignore=True) / (1024 * 1024)
                    for handle in self.handles]
        else:
            vram = [pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024 * 1024)
                    for handle in self.handles]
        self.shutdown()
        return vram

    def get_card_most_free(self):
        """ Return the card and available VRAM for card with
            most VRAM free """
        free_vram = self.get_free()
        vram_free = max(free_vram)
        card_id = free_vram.index(vram_free)
        return {"card_id": card_id,
                "device": self.devices[card_id],
                "free": vram_free,
                "total": self.vram[card_id]}

    
    def print_info(self):
        """ Output GPU info in verbose mode """
        print("GPU Driver:       {}".format(self.driver))
        print("GPU Device count: {}".format(self.device_count))
        print("GPU Devices:      {}".format(self.devices))
        print("GPU VRAM:         {}".format(self.vram))
