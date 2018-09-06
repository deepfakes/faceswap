#!/usr/bin python3
""" GPU VRAM allocator calculations """

from lib.gpu_stats import GPUStats


class GPUMem():
    """ Sets the scale to factor for dlib images
        and the ratio of vram to use for tensorflow """

    def __init__(self):
        self.initialized = False
        self.verbose = False
        self.stats = GPUStats()
        self.dlib_buffer = 64
        self.vram_free = None
        self.vram_total = None
        self.scale_to = None

        self.device = self.set_device()

        if self.device == -1:
            # Limit ram usage to 2048 for CPU
            self.vram_total = 2048
        else:
            self.vram_total = self.stats.vram[self.device]

        self.get_available_vram()

    def set_device(self):
        """ Set the default device """
        if self.stats.device_count == 0:
            return -1
        return 0
        # TF selects first device, so this is used for stats
        # TODO select and use device with most available VRAM
        # TODO create virtual devices/allow multiple GPUs for
        # parallel processing

    def set_device_with_max_free_vram(self):
        """ Set the device with the most available free vram """
        # TODO Implement this to select the device with most available VRAM
        free_mem = self.stats.get_free()
        self.vram_free = max(free_mem)
        self.device = free_mem.index(self.vram_free)

    def get_available_vram(self):
        """ Recalculate the available vram """
        if self.device == -1:
            # Limit RAM to 2GB for non-gpu
            self.vram_free = 2048
        else:
            free_mem = self.stats.get_free()
            self.vram_free = free_mem[self.device]

        if self.verbose:
            if self.device == -1:
                print("No GPU. Limiting RAM usage to "
                      "{}MB".format(self.vram_free))
            print("GPU VRAM free:    {}".format(self.vram_free))

    def output_stats(self):
        """ Output stats in verbose mode """
        if not self.verbose:
            return
        print("\n----- Initial GPU Stats -----")
        if self.device == -1:
            print("No GPU. Limiting RAM usage to {}MB".format(self.vram_free))
        self.stats.print_info()
        print("GPU VRAM free:    {}".format(self.vram_free))
        print("-----------------------------\n")

    def get_tensor_gpu_ratio(self):
        """ Set the ratio of GPU memory to use for tensorflow session for
            keras points predictor.

            Ideally 2304MB is required, but will run with less
            (with warnings).

            This is only required if running with DLIB. MTCNN will share
            the tensorflow session. """
        if self.vram_free < 2030:
            ratio = 1024.0 / self.vram_total
        elif self.vram_free < 3045:
            ratio = 1560.0 / self.vram_total
        elif self.vram_free < 4060:
            ratio = 2048.0 / self.vram_total
        else:
            ratio = 2304.0 / self.vram_total

        return ratio

    def set_scale_to(self, detector):
        """ Set the size to scale images down to for specific detector
            and available VRAM

            DLIB VRAM allocation is linear to pixel count

            MTCNN is weird. Not linear at low levels,
            then fairly linear up to 3360x1890 then
            requirements drop again.
            As 3360x1890 is hi-res, just this scale is
            used for calculating image scaling """

        # MTCNN VRAM Usage Stats
        # Crudely Calculated at default values
        # The formula may need ammending, but it should
        # work for most use cases
        # 480x270 = 267.56 MB
        # 960x540 = 333.18 MB
        # 1280x720 = 592.32 MB
        # 1440x810 = 746.56 MB
        # 1920x1080 = 1.30 GB
        # 2400x1350 = 2.03 GB
        # 2880x1620 = 2.93 GB
        # 3360x1890 = 3.98 GB
        # 3840x2160 = 2.62 GB <--??
        # 4280x2800 = 3.69 GB

        detector = "dlib" if detector in ("dlib-cnn",
                                          "dlib-hog",
                                          "dlib-all") else detector
        gradient = 3483.2 / 9651200  # MTCNN
        constant = 1.007533156  # MTCNN
        if detector == "dlib":
            self.get_available_vram()
            gradient = 213 / 524288
            constant = 307

        free_mem = self.vram_free - self.dlib_buffer  # overhead buffer
        if self.verbose:
            print("Allocating for Detector: {}".format(free_mem))

        self.scale_to = int((free_mem - constant) / gradient)

        if self.scale_to < 4097:
            raise ValueError("Images would be shrunk too much "
                             "for successful extraction")
