#!/usr/bin python3
""" GPU VRAM allocator calculations """

from lib import gpu_stats


class GPUMem(object):
    """ Sets the scale to factor for dlib images
        and the ratio of vram to use for tensorflow """

    def __init__(self):
        self.verbose = False
        self.output_shown = False
        self.stats = gpu_stats.GPUStats()
        self.gpu_memory = min(self.stats.get_free())
        self.tensorflow_ratio = self.set_tensor_gpu_ratio()
        self.scale_to = None

    def output_stats(self):
        """ Output stats in verbose mode """
        if self.output_shown or not self.verbose:
            return
        print("\n----- Initial GPU Stats -----")
        self.stats.print_info()
        print("GPU VRAM free:    {}".format(self.gpu_memory))
        print("-----------------------------\n")
        self.output_shown = True

    def get_available_vram(self):
        """ Update self.gpu_memory to current available """
        # TODO don't go for smallest card
        self.gpu_memory = min(self.stats.get_free())
        if self.verbose:
            print("GPU VRAM free:    {}".format(self.gpu_memory))

    def set_tensor_gpu_ratio(self):
        """ Set the ratio of GPU memory to use
            for tensorflow session

            Ideally at least 2304MB is required, but
            will run with less (with warnings) """

        if self.gpu_memory < 2030:
            ratio = 1024.0 / self.gpu_memory
        elif self.gpu_memory < 3045:
            ratio = 1560.0 / self.gpu_memory
        elif self.gpu_memory < 4060:
            ratio = 2048.0 / self.gpu_memory
        else:
            ratio = 2304.0 / self.gpu_memory
        return ratio

    def set_scale_to(self):
        """ Set the size to scale images down to for specific
            gfx cards.
            DLIB VRAM allocation is linear to pixel count """
        self.get_available_vram()
        buffer = 64  # 64MB overhead buffer
        free_mem = self.gpu_memory - buffer
        gradient = 213 / 524288
        constant = 307
        self.scale_to = int((free_mem - constant) / gradient)
