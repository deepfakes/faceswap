#!/usr/bin python3

""" PlaidML tools

    Must be kept separate from keras as the keras backend needs to be set from this module
"""

import json
import logging
import os

import plaidml

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PlaidMLStats():
    """ Stats for plaidML """
    def __init__(self, loglevel=logging.INFO, set_for_test=False):
        logger.debug("Initializing: %s: (loglevel: %s, set_for_test: %s)", self.__class__.__name__,
                     loglevel, set_for_test)
        self.set_plaidml_logger()
        self.ctx = plaidml.Context()
        self.set_verbosity(loglevel)
        if set_for_test:
            plaidml.settings._setup_for_test(  # pylint: disable=no-member, protected-access
                plaidml.settings.user_settings)  # pylint: disable=no-member
        self.plaidml_devices = self.get_devices()
        self.device_details = [json.loads(device.details.decode())
                               for device in self.plaidml_devices]
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def device_count(self):
        """ Return count of PlaidML Devices """
        return len(self.plaidml_devices)

    @property
    def drivers(self):
        """ Return all PlaidML device drivers """
        return [device.get("driverVersion", "No Driver Found") for device in self.device_details]

    @property
    def vram(self):
        """ Return Total VRAM for all PlaidML Devices """
        return [int(device.get("maxMemAllocSize", 0)) / (1024 * 1024)
                for device in self.device_details]

    @property
    def ids(self):
        """ Return all PlaidML Device IDs """
        return [device.id.decode() for device in self.plaidml_devices]

    @staticmethod
    def set_plaidml_logger():
        """ Set PlaidMLs default logger to Faceswap Logger and prevent propagation """
        if plaidml.DEFAULT_LOG_HANDLER == logger:
            return
        logger.debug("Setting PlaidML Default Logger")
        plaidml.DEFAULT_LOG_HANDLER = logger
        plaidml.DEFAULT_LOG_HANDLER.propagate = 0
        logger.debug("Set PlaidML Default Logger")

    def get_devices(self):
        """ Return list of dicts of discovered GPU Devices """
        plaidml.settings.experimental = False
        devices, _ = plaidml.devices(self.ctx, limit=100, return_all=True)
        plaidml.settings.experimental = True
        exp_devices, _ = plaidml.devices(self.ctx, limit=100, return_all=True)
        all_devices = devices + exp_devices
        all_devices = [device for device in all_devices
                       if json.loads(device.details.decode())["type"].lower() != "cpu"]
        return all_devices

    @staticmethod
    def set_verbosity(loglevel):
        """ Set the PlaidML Verbosity """
        logger.debug("Setting PlaidML Loglevel: %s", loglevel)
        numeric_level = getattr(logging, loglevel.upper(), None)
        if numeric_level < 10:
            # DEBUG Logging
            plaidml._internal_set_vlog(1)  # pylint: disable=protected-access
        elif numeric_level < 20:
            # INFO Logging
            plaidml._internal_set_vlog(0)  # pylint: disable=protected-access
        else:
            # WARNING Logging
            plaidml.quiet()


def plaidml_setup(loglevel):
    """ Sets up PlaidML, setting the primary GPU to the one with mode VRAM """
    logger.info("Setting up for PlaidML")
    logger.info("Setting Keras Backend to PlaidML")
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    stats = PlaidMLStats(loglevel=loglevel, set_for_test=True)
    biggest_gpu_idx = stats.vram.index(max(stats.vram))
    selected_gpu = stats.ids[biggest_gpu_idx]
    logger.info("Using GPU: %s", selected_gpu)
    plaidml.settings.device_ids = [selected_gpu]
    plaidml.settings.save(plaidml.settings.user_settings)  # pylint: disable=no-member
    logger.info("Successfully set up for PlaidML")
