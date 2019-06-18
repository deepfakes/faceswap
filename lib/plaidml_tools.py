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
    def __init__(self, loglevel=logging.INFO):
        logger.debug("Initializing: %s: (loglevel: %s)", self.__class__.__name__, loglevel)
        self.set_plaidml_logger()
        self.ctx = plaidml.Context()
        self.set_verbosity(loglevel)
        self.supported_devices = self.get_supported_devices()
        self.devices = self.get_all_devices()

        self.device_details = [json.loads(device.details.decode()) for device in self.devices]
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def active_devices(self):
        """ Return the active device IDs """
        return plaidml.settings.device_ids

    @property
    def device_count(self):
        """ Return count of PlaidML Devices """
        return len(self.devices)

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
        return [device.id.decode() for device in self.devices]

    @staticmethod
    def set_plaidml_logger():
        """ Set PlaidMLs default logger to Faceswap Logger and prevent propagation """
        if plaidml.DEFAULT_LOG_HANDLER == logger:
            return
        logger.debug("Setting PlaidML Default Logger")
        plaidml.DEFAULT_LOG_HANDLER = logging.getLogger("plaidml_root")
        plaidml.DEFAULT_LOG_HANDLER.propagate = 0
        logger.debug("Set PlaidML Default Logger")

    def get_supported_devices(self):
        """ Return a list of supported devices """
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = False
        devices, _ = plaidml.devices(self.ctx, limit=100, return_all=True)
        plaidml.settings.experimental = experimental_setting

        supported = [device for device in devices
                     if json.loads(device.details.decode())["type"].lower() != "cpu"]
        logger.debug(supported)
        return supported

    def get_all_devices(self):
        """ Return list of supported and experimental devices """
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = True
        devices, _ = plaidml.devices(self.ctx, limit=100, return_all=True)
        plaidml.settings.experimental = experimental_setting

        experimental = [device for device in devices
                        if json.loads(device.details.decode())["type"].lower() != "cpu"]
        logger.debug("Experimental Devices: %s", experimental)
        all_devices = experimental + self.supported_devices
        logger.debug(all_devices)
        return all_devices

    @staticmethod
    def set_verbosity(loglevel):
        """ Set the PlaidML Verbosity """
        logger.debug("Setting PlaidML Loglevel: %s", loglevel)
        numeric_level = getattr(logging, loglevel.upper(), None)
        if numeric_level < 10:
            # DEBUG Logging
            plaidml._internal_set_vlog(1)  # pylint:disable=protected-access
        elif numeric_level < 20:
            # INFO Logging
            plaidml._internal_set_vlog(0)  # pylint:disable=protected-access
        else:
            # WARNING Logging
            plaidml.quiet()

    def load_active_devices(self):
        """ Load settings from PlaidML.settings.usersettings or select biggest gpu """
        if not os.path.exists(plaidml.settings.user_settings):  # pylint:disable=no-member
            logger.debug("Setting largest PlaidML device")
            self.set_largest_gpu()
        else:
            logger.debug("Setting PlaidML devices from user_settings")

    def set_largest_gpu(self):
        """ Get the GPU with largest VRAM. Prioritise supported over experimental """
        max_vram = max(self.vram)
        indices = [idx for idx, vram in enumerate(self.vram) if vram == max_vram]
        logger.debug("Device indices with max vram (%s): %s", max_vram, indices)
        selected_gpu = None
        for idx in indices:
            device = self.devices[idx]
            if device in self.supported_devices:
                selected_gpu = self.ids[idx]
                break

        if not selected_gpu:
            logger.debug("No GPUs found in supported. Setting to Experimental")
            plaidml.settings.experimental = True
            selected_gpu = self.ids[indices[0]]
        logger.info("Setting GPU to largest available. If you want to override this selection, "
                    "run `plaidml-setup` from the command line.")
        plaidml.settings.device_ids = [selected_gpu]


def setup_plaidml(loglevel):
    """ Setup plaidml for AMD Cards """
    logger.info("Setting up for PlaidML")
    logger.info("Setting Keras Backend to PlaidML")
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    plaid = PlaidMLStats(loglevel)
    plaid.load_active_devices()
    logger.info("Using GPU: %s", plaid.active_devices)
    logger.info("Successfully set up for PlaidML")
