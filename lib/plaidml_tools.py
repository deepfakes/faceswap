#!/usr/bin python3

""" PlaidML tools

    Must be kept separate from keras as the keras backend needs to be set from this module
"""

import json
import logging
import os

import plaidml

_INIT = False
_LOGGER = None


class PlaidMLStats():
    """ Stats for plaidML """
    def __init__(self, loglevel="INFO", log=True):
        if not _INIT and log:
            # Logger is held internally, as we don't want to log
            # when obtaining system stats on crash
            global _LOGGER  # pylint:disable=global-statement
            _LOGGER = logging.getLogger(__name__)  # pylint:disable=invalid-name
            _LOGGER.debug("Initializing: %s: (loglevel: %s, log: %s)",
                          self.__class__.__name__, loglevel, log)
        self.initialize(loglevel)
        self.ctx = plaidml.Context()
        self.supported_devices = self.get_supported_devices()
        self.devices = self.get_all_devices()

        self.device_details = [json.loads(device.details.decode()) for device in self.devices]
        if _LOGGER:
            _LOGGER.debug("Initialized: %s", self.__class__.__name__)

    # PROPERTIES
    @property
    def active_devices(self):
        """ Return the active device IDs """
        return [idx for idx, d_id in enumerate(self.ids) if d_id in plaidml.settings.device_ids]

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
        return [int(device.get("globalMemSize", 0)) / (1024 * 1024)
                for device in self.device_details]

    @property
    def max_alloc(self):
        """ Return Maximum allowed VRAM allocation for all PlaidML Devices """
        return [int(device.get("maxMemAllocSize", 0)) / (1024 * 1024)
                for device in self.device_details]

    @property
    def ids(self):
        """ Return all PlaidML Device IDs """
        return [device.id.decode() for device in self.devices]

    @property
    def names(self):
        """ Return all PlaidML Device Names """
        return ["{} - {} ({})".format(
            device.get("vendor", "unknown"),
            device.get("name", "unknown"),
            "supported" if idx in self.supported_indices else "experimental")
                for idx, device in enumerate(self.device_details)]

    @property
    def supported_indices(self):
        """ Return the indices from self.devices of GPUs categorized as supported """
        retval = [idx for idx, device in enumerate(self.devices)
                  if device in self.supported_devices]
        if _LOGGER:
            _LOGGER.debug(retval)
        return retval

    @property
    def experimental_indices(self):
        """ Return the indices from self.devices of GPUs categorized as experimental """
        retval = [idx for idx, device in enumerate(self.devices)
                  if device not in self.supported_devices]
        if _LOGGER:
            _LOGGER.debug(retval)
        return retval

    # INITIALIZATION
    def initialize(self, loglevel):
        """ Initialize PlaidML """
        global _INIT  # pylint:disable=global-statement
        if _INIT:
            if _LOGGER:
                _LOGGER.debug("PlaidML already initialized")
            return
        if _LOGGER:
            _LOGGER.debug("Initializing PlaidML")
        self.set_plaidml_logger()
        self.set_verbosity(loglevel)
        _INIT = True
        if _LOGGER:
            _LOGGER.debug("Initialized PlaidML")

    @staticmethod
    def set_plaidml_logger():
        """ Set PlaidMLs default logger to Faceswap Logger and prevent propagation """
        if _LOGGER:
            _LOGGER.debug("Setting PlaidML Default Logger")
        plaidml.DEFAULT_LOG_HANDLER = logging.getLogger("plaidml_root")
        plaidml.DEFAULT_LOG_HANDLER.propagate = 0
        if _LOGGER:
            _LOGGER.debug("Set PlaidML Default Logger")

    @staticmethod
    def set_verbosity(loglevel):
        """ Set the PlaidML Verbosity """
        if _LOGGER:
            _LOGGER.debug("Setting PlaidML Loglevel: %s", loglevel)
        if isinstance(loglevel, int):
            numeric_level = loglevel
        else:
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

    def get_supported_devices(self):
        """ Return a list of supported devices """
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = False
        devices, _ = plaidml.devices(self.ctx, limit=100, return_all=True)
        plaidml.settings.experimental = experimental_setting

        supported = [device for device in devices
                     if json.loads(device.details.decode()).get("type", "cpu").lower() == "gpu"]
        if _LOGGER:
            _LOGGER.debug(supported)
        return supported

    def get_all_devices(self):
        """ Return list of supported and experimental devices """
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = True
        devices, _ = plaidml.devices(self.ctx, limit=100, return_all=True)
        plaidml.settings.experimental = experimental_setting

        experimental = [device for device in devices
                        if json.loads(device.details.decode()).get("type", "cpu").lower() == "gpu"]
        if _LOGGER:
            _LOGGER.debug("Experimental Devices: %s", experimental)
        all_devices = experimental + self.supported_devices
        if _LOGGER:
            _LOGGER.debug(all_devices)
        return all_devices

    def load_active_devices(self):
        """ Load settings from PlaidML.settings.usersettings or select biggest gpu """
        if not os.path.exists(plaidml.settings.user_settings):  # pylint:disable=no-member
            if _LOGGER:
                _LOGGER.debug("Setting largest PlaidML device")
            self.set_largest_gpu()
        else:
            if _LOGGER:
                _LOGGER.debug("Setting PlaidML devices from user_settings")

    def set_largest_gpu(self):
        """ Get a supported GPU with largest VRAM. If no supported, get largest experimental """
        category = "supported" if self.supported_devices else "experimental"
        if _LOGGER:
            _LOGGER.debug("Obtaining largest %s device", category)
        indices = getattr(self, "{}_indices".format(category))
        if not indices:
            _LOGGER.error("Failed to automatically detect your GPU.")
            _LOGGER.error("Please run `plaidml-setup` to set up your GPU.")
            exit()
        max_vram = max([self.vram[idx] for idx in indices])
        if _LOGGER:
            _LOGGER.debug("Max VRAM: %s", max_vram)
        gpu_idx = min([idx for idx, vram in enumerate(self.vram)
                       if vram == max_vram and idx in indices])
        if _LOGGER:
            _LOGGER.debug("GPU IDX: %s", gpu_idx)

        selected_gpu = self.ids[gpu_idx]
        if _LOGGER:
            _LOGGER.info("Setting GPU to largest available %s device. If you want to override "
                         "this selection, run `plaidml-setup` from the command line.", category)

        plaidml.settings.experimental = category == "experimental"
        plaidml.settings.device_ids = [selected_gpu]


def setup_plaidml(loglevel):
    """ Setup plaidml for AMD Cards """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.info("Setting up for PlaidML")
    logger.verbose("Setting Keras Backend to PlaidML")
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    plaid = PlaidMLStats(loglevel)
    plaid.load_active_devices()
    logger.info("Using GPU: %s", [plaid.ids[i] for i in plaid.active_devices])
    logger.info("Successfully set up for PlaidML")
