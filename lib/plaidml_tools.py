#!/usr/bin python3

""" PlaidML tools.

Statistics and setup for PlaidML on AMD devices.

This module must be kept separate from Keras, and be called prior to any Keras import, as the
plaidML Keras backend is set from this module.
"""

import json
import logging
import os
import sys

import plaidml

_INIT = False
_LOGGER = None
_EXCLUDE_DEVICES = []


class PlaidMLStats():
    """ Handles the initialization of PlaidML and the returning of GPU information for connected
    cards from the PlaidML library.

    This class is initialized early in Faceswap's Launch process from :func:`setup_plaidml`, with
    statistics made available from :class:`~lib.gpu_stats.GPUStats`

    Parameters
    ---------
    log_level: str, optional
        The requested Faceswap log level. Also dictates the level that PlaidML logging is set at.
        Default:`"INFO"`
    log: bool, optional
        Whether this class should output to the logger. If statistics are being accessed during a
        crash, then the logger may not be available, so this gives the option to turn logging off
        in those kinds of situations. Default:``True``
    """
    def __init__(self, log_level="INFO", log=True):
        if not _INIT and log:
            # Logger held internally, as we don't want to log when obtaining system stats on crash
            global _LOGGER  # pylint:disable=global-statement
            _LOGGER = logging.getLogger(__name__)
            _LOGGER.debug("Initializing: %s: (log_level: %s, log: %s)",
                          self.__class__.__name__, log_level, log)
        self._initialize(log_level)
        self._ctx = plaidml.Context()
        self._supported_devices = self._get_supported_devices()
        self._devices = self._get_all_devices()

        self._device_details = [json.loads(device.details.decode())
                                for device in self._devices if device.details]
        if self._devices and not self.active_devices:
            self._load_active_devices()
        if _LOGGER:
            _LOGGER.debug("Initialized: %s", self.__class__.__name__)

    # PROPERTIES
    @property
    def devices(self):
        """list:  The :class:`pladml._DeviceConfig` objects for GPUs that PlaidML has
        discovered. """
        return self._devices

    @property
    def active_devices(self):
        """ list: List of device indices for active GPU devices. """
        return [idx for idx, d_id in enumerate(self._ids)
                if d_id in plaidml.settings.device_ids and idx not in _EXCLUDE_DEVICES]

    @property
    def device_count(self):
        """ int: The total number of GPU Devices discovered. """
        return len(self._devices)

    @property
    def drivers(self):
        """ list: The driver versions for each GPU device that PlaidML has discovered. """
        return [device.get("driverVersion", "No Driver Found") for device in self._device_details]

    @property
    def vram(self):
        """ list: The VRAM of each GPU device that PlaidML has discovered. """
        return [int(device.get("globalMemSize", 0)) / (1024 * 1024)
                for device in self._device_details]

    @property
    def names(self):
        """ list: The name of each GPU device that PlaidML has discovered. """
        return ["{} - {} ({})".format(
            device.get("vendor", "unknown"),
            device.get("name", "unknown"),
            "supported" if idx in self._supported_indices else "experimental")
                for idx, device in enumerate(self._device_details)]

    @property
    def _ids(self):
        """ list: The device identification for each GPU device that PlaidML has discovered. """
        return [device.id.decode() for device in self._devices]

    @property
    def _experimental_indices(self):
        """ list: The indices corresponding to :attr:`_ids` of GPU devices marked as
        "experimental". """
        retval = [idx for idx, device in enumerate(self.devices)
                  if device not in self._supported_indices]
        if _LOGGER:
            _LOGGER.debug(retval)
        return retval

    @property
    def _supported_indices(self):
        """ list: The indices corresponding to :attr:`_ids` of GPU devices marked as
        "supported". """
        retval = [idx for idx, device in enumerate(self._devices)
                  if device in self._supported_devices]
        if _LOGGER:
            _LOGGER.debug(retval)
        return retval

    # INITIALIZATION
    def _initialize(self, log_level):
        """ Initialize PlaidML.

        Set PlaidML to use Faceswap's logger, and set the logging level

        Parameters
        ----------
        log_level: str, optional
            The requested Faceswap log level. Also dictates the level that PlaidML logging is set
            at.
        """
        global _INIT  # pylint:disable=global-statement
        if _INIT:
            if _LOGGER:
                _LOGGER.debug("PlaidML already initialized")
            return
        if _LOGGER:
            _LOGGER.debug("Initializing PlaidML")
        self._set_plaidml_logger()
        self._set_verbosity(log_level)
        _INIT = True
        if _LOGGER:
            _LOGGER.debug("Initialized PlaidML")

    @classmethod
    def _set_plaidml_logger(cls):
        """ Set PlaidMLs default logger to Faceswap Logger and prevent propagation. """
        if _LOGGER:
            _LOGGER.debug("Setting PlaidML Default Logger")
        plaidml.DEFAULT_LOG_HANDLER = logging.getLogger("plaidml_root")
        plaidml.DEFAULT_LOG_HANDLER.propagate = 0
        if _LOGGER:
            _LOGGER.debug("Set PlaidML Default Logger")

    @classmethod
    def _set_verbosity(cls, log_level):
        """ Set the PlaidML logging verbosity

        log_level: str
            The requested Faceswap log level. Also dictates the level that PlaidML logging is set
            at.
        """
        if _LOGGER:
            _LOGGER.debug("Setting PlaidML Loglevel: %s", log_level)
        if isinstance(log_level, int):
            numeric_level = log_level
        else:
            numeric_level = getattr(logging, log_level.upper(), None)
        if numeric_level < 10:
            # DEBUG Logging
            plaidml._internal_set_vlog(1)  # pylint:disable=protected-access
        elif numeric_level < 20:
            # INFO Logging
            plaidml._internal_set_vlog(0)  # pylint:disable=protected-access
        else:
            # WARNING Logging
            plaidml.quiet()

    def _get_supported_devices(self):
        """ Obtain GPU devices from PlaidML that are marked as "supported".

        Returns
        -------
        list
            The :class:`pladml._DeviceConfig` objects for GPUs that PlaidML has discovered.
        """
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = False
        devices = plaidml.devices(self._ctx, limit=100, return_all=True)[0]
        plaidml.settings.experimental = experimental_setting

        supported = [device for device in devices
                     if device.details
                     and json.loads(device.details.decode()).get("type", "cpu").lower() == "gpu"]
        if _LOGGER:
            _LOGGER.debug(supported)
        return supported

    def _get_all_devices(self):
        """ Obtain all available (experimental and supported) GPU devices from PlaidML.

        Returns
        -------
        list
            The :class:`pladml._DeviceConfig` objects for GPUs that PlaidML has discovered.
        """
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = True
        devices, _ = plaidml.devices(self._ctx, limit=100, return_all=True)
        plaidml.settings.experimental = experimental_setting

        experi = [device for device in devices
                  if device.details
                  and json.loads(device.details.decode()).get("type", "cpu").lower() == "gpu"]
        if _LOGGER:
            _LOGGER.debug("Experimental Devices: %s", experi)
        all_devices = experi + self._supported_devices
        if _LOGGER:
            _LOGGER.debug(all_devices)
        return all_devices

    def _load_active_devices(self):
        """ If the plaidml user configuration settings exist, then set the default GPU from the
        settings file, Otherwise set the GPU to be the one with most VRAM. """
        if not os.path.exists(plaidml.settings.user_settings):  # pylint:disable=no-member
            if _LOGGER:
                _LOGGER.debug("Setting largest PlaidML device")
            self._set_largest_gpu()
        else:
            if _LOGGER:
                _LOGGER.debug("Setting PlaidML devices from user_settings")

    def _set_largest_gpu(self):
        """ Set the default GPU to be a supported device with the most available VRAM. If no
        supported device is available, then set the GPU to be the an experimental device with the
        most VRAM available. """
        category = "supported" if self._supported_devices else "experimental"
        if _LOGGER:
            _LOGGER.debug("Obtaining largest %s device", category)
        indices = getattr(self, "_{}_indices".format(category))
        if not indices:
            _LOGGER.error("Failed to automatically detect your GPU.")
            _LOGGER.error("Please run `plaidml-setup` to set up your GPU.")
            sys.exit(1)
        max_vram = max([self.vram[idx] for idx in indices])
        if _LOGGER:
            _LOGGER.debug("Max VRAM: %s", max_vram)
        gpu_idx = min([idx for idx, vram in enumerate(self.vram)
                       if vram == max_vram and idx in indices])
        if _LOGGER:
            _LOGGER.debug("GPU IDX: %s", gpu_idx)

        selected_gpu = self._ids[gpu_idx]
        if _LOGGER:
            _LOGGER.info("Setting GPU to largest available %s device. If you want to override "
                         "this selection, run `plaidml-setup` from the command line.", category)

        plaidml.settings.experimental = category == "experimental"
        plaidml.settings.device_ids = [selected_gpu]


def setup_plaidml(log_level, exclude_devices):
    """ Setup PlaidML for AMD Cards.

    Sets the Keras backend to PlaidML, loads the plaidML backend and makes GPU Device information
    from PlaidML available to :class:`~lib.gpu_stats.GPUStats`.


    Parameters
    ----------
    log_level: str
        Faceswap's log level. Used for setting the log level inside PlaidML
    exclude_devices: list
        A list of integers of device IDs that should not be used by Faceswap
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.info("Setting up for PlaidML")
    logger.verbose("Setting Keras Backend to PlaidML")
    # Add explicitly excluded devices to list. The contents have already been checked in GPUStats
    if exclude_devices:
        _EXCLUDE_DEVICES.extend(int(idx) for idx in exclude_devices)
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    plaid = PlaidMLStats(log_level)
    logger.info("Using GPU(s): %s", [plaid.names[i] for i in plaid.active_devices])
    logger.info("Successfully set up for PlaidML")
