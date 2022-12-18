#!/usr/bin/env python3
""" Collects and returns Information on available AMD GPUs. """
import json
import logging
import os
import sys

from typing import List, Optional

import plaidml

from ._base import _GPUStats, _EXCLUDE_DEVICES


_PLAIDML_INITIALIZED: bool = False


def setup_plaidml(log_level: str, exclude_devices: List[int]) -> None:
    """ Setup PlaidML for AMD Cards.

    Sets the Keras backend to PlaidML, loads the plaidML backend and makes GPU Device information
    from PlaidML available to :class:`AMDStats`.

    Parameters
    ----------
    log_level: str
        Faceswap's log level. Used for setting the log level inside PlaidML
    exclude_devices: list
        A list of integers of device IDs that should not be used by Faceswap
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.info("Setting up for PlaidML")
    logger.verbose("Setting Keras Backend to PlaidML")  # type:ignore
    # Add explicitly excluded devices to list. The contents are checked in AMDstats
    if exclude_devices:
        _EXCLUDE_DEVICES.extend(int(idx) for idx in exclude_devices)
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    stats = AMDStats(log_level=log_level)
    logger.info("Using GPU(s): %s", [stats.names[i] for i in stats.active_devices])
    logger.info("Successfully set up for PlaidML")


class AMDStats(_GPUStats):
    """ Holds information and statistics about AMD GPU(s) available on the currently
    running system.

    Notes
    -----
    The quality of data that returns is very much dependent on the OpenCL implementation used
    for a particular OS. Some data is just not available at all, so assumptions and substitutions
    are made where required. PlaidML is used as an interface into OpenCL to obtain the required
    information.

    PlaidML is explicitly initialized inside this class, as it can be called from the command line
    arguments to list available GPUs. PlaidML needs to be set up and configured to obtain reliable
    information. As the function :func:`setup_plaidml` is called very early within the Faceswap
    and launch process and it references this class, initial PlaidML setup can all be handled here.

    Parameters
    ----------
    log: bool, optional
        Whether the class should output information to the logger. There may be occasions where the
        logger has not yet been set up when this class is queried. Attempting to log in these
        instances will raise an error. If GPU stats are being queried prior to the logger being
        available then this parameter should be set to ``False``. Otherwise set to ``True``.
        Default: ``True``
    """
    def __init__(self, log: bool = True, log_level: str = "INFO") -> None:

        self._log_level: str = log_level.upper()

        # Following attributes are set in :func:``_initialize``
        self._ctx: Optional[plaidml.Context] = None
        self._supported_devices: List[plaidml._DeviceConfig] = []
        self._all_devices: List[plaidml._DeviceConfig] = []
        self._device_details: List[dict] = []

        super().__init__(log=log)

    @property
    def active_devices(self) -> List[int]:
        """ list: The active device ids in use. """
        return self._active_devices

    @property
    def _plaid_ids(self) -> List[str]:
        """ list: The device identification for each GPU device that PlaidML has discovered. """
        return [device.id.decode("utf-8", errors="replace") for device in self._all_devices]

    @property
    def _experimental_indices(self) -> List[int]:
        """ list: The indices corresponding to :attr:`_ids` of GPU devices marked as
        "experimental". """
        retval = [idx for idx, device in enumerate(self._all_devices)
                  if device not in self._supported_indices]
        return retval

    @property
    def _supported_indices(self) -> List[int]:
        """ list: The indices corresponding to :attr:`_ids` of GPU devices marked as
        "supported". """
        retval = [idx for idx, device in enumerate(self._all_devices)
                  if device in self._supported_devices]
        return retval

    @property
    def _all_vram(self) -> List[int]:
        """ list: The VRAM of each GPU device that PlaidML has discovered. """
        return [int(int(device.get("globalMemSize", 0)) / (1024 * 1024))
                for device in self._device_details]

    @property
    def names(self) -> List[str]:
        """ list: The name of each GPU device that PlaidML has discovered. """
        return [f"{device.get('vendor', 'unknown')} - {device.get('name', 'unknown')} "
                f"({ 'supported' if idx in self._supported_indices else 'experimental'})"
                for idx, device in enumerate(self._device_details)]

    def _initialize(self) -> None:
        """ Initialize PlaidML for AMD GPUs.

        If :attr:`_is_initialized` is ``True`` then this function just returns performing no
        action.

        if ``False`` then PlaidML is setup, if not already, and GPU information is extracted
        from the PlaidML context.
        """
        if self._is_initialized:
            return
        self._log("debug", "Initializing PlaidML for AMD GPU.")

        self._initialize_plaidml()

        self._ctx = plaidml.Context()
        self._supported_devices = self._get_supported_devices()
        self._all_devices = self._get_all_devices()
        self._device_details = self._get_device_details()
        self._select_device()

        super()._initialize()

    def _initialize_plaidml(self) -> None:
        """ Initialize PlaidML on first call to this class and set global
        :attr:``_PLAIDML_INITIALIZED`` to ``True``. If PlaidML has already been initialized then
        return performing no action. """
        global _PLAIDML_INITIALIZED  # pylint:disable=global-statement

        if _PLAIDML_INITIALIZED:
            return

        self._log("debug", "Performing first time PlaidML setup.")
        self._set_plaidml_logger()

        _PLAIDML_INITIALIZED = True

    def _set_plaidml_logger(self) -> None:
        """ Set PlaidMLs default logger to Faceswap Logger, prevent propagation and set the correct
        log level. """
        self._log("debug", "Setting PlaidML Default Logger")

        plaidml.DEFAULT_LOG_HANDLER = logging.getLogger("plaidml_root")
        plaidml.DEFAULT_LOG_HANDLER.propagate = False

        numeric_level = getattr(logging, self._log_level, None)
        if numeric_level < 10:  # DEBUG Logging
            plaidml._internal_set_vlog(1)  # pylint:disable=protected-access
        elif numeric_level < 20:  # INFO Logging
            plaidml._internal_set_vlog(0)  # pylint:disable=protected-access
        else:  # WARNING LOGGING
            plaidml.quiet()

    def _get_supported_devices(self) -> List[plaidml._DeviceConfig]:
        """ Obtain GPU devices from PlaidML that are marked as "supported".

        Returns
        -------
        list_LOGGER.
            The :class:`plaidml._DeviceConfig` objects for all supported GPUs that PlaidML has
            discovered.
        """
        experimental_setting = plaidml.settings.experimental

        plaidml.settings.experimental = False
        devices = plaidml.devices(self._ctx, limit=100, return_all=True)[0]
        plaidml.settings.experimental = experimental_setting

        supported = [d for d in devices
                     if d.details
                     and json.loads(
                        d.details.decode("utf-8",
                                         errors="replace")).get("type", "cpu").lower() == "gpu"]

        self._log("debug", f"Obtained supported devices: {supported}")
        return supported

    def _get_all_devices(self) -> List[plaidml._DeviceConfig]:
        """ Obtain all available (experimental and supported) GPU devices from PlaidML.

        Returns
        -------
        list
            The :class:`pladml._DeviceConfig` objects for GPUs that PlaidML has discovered.
        """
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = True
        devices = plaidml.devices(self._ctx, limit=100, return_all=True)[0]
        plaidml.settings.experimental = experimental_setting

        experi = [d for d in devices
                  if d.details
                  and json.loads(
                    d.details.decode("utf-8",
                                     errors="replace")).get("type", "cpu").lower() == "gpu"]

        self._log("debug", f"Obtained experimental Devices: {experi}")

        all_devices = experi + self._supported_devices
        all_devices = all_devices if all_devices else self._get_fallback_devices()  # Use CPU

        self._log("debug", f"Obtained all Devices: {all_devices}")
        return all_devices

    def _get_fallback_devices(self) -> List[plaidml._DeviceConfig]:
        """ Called if a GPU has not been discovered. Return any devices we can run on.

        Returns
        -------
        list:
            The :class:`pladml._DeviceConfig` fallaback objects that PlaidML has discovered.
        """
        # Try get a supported device
        experimental_setting = plaidml.settings.experimental
        plaidml.settings.experimental = False
        devices = plaidml.devices(self._ctx, limit=100, return_all=True)[0]

        # Try get any device
        if not devices:
            plaidml.settings.experimental = True
            devices = plaidml.devices(self._ctx, limit=100, return_all=True)[0]

        plaidml.settings.experimental = experimental_setting

        if not devices:
            raise RuntimeError("No valid devices could be found for plaidML.")

        self._log("warning", f"PlaidML could not find a GPU. Falling back to: "
                  f"{[d.id.decode('utf-8', errors='replace') for d in devices]}")
        return devices

    def _get_device_details(self) -> List[dict]:
        """ Obtain the device details for all connected AMD GPUS.

        Returns
        -------
        list
            The `dict` device detail for all GPUs that PlaidML has discovered.
        """
        details = []
        for dev in self._all_devices:
            if dev.details:
                details.append(json.loads(dev.details.decode("utf-8", errors="replace")))
            else:
                details.append(dict(vendor=dev.id.decode("utf-8", errors="replace"),
                                    name=dev.description.decode("utf-8", errors="replace"),
                                    globalMemSize=4 * 1024 * 1024 * 1024))  # 4GB dummy ram
        self._log("debug", f"Obtained Device details: {details}")
        return details

    def _select_device(self) -> None:
        """
        If the plaidml user configuration settings exist, then set the default GPU from the
        settings file, Otherwise set the GPU to be the one with most VRAM. """
        if os.path.exists(plaidml.settings.user_settings):  # pylint:disable=no-member
            self._log("debug", "Setting PlaidML devices from user_settings")
        else:
            self._select_largest_gpu()

    def _select_largest_gpu(self) -> None:
        """ Set the default GPU to be a supported device with the most available VRAM. If no
        supported device is available, then set the GPU to be an experimental device with the
        most VRAM available. """
        category = "supported" if self._supported_devices else "experimental"
        self._log("debug", f"Obtaining largest {category} device")

        indices = getattr(self, f"_{category}_indices")
        if not indices:
            self._log("error", "Failed to automatically detect your GPU.")
            self._log("error", "Please run `plaidml-setup` to set up your GPU.")
            sys.exit(1)

        max_vram = max(self._all_vram[idx] for idx in indices)
        self._log("debug", f"Max VRAM: {max_vram}")

        gpu_idx = min(idx for idx, vram in enumerate(self._all_vram)
                      if vram == max_vram and idx in indices)
        self._log("debug", f"GPU IDX: {gpu_idx}")

        selected_gpu = self._plaid_ids[gpu_idx]
        self._log("info", f"Setting GPU to largest available {category} device. If you want to "
                          "override this selection, run `plaidml-setup` from the command line.")

        plaidml.settings.experimental = category == "experimental"
        plaidml.settings.device_ids = [selected_gpu]

    def _get_device_count(self) -> int:
        """ Detect the number of AMD GPUs available from PlaidML.

        Returns
        -------
        int
            The total number of AMD GPUs available
        """
        retval = len(self._all_devices)
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_active_devices(self) -> List[int]:
        """ Obtain the indices of active GPUs (those that have not been explicitly excluded by
        PlaidML or explicitly excluded in the command line arguments).

        Returns
        -------
        list
            The list of device indices that are available for Faceswap to use
        """
        devices = [idx for idx, d_id in enumerate(self._plaid_ids)
                   if d_id in plaidml.settings.device_ids and idx not in _EXCLUDE_DEVICES]
        self._log("debug", f"Active GPU Devices: {devices}")
        return devices

    def _get_handles(self) -> list:
        """ AMD Doesn't really use device handles, so we just return the all devices list

        Returns
        -------
        list
            The list of all AMD discovered GPUs
        """
        handles = self._all_devices
        self._log("debug", f"AMD GPU Handles found: {handles}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the AMD driver version currently in use.

        Returns
        -------
        str
            The current AMD GPU driver versions
        """
        drivers = "|".join([device.get("driverVersion", "No Driver Found")
                            for device in self._device_details])
        self._log("debug", f"GPU Drivers: {drivers}")
        return drivers

    def _get_device_names(self) -> List[str]:
        """ Obtain the list of names of connected AMD GPUs as identified in :attr:`_handles`.

        Returns
        -------
        list
            The list of connected Nvidia GPU names
        """
        names = self.names
        self._log("debug", f"GPU Devices: {names}")
        return names

    def _get_vram(self) -> List[int]:
        """ Obtain the VRAM in Megabytes for each connected AMD GPU as identified in
        :attr:`_handles`.

        Returns
        -------
        list
            The VRAM in Megabytes for each connected Nvidia GPU
        """
        vram = self._all_vram
        self._log("debug", f"GPU VRAM: {vram}")
        return vram

    def _get_free_vram(self) -> List[int]:
        """ Obtain the amount of VRAM that is available, in Megabytes, for each connected AMD
        GPU.

        Notes
        -----
        There is no useful way to get free VRAM on PlaidML. OpenCL loads and unloads VRAM as
        required, so this returns the total memory available per card for AMD GPUs, which is
        not particularly useful.

        Returns
        -------
        list
             List of `float`s containing the amount of VRAM available, in Megabytes, for each
             connected GPU as corresponding to the values in :attr:`_handles
        """
        vram = self._all_vram
        self._log("debug", f"GPU VRAM free: {vram}")
        return vram
