#!/usr/bin/env python3
""" Collects and returns Information about connected AMD GPUs for ROCm using sysfs and from
modinfo

As no ROCm compatible hardware was available for testing, this just returns information on all AMD
GPUs discovered on the system regardless of ROCm compatibility.

It is a good starting point but may need to be refined over time
"""
import os
import re
from subprocess import run

from ._base import _GPUStats

_DEVICE_LOOKUP = {  # ref: https://gist.github.com/roalercon/51f13a387f3754615cce
    int("0x130F", 0): "AMD Radeon(TM) R7 Graphics",
    int("0x1313", 0): "AMD Radeon(TM) R7 Graphics",
    int("0x1316", 0): "AMD Radeon(TM) R5 Graphics",
    int("0x6600", 0): "AMD Radeon HD 8600/8700M",
    int("0x6601", 0): "AMD Radeon (TM) HD 8500M/8700M",
    int("0x6604", 0): "AMD Radeon R7 M265 Series",
    int("0x6605", 0): "AMD Radeon R7 M260 Series",
    int("0x6606", 0): "AMD Radeon HD 8790M",
    int("0x6607", 0): "AMD Radeon (TM) HD8530M",
    int("0x6610", 0): "AMD Radeon HD 8670 Graphics",
    int("0x6611", 0): "AMD Radeon HD 8570 Graphics",
    int("0x6613", 0): "AMD Radeon R7 200 Series",
    int("0x6640", 0): "AMD Radeon HD 8950",
    int("0x6658", 0): "AMD Radeon R7 200 Series",
    int("0x665C", 0): "AMD Radeon HD 7700 Series",
    int("0x665D", 0): "AMD Radeon R7 200 Series",
    int("0x6660", 0): "AMD Radeon HD 8600M Series",
    int("0x6663", 0): "AMD Radeon HD 8500M Series",
    int("0x6664", 0): "AMD Radeon R5 M200 Series",
    int("0x6665", 0): "AMD Radeon R5 M230 Series",
    int("0x6667", 0): "AMD Radeon R5 M200 Series",
    int("0x666F", 0): "AMD Radeon HD 8500M",
    int("0x6704", 0): "AMD FirePro V7900 (FireGL V)",
    int("0x6707", 0): "AMD FirePro V5900 (FireGL V)",
    int("0x6718", 0): "AMD Radeon HD 6900 Series",
    int("0x6719", 0): "AMD Radeon HD 6900 Series",
    int("0x671D", 0): "AMD Radeon HD 6900 Series",
    int("0x671F", 0): "AMD Radeon HD 6900 Series",
    int("0x6720", 0): "AMD Radeon HD 6900M Series",
    int("0x6738", 0): "AMD Radeon HD 6800 Series",
    int("0x6739", 0): "AMD Radeon HD 6800 Series",
    int("0x673E", 0): "AMD Radeon HD 6700 Series",
    int("0x6740", 0): "AMD Radeon HD 6700M Series",
    int("0x6741", 0): "AMD Radeon 6600M and 6700M Series",
    int("0x6742", 0): "AMD Radeon HD 5570",
    int("0x6743", 0): "AMD Radeon E6760",
    int("0x6749", 0): "AMD FirePro V4900 (FireGL V)",
    int("0x674A", 0): "AMD FirePro V3900 (ATI FireGL)",
    int("0x6750", 0): "AMD Radeon HD 6500 series",
    int("0x6751", 0): "AMD Radeon HD 7600A Series",
    int("0x6758", 0): "AMD Radeon HD 6670",
    int("0x6759", 0): "AMD Radeon HD 6570 Graphics",
    int("0x675B", 0): "AMD Radeon HD 7600 Series",
    int("0x675D", 0): "AMD Radeon HD 7500 Series",
    int("0x675F", 0): "AMD Radeon HD 5500 Series",
    int("0x6760", 0): "AMD Radeon HD 6400M Series",
    int("0x6761", 0): "AMD Radeon HD 6430M",
    int("0x6763", 0): "AMD Radeon E6460",
    int("0x6770", 0): "AMD Radeon HD 6400 Series",
    int("0x6771", 0): "AMD Radeon R5 235X",
    int("0x6772", 0): "AMD Radeon HD 7400A Series",
    int("0x6778", 0): "AMD Radeon HD 7000 series",
    int("0x6779", 0): "AMD Radeon HD 6450",
    int("0x677B", 0): "AMD Radeon HD 7400 Series",
    int("0x6780", 0): "AMD FirePro W9000 (FireGL V)",
    int("0x678A", 0): "AMD FirePro S10000 (FireGL V)",
    int("0x6798", 0): "AMD Radeon HD 7900 Series",
    int("0x679A", 0): "AMD Radeon HD 7900 Series",
    int("0x679B", 0): "AMD Radeon HD 7900 Series",
    int("0x679E", 0): "AMD Radeon HD 7800 Series",
    int("0x67B0", 0): "AMD Radeon R9 200 Series",
    int("0x67B1", 0): "AMD Radeon R9 200 Series",
    int("0x6800", 0): "AMD Radeon HD 7970M",
    int("0x6801", 0): "AMD Radeon(TM) HD8970M",
    int("0x6808", 0): "AMD FirePro S7000 (FireGL V)",
    int("0x6809", 0): "AMD FirePro R5000 (FireGL V)",
    int("0x6810", 0): "AMD Radeon R9 200 Series",
    int("0x6811", 0): "AMD Radeon R9 200 Series",
    int("0x6818", 0): "AMD Radeon HD 7800 Series",
    int("0x6819", 0): "AMD Radeon HD 7800 Series",
    int("0x6820", 0): "AMD Radeon HD 8800M Series",
    int("0x6821", 0): "AMD Radeon HD 8800M Series",
    int("0x6822", 0): "AMD Radeon E8860",
    int("0x6823", 0): "AMD Radeon HD 8800M Series",
    int("0x6825", 0): "AMD Radeon HD 7800M Series",
    int("0x6827", 0): "AMD Radeon HD 7800M Series",
    int("0x6828", 0): "AMD FirePro W600",
    int("0x682B", 0): "AMD Radeon HD 8800M Series",
    int("0x682D", 0): "AMD Radeon HD 7700M Series",
    int("0x682F", 0): "AMD Radeon HD 7700M Series",
    int("0x6835", 0): "AMD Radeon R7 Series / HD 9000 Series",
    int("0x6837", 0): "AMD Radeon HD 6570",
    int("0x683D", 0): "AMD Radeon HD 7700 Series",
    int("0x683F", 0): "AMD Radeon HD 7700 Series",
    int("0x6840", 0): "AMD Radeon HD 7600M Series",
    int("0x6841", 0): "AMD Radeon HD 7500M/7600M Series",
    int("0x6842", 0): "AMD Radeon HD 7000M Series",
    int("0x6843", 0): "AMD Radeon HD 7670M",
    int("0x6858", 0): "AMD Radeon HD 7400 Series",
    int("0x6859", 0): "AMD Radeon HD 7400 Series",
    int("0x6888", 0): "ATI FirePro V8800 (FireGL V)",
    int("0x6889", 0): "ATI FirePro V7800 (FireGL V)",
    int("0x688A", 0): "ATI FirePro V9800 (FireGL V)",
    int("0x688C", 0): "AMD FireStream 9370",
    int("0x688D", 0): "AMD FireStream 9350",
    int("0x6898", 0): "AMD Radeon HD 5800 Series",
    int("0x6899", 0): "AMD Radeon HD 5800 Series",
    int("0x689B", 0): "AMD Radeon HD 6800 Series",
    int("0x689C", 0): "AMD Radeon HD 5900 Series",
    int("0x689E", 0): "AMD Radeon HD 5800 Series",
    int("0x68A0", 0): "AMD Mobility Radeon HD 5800 Series",
    int("0x68A1", 0): "AMD Mobility Radeon HD 5800 Series",
    int("0x68A8", 0): "AMD Radeon HD 6800M Series",
    int("0x68A9", 0): "ATI FirePro V5800 (FireGL V)",
    int("0x68B8", 0): "AMD Radeon HD 5700 Series",
    int("0x68B9", 0): "AMD Radeon HD 5600/5700",
    int("0x68BA", 0): "AMD Radeon HD 6700 Series",
    int("0x68BE", 0): "AMD Radeon HD 5700 Series",
    int("0x68BF", 0): "AMD Radeon HD 6700 Green Edition",
    int("0x68C0", 0): "AMD Mobility Radeon HD 5000",
    int("0x68C1", 0): "AMD Mobility Radeon HD 5000 Series",
    int("0x68C7", 0): "AMD Mobility Radeon HD 5570",
    int("0x68C8", 0): "ATI FirePro V4800 (FireGL V)",
    int("0x68C9", 0): "ATI FirePro 3800 (FireGL) Graphics Adapter",
    int("0x68D8", 0): "AMD Radeon HD 5670",
    int("0x68D9", 0): "AMD Radeon HD 5570",
    int("0x68DA", 0): "AMD Radeon HD 5500 Series",
    int("0x68E0", 0): "AMD Mobility Radeon HD 5000 Series",
    int("0x68E1", 0): "AMD Mobility Radeon HD 5000 Series",
    int("0x68E4", 0): "AMD Radeon HD 5450",
    int("0x68E5", 0): "AMD Radeon HD 6300M Series",
    int("0x68F1", 0): "AMD FirePro 2460",
    int("0x68F2", 0): "AMD FirePro 2270 (ATI FireGL)",
    int("0x68F9", 0): "AMD Radeon HD 5450",
    int("0x68FA", 0): "AMD Radeon HD 7300 Series",
    int("0x9640", 0): "AMD Radeon HD 6550D",
    int("0x9641", 0): "AMD Radeon HD 6620G",
    int("0x9642", 0): "AMD Radeon HD 6370D",
    int("0x9643", 0): "AMD Radeon HD 6380G",
    int("0x9644", 0): "AMD Radeon HD 6410D",
    int("0x9645", 0): "AMD Radeon HD 6410D",
    int("0x9647", 0): "AMD Radeon HD 6520G",
    int("0x9648", 0): "AMD Radeon HD 6480G",
    int("0x9649", 0): "AMD Radeon(TM) HD 6480G",
    int("0x964A", 0): "AMD Radeon HD 6530D",
    int("0x9802", 0): "AMD Radeon HD 6310 Graphics",
    int("0x9803", 0): "AMD Radeon HD 6250 Graphics",
    int("0x9804", 0): "AMD Radeon HD 6250 Graphics",
    int("0x9805", 0): "AMD Radeon HD 6250 Graphics",
    int("0x9806", 0): "AMD Radeon HD 6320 Graphics",
    int("0x9807", 0): "AMD Radeon HD 6290 Graphics",
    int("0x9808", 0): "AMD Radeon HD 7340 Graphics",
    int("0x9809", 0): "AMD Radeon HD 7310 Graphics",
    int("0x980A", 0): "AMD Radeon HD 7290 Graphics",
    int("0x9830", 0): "AMD Radeon HD 8400",
    int("0x9831", 0): "AMD Radeon(TM) HD 8400E",
    int("0x9832", 0): "AMD Radeon HD 8330",
    int("0x9833", 0): "AMD Radeon(TM) HD 8330E",
    int("0x9834", 0): "AMD Radeon HD 8210",
    int("0x9835", 0): "AMD Radeon(TM) HD 8210E",
    int("0x9836", 0): "AMD Radeon HD 8280",
    int("0x9837", 0): "AMD Radeon(TM) HD 8280E",
    int("0x9838", 0): "AMD Radeon HD 8240",
    int("0x9839", 0): "AMD Radeon HD 8180",
    int("0x983D", 0): "AMD Radeon HD 8250",
    int("0x9900", 0): "AMD Radeon HD 7660G",
    int("0x9901", 0): "AMD Radeon HD 7660D",
    int("0x9903", 0): "AMD Radeon HD 7640G",
    int("0x9904", 0): "AMD Radeon HD 7560D",
    int("0x9906", 0): "AMD FirePro A300 Series (FireGL V) Graphics Adapter",
    int("0x9907", 0): "AMD Radeon HD 7620G",
    int("0x9908", 0): "AMD Radeon HD 7600G",
    int("0x990A", 0): "AMD Radeon HD 7500G",
    int("0x990B", 0): "AMD Radeon HD 8650G",
    int("0x990C", 0): "AMD Radeon HD 8670D",
    int("0x990D", 0): "AMD Radeon HD 8550G",
    int("0x990E", 0): "AMD Radeon HD 8570D",
    int("0x990F", 0): "AMD Radeon HD 8610G",
    int("0x9910", 0): "AMD Radeon HD 7660G",
    int("0x9913", 0): "AMD Radeon HD 7640G",
    int("0x9917", 0): "AMD Radeon HD 7620G",
    int("0x9918", 0): "AMD Radeon HD 7600G",
    int("0x9919", 0): "AMD Radeon HD 7500G",
    int("0x9990", 0): "AMD Radeon HD 7520G",
    int("0x9991", 0): "AMD Radeon HD 7540D",
    int("0x9992", 0): "AMD Radeon HD 7420G",
    int("0x9993", 0): "AMD Radeon HD 7480D",
    int("0x9994", 0): "AMD Radeon HD 7400G",
    int("0x9995", 0): "AMD Radeon HD 8450G",
    int("0x9996", 0): "AMD Radeon HD 8470D",
    int("0x9997", 0): "AMD Radeon HD 8350G",
    int("0x9998", 0): "AMD Radeon HD 8370D",
    int("0x9999", 0): "AMD Radeon HD 8510G",
    int("0x999A", 0): "AMD Radeon HD 8410G",
    int("0x999B", 0): "AMD Radeon HD 8310G",
    int("0x999C", 0): "AMD Radeon HD 8650D",
    int("0x999D", 0): "AMD Radeon HD 8550D",
    int("0x99A0", 0): "AMD Radeon HD 7520G",
    int("0x99A2", 0): "AMD Radeon HD 7420G",
    int("0x99A4", 0): "AMD Radeon HD 7400G"}


class ROCm(_GPUStats):
    """ Holds information and statistics about GPUs connected using sysfs

    Parameters
    ----------
    log: bool, optional
        Whether the class should output information to the logger. There may be occasions where the
        logger has not yet been set up when this class is queried. Attempting to log in these
        instances will raise an error. If GPU stats are being queried prior to the logger being
        available then this parameter should be set to ``False``. Otherwise set to ``True``.
        Default: ``True``
    """
    def __init__(self, log: bool = True) -> None:
        self._vendor_id = "0x1002"  # AMD VendorID
        self._sysfs_paths: list[str] = []
        super().__init__(log=log)

    def _from_sysfs_file(self, path: str) -> str:
        """ Obtain the value from a sysfs file. On permission error or file doesn't exist, log and
        return empty value

        Parameters
        ----------
        path: str
            The path to a sysfs file to obtain the value from

        Returns
        -------
        str
            The obtained value from the given path
        """
        if not os.path.isfile(path):
            self._log("debug", f"File '{path}' does not exist. Returning empty string")
            return ""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as sysfile:
                val = sysfile.read().strip()
        except PermissionError:
            self._log("debug", f"Permission error accessing file '{path}'. Returning empty string")
            val = ""
        return val

    def _get_sysfs_paths(self) -> list[str]:
        """ Obtain a list of sysfs paths to AMD branded GPUs connected to the system

        Returns
        -------
        list[str]
            List of full paths to the sysfs entries for connected AMD GPUs
        """
        base_dir = "/sys/class/drm/"

        retval: list[str] = []
        if not os.path.exists(base_dir):
            self._log("warning", f"sysfs not found at '{base_dir}'")
            return retval

        for folder in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder, "device")
            vendor_path = os.path.join(folder_path, "vendor")
            if not os.path.isdir(vendor_path) and not re.match(r"^card\d+$", folder):
                self._log("debug", f"skipping path '{folder_path}'")
                continue

            vendor_id = self._from_sysfs_file(vendor_path)
            if vendor_id != self._vendor_id:
                self._log("debug", f"Skipping non AMD Vendor '{vendor_id}' for device: '{folder}'")
                continue

            retval.append(folder_path)

        self._log("debug", f"sysfs AMD devices: {retval}")
        return retval

    def _initialize(self) -> None:
        """ Initialize sysfs for ROCm backend.

        If :attr:`_is_initialized` is ``True`` then this function just returns performing no
        action.

        if ``False`` then the location of AMD cards within sysfs is collected
        """
        if self._is_initialized:
            return
        self._log("debug", "Initializing sysfs for AMDGPU (ROCm).")
        self._sysfs_paths = self._get_sysfs_paths()
        super()._initialize()

    def _get_device_count(self) -> int:
        """ The number of AMD cards found in sysfs

        Returns
        -------
        int
            The total number of GPUs available
        """
        retval = len(self._sysfs_paths)
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_handles(self) -> list:
        """ The sysfs doesn't use device handles, so we just return the list of the sysfs locations
        per card

        Returns
        -------
        list
            The list of all discovered GPUs
        """
        handles = self._sysfs_paths
        self._log("debug", f"sysfs GPU Handles found: {handles}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the driver versions currently in use from modinfo

        Returns
        -------
        str
            The current AMDGPU driver versions
        """
        retval = ""
        cmd = ["modinfo", "amdgpu"]
        try:
            proc = run(cmd,
                       check=True,
                       timeout=5,
                       capture_output=True,
                       encoding="utf-8",
                       errors="ignore")
            for line in proc.stdout.split("\n"):
                if line.startswith("version:"):
                    retval = line.split()[-1]
                    break
        except Exception as err:  # pylint:disable=broad-except
            self._log("debug", f"Error reading modinfo: '{str(err)}'")

        self._log("debug", f"GPU Drivers: {retval}")
        return retval

    def _get_device_names(self) -> list[str]:
        """ Obtain the list of names of connected GPUs as identified in :attr:`_handles`.

        Returns
        -------
        list
            The list of connected AMD GPU names
        """
        retval = []
        for device in self._sysfs_paths:
            name = self._from_sysfs_file(os.path.join(device, "product_name"))
            number = self._from_sysfs_file(os.path.join(device, "product_number"))
            if name or number:  # product_name or product_number populated
                self._log("debug", f"Got name from product_name: '{name}', product_number: "
                                   f"'{number}'")
                retval.append(f"{name + ' ' if name else ''}{number}")
                continue

            device_id = self._from_sysfs_file(os.path.join(device, "device"))
            self._log("debug", f"Got device_id: '{device_id}'")

            if not device_id:  # Can't get device name
                retval.append("Not found")
                continue
            try:
                lookup = int(device_id, 0)
            except ValueError:
                retval.append(device_id)
                continue

            device_name = _DEVICE_LOOKUP.get(lookup, device_id)
            retval.append(device_name)

        self._log("debug", f"Device names: {retval}")
        return retval

    def _get_active_devices(self) -> list[int]:
        """ Obtain the indices of active GPUs (those that have not been explicitly excluded by
        HIP_VISIBLE_DEVICES environment variable or explicitly excluded in the command line
        arguments).

        Returns
        -------
        list
            The list of device indices that are available for Faceswap to use
        """
        devices = super()._get_active_devices()
        env_devices = os.environ.get("HIP_VISIBLE_DEVICES ")
        if env_devices:
            new_devices = [int(i) for i in env_devices.split(",")]
            devices = [idx for idx in devices if idx in new_devices]
        self._log("debug", f"Active GPU Devices: {devices}")
        return devices

    def _get_vram(self) -> list[int]:
        """ Obtain the VRAM in Megabytes for each connected AMD GPU as identified in
        :attr:`_handles`.

        Returns
        -------
        list
            The VRAM in Megabytes for each connected Nvidia GPU
        """
        retval = []
        for device in self._sysfs_paths:
            query = self._from_sysfs_file(os.path.join(device, "mem_info_vram_total"))
            try:
                vram = int(query)
            except ValueError:
                self._log("debug", f"Couldn't extract VRAM from string: '{query}'", )
                vram = 0
            retval.append(int(vram / (1024 * 1024)))

        self._log("debug", f"GPU VRAM: {retval}")
        return retval

    def _get_free_vram(self) -> list[int]:
        """ Obtain the amount of VRAM that is available, in Megabytes, for each connected AMD
        GPU.

        Returns
        -------
        list
             List of `float`s containing the amount of VRAM available, in Megabytes, for each
             connected GPU as corresponding to the values in :attr:`_handles
        """
        retval = []
        total_vram = self._get_vram()
        for device, vram in zip(self._sysfs_paths, total_vram):
            if not vram:
                retval.append(0)
                continue
            query = self._from_sysfs_file(os.path.join(device, "mem_info_vram_used"))
            try:
                used = int(query)
            except ValueError:
                self._log("debug", f"Couldn't extract used VRAM from string: '{query}'")
                used = 0

            retval.append(vram - int(used / (1024 * 1024)))
        self._log("debug", f"GPU VRAM free: {retval}")
        return retval
