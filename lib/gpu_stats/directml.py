#!/usr/bin/env python3
""" Collects and returns Information on DirectX 12 hardware devices for DirectML. """
from __future__ import annotations
import os
import sys
import typing as T
assert sys.platform == "win32"

import ctypes
from ctypes import POINTER, Structure, windll
from dataclasses import dataclass
from enum import Enum, IntEnum

from comtypes import COMError, IUnknown, GUID, STDMETHOD, HRESULT  # pylint:disable=import-error

from ._base import _GPUStats

if T.TYPE_CHECKING:
    from collections.abc import Callable

# Monkey patch default ctypes.c_uint32 value to Enum ctypes property for easier tracking of types
# We can't just subclass as the attribute will be assumed to be part of the Enumeration, so we
# attach it directly and suck up the typing errors.
setattr(Enum, "ctype", ctypes.c_uint32)


#############################
# CTYPES SUPPORTING OBJECTS #
#############################
# GUIDs
@dataclass
class LookupGUID:
    """ GUIDs that are required for creating COM objects which are used and discarded.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12device2
    """
    IDXGIDevice = GUID("{54ec77fa-1377-44e6-8c32-88fd5f44c84c}")
    ID3D12Device = GUID("{189819f1-1db6-4b57-be54-1821339b85f7}")


# ENUMS
class DXGIGpuPreference(IntEnum):
    """ The preference of GPU for the app to run on.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_6/ne-dxgi1_6-dxgi_gpu_preference
    """
    DXGI_GPU_PREFERENCE_UNSPECIFIED = 0
    DXGI_GPU_PREFERENCE_MINIMUM_POWER = 1
    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE = 2


class DXGIAdapterFlag(IntEnum):
    """ Identifies the type of DXGI adapter.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi/ne-dxgi-dxgi_adapter_flag
    """
    DXGI_ADAPTER_FLAG_NONE = 0
    DXGI_ADAPTER_FLAG_REMOTE = 1
    DXGI_ADAPTER_FLAG_SOFTWARE = 2
    DXGI_ADAPTER_FLAG_FORCE_DWORD = 0xffffffff


class DXGIMemorySegmentGroup(IntEnum):
    """ Constants that specify an adapter's memory segment grouping.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ne-dxgi1_4-dxgi_memory_segment_group
    """
    DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
    DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL = 1


class D3DFeatureLevel(Enum):
    """ Describes the set of features targeted by a Direct3D device.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/d3dcommon/ne-d3dcommon-d3d_feature_level
    """
    D3D_FEATURE_LEVEL_1_0_CORE = 0x1000
    D3D_FEATURE_LEVEL_9_1 = 0x9100
    D3D_FEATURE_LEVEL_9_2 = 0x9200
    D3D_FEATURE_LEVEL_9_3 = 0x9300
    D3D_FEATURE_LEVEL_10_0 = 0xa000
    D3D_FEATURE_LEVEL_10_1 = 0xa100
    D3D_FEATURE_LEVEL_11_0 = 0xb000
    D3D_FEATURE_LEVEL_11_1 = 0xb100
    D3D_FEATURE_LEVEL_12_0 = 0xc000
    D3D_FEATURE_LEVEL_12_1 = 0xc100
    D3D_FEATURE_LEVEL_12_2 = 0xc200


class VendorID(Enum):
    """ DirectX VendorID Enum """
    AMD = 0x1002
    NVIDIA = 0x10DE
    MICROSOFT = 0x1414
    QUALCOMM = 0x4D4F4351
    INTEL = 0x8086


# STRUCTS
class StructureRepr(Structure):
    """ Override the standard structure class to add a useful __repr__ for logging """
    def __repr__(self) -> str:
        """ Output the class name and the structure contents """
        content = ["=".join([field[0], str(getattr(self, field[0]))])
                   for field in self._fields_]
        if self.__dict__:  # Add manually added parameters
            content.extend("=".join([key, str(val)]) for key, val in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(content)})"


class LUID(StructureRepr):  # pylint:disable=too-few-public-methods
    """ Local Identifier for an adaptor

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-luid """
    _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]


class DriverVersion(StructureRepr):  # pylint:disable=too-few-public-methods
    """ Stucture (based off LARGE_INTEGER) to hold the driver version

    Reference
    ---------
    https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-large_integer-r1"""
    _fields_ = [("parts_a", ctypes.c_uint16),
                ("parts_b", ctypes.c_uint16),
                ("parts_c", ctypes.c_uint16),
                ("parts_d", ctypes.c_uint16)]


class DXGIAdapterDesc1(StructureRepr):  # pylint:disable=too-few-public-methods
    """ Describes an adapter (or video card) using DXGI 1.1

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi/ns-dxgi-DXGIAdapterDesc1 """
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", LUID),
        ("Flags", DXGIAdapterFlag.ctype)]  # type:ignore[attr-defined] # pylint:disable=no-member


class DXGIQueryVideoMemoryInfo(StructureRepr):  # pylint:disable=too-few-public-methods
    """ Describes the current video memory budgeting parameters.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/ns-dxgi1_4-dxgi_query_video_memory_info
    """
    _fields_ = [("Budget", ctypes.c_uint64),
                ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64),
                ("CurrentReservation", ctypes.c_uint64)]


# COM OBjects
class IDXObject(IUnknown):  # pylint:disable=too-few-public-methods
    """ Base interface for all DXGI objects.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nn-dxgi-idxgiobject
    """
    _iid_ = GUID("{aec22fb8-76f3-4639-9be0-28eb43a67a2e}")
    _methods_ = [STDMETHOD(HRESULT, "SetPrivateData",
                           [GUID, ctypes.c_uint, POINTER(ctypes.c_void_p)]),
                 STDMETHOD(HRESULT, "SetPrivateDataInterface", [GUID, POINTER(IUnknown)]),
                 STDMETHOD(HRESULT, "GetPrivateData",
                           [GUID, POINTER(ctypes.c_uint), POINTER(ctypes.c_void_p)]),
                 STDMETHOD(HRESULT, "GetParent", [GUID, POINTER(POINTER(ctypes.c_void_p))])]


class IDXGIFactory6(IDXObject):  # pylint:disable=too-few-public-methods
    """ Implements methods for generating DXGI objects

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nn-dxgi-idxgifactory
    """
    _iid_ = GUID("{c1b6694f-ff09-44a9-b03c-77900a0a1d17}")

    _methods_ = [STDMETHOD(HRESULT, "EnumAdapters"),  # IDXGIFactory
                 STDMETHOD(HRESULT, "MakeWindowAssociation"),
                 STDMETHOD(HRESULT, "GetWindowAssociation"),
                 STDMETHOD(HRESULT, "CreateSwapChain"),
                 STDMETHOD(HRESULT, "CreateSoftwareAdapter"),
                 STDMETHOD(HRESULT, "EnumAdapters1"),  # IDXGIFactory1
                 STDMETHOD(ctypes.c_bool, "IsCurrent"),
                 STDMETHOD(ctypes.c_bool, "IsWindowedStereoEnabled"),  # IDXGIFactory2
                 STDMETHOD(HRESULT, "CreateSwapChainForHwnd"),
                 STDMETHOD(HRESULT, "CreateSwapChainForCoreWindow"),
                 STDMETHOD(HRESULT, "GetSharedResourceAdapterLuid"),
                 STDMETHOD(HRESULT, "RegisterStereoStatusWindow"),
                 STDMETHOD(HRESULT, "RegisterStereoStatusEvent"),
                 STDMETHOD(None, "UnregisterStereoStatus"),
                 STDMETHOD(HRESULT, "RegisterOcclusionStatusWindow"),
                 STDMETHOD(HRESULT, "RegisterOcclusionStatusEvent"),
                 STDMETHOD(None, "UnregisterOcclusionStatus"),
                 STDMETHOD(HRESULT, "CreateSwapChainForComposition"),
                 STDMETHOD(ctypes.c_uint, "GetCreationFlags"),  # IDXGIFactory3
                 STDMETHOD(HRESULT, "EnumAdapterByLuid",  # IDXGIFactory4
                           [LUID, GUID, POINTER(POINTER(ctypes.c_void_p))]),
                 STDMETHOD(HRESULT, "EnumWarpAdapter"),
                 STDMETHOD(HRESULT, "CheckFeatureSupport"),  # IDXGIFactory5
                 STDMETHOD(HRESULT,  # IDXGIFactory6
                           "EnumAdapterByGpuPreference",
                           [ctypes.c_uint,
                            DXGIGpuPreference.ctype,  # type:ignore[attr-defined] # pylint:disable=no-member  # noqa:E501
                            GUID,
                            POINTER(ctypes.c_void_p)])]


class IDXGIAdapter3(IDXObject):  # pylint:disable=too-few-public-methods
    """ Represents a display sub-system (including one or more GPU's, DACs and video memory).

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nn-dxgi1_4-idxgiadapter3
    """
    _iid_ = GUID("{645967a4-1392-4310-a798-8053ce3e93fd}")
    _methods_ = [STDMETHOD(HRESULT, "EnumOutputs"),  # v1.0 Methods
                 STDMETHOD(HRESULT, "GetDesc"),
                 STDMETHOD(HRESULT, "CheckInterfaceSupport",  # v1.1 Methods
                           [GUID, POINTER(DriverVersion)]),
                 STDMETHOD(HRESULT, "GetDesc1", [POINTER(DXGIAdapterDesc1)]),
                 STDMETHOD(HRESULT, "GetDesc2"),  # v1.2 Methods
                 STDMETHOD(HRESULT,    # v1.3 Methods
                           "RegisterHardwareContentProtectionTeardownStatusEvent"),
                 STDMETHOD(None, "UnregisterHardwareContentProtectionTeardownStatus"),
                 STDMETHOD(HRESULT,
                           "QueryVideoMemoryInfo",
                           [ctypes.c_uint,
                            DXGIMemorySegmentGroup.ctype,  # type:ignore[attr-defined] # pylint:disable=no-member  # noqa:E501
                            POINTER(DXGIQueryVideoMemoryInfo)]),
                 STDMETHOD(HRESULT, "SetVideoMemoryReservation"),
                 STDMETHOD(HRESULT, "RegisterVideoMemoryBudgetChangeNotificationEvent"),
                 STDMETHOD(None, "UnregisterVideoMemoryBudgetChangeNotification")]


###########################
# PYTHON COLLATED OBJECTS #
###########################
@dataclass
class Device:
    """ Holds information about a device attached to an adapter.

    Parameters
    ----------
    description: :class:`DXGIAdapterDesc1`
        The information returned from DXGI.dll about the device
    driver_version: str
        The driver version of the device
    local_mem: :class:`DXGIQueryVideoMemoryInfo`
        The amount of local memory currently available
    non_local_mem: :class:`DXGIQueryVideoMemoryInfo`
        The amount of non-local memory currently available
    is_d3d12: bool
        ``True`` if the device supports DirectX12
    is_compute_only: bool
        ``True`` if the device is only compute (no graphics)
    """
    description: DXGIAdapterDesc1
    driver_version: str
    local_mem: DXGIQueryVideoMemoryInfo
    non_local_mem: DXGIQueryVideoMemoryInfo
    is_d3d12: bool
    is_compute_only: bool = False

    @property
    def is_software_adapter(self) -> bool:
        """ bool: ``True`` if this is a software adapter. """
        return self.description.Flags == DXGIAdapterFlag.DXGI_ADAPTER_FLAG_SOFTWARE.value

    @property
    def is_valid(self) -> bool:
        """ bool: ``True`` if this adapter is a hardware adaptor and is not the basic renderer """
        if self.is_software_adapter:
            return False

        if (self.description.VendorId == VendorID.MICROSOFT.value and
                self.description.DeviceId == 0x8c):
            return False

        return True


class Adapters():  # pylint:disable=too-few-public-methods
    """ Wrapper to obtain connected DirectX Graphics interface adapters from Windows

    Parameters
    ----------
    log_func: :func:`~lib.gpu_stats._base._log`
        The logging function to use from the parent GPUStats class
    """
    def __init__(self, log_func: Callable[[str, str], None]) -> None:
        self._log = log_func
        self._log("debug", f"Initializing {self.__class__.__name__}: (log_func: {log_func})")

        self._factory = self._get_factory()
        self._adapters = self._get_adapters()
        self._devices = self._process_adapters()

        self._valid_adaptors: list[Device] = []
        self._log("debug", f"Initialized {self.__class__.__name__}")

    def _get_factory(self) -> ctypes._Pointer:
        """ Get a DXGI 1.1 Factory object

        Reference
        ---------
        https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-createdxgifactory1

        Returns
        -------
        :class:`ctypes._Pointer`
            A pointer to a :class:`IDXGIFactory6` COM instance
        """
        factory_func = windll.dxgi.CreateDXGIFactory
        factory_func.argtypes = (GUID, POINTER(ctypes.c_void_p))
        factory_func.restype = HRESULT
        handle = ctypes.c_void_p(0)
        factory_func(IDXGIFactory6._iid_,  ctypes.byref(handle))  # pylint:disable=protected-access
        retval = ctypes.POINTER(IDXGIFactory6)(T.cast(IDXGIFactory6, handle.value))
        self._log("debug", f"factory: {retval}")
        return retval

    @property
    def valid_adapters(self) -> list[Device]:
        """ list[:class:`Device`]: DirectX 12 compatible hardware :class:`Device` objects """
        if self._valid_adaptors:
            return self._valid_adaptors

        for device in self._devices:
            if not device.is_valid:
                # Sorted by most performant so everything after first basic adapter is skipped
                break
            if not device.is_d3d12:
                continue
            self._valid_adaptors.append(device)
        self._log("debug", f"valid_adaptors: {self._valid_adaptors}")
        return self._valid_adaptors

    def _get_adapters(self) -> list[ctypes._Pointer]:
        """ Obtain DirectX 12 supporting hardware adapter objects and add a Device class for
        obtaining details

        Returns
        -------
        list
            List of :class:`ctypes._Pointer` objects
        """
        idx = 0
        retval = []
        while True:
            try:
                handle = ctypes.c_void_p(0)
                success = self._factory.EnumAdapterByGpuPreference(  # type:ignore[attr-defined]
                    idx,
                    DXGIGpuPreference.DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE.value,
                    IDXGIAdapter3._iid_,  # pylint:disable=protected-access
                    ctypes.byref(handle))
                if success != 0:
                    raise AttributeError("Error calling EnumAdapterByGpuPreference. Result: "
                                         f"{hex(ctypes.c_ulong(success).value)}")
                adapter = POINTER(IDXGIAdapter3)(T.cast(IDXGIAdapter3, handle.value))
                self._log("debug", f"found adapter: {adapter}")
                retval.append(adapter)
            except COMError as err:
                err_code = hex(ctypes.c_ulong(err.hresult).value)  # pylint:disable=no-member
                self._log(
                    "debug",
                    "COM Error. Breaking: "
                    f"{err.text}({err_code})")  # pylint:disable=no-member
                break
            finally:
                idx += 1

        self._log("debug", f"adapters: {retval}")
        return retval

    def _query_adapter(self, func: Callable[[T.Any], T.Any], *args: T.Any) -> None:
        """ Query an adapter function, logging if the HRESULT is not a success

        Parameters
        ----------
        func: Callable[[Any], Any]
            The adaptor function to call
        args: Any
            The arguments to pass to the adaptor function
        """
        check = func(*args)
        if check:
            self._log("debug", f"Failed HRESULT for func {func}({args}): "
                               f"{hex(ctypes.c_ulong(check).value)}")

    def _test_d3d12(self, adapter: ctypes._Pointer) -> bool:
        """ Test whether the given adapter supports DirectX 12

        Parameters
        ----------
        adapter: :class:`ctypes._Pointer`
            A pointer to an adapter instance

        Returns
        -------
        bool
            ``True`` if the given adapter supports DirectX 12
        """
        factory_func = windll.d3d12.D3D12CreateDevice
        factory_func.argtypes = (
            POINTER(IUnknown),
            D3DFeatureLevel.ctype,  # type:ignore[attr-defined] # pylint:disable=no-member
            GUID,
            POINTER(ctypes.c_void_p))
        handle = ctypes.c_void_p(0)
        factory_func.restype = HRESULT
        success = factory_func(adapter,
                               D3DFeatureLevel.D3D_FEATURE_LEVEL_11_0.value,
                               LookupGUID.ID3D12Device,
                               ctypes.byref(handle))
        return success in (0, 1)

    def _process_adapters(self) -> list[Device]:
        """ Process the adapters to add discovered information.

        Returns
        -------
        list[:class:`Device`]
            List of device of objects found in the adapters
        """
        retval = []
        for adapter in self._adapters:
            # Description
            desc = DXGIAdapterDesc1()
            self._query_adapter(adapter.GetDesc1, ctypes.byref(desc))  # type:ignore[attr-defined]

            # Driver Version
            driver = DriverVersion()
            self._query_adapter(adapter.CheckInterfaceSupport,  # type:ignore[attr-defined]
                                LookupGUID.IDXGIDevice,
                                ctypes.byref(driver))
            driver_version = f"{driver.parts_d}.{driver.parts_c}.{driver.parts_b}.{driver.parts_a}"

            # Current Memory
            local_mem = DXGIQueryVideoMemoryInfo()
            self._query_adapter(adapter.QueryVideoMemoryInfo,  # type:ignore[attr-defined]
                                0,
                                DXGIMemorySegmentGroup.DXGI_MEMORY_SEGMENT_GROUP_LOCAL.value,
                                local_mem)
            non_local_mem = DXGIQueryVideoMemoryInfo()
            self._query_adapter(
                adapter.QueryVideoMemoryInfo,  # type:ignore[attr-defined]
                0,
                DXGIMemorySegmentGroup.DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL.value,
                non_local_mem)

            # is_d3d12
            is_d3d12 = self._test_d3d12(adapter)

            retval.append(Device(desc, driver_version, local_mem, non_local_mem, is_d3d12))

        return retval


class DirectML(_GPUStats):
    """ Holds information and statistics about GPUs connected using Windows API

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
        self._devices: list[Device] = []
        super().__init__(log=log)

    @property
    def _all_vram(self) -> list[int]:
        """ list: The VRAM of each GPU device that the DX API has discovered. """
        return [int(device.description.DedicatedVideoMemory / (1024 * 1024))
                for device in self._devices]

    @property
    def names(self) -> list[str]:
        """ list: The name of each GPU device that the DX API has discovered. """
        return [device.description.Description for device in self._devices]

    def _get_active_devices(self) -> list[int]:
        """ Obtain the indices of active GPUs (those that have not been explicitly excluded by
        DML_VISIBLE_DEVICES environment variable or explicitly excluded in the command line
        arguments).

        Returns
        -------
        list
            The list of device indices that are available for Faceswap to use
        """
        devices = super()._get_active_devices()
        env_devices = os.environ.get("DML_VISIBLE_DEVICES")
        if env_devices:
            new_devices = [int(i) for i in env_devices.split(",")]
            devices = [idx for idx in devices if idx in new_devices]
        self._log("debug", f"Active GPU Devices: {devices}")
        return devices

    def _get_devices(self) -> list[Device]:
        """ Obtain all detected DX API devices.

        Returns
        -------
        list
            The :class:`~dx_lib.Device` objects for GPUs that the DX API has discovered.
        """
        adapters = Adapters(log_func=self._log)
        devices = adapters.valid_adapters
        self._log("debug", f"Obtained Devices: {devices}")
        return devices

    def _initialize(self) -> None:
        """ Initialize DX Core for DirectML backend.

        If :attr:`_is_initialized` is ``True`` then this function just returns performing no
        action.

        if ``False`` then DirectML is setup, if not already, and GPU information is extracted
        from the DirectML context.
        """
        if self._is_initialized:
            return
        self._log("debug", "Initializing Win DX API for DirectML.")
        self._devices = self._get_devices()
        super()._initialize()

    def _get_device_count(self) -> int:
        """ Detect the number of GPUs available from the DX API.

        Returns
        -------
        int
            The total number of GPUs available
        """
        retval = len(self._devices)
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_handles(self) -> list:
        """ The DX API doesn't really use device handles, so we just return the all devices list

        Returns
        -------
        list
            The list of all discovered GPUs
        """
        handles = self._devices
        self._log("debug", f"DirectML GPU Handles found: {handles}")
        return handles

    def _get_driver(self) -> str:
        """ Obtain the driver versions currently in use.

        Returns
        -------
        str
            The current DirectX 12 GPU driver versions
        """
        drivers = "|".join([device.driver_version if device.driver_version else "No Driver Found"
                            for device in self._devices])
        self._log("debug", f"GPU Drivers: {drivers}")
        return drivers

    def _get_device_names(self) -> list[str]:
        """ Obtain the list of names of connected GPUs as identified in :attr:`_handles`.

        Returns
        -------
        list
            The list of connected Nvidia GPU names
        """
        names = self.names
        self._log("debug", f"GPU Devices: {names}")
        return names

    def _get_vram(self) -> list[int]:
        """ Obtain the VRAM in Megabytes for each connected DirectML GPU as identified in
        :attr:`_handles`.

        Returns
        -------
        list
            The VRAM in Megabytes for each connected Nvidia GPU
        """
        vram = self._all_vram
        self._log("debug", f"GPU VRAM: {vram}")
        return vram

    def _get_free_vram(self) -> list[int]:
        """ Obtain the amount of VRAM that is available, in Megabytes, for each connected DirectX
        12 supporting GPU.

        Returns
        -------
        list
             List of `float`s containing the amount of VRAM available, in Megabytes, for each
             connected GPU as corresponding to the values in :attr:`_handles
        """
        vram = [int(device.local_mem.Budget / (1024 * 1024)) for device in self._devices]
        self._log("debug", f"GPU VRAM free: {vram}")
        return vram
