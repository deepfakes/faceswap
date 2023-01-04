#!/usr/bin/env python3
""" Collects and returns Information on DirectX 12 hardware devices for DirectML. """
import os
import ctypes
from ctypes import POINTER, Structure, windll
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Callable, cast, List

from comtypes import IUnknown, GUID, STDMETHOD, HRESULT

from ._base import _GPUStats

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
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore-adapter-attribute-guids
    https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12device2
    """
    d3d11_graphics = GUID("{8c47866b-7583-450d-f0f0-6bada895af4b}")
    d3d12_graphics = GUID("{0c9ece4d-2f6e-4f01-8c96-e89e331b47b1}")
    d3d12_core_compute = GUID("{248e2800-a793-4724-abaa-23a6de1be090}")
    id3d12device = GUID("{189819f1-1db6-4b57-be54-1821339b85f7}")


# ENUMS
class DXCoreAdapterPreference(IntEnum):
    """ Constants that specify DXCore adapter preferences to be used as list-sorting criteria in
    :func:`IDXCoreAdapterList.Sort`

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/ne-dxcore_interface-dxcoreadapterpreference
    """
    # pylint:disable=invalid-name
    Hardware = 0
    MinimumPower = 1
    HighPerformance = 2


class DXCoreAdapterProperty(IntEnum):
    """ Defines constants that specify DXCore adapter properties. Use in
    :func:`IDXCoreAdapter.GetPropertySize` to retrieve the buffer size necessary to receive the
    value of the corresponding property. Use in :func:`IDXCoreAdapter.GetProperty` to retrieve the
    property's value

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/ne-dxcore_interface-dxcoreadapterproperty
    """
    # pylint:disable=invalid-name
    InstanceLuid = 0
    DriverVersion = 1
    DriverDescription = 2
    HardwareID = 3
    KmdModelVersion = 4
    ComputePreemptionGranularity = 5
    GraphicsPreemptionGranularity = 6
    DedicatedAdapterMemory = 7
    DedicatedSystemMemory = 8
    SharedSystemMemory = 9
    AcgCompatible = 10
    IsHardware = 11
    IsIntegrated = 12
    IsDetachable = 13


class DXCoreAdapterState(IntEnum):
    """ Specify kinds of DXCore adapter states. Pass one of these constants to
    :func:`IDXCoreAdapter.QueryState` to retrieve the adapter state item for that state.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/ne-dxcore_interface-dxcoreadapterstate
    """
    # pylint:disable=invalid-name
    IsDriverUpdateInProgress = 0
    AdapterMemoryBudget = 1


class DXCoreSegmentGroup(IntEnum):
    """ Constants that specify an adapter's memory segment grouping.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ne-dxcore_interface-dxcoresegmentgroup
    """
    # pylint:disable=invalid-name
    Local = 0
    NonLocal = 1


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
class StructureRepr(Structure):  # pylint:disable=too-few-public-methods
    """ Override the standard structure class to add a useful __repr__ for logging """
    def __repr__(self) -> str:
        """ Output the class name and the structure contents """
        content = ["=".join([field[0], str(getattr(self, field[0]))])
                   for field in self._fields_]
        if self.__dict__:  # Add manually added parameters
            content.extend("=".join([key, str(val)]) for key, val in self.__dict__.items())
        return f"{self.__class__.__name__}({', '.join(content)})"


class LUID(StructureRepr):  # pylint:disable=too-few-public-methods
    """Local Identifier for an adaptor

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-luid """
    _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]


class DXCoreHardwareID(StructureRepr):  # pylint:disable=too-few-public-methods
    """ Represents the PnP hardware ID parts for an adapter

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/ns-dxcore_interface-dxcorehardwareid
    """
    _fields_ = [("vendorID", ctypes.c_uint32),
                ("deviceID", ctypes.c_uint32),
                ("subSysID", ctypes.c_uint32),
                ("revision", ctypes.c_uint32)]


class DriverVersion(StructureRepr):  # pylint:disable=too-few-public-methods
    """ Stucture (based off LARGE_INTEGER) to hold the driver version

    Reference
    ---------
    https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-large_integer-r1"""
    _fields_ = [("parts_a", ctypes.c_uint16),
                ("parts_b", ctypes.c_uint16),
                ("parts_c", ctypes.c_uint16),
                ("parts_d", ctypes.c_uint16)]


class DXCoreAdapterMemoryBudget(StructureRepr):  # pylint:disable=too-few-public-methods
    """ Specifies the AdapterMemoryBudget adapter state, which retrieves or requests the adapter
    memory budget on the adapter.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/ns-dxcore_interface-dxcoreadaptermemorybudget
    """
    _fields_ = [("budget", ctypes.c_uint64),
                ("currentUsage", ctypes.c_uint64),
                ("availableForReservation", ctypes.c_uint64),
                ("currentReservation", ctypes.c_uint64)]


class DXCoreAdapterMemoryBudgetNodeSegmentGroup(StructureRepr):  # pylint:disable=too-few-public-methods  # noqa:E501
    """ Describes a memory segment group for an adapter.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/api/dxcore_interface/ns-dxcore_interface-dxcoreadaptermemorybudgetnodesegmentgroup
    """
    _fields_ = [("nodeIndex", ctypes.c_uint32),
                ("segmentGroup", DXCoreSegmentGroup.ctype)]  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501


# COM OBjects
class IDXCoreAdapter(IUnknown):
    """ Implements methods for retrieving details about an adapter item.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/nn-dxcore_interface-idxcoreadapter
    """
    _iid_ = GUID("{f0db4c7f-fe5a-42a2-bd62-f2a6cf6fc83e}")
    _methods_ = [STDMETHOD(ctypes.c_bool, "IsValid"),
                 STDMETHOD(ctypes.c_bool, "IsAttributeSupported", [GUID]),
                 STDMETHOD(ctypes.c_bool, "IsPropertySupported",
                           [DXCoreAdapterProperty.ctype]),  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                 STDMETHOD(HRESULT, "GetProperty",
                           [DXCoreAdapterProperty.ctype,    # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                            ctypes.c_size_t,
                            POINTER(ctypes.c_void_p)]),
                 STDMETHOD(HRESULT, "GetPropertySize",
                           [DXCoreAdapterProperty.ctype,  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                            POINTER(ctypes.c_size_t)]),
                 STDMETHOD(ctypes.c_bool, "IsQueryStateSupported",
                           [DXCoreAdapterState.ctype]),  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                 STDMETHOD(HRESULT, "QueryState",
                           [DXCoreAdapterState.ctype,  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                            ctypes.c_size_t,
                            POINTER(ctypes.c_void_p),
                            ctypes.c_size_t,
                            POINTER(ctypes.c_void_p)]),
                 STDMETHOD(ctypes.c_bool, "IsSetStateSupported",
                           [DXCoreAdapterState.ctype]),  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                 STDMETHOD(HRESULT, "SetState",
                           [DXCoreAdapterState.ctype,  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                            ctypes.c_size_t,
                            POINTER(ctypes.c_void_p),
                            ctypes.c_size_t,
                            POINTER(ctypes.c_void_p)]),
                 STDMETHOD(HRESULT, "GetFactory", [GUID, POINTER(ctypes.c_void_p)])]


class IDXCoreAdapterList(IUnknown):
    """ Implements methods for retrieving adapter items from a generated list, as well as details
    about the list.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/nn-dxcore_interface-idxcoreadapterlist
    """
    _iid_ = GUID("{526c7776-40e9-459b-b711-f32ad76dfc28}")
    _methods_ = [STDMETHOD(HRESULT, "GetAdapter",
                           [ctypes.c_uint32, GUID, POINTER(ctypes.c_void_p)]),
                 STDMETHOD(ctypes.c_uint32, "GetAdapterCount"),
                 STDMETHOD(ctypes.c_bool, "IsStale"),
                 STDMETHOD(HRESULT, "GetFactory", [GUID, POINTER(ctypes.c_void_p)]),
                 STDMETHOD(HRESULT, "Sort",
                           [ctypes.c_uint32,
                            POINTER(DXCoreAdapterPreference.ctype)]),  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                 STDMETHOD(ctypes.c_bool, "IsAdapterPreferenceSupported",
                           [DXCoreAdapterPreference.ctype])]  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501


class IDXCoreAdapterFactory(IUnknown):
    """ Implements methods for generating DXCore adapter enumeration objects, and for retrieving
    their details.

    Reference
    ---------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/nn-dxcore_interface-idxcoreadapterfactory
    """
    _iid_ = GUID("{78ee5945-c36e-4b13-a669-005dd11c0f06}")
    _methods_ = [STDMETHOD(HRESULT, "CreateAdapterList",
                           [ctypes.c_uint32, POINTER(GUID), GUID, POINTER(ctypes.c_void_p)]),
                 STDMETHOD(HRESULT, "GetAdapterByLuid", [LUID, GUID, POINTER(ctypes.c_void_p)]),
                 STDMETHOD(ctypes.c_bool, "IsNotificationTypeSupported", [ctypes.c_uint32]),
                 STDMETHOD(HRESULT, "RegisterEventNotification"),  # unbound
                 STDMETHOD(HRESULT, "UnregisterEventNotification", [ctypes.c_uint32])]


###########################
# PYTHON COLLATED OBJECTS #
###########################
@dataclass
class VRam:
    """ Object for holding information about VRAM amounts as bytes

    Parameters
    ----------
    dedicated_adapter_memory: int, optional
        The total amount dedicated adapter memory that is not shared with the CPU. Default: `0`
    dedicated_system_memory: int, optional
        The total amount of dedicated system memory that is not shared with the CPU. Default: `0`
    shared_system_memory: int, optional
        The amount of shared system memory. This is the maximum value of system memory that may be
        consumed by the adapter during operation. Default: `0`
    local_budget: int, optional
        The OS-provided adapter local memory budget. Default: `0`
    local_current_usage: int, optional
        The applicaton's current adapter local memory usage. Default: `0`
    local_available_for_reservation: int, optional
        The amount of adapter memory, that the application has available locally for reservation.
        Default: `0`
    local_current_reservation: int, optional
        The amount of adapter memory that is reserved locally by the application. Default: `0`
    non_local_budget: int, optional
        The OS-provided adapter non-local memory budget. Default: `0`
    non_local_current_usage: int, optional
        The applicaton's current adapter non-local memory usage. Default: `0`
    non_local_available_for_reservation: int, optional
        The amount of adapter memory, that the application has available non-locally for
        reservation. Default: `0`
    non_local_current_reservation: int, optional
        The amount of adapter memory that is reserved non-locally by the application. Default: `0`

    References
    ----------
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/ne-dxcore_interface-dxcoreadapterproperty
    https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/ns-dxcore_interface-dxcoreadaptermemorybudget
    """
    dedicated_adapter_memory: int = 0
    dedicated_system_memory: int = 0
    shared_system_memory: int = 0
    local_budget: int = 0
    local_current_usage: int = 0
    local_available_for_reservation: int = 0
    local_current_reservation: int = 0
    non_local_budget: int = 0
    non_local_current_usage: int = 0
    non_local_available_for_reservation: int = 0
    non_local_current_reservation: int = 0


class Device():  # pylint:disable=too-few-public-methods
    """ Holds the information about a device.

    Parameters
    ----------
    adapter: :class:`ctypes._Pointer`
        A ctypes pointer pointing at the adapter to be analysed
    log_func: :func:`~lib.gpu_stats._base._log`
        The logging function to use from the parent GPUStats class

    Attributes
    ----------
    is_hardware: bool
        ``True`` if the device is a hardware adapter. ``False`` if it is software
    compute_only: bool
        ``True`` if the device is a compute-only device. ``False`` if it is a GPU
    hardware_id: :class:`DXCoreHardwareID`
        The hardware identification information
    is_basic_render_vendor: bool
        ``True`` if the device is a basic render vendor
    is_basic_render_device: bool
        ``True`` if the device is a basic render device
    driver_version: str
        The driver version as a 4 part string
    d3d12_device: bool
        ``True`` if the device is DirectX 12 compatible
    driver_details: str
        The vendor and model name of the device
    luid: LUID
        The internal LUID of the device
    vram: :class:`VRam`
        The vram statistics of the device
    """
    def __init__(self, adapter: ctypes._Pointer, log_func: Callable[[str, str], None]) -> None:
        self._log = log_func
        self._log("debug", f"Initializing {self.__class__.__name__}: (adapter: {adapter}, "
                           f"log_func: {log_func})")
        self._adapter = adapter
        self.is_hardware: bool = self._get_property(DXCoreAdapterProperty.IsHardware,
                                                    ctypes.c_bool).value
        self.compute_only = not adapter.IsAttributeSupported(  # type:ignore[attr-defined]
                                                             LookupGUID.d3d12_graphics)
        self.hardware_id: DXCoreHardwareID = self._get_property(DXCoreAdapterProperty.HardwareID,
                                                                DXCoreHardwareID)
        self.is_basic_render_vendor: bool = self.hardware_id.vendorID == VendorID.MICROSOFT.value
        self.is_basic_render_device: bool = self.hardware_id.deviceID == 0x8c

        self.driver_version = self._get_driver_version()
        self.d3d12_device = self._try_create_d3d12_device()
        self.driver_details = self._get_driver_details()
        self.luid: LUID = self._get_property(DXCoreAdapterProperty.InstanceLuid, LUID)
        self.vram = self._get_vram()
        self._log("debug", f"Initialized {self.__class__.__name__}")

    def __repr__(self):
        """ Nicely format the __repr__ for logging purposes """
        items = ", ".join(f"{k}={v}" for k, v in self.__dict__.items()
                          if k[0] != "_" and k != "driver_details")
        retval = f"<{self.__class__.__name__}({self.driver_details}): ({items})>"
        return retval

    def _get_property(self, adapter_property: DXCoreAdapterProperty, ret_type: Any) -> Any:
        """ Obtain a property from an adapter

        Parameters
        ----------
        adapter_property: :class:`DXCoreAdapterProperty`
            Enum for the property to query
        ret_type: ctypes type
            The ctypes object that the data should be returned in

        Returns
        -------
        ctypes type
            The property value for the requested property
        """
        if issubclass(ret_type, ctypes.Array):
            # TODO For arrays a standard void pointer silently exits when returning the contents.
            # Ideally we would have a standardized way of creating the handle, but too much time
            # has been spent on this, so we just have switch logic depending on our limited use
            # case
            buffer = ret_type()
            handle = ctypes.c_void_p.from_buffer(buffer)
        else:
            handle = ctypes.c_void_p(0)
        success = self._adapter.GetProperty(adapter_property.value,  # type:ignore[attr-defined]
                                            ctypes.sizeof(ret_type),
                                            ctypes.byref(handle))

        if success != 0:  # Return empty requested object on failure
            try:
                retval = ret_type().value
            except AttributeError:
                retval = ret_type()
            return retval

        return POINTER(ret_type)(handle).contents

    def _get_driver_version(self) -> str:
        """ Obtain the driver version from the adapter API

        Returns
        -------
        str
            The 4 part driver version
        """
        driver = self._get_property(DXCoreAdapterProperty.DriverVersion, DriverVersion)
        self._log("debug", f"driver: {driver}")
        return f"{driver.parts_d}.{driver.parts_c}.{driver.parts_b}.{driver.parts_a}"

    def _try_create_d3d12_device(self) -> bool:
        """ Attempt to create a D3D12 device. Failure means the device does not support DirectX 12,
        success means it does.

        Returns
        -------
        bool
            ``True`` if the device supports DirectX 12 otherwise ``False``
        """
        feature_level = (D3DFeatureLevel.D3D_FEATURE_LEVEL_1_0_CORE if self.compute_only
                         else D3DFeatureLevel.D3D_FEATURE_LEVEL_11_0)
        func = windll.d3d12.D3D12CreateDevice
        func.argtypes = (POINTER(IDXCoreAdapter),
                         D3DFeatureLevel.ctype,  # type:ignore[attr-defined]  # pylint:disable=no-member  # noqa:E501
                         GUID)
        func.restype = HRESULT
        check = func(self._adapter, feature_level.value, LookupGUID.id3d12device)
        self._log("debug", f"d3d12 check result: {check}")
        retval = check in (0, 1)  # Should be 'S_FALSE' but leave S_OK anyway
        return retval

    def _get_driver_details(self) -> str:
        """ Obtain the driver details (device model name and number)

        Returns
        -------
        str
            The device clear text identifier
        """
        handle = ctypes.c_size_t(0)
        success = self._adapter.GetPropertySize(  # type:ignore[attr-defined]
                                                DXCoreAdapterProperty.DriverDescription.value,
                                                ctypes.byref(handle))
        if success != 0:
            retval = "Driver not Found"
        else:
            size = handle.value
            details = self._get_property(DXCoreAdapterProperty.DriverDescription,
                                         ctypes.c_char * size)
            retval = details.value.decode(encoding="utf-8", errors="ignore")
        self._log("debug", f"driver_details: {retval}")
        return retval

    def _get_current_vram(self, vram: VRam) -> None:
        """ Obtain statistics on current VRAM usage and populate to the given object

        Parameters
        ----------
        vram: :class:`VRam`
            The VRAM object for the device that is to be populated with current usage stats
        """
        if not self._adapter.IsQueryStateSupported(  # type:ignore[attr-defined]
                DXCoreAdapterState.AdapterMemoryBudget.value):
            self._log("debug", "QueryState not supported")
            return  # Just leave them all set to 0

        for segment in ("Local", "NonLocal"):
            in_ = DXCoreAdapterMemoryBudgetNodeSegmentGroup()
            in_.nodeIndex = 0  # pylint:disable=invalid-name,attribute-defined-outside-init
            in_.segmentGroup = getattr(DXCoreSegmentGroup,  segment).value  # pylint:disable=invalid-name,attribute-defined-outside-init  # noqa:E501
            in_ptr = ctypes.c_void_p.from_buffer(in_)

            out = DXCoreAdapterMemoryBudget()
            handle = ctypes.c_void_p.from_buffer(out)
            success = self._adapter.QueryState(  # type:ignore[attr-defined]
                DXCoreAdapterState.AdapterMemoryBudget.value,
                ctypes.c_size_t(ctypes.sizeof(in_)),
                ctypes.byref(in_ptr),
                ctypes.c_size_t(ctypes.sizeof(out)),
                ctypes.byref(handle))
            if success != 0:
                continue  # Just leave set at 0
            setattr(vram, f"{segment.lower()}_budget", out.budget)
            setattr(vram, f"{segment.lower()}_current_usage", out.currentUsage)
            setattr(vram,
                    f"{segment.lower()}_available_for_reservation",
                    out.availableForReservation)
            setattr(vram, f"{segment.lower()}_current_reservation", out.currentReservation)
            self._log("debug", f"{segment.lower()} free vram populated: {vram}")

    def _get_vram(self) -> VRam:
        """ Obtain total and available VRAM and populate to a :class:`VRam` object

        Returns
        -------
        :class:`VRam`
            Object holding VRAM information for a device
        """
        dedicated_adapter_memory = self._get_property(DXCoreAdapterProperty.DedicatedAdapterMemory,
                                                      ctypes.c_uint64)
        dedicated_system_memory = self._get_property(DXCoreAdapterProperty.DedicatedSystemMemory,
                                                     ctypes.c_uint64)
        shared_system_memory = self._get_property(DXCoreAdapterProperty.SharedSystemMemory,
                                                  ctypes.c_uint64)
        vram = VRam(dedicated_adapter_memory=dedicated_adapter_memory.value,
                    dedicated_system_memory=dedicated_system_memory.value,
                    shared_system_memory=shared_system_memory.value)
        self._log("debug", f"total vram populated: {vram}")
        self._get_current_vram(vram)
        self._log("debug", f"total vram populated: {vram}")
        return vram


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
        self._adapter_list = self._create_adapter_list()
        self._sort_adapter_list()
        self._adapters = self._get_adapters()
        self._valid_adaptors: List[Device] = []
        self._log("debug", f"Initialized {self.__class__.__name__}")

    def _get_factory(self) -> ctypes._Pointer:
        """ Get a DXGI 1.1 Factory object

        Reference
        ---------
        https://learn.microsoft.com/en-us/windows/win32/api/dxgi/nf-dxgi-createdxgifactory1

        Returns
        -------
        :class:`ctypes._Pointer`
            A pointer to a :class:`IDXGIFactory` COM instance
        """
        factory_func = windll.dxcore.DXCoreCreateAdapterFactory
        factory_func.argtypes = (GUID, POINTER(ctypes.c_void_p))
        factory_func.restype = HRESULT
        handle = ctypes.c_void_p(0)

        factory_func(IDXCoreAdapterFactory._iid_,  # pylint:disable=protected-access
                     ctypes.byref(handle))
        retval = ctypes.POINTER(IDXCoreAdapterFactory)(cast(IDXCoreAdapterFactory, handle.value))
        self._log("debug", f"factory: {retval}")

        return retval

    @property
    def valid_adapters(self) -> List[Device]:
        """ list: DirectX 12 compatible hardware :class:`Device` objects"""
        if self._valid_adaptors:
            return self._valid_adaptors

        for device in self._adapters:
            if not device.is_hardware or (device.is_basic_render_vendor
                                          and device.is_basic_render_device):
                # Sorted by most performant so everything after first basic adapter is skipped
                break
            self._valid_adaptors.append(device)
        self._log("debug", f"valid_adaptors: {self._valid_adaptors}")
        return self._valid_adaptors

    def _create_adapter_list(self) -> ctypes._Pointer:
        """ Obtain a list of connected adapters

        Returns
        -------
        :class:`ctypes._Pointer`
            The pointer to the adapter list

        Reference
        ---------
        https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/nf-dxcore_interface-idxcoreadapterfactory-createadapterlist
        """
        attribute = LookupGUID.d3d12_core_compute
        handle = ctypes.c_void_p(0)
        self._factory.CreateAdapterList(  # type:ignore[attr-defined]
            1,
            ctypes.byref(attribute),
            IDXCoreAdapterList._iid_,  # pylint:disable=protected-access
            ctypes.byref(handle))
        retval = ctypes.POINTER(IDXCoreAdapterList)(cast(IDXCoreAdapterList, handle.value))
        self._log("debug", f"adapter_list: {retval}")
        return retval

    def _sort_adapter_list(self) -> None:
        """ Sort the adapter list, in place, by most performant

        References
        ----------
        https://learn.microsoft.com/en-us/windows/win32/dxcore/dxcore_interface/nf-dxcore_interface-idxcoreadapterlist-sort
        """
        preference = ctypes.c_uint32(DXCoreAdapterPreference.HighPerformance.value)
        self._adapter_list.Sort(1, ctypes.byref(preference))  # type:ignore[attr-defined]
        self._log("debug", f"sorted adapter_list: {self._adapter_list}")

    def _get_adapters(self) -> List[Device]:
        """ Obtain DirectX 12 supporting hardware adapter objects and add to a Device class for
        obtaining details

        Returns
        -------
        list
            List of :class:`Device` objects
        """
        num_adapters = self._adapter_list.GetAdapterCount()  # type:ignore[attr-defined]
        retval = []
        for idx in range(num_adapters):
            handle = ctypes.c_void_p(0)
            success = self._adapter_list.GetAdapter(  # type:ignore[attr-defined]
                idx,
                IDXCoreAdapter._iid_,  # pylint:disable=protected-access
                ctypes.byref(handle))
            if success != 0:
                continue
            retval.append(Device(POINTER(IDXCoreAdapter)(cast(IDXCoreAdapter, handle.value)),
                                 self._log))
        self._log("debug", f"adapters: {retval}")
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
        self._devices: List[Device] = []
        super().__init__(log=log)

    @property
    def _all_vram(self) -> List[int]:
        """ list: The VRAM of each GPU device that DX Core has discovered. """
        return [int(device.vram.dedicated_adapter_memory / (1024 * 1024))
                for device in self._devices]

    @property
    def names(self) -> List[str]:
        """ list: The name of each GPU device that DX Core has discovered. """
        return [device.driver_details for device in self._devices]

    def _get_active_devices(self) -> List[int]:
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

    def _get_devices(self) -> List[Device]:
        """ Obtain all detected DX12 devices.

        Returns
        -------
        list
            The :class:`~dx_lib.Device` objects for GPUs that DX Core has discovered.
        """
        adapters = Adapters(log_func=self._log)
        devices = adapters.valid_adapters
        self._log("debug", f"Obtained Devices: {devices}")
        return devices

    def _initialize(self) -> None:
        """ Initialize DX Core for DirectML backend.

        If :attr:`_is_initialized` is ``True`` then this function just returns performing no
        action.

        if ``False`` then PlaidML is setup, if not already, and GPU information is extracted
        from the PlaidML context.
        """
        if self._is_initialized:
            return
        self._log("debug", "Initializing DX Core for DirectML.")
        self._devices = self._get_devices()
        super()._initialize()

    def _get_device_count(self) -> int:
        """ Detect the number of GPUs available from DX Core.

        Returns
        -------
        int
            The total number of GPUs available
        """
        retval = len(self._devices)
        self._log("debug", f"GPU Device count: {retval}")
        return retval

    def _get_handles(self) -> list:
        """ DX Core Doesn't really use device handles, so we just return the all devices list

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

    def _get_device_names(self) -> List[str]:
        """ Obtain the list of names of connected GPUs as identified in :attr:`_handles`.

        Returns
        -------
        list
            The list of connected Nvidia GPU names
        """
        names = self.names
        self._log("debug", f"GPU Devices: {names}")
        return names

    def _get_vram(self) -> List[int]:
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

    def _get_free_vram(self) -> List[int]:
        """ Obtain the amount of VRAM that is available, in Megabytes, for each connected DirectX
        12 supporting GPU.

        Returns
        -------
        list
             List of `float`s containing the amount of VRAM available, in Megabytes, for each
             connected GPU as corresponding to the values in :attr:`_handles
        """
        vram = [int(device.vram.local_budget / (1024 * 1024)) for device in self._devices]
        self._log("debug", f"GPU VRAM free: {vram}")
        return vram
