#! /usr/env/bin/python
"""
Queries information about system installed Machine Learning Libraries.
NOTE: Only packages from Python's Standard Library should be imported in this module
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import typing as T

from abc import ABC, abstractmethod
from shutil import which

from lib.utils import get_module_objects

from .system import _lines_from_command

if platform.system() == "Windows":
    import winreg  # pylint:disable=import-error
else:
    winreg = None  # type:ignore[assignment]  # pylint:disable=invalid-name

if T.TYPE_CHECKING:
    from winreg import HKEYType  # type:ignore[attr-defined]

logger = logging.getLogger(__name__)


_TORCH_ROCM_REQUIREMENTS = {">=2.2.1,<2.4.0": ((6, 0), (6, 0))}
"""dict[str, tuple[tuple[int, int], tuple[int, int]]]: Minumum and maximum ROCm versions """


def _check_dynamic_linker(lib: str) -> list[str]:
    """ Locate the folders that contain a given library in ldconfig and $LD_LIBRARY_PATH

    Parameters
    ----------
    lib: str The library to locate

    Returns
    -------
    list[str]
        All real existing folders from ldconfig or $LD_LIBRARY_PATH that contain the given lib
    """
    paths: set[str] = set()
    ldconfig = which("ldconfig")
    if ldconfig:
        paths.update({os.path.realpath(os.path.dirname(line.split("=>")[-1].strip()))
                      for line in _lines_from_command([ldconfig, "-p"])
                      if lib in line and "=>" in line})

    if not os.environ.get("LD_LIBRARY_PATH"):
        return list(paths)

    paths.update({os.path.realpath(path)
                  for path in os.environ["LD_LIBRARY_PATH"].split(":")
                  if path and os.path.exists(path)
                  for fname in os.listdir(path)
                  if lib in fname})
    return list(paths)


def _files_from_folder(folder: str, prefix: str) -> list[str]:
    """ Obtain all filenames from the given folder that start with the given prefix

    Parameters
    ----------
    folder : str
        The folder to search for files in
    prefix : str
        The filename prefix to search for

    Returns
    -------
    list[str]
        All filenames that exist in the given folder with the given prefic
    """
    if not os.path.exists(folder):
        return []
    return [f for f in os.listdir(folder) if f.startswith(prefix)]


class _Alternatives:
    """ Holds output from the update-alternatives command for the given package

    Parameters
    ----------
    package : str
        The package to query update-alternatives for information
    """
    def __init__(self, package: str) -> None:
        self._package = package
        self._bin = which("update-alternatives")
        self._default_marker = "link currently points to"
        self._alternatives_marker = "priority"
        self._output: list[str] | None = None

    @property
    def alternatives(self) -> list[str]:
        """ list[str] : Full path to alternatives listed for the given package """
        if self._output is None:
            self._query()
        if not self._output:
            return []
        retval = [line.rsplit(" - ", maxsplit=1)[0] for line in self._output
                  if self._alternatives_marker in line.lower()]
        logger.debug("Versions from 'update-alternatives' for '%s': %s", self._package, retval)
        return retval

    @property
    def default(self) -> str:
        """ str : Full path to the default package """
        if self._output is None:
            self._query()
        if not self._output:
            return ""
        retval = next((x for x in self._output
                       if x.startswith(self._default_marker)), "").replace(self._default_marker,
                                                                           "").strip()
        logger.debug("Default from update-alternatives for '%s': %s", self._package, retval)
        return retval

    def _query(self) -> None:
        """ Query update-alternatives for the given package and place stripped output into
        :attr:`_output` """
        if not self._bin:
            self._output = []
            return
        cmd = [self._bin, "--display", self._package]
        retval = [line.strip() for line in _lines_from_command(cmd)]
        logger.debug("update-alternatives output for command %s: %s",
                     cmd, retval)
        self._output = retval


class _Cuda(ABC):
    """ Find the location of system installed Cuda and cuDNN on Windows and Linux. """
    def __init__(self) -> None:
        self.versions: list[tuple[int, int]] = []
        """ list[tuple[int, int]] : All detected globally installed Cuda versions """
        self.version: tuple[int, int] = (0, 0)
        """ tuple[int, int] : Default installed Cuda version. (0, 0) if not detected """
        self.cudnn_versions: dict[tuple[int, int], tuple[int, int, int]] = {}
        """ dict[tuple[int, int], tuple[int, int, int]] : Detected cuDNN version for each installed
        Cuda. key (0, 0) denotes globally installed cudnn """
        self._paths: list[str] = []
        """ list[str] : list of path to Cuda install folders relating to :attr:`versions` """

        self._version_file = "version.json"
        self._lib = "libcudart.so"
        self._cudnn_header = "cudnn_version.h"
        self._alternatives = _Alternatives("cuda")
        self._re_cudnn = re.compile(r"#define CUDNN_(MAJOR|MINOR|PATCHLEVEL)\s+(\d+)")

        if platform.system() in ("Windows", "Linux"):
            self._get_versions()
            self._get_version()
            self._get_cudnn_versions()

    def __repr__(self) -> str:
        """ Pretty representation of this class """
        attrs = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items()
                          if not k.startswith("_"))
        return f"{self.__class__.__name__}({attrs})"

    @classmethod
    def _tuple_from_string(cls, version: str) -> tuple[int, int] | None:
        """ Convert a Cuda version string to a version tuple

        Parameters
        ----------
        version : str
            The Cuda version string to convert

        Returns
        -------
        tuple[int, int] | None
            The converted Cuda version string. ``None`` if not a valid version string
        """
        if version.startswith("."):
            version = version[1:]
        split = version.split(".")
        if len(split) not in (2, 3):
            return None
        split = split[:2]
        if not all(x.isdigit() for x in split):
            return None
        return (int(split[0]), int(split[1]))

    @abstractmethod
    def get_versions(self) -> dict[tuple[int, int], str]:
        """ Overide to Attempt to detect all installed Cuda versions on Linux or Windows systems

        Returns
        -------
        dict[tuple[int, int], str]
            The Cuda versions to the folder path on the system
        """

    @abstractmethod
    def get_version(self) -> tuple[int, int] | None:
        """ Override to attempt to locate the default Cuda version on Linux or Windows

        Returns
        -------
        tuple[int, int] | None
            The Default global Cuda version or ``None`` if not found
        """

    @abstractmethod
    def get_cudnn_versions(self) -> dict[tuple[int, int], tuple[int, int, int]]:
        """ Override to attempt to locate any installed cuDNN versions

        Returns
        -------
        dict[tuple[int, int], tuple[int, int, int]]
            Detected cuDNN version for each installed Cuda. key (0, 0) denotes globally installed
            cudnn
        """

    def version_from_version_file(self, folder: str) -> tuple[int, int] | None:
        """ Attempt to get an installed Cuda version from its version.json file

        Parameters
        ----------
        folder : str
            Full path to the folder to check for a version file

        Returns
        -------
        tuple[int, int] | None
            The detected Cuda version or ``None`` if not detected
        """
        vers_file = os.path.join(folder, self._version_file)
        if not os.path.exists(vers_file):
            return None
        with open(vers_file, "r", encoding="utf-8", errors="replace") as f:
            vers = json.load(f)
        retval = self._tuple_from_string(vers.get("cuda_cudart", {}).get("version"))
        logger.debug("Version from '%s': %s", vers_file, retval)
        return retval

    def _version_from_nvcc(self) -> tuple[int, int] | None:
        """ Obtain the version from NVCC output if it is on PATH

        Returns
        -------
        tuple[int, int] | None
            The detected default Cuda version. ``None`` if not version detected
        """
        retval = None
        nvcc = which("nvcc")
        if not nvcc:
            return retval

        for line in _lines_from_command([nvcc, "-V"]):
            vers = re.match(r".*release (\d+\.\d+)", line)
            if vers is not None:
                retval = self._tuple_from_string(vers.group(1))
                break
        logger.debug("Version from NVCC '%s': %s", nvcc, retval)
        return retval

    def _get_versions(self) -> None:
        """ Attempt to detect all installed Cuda versions and populate to :attr:`versions` """
        versions = self.get_versions()
        if versions:
            logger.debug("Cuda Versions: %s", versions)
            self.versions = list(versions)
            self._paths = list(versions.values())
            return
        logger.debug("Could not locate any Cuda versions")

    def _get_version(self) -> None:
        """ Attempt to detect the default Cuda version and populate to :attr:`version` """
        version: tuple[int, int] | None = None
        if len(self.versions) == 1:
            version = self.versions[0]
            logger.debug("Only 1 installed Cuda version: %s", version)
        if not version:
            version = self._version_from_nvcc()
        if not version:
            version = self.get_version()
        if version:
            self.version = version
        logger.debug("Cuda version: %s", self.version if version else "not detected")

    def _get_cudnn_versions(self) -> None:
        """ Attempt to locate any installed cuDNN versions and add to :attr`cudnn_versions` """
        versions = self.get_cudnn_versions()
        if versions:
            logger.debug("cudnn versions: %s", versions)
            self.cudnn_versions = versions
            return
        logger.debug("No cudnn versions found")

    def cudnn_version_from_header(self, folder: str) -> tuple[int, int, int] | None:
        """ Attempt to detect the cuDNN version from the version header file

        Parameters
        ----------
        folder : str
            The folder to check for the cuDNN header file

        Returns
        -------
        tuple[int, int, int] | None
            The cuDNN version found from the given folder or ``None`` if not detected
        """
        path = os.path.join(folder, self._cudnn_header)
        if not os.path.exists(path):
            logger.debug("cudnn file '%s' does not exist", path)
            return None

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            file = f.read()
        version = {v[0]: int(v[1]) if v[1].isdigit() else 0
                   for v in self._re_cudnn.findall(file)}
        if not version:
            logger.debug("cudnn version could not be found in '%s'", path)
            return None

        logger.debug("cudnn version from '%s': %s", path, version)
        retval = (version.get("MAJOR", 0), version.get("MINOR", 0), version.get("PATCHLEVEL", 0))
        logger.debug("cudnn versions: %s", retval)
        return retval


class CudaLinux(_Cuda):
    """ Find the location of system installed Cuda and cuDNN on Linux. """
    def __init__(self) -> None:
        self._folder_prefix = "cuda-"
        super().__init__()

    def _version_from_lib(self, folder: str) -> tuple[int, int] | None:
        """ Attempt to locate the version from the existence of libcudart.so within a Cuda
        targets/x86_64-linux/lib folder

        Parameters
        ----------
        folder : str
            Full file path to the Cuda folder

        Returns
        -------
        tuple[int, int] | None
            The Cuda version identified by the existence of the libcudart.so file. ``None`` if
            not detected
        """
        lib_folder = os.path.join(folder, "targets", "x86_64-linux", "lib")
        lib_versions = [f.replace(self._lib, "")
                        for f in _files_from_folder(lib_folder, self._lib)]
        if not lib_versions:
            return None
        versions = [self._tuple_from_string(f[1:])
                    for f in lib_versions if f and f.startswith(".")]
        valid = [v for v in versions if v is not None]
        if not valid or not len(set(valid)) == 1:
            return None
        retval = valid[0]
        logger.debug("Version from '%s': %s", os.path.join(lib_folder, self._lib), retval)
        return retval

    def _versions_from_usr(self) -> dict[tuple[int, int], str]:
        """ Attempt to detect all installed Cuda versions from the /usr/local folder

        Scan /usr/local for cuda-x.x folders containing either a version.json file or
        include/lib/libcudart.so.x.

        Returns
        -------
        dict[tuple[int, int], str]
            A dictionary of detected Cuda versions to their install paths
        """
        retval: dict[tuple[int, int], str] = {}
        usr = os.path.join(os.sep, "usr", "local")

        for folder in _files_from_folder(usr, self._folder_prefix):
            path = os.path.join(usr, folder)
            if os.path.islink(path):
                continue
            version = self.version_from_version_file(path) or self._version_from_lib(path)
            if version is not None:
                retval[version] = path
        return retval

    def _versions_from_alternatives(self) -> dict[tuple[int, int], str]:
        """ Attempt to detect all installed Cuda versions from update-alternatives

        Returns
        -------
        list[tuple[int, int, int]]
            A dictionary of detected Cuda versions to their install paths found in
            update-alternatives
        """
        retval: dict[tuple[int, int], str] = {}
        alts = self._alternatives.alternatives
        for path in alts:
            vers = self.version_from_version_file(path) or self._version_from_lib(path)
            if vers is not None:
                retval[vers] = path
        logger.debug("Versions from 'update-alternatives': %s", retval)
        return retval

    def _parent_from_targets(self, folder: str) -> str:
        """ Obtain the Cuda parent folder from a path obtained from child targets folder

        Parameters
        ----------
        folder : str
            Full path to a folder that has a 'targets' folder in its path

        Returns
        -------
        str
            The potential parent Cuda folder, or an empty string if not detected
        """
        split = folder.split(os.sep)
        return os.sep.join(split[:split.index("targets")]) if "targets" in split else ""

    def _versions_from_dynamic_linker(self) -> dict[tuple[int, int], str]:
        """ Attempt to detect all installed Cuda versions from ldconfig

        Returns
        -------
        dict[tuple[int, int], str]
            The Cuda version to the folder path found from ldconfig
        """
        retval: dict[tuple[int, int], str] = {}
        folders = _check_dynamic_linker(self._lib)
        cuda_roots = [self._parent_from_targets(f) for f in folders]
        for path in cuda_roots:
            if not path:
                continue
            version = self.version_from_version_file(path) or self._version_from_lib(path)
            if version is not None:
                retval[version] = path

        logger.debug("Versions from 'ld_config': %s", retval)
        return retval

    def get_versions(self) -> dict[tuple[int, int], str]:
        """ Attempt to detect all installed Cuda versions on Linux systems

        Returns
        -------
        dict[tuple[int, int], str]
            The Cuda version to the folder path on Linux
        """
        versions = (self._versions_from_usr() |
                    self._versions_from_alternatives() |
                    self._versions_from_dynamic_linker())
        return {k: versions[k] for k in sorted(versions)}

    def _version_from_alternatives(self) -> tuple[int, int] | None:
        """ Attempt to get the default Cuda version from update-alternatives

        Returns
        -------
        tuple[int, int] | None
            The detected default Cuda version. ``None`` if not version detected
        """
        default = self._alternatives.default
        if not default:
            return None
        retval = self.version_from_version_file(default) or self._version_from_lib(default)
        logger.debug("Version from update-alternatives: %s", retval)
        return retval

    def _version_from_link(self) -> tuple[int, int] | None:
        """ Attempt to get the default Cuda version from the /usr/local/cuda file

        Returns
        -------
        tuple[int, int] | None
            The detected default Cuda version. ``None`` if not version detected
        """
        path = os.path.join(os.sep, "usr", "local", "cuda")
        if not os.path.exists(path):
            return None
        real_path = os.path.abspath(os.path.realpath(path)) if os.path.islink(path) else path
        retval = self.version_from_version_file(real_path) or self._version_from_lib(real_path)
        logger.debug("Version from symlink: %s", retval)
        return retval

    def _version_from_dynamic_linker(self) -> tuple[int, int] | None:
        """ Attempt to get the default version from ldconfig or $LD_LIBRARY_PATH

        Returns
        -------
        tuple[int, int, int] | None
            The detected default ROCm version. ``None`` if not version detected
        """
        paths = _check_dynamic_linker(self._lib)
        if len(paths) != 1:  # Multiple or None
            return None
        root = self._parent_from_targets(paths[0])
        retval = self.version_from_version_file(root) or self._version_from_lib(root)
        logger.debug("Version from ld_config: %s", retval)
        return retval

    def get_version(self) -> tuple[int, int] | None:
        """ Attempt to locate the default Cuda version on Linux

        Checks, in order: update-alternatives, /usr/local/cuda, ldconfig, nvcc

        Returns
        -------
        tuple[int, int] | None
            The Default global Cuda version or ``None`` if not found
        """
        return (self._version_from_alternatives() or
                self._version_from_link() or
                self._version_from_dynamic_linker())

    def get_cudnn_versions(self) -> dict[tuple[int, int], tuple[int, int, int]]:
        """ Attempt to locate any installed cuDNN versions on Linux

        Returns
        -------
        dict[tuple[int, int], tuple[int, int, int]]
            Detected cuDNN version for each installed Cuda. key (0, 0) denotes globally installed
            cudnn
        """
        retval: dict[tuple[int, int], tuple[int, int, int]] = {}
        gbl = ["/usr/include", "/usr/local/include"]
        lcl = [os.path.join(f, "include") for f in self._paths]
        for root in gbl + lcl:
            for folder, _, filenames in os.walk(root):
                if self._cudnn_header not in filenames:
                    continue
                version = self.cudnn_version_from_header(folder)
                if not version:
                    continue
                cuda_vers = ((0, 0) if root in gbl
                             else self.versions[self._paths.index(os.path.dirname(root))])
                retval[cuda_vers] = version
        return retval


class CudaWindows(_Cuda):
    """ Find the location of system installed Cuda and cuDNN on Windows. """

    @classmethod
    def _enum_subkeys(cls, key: HKEYType) -> T.Generator[str, None, None]:
        """ Iterate through a Registry key's sub-keys

        Parameters
        ----------
        key : :class:`winreg.HKEYType`
            The Registry key to iterate

        Yields
        ------
        str
            A sub-key name from the given registry key
        """
        assert winreg is not None
        i = 0
        while True:
            try:
                yield winreg.EnumKey(key, i)  # type:ignore[attr-defined]
            except OSError:
                break
            i += 1

    def get_versions(self) -> dict[tuple[int, int], str]:
        """ Attempt to detect all installed Cuda versions on Windows systems from the registry

        Returns
        -------
        dict[tuple[int, int], str]
            The Cuda version to the folder path on Windows
        """
        retval: dict[tuple[int, int], str] = {}
        assert winreg is not None
        reg_key = r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA"
        paths = {k.lower().replace("cuda_path_", "").replace("_", "."): v
                 for k, v in os.environ.items()
                 if "cuda_path_v" in k.lower()}
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,  # type:ignore[attr-defined]
                                reg_key) as key:
                for version in self._enum_subkeys(key):
                    vers_tuple = self._tuple_from_string(version[1:])
                    if vers_tuple is not None:
                        retval[vers_tuple] = paths.get(version, "")
        except FileNotFoundError:
            logger.debug("Could not find Windows Registry key '%s'", reg_key)
        return {k: retval[k] for k in sorted(retval)}

    def get_version(self) -> tuple[int, int] | None:
        """ Attempt to get the default Cuda version from the Environment Variable

        Returns
        -------
        tuple[int, int] | None
            The Default global Cuda version or ``None`` if not found
        """
        path = os.environ.get("CUDA_PATH")
        if not path or path not in self._paths:
            return None

        retval = self.versions[self._paths.index(path)]
        logger.debug("Version from CUDA_PATH Environment Variable: %s", path)
        return retval

    def _get_cudnn_paths(self) -> list[str]:  # noqa[C901]
        """ Attempt to locate the locations of cuDNN installs for Windows

        Returns
        -------
        list[str]
            Full path to existing cuDNN installs under Windows
        """
        assert winreg is not None
        paths: set[str] = set()
        cudnn_key = "cudnn_cuda"
        reg_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
        lookups = (winreg.HKEY_LOCAL_MACHINE,  # type:ignore[attr-defined]
                   winreg.HKEY_CURRENT_USER)  # type:ignore[attr-defined]
        for lookup in lookups:
            try:
                key = winreg.OpenKey(lookup, reg_key)  # type:ignore[attr-defined]
            except FileNotFoundError:
                continue
            for name in self._enum_subkeys(key):
                if cudnn_key not in name.lower():
                    logger.debug("Skipping subkey '%s'", name)
                    continue
                try:
                    subkey = winreg.OpenKey(key, name)  # type:ignore[attr-defined]
                    logger.debug("Skipping subkey not found '%s'", name)
                except FileNotFoundError:
                    continue
                logger.debug("Parsing cudnn key '%s'", cudnn_key)
                try:
                    path, _ = winreg.QueryValueEx(subkey,  # type:ignore[attr-defined]
                                                  "InstallLocation")
                except (FileNotFoundError, OSError):
                    logger.debug("Skipping missing InstallLocation for sub-key '%s'", subkey)
                    continue
                if not os.path.isdir(path):
                    logger.debug("Skipping non-existant path '%s'", path)
                    continue
                paths.add(path)
        retval = list(paths)
        logger.debug("cudnn install paths: %s", retval)
        return retval

    def get_cudnn_versions(self) -> dict[tuple[int, int], tuple[int, int, int]]:
        """ Attempt to locate any installed cuDNN versions on Windows

        Returns
        -------
        dict[tuple[int, int], tuple[int, int, int]]
            Detected cuDNN version for each installed Cuda. key (0, 0) denotes globally installed
            cudnn
        """
        retval: dict[tuple[int, int], tuple[int, int, int]] = {}
        gbl = self._get_cudnn_paths()
        lcl = [os.path.join(f, "include") for f in self._paths]
        for root in gbl + lcl:
            for folder, _, filenames in os.walk(root):
                if self._cudnn_header not in filenames:
                    continue
                version = self.cudnn_version_from_header(folder)
                if not version:
                    continue
                cuda_vers = ((0, 0) if root in gbl
                             else self.versions[self._paths.index(os.path.dirname(root))])
                retval[cuda_vers] = version
        return retval


def get_cuda_finder() -> type[_Cuda]:
    """Create a platform-specific CUDA object.

    Returns
    -------
    type[_Cuda]
        The OS specific finder for system-wide Cuda
    """
    if platform.system().lower() == "windows":
        return CudaWindows
    return CudaLinux


Cuda = get_cuda_finder()


class ROCm():
    """ Find the location of system installed ROCm on Linux """
    def __init__(self) -> None:
        self.version_min = min(v[0] for v in _TORCH_ROCM_REQUIREMENTS.values())
        self.version_max = max(v[1] for v in _TORCH_ROCM_REQUIREMENTS.values())
        self.versions: list[tuple[int, int, int]] = []
        """ list[tuple[int, int, int]] : All detected ROCm installed versions """
        self.version: tuple[int, int, int] = (0, 0, 0)
        """ tuple[int, int, int] : Default ROCm installed version. (0, 0, 0) if not detected """

        self._folder_prefix = "rocm-"
        self._version_files = ["version-rocm", "version"]
        self._lib = "librocm-core.so"
        self._alternatives = _Alternatives("rocm")
        self._re_version = re.compile(r"(\d+\.\d+\.\d+)(?=$|[-.])")
        self._re_config = re.compile(r"\sroc-(\d+\.\d+\.\d+)(?=\s|[-.])")
        if platform.system() == "Linux":
            self._rocm_check()

    def __repr__(self) -> str:
        """ Pretty representation of this class """
        attrs = ", ".join(f"{k}={repr(v)}" for k, v in self.__dict__.items()
                          if not k.startswith("_"))
        return f"{self.__class__.__name__}({attrs})"

    @property
    def valid_versions(self) -> list[tuple[int, int, int]]:
        """ list[tuple[int, int, int]] """
        return [v for v in self.versions if self.version_min <= v[:2] <= self.version_max]

    @property
    def valid_installed(self) -> bool:
        """ bool : ``True`` if a valid version of ROCm is installed """
        return any(self.valid_versions)

    @property
    def is_valid(self):
        """ bool : ``True`` if the default ROCm version is valid """
        return self.version_min <= self.version[:2] <= self.version_max

    @classmethod
    def _tuple_from_string(cls, version: str) -> tuple[int, int, int] | None:
        """ Convert a ROCm version string to a version tuple

        Parameters
        ----------
        version : str
            The ROCm version string to convert

        Returns
        -------
        tuple[int, int, int] | None
            The converted ROCm version string. ``None`` if not a valid version string
        """
        split = version.split(".")
        if len(split) != 3:
            return None
        if not all(x.isdigit() for x in split):
            return None
        return (int(split[0]), int(split[1]), int(split[2]))

    def _version_from_string(self, string: str) -> tuple[int, int, int] | None:
        """ Obtain the ROCm version from the end of a string

        Parameters
        ----------
        string : str
            The string to test for a valid ROCm version

        Returns
        -------
        tuple[int, int, int] | None
            The ROCm version from the end of the string or ``None`` if not detected
        """
        re_vers = self._re_version.search(string)
        if re_vers is None:
            return None
        return self._tuple_from_string(re_vers.group(1))

    def _version_from_info(self, folder: str) -> tuple[int, int, int] | None:
        """ Attempt to locate the version from a version file within a ROCm .info folder

        Parameters
        ----------
        file_path : str
            Full path to the ROCm .info folder

        Returns
        -------
        tuple[int, int, int] | None
            The ROCm version extracted from a version file within the .info folder. ``None`` if
            not detected
        """
        info_loc = [os.path.join(folder, ".info", v) for v in self._version_files]
        for info_file in info_loc:
            if not os.path.exists(info_file):
                continue
            with open(info_file, "r", encoding="utf-8") as f:
                vers_string = f.read().strip()
            if not vers_string:
                continue
            retval = self._tuple_from_string(vers_string.split("-", maxsplit=1)[0])
            if retval is None:
                continue
            logger.debug("Version from '%s': %s", info_file, retval)
            return retval
        return None

    def _version_from_lib(self, folder: str) -> tuple[int, int, int] | None:
        """ Attempt to locate the version from the existence of librocm-core.so within a ROCm
        lib folder

        Parameters
        ----------
        folder : str
            Full file path to the ROCm folder

        Returns
        -------
        tuple[int, int, int] | None
            The ROCm version identified by the existence of the librocm-core.so file. ``None`` if
            not detected
        """
        lib_folder = os.path.join(folder, "lib")
        lib_files = _files_from_folder(lib_folder, self._lib)
        if not lib_files:
            return None

        # librocm-core naming is librocm-core.so.1.0.##### which is ambiguous. Get from folder
        rocm_folder = os.path.basename(folder)
        if not rocm_folder.startswith(self._folder_prefix):
            return None
        retval = self._version_from_string(rocm_folder)
        logger.debug("Version from '%s': %s", os.path.join(lib_folder, self._lib), retval)
        return retval

    def _versions_from_opt(self) -> list[tuple[int, int, int]]:
        """ Attempt to detect all installed ROCm versions from the /opt folder

        Scan /opt for rocm.x.x.x folders containing either .info or lib/librocm-core.so.x

        Returns
        -------
        list[tuple[int, int, int]]
            Any ROCm versions found in the /opt folder
        """
        retval: list[tuple[int, int, int]] = []
        opt = os.path.join(os.sep, "opt")

        for folder in _files_from_folder(opt, self._folder_prefix):
            path = os.path.join(opt, folder)
            version = self._version_from_info(path) or self._version_from_lib(path)
            if version is not None:
                retval.append(version)

        return retval

    def _versions_from_alternatives(self) -> list[tuple[int, int, int]]:
        """ Attempt to detect all installed ROCm versions from update-alternatives

        Returns
        -------
        list[tuple[int, int, int]]
            Any ROCm versions found in update-alternatives
        """
        alts = self._alternatives.alternatives
        if not alts:
            return []
        versions = [self._version_from_string(c) for c in alts]
        retval = list(set(v for v in versions if v is not None))
        logger.debug("Versions from 'update-alternatives': %s", retval)
        return retval

    def _versions_from_dynamic_linker(self) -> list[tuple[int, int, int]]:
        """ Attempt to detect all installed ROCm versions from ldconfig

        Returns
        -------
        dict[tuple[int, int], str]
            The ROCm versions found from ldconfig
        """
        retval: list[tuple[int, int, int]] = []
        folders = _check_dynamic_linker(self._lib)
        for folder in folders:
            path = os.path.dirname(folder)
            version = self._version_from_info(path) or self._version_from_lib(path)
            if version is not None:
                retval.append(version)

        logger.debug("Versions from 'ld_config': %s", retval)
        return retval

    def _get_versions(self) -> None:
        """ Attempt to detect all installed ROCm versions and populate to :attr:`rocm_versions` """
        versions = list(sorted(set(self._versions_from_opt()) |
                               set(self._versions_from_alternatives()) |
                               set(self._versions_from_dynamic_linker())))
        if versions:
            logger.debug("ROCm Versions: %s", versions)
            self.versions = versions
            return
        logger.debug("Could not locate any ROCm versions")

    def _version_from_hipconfig(self) -> tuple[int, int, int] | None:
        """ Attempt to get the default version from hipconfig

        Returns
        -------
        tuple[int, int, int] | None
            The detected default ROCm version. ``None`` if not version detected
        """
        retval: tuple[int, int, int] | None = None
        exe = which("hipconfig")
        if not exe:
            return retval
        lines = _lines_from_command([exe, "--full"])
        if not lines:
            return retval
        for line in lines:
            line = line.strip()
            if line.startswith("ROCM_PATH"):
                path = line.split(":", maxsplit=1)[-1]
                retval = self._version_from_info(path) or self._version_from_lib(path)
            match = self._re_config.search(line)

            if match is not None:
                retval = self._tuple_from_string(match.group(1))

        logger.debug("Version from hipconfig: %s", retval)
        return retval

    def _version_from_alternatives(self) -> tuple[int, int, int] | None:
        """ Attempt to get the default version from update-alternatives

        Returns
        -------
        tuple[int, int, int] | None
            The detected default ROCm version. ``None`` if not version detected
        """
        default = self._alternatives.default
        if not default:
            return None
        retval = self._version_from_string(default.rsplit(os.sep, maxsplit=1)[-1])
        logger.debug("Version from update-alternatives: %s", retval)
        return retval

    def _version_from_link(self) -> tuple[int, int, int] | None:
        """ Attempt to get the default version from the /opt/rocm file

        Returns
        -------
        tuple[int, int, int] | None
            The detected default ROCm version. ``None`` if not version detected
        """
        path = os.path.join(os.sep, "opt", "rocm")
        if not os.path.exists(path):
            return None
        real_path = os.path.abspath(os.path.realpath(path)) if os.path.islink(path) else path
        retval = self._version_from_info(real_path) or self._version_from_lib(real_path)
        logger.debug("Version from symlink: %s", retval)
        return retval

    def _version_from_dynamic_linker(self) -> tuple[int, int, int] | None:
        """ Attempt to get the default version from ldconfig or $LD_LIBRARY_PATH

        Returns
        -------
        tuple[int, int, int] | None
            The detected default ROCm version. ``None`` if not version detected
        """
        paths = _check_dynamic_linker("librocm-core.so.")
        if len(paths) != 1:  # Multiple or None
            return None
        path = os.path.dirname(paths[0])
        retval = self._version_from_info(path) or self._version_from_lib(path)
        logger.debug("Version from ld_config: %s", retval)
        return retval

    def _get_version(self) -> None:
        """ Attempt to detect the default ROCm version """
        version = (self._version_from_hipconfig() or
                   self._version_from_alternatives() or
                   self._version_from_link() or
                   self._version_from_dynamic_linker())
        if version is not None:
            logger.debug("ROCm default version: %s", version)
            self.version = version
            return
        logger.debug("Could not locate default ROCm version")

    def _rocm_check(self) -> None:
        """ Attempt to locate the installed ROCm versions and the default ROCm version """
        self._get_versions()
        self._get_version()
        logger.debug("ROCm Versions: %s, Version: %s", self.versions, self.version)


__all__ = get_module_objects(__name__)


if __name__ == "__main__":
    print(Cuda())
    print(ROCm())
