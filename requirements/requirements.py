#! /usr/env/bin/python3
""" Parses the contents of python requirements.txt files and holds the information in a parsable
format

NOTE: Only packages from the Python Standard Library should be imported in this module
"""
from __future__ import annotations

import logging
import typing as T
import os

from importlib import import_module, util as import_util

if T.TYPE_CHECKING:
    from packaging.markers import Marker
    from packaging.requirements import Requirement
    from packaging.specifiers import Specifier

logger = logging.getLogger(__name__)


PYTHON_VERSIONS: dict[str, tuple[int, int]] = {"rocm_60": (3, 12)}
""" dict[str, tuple[int, int]] : Mapping of requirement file names to the maximum supported
Python version, if below the project maximum """


class Requirements:
    """ Parse requirement information

    Parameters
    ----------
    include_dev : bool, optional
        ``True`` to additionally load requirements from the dev requirements file
    """
    def __init__(self, include_dev: bool = False) -> None:
        self._include_dev = include_dev
        self._marker: type[Marker] | None = None
        self._requirement: type[Requirement] | None = None
        self._specifier: type[Specifier] | None = None
        self._global_options: dict[str, list[str]] = {}
        self._requirements: dict[str, list[Requirement]] = {}

    @property
    def packaging_available(self) -> bool:
        """ bool : ``True`` if the packaging Library is available otherwise ``False`` """
        if self._requirement is not None:
            return True
        return import_util.find_spec("packaging") is not None

    @property
    def requirements(self) -> dict[str, list[Requirement]]:
        """ dict[str, list[Requirement]] : backend type as key, list of required packages as
        value """
        if not self._requirements:
            self._load_requirements()
        return self._requirements

    @property
    def global_options(self) -> dict[str, list[str]]:
        """ dict[str, list[str]] : The global pip install options for each backend """
        if not self._requirements:
            self._load_requirements()
        return self._global_options

    def __repr__(self) -> str:
        """ Pretty print the required packages for logging """
        props = ", ".join(
            f"{k}={repr(getattr(self, k))}"
            for k, v in self.__class__.__dict__.items()
            if isinstance(v, property) and not k.startswith("_"))
        return f"{self.__class__.__name__}({props})"

    def _import_packaging(self) -> None:
        """ Import the packaging library and set the required classes to class attributes. """
        if self._requirement is not None:
            return

        logger.debug("Importing packaging library")
        mark_mod = import_module("packaging.markers")
        req_mod = import_module("packaging.requirements")
        spec_mod = import_module("packaging.specifiers")
        self._marker = mark_mod.Marker
        self._requirement = req_mod.Requirement
        self._specifier = spec_mod.Specifier

    @classmethod
    def _parse_file(cls, file_path: str) -> tuple[list[str], list[str]]:
        """ Parse a requirements file

        Parameters
        ----------
        file_path : str
            The full path to a requirements file to parse

        Returns
        -------
        global_options : list[str]
            Any global options collected from the requirements file
        requirements : list[str]
            The requirements strings from the requirments file
        """
        global_options = []
        requirements = []
        with open(file_path, encoding="utf8") as f:
            for line in f:
                line = line.strip()  # Skip blanks, comments and nested requirement files
                if not line or line.startswith(("#", "-r")):
                    continue

                line = line.split("#", maxsplit=1)[0]  # Strip inline comments

                if line.startswith("-"):  # Collect global option
                    global_options.append(line)
                    continue
                requirements.append(line)  # Collect requirement

        logger.debug("Parsed requirements file '%s'. global_options: %s, requirements: %s",
                     os.path.basename(file_path), global_options, requirements)
        return global_options, requirements

    def parse_requirements(self, packages: list[str]) -> list[Requirement]:
        """ Drop in replacement for deprecated pkg_resources.parse_requirements

        Parameters
        ----------
        packages: list[str]
            List of packages formatted from a requirements.txt file

        Returns
        -------
        list[:class:`packaging.Requirement`]
            List of Requirement objects
        """
        self._import_packaging()
        assert self._requirement is not None
        requirements = [self._requirement(p) for p in packages]
        retval = [r for r in requirements if r.marker is None or r.marker.evaluate()]
        if len(retval) != len(requirements):
            logger.debug("Filtered invalid packages %s",
                         [(r.name, r.marker) for r in set(requirements).difference(set(retval))])
        logger.debug("Parsed requirements %s: %s", packages, retval)
        return retval

    def _parse_options(self, options: list[str]) -> list[str]:
        """ Parse global options from a requirements file and only return valid options

        Parameters
        ----------
        options: list[str]
            List of global options formatted from a requirements.txt file

        Returns
        -------
        list[str]
            List of global options valid for the running system
        """
        if not options:
            return options
        assert self._marker is not None
        retval = []
        for opt in options:
            if ";" not in opt:
                retval.append(opt)
                continue
            directive, marker = opt.split(";", maxsplit=1)
            if not self._marker(marker.strip()).evaluate():
                logger.debug("Filtered invalid option: '%s'", opt)
                continue
            retval.append(directive.strip())

        logger.debug("Selected options: %s", retval)
        return retval

    def _load_requirements(self) -> None:
        """ Parse the requirements files and populate information to :attr:`_requirements` """
        req_path = os.path.dirname(os.path.realpath(__file__))
        base_file = os.path.join(req_path, "_requirements_base.txt")
        req_files = [os.path.join(req_path, f)
                     for f in os.listdir(req_path)
                     if f.startswith("requirements_")
                     and os.path.splitext(f)[-1] == ".txt"]

        opts_base, reqs_base = self._parse_file(base_file)
        parsed_reqs_base = self.parse_requirements(reqs_base)
        parsed_opts_base = self._parse_options(opts_base)

        if self._include_dev:
            opts_dev, reqs_dev = self._parse_file(os.path.join(req_path, "_requirements_dev.txt"))
            opts_base += opts_dev
            parsed_reqs_base += self.parse_requirements(reqs_dev)
            parsed_opts_base += self._parse_options(opts_dev)

        for req_file in req_files:
            backend = os.path.splitext(os.path.basename(req_file))[0].replace("requirements_", "")
            assert backend
            opts, reqs = self._parse_file(req_file)
            self._requirements[backend] = parsed_reqs_base + self.parse_requirements(reqs)
            self._global_options[backend] = parsed_opts_base + self._parse_options(opts)
            logger.debug("[%s] Requirements: %s , Options: %s",
                         backend, self._requirements[backend], self._global_options[backend])


if __name__ == "__main__":
    print(Requirements(include_dev=True))
