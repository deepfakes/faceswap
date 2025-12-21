#!/usr/bin/env python3
""" Default configurations for convert """

import logging
import os

from lib.config import FaceswapConfig

logger = logging.getLogger(__name__)


class _Config(FaceswapConfig):
    """ Config File for Convert """

    def set_defaults(self, helptext=""):
        """ Set the default values for config """
        super().set_defaults(helptext=helptext)
        self._defaults_from_plugin(os.path.dirname(__file__))


_CONFIG: _Config | None = None


def load_config(config_file: str | None = None) -> _Config:
    """ Load the Extraction configuration .ini file

    Parameters
    ----------
    config_file : str | None, optional
        Path to a custom .ini configuration file to load. Default: ``None`` (use default
        configuration file)

    Returns
    -------
    :class:`_Config`
        The loaded convert config object
    """
    global _CONFIG  # pylint:disable=global-statement
    if _CONFIG is None:
        _CONFIG = _Config(configfile=config_file)
    return _CONFIG
