#!/usr/bin/env python3
""" Base class for Faceswap :mod:`~plugins.convert.mask` Plugins """

import logging

import numpy as np

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _get_config(plugin_name, configfile=None):
    """ Return the :attr:`lib.config.FaceswapConfig.config_dict` for the requested plugin.

    Parameters
    ----------
    plugin_name: str
        The name of the plugin to retrieve the config for
    configfile: str, optional
        Optional location of custom configuration ``ini`` file. If ``None`` then use the default
        config location. Default: ``None``

    Returns
    -------
    dict
        The configuration in dictionary form for the given plugin_name from
         :attr:`lib.config.FaceswapConfig.config_dict`
    """
    return Config(plugin_name, configfile=configfile).config_dict


class Adjustment():
    """ Parent class for Mask Adjustment Plugins.

    All mask plugins must inherit from this class.

    Parameters
    ----------
    mask_type: str
        The type of mask that this plugin is being used for
    output_size: int
        The size, in pixels, of the output from the Faceswap model.
    configfile: str, Optional
        Optional location of custom configuration ``ini`` file. If ``None`` then use the default
        config location. Default: ``None``
    config: :class:`lib.config.FaceswapConfig`, Optional
        Optional pre-loaded :class:`lib.config.FaceswapConfig`. If passed, then this will be used
        over any configuration on disk. If ``None`` then it is ignored. Default: ``None``


    Attributes
    ----------
    config: dict
        The configuration dictionary for this plugin.
    mask_type: str
        The type of mask that this plugin is being used for.
    """
    def __init__(self, mask_type, output_size, configfile=None, config=None):
        logger.debug("Initializing %s: (arguments: '%s', output_size: %s, "
                     "configfile: %s, config: %s)", self.__class__.__name__, mask_type,
                     output_size, configfile, config)
        self.config = self._set_config(configfile, config)
        logger.debug("config: %s", self.config)
        self.mask_type = mask_type
        self._dummy = np.zeros((output_size, output_size, 3), dtype='float32')
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def dummy(self):
        """:class:`numpy.ndarray`: A dummy mask of all zeros of the shape:
        (:attr:`output_size`, :attr:`output_size`, `3`)
        """
        return self._dummy

    @property
    def skip(self):
        """bool: ``True`` if the blur type config attribute is ``None`` otherwise ``False`` """
        return self.config.get("type", None) is None

    def _set_config(self, configfile, config):
        """ Set the correct configuration for the plugin based on whether a config file
        or pre-loaded config has been passed in.

        Parameters
        ----------
        configfile: str
            Location of custom configuration ``ini`` file. If ``None`` then use the
            default config location
        config: :class:`lib.config.FaceswapConfig`
            Pre-loaded :class:`lib.config.FaceswapConfig`. If passed, then this will be
            used over any configuration on disk. If ``None`` then it is ignored.

        Returns
        -------
        dict
            The configuration in dictionary form for the given from
            :attr:`lib.config.FaceswapConfig.config_dict`
        """
        section = ".".join(self.__module__.split(".")[-2:])
        if config is None:
            retval = _get_config(section, configfile=configfile)
        else:
            config.section = section
            retval = config.config_dict
            config.section = None
        logger.debug("Config: %s", retval)
        return retval

    def process(self, *args, **kwargs):
        """ Override for specific mask adjustment plugin processes.

        Input parameters will vary from plugin to plugin.

        Should return a :class:`numpy.ndarray` mask with the plugin's actions applied
        """
        raise NotImplementedError

    def run(self, *args, **kwargs):
        """ Perform selected adjustment on face """
        logger.trace("Performing mask adjustment: (plugin: %s, args: %s, kwargs: %s",
                     self.__module__, args, kwargs)
        retval = self.process(*args, **kwargs)
        return retval
