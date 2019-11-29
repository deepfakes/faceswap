#!/usr/bin/env python3
""" Plugin loader for Faceswap extract, training and convert tasks """

import logging
import os
from importlib import import_module

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PluginLoader():
    """ Retrieve, or get information on, Faceswap plugins

    Return a specific plugin, list available plugins, or get the default plugin for a
    task.

    Example
    -------
    >>> from plugins.plugin_loader import PluginLoader
    >>> align_plugins = PluginLoader.get_available_extractors('align')
    >>> aligner = PluginLoader.get_aligner('cv2-dnn')
    """
    @staticmethod
    def get_detector(name, disable_logging=False):
        """ Return requested detector plugin

        Parameters
        ----------
        name: str
            The name of the requested detector plugin
        disable_logging: bool, optional
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        :class:`plugins.extract.detect` object:
            An extraction detector plugin
        """
        return PluginLoader._import("extract.detect", name, disable_logging)

    @staticmethod
    def get_aligner(name, disable_logging=False):
        """ Return requested aligner plugin

        Parameters
        ----------
        name: str
            The name of the requested aligner plugin
        disable_logging: bool, optional
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        :class:`plugins.extract.align` object:
            An extraction aligner plugin
        """
        return PluginLoader._import("extract.align", name, disable_logging)

    @staticmethod
    def get_masker(name, disable_logging=False):
        """ Return requested masker plugin

        Parameters
        ----------
        name: str
            The name of the requested masker plugin
        disable_logging: bool, optional
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        :class:`plugins.extract.mask` object:
            An extraction masker plugin
        """
        return PluginLoader._import("extract.mask", name, disable_logging)

    @staticmethod
    def get_model(name, disable_logging=False):
        """ Return requested training model plugin

        Parameters
        ----------
        name: str
            The name of the requested training model plugin
        disable_logging: bool, optional
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        :class:`plugins.train.model` object:
            A training model plugin
        """
        return PluginLoader._import("train.model", name, disable_logging)

    @staticmethod
    def get_trainer(name, disable_logging=False):
        """ Return requested training trainer plugin

        Parameters
        ----------
        name: str
            The name of the requested training trainer plugin
        disable_logging: bool, optional
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        :class:`plugins.train.trainer` object:
            A training trainer plugin
        """
        return PluginLoader._import("train.trainer", name, disable_logging)

    @staticmethod
    def get_converter(category, name, disable_logging=False):
        """ Return requested converter plugin

        Converters work slightly differently to other faceswap plugins. They are created to do a
        specific task (e.g. color adjustment, mask blending etc.), so multiple plugins will be
        loaded in the convert phase, rather than just one plugin for the other phases.

        Parameters
        ----------
        name: str
            The name of the requested converter plugin
        disable_logging: bool, optional
            Whether to disable the INFO log message that the plugin is being imported.
            Default: `False`

        Returns
        -------
        :class:`plugins.convert` object:
            A converter sub plugin
        """
        return PluginLoader._import("convert.{}".format(category), name, disable_logging)

    @staticmethod
    def _import(attr, name, disable_logging):
        """ Import the plugin's module

        Parameters
        ----------
        name: str
            The name of the requested converter plugin
        disable_logging: bool
            Whether to disable the INFO log message that the plugin is being imported.

        Returns
        -------
        :class:`plugin` object:
            A plugin
        """
        name = name.replace("-", "_")
        ttl = attr.split(".")[-1].title()
        if not disable_logging:
            logger.info("Loading %s from %s plugin...", ttl, name.title())
        attr = "model" if attr == "Trainer" else attr.lower()
        mod = ".".join(("plugins", attr, name))
        module = import_module(mod)
        return getattr(module, ttl)

    @staticmethod
    def get_available_extractors(extractor_type, add_none=False):
        """ Return a list of available extractors of the given type

        Parameters
        ----------
        extractor_type: {'aligner', 'detector', 'masker'}
            The type of extractor to return the plugins for
        add_none: bool, optional
            Append "none" to the list of returned plugins. Default: False
        Returns
        -------
        list:
            A list of the available extractor plugin names for the given type
        """
        extractpath = os.path.join(os.path.dirname(__file__),
                                   "extract",
                                   extractor_type)
        extractors = sorted(item.name.replace(".py", "").replace("_", "-")
                            for item in os.scandir(extractpath)
                            if not item.name.startswith("_")
                            and not item.name.endswith("defaults.py")
                            and item.name.endswith(".py"))
        if add_none:
            extractors.insert(0, "none")
        return extractors

    @staticmethod
    def get_available_models():
        """ Return a list of available training models

        Returns
        -------
        list:
            A list of the available training model plugin names
        """
        modelpath = os.path.join(os.path.dirname(__file__), "train", "model")
        models = sorted(item.name.replace(".py", "").replace("_", "-")
                        for item in os.scandir(modelpath)
                        if not item.name.startswith("_")
                        and not item.name.endswith("defaults.py")
                        and item.name.endswith(".py"))
        return models

    @staticmethod
    def get_default_model():
        """ Return the default training model plugin name

        Returns
        -------
        str:
            The default faceswap training model

        """
        models = PluginLoader.get_available_models()
        return 'original' if 'original' in models else models[0]

    @staticmethod
    def get_available_convert_plugins(convert_category, add_none=True):
        """ Return a list of available converter plugins in the given category

        Parameters
        ----------
        convert_category: {'color', 'mask', 'scaling', 'writer'}
            The category of converter plugin to return the plugins for
        add_none: bool, optional
            Append "none" to the list of returned plugins. Default: True

        Returns
        -------
        list
            A list of the available converter plugin names in the given category
        """

        convertpath = os.path.join(os.path.dirname(__file__),
                                   "convert",
                                   convert_category)
        converters = sorted(item.name.replace(".py", "").replace("_", "-")
                            for item in os.scandir(convertpath)
                            if not item.name.startswith("_")
                            and not item.name.endswith("defaults.py")
                            and item.name.endswith(".py"))
        if add_none:
            converters.insert(0, "none")
        return converters
