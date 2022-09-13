#!/usr/bin/env python3
""" Plugin loader for Faceswap extract, training and convert tasks """

import logging
import os
import sys
from importlib import import_module
from typing import Callable, List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from plugins.extract.detect._base import Detector
    from plugins.extract.align._base import Aligner
    from plugins.extract.mask._base import Masker
    from plugins.train.model._base import ModelBase
    from plugins.train.trainer._base import TrainerBase

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

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
    def get_detector(name: str, disable_logging: bool = False) -> Type["Detector"]:
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
    def get_aligner(name: str, disable_logging: bool = False) -> Type["Aligner"]:
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
    def get_masker(name: str, disable_logging: bool = False) -> Type["Masker"]:
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
    def get_model(name: str, disable_logging: bool = False) -> Type["ModelBase"]:
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
    def get_trainer(name: str, disable_logging: bool = False) -> Type["TrainerBase"]:
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
    def get_converter(category: str, name: str, disable_logging: bool = False) -> Callable:
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
        return PluginLoader._import(f"convert.{category}", name, disable_logging)

    @staticmethod
    def _import(attr: str, name: str, disable_logging: bool):
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
    def get_available_extractors(extractor_type: Literal["align", "detect", "mask"],
                                 add_none: bool = False,
                                 extend_plugin: bool = False) -> List[str]:
        """ Return a list of available extractors of the given type

        Parameters
        ----------
        extractor_type: {'align', 'detect', 'mask'}
            The type of extractor to return the plugins for
        add_none: bool, optional
            Append "none" to the list of returned plugins. Default: False
        extend_plugin: bool, optional
            Some plugins have configuration options that mean that multiple 'pseudo-plugins'
            can be generated based on their settings. An example of this is the bisenet-fp mask
            which, whilst selected as 'bisenet-fp' can be stored as 'bisenet-fp-face' and
            'bisenet-fp-head' depending on whether hair has been included in the mask or not.
            ``True`` will generate each pseudo-plugin, ``False`` will generate the original
            plugin name. Default: ``False``

        Returns
        -------
        list:
            A list of the available extractor plugin names for the given type
        """
        extractpath = os.path.join(os.path.dirname(__file__),
                                   "extract",
                                   extractor_type)
        extractors = [item.name.replace(".py", "").replace("_", "-")
                      for item in os.scandir(extractpath)
                      if not item.name.startswith("_")
                      and not item.name.endswith("defaults.py")
                      and item.name.endswith(".py")]
        extendable = ["bisenet-fp", "custom"]
        if extend_plugin and extractor_type == "mask" and any(ext in extendable
                                                              for ext in extractors):
            for msk in extendable:
                extractors.remove(msk)
                extractors.extend([f"{msk}_face", f"{msk}_head"])

        extractors = sorted(extractors)
        if add_none:
            extractors.insert(0, "none")
        return extractors

    @staticmethod
    def get_available_models() -> List[str]:
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
    def get_default_model() -> str:
        """ Return the default training model plugin name

        Returns
        -------
        str:
            The default faceswap training model

        """
        models = PluginLoader.get_available_models()
        return 'original' if 'original' in models else models[0]

    @staticmethod
    def get_available_convert_plugins(convert_category: str, add_none: bool = True) -> List[str]:
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
