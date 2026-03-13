#!/usr/bin/env python3
""" Plugin loader for Faceswap extract, training and convert tasks """
from __future__ import annotations
import ast
import logging
import os
import typing as T

from importlib import import_module

from lib.utils import full_path_split, get_module_objects, PROJECT_ROOT

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from plugins.extract.base import ExtractPlugin
    from plugins.train.model._base import ModelBase
    from plugins.train.trainer._base import TrainerBase

logger = logging.getLogger(__name__)


def get_extractors() -> dict[str, list[str]]:  # noqa[C901]
    """ Obtain a dictionary of all available extraction plugins by plugin type

    Returns
    -------
    dict[str, list[:class:`plugins.extract._base.ExtractPlugin`]]
        A list of all available plugins for each extraction plugin type
    """
    root = os.path.join(PROJECT_ROOT, "plugins", "extract")
    folders = sorted(os.path.join(root, fldr) for fldr in os.listdir(root)
                     if os.path.isdir(os.path.join(root, fldr))
                     and not fldr.startswith("_"))
    retval: dict[str, list[str]] = {}
    for fldr in folders:
        files = sorted(os.path.join(fldr, fname) for fname in os.listdir(fldr)
                       if os.path.isfile(os.path.join(fldr, fname))
                       and fname.endswith(".py")
                       and not fname.startswith("_")
                       and not fname.endswith("_defaults.py"))
        mods = []
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as pfile:
                    tree = ast.parse(pfile.read())
            except Exception:  # pylint:disable=broad-except
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for base in node.bases:
                    if not isinstance(base, ast.Name):
                        continue
                    if base.id in ("ExtractPlugin", "FacePlugin"):
                        rel_path = os.path.splitext(fpath.replace(PROJECT_ROOT, "")[1:])[0]
                        mods.append(".".join(full_path_split(rel_path) + [node.name]))
        if mods:
            retval[os.path.basename(fldr)] = list(sorted(mods))
    logger.debug("Extraction plugins: %s", retval)
    return retval


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
    extract_plugins = get_extractors()

    @classmethod
    def get_extractor(cls,
                      plugin_type: T.Literal["align", "detect", "identity", "mask"],
                      name: str) -> ExtractPlugin:
        """ Return requested extractor plugin

        Parameters
        ----------
        type : Literal["align", "detect", "identity", "mask"]
            The type of extractor plugin to obtain
        name: str
            The name of the requested extractor plugin

        Returns
        -------
        type[:class:`plugins.extract.ExtractPlugin`]
            An extraction plugin

        Raises
        ------
        ValueError
            If an invalid plugin type or plugin name is selected
        """
        if plugin_type not in cls.extract_plugins:
            raise ValueError(f"{plugin_type} is not a valid plugin type. Select from "
                             f"{list(cls.extract_plugins)}")
        plugins = cls.extract_plugins[plugin_type]
        mods = [p.split(".")[-2] for p in plugins]
        real_name = name.lower().replace("-", "_")
        if real_name not in mods:
            raise ValueError(f"{name} is not a valid {plugin_type} plugin. Select from {mods}")

        mod, obj = plugins[mods.index(real_name)].rsplit(".", maxsplit=1)
        logger.debug("Loading '%s' from '%s'", plugin_type, name)

        module = import_module(mod)

        retval = getattr(module, obj)()
        logger.info("Loading %s from %s", plugin_type.title(), retval.name)
        return retval

    @staticmethod
    def get_model(name: str, disable_logging: bool = False) -> type[ModelBase]:
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
    def get_trainer(name: str, disable_logging: bool = False) -> type[TrainerBase]:
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
            The name of the requested plugin
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

    @classmethod
    def get_available_extractors(cls,
                                 extractor_type: T.Literal["align", "detect", "identity", "mask"],
                                 add_none: bool = False,
                                 extend_plugin: bool = False) -> list[str]:
        """ Return a list of available extractors of the given type

        Parameters
        ----------
        extractor_type : Literal["align", "detect", "identity", "mask"]
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
        if extractor_type not in cls.extract_plugins:
            raise ValueError(f"{extractor_type} is not a valid plugin type. Select from "
                             f"{list(cls.extract_plugins)}")
        plugins = [x.split(".")[-2].replace("_", "-") for x in cls.extract_plugins[extractor_type]]
        if extend_plugin and extractor_type == "mask":
            extendable = ["bisenet-fp", "custom"]
            for plugin in extendable:
                if plugin not in plugins:
                    continue
                plugins.remove(plugin)
                plugins.extend([f"{plugin}_face", f"{plugin}_head"])
        plugins = sorted(plugins)
        if add_none:
            plugins.insert(0, "none")
        return plugins

    @staticmethod
    def get_available_models() -> list[str]:
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
    def get_available_convert_plugins(convert_category: str, add_none: bool = True) -> list[str]:
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


__all__ = get_module_objects(__name__)
