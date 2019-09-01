#!/usr/bin/env python3
""" Plugin loader for extract, training and model tasks """

import logging
import os
from importlib import import_module

from lib.utils import get_backend

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PluginLoader():
    """ Plugin loader for extract, training and model tasks """
    @staticmethod
    def get_detector(name, disable_logging=False):
        """ Return requested detector plugin """
        return PluginLoader._import("extract.detect", name, disable_logging)

    @staticmethod
    def get_aligner(name, disable_logging=False):
        """ Return requested detector plugin """
        return PluginLoader._import("extract.align", name, disable_logging)

    @staticmethod
    def get_model(name, disable_logging=False):
        """ Return requested model plugin """
        return PluginLoader._import("train.model", name, disable_logging)

    @staticmethod
    def get_trainer(name, disable_logging=False):
        """ Return requested trainer plugin """
        return PluginLoader._import("train.trainer", name, disable_logging)

    @staticmethod
    def get_converter(category, name, disable_logging=False):
        """ Return the converter sub plugin """
        return PluginLoader._import("convert.{}".format(category), name, disable_logging)

    @staticmethod
    def _import(attr, name, disable_logging):
        """ Import the plugin's module """
        name = name.replace("-", "_")
        ttl = attr.split(".")[-1].title()
        if not disable_logging:
            logger.info("Loading %s from %s plugin...", ttl, name.title())
        attr = "model" if attr == "Trainer" else attr.lower()
        mod = ".".join(("plugins", attr, name))
        module = import_module(mod)
        return getattr(module, ttl)

    @staticmethod
    def get_available_extractors(extractor_type):
        """ Return a list of available aligners/detectors """
        extractpath = os.path.join(os.path.dirname(__file__),
                                   "extract",
                                   extractor_type)
        extractors = sorted(item.name.replace(".py", "").replace("_", "-")
                            for item in os.scandir(extractpath)
                            if not item.name.startswith("_")
                            and not item.name.endswith("defaults.py")
                            and item.name.endswith(".py")
                            and item.name != "manual.py")
        # TODO Remove this hacky fix when we move them to the same models
        multi_versions = [extractor.replace("-amd", "")
                          for extractor in extractors if extractor.endswith("-amd")]
        if get_backend() == "amd":
            for extractor in multi_versions:
                extractors.remove(extractor)
        else:
            for extractor in multi_versions:
                extractors.remove("{}-amd".format(extractor))
        return extractors

    @staticmethod
    def get_available_models():
        """ Return a list of available models """
        modelpath = os.path.join(os.path.dirname(__file__), "train", "model")
        models = sorted(item.name.replace(".py", "").replace("_", "-")
                        for item in os.scandir(modelpath)
                        if not item.name.startswith("_")
                        and not item.name.endswith("defaults.py")
                        and item.name.endswith(".py"))
        return models

    @staticmethod
    def get_default_model():
        """ Return the default model """
        models = PluginLoader.get_available_models()
        return 'original' if 'original' in models else models[0]

    @staticmethod
    def get_available_convert_plugins(convert_category, add_none=True):
        """ Return a list of available models """
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
