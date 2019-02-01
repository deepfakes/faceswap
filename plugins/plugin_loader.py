#!/usr/bin/env python3
""" Plugin loader for extract, training and model tasks """

import logging
import os
from importlib import import_module

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PluginLoader():
    """ Plugin loader for extract, training and model tasks """
    @staticmethod
    def get_detector(name):
        """ Return requested detector plugin """
        return PluginLoader._import("extract.detect", name)

    @staticmethod
    def get_aligner(name):
        """ Return requested detector plugin """
        return PluginLoader._import("extract.align", name)

    @staticmethod
    def get_converter(name):
        """ Return requested converter plugin """
        return PluginLoader._import("convert", name)

    @staticmethod
    def get_model(name):
        """ Return requested model plugin """
        return PluginLoader._import("train.model", name)

    @staticmethod
    def get_trainer(name):
        """ Return requested trainer plugin """
        return PluginLoader._import("train.trainer", name)

    @staticmethod
    def _import(attr, name):
        """ Import the plugin's module """
        name = name.replace("-", "_")
        ttl = attr.split(".")[-1].title()
        logger.info("Loading %s from %s plugin...", ttl, name.title())
        attr = "model" if attr == "Trainer" else attr.lower()
        mod = ".".join(("plugins", attr, name))
        module = import_module(mod)
        return getattr(module, ttl)

    @staticmethod
    def get_available_models():
        """ Return a list of available models """
        modelpath = os.path.join(os.path.dirname(__file__), "train", "model")
        models = sorted(item.name.replace(".py", "").replace("_", "-")
                        for item in os.scandir(modelpath)
                        if not item.name.startswith("_")
                        and item.name.endswith(".py"))
        return models

    @staticmethod
    def get_available_converters():
        """ Return a list of available converters """
        converter_path = os.path.join(os.path.dirname(__file__), "convert")
        converters = sorted(item.name.replace(".py", "").replace("_", "-")
                            for item in os.scandir(converter_path)
                            if not item.name.startswith("_")
                            and item.name.endswith(".py"))
        return converters

    @staticmethod
    def get_available_extractors(extractor_type):
        """ Return a list of available models """
        extractpath = os.path.join(os.path.dirname(__file__),
                                   "extract",
                                   extractor_type)
        extractors = sorted(item.name.replace(".py", "").replace("_", "-")
                            for item in os.scandir(extractpath)
                            if not item.name.startswith("_")
                            and item.name.endswith(".py")
                            and item.name != "manual.py")
        return extractors

    @staticmethod
    def get_default_model():
        """ Return the default model """
        models = PluginLoader.get_available_models()
        return 'original' if 'original' in models else models[0]
