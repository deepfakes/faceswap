#!/usr/bin/env python3
""" Plugin loader for extract, training and model tasks """

import os
from importlib import import_module


class PluginLoader():
    """ Plugin loader for extract, training and model tasks """
    @staticmethod
    def get_detector(name):
        """ Return requested detector plugin """
        if name.lower().startswith("dlib"):
            name = "dlib"
        return PluginLoader._import("extract.detect", name)

    @staticmethod
    def get_aligner(name):
        """ Return requested detector plugin """
        return PluginLoader._import("extract.alignments", name)

    @staticmethod
    def get_extractor(name):
        """ Return requested extractor plugin """
        return PluginLoader._import("extract", name)

    @staticmethod
    def get_converter(name):
        """ Return requested converter plugin """
        return PluginLoader._import("Convert", "Convert_{0}".format(name))

    @staticmethod
    def get_model(name):
        """ Return requested model plugin """
        return PluginLoader._import("Model", "Model_{0}".format(name))

    @staticmethod
    def get_trainer(name):
        """ Return requested trainer plugin """
        return PluginLoader._import("Trainer", "Model_{0}".format(name))

    @staticmethod
    def _import(attr, name):
        """ Import the plugin's module """
        ttl = attr.split(".")[-1].title()
        print("Loading {} from {} plugin...".format(ttl, name.title()))
        mod = ".".join(("plugins", attr.lower(), name))
        module = import_module(mod)
        return getattr(module, ttl)

    @staticmethod
    def get_available_models():
        """ Return a list of available models """
        models = ()
        modelpath = os.path.join(os.path.dirname(__file__), "model")
        for modeldir in next(os.walk(modelpath))[1]:
            if modeldir[0:6].lower() == 'model_':
                models += (modeldir[6:],)
        return models

    @staticmethod
    def get_default_model():
        """ Return the default model """
        models = PluginLoader.get_available_models()
        return 'Original' if 'Original' in models else models[0]
