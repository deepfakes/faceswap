#!/usr/bin/env python3
""" Parent class for Adjustments for faceswap.py converter
    Based on: https://gist.github.com/anonymous/d3815aba83a8f79779451262599b0955
    found on https://www.reddit.com/r/deepfakes/ """

import logging

import numpy as np

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = Config(None)


class Adjustment():
    """ Parent class for adjustments """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s')", self.__class__.__name__, arguments)

        self.args = arguments
        self.config = self.get_config()
        self.funcs = list()
        self.func_constants = dict()
        self.add_functions()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def func_names(self):
        """ Override with names of functions. Should match naming schema in config.py
            and be set in the order that they will be executed """
        raise NotImplementedError

    def get_config(self):
        """ Return the config dict for the requested master section """
        master_section = self.__class__.__name__.lower()
        logger.debug("returning config for section: '%s'", master_section)
        config = _CONFIG.get_master_section_config_dict(master_section)
        logger.debug("Got config for master_section '%s': %s", master_section, config)
        return config

    def add_functions(self):
        """ Add the functions to be performed on the swap box """
        for action in self.func_names:
            logger.debug("Adding function: '%s'", action)
            getattr(self, "add_{}_func".format(action))(action)

    def add_function(self, action, do_add):
        """ Add the specified function to self.funcs """
        if not do_add:
            logger.debug("'%s' not selected", action)
            return
        logger.debug("Adding: '%s'", action)
        self.funcs.append(getattr(self, action))

    def do_actions(self, *args, **kwargs):
        """ Perform selected adjustments on face """
        logger.trace("Performing image adjustments")
        # Remove Mask for processing
        reinsert_mask = False
        new_face = kwargs["new_face"]
        if new_face.shape[2] == 4:
            reinsert_mask = True
            final_mask = new_face[:, :, -1]
            new_face = new_face[:, :, :3]
            kwargs["new_face"] = new_face

        for func in self.funcs:
            new_face = func(*args, **kwargs)
            new_face = np.clip(new_face, 0.0, 1.0)

        if reinsert_mask and new_face.shape[2] != 4:
            # Reinsert Mask
            new_face = np.concatenate((new_face, np.expand_dims(final_mask, axis=-1)), -1)
        logger.trace("Performed image adjustments")
        return new_face
