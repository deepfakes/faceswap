#!/usr/bin/env python3
""" Default configurations for models """

import logging
import os
import sys

from importlib import import_module

from lib.config import FaceswapConfig
from lib.model.masks import get_available_masks
from lib.utils import full_path_split

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

ADDITIONAL_INFO = ("\nNB: Unless specifically stated, values changed here will only take effect "
                   "when creating a new model.")


class Config(FaceswapConfig):
    """ Config File for Models """
    # pylint: disable=too-many-statements
    def set_defaults(self):
        """ Set the default values for config """
        logger.debug("Setting defaults")
        self.set_globals()
        current_dir = os.path.dirname(__file__)
        for dirpath, _, filenames in os.walk(current_dir):
            default_files = [fname for fname in filenames if fname.endswith("_defaults.py")]
            if not default_files:
                continue
            base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
            import_path = ".".join(full_path_split(dirpath.replace(base_path, ""))[1:])
            plugin_type = import_path.split(".")[-1]
            for filename in default_files:
                self.load_module(filename, import_path, plugin_type)

    def set_globals(self):
        """ Set the global options for training """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(title=section,
                         info="Options that apply to all models" + ADDITIONAL_INFO)
        self.add_item(section=section, title="mask_type", datatype=str, default="none",
                      choices=get_available_masks(), gui_radio=True,
                      info="The mask to be used for training:"
                           "\n\t none: Doesn't use any mask."
                           "\n\t components: An improved face hull mask using a facehull of 8 "
                           "facial parts"
                           "\n\t dfl_full: An improved face hull mask using a facehull of 3 "
                           "facial parts"
                           "\n\t extended: Based on components mask. Extends the eyebrow points "
                           "to further up the forehead. May perform badly on difficult angles."
                           "\n\t facehull: Face cutout based on landmarks")
        self.add_item(
            section=section, title="icnr_init", datatype=bool, default=False,
            info="Use ICNR Kernel Initializer for upscaling.\nThis can help reduce the "
                 "'checkerboard effect' when upscaling the image.")
        self.add_item(
            section=section, title="subpixel_upscaling", datatype=bool, default=False,
            info="Use subpixel upscaling rather than pixel shuffler.\n"
                 "Might increase speed at cost of VRAM")
        self.add_item(
            section=section, title="reflect_padding", datatype=bool, default=False,
            info="Use reflect padding rather than zero padding. Only enable this option if the "
                 "model you are training has a distinct line appearing around the edge of the "
                 "swap area.")
        self.add_item(
            section=section, title="dssim_loss", datatype=bool, default=True,
            info="Use DSSIM for Loss rather than Mean Absolute Error\n"
                 "May increase overall quality.")
        self.add_item(
            section=section, title="penalized_mask_loss", datatype=bool, default=True,
            info="If using a mask, This penalizes the loss for the masked area, to give higher "
                 "priority to the face area. \nShould increase overall quality and speed up "
                 "training. This should probably be left at True")
        logger.debug("Set global config")

    def load_module(self, filename, module_path, plugin_type):
        """ Load the defaults module and add defaults """
        logger.debug("Adding defaults: (filename: %s, module_path: %s, plugin_type: %s",
                     filename, module_path, plugin_type)
        module = os.path.splitext(filename)[0]
        section = ".".join((plugin_type, module.replace("_defaults", "")))
        logger.debug("Importing defaults module: %s.%s", module_path, module)
        mod = import_module("{}.{}".format(module_path, module))
        helptext = mod._HELPTEXT  # pylint:disable=protected-access
        helptext += ADDITIONAL_INFO if module_path.endswith("model") else ""
        self.add_section(title=section, info=helptext)
        for key, val in mod._DEFAULTS.items():  # pylint:disable=protected-access
            self.add_item(section=section, title=key, **val)
        logger.debug("Added defaults: %s", section)
