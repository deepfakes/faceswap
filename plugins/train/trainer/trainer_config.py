#!/usr/bin/env python3
""" Default configurations for trainers """
import gettext
import logging

from lib.config import ConfigItem
from lib.utils import get_module_objects

logger = logging.getLogger(__name__)


# LOCALES
_LANG = gettext.translation("plugins.train.trainer.train_config",
                            localedir="locales", fallback=True)
_ = _LANG.gettext


def get_defaults() -> tuple[str, str, dict[str, ConfigItem]]:
    """ Obtain the default values for adding to the config.ini file

    Returns
    -------
    helptext : str
        The help text for the training config section
    section : str
        The section name for the config items
    defaults : dict[str, :class:`lib.config.objects.ConfigItem`]
        The option names and config items
    """
    section = "trainer.augmentation"
    helptext = _(
        "Data Augmentation Options.\n"
        "WARNING: The defaults for augmentation will be fine for 99.9% of use cases. "
        "Only change them if you absolutely know what you are doing!")
    defaults = {k: v for k, v in globals().items()
                if isinstance(v, ConfigItem)}
    logger.debug("Training config. Helptext: %s, options: %s", helptext, defaults)
    return helptext, section, defaults


preview_images = ConfigItem(
    datatype=int,
    default=14,
    group=_("evaluation"),
    info=_("Number of sample faces to display for each side in the preview when training."),
    rounding=2,
    min_max=(2, 16))

mask_opacity = ConfigItem(
    datatype=int,
    default=30,
    group=_("evaluation"),
    info=_("The opacity of the mask overlay in the training preview. Lower values are more "
           "transparent."),
    rounding=2,
    min_max=(0, 100))

mask_color = ConfigItem(
    datatype=str,
    default="#ff0000",
    choices="colorchooser",
    group=_("evaluation"),
    info=_("The RGB hex color to use for the mask overlay in the training preview."))

zoom_amount = ConfigItem(
    datatype=int,
    default=5,
    group=_("image augmentation"),
    info=_("Percentage amount to randomly zoom each training image in and out."),
    rounding=1,
    min_max=(0, 25))

rotation_range = ConfigItem(
    datatype=int,
    default=10,
    group=_("image augmentation"),
    info=_("Percentage amount to randomly rotate each training image."),
    rounding=1,
    min_max=(0, 25))

shift_range = ConfigItem(
    datatype=int,
    default=5,
    group=_("image augmentation"),
    info=_("Percentage amount to randomly shift each training image horizontally and "
           "vertically."),
    rounding=1,
    min_max=(0, 25))

flip_chance = ConfigItem(
    datatype=int,
    default=50,
    group=_("image augmentation"),
    info=_("Percentage chance to randomly flip each training image horizontally.\n"
           "NB: This is ignored if the 'no-flip' option is enabled"),
    rounding=1,
    min_max=(0, 75))

color_lightness = ConfigItem(
    datatype=int,
    default=30,
    group=_("color augmentation"),
    info=_("Percentage amount to randomly alter the lightness of each training image.\n"
           "NB: This is ignored if the 'no-augment-color' option is enabled"),
    rounding=1,
    min_max=(0, 75))

color_ab = ConfigItem(
    datatype=int,
    default=8,
    group=_("color augmentation"),
    info=_("Percentage amount to randomly alter the 'a' and 'b' colors of the L*a*b* color "
           "space of each training image.\nNB: This is ignored if the 'no-augment-color' option"
           "is enabled"),
    rounding=1,
    min_max=(0, 50))

color_clahe_chance = ConfigItem(
    datatype=int,
    default=50,
    group=_("color augmentation"),
    info=_("Percentage chance to perform Contrast Limited Adaptive Histogram Equalization on "
           "each training image.\nNB: This is ignored if the 'no-augment-color' option is "
           "enabled"),
    rounding=1,
    min_max=(0, 75),
    fixed=False)

color_clahe_max_size = ConfigItem(
    datatype=int,
    default=4,
    group=_("color augmentation"),
    info=_("The grid size dictates how much Contrast Limited Adaptive Histogram Equalization is "
           "performed on any training image selected for clahe. Contrast will be applied "
           "randomly with a gridsize of 0 up to the maximum. This value is a multiplier "
           "calculated from the training image size.\nNB: This is ignored if the "
           "'no-augment-color' option is enabled"),
    rounding=1,
    min_max=(1, 8))


__all__ = get_module_objects(__name__)
