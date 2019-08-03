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
        """
        Set the global options for training

        Loss Documentation
        MAE https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine
            -learners-should-know-4fb140e9d4b0
        MSE https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine
            -learners-should-know-4fb140e9d4b0
        LogCosh https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine
                -learners-should-know-4fb140e9d4b0
        Smooth L1 https://arxiv.org/pdf/1701.03077.pdf
        L_inf_norm https://medium.com/@montjoile/l0-norm-l1-norm-l2-norm-l-infinity
                   -norm-7a7d18a4f40c
        SSIM http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
        GMSD https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
        """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(title=section,
                         info="Options that apply to all models" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="icnr_init", datatype=bool, default=False,
            info="Use ICNR to tile the default initializer in a repeating pattern. "
                 "This strategy is designed for pairing with sub-pixel / pixel shuffler "
                 "to reduce the 'checkerboard effect' in image reconstruction. "
                 "\n\t https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf")
        self.add_item(
            section=section, title="conv_aware_init", datatype=bool, default=False,
            info="Use Convolution Aware Initialization for convolutional layers. "
                 "This can help eradicate the vanishing and exploding gradient problem "
                 "as well as lead to higher accuracy, lower loss and faster convergence. "
                 "NB This can use more VRAM when creating a new model so you may want to "
                 "lower the batch size for the first run. The batch size can be raised "
                 "again when reloading the model. "
                 "\n\t NB: Building the model will likely take several minutes as the "
                 "calculations for this initialization technique are expensive.")
        self.add_item(
            section=section, title="subpixel_upscaling", datatype=bool, default=False,
            info="Use subpixel upscaling rather than pixel shuffler. These techniques "
                 "are both designed to produce better resolving upscaling than other "
                 "methods. Each perform the same operations, but using different TF opts."
                 "\n\t https://arxiv.org/pdf/1609.05158.pdf")
        self.add_item(
            section=section, title="reflect_padding", datatype=bool, default=False,
            info="Use reflection padding rather than zero padding with convolutions. "
                 "Each convolution must pad the image boundaries to maintain the proper "
                 "sizing. More complex padding schemes can reduce artifacts at the "
                 "border of the image."
                 "\n\t http://www-cs.engr.ccny.cuny.edu/~wolberg/cs470/hw/hw2_pad.txt")
        self.add_item(
            section=section, title="penalized_mask_loss", datatype=bool, default=True,
            info="Image loss function is weighted by mask presence. For areas of "
                 "the image without the facial mask, reconstuction errors will be "
                 "ignored while the masked face area is prioritized. May increase "
                 "overall quality by focusing attention on the core face area.")
        self.add_item(
            section=section, title="loss_function", datatype=str,
            default="mae",
            choices=["mae", "mse", "logcosh", "smooth_l1", "l_inf_norm", "ssim", "gmsd",
                     "pixel_gradient_diff"],
            info="\n\t MAE - Mean absolute error will guide reconstructions of each pixel "
                 "towards its median value in the training dataset. Robust to outliers but as "
                 "a median, it can potentially ignore some infrequent image types in the dataset."
                 "\n\t MSE - Mean squared error will guide reconstructions of each pixel "
                 "towards its average value in the training dataset. As an avg, it will be "
                 "suspectible to outliers and typically produces slightly blurrier results."
                 "\n\t LogCosh - log(cosh(x)) acts similiar to MSE for small errors and to "
                 "MAE for large errors. Like MSE, it is very stable and prevents overshoots "
                 "when errors are near zero. Like MAE, it is robust to outliers."
                 "\n\t Smooth_L1 --- Modification of the MAE loss to correct two of its "
                 "disadvantages. This loss has improved stability and guidance for small errors."
                 "\n\t L_inf_norm --- The L_inf norm will reduce the largest individual pixel "
                 "error in an image. As each largest error is minimized sequentially, the "
                 "overall error is improved. This loss will be extremely focused on outliers."
                 "\n\t SSIM - Structural Similarity Index Metric is a perception-based "
                 "loss that considers changes in texture, luminance, contrast, and local spatial "
                 "statistics of an image. Potentially delivers more realistic looking images."
                 "\n\t GMSD - Gradient Magnitude Similarity Deviation seeks to match "
                 "the global standard deviation of the pixel to pixel differences between two "
                 "images. Similiar in approach to SSIM."
                 "\n\t Pixel_Gradient_Difference - Instead of minimizing the difference between "
                 "the absolute value of each pixel in two reference images, compute the pixel to "
                 "pixel spatial difference in each image and then minimize that difference "
                 "between two images. Allows for large color shifts,but maintains the structure "
                 "of the image.\n")
        self.add_item(section=section, title="mask_type", datatype=str, default="none",
                      choices=get_available_masks(),
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
            section=section, title="learning_rate", datatype=float, default=5e-5,
            min_max=(1e-6, 1e-4), rounding=6, fixed=False,
            info="Learning rate - how fast your network will learn (how large are "
                 "the modifications to the model weights after one batch of training). "
                 "Values that are too large might result in model crashes and the "
                 "inability of the model to find the best solution. "
                 "Values that are too small might be unable to escape from dead-ends "
                 "and find the best global minimum.")
        self.add_item(
            section=section, title="coverage", datatype=float, default=68.75,
            min_max=(62.5, 100.0), rounding=2, fixed=True,
            info="How much of the extracted image to train on. A lower coverage will limit the "
                 "model's scope to a zoomed-in central area while higher amounts can include the "
                 "entire face. A trade-off exists between lower amounts given more detail "
                 "versus higher amounts avoiding noticeable swap transitions. Sensible values to "
                 "use are:"
                 "\n\t62.5%% spans from eyebrow to eyebrow."
                 "\n\t75.0%% spans from temple to temple."
                 "\n\t87.5%% spans from ear to ear."
                 "\n\t100.0%% is a mugshot.")

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
