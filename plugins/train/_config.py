#!/usr/bin/env python3
""" Default configurations for models """

import logging
import os
import sys

from importlib import import_module

from lib.config import FaceswapConfig
from lib.utils import full_path_split
from plugins.plugin_loader import PluginLoader

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
            section=section, title="coverage", datatype=float, default=68.75,
            min_max=(62.5, 100.0), rounding=2, fixed=True, group="face",
            info="How much of the extracted image to train on. A lower coverage will limit the "
                 "model's scope to a zoomed-in central area while higher amounts can include the "
                 "entire face. A trade-off exists between lower amounts given more detail "
                 "versus higher amounts avoiding noticeable swap transitions. Sensible values to "
                 "use are:"
                 "\n\t62.5%% spans from eyebrow to eyebrow."
                 "\n\t75.0%% spans from temple to temple."
                 "\n\t87.5%% spans from ear to ear."
                 "\n\t100.0%% is a mugshot.")
        self.add_item(
            section=section, title="mask_type", datatype=str, default="none",
            choices=PluginLoader.get_available_extractors("mask", add_none=True), group="mask",
            gui_radio=True,
            info="The mask to be used for training. If you have selected 'Learn Mask' or "
                 "'Penalized Mask Loss' you must select a value other than 'none'. The required "
                 "mask should have been selected as part of the Extract process. If it does not "
                 "exist in the alignments file then it will be generated prior to training "
                 "commencing."
                 "\n\tnone: Don't use a mask."
                 "\n\tcomponents: Mask designed to provide facial segmentation based on the "
                 "positioning of landmark locations. A convex hull is constructed around the "
                 "exterior of the landmarks to create a mask."
                 "\n\textended: Mask designed to provide facial segmentation based on the "
                 "positioning of landmark locations. A convex hull is constructed around the "
                 "exterior of the landmarks and the mask is extended upwards onto the forehead."
                 "\n\tvgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                 "faces clear of obstructions. Profile faces and obstructions may result in "
                 "sub-par performance."
                 "\n\tvgg-obstructed: Mask designed to provide smart segmentation of mostly "
                 "frontal faces. The mask model has been specifically trained to recognize "
                 "some facial obstructions (hands and eyeglasses). Profile faces may result in "
                 "sub-par performance."
                 "\n\tunet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                 "faces. The mask model has been trained by community members and will need "
                 "testing for further description. Profile faces may result in sub-par "
                 "performance.")
        self.add_item(
            section=section, title="mask_blur_kernel", datatype=int, min_max=(0, 9),
            rounding=1, default=3, group="mask",
            info="Apply gaussian blur to the mask input. This has the effect of smoothing the "
                 "edges of the mask, which can help with poorly calculated masks and give less "
                 "of a hard edge to the predicted mask. The size is in pixels (calculated from "
                 "a 128px mask). Set to 0 to not apply gaussian blur. This value should be odd, "
                 "if an even number is passed in then it will be rounded to the next odd number.")
        self.add_item(
            section=section, title="mask_threshold", datatype=int, default=4,
            min_max=(0, 50), rounding=1, group="mask",
            info="Sets pixels that are near white to white and near black to black. Set to 0 for "
                 "off.")
        self.add_item(
            section=section, title="learn_mask", datatype=bool, default=False, group="mask",
            info="Dedicate a portion of the model to learning how to duplicate the input "
                 "mask. Increases VRAM usage in exchange for learning a quick ability to try "
                 "to replicate more complex mask models.")
        self.add_item(
            section=section, title="icnr_init", datatype=bool,
            default=False, group="initialization",
            info="Use ICNR to tile the default initializer in a repeating pattern. "
                 "This strategy is designed for pairing with sub-pixel / pixel shuffler "
                 "to reduce the 'checkerboard effect' in image reconstruction. "
                 "\n\t https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf")
        self.add_item(
            section=section, title="conv_aware_init", datatype=bool,
            default=False, group="initialization",
            info="Use Convolution Aware Initialization for convolutional layers. "
                 "This can help eradicate the vanishing and exploding gradient problem "
                 "as well as lead to higher accuracy, lower loss and faster convergence.\nNB:"
                 "\n\t This can use more VRAM when creating a new model so you may want to "
                 "lower the batch size for the first run. The batch size can be raised "
                 "again when reloading the model. "
                 "\n\t Multi-GPU is not supported for this option, so you should start the model "
                 "on a single GPU. Once training has started, you can stop training, enable "
                 "multi-GPU and resume."
                 "\n\t Building the model will likely take several minutes as the calculations "
                 "for this initialization technique are expensive. This will only impact starting "
                 "a new model.")
        self.add_item(
            section=section, title="reflect_padding", datatype=bool,
            default=False, group="network",
            info="Use reflection padding rather than zero padding with convolutions. "
                 "Each convolution must pad the image boundaries to maintain the proper "
                 "sizing. More complex padding schemes can reduce artifacts at the "
                 "border of the image."
                 "\n\t http://www-cs.engr.ccny.cuny.edu/~wolberg/cs470/hw/hw2_pad.txt")
        self.add_item(
            section=section, title="penalized_mask_loss", datatype=bool,
            default=True, group="loss",
            info="Image loss function is weighted by mask presence. For areas of "
                 "the image without the facial mask, reconstuction errors will be "
                 "ignored while the masked face area is prioritized. May increase "
                 "overall quality by focusing attention on the core face area.")
        self.add_item(
            section=section, title="loss_function", datatype=str, group="loss",
            default="mae",
            choices=["mae", "mse", "logcosh", "smooth_loss", "l_inf_norm", "ssim", "gmsd",
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
        self.add_item(
            section=section, title="learning_rate", datatype=float, default=5e-5,
            min_max=(1e-6, 1e-4), rounding=6, fixed=False, group="optimizer",
            info="Learning rate - how fast your network will learn (how large are "
                 "the modifications to the model weights after one batch of training). "
                 "Values that are too large might result in model crashes and the "
                 "inability of the model to find the best solution. "
                 "Values that are too small might be unable to escape from dead-ends "
                 "and find the best global minimum.")

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
