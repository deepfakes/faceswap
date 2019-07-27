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
            info="\nUse ICNR to tile the default initializer in a repeating pattern. \n"
                 "This strategy is designed for  sub-pixel / pixel shuffler upscaling \n"
                 "and should only be used on upscaling layers. This can help reduce the \n"
                 "'checkerboard effect' when upscaling the image in the decoder.\n"
                 "https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf \n")
        self.add_item(
            section=section, title="conv_aware_init", datatype=bool, default=False,
            info="Use Convolution Aware Initialization for convolutional layers\nThis can help "
                 "eradicate the vanishing and exploding gradient problem as well as lead to "
                 "higher accuracy, lower loss and faster convergence."
                 "\nNB This can use more VRAM when creating a new model so you may want to lower "
                 "the batch size for the first run. The batch size can be raised again when "
                 "reloading the model."
                 "\nNB: Building the model will likely take several minutes as the caluclations "
                 "for this initialization technique are expensive.")
        self.add_item(
            section=section, title="subpixel_upscaling", datatype=bool, default=False,
            info="\nUse subpixel upscaling rather than pixel shuffler. These techniques \n"
                 "are both designed to produce better resolving upscaling than other \n"
                 "methods. Each perform the same operations, but using different TF opts.\n"
                 "https://arxiv.org/pdf/1609.05158.pdf \n")
        self.add_item(
            section=section, title="reflect_padding", datatype=bool, default=False,
            info="\nUse reflection padding rather than zero padding when either \n"
                 "downscaling or using simple convolutions. Each convolution must \n"
                 "pad the image/feature boundaries to maintain the proper sizing. \n"
                 "More complex padding schemes can reduce artifacts at the border \n"
                 "of the image.\n"
                 "http://www-cs.engr.ccny.cuny.edu/~wolberg/cs470/hw/hw2_pad.txt \n")
        self.add_item(
            section=section, title="penalized_mask_loss", datatype=bool, default=True,
            info="\nImage loss function is weighted by mask presence. For areas of \n"
                 "the image without the facial mask, reconstuction errors will be \n"
                 "ignored while the masked face area is prioritized. May increase \n"
                 "overall quality by focusing attention on the core face area.\n")
        self.add_item(
            section=section, title="image_loss_function", datatype=str,
            default="Mean_Absolute_Error",
            choices=["Mean_Absolute_Error", "Mean_Squared_Error", "LogCosh",
                     "Smooth_L1", "L_inf_norm", "SSIM", "GMSD", "Pixel_Gradient_Difference"],
            info="\nGeneral Loss Discussion \n"
                 "Whenever we train a machine learning model, our goal is to find \n"
                 "the point that minimizes a loss function. Of course, any function \n"
                 "reaches a minimum when the prediction is exactly equal to the true \n"
                 "value. The difference in loss functions is what type of artifacts \n"
                 "they avoid and/or cause and how susceptible they are to outliers"
                 "http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/\n"
                 "lecturenote10.html\n"
                 "\nMean_Absolute_Error ---\n"
                 "MAE is the sum of absolute differences between our target and \n"
                 "predicted values. This loss will guide reconstructions of each \n"
                 "pixel towards its median value in the training dataset. Robust to \n"
                 "outliers but as a median, it can potentially ignore a minority of \n"
                 "the dataset"
                 "https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine\n"
                 "-learners-should-know-4fb140e9d4b0\n"
                 "\nMean_Squared_Error ---\n"
                 "MSE is the sum of squared distances between our target and \n"
                 "predicted values. This loss will guide reconstructions of each \n"
                 "pixel towards its average value in the training dataset. As an avg. \n"
                 "it will be suspectible to outliers and typically produces slightly \n"
                 "blurrier results."
                 "https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine\n"
                 "-learners-should-know-4fb140e9d4b0\n"
                 "\nLogCosh ---\n"
                 "log(cosh(x)) is approximately equal to MSE / 2 for small errors \n"
                 "and to MAE - log(2) for large errors. Like MSE, it is differentiable \n"
                 "and has a declining gradient for increasingly small errors. Like MAE, \n"
                 "it is robust to outliers."
                 "https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine\n"
                 "-learners-should-know-4fb140e9d4b0\n"
                 "\nSmooth_L1 ---\n"
                 "Modification of the MAE loss to correct two of its disadvantages. \n"
                 "This loss is differentiable at zero and has a non-constant loss \n"
                 "gradient near zero but closely tracks the MAE loss for medium/large \n"
                 "errors."
                 "https://arxiv.org/pdf/1701.03077.pdf\n"
                 "\nL_inf_norm ---\n"
                 "MAE is also known as the L1_norm, and MSE is also known as the L2_norm. \n"
                 "They respectively minimize the absolute value of the error between \n"
                 "images raised to the L power. The L_inf norm therefore has a loss \n"
                 "function that is equal to the maximum pixel error between the two \n"
                 "images, wherever that pixel is located. Minimzing this loss will \n"
                 "limit the maximum error and will be extrenely focused on outliers."
                 "https://medium.com/@montjoile/l0-norm-l1-norm-l2-norm-l-infinity\n"
                 "-norm-7a7d18a4f40c\n"
                 "\nSSIM ---\n Use Structural Similarity Index Metric as a loss function \n"
                 "for training the neural net's image reconstruction in lieu of \n"
                 "Mean Absolute Error. Potentially better textural, second-order \n"
                 "statistics, and translation invariance than MAE.\n"
                 "http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf\n"
                 "\nGMSD ---\n"
                 "Gradient magnitude similarity deviation (GMSD). The image pixel \n"
                 "to pixel gradients are sensitive to image distortions, where \n"
                 "different local structures in a distorted image suffer different \n"
                 "degrees of degradations. Global variation of gradient based local \n"
                 "quality maps are used for overall image quality prediction. The \n"
                 "pixel-wise gradient magnitude similarity (GMS) between the reference \n"
                 "and distorted images combined with a novel pooling strategy – the \n"
                 "standard deviation of the GMS map – can predict perceptual image \n"
                 "quality accurately."
                 "https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf\n"
                 "\nPixel_Gradient_Difference ---\n"
                 "Instead of minimizing the difference between the absolute value of \n"
                 "each pixel in two reference images, compute the pixel to pixel \n"
                 "spatial difference in each image and then minimize the difference \n"
                 "between these two gradient maps. Allows for large color shifts,but \n"
                 "maintains the structure of the image."
                 )
        self.add_item(
            section=section, title="learning_rate", datatype=float, default=5e-5,
            min_max=(1e-6, 1e-4), rounding=6, fixed=False,
            info="Learning rate - how fast your network will learn (how large are \n"
                 "the modifications to the model weights after one batch of training).\n"
                 "Values that are too large might result in model crashes and the \n"
                 "inability of the model to find the best solution.\n"
                 "Values that are too small might be unable to escape from dead-ends \n"
                 "and find the best global minimum.")

        # << ORIGINAL MODEL OPTIONS >> #
        section = "model.original"
        self.add_section(title=section,
                         info="Original Faceswap Model" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other.")

        # << LIGHTWEIGHT MODEL OPTIONS >> #
        section = "model.lightweight"
        self.add_section(title=section,
                         info="A lightweight version of the Original Faceswap Model, designed to "
                              "run on lower end GPUs (~2GB).\nDon't expect great results, but it "
                              "allows users with lower end cards to play with the "
                              "software." + ADDITIONAL_INFO)

        # << DFAKER OPTIONS >> #
        section = "model.dfaker"
        self.add_section(title=section,
                         info="Dfaker Model (Adapted from https://github.com/dfaker/df)" +
                         ADDITIONAL_INFO)

        # << DFL MODEL OPTIONS >> #
        section = "model.dfl_h128"
        self.add_section(title=section,
                         info="DFL H128 Model (Adapted from "
                              "https://github.com/iperov/DeepFaceLab)" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other.")

        # << IAE MODEL OPTIONS >> #
        section = "model.iae"
        self.add_section(title=section,
                         info="Intermediate Auto Encoder. Based on Original Model, uses "
                              "intermediate layers to try to better get details" + ADDITIONAL_INFO)

        # << UNBALANCED MODEL OPTIONS >> #
        section = "model.unbalanced"
        self.add_section(title=section,
                         info="An unbalanced model with adjustable input size options.\nThis is "
                              "an unbalanced model so b>a swaps may not work "
                              "well" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other. NB: lowmem will "
                 "override cutom nodes and complexity settings.")
        self.add_item(
            section=section, title="clipnorm", datatype=bool, default=False,
            info="Controls gradient clipping of the optimizer. Can prevent model corruption at "
                 "the expense of VRAM")
        self.add_item(
            section=section, title="nodes", datatype=int, default=1024, rounding=64,
            min_max=(512, 4096),
            info="Number of nodes for decoder. Don't change this unless you "
                 "know what you are doing!")
        self.add_item(
            section=section, title="complexity_encoder", datatype=int, default=128,
            rounding=16, min_max=(64, 1024),
            info="Encoder Convolution Layer Complexity. sensible ranges: "
                 "128 to 160")
        self.add_item(
            section=section, title="complexity_decoder_a", datatype=int, default=384,
            rounding=16, min_max=(64, 1024),
            info="Decoder A Complexity.")
        self.add_item(
            section=section, title="complexity_decoder_b", datatype=int, default=512,
            rounding=16, min_max=(64, 1024),
            info="Decoder B Complexity.")
        self.add_item(
            section=section, title="input_size", datatype=int, default=128,
            rounding=64, min_max=(64, 512),
            info="Resolution (in pixels) of the image to train on.\n"
                 "BE AWARE Larger resolution will dramatically increase"
                 "VRAM requirements.\n"
                 "Make sure your resolution is divisible by 64 (e.g. 64, 128, 256 etc.).\n"
                 "NB: Your faceset must be at least 1.6x larger than your required input size.\n"
                 "    (e.g. 160 is the maximum input size for a 256x256 faceset)")

        # << PEGASUS MODEL OPTIONS >> #
        section = "model.realface"
        self.add_section(title=section,
                         info="An extra detailed variant of Original model.\n"
                              "Incorporates ideas from Bryanlyon and inspiration from the Villain "
                              "model.\n"
                              "Requires about 6GB-8GB of VRAM (batchsize 8-16)." + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="input_size", datatype=int, default=64,
            rounding=16, min_max=(64, 128),
            info="Resolution (in pixels) of the input image to train on.\n"
                 "BE AWARE Larger resolution will dramatically increase"
                 "VRAM requirements.\n"
                 "Higher resolutions may increase prediction accuracy, but does not effect the "
                 "resulting output size.\n"
                 "Must be between 64 and 128 and be divisible by 16.")
        self.add_item(
            section=section, title="output_size", datatype=int, default=128,
            rounding=16, min_max=(64, 256),
            info="Output image resolution (in pixels).\n"
                 "Be aware that larger resolution will increase VRAM requirements.\n"
                 "NB: Must be between 64 and 256 and be divisible by 16.")
        self.add_item(
            section=section, title="dense_nodes", datatype=int, default=1536, rounding=64,
            min_max=(768, 2048),
            info="Number of nodes for decoder. Might affect your model's ability to learn in "
                 "general.\n"
                 "Note that: Lower values will affect the ability to predict details.")
        self.add_item(
            section=section, title="complexity_encoder", datatype=int, default=128,
            min_max=(96, 160), rounding=4,
            info="Encoder Convolution Layer Complexity. sensible ranges: "
                 "128 to 150")
        self.add_item(
            section=section, title="complexity_decoder", datatype=int, default=512,
            rounding=4, min_max=(512, 544),
            info="Decoder Complexity.")

        # << VILLAIN MODEL OPTIONS >> #
        section = "model.villain"
        self.add_section(title=section,
                         info="A Higher resolution version of the Original "
                              "Model by VillainGuy.\nExtremely VRAM heavy. Full model requires "
                              "9GB+ for batchsize 16" + ADDITIONAL_INFO)
        self.add_item(
            section=section, title="lowmem", datatype=bool, default=False,
            info="Lower memory mode. Set to 'True' if having issues with VRAM useage.\nNB: Models "
                 "with a changed lowmem mode are not compatible with each other.")

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
