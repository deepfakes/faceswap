#!/usr/bin/env python3
""" Default configurations for models """

import logging
import os

from lib.config import FaceswapConfig
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
        self._set_globals()
        self._set_loss()
        self._defaults_from_plugin(os.path.dirname(__file__))

    def _set_globals(self):
        """ Set the global options for training """
        logger.debug("Setting global config")
        section = "global"
        self.add_section(title=section,
                         info="Options that apply to all models" + ADDITIONAL_INFO)
        self.add_item(
            section=section,
            title="centering",
            datatype=str,
            gui_radio=True,
            default="face",
            choices=["face", "head", "legacy"],
            fixed=True,
            group="face",
            info="How to center the training image. The extracted images are centered on the "
                 "middle of the skull based on the face's estimated pose. A subsection of these "
                 "images are used for training. The centering used dictates how this subsection "
                 "will be cropped from the aligned images."
                 "\n\tface: Centers the training image on the center of the face, adjusting for "
                 "pitch and yaw."
                 "\n\thead: Centers the training image on the center of the head, adjusting for "
                 "pitch and yaw. NB: You should only select head centering if you intend to "
                 "include the full head (including hair) in the final swap. This may give mixed "
                 "results. Additionally, it is only worth choosing head centering if you are "
                 "training with a mask that includes the hair (e.g. BiSeNet-FP-Head)."
                 "\n\tlegacy: The 'original' extraction technique. Centers the training image "
                 "near the tip of the nose with no adjustment. Can result in the edges of the "
                 "face appearing outside of the training area.")
        self.add_item(
            section=section,
            title="coverage",
            datatype=float,
            default=87.5,
            min_max=(62.5, 100.0),
            rounding=2,
            fixed=True,
            group="face",
            info="How much of the extracted image to train on. A lower coverage will limit the "
                 "model's scope to a zoomed-in central area while higher amounts can include the "
                 "entire face. A trade-off exists between lower amounts given more detail "
                 "versus higher amounts avoiding noticeable swap transitions. For 'Face' "
                 "centering you will want to leave this above 75%. For Head centering you will "
                 "most likely want to set this to 100%. Sensible values for 'Legacy' "
                 "centering are:"
                 "\n\t62.5% spans from eyebrow to eyebrow."
                 "\n\t75.0% spans from temple to temple."
                 "\n\t87.5% spans from ear to ear."
                 "\n\t100.0% is a mugshot.")

        self.add_item(
            section=section,
            title="icnr_init",
            datatype=bool,
            default=False,
            group="initialization",
            info="Use ICNR to tile the default initializer in a repeating pattern. "
                 "This strategy is designed for pairing with sub-pixel / pixel shuffler "
                 "to reduce the 'checkerboard effect' in image reconstruction. "
                 "\n\t https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf")
        self.add_item(
            section=section,
            title="conv_aware_init",
            datatype=bool,
            default=False,
            group="initialization",
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
            section=section,
            title="optimizer",
            datatype=str,
            gui_radio=True,
            group="optimizer",
            default="adam",
            choices=["adabelief", "adam", "nadam", "rms-prop"],
            info="The optimizer to use."
                 "\n\t adabelief - Adapting Stepsizes by the Belief in Observed Gradients. An "
                 "optimizer with the aim to converge faster, generalize better and remain more "
                 "stable. (https://arxiv.org/abs/2010.07468). NB: Epsilon for AdaBelief needs to "
                 "be set to a smaller value than other Optimizers. Generally setting the 'Epsilon "
                 "Exponent' to around '-16' should work."
                 "\n\t adam - Adaptive Moment Optimization. A stochastic gradient descent method "
                 "that is based on adaptive estimation of first-order and second-order moments."
                 "\n\t nadam - Adaptive Moment Optimization with Nesterov Momentum. Much like "
                 "Adam but uses a different formula for calculating momentum."
                 "\n\t rms-prop - Root Mean Square Propagation. Maintains a moving (discounted) "
                 "average of the square of the gradients. Divides the gradient by the root of "
                 "this average.")
        self.add_item(
            section=section,
            title="learning_rate",
            datatype=float,
            default=5e-5,
            min_max=(1e-6, 1e-4),
            rounding=6,
            fixed=False,
            group="optimizer",
            info="Learning rate - how fast your network will learn (how large are the "
                 "modifications to the model weights after one batch of training). Values that "
                 "are too large might result in model crashes and the inability of the model to "
                 "find the best solution. Values that are too small might be unable to escape "
                 "from dead-ends and find the best global minimum.")
        self.add_item(
            section=section,
            title="epsilon_exponent",
            datatype=int,
            default=-7,
            min_max=(-20, 0),
            rounding=1,
            fixed=False,
            group="optimizer",
            info="The epsilon adds a small constant to weight updates to attempt to avoid 'divide "
                 "by zero' errors. Unless you are using the AdaBelief Optimizer, then Generally "
                 "this option should be left at default value, For AdaBelief, setting this to "
                 "around '-16' should work.\n"
                 "In all instances if you are getting 'NaN' loss values, and have been unable to "
                 "resolve the issue any other way (for example, increasing batch size, or "
                 "lowering learning rate), then raising the epsilon can lead to a more stable "
                 "model. It may, however, come at the cost of slower training and a less accurate "
                 "final result.\n"
                 "NB: The value given here is the 'exponent' to the epsilon. For example, "
                 "choosing '-7' will set the epsilon to 1e-7. Choosing '-3' will set the epsilon "
                 "to 0.001 (1e-3).")
        self.add_item(
            section=section,
            title="reflect_padding",
            datatype=bool,
            default=False,
            group="network",
            info="Use reflection padding rather than zero padding with convolutions. "
                 "Each convolution must pad the image boundaries to maintain the proper "
                 "sizing. More complex padding schemes can reduce artifacts at the "
                 "border of the image."
                 "\n\t http://www-cs.engr.ccny.cuny.edu/~wolberg/cs470/hw/hw2_pad.txt")
        self.add_item(
            section=section,
            title="allow_growth",
            datatype=bool,
            default=False,
            group="network",
            fixed=False,
            info="[Nvidia Only]. Enable the Tensorflow GPU 'allow_growth' configuration option. "
                 "This option prevents Tensorflow from allocating all of the GPU VRAM at launch "
                 "but can lead to higher VRAM fragmentation and slower performance. Should only "
                 "be enabled if you are receiving errors regarding 'cuDNN fails to initialize' "
                 "when commencing training.")
        self.add_item(
            section=section,
            title="mixed_precision",
            datatype=bool,
            default=False,
            group="network",
            info="[Nvidia Only], NVIDIA GPUs can run operations in float16 faster than in "
                 "float32. Mixed precision allows you to use a mix of float16 with float32, to "
                 "get the performance benefits from float16 and the numeric stability benefits "
                 "from float32.\n\nWhile mixed precision will run on most Nvidia models, it will "
                 "only speed up training on more recent GPUs. Those with compute capability 7.0 "
                 "or higher will see the greatest performance benefit from mixed precision "
                 "because they have Tensor Cores. Older GPUs offer no math performance benefit "
                 "for using mixed precision, however memory and bandwidth savings can enable some "
                 "speedups. Generally RTX GPUs and later will offer the most benefit.")
        self.add_item(
            section=section,
            title="nan_protection",
            datatype=bool,
            default=True,
            group="network",
            info="If a 'NaN' is generated in the model, this means that the model has corrupted "
                 "and the model is likely to start deteriorating from this point on. Enabling NaN "
                 "protection will stop training immediately in the event of a NaN. The last save "
                 "will not contain the NaN, so you may still be able to rescue your model.",
            fixed=False)
        self.add_item(
            section=section,
            title="convert_batchsize",
            datatype=int,
            default=16,
            min_max=(1, 32),
            rounding=1,
            fixed=False,
            group="convert",
            info="[GPU Only]. The number of faces to feed through the model at once when running "
                 "the Convert process.\n\nNB: Increasing this figure is unlikely to improve "
                 "convert speed, however, if you are getting Out of Memory errors, then you may "
                 "want to reduce the batch size.")

    def _set_loss(self):
        """ Set the default loss options.

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
        MSSIM https://www.cns.nyu.edu/pub/eero/wang03b.pdf
        GMSD https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf
        """
        logger.debug("Setting Loss config")
        section = "global.loss"
        self.add_section(title=section,
                         info="Loss configuration options\n"
                              "Loss is the mechanism by which a Neural Network judges how well it "
                              "thinks that it is recreating a face." + ADDITIONAL_INFO)
        self.add_item(
            section=section,
            title="loss_function",
            datatype=str,
            group="loss",
            default="ssim",
            choices=["mae", "mse", "logcosh", "smooth_loss", "l_inf_norm", "ssim", "ms_ssim",
                     "gmsd", "pixel_gradient_diff"],
            info="The loss function to use."
                 "\n\t MAE - Mean absolute error will guide reconstructions of each pixel "
                 "towards its median value in the training dataset. Robust to outliers but as "
                 "a median, it can potentially ignore some infrequent image types in the dataset."
                 "\n\t MSE - Mean squared error will guide reconstructions of each pixel "
                 "towards its average value in the training dataset. As an avg, it will be "
                 "susceptible to outliers and typically produces slightly blurrier results."
                 "\n\t LogCosh - log(cosh(x)) acts similar to MSE for small errors and to "
                 "MAE for large errors. Like MSE, it is very stable and prevents overshoots "
                 "when errors are near zero. Like MAE, it is robust to outliers. NB: Due to a bug "
                 "in PlaidML, this loss does not work on AMD cards."
                 "\n\t Smooth_L1 --- Modification of the MAE loss to correct two of its "
                 "disadvantages. This loss has improved stability and guidance for small errors."
                 "\n\t L_inf_norm --- The L_inf norm will reduce the largest individual pixel "
                 "error in an image. As each largest error is minimized sequentially, the "
                 "overall error is improved. This loss will be extremely focused on outliers."
                 "\n\t SSIM - Structural Similarity Index Metric is a perception-based "
                 "loss that considers changes in texture, luminance, contrast, and local spatial "
                 "statistics of an image. Potentially delivers more realistic looking images."
                 "\n\t MS_SSIM - Multiscale Structural Similarity Index Metric is similar to SSIM "
                 "except that it performs the calculations along multiple scales of the input "
                 "image. NB: This loss currently does not work on AMD Cards."
                 "\n\t GMSD - Gradient Magnitude Similarity Deviation seeks to match "
                 "the global standard deviation of the pixel to pixel differences between two "
                 "images. Similar in approach to SSIM. NB: This loss does not currently work on "
                 "AMD cards."
                 "\n\t Pixel_Gradient_Difference - Instead of minimizing the difference between "
                 "the absolute value of each pixel in two reference images, compute the pixel to "
                 "pixel spatial difference in each image and then minimize that difference "
                 "between two images. Allows for large color shifts, but maintains the structure "
                 "of the image.")
        self.add_item(
            section=section,
            title="mask_loss_function",
            datatype=str,
            group="loss",
            default="mse",
            choices=["mae", "mse"],
            info="The loss function to use when learning a mask."
                 "\n\t MAE - Mean absolute error will guide reconstructions of each pixel "
                 "towards its median value in the training dataset. Robust to outliers but as "
                 "a median, it can potentially ignore some infrequent image types in the dataset."
                 "\n\t MSE - Mean squared error will guide reconstructions of each pixel "
                 "towards its average value in the training dataset. As an average, it will be "
                 "susceptible to outliers and typically produces slightly blurrier results.")
        self.add_item(
            section=section,
            title="l2_reg_term",
            datatype=int,
            group="loss",
            min_max=(0, 400),
            rounding=1,
            default=100,
            info="The amount of L2 Regularization to apply as a penalty to Structural Similarity "
                 "loss functions.\n\nNB: You should only adjust this if you know what you are "
                 "doing!\n\n"
                 "L2 regularization applies a penalty term to the given Loss function. This "
                 "penalty will only be applied if SSIM, MS-SSIM or GMSD is selected for the main "
                 "loss function, otherwise it is ignored."
                 "\n\nThe value given here is as a percentage weight of the main loss function. "
                 "For example:"
                 "\n\t 100 - Will give equal weighting to the main loss and the penalty function. "
                 "\n\t 25 - Will give the penalty function 1/4 of the weight of the main loss "
                 "function. "
                 "\n\t 400 - Will give the penalty function 4x as much importance as the main "
                 "loss function."
                 "\n\t 0 - Disables L2 Regularization altogether.")
        self.add_item(
            section=section,
            title="eye_multiplier",
            datatype=int,
            group="loss",
            min_max=(1, 40),
            rounding=1,
            default=3,
            fixed=False,
            info="The amount of priority to give to the eyes.\n\nThe value given here is as a "
                 "multiplier of the main loss score. For example:"
                 "\n\t 1 - The eyes will receive the same priority as the rest of the face. "
                 "\n\t 10 - The eyes will be given a score 10 times higher than the rest of the "
                 "face."
                 "\n\nNB: Penalized Mask Loss must be enable to use this option.")
        self.add_item(
            section=section,
            title="mouth_multiplier",
            datatype=int,
            group="loss",
            min_max=(1, 40),
            rounding=1,
            default=2,
            fixed=False,
            info="The amount of priority to give to the mouth.\n\nThe value given here is as a "
                 "multiplier of the main loss score. For Example:"
                 "\n\t 1 - The mouth will receive the same priority as the rest of the face. "
                 "\n\t 10 - The mouth will be given a score 10 times higher than the rest of the "
                 "face."
                 "\n\nNB: Penalized Mask Loss must be enable to use this option.")
        self.add_item(
            section=section,
            title="penalized_mask_loss",
            datatype=bool,
            default=True,
            group="loss",
            info="Image loss function is weighted by mask presence. For areas of "
                 "the image without the facial mask, reconstruction errors will be "
                 "ignored while the masked face area is prioritized. May increase "
                 "overall quality by focusing attention on the core face area.")
        self.add_item(
            section=section,
            title="mask_type",
            datatype=str,
            default="extended",
            choices=PluginLoader.get_available_extractors("mask",
                                                          add_none=True, extend_plugin=True),
            group="mask",
            gui_radio=True,
            info="The mask to be used for training. If you have selected 'Learn Mask' or "
                 "'Penalized Mask Loss' you must select a value other than 'none'. The required "
                 "mask should have been selected as part of the Extract process. If it does not "
                 "exist in the alignments file then it will be generated prior to training "
                 "commencing."
                 "\n\tnone: Don't use a mask."
                 "\n\tbisenet-fp-face: Relatively lightweight NN based mask that provides more "
                 "refined control over the area to be masked (configurable in mask settings). "
                 "Use this version of bisenet-fp if your model is trained with 'face' or "
                 "'legacy' centering."
                 "\n\tbisenet-fp-head: Relatively lightweight NN based mask that provides more "
                 "refined control over the area to be masked (configurable in mask settings). "
                 "Use this version of bisenet-fp if your model is trained with 'head' centering."
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
            section=section,
            title="mask_blur_kernel",
            datatype=int,
            min_max=(0, 9),
            rounding=1,
            default=3,
            group="mask",
            info="Apply gaussian blur to the mask input. This has the effect of smoothing the "
                 "edges of the mask, which can help with poorly calculated masks and give less "
                 "of a hard edge to the predicted mask. The size is in pixels (calculated from "
                 "a 128px mask). Set to 0 to not apply gaussian blur. This value should be odd, "
                 "if an even number is passed in then it will be rounded to the next odd number.")
        self.add_item(
            section=section,
            title="mask_threshold",
            datatype=int,
            default=4,
            min_max=(0, 50),
            rounding=1,
            group="mask",
            info="Sets pixels that are near white to white and near black to black. Set to 0 for "
                 "off.")
        self.add_item(
            section=section,
            title="learn_mask",
            datatype=bool,
            default=False,
            group="mask",
            info="Dedicate a portion of the model to learning how to duplicate the input "
                 "mask. Increases VRAM usage in exchange for learning a quick ability to try "
                 "to replicate more complex mask models.")
