#!/usr/bin/env python3
""" Default configurations for models """

import gettext
import logging
import os

from dataclasses import dataclass

from lib.config import ConfigItem, FaceswapConfig, GlobalSection
from plugins.plugin_loader import PluginLoader
from plugins.train.trainer import trainer_config

# LOCALES
_LANG = gettext.translation("plugins.train._config", localedir="locales", fallback=True)
_ = _LANG.gettext

logger = logging.getLogger(__name__)


_ADDITIONAL_INFO = _("\nNB: Unless specifically stated, values changed here will only take effect "
                     "when creating a new model.")


class _Config(FaceswapConfig):
    """ Config File for Models """
    # pylint:disable=too-many-statements
    def set_defaults(self, helptext="") -> None:
        """ Set the default values for config """
        super().set_defaults(helptext=_("Options that apply to all models") + _ADDITIONAL_INFO)
        self._defaults_from_plugin(os.path.dirname(__file__))

        train_helptext, section, train_opts = trainer_config.get_defaults()
        self.add_section(section, train_helptext)
        for k, v in train_opts.items():
            self.add_item(section, k, v)


centering = ConfigItem(
    datatype=str,
    default="face",
    gui_radio=True,
    group=_("face"),
    info=_(
        "How to center the training image. The extracted images are centered on the middle of the "
        "skull based on the face's estimated pose. A subsection of these images are used for "
        "training. The centering used dictates how this subsection will be cropped from the "
        "aligned images."
        "\n\tface: Centers the training image on the center of the face, adjusting for pitch and "
        "yaw."
        "\n\thead: Centers the training image on the center of the head, adjusting for pitch and "
        "yaw. NB: You should only select head centering if you intend to include the full head ("
        "including hair) in the final swap. This may give mixed results. Additionally, it is only "
        "worth choosing head centering if you are training with a mask that includes the hair ("
        "e.g. BiSeNet-FP-Head)."
        "\n\tlegacy: The 'original' extraction technique. Centers the training image near the tip "
        "of the nose with no adjustment. Can result in the edges of the face appearing outside of "
        "the training area."),
    choices=["face", "head", "legacy"],
    fixed=True)


coverage = ConfigItem(
    datatype=float,
    default=100.0,
    group=_("face"),
    info=_(
        "How much of the extracted image to train on. A lower coverage will limit the model's "
        "scope to a zoomed-in central area while higher amounts can include the entire face. A "
        "trade-off exists between lower amounts given more detail versus higher amounts avoiding "
        "noticeable swap transitions. For 'Face' centering you will want to leave this above 75%. "
        "For Head centering you will most likely want to set this to 100%. Sensible values for "
        "'Legacy' centering are:"
        "\n\t62.5% spans from eyebrow to eyebrow."
        "\n\t75.0% spans from temple to temple."
        "\n\t87.5% spans from ear to ear."
        "\n\t100.0% is a mugshot."),
    min_max=(62.5, 100.0),
    rounding=2,
    fixed=True)


vertical_offset = ConfigItem(
    datatype=int,
    default=0,
    group=_("face"),
    info=_(
        "How much to adjust the vertical position of the aligned face as a percentage of face "
        "image size. Negative values move the face up (expose more chin and less forehead). "
        "Positive values move the face down (expose less chin and more forehead)"),
    min_max=(-25, 25),
    rounding=1,
    fixed=True)


icnr_init = ConfigItem(
    datatype=bool,
    default=False,
    group=_("initialization"),
    info=_(
        "Use ICNR to tile the default initializer in a repeating pattern. This strategy is "
        "designed for pairing with sub-pixel / pixel shuffler to reduce the 'checkerboard effect' "
        "in image reconstruction. \n\t https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf"))


conv_aware_init = ConfigItem(
    datatype=bool,
    default=False,
    group=_("initialization"),
    info=_(
        "Use Convolution Aware Initialization for convolutional layers. This can help eradicate "
        "the vanishing and exploding gradient problem as well as lead to higher accuracy, lower "
        "loss and faster convergence.\nNB:\n\t This can use more VRAM when creating a new model "
        "so you may want to lower the batch size for the first run. The batch size can be raised "
        "again when reloading the model."
        "\n\t Multi-GPU is not supported for this option, so you should start the model on a "
        "single GPU. Once training has started, you can stop training, enable multi-GPU and "
        "resume."
        "\n\t Building the model will likely take several minutes as the calculations for this "
        "initialization technique are expensive. This will only impact starting a new model."))


lr_finder_iterations = ConfigItem(
    datatype=int,
    default=1000,
    group=_("Learning Rate Finder"),
    info=_(
        "The number of iterations to process to find the optimal learning rate. Higher values "
        "will take longer, but will be more accurate."),
    min_max=(100, 10000),
    rounding=100,
    fixed=True)


lr_finder_mode = ConfigItem(
    datatype=str,
    default="set",
    group=_("Learning Rate Finder"),
    info=_(
        "The operation mode for the learning rate finder. Only applicable to new models. For "
        "existing models this will always default to 'set'."
        "\n\tset - Train with the discovered optimal learning rate."
        "\n\tgraph_and_set - Output a graph in the training folder showing the discovered "
        "learning rates and train with the optimal learning rate."
        "\n\tgraph_and_exit - Output a graph in the training folder with the discovered learning "
        "rates and exit."),
    gui_radio=True,
    choices=["set", "graph_and_set", "graph_and_exit"],
    fixed=True)


lr_finder_strength = ConfigItem(
    datatype=str,
    default="default",
    group=_("Learning Rate Finder"),
    info=_(
        "How aggressively to set the Learning Rate. More aggressive can learn faster, but is more "
        "likely to lead to exploding gradients."
        "\n\tdefault - The default optimal learning rate. A safe choice for nearly all use cases."
        "\n\taggressive - Set's a higher learning rate than the default. May learn faster but "
        "with a higher chance of exploding gradients."
        "\n\textreme - The highest optimal learning rate. A much higher risk of exploding "
        "gradients."),
    gui_radio=True,
    choices=["default", "aggressive", "extreme"],
    fixed=True)


reflect_padding = ConfigItem(
    datatype=bool,
    default=False,
    group=_("network"),
    info=_(
        "Use reflection padding rather than zero padding with convolutions. Each convolution must "
        "pad the image boundaries to maintain the proper sizing. More complex padding schemes can "
        "reduce artifacts at the border of the image."
        "\n\t http://www-cs.engr.ccny.cuny.edu/~wolberg/cs470/hw/hw2_pad.txt"))


mixed_precision = ConfigItem(
    datatype=bool,
    default=False,
    group=_("network"),
    info=_(
        "NVIDIA GPUs can run operations in float16 faster than in float32. Mixed precision allows "
        "you to use a mix of float16 with float32, to get the performance benefits from float16 "
        "and the numeric stability benefits from float32.\n\nThis is untested on non-Nvidia "
        "cards, but will run on most Nvidia models. it will only speed up training on more recent "
        "GPUs. Those with compute capability 7.0 or higher will see the greatest performance "
        "benefit from mixed precision because they have Tensor Cores. Older GPUs offer no math "
        "performance benefit for using mixed precision, however memory and bandwidth savings can "
        "enable some speedups. Generally RTX GPUs and later will offer the most benefit."),
    fixed=False)


nan_protection = ConfigItem(
    datatype=bool,
    default=True,
    group=_("network"),
    info=_(
        "If a 'NaN' is generated in the model, this means that the model has corrupted and the "
        "model is likely to start deteriorating from this point on. Enabling NaN protection will "
        "stop training immediately in the event of a NaN. The last save will not contain the NaN, "
        "so you may still be able to rescue your model."),
    fixed=False)


convert_batchsize = ConfigItem(
    datatype=int,
    default=16,
    group=_("convert"),
    info=_(
        "[GPU Only]. The number of faces to feed through the model at once when running the "
        "Convert process.\n\nNB: Increasing this figure is unlikely to improve convert speed, "
        "however, if you are getting Out of Memory errors, then you may want to reduce the batch "
        "size."),
    min_max=(1, 32),
    rounding=1,
    fixed=False)


_LOSS_HELP = {
    "ffl": _(
        "Focal Frequency Loss. Analyzes the frequency spectrum of the images rather than the "
        "images themselves. This loss function can be used on its own, but the original paper "
        "found increased benefits when using it as a complementary loss to another spacial loss "
        "function (e.g. MSE). Ref: Focal Frequency Loss for Image Reconstruction and Synthesis "
        "https://arxiv.org/pdf/2012.12821.pdf NB: This loss does not currently work on AMD "
        "cards."),
    "flip": _(
        "Nvidia FLIP. A perceptual loss measure that approximates the difference perceived by "
        "humans as they alternate quickly (or flip) between two images. Used on its own and this "
        "loss function creates a distinct grid on the output. However it can be helpful when "
        "used as a complimentary loss function. Ref: FLIP: A Difference Evaluator for "
        "Alternating Images: "
        "https://research.nvidia.com/sites/default/files/node/3260/FLIP_Paper.pdf"),
    "gmsd": _(
        "Gradient Magnitude Similarity Deviation seeks to match the global standard deviation of "
        "the pixel to pixel differences between two images. Similar in approach to SSIM. Ref: "
        "Gradient Magnitude Similarity Deviation: An Highly Efficient Perceptual Image Quality "
        "Index https://arxiv.org/ftp/arxiv/papers/1308/1308.3052.pdf"),
    "l_inf_norm": _(
        "The L_inf norm will reduce the largest individual pixel error in an image. As "
        "each largest error is minimized sequentially, the overall error is improved. This loss "
        "will be extremely focused on outliers."),
    "laploss": _(
        "Laplacian Pyramid Loss. Attempts to improve results by focussing on edges using "
        "Laplacian Pyramids. As this loss function gives priority to edges over other low-"
        "frequency information, like color, it should not be used on its own. The original "
        "implementation uses this loss as a complimentary function to MSE. "
        "Ref: Optimizing the Latent Space of Generative Networks "
        "https://arxiv.org/abs/1707.05776"),
    "lpips_alex": _(
        "LPIPS is a perceptual loss that uses the feature outputs of other pretrained models as a "
        "loss metric. Be aware that this loss function will use more VRAM. Used on its own and "
        "this loss will create a distinct moire pattern on the output, however it can be helpful "
        "as a complimentary loss function. The output of this function is strong, so depending "
        "on your chosen primary loss function, you are unlikely going to want to set the weight "
        "above about 25%. Ref: The Unreasonable Effectiveness of Deep Features as a Perceptual "
        "Metric http://arxiv.org/abs/1801.03924\nThis variant uses the AlexNet backbone. A fairly "
        "light and old model which performed best in the paper's original implementation.\nNB: "
        "For AMD Users the final linear layer is not implemented."),
    "lpips_squeeze": _(
        "Same as lpips_alex, but using the SqueezeNet backbone. A more lightweight "
        "version of AlexNet.\nNB: For AMD Users the final linear layer is not implemented."),
    "lpips_vgg16": _(
        "Same as lpips_alex, but using the VGG16 backbone. A more heavyweight model.\n"
        "NB: For AMD Users the final linear layer is not implemented."),
    "logcosh": _(
        "log(cosh(x)) acts similar to MSE for small errors and to MAE for large errors. Like "
        "MSE, it is very stable and prevents overshoots when errors are near zero. Like MAE, it "
        "is robust to outliers."),
    "mae": _(
        "Mean absolute error will guide reconstructions of each pixel towards its median value in "
        "the training dataset. Robust to outliers but as a median, it can potentially ignore some "
        "infrequent image types in the dataset."),
    "mse": _(
        "Mean squared error will guide reconstructions of each pixel towards its average value in "
        "the training dataset. As an avg, it will be susceptible to outliers and typically "
        "produces slightly blurrier results. Ref: Multi-Scale Structural Similarity for Image "
        "Quality Assessment https://www.cns.nyu.edu/pub/eero/wang03b.pdf"),
    "ms_ssim": _(
        "Multiscale Structural Similarity Index Metric is similar to SSIM except that it "
        "performs the calculations along multiple scales of the input image."),
    "smooth_loss": _(
        "Smooth_L1 is a modification of the MAE loss to correct two of its disadvantages. "
        "This loss has improved stability and guidance for small errors. Ref: A General and "
        "Adaptive Robust Loss Function https://arxiv.org/pdf/1701.03077.pdf"),
    "ssim": _(
        "Structural Similarity Index Metric is a perception-based loss that considers changes in "
        "texture, luminance, contrast, and local spatial statistics of an image. Potentially "
        "delivers more realistic looking images. Ref: Image Quality Assessment: From Error "
        "Visibility to Structural Similarity http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf"),
    "pixel_gradient_diff": _(
        "Instead of minimizing the difference between the absolute value of each "
        "pixel in two reference images, compute the pixel to pixel spatial difference in each "
        "image and then minimize that difference between two images. Allows for large color "
        "shifts, but maintains the structure of the image."),
    "none": _("Do not use an additional loss function.")}

_NON_PRIMARY_LOSS = ["flip", "lpips_alex", "lpips_squeeze", "lpips_vgg16", "none"]


@dataclass
class Loss(GlobalSection):
    """ global.loss configuration section
    Loss Documentation
    MAE https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    MSE https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    LogCosh https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0
    L_inf_norm https://medium.com/@montjoile/l0-norm-l1-norm-l2-norm-l-infinity-norm-7a7d18a4f40c
    """  # pylint:disable=line-too-long  # noqa[E501]

    helptext = _(
        "Loss configuration options\n"
        "Loss is the mechanism by which a Neural Network judges how well it thinks that it "
        "is recreating a face.") + _ADDITIONAL_INFO
    loss_function = ConfigItem(
        datatype=str,
        default="ssim",
        group=_("loss"),
        info=(_("The loss function to use.") +
              "\n\n\t" + "\n\n\t".join(f"{k}: {v}"
                                       for k, v in sorted(_LOSS_HELP.items())
                                       if k not in _NON_PRIMARY_LOSS)),
        choices=[x for x in sorted(_LOSS_HELP) if x not in _NON_PRIMARY_LOSS],
        fixed=False)
    loss_function_2 = ConfigItem(
        datatype=str,
        default="mse",
        group=_("loss"),
        info=_(
            "The second loss function to use. If using a structural based loss (such as "
            "SSIM, MS-SSIM or GMSD) it is common to add an L1 regularization(MAE) or L2 "
            "regularization (MSE) function. You can adjust the weighting of this loss "
            "function with the loss_weight_2 option." +
            "\n\n\t" + "\n\n\t".join(f"{k}: {v}" for k, v in sorted(_LOSS_HELP.items()))),
        choices=list(sorted(_LOSS_HELP)),
        fixed=False)
    loss_weight_2 = ConfigItem(
        datatype=int,
        default=100,
        group=_("loss"),
        info=_(
            "The amount of weight to apply to the second loss function.\n\n"
            "\n\nThe value given here is as a percentage denoting how much the selected "
            "function should contribute to the overall loss cost of the model. For "
            "example:"
            "\n\t 100 - The loss calculated for the second loss function will be applied "
            "at its full amount towards the overall loss score. "
            "\n\t 25 - The loss calculated for the second loss function will be reduced "
            "by a quarter prior to adding to the overall loss score. "
            "\n\t 400 - The loss calculated for the second loss function will be "
            "mulitplied 4 times prior to adding to the overall loss score. "
            "\n\t 0 - Disables the second loss function altogether."),
        min_max=(0, 400),
        rounding=1,
        fixed=False)
    loss_function_3 = ConfigItem(
        datatype=str,
        default="none",
        group=_("loss"),
        info=_("The third loss function to use. You can adjust the weighting of this loss "
               "function with the loss_weight_3 option." +
               "\n\n\t" +
               "\n\n\t".join(f"{k}: {v}" for k, v in sorted(_LOSS_HELP.items()))),
        choices=list(sorted(_LOSS_HELP)),
        fixed=False)
    loss_weight_3 = ConfigItem(
        datatype=int,
        default=0,
        group=_("loss"),
        info=_(
            "The amount of weight to apply to the third loss function.\n\n"
            "\n\nThe value given here is as a percentage denoting how much the selected "
            "function should contribute to the overall loss cost of the model. For "
            "example:"
            "\n\t 100 - The loss calculated for the third loss function will be applied "
            "at its full amount towards the overall loss score. "
            "\n\t 25 - The loss calculated for the third loss function will be reduced "
            "by a quarter prior to adding to the overall loss score. "
            "\n\t 400 - The loss calculated for the third loss function will be "
            "mulitplied 4 times prior to adding to the overall loss score. "
            "\n\t 0 - Disables the third loss function altogether."),
        min_max=(0, 400),
        rounding=1,
        fixed=False)
    loss_function_4 = ConfigItem(
        datatype=str,
        default="none",
        group=_("loss"),
        info=_(
            "The fourth loss function to use. You can adjust the weighting of this "
            "loss function with the loss_weight_3 option." +
            "\n\n\t" +
            "\n\n\t".join(f"{k}: {v}" for k, v in sorted(_LOSS_HELP.items()))),
        choices=list(sorted(_LOSS_HELP)),
        fixed=False)
    loss_weight_4 = ConfigItem(
        datatype=int,
        default=0,
        group=_("loss"),
        info=_(
            "The amount of weight to apply to the fourth loss function.\n\n"
            "\n\nThe value given here is as a percentage denoting how much the selected "
            "function should contribute to the overall loss cost of the model. For "
            "example:"
            "\n\t 100 - The loss calculated for the fourth loss function will be applied "
            "at its full amount towards the overall loss score. "
            "\n\t 25 - The loss calculated for the fourth loss function will be reduced "
            "by a quarter prior to adding to the overall loss score. "
            "\n\t 400 - The loss calculated for the fourth loss function will be "
            "mulitplied 4 times prior to adding to the overall loss score. "
            "\n\t 0 - Disables the fourth loss function altogether."),
        min_max=(0, 400),
        rounding=1,
        fixed=False)
    mask_loss_function = ConfigItem(
        datatype=str,
        default="mse",
        group=_("loss"),
        info=_(
            "The loss function to use when learning a mask."
            "\n\t MAE - Mean absolute error will guide reconstructions of each pixel "
            "towards its median value in the training dataset. Robust to outliers but as "
            "a median, it can potentially ignore some infrequent image types in the "
            "dataset."
            "\n\t MSE - Mean squared error will guide reconstructions of each pixel "
            "towards its average value in the training dataset. As an average, it will be "
            "susceptible to outliers and typically produces slightly blurrier results."),
        choices=["mae", "mse"],
        fixed=False)
    eye_multiplier = ConfigItem(
        datatype=int,
        default=3,
        group=_("loss"),
        info=_(
            "The amount of priority to give to the eyes.\n\nThe value given here is as a "
            "multiplier of the main loss score. For example:"
            "\n\t 1 - The eyes will receive the same priority as the rest of the face. "
            "\n\t 10 - The eyes will be given a score 10 times higher than the rest of "
            "the face."
            "\n\nNB: Penalized Mask Loss must be enable to use this option."),
        min_max=(1, 40),
        rounding=1,
        fixed=False)
    mouth_multiplier = ConfigItem(
        datatype=int,
        default=2,
        group=_("loss"),
        info=_(
            "The amount of priority to give to the mouth.\n\nThe value given here is as a "
            "multiplier of the main loss score. For Example:"
            "\n\t 1 - The mouth will receive the same priority as the rest of the face. "
            "\n\t 10 - The mouth will be given a score 10 times higher than the rest of "
            "the face."
            "\n\nNB: Penalized Mask Loss must be enable to use this option."),
        min_max=(1, 40),
        rounding=1,
        fixed=False)
    penalized_mask_loss = ConfigItem(
        datatype=bool,
        default=True,
        group=_("loss"),
        info=_(
            "Image loss function is weighted by mask presence. For areas of "
            "the image without the facial mask, reconstruction errors will be "
            "ignored while the masked face area is prioritized. May increase "
            "overall quality by focusing attention on the core face area."))
    mask_type = ConfigItem(
        datatype=str,
        default="extended",
        group=_("mask"),
        info=_(
            "The mask to be used for training. If you have selected 'Learn Mask' or "
            "'Penalized Mask Loss' you must select a value other than 'none'. The "
            "required mask should have been selected as part of the Extract process. If "
            "it does not exist in the alignments file then it will be generated prior to "
            "training commencing."
            "\n\tnone: Don't use a mask."
            "\n\tbisenet-fp_face: Relatively lightweight NN based mask that provides more "
            "refined control over the area to be masked (configurable in mask settings). "
            "Use this version of bisenet-fp if your model is trained with 'face' or "
            "'legacy' centering."
            "\n\tbisenet-fp_head: Relatively lightweight NN based mask that provides more "
            "refined control over the area to be masked (configurable in mask settings). "
            "Use this version of bisenet-fp if your model is trained with 'head' "
            "centering."
            "\n\tcomponents: Mask designed to provide facial segmentation based on the "
            "positioning of landmark locations. A convex hull is constructed around the "
            "exterior of the landmarks to create a mask."
            "\n\tcustom_face: Custom user created, face centered mask."
            "\n\tcustom_head: Custom user created, head centered mask."
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
            "performance."),
        choices=PluginLoader.get_available_extractors("mask",
                                                      add_none=True, extend_plugin=True),
        gui_radio=True)
    mask_dilation = ConfigItem(
        datatype=float,
        default=0.0,
        group=_("mask"),
        info=_(
            "Dilate or erode the mask. Negative values erode the mask (make it smaller). "
            "Positive values dilate the mask (make it larger). The value given is a "
            "percentage of the total mask size."),
        min_max=(-5.0, 5.0),
        rounding=1,
        fixed=False)
    mask_blur_kernel = ConfigItem(
        datatype=int,
        default=3,
        group=_("mask"),
        info=_(
            "Apply gaussian blur to the mask input. This has the effect of smoothing the "
            "edges of the mask, which can help with poorly calculated masks and give less "
            "of a hard edge to the predicted mask. The size is in pixels (calculated from "
            "a 128px mask). Set to 0 to not apply gaussian blur. This value should be "
            "odd, if an even number is passed in then it will be rounded to the next odd "
            "number."),
        min_max=(0, 9),
        rounding=1,
        fixed=False)
    mask_threshold = ConfigItem(
        datatype=int,
        default=4,
        group=_("mask"),
        info=_(
            "Sets pixels that are near white to white and near black to black. Set to 0 "
            "for off."),
        min_max=(0, 50),
        rounding=1,
        fixed=False)
    learn_mask = ConfigItem(
        datatype=bool,
        default=False,
        group=_("mask"),
        info=_(
            "Dedicate a portion of the model to learning how to duplicate the input "
            "mask. Increases VRAM usage in exchange for learning a quick ability to try "
            "to replicate more complex mask models."))


@dataclass
class Optimizer(GlobalSection):
    """ global.optimizer configuration section """
    helptext = (_("Optimizer configuration options\n"
                  "The optimizer applies the output of the loss function to the model.\n")
                + _ADDITIONAL_INFO)
    optimizer = ConfigItem(
        datatype=str,
        default="adam",
        group=_("optimizer"),
        info=_(
            "The optimizer to use."
            "\n\t adabelief - Adapting Stepsizes by the Belief in Observed Gradients. An "
            "optimizer with the aim to converge faster, generalize better and remain more "
            "stable. (https://arxiv.org/abs/2010.07468). NB: Epsilon for AdaBelief needs "
            "to be set to a smaller value than other Optimizers. Generally setting the "
            "'Epsilon Exponent' to around '-16' should work."
            "\n\t adam - Adaptive Moment Optimization. A stochastic gradient descent "
            "method that is based on adaptive estimation of first-order and second-order "
            "moments."
            "\n\t adamax - a variant of Adam based on the infinity norm. Due to its "
            "capability of adjusting the learning rate based on data characteristics, it "
            "is suited to learn time-variant process, "
            "parameters follow those provided in the paper"
            "\n\t adamw - Like 'adam' but with an added method to decay weights per the "
            "techniques discussed in the paper (https://arxiv.org/abs/1711.05101). NB: "
            "Weight decay should be set at 0.004 for default implementation."
            "\n\t lion - A method that uses the sign operator to control the magnitude of "
            "the update, rather than relying on second-order moments (Adam). saves VRAM "
            "by only tracking the momentum. Performance gains should be better with "
            "larger batch sizes. A suitable learning rate for Lion is typically 3-10x "
            "smaller than that for AdamW. The weight decay for Lion should be 3-10x "
            "larger than that for AdamW to maintain a similar strength."
            "\n\t nadam - Adaptive Moment Optimization with Nesterov Momentum. Much like "
            "Adam but uses a different formula for calculating momentum."
            "\n\t rms-prop - Root Mean Square Propagation. Maintains a moving "
            "(discounted) average of the square of the gradients. Divides the gradient by "
            "the root of this average."),
        choices=["adabelief", "adam", "adamax", "adamw", "lion", "nadam", "rms-prop"],
        gui_radio=True,
        fixed=True)
    learning_rate = ConfigItem(
        datatype=float,
        default=5e-5,
        group=_("optimizer"),
        info=_(
            "Learning rate - how fast your network will learn (how large are the "
            "modifications to the model weights after one batch of training). Values that "
            "are too large might result in model crashes and the inability of the model "
            "to find the best solution. Values that are too small might be unable to "
            "escape from dead-ends and find the best global minimum."),
        min_max=(1e-6, 1e-4),
        rounding=6,
        fixed=False)
    epsilon_exponent = ConfigItem(
        datatype=int,
        default=-7,
        group=_("optimizer"),
        info=_(
            "The epsilon adds a small constant to weight updates to attempt to avoid "
            "'divide by zero' errors. Unless you are using the AdaBelief Optimizer, then "
            "Generally this option should be left at default value, For AdaBelief, "
            "setting this to around '-16' should work.\n"
            "In all instances if you are getting 'NaN' loss values, and have been unable "
            "to resolve the issue any other way (for example, increasing batch size, or "
            "lowering learning rate), then raising the epsilon can lead to a more stable "
            "model. It may, however, come at the cost of slower training and a less "
            "accurate final result.\n"
            "Note: The value given here is the 'exponent' to the epsilon. For example, "
            "choosing '-7' will set the epsilon to 1e-7. Choosing '-3' will set the "
            "epsilon to 0.001 (1e-3).\n"
            "Note: Not used by the Lion optimizer"),
        min_max=(-20, 0),
        rounding=1,
        fixed=False)
    save_optimizer = ConfigItem(
        datatype=str,
        default="exit",
        group=_("optimizer"),
        info=_(
            "When to save the Optimizer Weights. Saving the optimizer weights is not "
            "necessary and will increase the model file size 3x (and by extension the "
            "amount of time it takes to save the model). However, it can be useful to "
            "save these weights if you want to guarantee that a resumed model carries off "
            "exactly from where it left off, rather than spending a few hundred "
            "iterations catching up."
            "\n\t never - Don't save optimizer weights."
            "\n\t always - Save the optimizer weights at every save iteration. Model "
            "saving will take longer, due to the increased file size, but you will always "
            "have the last saved optimizer state in your model file."
            "\n\t exit - Only save the optimizer weights when explicitly terminating a "
            "model. This can be when the model is actively stopped or when the target "
            "iterations are met. Note: If the training session ends because of another "
            "reason (e.g. power outage, Out of Memory Error, NaN detected) then the "
            "optimizer weights will NOT be saved."),
        gui_radio=True,
        choices=["never", "always", "exit"],
        fixed=False)
    gradient_clipping = ConfigItem(
        datatype=str,
        default="none",
        group=_("clipping"),
        info=_(
            "Apply clipping to the gradients. Can help prevent NaNs and improve model "
            "optimization at the expense of VRAM."
            "\n\tautoclip: Analyzes the gradient weights and adjusts the normalization "
            "value dynamically to fit the data"
            "\n\tglobal_norm: Clips the gradient of each weight so that the global norm "
            "is no higher than the given value."
            "\n\tnorm: Clips the gradient of each weight so that its norm is no higher "
            "than the given value."
            "\n\tvalue: Clips the gradient of each weight so that it is no higher than "
            "the given value."
            "\n\tnone: Don't perform any clipping to the gradients."),
        choices=["autoclip", "global_norm", "norm", "value", "none"],
        gui_radio=True,
        fixed=False)
    clipping_value = ConfigItem(
        datatype=float,
        default=1.0,
        group=_("clipping"),
        info=_(
            "The amount of clipping to perform."
            "\n\tautoclip: The percentile to clip at. A value of 1.0 will clip at the "
            "10th percentile a value of 2.5 will clip at the 25th percentile etc. "
            "Default: 1.0"
            "\n\tglobal_norm: The gradient of each weight is clipped so that the global "
            "norm is no higher than this value."
            "\n\tnorm: The gradient of each weight is clipped so that its norm is no "
            "higher than this value."
            "\n\tvalue: The gradient of each weight is clipped to be no higher than this "
            "value."
            "\n\tnone: This option is ignored."),
        min_max=(0.0, 10.0),
        rounding=1,
        fixed=False)
    autoclip_history = ConfigItem(
        datatype=int,
        default=10000,
        group=_("clipping"),
        info=_(
            "The maximum number of prior iterations for autoclipper to analyze when "
            "calculating the normalization amount. 0 to always include all prior "
            "iterations."),
        min_max=(0, 100000),
        rounding=1000,
        fixed=False)
    weight_decay = ConfigItem(
        datatype=float,
        default=0.0,
        group=_("updates"),
        info=_("If set, weight decay is applied. 0.0 for no weight decay. Default is 0.0 "
               "for all optimizers except AdamW (0.004)"),
        min_max=(0.0, 1.0),
        rounding=4,
        fixed=False)
    gradient_accumulation = ConfigItem(
        datatype=int,
        default=1,
        group=_("updates"),
        info=_(
            "Values above 1 will enable Gradient Accumulation. Updates will not be at "
            "every iteration; instead they will occur every number of iterations given "
            "here. The update will be the average value of the gradients since the last "
            "update. Can be useful when your batch size is very small, in order to reduce "
            "gradient noise at each update iteration."),
        min_max=(1, 100),
        rounding=1,
        fixed=False)
    use_ema = ConfigItem(
        datatype=bool,
        default=False,
        group=_("exponential moving average"),
        info=_(
            "Enable exponential moving average (EMA). EMA consists of computing an "
            "exponential moving average of the weights of the model (as the weight values "
            "change after each training batch), and periodically overwriting the weights "
            "with their moving average"),
        fixed=True)
    ema_momentum = ConfigItem(
        datatype=float,
        default=0.99,
        group=_("exponential moving average"),
        info=_(
            "Only used if use_ema is enabled. This is the momentum to use when computing "
            "the EMA of the model's weights: new_average = ema_momentum * old_average + "
            "(1 - ema_momentum) * current_variable_value."),
        min_max=(0.0, 1.0),
        rounding=4,
        fixed=True)
    ema_frequency = ConfigItem(
        datatype=int,
        default=100,
        group=_("exponential moving average"),
        info=_(
            "Only used if use_ema is enabled. Set the number of iterations, to overwrite "
            "the model variable by its moving average. "),
        min_max=(10, 10000),
        rounding=10,
        fixed=True)
    ada_beta_1 = ConfigItem(
        datatype=float,
        default=0.9,
        group=_("optimizer specific"),
        info=_(
            "The exponential decay rate for the 1st moment estimates. Used for the "
            "following Optimizers: AdaBelief, Adam, Adamax, AdamW, Lion, nAdam. Ignored "
            "for all others."),
        min_max=(0.0, 1.0),
        rounding=4,
        fixed=True)
    ada_beta_2 = ConfigItem(
        datatype=float,
        default=0.999,
        group=_("optimizer specific"),
        info=_(
            "The exponential decay rate for the 2nd moment estimates. Used for the "
            "following Optimizers:  AdaBelief, Adam, Adamax, AdamW, Lion, nAdam. Ignored "
            "for all others."),
        min_max=(0.0, 1.0),
        rounding=4,
        fixed=True)
    ada_amsgrad = ConfigItem(
        datatype=bool,
        default=False,
        group=_("optimizer specific"),
        info=_(
            "Whether to apply AMSGrad variant of the algorithm from the paper 'On the "
            "Convergence of Adam and beyond. Used for the following Optimizers: "
            "AdaBelief, Adam, AdamW. Ignored for all others.'"),
        fixed=True)


# pylint:disable=duplicate-code
_IS_LOADED: bool = False


def load_config(config_file: str | None = None) -> None:
    """ Load the Train configuration .ini file

    Parameters
    ----------
    config_file : str | None, optional
        Path to a custom .ini configuration file to load. Default: ``None`` (use default
        configuration file)
    """
    global _IS_LOADED  # pylint:disable=global-statement
    if not _IS_LOADED:
        _Config(configfile=config_file)
    _IS_LOADED = True
