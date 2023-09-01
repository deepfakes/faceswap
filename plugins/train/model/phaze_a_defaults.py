#!/usr/bin/env python3
"""
    The default options for the faceswap Phaze-A Model plugin.

    Defaults files should be named <plugin_name>_defaults.py
    Any items placed into this file will automatically get added to the relevant config .ini files
    within the faceswap/config folder.

    The following variables should be defined:
        "_HELPTEXT: A string describing what this plugin does
        "_DEFAULTS: A dictionary containing the options, defaults and meta information. The
               "   dictionary should be defined as:
               "       {<option_name>: {<metadata>}}

               "   <option_name> should always be lower text.
               "   <metadata> dictionary requirements are listed below.

    The following keys are expected for the _DEFAULTS <metadata> dict:
        "datatype:  [required] A python type class. This limits the type of data that can be
               "   provided in the .ini file and ensures that the value is returned in the
               "   correct type to faceswap. Valid data types are: <class 'int'>, <class 'float'>,
               "   <class 'str'>, <class 'bool'>.
        "default:   [required] The default value for this option.
        "info:      [required] A string describing what this option does.
        "choices:   [optional] If this option's datatype is of <class 'str'> then valid
               "   selections can be defined here. This validates the option and also enables
               "   a combobox / radio option in the GUI.
        "gui_radio: [optional] If <choices> are defined, this indicates that the GUI should use
               "   radio buttons rather than a combobox to display this option.
        "min_max:   [partial] For <class 'int'> and <class 'float'> data types this is required
               "   otherwise it is ignored. Should be a tuple of min and max accepted values.
               "   This is used for controlling the GUI slider range. Values are not enforced.
        "rounding:  [partial] For <class 'int'> and <class 'float'> data types this is
               "   required otherwise it is ignored. Used for the GUI slider. For floats, this
               "   is the number of decimal places to display. For ints this is the step size.
        "fixed:     [optional] [train only]. Training configurations are fixed when the model is
               "   created, and then reloaded from the state file. Marking an item as fixed=False
               "   indicates that this value can be changed for existing models, and will override
               "   the value saved in the state file with the updated value in config. If not
               "   provided this will default to True.
"""

_HELPTEXT: str = (
    "Phaze-A Model by TorzDF, with thanks to BirbFakes.\n"
    "Allows for the experimentation of various standard Networks as the encoder and takes "
    "inspiration from Nvidia's StyleGAN for the Decoder. It is highly recommended to research to "
    "understand the parameters better.")

_ENCODERS: list[str] = sorted([
    "clipv_vit-b-16", "clipv_vit-b-32", "clipv_vit-l-14", "clipv_vit-l-14-336px",
    "clipv_farl-b-16-16", "clipv_farl-b-16-64",
    "densenet121", "densenet169", "densenet201", "efficientnet_b0", "efficientnet_b1",
    "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6",
    "efficientnet_b7", "efficientnet_v2_b0", "efficientnet_v2_b1", "efficientnet_v2_b2",
    "efficientnet_v2_b3", "efficientnet_v2_l", "efficientnet_v2_m", "efficientnet_v2_s",
    "inception_resnet_v2", "inception_v3", "mobilenet", "mobilenet_v2", "mobilenet_v3_large",
    "mobilenet_v3_small", "nasnet_large", "nasnet_mobile", "resnet50", "resnet50_v2", "resnet101",
    "resnet101_v2", "resnet152", "resnet152_v2", "vgg16", "vgg19", "xception", "fs_original"])

_DEFAULTS = {
    # General
    "output_size": {
        "default": 128,
        "info": (
            "Resolution (in pixels) of the output image to generate.\n"
            "BE AWARE Larger resolution will dramatically increase VRAM requirements."),
        "datatype": int,
        "rounding": 16,
        "min_max": (64, 2048),
        "group": "general",
        "fixed": True},
    "shared_fc": {
        "default": "none",
        "info": (
            "Whether to create a shared fully connected layer. This layer will have the same "
            "structure as the fully connected layers used for each side of the model. A shared "
            "fully connected layer looks for patterns that are common to both sides. NB: "
            "Enabling this option only makes sense if 'split fc' is selected."
            "\n\tnone - Do not create a Fully Connected layer for shared data. (Original method)"
            "\n\tfull - Create an exclusive Fully Connected layer for shared data. (IAE method)"
            "\n\thalf - Use the 'fc_a' layer for shared data. This saves VRAM by re-using the "
            "'A' side's fully connected model for the shared data. However, this will lead to "
            "an 'unbalanced' model and can lead to more identity bleed (DFL method)"),
        "datatype": str,
        "choices": ["none", "full", "half"],
        "gui_radio": True,
        "group": "general",
        "fixed": True},
    "enable_gblock": {
        "default": True,
        "info": (
            "Whether to enable the G-Block. If enabled, this will create a shared fully "
            "connected layer (configurable in the 'G-Block hidden layers' section) to look for "
            "patterns in the combined data, before feeding a block prior to the decoder for "
            "merging this shared and combined data."
            "\n\tTrue - Use the G-Block in the Decoder. A combined fully connected layer will be "
            "created to feed this block which can be configured below."
            "\n\tFalse - Don't use the G-Block in the decoder. No combined fully connected layer "
            "will be created."),
        "datatype": bool,
        "group": "general",
        "fixed": True},
    "split_fc": {
        "default": True,
        "info": (
            "Whether to use a single shared Fully Connected layer or separate Fully Connected "
            "layers for each side."
            "\n\tTrue - Use separate Fully Connected layers for Face A and Face B. This is more "
            "similar to the 'IAE' style of model."
            "\n\tFalse - Use combined Fully Connected layers for both sides. This is more "
            "similar to the original Faceswap architecture."),
        "datatype": bool,
        "group": "general",
        "fixed": True},
    "split_gblock": {
        "default": False,
        "info": (
            "If the G-Block is enabled, Whether to use a single G-Block shared between both "
            "sides, or whether to have a separate G-Block (one for each side). NB: The Fully "
            "Connected layer that feeds the G-Block will always be shared."
            "\n\tTrue - Use separate G-Blocks for Face A and Face B."
            "\n\tFalse - Use a combined G-Block layers for both sides."),
        "datatype": bool,
        "group": "general",
        "fixed": True},
    "split_decoders": {
        "default": False,
        "info": (
            "Whether to use a single decoder or split decoders."
            "\n\tTrue - Use a separate decoder for Face A and Face B. This is more similar to "
            "the original Faceswap architecture."
            "\n\tFalse - Use a combined Decoder. This is more similar to 'IAE' style "
            "architecture."),
        "datatype": bool,
        "group": "general",
        "fixed": True},

    # Encoder
    "enc_architecture": {
        "default": "fs_original",
        "info": (
            "The encoder architecture to use. See the relevant config sections for specific "
            "architecture tweaking.\nNB: For keras based pre-built models, the global "
            "initializers and padding options will be ignored for the selected encoder."
            "\n\n\tCLIPv: This is an implementation of the Visual encoder from the CLIP "
            "transformer. The ViT weights are trained on imagenet whilst the FaRL weights are "
            "trained on face related tasks. All have a default input size of 224px except for "
            "ViT-L-14-336px that has an input size of 336px. Ref: Learning Transferable Visual "
            "Models From Natural Language Supervision (2021): https://arxiv.org/abs/2103.00020"
            "\n\n\tdensenet: (32px -224px). Ref: Densely Connected Convolutional Networks "
            "(2016): https://arxiv.org/abs/1608.06993?source=post_page"
            "\n\n\tefficientnet: [Tensorflow 2.3+ only] EfficientNet has numerous variants (B0 - "
            "B8) that increases the model width, depth and dimensional space at each step. The "
            "minimum input resolution is 32px for all variants. The maximum input resolution for "
            "each variant is: b0: 224px, b1: 240px, b2: 260px, b3: 300px, b4: 380px, b5: 456px, "
            "b6: 528px, b7 600px. Ref: Rethinking Model Scaling for Convolutional Neural "
            "Networks (2020): https://arxiv.org/abs/1905.11946"
            "\n\n\tefficientnet_v2: [Tensorflow 2.8+ only] EfficientNetV2 is the follow up to "
            "efficientnet. It has numerous variants (B0 - B3 and Small, Medium and Large) that "
            "increases the model width, depth and dimensional space at each step. The minimum "
            "input resolution is 32px for all variants. The maximum input resolution for each "
            "variant is: b0: 224px, b1: 240px, b2: 260px, b3: 300px, s: 384px, m: 480px, l: "
            "480px. Ref: EfficientNetV2: Smaller Models and Faster Training (2021): "
            "https://arxiv.org/abs/2104.00298"
            "\n\n\tfs_original: (32px - 1024px). A configurable variant of the original facewap "
            "encoder. ImageNet weights cannot be loaded for this model. Additional parameters "
            "can be configured with the 'fs_enc' options. A version of this encoder is used in "
            "the following models: Original, Original (lowmem), Dfaker, DFL-H128, DFL-SAE, IAE, "
            "Lightweight."
            "\n\n\tinception_resnet_v2: (75px - 299px). Ref: Inception-ResNet and the Impact of "
            "Residual Connections on Learning (2016): https://arxiv.org/abs/1602.07261"
            "\n\n\tinceptionV3: (75px - 299px). Ref: Rethinking the Inception Architecture for "
            "Computer Vision (2015): https://arxiv.org/abs/1512.00567"
            "\n\n\tmobilenet: (32px - 224px). Additional MobileNet parameters can be set with "
            "the 'mobilenet' options. Ref: MobileNets: Efficient Convolutional Neural Networks "
            "for Mobile Vision Applications (2017): https://arxiv.org/abs/1704.04861"
            "\n\n\tmobilenet_v2: (32px - 224px). Additional MobileNet parameters can be set with "
            "the 'mobilenet' options. Ref: MobileNetV2: Inverted Residuals and Linear "
            "Bottlenecks (2018): https://arxiv.org/abs/1801.04381"
            "\n\n\tmobilenet_v3: (32px - 224px). Additional MobileNet parameters can be set with "
            "the 'mobilenet' options. Ref: Searching for MobileNetV3 (2019): "
            "https://arxiv.org/pdf/1905.02244.pdf"
            "\n\n\tnasnet: (32px - 331px (large) or 224px (mobile)). Ref: Learning Transferable "
            "Architectures for Scalable Image Recognition (2017): "
            "https://arxiv.org/abs/1707.07012"
            "\n\n\tresnet: (32px - 224px). Deep Residual Learning for Image Recognition (2015): "
            "https://arxiv.org/abs/1512.03385"
            "\n\n\tvgg: (32px - 224px). Very Deep Convolutional Networks for Large-Scale Image "
            "Recognition (2014): https://arxiv.org/abs/1409.1556"
            "\n\n\txception: (71px - 229px). Ref: Deep Learning with Depthwise Separable "
            "Convolutions (2017): https://arxiv.org/abs/1409.1556.\n"),
        "datatype": str,
        "choices": _ENCODERS,
        "gui_radio": False,
        "group": "encoder",
        "fixed": True},
    "enc_scaling": {
        "default": 7,
        "info": (
            "Input scaling for the encoder. Some of the encoders have large input sizes, which "
            "often are not helpful for Faceswap. This setting scales the dimensional space that "
            "the encoder works in. For example an encoder with a maximum input size of 224px "
            "will be input an image of 112px at 50%% scaling. See the Architecture tooltip for "
            "the minimum and maximum sizes for each encoder. NB: The input size will be rounded "
            "down to the nearest 16 pixels."),
        "datatype": int,
        "min_max": (0, 200),
        "rounding": 1,
        "group": "encoder",
        "fixed": True},
    "enc_load_weights": {
        "default": True,
        "info": (
            "Load pre-trained weights trained on ImageNet data. Only available for non-"
            "Faceswap encoders (i.e. those not beginning with 'fs'). NB: If you use the global "
            "'load weights' option and have selected to load weights from a previous model's "
            "'encoder' or 'keras_encoder' then the weights loaded here will be replaced by the "
            "weights loaded from your saved model."),
        "datatype": bool,
        "group": "encoder",
        "fixed": True},

    # Bottleneck
    "bottleneck_type": {
        "default": "dense",
        "info": (
            "The type of layer to use for the bottleneck."
            "\n\taverage_pooling: Use a Global Average Pooling 2D layer for the bottleneck."
            "\n\tdense: Use a Dense layer for the bottleneck (the traditional Faceswap method). "
            "You can set the size of the Dense layer with the 'bottleneck_size' parameter."
            "\n\tmax_pooling: Use a Global Max Pooling 2D layer for the bottleneck."
            "\n\flatten: Don't use a bottleneck at all. Some encoders output in a size that make "
            "a bottleneck unnecessary. This option flattens the output from the encoder, with no "
            "further operations"),
        "datatype": str,
        "group": "bottleneck",
        "gui_radio": True,
        "choices": ["average_pooling", "dense", "max_pooling", "flatten"],
        "fixed": True},
    "bottleneck_norm": {
        "default": "none",
        "info": (
            "Apply a normalization layer after encoder output and prior to the bottleneck."
            "\n\tnone - Do not apply a normalization layer"
            "\n\tinstance - Apply Instance Normalization"
            "\n\tlayer - Apply Layer Normalization (Ba et al., 2016)"
            "\n\trms - Apply Root Mean Squared Layer Normalization (Zhang et al., 2019). A "
            "simplified version of Layer Normalization with reduced overhead."),
        "datatype": str,
        "gui_radio": True,
        "choices": ["none", "instance", "layer", "rms"],
        "group": "bottleneck",
        "fixed": True},
    "bottleneck_size": {
        "default": 1024,
        "info": (
            "If using a Dense layer for the bottleneck, then this is the number of nodes to "
            "use."),
        "datatype": int,
        "rounding": 128,
        "min_max": (128, 4096),
        "group": "bottleneck",
        "fixed": True},
    "bottleneck_in_encoder": {
        "default": True,
        "info": (
            "Whether to place the bottleneck in the Encoder or to place it with the other "
            "hidden layers. Placing the bottleneck in the encoder means that both sides will "
            "share the same bottleneck. Placing it with the other fully connected layers means "
            "that each fully connected layer will each get their own bottleneck. This may be "
            "combined or split depending on your overall architecture configuration settings."),
        "datatype": bool,
        "group": "bottleneck",
        "fixed": True},

    # Intermediate Layers
    "fc_depth": {
        "default": 1,
        "info": (
            "The number of consecutive Dense (fully connected) layers to include in each "
            "side's intermediate layer."),
        "datatype": int,
        "rounding": 1,
        "min_max": (0, 16),
        "group": "hidden layers",
        "fixed": True},
    "fc_min_filters": {
        "default": 1024,
        "info": (
            "The number of filters to use for the initial fully connected layer. The number of "
            "nodes actually used is: fc_min_filters x fc_dimensions x fc_dimensions.\nNB: This "
            "value may be scaled down, depending on output resolution."),
        "datatype": int,
        "rounding": 16,
        "min_max": (16, 5120),
        "group": "hidden layers",
        "fixed": True},
    "fc_max_filters": {
        "default": 1024,
        "info": (
            "This is the number of filters to be used in the final reshape layer at the end of "
            "the fully connected layers. The actual number of nodes used for the final fully "
            "connected layer is: fc_min_filters x fc_dimensions x fc_dimensions.\nNB: This value "
            "may be scaled down, depending on output resolution."),
        "datatype": int,
        "rounding": 64,
        "min_max": (128, 5120),
        "group": "hidden layers",
        "fixed": True},
    "fc_dimensions": {
        "default": 4,
        "info": (
            "The height and width dimension for the final reshape layer at the end of the "
            "fully connected layers.\nNB: The total number of nodes within the final fully "
            "connected layer will be: fc_dimensions x fc_dimensions x fc_max_filters."),
        "datatype": int,
        "rounding": 1,
        "min_max": (1, 16),
        "group": "hidden layers",
        "fixed": True},
    "fc_filter_slope": {
        "default": -0.5,
        "info": (
            "The rate that the filters move from the minimum number of filters to the maximum "
            "number of filters. EG:\n"
            "Negative numbers will change the number of filters quicker at first and slow down "
            "each layer.\n"
            "Positive numbers will change the number of filters slower at first but then speed "
            "up each layer.\n"
            "0.0 - This will change at a linear rate (i.e. the same number of filters will be "
            "changed at each layer)."),
        "datatype": float,
        "min_max": (-.99, .99),
        "rounding": 2,
        "group": "hidden layers",
        "fixed": True},
    "fc_dropout": {
        "default": 0.0,
        "info": (
            "Dropout is a form of regularization that can prevent a model from over-fitting "
            "and help to keep neurons 'alive'. 0.5 will dropout half the connections between each "
            "fully connected layer, 0.25 will dropout a quarter of the connections etc. Set to "
            "0.0 to disable."),
        "datatype": float,
        "rounding": 2,
        "min_max": (0.0, 0.99),
        "group": "hidden layers",
        "fixed": False},
    "fc_upsampler": {
        "default": "upsample2d",
        "info": (
            "The type of dimensional upsampling to perform at the end of the fully connected "
            "layers, if upsamples > 0. The number of filters used for the upscale layers will be "
            "the value given in 'fc_upsample_filters'."
            "\n\tupsample2d - A lightweight and VRAM friendly method. 'quick and dirty' but does "
            "not learn any parameters"
            "\n\tsubpixel - Sub-pixel upscaler using depth-to-space which may require more "
            "VRAM."
            "\n\tresize_images - Uses the Keras resize_image function to save about half as much "
            "vram as the heaviest methods."
            "\n\tupscale_fast - Developed by Andenixa. Focusses on speed to upscale, but "
            "requires more VRAM."
            "\n\tupscale_hybrid - Developed by Andenixa. Uses a combination of PixelShuffler and "
            "Upsampling2D to upscale, saving about 1/3rd of VRAM of the heaviest methods."),
        "datatype": str,
        "choices": ["resize_images", "subpixel", "upscale_fast", "upscale_hybrid", "upsample2d"],
        "group": "hidden layers",
        "gui_radio": False,
        "fixed": True},
    "fc_upsamples": {
        "default": 1,
        "info": (
            "Some upsampling can occur within the Fully Connected layers rather than in the "
            "Decoder to increase the dimensional space. Set how many upscale layers should occur "
            "within the Fully Connected layers."),
        "datatype": int,
        "min_max": (0, 4),
        "rounding": 1,
        "group": "hidden layers",
        "fixed": True},
    "fc_upsample_filters": {
        "default": 512,
        "info": (
            "If you have selected an upsampler which requires filters (i.e. any upsampler with "
            "the exception of Upsampling2D), then this is the number of filters to be used for "
            "the upsamplers within the fully connected layers,  NB: This value may be scaled "
            "down, depending on output resolution. Also note, that this figure will dictate the "
            "number of filters used for the G-Block, if selected."),
        "datatype": int,
        "rounding": 64,
        "min_max": (128, 5120),
        "group": "hidden layers",
        "fixed": True},

    # G-Block
    "fc_gblock_depth": {
        "default": 3,
        "info": (
            "The number of consecutive Dense (fully connected) layers to include in the "
            "G-Block shared layer."),
        "datatype": int,
        "rounding": 1,
        "min_max": (1, 16),
        "group": "g-block hidden layers",
        "fixed": True},
    "fc_gblock_min_nodes": {
        "default": 512,
        "info": "The number of nodes to use for the initial G-Block shared fully connected layer.",
        "datatype": int,
        "rounding": 64,
        "min_max": (128, 5120),
        "group": "g-block hidden layers",
        "fixed": True},
    "fc_gblock_max_nodes": {
        "default": 512,
        "info": "The number of nodes to use for the final G-Block shared fully connected layer.",
        "datatype": int,
        "rounding": 64,
        "min_max": (128, 5120),
        "group": "g-block hidden layers",
        "fixed": True},
    "fc_gblock_filter_slope": {
        "default": -0.5,
        "info": (
            "The rate that the filters move from the minimum number of filters to the maximum "
            "number of filters for the G-Block shared layers. EG:\n"
            "Negative numbers will change the number of filters quicker at first and slow down "
            "each layer.\n"
            "Positive numbers will change the number of filters slower at first but then speed "
            "up each layer.\n"
            "0.0 - This will change at a linear rate (i.e. the same number of filters will be "
            "changed at each layer)."),
        "datatype": float,
        "min_max": (-.99, .99),
        "rounding": 2,
        "group": "g-block hidden layers",
        "fixed": True},
    "fc_gblock_dropout": {
        "default": 0.0,
        "info": (
            "Dropout is a regularization technique that can prevent a model from over-fitting "
            "and help to keep neurons 'alive'. 0.5 will dropout half the connections between "
            "each fully connected layer, 0.25 will dropout a quarter of the connections etc. Set "
            "to 0.0 to disable."),
        "datatype": float,
        "rounding": 2,
        "min_max": (0.0, 0.99),
        "group": "g-block hidden layers",
        "fixed": False},

    # Decoder
    "dec_upscale_method": {
        "default": "subpixel",
        "info": (
            "The method to use for the upscales within the decoder. Images are upscaled "
            "multiple times within the decoder as the network learns to reconstruct the face."
            "\n\tsubpixel - Sub-pixel upscaler using depth-to-space which requires more "
            "VRAM."
            "\n\tresize_images - Uses the Keras resize_image function to save about half as much "
            "vram as the heaviest methods."
            "\n\tupscale_fast - Developed by Andenixa. Focusses on speed to upscale, but "
            "requires more VRAM."
            "\n\tupscale_hybrid - Developed by Andenixa. Uses a combination of PixelShuffler and "
            "Upsampling2D to upscale, saving about 1/3rd of VRAM of the heaviest methods."
            "\n\tupscale_dny - An alternative upscale implementation using Upsampling2D to "
            "upsale."),
        "datatype": str,
        "choices": ["subpixel", "resize_images", "upscale_fast", "upscale_hybrid", "upscale_dny"],
        "gui_radio": True,
        "group": "decoder",
        "fixed": True},
    "dec_upscales_in_fc": {
        "default": 0,
        "min_max": (0, 6),
        "rounding": 1,
        "info": (
            "It is possible to place some of the upscales at the end of the fully connected "
            "model. For models with split decoders, but a shared fully connected layer, this "
            "would have the effect of saving some VRAM but possibly at the cost of introducing "
            "artefacts. For models with a shared decoder but split fully connected layers, this "
            "would have the effect of increasing VRAM usage by processing some of the upscales "
            "for each side rather than together."),
        "datatype": int,
        "group": "decoder",
        "fixed": True},
    "dec_norm": {
        "default": "none",
        "info": (
            "Normalization to apply to apply after each upscale."
            "\n\tnone - Do not apply a normalization layer"
            "\n\tbatch - Apply Batch Normalization"
            "\n\tgroup - Apply Group Normalization"
            "\n\tinstance - Apply Instance Normalization"
            "\n\tlayer - Apply Layer Normalization (Ba et al., 2016)"
            "\n\trms - Apply Root Mean Squared Layer Normalization (Zhang et al., 2019). A "
            "simplified version of Layer Normalization with reduced overhead."),
        "datatype": str,
        "gui_radio": True,
        "choices": ["none", "batch", "group", "instance", "layer", "rms"],
        "group": "decoder",
        "fixed": True},
    "dec_min_filters": {
        "default": 64,
        "info": (
            "The minimum number of filters to use in decoder upscalers (i.e. the number of "
            "filters to use for the final upscale layer)."),
        "datatype": int,
        "min_max": (16, 512),
        "rounding": 16,
        "group": "decoder",
        "fixed": True},
    "dec_max_filters": {
        "default": 512,
        "info": (
            "The maximum number of filters to use in decoder upscalers (i.e. the number of "
            "filters to use for the first upscale layer)."),
        "datatype": int,
        "min_max": (256, 5120),
        "rounding": 64,
        "group": "decoder",
        "fixed": True},
    "dec_slope_mode": {
        "default": "full",
        "info": (
            "Alters the action of the filter slope.\n"
            "\n\tfull: The number of filters at each upscale layer will reduce from the chosen "
            "max_filters at the first layer to the chosen min_filters at the last layer as "
            "dictated by the dec_filter_slope."
            "\n\tcap_max: The filters will decline at a fixed rate from each upscale to the next "
            "based on the filter_slope setting. If there are more upscales than filters, "
            "then the earliest upscales will be capped at the max_filter value until the filters "
            "can reduce to the min_filters value at the final upscale. (EG: 512 -> 512 -> 512 -> "
            "256 -> 128 -> 64)."
            "\n\tcap_min: The filters will decline at a fixed rate from each upscale to the next "
            "based on the filter_slope setting. If there are more upscales than filters, then "
            "the earliest upscales will drop their filters until the min_filter value is met and "
            "repeat the min_filter value for the remaining upscales. (EG: 512 -> 256 -> 128 -> "
            "64 -> 64 -> 64)."),
        "choices": ["full", "cap_max", "cap_min"],
        "group": "decoder",
        "fixed": True,
        "gui_radio": True},
    "dec_filter_slope": {
        "default": -0.45,
        "info": (
            "The rate that the filters reduce at each upscale layer.\n"
            "\n\tFull Slope Mode: Negative numbers will drop the number of filters quicker at "
            "first and slow down each upscale. Positive numbers will drop the number of filters "
            "slower at first but then speed up each upscale. A value of 0.0 will reduce at a "
            "linear rate (i.e. the same number of filters will be reduced at each upscale).\n"
            "\n\tCap Min/Max Slope Mode: Only positive values will work here. Negative values "
            "will automatically be converted to their positive counterpart. A value of 0.5 will "
            "halve the number of filters at each upscale until the minimum value is reached. A "
            "value of 0.33 will be reduce the number of filters by a third until the minimum "
            "value is reached etc."),
        "datatype": float,
        "min_max": (-.99, .99),
        "rounding": 2,
        "group": "decoder",
        "fixed": True},
    "dec_res_blocks": {
        "default": 1,
        "info": (
            "The number of Residual Blocks to apply to each upscale layer. Set to 0 to disable "
            "residual blocks entirely."),
        "datatype": int,
        "rounding": 1,
        "min_max": (0, 8),
        "group": "decoder",
        "fixed": True},
    "dec_output_kernel": {
        "default": 5,
        "info": "The kernel size to apply to the final Convolution layer.",
        "datatype": int,
        "rounding": 2,
        "min_max": (1, 9),
        "group": "decoder",
        "fixed": True},
    "dec_gaussian": {
        "default": True,
        "info": (
            "Gaussian Noise acts as a regularization technique for preventing overfitting of "
            "data."
            "\n\tTrue - Apply a Gaussian Noise layer to each upscale."
            "\n\tFalse - Don't apply a Gaussian Noise layer to each upscale."),
        "datatype": bool,
        "group": "decoder",
        "fixed": True},
    "dec_skip_last_residual": {
        "default": True,
        "info": (
            "If Residual blocks have been enabled, enabling this option will not apply a "
            "Residual block to the final upscaler."
            "\n\tTrue - Don't apply a Residual block to the final upscale."
            "\n\tFalse - Apply a Residual block to all upscale layers."),
        "datatype": bool,
        "group": "decoder",
        "fixed": True},

    # Weight management
    "freeze_layers": {
        "default": "keras_encoder",
        "info": (
            "If the command line option 'freeze-weights' is enabled, then the layers indicated "
            "here will be frozen the next time the model starts up. NB: Not all architectures "
            "contain all of the layers listed here, so any layers marked for freezing that are "
            "not within your chosen architecture will be ignored. EG:\n If 'split fc' has "
            "been selected, then 'fc_a' and 'fc_b' are available for freezing. If it has "
            "not been selected then 'fc_both' is available for freezing."),
        "datatype": list,
        "choices": ["encoder", "keras_encoder", "fc_a", "fc_b", "fc_both", "fc_shared",
                    "fc_gblock", "g_block_a", "g_block_b", "g_block_both", "decoder_a",
                    "decoder_b", "decoder_both"],
        "group": "weights",
        "fixed": False},
    "load_layers": {
        "default": "encoder",
        "info": (
            "If the command line option 'load-weights' is populated, then the layers indicated "
            "here will be loaded from the given weights file if starting a new model. NB Not all "
            "architectures contain all of the layers listed here, so any layers marked for "
            "loading that are not within your chosen architecture will be ignored. EG:\n If "
            "'split fc' has been selected, then 'fc_a' and 'fc_b' are available for loading. If "
            "it has not been selected then 'fc_both' is available for loading."),
        "datatype": list,
        "choices": ["encoder", "fc_a", "fc_b", "fc_both", "fc_shared", "fc_gblock", "g_block_a",
                    "g_block_b", "g_block_both", "decoder_a", "decoder_b", "decoder_both"],
        "group": "weights",
        "fixed": True},

    # # SPECIFIC ENCODER SETTINGS # #
    # Faceswap Original
    "fs_original_depth": {
        "default": 4,
        "info": "Faceswap Encoder only: The number of convolutions to perform within the encoder.",
        "datatype": int,
        "min_max": (2, 10),
        "rounding": 1,
        "group": "faceswap encoder configuration",
        "fixed": True},
    "fs_original_min_filters": {
        "default": 128,
        "info": (
            "Faceswap Encoder only: The minumum number of filters to use for encoder "
            "convolutions. (i.e. the number of filters to use for the first encoder layer)."),
        "datatype": int,
        "min_max": (16, 2048),
        "rounding": 64,
        "group": "faceswap encoder configuration",
        "fixed": True},
    "fs_original_max_filters": {
        "default": 1024,
        "info": (
            "Faceswap Encoder only: The maximum number of filters to use for encoder "
            "convolutions. (i.e. the number of filters to use for the final encoder layer)."),
        "datatype": int,
        "min_max": (256, 8192),
        "rounding": 128,
        "group": "faceswap encoder configuration",
        "fixed": True},
    "fs_original_use_alt": {
        "default": False,
        "info": (
            "Use a slightly alternate version of the Faceswap Encoder."
            "\n\tTrue - Use the alternate variation of the Faceswap Encoder."
            "\n\tFalse - Use the original Faceswap Encoder."),
        "datatype": bool,
        "group": "faceswap encoder configuration",
        "fixed": True},

    # MobileNet
    "mobilenet_width": {
        "default": 1.0,
        "info": (
            "The width multiplier for mobilenet encoders. Controls the width of the "
            "network. Values less than 1.0 proportionally decrease the number of filters within "
            "each layer. Values greater than 1.0 proportionally increase the number of filters "
            "within each layer. 1.0 is the default number of layers used within the paper.\n"
            "NB: This option is ignored for any non-mobilenet encoders.\n"
            "NB: If loading ImageNet weights, then for MobilenetV1 only values of '0.25', "
            "'0.5', '0.75' or '1.0 can be selected. For MobilenetV2 only values of '0.35', "
            "'0.50', '0.75', '1.0', '1.3' or '1.4' can be selected. For mobilenet_v3 only values "
            "of '0.75' or '1.0' can be selected"),
        "datatype": float,
        "min_max": (0.1, 2.0),
        "rounding": 2,
        "group": "mobilenet encoder configuration",
        "fixed": True},
    "mobilenet_depth": {
        "default": 1,
        "info": (
            "The depth multiplier for MobilenetV1 encoder. This is the depth multiplier "
            "for depthwise convolution (known as the resolution multiplier within the original "
            "paper).\n"
            "NB: This option is only used for MobilenetV1 and is ignored for all other "
            "encoders.\n"
            "NB: If loading ImageNet weights, this must be set to 1."),
        "datatype": int,
        "min_max": (1, 10),
        "rounding": 1,
        "group": "mobilenet encoder configuration",
        "fixed": True},
    "mobilenet_dropout": {
        "default": 0.001,
        "info": (
            "The dropout rate for MobilenetV1 encoder.\n"
            "NB: This option is only used for MobilenetV1 and is ignored for all other "
            "encoders."),
        "datatype": float,
        "min_max": (0.001, 2.0),
        "rounding": 3,
        "group": "mobilenet encoder configuration",
        "fixed": True},
    "mobilenet_minimalistic": {
        "default": False,
        "info": (
            "Use a minimilist version of MobilenetV3.\n"
            "In addition to large and small models MobilenetV3 also contains so-called "
            "minimalistic models, these models have the same per-layer dimensions characteristic "
            "as MobilenetV3 however, they don't utilize any of the advanced blocks "
            "(squeeze-and-excite units, hard-swish, and 5x5 convolutions). While these models "
            "are less efficient on CPU, they are much more performant on GPU/DSP.\n"
            "NB: This option is only used for MobilenetV3 and is ignored for all other "
            "encoders.\n"),
        "datatype": bool,
        "group": "mobilenet encoder configuration",
        "fixed": True},
    }
