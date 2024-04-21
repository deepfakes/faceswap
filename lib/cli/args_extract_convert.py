#!/usr/bin/env python3
""" The Command Line Argument options for extracting and converting with faceswap.py """
import argparse
import gettext
import typing as T

from lib.utils import get_backend
from plugins.plugin_loader import PluginLoader

from .actions import (DirFullPaths, DirOrFileFullPaths, DirOrFilesFullPaths, FileFullPaths,
                      FilesFullPaths, MultiOption, Radio, Slider)
from .args import FaceSwapArgs


# LOCALES
_LANG = gettext.translation("lib.cli.args_extract_convert", localedir="locales", fallback=True)
_ = _LANG.gettext


class ExtractConvertArgs(FaceSwapArgs):
    """ Parent class to capture arguments that will be used in both extract and convert processes.

    Extract and Convert share a fair amount of arguments, so arguments that can be used in both of
    these processes should be placed here.

    No further processing is done in this class (this is handled by the children), this just
    captures the shared arguments.
    """

    @staticmethod
    def get_argument_list() -> list[dict[str, T.Any]]:
        """ Returns the argument list for shared Extract and Convert arguments.

        Returns
        -------
        list
            The list of command line options for the given Extract and Convert
        """
        argument_list: list[dict[str, T.Any]] = []
        argument_list.append({
            "opts": ("-i", "--input-dir"),
            "action": DirOrFileFullPaths,
            "filetypes": "video",
            "dest": "input_dir",
            "required": True,
            "group": _("Data"),
            "help": _(
                "Input directory or video. Either a directory containing the image files you wish "
                "to process or path to a video file. NB: This should be the source video/frames "
                "NOT the source faces.")})
        argument_list.append({
            "opts": ("-o", "--output-dir"),
            "action": DirFullPaths,
            "dest": "output_dir",
            "required": True,
            "group": _("Data"),
            "help": _("Output directory. This is where the converted files will be saved.")})
        argument_list.append({
            "opts": ("-p", "--alignments"),
            "action": FileFullPaths,
            "filetypes": "alignments",
            "type": str,
            "dest": "alignments_path",
            "group": _("Data"),
            "help": _(
                "Optional path to an alignments file. Leave blank if the alignments file is at "
                "the default location.")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-al", ),
            "action": FileFullPaths,
            "filetypes": "alignments",
            "type": str,
            "dest": "depr_alignments_al_p",
            "help": argparse.SUPPRESS})
        return argument_list


class ExtractArgs(ExtractConvertArgs):
    """ Creates the command line arguments for extraction.

    This class inherits base options from :class:`ExtractConvertArgs` where arguments that are used
    for both Extract and Convert should be placed.

    Commands explicit to Extract should be added in :func:`get_optional_arguments`
    """

    @staticmethod
    def get_info() -> str:
        """ The information text for the Extract command.

        Returns
        -------
        str
            The information text for the Extract command.
        """
        return _("Extract faces from image or video sources.\n"
                 "Extraction plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_optional_arguments() -> list[dict[str, T.Any]]:
        """ Returns the argument list unique to the Extract command.

        Returns
        -------
        list
            The list of optional command line options for the Extract command
        """
        if get_backend() == "cpu":
            default_detector = "mtcnn"
            default_aligner = "cv2-dnn"
        else:
            default_detector = "s3fd"
            default_aligner = "fan"

        argument_list: list[dict[str, T.Any]] = []
        argument_list.append({
            "opts": ("-b", "--batch-mode"),
            "action": "store_true",
            "dest": "batch_mode",
            "default": False,
            "group": _("Data"),
            "help": _(
                "R|If selected then the input_dir should be a parent folder containing multiple "
                "videos and/or folders of images you wish to extract from. The faces will be "
                "output to separate sub-folders in the output_dir.")})
        argument_list.append({
            "opts": ("-D", "--detector"),
            "action": Radio,
            "type": str.lower,
            "default": default_detector,
            "choices": PluginLoader.get_available_extractors("detect"),
            "group": _("Plugins"),
            "help": _(
                "R|Detector to use. Some of these have configurable settings in "
                "'/config/extract.ini' or 'Settings > Configure Extract 'Plugins':"
                "\nL|cv2-dnn: A CPU only extractor which is the least reliable and least resource "
                "intensive. Use this if not using a GPU and time is important."
                "\nL|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources than "
                "other GPU detectors but can often return more false positives."
                "\nL|s3fd: Best detector. Slow on CPU, faster on GPU. Can detect more faces and "
                "fewer false positives than other GPU detectors, but is a lot more resource "
                "intensive."
                "\nL|external: Import a face detection bounding box from a json file. ("
                "configurable in Detect settings)")})
        argument_list.append({
            "opts": ("-A", "--aligner"),
            "action": Radio,
            "type": str.lower,
            "default": default_aligner,
            "choices": PluginLoader.get_available_extractors("align"),
            "group": _("Plugins"),
            "help": _(
                "R|Aligner to use."
                "\nL|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, but "
                "less accurate. Only use this if not using a GPU and time is important."
                "\nL|fan: Best aligner. Fast on GPU, slow on CPU."
                "\nL|external: Import 68 point 2D landmarks or an aligned bounding box from a "
                "json file. (configurable in Align settings)")})
        argument_list.append({
            "opts": ("-M", "--masker"),
            "action": MultiOption,
            "type": str.lower,
            "nargs": "+",
            "choices": [mask for mask in PluginLoader.get_available_extractors("mask")
                        if mask not in ("components", "extended")],
            "group": _("Plugins"),
            "help": _(
                "R|Additional Masker(s) to use. The masks generated here will all take up GPU "
                "RAM. You can select none, one or multiple masks, but the extraction may take "
                "longer the more you select. NB: The Extended and Components (landmark based) "
                "masks are automatically generated on extraction."
                "\nL|bisenet-fp: Relatively lightweight NN based mask that provides more refined "
                "control over the area to be masked including full head masking (configurable in "
                "mask settings)."
                "\nL|custom: A dummy mask that fills the mask area with all 1s or 0s ("
                "configurable in settings). This is only required if you intend to manually edit "
                "the custom masks yourself in the manual tool. This mask does not use the GPU so "
                "will not use any additional VRAM."
                "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                "faces clear of obstructions. Profile faces and obstructions may result in "
                "sub-par performance."
                "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
                "frontal faces. The mask model has been specifically trained to recognize some "
                "facial obstructions (hands and eyeglasses). Profile faces may result in sub-par "
                "performance."
                "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                "faces. The mask model has been trained by community members and will need "
                "testing for further description. Profile faces may result in sub-par "
                "performance."
                "\nThe auto generated masks are as follows:"
                "\nL|components: Mask designed to provide facial segmentation based on the "
                "positioning of landmark locations. A convex hull is constructed around the "
                "exterior of the landmarks to create a mask."
                "\nL|extended: Mask designed to provide facial segmentation based on the "
                "positioning of landmark locations. A convex hull is constructed around the "
                "exterior of the landmarks and the mask is extended upwards onto the forehead."
                "\n(eg: `-M unet-dfl vgg-clear`, `--masker vgg-obstructed`)")})
        argument_list.append({
            "opts": ("-O", "--normalization"),
            "action": Radio,
            "type": str.lower,
            "dest": "normalization",
            "default": "none",
            "choices": ["none", "clahe", "hist", "mean"],
            "group": _("Plugins"),
            "help": _(
                "R|Performing normalization can help the aligner better align faces with "
                "difficult lighting conditions at an extraction speed cost. Different methods "
                "will yield different results on different sets. NB: This does not impact the "
                "output face, just the input to the aligner."
                "\nL|none: Don't perform normalization on the face."
                "\nL|clahe: Perform Contrast Limited Adaptive Histogram Equalization on the face."
                "\nL|hist: Equalize the histograms on the RGB channels."
                "\nL|mean: Normalize the face colors to the mean.")})
        argument_list.append({
            "opts": ("-R", "--re-feed"),
            "action": Slider,
            "min_max": (0, 10),
            "rounding": 1,
            "type": int,
            "dest": "re_feed",
            "default": 0,
            "group": _("Plugins"),
            "help": _(
                "The number of times to re-feed the detected face into the aligner. Each time the "
                "face is re-fed into the aligner the bounding box is adjusted by a small amount. "
                "The final landmarks are then averaged from each iteration. Helps to remove "
                "'micro-jitter' but at the cost of slower extraction speed. The more times the "
                "face is re-fed into the aligner, the less micro-jitter should occur but the "
                "longer extraction will take.")})
        argument_list.append({
            "opts": ("-a", "--re-align"),
            "action": "store_true",
            "dest": "re_align",
            "default": False,
            "group": _("Plugins"),
            "help": _(
                "Re-feed the initially found aligned face through the aligner. Can help produce "
                "better alignments for faces that are rotated beyond 45 degrees in the frame or "
                "are at extreme angles. Slows down extraction.")})
        argument_list.append({
            "opts": ("-r", "--rotate-images"),
            "type": str,
            "dest": "rotate_images",
            "default": None,
            "group": _("Plugins"),
            "help": _(
                "If a face isn't found, rotate the images to try to find a face. Can find more "
                "faces at the cost of extraction speed. Pass in a single number to use increments "
                "of that size up to 360, or pass in a list of numbers to enumerate exactly what "
                "angles to check.")})
        argument_list.append({
            "opts": ("-I", "--identity"),
            "action": "store_true",
            "default": False,
            "group": _("Plugins"),
            "help": _(
                "Obtain and store face identity encodings from VGGFace2. Slows down extract a "
                "little, but will save time if using 'sort by face'")})
        argument_list.append({
            "opts": ("-m", "--min-size"),
            "action": Slider,
            "min_max": (0, 1080),
            "rounding": 20,
            "type": int,
            "dest": "min_size",
            "default": 0,
            "group": _("Face Processing"),
            "help": _(
                "Filters out faces detected below this size. Length, in pixels across the "
                "diagonal of the bounding box. Set to 0 for off")})
        argument_list.append({
            "opts": ("-n", "--nfilter"),
            "action": DirOrFilesFullPaths,
            "filetypes": "image",
            "dest": "nfilter",
            "default": None,
            "nargs": "+",
            "group": _("Face Processing"),
            "help": _(
                "Optionally filter out people who you do not wish to extract by passing in images "
                "of those people. Should be a small variety of images at different angles and in "
                "different conditions. A folder containing the required images or multiple image "
                "files, space separated, can be selected.")})
        argument_list.append({
            "opts": ("-f", "--filter"),
            "action": DirOrFilesFullPaths,
            "filetypes": "image",
            "dest": "filter",
            "default": None,
            "nargs": "+",
            "group": _("Face Processing"),
            "help": _(
                "Optionally select people you wish to extract by passing in images of that "
                "person. Should be a small variety of images at different angles and in different "
                "conditions A folder containing the required images or multiple image files, "
                "space separated, can be selected.")})
        argument_list.append({
            "opts": ("-l", "--ref_threshold"),
            "action": Slider,
            "min_max": (0.01, 0.99),
            "rounding": 2,
            "type": float,
            "dest": "ref_threshold",
            "default": 0.60,
            "group": _("Face Processing"),
            "help": _(
                "For use with the optional nfilter/filter files. Threshold for positive face "
                "recognition. Higher values are stricter.")})
        argument_list.append({
            "opts": ("-z", "--size"),
            "action": Slider,
            "min_max": (256, 1024),
            "rounding": 64,
            "type": int,
            "default": 512,
            "group": _("output"),
            "help": _(
                "The output size of extracted faces. Make sure that the model you intend to train "
                "supports your required size. This will only need to be changed for hi-res "
                "models.")})
        argument_list.append({
            "opts": ("-N", "--extract-every-n"),
            "action": Slider,
            "min_max": (1, 100),
            "rounding": 1,
            "type": int,
            "dest": "extract_every_n",
            "default": 1,
            "group": _("output"),
            "help": _(
                "Extract every 'nth' frame. This option will skip frames when extracting faces. "
                "For example a value of 1 will extract faces from every frame, a value of 10 will "
                "extract faces from every 10th frame.")})
        argument_list.append({
            "opts": ("-v", "--save-interval"),
            "action": Slider,
            "min_max": (0, 1000),
            "rounding": 10,
            "type": int,
            "dest": "save_interval",
            "default": 0,
            "group": _("output"),
            "help": _(
                "Automatically save the alignments file after a set amount of frames. By default "
                "the alignments file is only saved at the end of the extraction process. NB: If "
                "extracting in 2 passes then the alignments file will only start to be saved out "
                "during the second pass. WARNING: Don't interrupt the script when writing the "
                "file because it might get corrupted. Set to 0 to turn off")})
        argument_list.append({
            "opts": ("-B", "--debug-landmarks"),
            "action": "store_true",
            "dest": "debug_landmarks",
            "default": False,
            "group": _("output"),
            "help": _("Draw landmarks on the ouput faces for debugging purposes.")})
        argument_list.append({
            "opts": ("-P", "--singleprocess"),
            "action": "store_true",
            "default": False,
            "backend": ("nvidia", "directml", "rocm", "apple_silicon"),
            "group": _("settings"),
            "help": _(
                "Don't run extraction in parallel. Will run each part of the extraction process "
                "separately (one after the other) rather than all at the same time. Useful if "
                "VRAM is at a premium.")})
        argument_list.append({
            "opts": ("-s", "--skip-existing"),
            "action": "store_true",
            "dest": "skip_existing",
            "default": False,
            "group": _("settings"),
            "help": _(
                "Skips frames that have already been extracted and exist in the alignments file")})
        argument_list.append({
            "opts": ("-e", "--skip-existing-faces"),
            "action": "store_true",
            "dest": "skip_faces",
            "default": False,
            "group": _("settings"),
            "help": _("Skip frames that already have detected faces in the alignments file")})
        argument_list.append({
            "opts": ("-K", "--skip-saving-faces"),
            "action": "store_true",
            "dest": "skip_saving_faces",
            "default": False,
            "group": _("settings"),
            "help": _("Skip saving the detected faces to disk. Just create an alignments file")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-min", ),
            "type": int,
            "dest": "depr_min-size_min_m",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-een", ),
            "type": int,
            "dest": "depr_extract-every-n_een_N",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-nm",),
            "type": str.lower,
            "dest": "depr_normalization_nm_O",
            "choices": ["none", "clahe", "hist", "mean"],
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-rf", ),
            "type": int,
            "dest": "depr_re-feed_rf_R",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-sz", ),
            "type": int,
            "dest": "depr_size_sz_z",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-si", ),
            "type": int,
            "dest": "depr_save-interval_si_v",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-dl", ),
            "action": "store_true",
            "dest": "depr_debug-landmarks_dl_B",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-sp", ),
            "dest": "depr_singleprocess_sp_P",
            "action": "store_true",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-sf", ),
            "action": "store_true",
            "dest": "depr_skip-existing-faces_sf_e",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-ssf", ),
            "action": "store_true",
            "dest": "depr_skip-saving-faces_ssf_K",
            "help": argparse.SUPPRESS})
        return argument_list


class ConvertArgs(ExtractConvertArgs):
    """ Creates the command line arguments for conversion.

    This class inherits base options from :class:`ExtractConvertArgs` where arguments that are used
    for both Extract and Convert should be placed.

    Commands explicit to Convert should be added in :func:`get_optional_arguments`
    """

    @staticmethod
    def get_info() -> str:
        """ The information text for the Convert command.

        Returns
        -------
        str
            The information text for the Convert command.
        """
        return _("Swap the original faces in a source video/images to your final faces.\n"
                 "Conversion plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_optional_arguments() -> list[dict[str, T.Any]]:
        """ Returns the argument list unique to the Convert command.

        Returns
        -------
        list
            The list of optional command line options for the Convert command
        """

        argument_list: list[dict[str, T.Any]] = []
        argument_list.append({
            "opts": ("-r", "--reference-video"),
            "action": FileFullPaths,
            "filetypes": "video",
            "type": str,
            "dest": "reference_video",
            "group": _("Data"),
            "help": _(
                "Only required if converting from images to video. Provide The original video "
                "that the source frames were extracted from (for extracting the fps and audio).")})
        argument_list.append({
            "opts": ("-m", "--model-dir"),
            "action": DirFullPaths,
            "dest": "model_dir",
            "required": True,
            "group": _("Data"),
            "help": _(
                "Model directory. The directory containing the trained model you wish to use for "
                "conversion.")})
        argument_list.append({
            "opts": ("-c", "--color-adjustment"),
            "action": Radio,
            "type": str.lower,
            "dest": "color_adjustment",
            "default": "avg-color",
            "choices": PluginLoader.get_available_convert_plugins("color", True),
            "group": _("Plugins"),
            "help": _(
                "R|Performs color adjustment to the swapped face. Some of these options have "
                "configurable settings in '/config/convert.ini' or 'Settings > Configure Convert "
                "Plugins':"
                "\nL|avg-color: Adjust the mean of each color channel in the swapped "
                "reconstruction to equal the mean of the masked area in the original image."
                "\nL|color-transfer: Transfers the color distribution from the source to the "
                "target image using the mean and standard deviations of the L*a*b* color space."
                "\nL|manual-balance: Manually adjust the balance of the image in a variety of "
                "color spaces. Best used with the Preview tool to set correct values."
                "\nL|match-hist: Adjust the histogram of each color channel in the swapped "
                "reconstruction to equal the histogram of the masked area in the original image."
                "\nL|seamless-clone: Use cv2's seamless clone function to remove extreme "
                "gradients at the mask seam by smoothing colors. Generally does not give very "
                "satisfactory results."
                "\nL|none: Don't perform color adjustment.")})
        argument_list.append({
            "opts": ("-M", "--mask-type"),
            "action": Radio,
            "type": str.lower,
            "dest": "mask_type",
            "default": "extended",
            "choices": PluginLoader.get_available_extractors("mask",
                                                             add_none=True,
                                                             extend_plugin=True) + ["predicted"],
            "group": _("Plugins"),
            "help": _(
                "R|Masker to use. NB: The mask you require must exist within the alignments file. "
                "You can add additional masks with the Mask Tool."
                "\nL|none: Don't use a mask."
                "\nL|bisenet-fp_face: Relatively lightweight NN based mask that provides more "
                "refined control over the area to be masked (configurable in mask settings). Use "
                "this version of bisenet-fp if your model is trained with 'face' or "
                "'legacy' centering."
                "\nL|bisenet-fp_head: Relatively lightweight NN based mask that provides more "
                "refined control over the area to be masked (configurable in mask settings). Use "
                "this version of bisenet-fp if your model is trained with 'head' centering."
                "\nL|custom_face: Custom user created, face centered mask."
                "\nL|custom_head: Custom user created, head centered mask."
                "\nL|components: Mask designed to provide facial segmentation based on the "
                "positioning of landmark locations. A convex hull is constructed around the "
                "exterior of the landmarks to create a mask."
                "\nL|extended: Mask designed to provide facial segmentation based on the "
                "positioning of landmark locations. A convex hull is constructed around the "
                "exterior of the landmarks and the mask is extended upwards onto the forehead."
                "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                "faces clear of obstructions. Profile faces and obstructions may result in sub-"
                "par performance."
                "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
                "frontal faces. The mask model has been specifically trained to recognize some "
                "facial obstructions (hands and eyeglasses). Profile faces may result in sub-par "
                "performance."
                "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                "faces. The mask model has been trained by community members and will need "
                "testing for further description. Profile faces may result in sub-par "
                "performance."
                "\nL|predicted: If the 'Learn Mask' option was enabled during training, this will "
                "use the mask that was created by the trained model.")})
        argument_list.append({
            "opts": ("-w", "--writer"),
            "action": Radio,
            "type": str,
            "default": "opencv",
            "choices": PluginLoader.get_available_convert_plugins("writer", False),
            "group": _("Plugins"),
            "help": _(
                "R|The plugin to use to output the converted images. The writers are configurable "
                "in '/config/convert.ini' or 'Settings > Configure Convert Plugins:'"
                "\nL|ffmpeg: [video] Writes out the convert straight to video. When the input is "
                "a series of images then the '-ref' (--reference-video) parameter must be set."
                "\nL|gif: [animated image] Create an animated gif."
                "\nL|opencv: [images] The fastest image writer, but less options and formats than "
                "other plugins."
                "\nL|patch: [images] Outputs the raw swapped face patch, along with the "
                "transformation matrix required to re-insert the face back into the original "
                "frame. Use this option if you wish to post-process and composite the final face "
                "within external tools."
                "\nL|pillow: [images] Slower than opencv, but has more options and supports more "
                "formats.")})
        argument_list.append({
            "opts": ("-O", "--output-scale"),
            "action": Slider,
            "min_max": (25, 400),
            "rounding": 1,
            "type": int,
            "dest": "output_scale",
            "default": 100,
            "group": _("Frame Processing"),
            "help": _(
                "Scale the final output frames by this amount. 100%% will output the frames at "
                "source dimensions. 50%% at half size 200%% at double size")})
        argument_list.append({
            "opts": ("-R", "--frame-ranges"),
            "type": str,
            "nargs": "+",
            "dest": "frame_ranges",
            "group": _("Frame Processing"),
            "help": _(
                "Frame ranges to apply transfer to e.g. For frames 10 to 50 and 90 to 100 use "
                "--frame-ranges 10-50 90-100. Frames falling outside of the selected range will "
                "be discarded unless '-k' (--keep-unchanged) is selected. NB: If you are "
                "converting from images, then the filenames must end with the frame-number!")})
        argument_list.append({
            "opts": ("-S", "--face-scale"),
            "action": Slider,
            "min_max": (-10.0, 10.0),
            "rounding": 2,
            "dest": "face_scale",
            "type": float,
            "default": 0.0,
            "group": _("Face Processing"),
            "help": _(
                "Scale the swapped face by this percentage. Positive values will enlarge the "
                "face, Negative values will shrink the face.")})
        argument_list.append({
            "opts": ("-a", "--input-aligned-dir"),
            "action": DirFullPaths,
            "dest": "input_aligned_dir",
            "default": None,
            "group": _("Face Processing"),
            "help": _(
                "If you have not cleansed your alignments file, then you can filter out faces by "
                "defining a folder here that contains the faces extracted from your input files/"
                "video. If this folder is defined, then only faces that exist within your "
                "alignments file and also exist within the specified folder will be converted. "
                "Leaving this blank will convert all faces that exist within the alignments "
                "file.")})
        argument_list.append({
            "opts": ("-n", "--nfilter"),
            "action": FilesFullPaths,
            "filetypes": "image",
            "dest": "nfilter",
            "default": None,
            "nargs": "+",
            "group": _("Face Processing"),
            "help": _(
                "Optionally filter out people who you do not wish to process by passing in an "
                "image of that person. Should be a front portrait with a single person in the "
                "image. Multiple images can be added space separated. NB: Using face filter will "
                "significantly decrease extraction speed and its accuracy cannot be guaranteed.")})
        argument_list.append({
            "opts": ("-f", "--filter"),
            "action": FilesFullPaths,
            "filetypes": "image",
            "dest": "filter",
            "default": None,
            "nargs": "+",
            "group": _("Face Processing"),
            "help": _(
                "Optionally select people you wish to process by passing in an image of that "
                "person. Should be a front portrait with a single person in the image. Multiple "
                "images can be added space separated. NB: Using face filter will significantly "
                "decrease extraction speed and its accuracy cannot be guaranteed.")})
        argument_list.append({
            "opts": ("-l", "--ref_threshold"),
            "action": Slider,
            "min_max": (0.01, 0.99),
            "rounding": 2,
            "type": float,
            "dest": "ref_threshold",
            "default": 0.4,
            "group": _("Face Processing"),
            "help": _(
                "For use with the optional nfilter/filter files. Threshold for positive face "
                "recognition. Lower values are stricter. NB: Using face filter will significantly "
                "decrease extraction speed and its accuracy cannot be guaranteed.")})
        argument_list.append({
            "opts": ("-j", "--jobs"),
            "action": Slider,
            "min_max": (0, 40),
            "rounding": 1,
            "type": int,
            "dest": "jobs",
            "default": 0,
            "group": _("settings"),
            "help": _(
                "The maximum number of parallel processes for performing conversion. Converting "
                "images is system RAM heavy so it is possible to run out of memory if you have a "
                "lot of processes and not enough RAM to accommodate them all. Setting this to 0 "
                "will use the maximum available. No matter what you set this to, it will never "
                "attempt to use more processes than are available on your system. If "
                "singleprocess is enabled this setting will be ignored.")})
        argument_list.append({
            "opts": ("-T", "--on-the-fly"),
            "action": "store_true",
            "dest": "on_the_fly",
            "default": False,
            "group": _("settings"),
            "help": _(
                "Enable On-The-Fly Conversion. NOT recommended. You should generate a clean "
                "alignments file for your destination video. However, if you wish you can "
                "generate the alignments on-the-fly by enabling this option. This will use an "
                "inferior extraction pipeline and will lead to substandard results. If an "
                "alignments file is found, this option will be ignored.")})
        argument_list.append({
            "opts": ("-k", "--keep-unchanged"),
            "action": "store_true",
            "dest": "keep_unchanged",
            "default": False,
            "group": _("Frame Processing"),
            "help": _(
                "When used with --frame-ranges outputs the unchanged frames that are not "
                "processed instead of discarding them.")})
        argument_list.append({
            "opts": ("-s", "--swap-model"),
            "action": "store_true",
            "dest": "swap_model",
            "default": False,
            "group": _("settings"),
            "help": _("Swap the model. Instead converting from of A -> B, converts B -> A")})
        argument_list.append({
            "opts": ("-P", "--singleprocess"),
            "action": "store_true",
            "default": False,
            "group": _("settings"),
            "help": _("Disable multiprocessing. Slower but less resource intensive.")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-sp", ),
            "action": "store_true",
            "dest": "depr_singleprocess_sp_P",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-ref", ),
            "type": str,
            "dest": "depr_reference-video_ref_r",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-fr", ),
            "type": str,
            "nargs": "+",
            "dest": "depr_frame-ranges_fr_R",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-osc", ),
            "type": int,
            "dest": "depr_output-scale_osc_O",
            "help": argparse.SUPPRESS})
        argument_list.append({
            "opts": ("-otf", ),
            "action": "store_true",
            "dest": "depr_on-the-fly_otf_T",
            "help": argparse.SUPPRESS})
        return argument_list
