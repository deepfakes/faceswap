#!/usr/bin/env python3
""" Command Line Arguments for tools """
import argparse
import gettext

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import (DirOrFileFullPaths, DirFullPaths, FileFullPaths, Radio, Slider)
from plugins.plugin_loader import PluginLoader


# LOCALES
_LANG = gettext.translation("tools.mask.cli", localedir="locales", fallback=True)
_ = _LANG.gettext

_HELPTEXT = _("This tool allows you to generate, import, export or preview masks for existing "
              "alignments.")


class MaskArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Mask tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return _("Mask tool\nGenerate, import, export or preview masks for existing alignments "
                 "files.")

    @staticmethod
    def get_argument_list():
        argument_list = []
        argument_list.append({
            "opts": ("-a", "--alignments"),
            "action": FileFullPaths,
            "type": str,
            "group": _("data"),
            "required": False,
            "filetypes": "alignments",
            "help": _(
                "Full path to the alignments file that contains the masks if not at the "
                "default location. NB: If the input-type is faces and you wish to update the "
                "corresponding alignments file, then you must provide a value here as the "
                "location cannot be automatically detected.")})
        argument_list.append({
            "opts": ("-i", "--input"),
            "action": DirOrFileFullPaths,
            "type": str,
            "group": _("data"),
            "filetypes": "video",
            "required": True,
            "help": _(
                "Directory containing extracted faces, source frames, or a video file.")})
        argument_list.append({
            "opts": ("-I", "--input-type"),
            "action": Radio,
            "type": str.lower,
            "choices": ("faces", "frames"),
            "dest": "input_type",
            "group": _("data"),
            "default": "frames",
            "help": _(
                "R|Whether the `input` is a folder of faces or a folder frames/video"
                "\nL|faces: The input is a folder containing extracted faces."
                "\nL|frames: The input is a folder containing frames or is a video")})
        argument_list.append({
            "opts": ("-B", "--batch-mode"),
            "action": "store_true",
            "dest": "batch_mode",
            "default": False,
            "group": _("data"),
            "help": _(
                "R|Run the mask tool on multiple sources. If selected then the other options "
                "should be set as follows:"
                "\nL|input: A parent folder containing either all of the video files to be "
                "processed, or containing sub-folders of frames/faces."
                "\nL|output-folder: If provided, then sub-folders will be created within the "
                "given location to hold the previews for each input."
                "\nL|alignments: Alignments field will be ignored for batch processing. The "
                "alignments files must exist at the default location (for frames). For batch "
                "processing of masks with 'faces' as the input type, then only the PNG header "
                "within the extracted faces will be updated.")})
        argument_list.append({
            "opts": ("-M", "--masker"),
            "action": Radio,
            "type": str.lower,
            "choices": PluginLoader.get_available_extractors("mask"),
            "default": "extended",
            "group": _("process"),
            "help": _(
                "R|Masker to use."
                "\nL|bisenet-fp: Relatively lightweight NN based mask that provides more "
                "refined control over the area to be masked including full head masking "
                "(configurable in mask settings)."
                "\nL|components: Mask designed to provide facial segmentation based on the "
                "positioning of landmark locations. A convex hull is constructed around the "
                "exterior of the landmarks to create a mask."
                "\nL|custom: A dummy mask that fills the mask area with all 1s or 0s "
                "(configurable in settings). This is only required if you intend to manually "
                "edit the custom masks yourself in the manual tool. This mask does not use the "
                "GPU."
                "\nL|extended: Mask designed to provide facial segmentation based on the "
                "positioning of landmark locations. A convex hull is constructed around the "
                "exterior of the landmarks and the mask is extended upwards onto the forehead."
                "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                "faces clear of obstructions. Profile faces and obstructions may result in "
                "sub-par performance."
                "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
                "frontal faces. The mask model has been specifically trained to recognize "
                "some facial obstructions (hands and eyeglasses). Profile faces may result in "
                "sub-par performance."
                "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                "faces. The mask model has been trained by community members. Profile faces "
                "may result in sub-par performance.")})
        argument_list.append({
            "opts": ("-p", "--processing"),
            "action": Radio,
            "type": str.lower,
            "choices": ("all", "missing", "output", "import"),
            "default": "all",
            "group": _("process"),
            "help": _(
                "R|The Mask tool process to perform."
                "\nL|all: Update the mask for all faces in the alignments file for the selected "
                "'masker'."
                "\nL|missing: Create a mask for all faces in the alignments file where a mask "
                "does not previously exist for the selected 'masker'."
                "\nL|output: Don't update the masks, just output the selected 'masker' for "
                "review/editing in external tools to the given output folder."
                "\nL|import: Import masks that have been edited outside of faceswap into the "
                "alignments file. Note: 'custom' must be the selected 'masker' and the masks must "
                "be in the same format as the 'input-type' (frames or faces)")})
        argument_list.append({
            "opts": ("-m", "--mask-path"),
            "action": DirFullPaths,
            "type": str,
            "group": _("import"),
            "help": _(
                "R|Import only. The path to the folder that contains masks to be imported."
                "\nL|How the masks are provided is not important, but they will be stored, "
                "internally, as 8-bit grayscale images."
                "\nL|If the input are images, then the masks must be named exactly the same as "
                "input frames/faces (excluding the file extension)."
                "\nL|If the input is a video file, then the filename of the masks is not "
                "important but should contain the frame number at the end of the filename (but "
                "before the file extension). The frame number can be separated from the rest of "
                "the filename by any non-numeric character and can be padded by any number of "
                "zeros. The frame number must correspond correctly to the frame number in the "
                "original video (starting from frame 1).")})
        argument_list.append({
            "opts": ("-c", "--centering"),
            "action": Radio,
            "type": str.lower,
            "choices": ("face", "head", "legacy"),
            "default": "face",
            "group": _("import"),
            "help": _(
                "R|Import/Output only. When importing masks, this is the centering to use. For "
                "output this is only used for outputting custom imported masks, and should "
                "correspond to the centering used when importing the mask. Note: For any job "
                "other than 'import' and 'output' this option is ignored as mask centering is "
                "handled internally."
                "\nL|face: Centers the mask on the center of the face, adjusting for "
                "pitch and yaw. Outside of requirements for full head masking/training, this "
                "is likely to be the best choice."
                "\nL|head: Centers the mask on the center of the head, adjusting for "
                "pitch and yaw. Note: You should only select head centering if you intend to "
                "include the full head (including hair) within the mask and are looking to "
                "train a full head model."
                "\nL|legacy: The 'original' extraction technique. Centers the mask near the "
                " of the nose with and crops closely to the face. Can result in the edges of "
                "the mask appearing outside of the training area.")})
        argument_list.append({
            "opts": ("-s", "--storage-size"),
            "dest": "storage_size",
            "action": Slider,
            "type": int,
            "group": _("import"),
            "min_max": (64, 1024),
            "default": 128,
            "rounding": 64,
            "help": _(
                "Import only. The size, in pixels to internally store the mask at.\nThe default "
                "is 128 which is fine for nearly all usecases. Larger sizes will result in larger "
                "alignments files and longer processing.")})
        argument_list.append({
            "opts": ("-o", "--output-folder"),
            "action": DirFullPaths,
            "dest": "output",
            "type": str,
            "group": _("output"),
            "help": _(
                "Optional output location. If provided, a preview of the masks created will "
                "be output in the given folder.")})
        argument_list.append({
            "opts": ("-b", "--blur_kernel"),
            "action": Slider,
            "type": int,
            "group": _("output"),
            "min_max": (0, 9),
            "default": 0,
            "rounding": 1,
            "help": _(
                "Apply gaussian blur to the mask output. Has the effect of smoothing the "
                "edges of the mask giving less of a hard edge. the size is in pixels. This "
                "value should be odd, if an even number is passed in then it will be rounded "
                "to the next odd number. NB: Only effects the output preview. Set to 0 for "
                "off")})
        argument_list.append({
            "opts": ("-t", "--threshold"),
            "action": Slider,
            "type": int,
            "group": _("output"),
            "min_max": (0, 50),
            "default": 0,
            "rounding": 1,
            "help": _(
                "Helps reduce 'blotchiness' on some masks by making light shades white "
                "and dark shades black. Higher values will impact more of the mask. NB: "
                "Only effects the output preview. Set to 0 for off")})
        argument_list.append({
            "opts": ("-O", "--output-type"),
            "action": Radio,
            "type": str.lower,
            "choices": ("combined", "masked", "mask"),
            "default": "combined",
            "group": _("output"),
            "help": _(
                "R|How to format the output when processing is set to 'output'."
                "\nL|combined: The image contains the face/frame, face mask and masked face."
                "\nL|masked: Output the face/frame as rgba image with the face masked."
                "\nL|mask: Only output the mask as a single channel image.")})
        argument_list.append({
            "opts": ("-f", "--full-frame"),
            "action": "store_true",
            "default": False,
            "group": _("output"),
            "help": _(
                "R|Whether to output the whole frame or only the face box when using "
                "output processing. Only has an effect when using frames as input.")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-it", ),
            "type": str,
            "dest": "depr_input-type_it_I",
            "help": argparse.SUPPRESS})
        return argument_list
