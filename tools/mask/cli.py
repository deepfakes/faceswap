#!/usr/bin/env python3
""" Command Line Arguments for tools """
import gettext

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import (DirOrFileFullPaths, DirFullPaths, FileFullPaths, Radio, Slider)
from plugins.plugin_loader import PluginLoader


# LOCALES
_LANG = gettext.translation("tools.mask.cli", localedir="locales", fallback=True)
_ = _LANG.gettext


_HELPTEXT = _("This command lets you generate masks for existing alignments.")


class MaskArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for Mask tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return _("Mask tool\nGenerate masks for existing alignments files.")

    @staticmethod
    def get_argument_list():
        argument_list = []
        argument_list.append(dict(
            opts=("-a", "--alignments"),
            action=FileFullPaths,
            type=str,
            group=_("data"),
            required=False,
            filetypes="alignments",
            help=_("Full path to the alignments file to add the mask to if not at the default "
                   "location. NB: If the input-type is faces and you wish to update the "
                   "corresponding alignments file, then you must provide a value here as the "
                   "location cannot be automatically detected.")))
        argument_list.append(dict(
            opts=("-i", "--input"),
            action=DirOrFileFullPaths,
            type=str,
            group=_("data"),
            filetypes="video",
            required=True,
            help=_("Directory containing extracted faces, source frames, or a video file.")))
        argument_list.append(dict(
            opts=("-it", "--input-type"),
            action=Radio,
            type=str.lower,
            choices=("faces", "frames"),
            dest="input_type",
            group=_("data"),
            default="frames",
            help=_("R|Whether the `input` is a folder of faces or a folder frames/video"
                   "\nL|faces: The input is a folder containing extracted faces."
                   "\nL|frames: The input is a folder containing frames or is a video")))
        argument_list.append(dict(
            opts=("-M", "--masker"),
            action=Radio,
            type=str.lower,
            choices=PluginLoader.get_available_extractors("mask"),
            default="extended",
            group=_("process"),
            help=_("R|Masker to use."
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
                   "faces. The mask model has been trained by community members and will need "
                   "testing for further description. Profile faces may result in sub-par "
                   "performance.")))
        argument_list.append(dict(
            opts=("-p", "--processing"),
            action=Radio,
            type=str.lower,
            choices=("all", "missing", "output"),
            default="missing",
            group=_("process"),
            help=_("R|Whether to update all masks in the alignments files, only those faces "
                   "that do not already have a mask of the given `mask type` or just to output "
                   "the masks to the `output` location."
                   "\nL|all: Update the mask for all faces in the alignments file."
                   "\nL|missing: Create a mask for all faces in the alignments file where a mask "
                   "does not previously exist."
                   "\nL|output: Don't update the masks, just output them for review in the given "
                   "output folder.")))
        argument_list.append(dict(
            opts=("-o", "--output-folder"),
            action=DirFullPaths,
            dest="output",
            type=str,
            group=_("output"),
            help=_("Optional output location. If provided, a preview of the masks created will "
                   "be output in the given folder.")))
        argument_list.append(dict(
            opts=("-b", "--blur_kernel"),
            action=Slider,
            type=int,
            group=_("output"),
            min_max=(0, 9),
            default=3,
            rounding=1,
            help=_("Apply gaussian blur to the mask output. Has the effect of smoothing the "
                   "edges of the mask giving less of a hard edge. the size is in pixels. This "
                   "value should be odd, if an even number is passed in then it will be rounded "
                   "to the next odd number. NB: Only effects the output preview. Set to 0 for "
                   "off")))
        argument_list.append(dict(
            opts=("-t", "--threshold"),
            action=Slider,
            type=int,
            group=_("output"),
            min_max=(0, 50),
            default=4,
            rounding=1,
            help=_("Helps reduce 'blotchiness' on some masks by making light shades white "
                   "and dark shades black. Higher values will impact more of the mask. NB: "
                   "Only effects the output preview. Set to 0 for off")))
        argument_list.append(dict(
            opts=("-ot", "--output-type"),
            action=Radio,
            type=str.lower,
            choices=("combined", "masked", "mask"),
            default="combined",
            group=_("output"),
            help=_("R|How to format the output when processing is set to 'output'."
                   "\nL|combined: The image contains the face/frame, face mask and masked face."
                   "\nL|masked: Output the face/frame as rgba image with the face masked."
                   "\nL|mask: Only output the mask as a single channel image.")))
        argument_list.append(dict(
            opts=("-f", "--full-frame"),
            action="store_true",
            default=False,
            group=_("output"),
            help=_("R|Whether to output the whole frame or only the face box when using "
                   "output processing. Only has an effect when using frames as input.")))

        return argument_list
