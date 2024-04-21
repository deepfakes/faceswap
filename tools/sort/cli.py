#!/usr/bin/env python3
""" Command Line Arguments for tools """
import argparse
import gettext

from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirFullPaths, SaveFileFullPaths, Radio, Slider


# LOCALES
_LANG = gettext.translation("tools.sort.cli", localedir="locales", fallback=True)
_ = _LANG.gettext


_HELPTEXT = _("This command lets you sort images using various methods.")
_SORT_METHODS = (
    "none", "blur", "blur-fft", "distance", "face", "face-cnn", "face-cnn-dissim",
    "yaw", "pitch", "roll", "hist", "hist-dissim", "color-black", "color-gray", "color-luma",
    "color-green", "color-orange", "size")

_GPTHRESHOLD = _(" Adjust the '-t' ('--threshold') parameter to control the strength of grouping.")
_GPCOLOR = _(" Adjust the '-b' ('--bins') parameter to control the number of bins for grouping. "
             "Each image is allocated to a bin by the percentage of color pixels that appear in "
             "the image.")
_GPDEGREES = _(" Adjust the '-b' ('--bins') parameter to control the number of bins for grouping. "
               "Each image is allocated to a bin by the number of degrees the face is orientated "
               "from center.")
_GPLINEAR = _(" Adjust the '-b' ('--bins') parameter to control the number of bins for grouping. "
              "The minimum and maximum values are taken for the chosen sort metric. The bins "
              "are then populated with the results from the group sorting.")
_METHOD_TEXT = {
    "blur": _("faces by blurriness."),
    "blur-fft": _("faces by fft filtered blurriness."),
    "distance": _("faces by the estimated distance of the alignments from an 'average' face. This "
                  "can be useful for eliminating misaligned faces. Sorts from most like an "
                  "average face to least like an average face."),
    "face": _("faces using VGG Face2 by face similarity. This uses a pairwise clustering "
              "algorithm to check the distances between 512 features on every face in your set "
              "and order them appropriately."),
    "face-cnn": _("faces by their landmarks."),
    "face-cnn-dissim": _("Like 'face-cnn' but sorts by dissimilarity."),
    "yaw": _("faces by Yaw (rotation left to right)."),
    "pitch": _("faces by Pitch (rotation up and down)."),
    "roll": _("faces by Roll (rotation). Aligned faces should have a roll value close to zero. "
              "The further the Roll value from zero the higher liklihood the face is misaligned."),
    "hist": _("faces by their color histogram."),
    "hist-dissim": _("Like 'hist' but sorts by dissimilarity."),
    "color-gray": _("images by the average intensity of the converted grayscale color channel."),
    "color-black": _("images by their number of black pixels. Useful when faces are near borders "
                     "and a large part of the image is black."),
    "color-luma": _("images by the average intensity of the converted Y color channel. Bright "
                    "lighting and oversaturated images will be ranked first."),
    "color-green": _("images by the average intensity of the converted Cg color channel. Green "
                     "images will be ranked first and red images will be last."),
    "color-orange": _("images by the average intensity of the converted Co color channel. Orange "
                      "images will be ranked first and blue images will be last."),
    "size": _("images by their size in the original frame. Faces further from the camera and from "
              "lower resolution sources will be sorted first, whilst faces closer to the camera "
              "and from higher resolution sources will be sorted last.")}

_BIN_TYPES = [
    (("face", "face-cnn", "face-cnn-dissim", "hist", "hist-dissim"), _GPTHRESHOLD),
    (("color-black", "color-gray", "color-luma", "color-green", "color-orange"), _GPCOLOR),
    (("yaw", "pitch", "roll"), _GPDEGREES),
    (("blur", "blur-fft", "distance", "size"), _GPLINEAR)]
_SORT_HELP = ""
_GROUP_HELP = ""

for method in sorted(_METHOD_TEXT):
    _SORT_HELP += f"\nL|{method}: {_('Sort')} {_METHOD_TEXT[method]}"
    _GROUP_HELP += (f"\nL|{method}: {_('Group')} {_METHOD_TEXT[method]} "
                    f"{next((x[1] for x in _BIN_TYPES if method in x[0]), '')}")


class SortArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for sort tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return _("Sort faces using a number of different techniques")

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = []
        argument_list.append({
            "opts": ('-i', '--input'),
            "action": DirFullPaths,
            "dest": "input_dir",
            "group": _("data"),
            "help": _("Input directory of aligned faces."),
            "required": True})
        argument_list.append({
            "opts": ('-o', '--output'),
            "action": DirFullPaths,
            "dest": "output_dir",
            "group": _("data"),
            "help": _(
                "Output directory for sorted aligned faces. If not provided and 'keep' is "
                "selected then a new folder called 'sorted' will be created within the input "
                "folder to house the output. If not provided and 'keep' is not selected then the "
                "images will be sorted in-place, overwriting the original contents of the "
                "'input_dir'")})
        argument_list.append({
            "opts": ("-B", "--batch-mode"),
            "action": "store_true",
            "dest": "batch_mode",
            "default": False,
            "group": _("data"),
            "help": _(
                "R|If selected then the input_dir should be a parent folder containing multiple "
                "folders of faces you wish to sort. The faces will be output to separate sub-"
                "folders in the output_dir")})
        argument_list.append({
            "opts": ('-s', '--sort-by'),
            "action": Radio,
            "type": str,
            "choices": _SORT_METHODS,
            "dest": 'sort_method',
            "group": _("sort settings"),
            "default": "face",
            "help": _(
                "R|Choose how images are sorted. Selecting a sort method gives the images a new "
                "filename based on the order the image appears within the given method."
                "\nL|'none': Don't sort the images. When a 'group-by' method is selected, "
                "selecting 'none' means that the files will be moved/copied into their respective "
                "bins, but the files will keep their original filenames. Selecting 'none' for "
                "both 'sort-by' and 'group-by' will do nothing" + _SORT_HELP + "\nDefault: face")})
        argument_list.append({
            "opts": ('-g', '--group-by'),
            "action": Radio,
            "type": str,
            "choices": _SORT_METHODS,
            "dest": 'group_method',
            "group": _("group settings"),
            "default": "none",
            "help": _(
                "R|Selecting a group by method will move/copy files into numbered bins based on "
                "the selected method."
                "\nL|'none': Don't bin the images. Folders will be sorted by the selected 'sort-"
                "by' but will not be binned, instead they will be sorted into a single folder. "
                "Selecting 'none' for both 'sort-by' and 'group-by' will do nothing" +
                _GROUP_HELP + "\nDefault: none")})
        argument_list.append({
            "opts": ('-k', '--keep'),
            "action": 'store_true',
            "dest": 'keep_original',
            "default": False,
            "group": _("data"),
            "help": _(
                "Whether to keep the original files in their original location. Choosing a 'sort-"
                "by' method means that the files have to be renamed. Selecting 'keep' means that "
                "the original files will be kept, and the renamed files will be created in the "
                "specified output folder. Unselecting keep means that the original files will be "
                "moved and renamed based on the selected sort/group criteria.")})
        argument_list.append({
            "opts": ('-t', '--threshold'),
            "action": Slider,
            "min_max": (-1.0, 10.0),
            "rounding": 2,
            "type": float,
            "dest": 'threshold',
            "group": _("group settings"),
            "default": -1.0,
            "help": _(
                "R|Float value. Minimum threshold to use for grouping comparison with 'face-cnn' "
                "'hist' and 'face' methods."
                "\nThe lower the value the more discriminating the grouping is. Leaving -1.0 will "
                "allow Faceswap to choose the default value."
                "\nL|For 'face-cnn' 7.2 should be enough, with 4 being very discriminating. "
                "\nL|For 'hist' 0.3 should be enough, with 0.2 being very discriminating. "
                "\nL|For 'face' between 0.1 (more bins) to 0.5 (fewer bins) should be about right."
                "\nBe careful setting a value that's too extrene in a directory with many images, "
                "as this could result in a lot of folders being created. Defaults: face-cnn 7.2, "
                "hist 0.3, face 0.25")})
        argument_list.append({
            "opts": ('-b', '--bins'),
            "action": Slider,
            "min_max": (1, 100),
            "rounding": 1,
            "type": int,
            "dest": 'num_bins',
            "group": _("group settings"),
            "default": 5,
            "help": _(
                "R|Integer value. Used to control the number of bins created for grouping by: any "
                "'blur' methods, 'color' methods or 'face metric' methods ('distance', 'size') "
                "and 'orientation; methods ('yaw', 'pitch'). For any other grouping "
                "methods see the '-t' ('--threshold') option."
                "\nL|For 'face metric' methods the bins are filled, according the the "
                "distribution of faces between the minimum and maximum chosen metric."
                "\nL|For 'color' methods the number of bins represents the divider of the "
                "percentage of colored pixels. Eg. For a bin number of '5': The first folder will "
                "have the faces with 0%% to 20%% colored pixels, second 21%% to 40%%, etc. Any "
                "empty bins will be deleted, so you may end up with fewer bins than selected."
                "\nL|For 'blur' methods folder 0 will be the least blurry, while the last folder "
                "will be the blurriest."
                "\nL|For 'orientation' methods the number of bins is dictated by how much 180 "
                "degrees is divided. Eg. If 18 is selected, then each folder will be a 10 degree "
                "increment. Folder 0 will contain faces looking the most to the left/down whereas "
                "the last folder will contain the faces looking the most to the right/up. NB: "
                "Some bins may be empty if faces do not fit the criteria. \nDefault value: 5")})
        argument_list.append({
            "opts": ('-l', '--log-changes'),
            "action": 'store_true',
            "group": _("settings"),
            "default": False,
            "help": _(
                "Logs file renaming changes if grouping by renaming, or it logs the file copying/"
                "movement if grouping by folders. If no log file is specified  with '--log-file', "
                "then a 'sort_log.json' file will be created in the input directory.")})
        argument_list.append({
            "opts": ('-f', '--log-file'),
            "action": SaveFileFullPaths,
            "filetypes": "alignments",
            "group": _("settings"),
            "dest": 'log_file_path',
            "default": 'sort_log.json',
            "help": _(
                "Specify a log file to use for saving the renaming or grouping information. If "
                "specified extension isn't 'json' or 'yaml', then json will be used as the "
                "serializer, with the supplied filename. Default: sort_log.json")})
        # Deprecated multi-character switches
        argument_list.append({
            "opts": ("-lf", ),
            "type": str,
            "dest": "depr_log-file_lf_f",
            "help": argparse.SUPPRESS})
        return argument_list
