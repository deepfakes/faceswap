#!/usr/bin/env python3
""" Command Line Arguments for tools """
from lib.cli.args import FaceSwapArgs
from lib.cli.actions import DirFullPaths, SaveFileFullPaths, Radio, Slider

_HELPTEXT = "This command lets you sort images using various methods."


class SortArgs(FaceSwapArgs):
    """ Class to parse the command line arguments for sort tool """

    @staticmethod
    def get_info():
        """ Return command information """
        return "Sort faces using a number of different techniques"

    @staticmethod
    def get_argument_list():
        """ Put the arguments in a list so that they are accessible from both argparse and gui """
        argument_list = list()
        argument_list.append({"opts": ('-i', '--input'),
                              "action": DirFullPaths,
                              "dest": "input_dir",
                              "group": "data",
                              "help": "Input directory of aligned faces.",
                              "required": True})

        argument_list.append({"opts": ('-o', '--output'),
                              "action": DirFullPaths,
                              "dest": "output_dir",
                              "group": "data",
                              "help": "Output directory for sorted aligned "
                                      "faces."})

        argument_list.append({"opts": ('-s', '--sort-by'),
                              "action": Radio,
                              "type": str,
                              "choices": ("blur", "face", "face-cnn", "face-cnn-dissim",
                                          "face-yaw", "hist", "hist-dissim", "color-gray",
                                          "color-luma", "color-green", "color-orange"),
                              "dest": 'sort_method',
                              "group": "sort settings",
                              "default": "face",
                              "help": "R|Sort by method. Choose how images are sorted. "
                                      "\nL|'blur': Sort faces by blurriness."
                                      "\nL|'face': Use VGG Face to sort by face similarity. This "
                                      "uses a pairwise clustering algorithm to check the "
                                      "distances between 4096 features on every face in your set "
                                      "and order them appropriately. WARNING: On very large "
                                      "datasets it is possible to run out of memory performing "
                                      "this calculation."
                                      "\nL|'face-cnn': Sort faces by their landmarks. You can "
                                      "adjust the threshold with the '-t' (--ref_threshold) "
                                      "option."
                                      "\nL|'face-cnn-dissim': Like 'face-cnn' but sorts by "
                                      "dissimilarity."
                                      "\nL|'face-yaw': Sort faces by Yaw (rotation left to right)."
                                      "\nL|'hist': Sort faces by their color histogram. You can "
                                      "adjust the threshold with the '-t' (--ref_threshold) "
                                      "option."
                                      "\nL|'hist-dissim': Like 'hist' but sorts by dissimilarity."
                                      "\nL|'color-gray': Sort images by the average intensity of "
                                      "the converted grayscale color channel."
                                      "\nL|'color-luma': Sort images by the average intensity of "
                                      "the converted Y color channel. Bright lighting and "
                                      "oversaturated images will be ranked first."
                                      "\nL|'color-green': Sort images by the average intensity of "
                                      "the converted Cg color channel. Green images will be "
                                      "ranked first and red images will be last."
                                      "\nL|'color-orange': Sort images by the average intensity "
                                      "of the converted Co color channel. Orange images will be "
                                      "ranked first and blue images will be last."
                                      "\nDefault: hist"})
        argument_list.append({"opts": ('-k', '--keep'),
                              "action": 'store_true',
                              "dest": 'keep_original',
                              "default": False,
                              "group": "output",
                              "help": "Keeps the original files in the input "
                                      "directory. Be careful when using this "
                                      "with rename grouping and no specified "
                                      "output directory as this would keep "
                                      "the original and renamed files in the "
                                      "same directory."})
        argument_list.append({"opts": ('-t', '--ref_threshold'),
                              "action": Slider,
                              "min_max": (-1.0, 10.0),
                              "rounding": 2,
                              "type": float,
                              "dest": 'min_threshold',
                              "group": "sort settings",
                              "default": -1.0,
                              "help": "Float value. "
                                      "Minimum threshold to use for grouping comparison with "
                                      "'face-cnn' and 'hist' methods. The lower the value the "
                                      "more discriminating the grouping is. Leaving -1.0 will "
                                      "allow the program set the default value automatically. "
                                      "For face-cnn 7.2 should be enough, with 4 being very "
                                      "discriminating. For hist 0.3 should be enough, with 0.2 "
                                      "being very discriminating. Be careful setting a value "
                                      "that's too low in a directory with many images, as this "
                                      "could result in a lot of directories being created. "
                                      "Defaults: face-cnn 7.2, hist 0.3"})

        argument_list.append({"opts": ('-fp', '--final-process'),
                              "action": Radio,
                              "type": str,
                              "choices": ("folders", "rename"),
                              "dest": 'final_process',
                              "default": "rename",
                              "group": "output",
                              "help": "R|Default: rename."
                                      "\nL|'folders': files are sorted using "
                                      "the -s/--sort-by method, then they "
                                      "are organized into folders using "
                                      "the -g/--group-by grouping method."
                                      "\nL|'rename': files are sorted using "
                                      "the -s/--sort-by then they are "
                                      "renamed."})

        argument_list.append({"opts": ('-g', '--group-by'),
                              "action": Radio,
                              "type": str,
                              "choices": ("blur", "face-cnn", "face-yaw", "hist"),
                              "dest": 'group_method',
                              "group": "output",
                              "default": "hist",
                              "help": "Group by method. "
                                      "When -fp/--final-processing by "
                                      "folders choose the how the images are "
                                      "grouped after sorting. "
                                      "Default: hist"})

        argument_list.append({"opts": ('-b', '--bins'),
                              "action": Slider,
                              "min_max": (1, 100),
                              "rounding": 1,
                              "type": int,
                              "dest": 'num_bins',
                              "group": "output",
                              "default": 5,
                              "help": "Integer value. "
                                      "Number of folders that will be used "
                                      "to group by blur and face-yaw. "
                                      "For blur folder 0 will be the least "
                                      "blurry, while the last folder will be "
                                      "the blurriest. "
                                      "For face-yaw the number of bins is by "
                                      "how much 180 degrees is divided. So "
                                      "if you use 18, then each folder will "
                                      "be a 10 degree increment. Folder 0 "
                                      "will contain faces looking the most "
                                      "to the left whereas the last folder "
                                      "will contain the faces looking the "
                                      "most to the right. "
                                      "If the number of images doesn't "
                                      "divide evenly into the number of "
                                      "bins, the remaining images get put in "
                                      "the last bin."
                                      "Default value: 5"})

        argument_list.append({"opts": ("-be", "--backend"),
                              "action": Radio,
                              "type": str.upper,
                              "choices": ("CPU", "GPU"),
                              "default": "GPU",
                              "group": "settings",
                              "help": "Backend to use for VGG Face inference."
                                      "Only used for sort by 'face'."})

        argument_list.append({"opts": ('-l', '--log-changes'),
                              "action": 'store_true',
                              "group": "settings",
                              "default": False,
                              "help": "Logs file renaming changes if "
                                      "grouping by renaming, or it logs the "
                                      "file copying/movement if grouping by "
                                      "folders. If no log file is specified "
                                      "with '--log-file', then a "
                                      "'sort_log.json' file will be created "
                                      "in the input directory."})

        argument_list.append({"opts": ('-lf', '--log-file'),
                              "action": SaveFileFullPaths,
                              "filetypes": "alignments",
                              "group": "settings",
                              "dest": 'log_file_path',
                              "default": 'sort_log.json',
                              "help": "Specify a log file to use for saving "
                                      "the renaming or grouping information. "
                                      "If specified extension isn't 'json' "
                                      "or 'yaml', then json will be used as "
                                      "the serializer, with the supplied "
                                      "filename. "
                                      "Default: sort_log.json"})

        return argument_list
