#!/usr/bin/env python3
"""
A tool that allows for sorting and grouping images in different ways.
"""
from __future__ import annotations
import logging
import os
import sys
import typing as T

from argparse import Namespace
from shutil import copyfile, rmtree

from tqdm import tqdm

# faceswap imports
from lib.serializer import Serializer, get_serializer_from_filename
from lib.utils import handle_deprecated_cliopts

from .sort_methods import SortBlur, SortColor, SortFace, SortHistogram, SortMultiMethod
from .sort_methods_aligned import SortDistance, SortFaceCNN, SortPitch, SortSize, SortYaw, SortRoll

if T.TYPE_CHECKING:
    from .sort_methods import SortMethod

logger = logging.getLogger(__name__)


class Sort():
    """ Sorts folders of faces based on input criteria

    Wrapper for the sort process to run in either batch mode or single use mode

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the extraction process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing: %s (args: %s)", self.__class__.__name__, arguments)
        self._args = handle_deprecated_cliopts(arguments)
        self._input_locations = self._get_input_locations()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _get_input_locations(self) -> list[str]:
        """ Obtain the full path to input locations. Will be a list of locations if batch mode is
        selected, or a containing a single location if batch mode is not selected.

        Returns
        -------
        list:
            The list of input location paths
        """
        if not self._args.batch_mode:
            return [self._args.input_dir]

        retval = [os.path.join(self._args.input_dir, fname)
                  for fname in os.listdir(self._args.input_dir)
                  if os.path.isdir(os.path.join(self._args.input_dir, fname))]
        logger.debug("Input locations: %s", retval)
        return retval

    def _output_for_input(self, input_location: str) -> str:
        """ Obtain the path to an output folder for faces for a given input location.

        If not running in batch mode, then the user supplied output location will be returned,
        otherwise a sub-folder within the user supplied output location will be returned based on
        the input filename

        Parameters
        ----------
        input_location: str
            The full path to an input video or folder of images
        """
        if not self._args.batch_mode or self._args.output_dir is None:
            return self._args.output_dir

        retval = os.path.join(self._args.output_dir, os.path.basename(input_location))
        logger.debug("Returning output: '%s' for input: '%s'", retval, input_location)
        return retval

    def process(self) -> None:
        """ The entry point for triggering the Sort Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.info('Starting, this may take a while...')
        inputs = self._input_locations
        if self._args.batch_mode:
            logger.info("Batch mode selected processing: %s", self._input_locations)
        for job_no, location in enumerate(self._input_locations):
            if self._args.batch_mode:
                logger.info("Processing job %s of %s: '%s'", job_no + 1, len(inputs), location)
                arguments = Namespace(**self._args.__dict__)
                arguments.input_dir = location
                arguments.output_dir = self._output_for_input(location)
            else:
                arguments = self._args
            sort = _Sort(arguments)
            sort.process()


class _Sort():
    """ Sorts folders of faces based on input criteria """
    def __init__(self, arguments: Namespace) -> None:
        logger.debug("Initializing %s: arguments: %s", self.__class__.__name__, arguments)
        self._processes = {"blur": SortBlur,
                           "blur_fft": SortBlur,
                           "distance": SortDistance,
                           "yaw": SortYaw,
                           "pitch": SortPitch,
                           "roll": SortRoll,
                           "size": SortSize,
                           "face": SortFace,
                           "face_cnn": SortFaceCNN,
                           "face_cnn_dissim": SortFaceCNN,
                           "hist": SortHistogram,
                           "hist_dissim": SortHistogram,
                           "color_black": SortColor,
                           "color_gray": SortColor,
                           "color_luma": SortColor,
                           "color_green": SortColor,
                           "color_orange": SortColor}

        self._args = self._parse_arguments(arguments)
        self._changes: dict[str, str] = {}
        self.serializer: Serializer | None = None

        if arguments.log_changes:
            self.serializer = get_serializer_from_filename(arguments.log_file_path)

        self._sorter = self._get_sorter()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_output_folder(self, arguments):
        """ Set the output folder correctly if it has not been provided
        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments passed to the sort process

        Returns
        -------
        :class:`argparse.Namespace`
            The command line arguments with output folder correctly set
        """
        logger.debug("setting output folder: %s", arguments.output_dir)
        input_dir = arguments.input_dir
        output_dir = arguments.output_dir
        sort_method = arguments.sort_method
        group_method = arguments.group_method

        needs_rename = sort_method != "none" and group_method == "none"

        if needs_rename and arguments.keep_original and (not output_dir or
                                                         output_dir == input_dir):
            output_dir = os.path.join(input_dir, "sorted")
            logger.warning("No output folder selected, but files need renaming. "
                           "Outputting to: '%s'", output_dir)
        elif not output_dir:
            output_dir = input_dir
            logger.warning("No output folder selected, files will be sorted in place in: '%s'",
                           output_dir)

        arguments.output_dir = output_dir
        logger.debug("Set output folder: %s", arguments.output_dir)
        return arguments

    def _parse_arguments(self, arguments):
        """ Parse the arguments and update/format relevant choices

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments passed to the sort process

        Returns
        -------
        :class:`argparse.Namespace`
            The formatted command line arguments
        """
        logger.debug("Cleaning arguments: %s", arguments)
        if arguments.sort_method == "none" and arguments.group_method == "none":
            logger.error("Both sort-by and group-by are 'None'. Nothing to do.")
            sys.exit(1)

        # Prepare sort, group and final process method names
        arguments.sort_method = arguments.sort_method.lower().replace("-", "_")
        arguments.group_method = arguments.group_method.lower().replace("-", "_")

        arguments = self._set_output_folder(arguments)

        if arguments.log_changes and arguments.log_file_path == "sort_log.json":
            # Assign default sort_log.json value if user didn't specify one
            arguments.log_file_path = os.path.join(self._args.input_dir, 'sort_log.json')

        logger.debug("Cleaned arguments: %s", arguments)
        return arguments

    def _get_sorter(self) -> SortMethod:
        """ Obtain a sorter/grouper combo for the selected sort/group by options

        Returns
        -------
        :class:`SortMethod`
            The sorter or combined sorter for sorting and grouping based on user selections
        """
        sort_method = self._args.sort_method
        group_method = self._args.group_method

        sort_method = group_method if sort_method == "none" else sort_method
        sorter = self._processes[sort_method](self._args,
                                              is_group=self._args.sort_method == "none")

        if sort_method != "none" and group_method != "none" and group_method != sort_method:
            grouper = self._processes[group_method](self._args, is_group=True)
            retval = SortMultiMethod(self._args, sorter, grouper)
            logger.debug("Got sorter + grouper: %s (%s, %s)", retval, sorter, grouper)

        else:

            retval = sorter

        logger.debug("Final sorter: %s", retval)
        return retval

    def _write_to_log(self, changes):
        """ Write the changes to log file """
        logger.info("Writing sort log to: '%s'", self._args.log_file_path)
        self.serializer.save(self._args.log_file_path, changes)

    def process(self) -> None:
        """ Main processing function of the sort tool

        This method dynamically assigns the functions that will be used to run
        the core process of sorting, optionally grouping, renaming/moving into
        folders. After the functions are assigned they are executed.
        """
        if self._args.group_method != "none":
            # Check if non-dissimilarity sort method and group method are not the same
            self._output_groups()
        else:
            self._output_non_grouped()

        if self._args.log_changes:
            self._write_to_log(self._changes)

        logger.info("Done.")

    def _sort_file(self, source: str, destination: str) -> None:
        """ Copy or move a file based on whether 'keep original' has been selected and log changes
        if required.

        Parameters
        ----------
        source: str
            The full path to the source file that is being sorted
        destination: str
            The full path to where the source file should be moved/renamed
        """
        try:
            if self._args.keep_original:
                copyfile(source, destination)
            else:
                os.rename(source, destination)
        except FileNotFoundError as err:
            logger.error("Failed to sort '%s' to '%s'. Original error: %s",
                         source, destination, str(err))

        if self._args.log_changes:
            self._changes[source] = destination

    def _output_groups(self) -> None:
        """ Move the files to folders.

        Obtains the bins and original filenames from :attr:`_sorter` and outputs into appropriate
        bins in the output location
        """
        is_rename = self._args.sort_method != "none"

        logger.info("Creating %s group folders in '%s'.",
                    len(self._sorter.binned), self._args.output_dir)
        bin_names = [f"_{b}" for b in self._sorter.bin_names]
        if is_rename:
            bin_names = [f"{name}_by_{self._args.sort_method}" for name in bin_names]
        for name in bin_names:
            folder = os.path.join(self._args.output_dir, name)
            if os.path.exists(folder):
                rmtree(folder)
            os.makedirs(folder)

        description = f"{'Copying' if self._args.keep_original else 'Moving'} into groups"
        description += " and renaming" if is_rename else ""

        pbar = tqdm(range(len(self._sorter.sorted_filelist)),
                    desc=description,
                    file=sys.stdout,
                    leave=False)
        idx = 0
        for bin_id, bin_ in enumerate(self._sorter.binned):
            pbar.set_description(f"{description}: Bin {bin_id + 1} of {len(self._sorter.binned)}")
            output_path = os.path.join(self._args.output_dir, bin_names[bin_id])
            if not bin_:
                logger.debug("Removing empty bin: %s", output_path)
                os.rmdir(output_path)
            for source in bin_:
                basename = os.path.basename(source)
                dst_name = f"{idx:06d}_{basename}" if is_rename else basename
                dest = os.path.join(output_path, dst_name)
                self._sort_file(source, dest)
                idx += 1
                pbar.update(1)

    # Output methods
    def _output_non_grouped(self) -> None:
        """ Output non-grouped files.

        These are files which are sorted but not binned, so just the filename gets updated
        """
        output_dir = self._args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        description = f"{'Copying' if self._args.keep_original else 'Moving'} and renaming"
        for idx, source in enumerate(tqdm(self._sorter.sorted_filelist,
                                          desc=description,
                                          file=sys.stdout,
                                          leave=False)):
            dest = os.path.join(output_dir, f"{idx:06d}_{os.path.basename(source)}")

            self._sort_file(source, dest)
