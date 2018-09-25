#!/usr/bin/env python3
""" Tools for manipulating the alignments seralized file """

# TODO merge alignments
from lib.utils import set_system_verbosity
from .lib_alignments import (AlignmentData, Check, Draw, Extract, Manual,
                             Reformat, RemoveAlignments, Rotate, Sort)


class Alignments():
    """ Perform tasks relating to alignments file """
    def __init__(self, arguments):
        self.args = arguments
        self.set_verbosity(arguments.verbose)

        dest_format = self.get_dest_format()
        self.alignments = AlignmentData(self.args.alignments_file,
                                        dest_format,
                                        self.args.verbose)

    @staticmethod
    def set_verbosity(verbose):
        """ Set the system output verbosity """
        lvl = '0' if verbose else '2'
        set_system_verbosity(lvl)

    def get_dest_format(self):
        """ Set the destination format for Alignments """
        dest_format = None
        if (hasattr(self.args, 'alignment_format')
                and self.args.alignment_format):
            dest_format = self.args.alignment_format
        return dest_format

    def process(self):
        """ Main processing function of the Align tool """
        if self.args.job.startswith("extract"):
            job = Extract
        elif self.args.job.startswith("remove-"):
            job = RemoveAlignments
        elif self.args.job.startswith("sort-"):
            job = Sort
        elif self.args.job in("missing-alignments", "missing-frames",
                              "multi-faces", "leftover-faces",
                              "no-faces"):
            job = Check
        else:
            job = globals()[self.args.job.title()]
        job = job(self.alignments, self.args)
        job.process()
