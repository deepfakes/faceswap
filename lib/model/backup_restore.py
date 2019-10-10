#!/usr/bin/env python3

""" Functions for backing up, restoring and snapshotting models """

import logging
import os
from datetime import datetime
from shutil import copyfile, copytree, rmtree

from lib.serializer import get_serializer
from lib.utils import get_folder

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Backup():
    """ Holds information about model location and functions for backing up
        Restoring and Snapshotting models """
    def __init__(self, model_dir, model_name):
        logger.debug("Initializing %s: (model_dir: '%s', model_name: '%s')",
                     self.__class__.__name__, model_dir, model_name)
        self.model_dir = str(model_dir)
        self.model_name = model_name
        logger.debug("Initialized %s", self.__class__.__name__)

    def check_valid(self, filename, for_restore=False):
        """ Check if the passed in filename is valid for a backup operation """
        fullpath = os.path.join(self.model_dir, filename)
        if not filename.startswith(self.model_name):
            # Any filename that does not start with the model name are invalid
            # for all operations
            retval = False
        elif for_restore and filename.endswith(".bk"):
            # Only filenames ending in .bk are valid for restoring
            retval = True
        elif not for_restore and ((os.path.isfile(fullpath) and not filename.endswith(".bk")) or
                                  (os.path.isdir(fullpath) and
                                   filename == "{}_logs".format(self.model_name))):
            # Only filenames that do not end with .bk or folders that are the logs folder
            # are valid for backup
            retval = True
        else:
            retval = False
        logger.debug("'%s' valid for backup operation: %s", filename, retval)
        return retval

    @staticmethod
    def backup_model(fullpath):
        """ Backup Model File
            Fullpath should be the path to an h5.py file or a state.json file """
        backupfile = fullpath + ".bk"
        logger.verbose("Backing up: '%s' to '%s'", fullpath, backupfile)
        if os.path.exists(backupfile):
            os.remove(backupfile)
        if os.path.exists(fullpath):
            os.rename(fullpath, backupfile)

    def snapshot_models(self, iterations):
        """ Take a snapshot of the model at current state and back up """
        logger.info("Saving snapshot")
        snapshot_dir = "{}_snapshot_{}_iters".format(self.model_dir, iterations)

        if os.path.isdir(snapshot_dir):
            logger.debug("Removing previously existing snapshot folder: '%s'", snapshot_dir)
            rmtree(snapshot_dir)

        dst = str(get_folder(snapshot_dir))
        for filename in os.listdir(self.model_dir):
            if not self.check_valid(filename, for_restore=False):
                logger.debug("Not snapshotting file: '%s'", filename)
                continue
            srcfile = os.path.join(self.model_dir, filename)
            dstfile = os.path.join(dst, filename)
            copyfunc = copytree if os.path.isdir(srcfile) else copyfile
            logger.debug("Saving snapshot: '%s' > '%s'", srcfile, dstfile)
            copyfunc(srcfile, dstfile)
        logger.info("Saved snapshot")

    def restore(self):
        """ Restores a model from backup.
            This will place all existing models/logs into a folder named:
                - "<model_name>_archived_<timestamp>"
            Copy all .bk files to replace original files
            Remove logs from after the restore session_id from the logs folder """
        archive_dir = self.move_archived()
        self.restore_files()
        self.restore_logs(archive_dir)

    def move_archived(self):
        """ Move archived files to archived folder and return archived folder name """
        logger.info("Archiving existing model files...")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(self.model_dir, "{}_archived_{}".format(self.model_name, now))
        os.mkdir(archive_dir)
        for filename in os.listdir(self.model_dir):
            if not self.check_valid(filename, for_restore=False):
                logger.debug("Not moving file to archived: '%s'", filename)
                continue
            logger.verbose("Moving '%s' to archived model folder: '%s'", filename, archive_dir)
            src = os.path.join(self.model_dir, filename)
            dst = os.path.join(archive_dir, filename)
            os.rename(src, dst)
        logger.verbose("Archived existing model files")
        return archive_dir

    def restore_files(self):
        """ Restore files from .bk """
        logger.info("Restoring models from backup...")
        for filename in os.listdir(self.model_dir):
            if not self.check_valid(filename, for_restore=True):
                logger.debug("Not restoring file: '%s'", filename)
                continue
            dstfile = os.path.splitext(filename)[0]
            logger.verbose("Restoring '%s' to '%s'", filename, dstfile)
            src = os.path.join(self.model_dir, filename)
            dst = os.path.join(self.model_dir, dstfile)
            copyfile(src, dst)
        logger.verbose("Restored models from backup")

    def restore_logs(self, archive_dir):
        """ Restore the log files since before archive """
        logger.info("Restoring Logs...")
        session_names = self.get_session_names()
        log_dirs = self.get_log_dirs(archive_dir, session_names)
        for log_dir in log_dirs:
            src = os.path.join(archive_dir, log_dir)
            dst = os.path.join(self.model_dir, log_dir)
            logger.verbose("Restoring logfile: %s", dst)
            copytree(src, dst)
        logger.verbose("Restored Logs")

    def get_session_names(self):
        """ Get the existing session names from state file """
        serializer = get_serializer("json")
        state_file = os.path.join(self.model_dir,
                                  "{}_state.{}".format(self.model_name, serializer.file_extension))
        state = serializer.load(state_file)
        session_names = ["session_{}".format(key)
                         for key in state["sessions"].keys()]
        logger.debug("Session to restore: %s", session_names)
        return session_names

    def get_log_dirs(self, archive_dir, session_names):
        """ Get the session logdir paths in the archive folder """
        archive_logs = os.path.join(archive_dir, "{}_logs".format(self.model_name))
        paths = [os.path.join(dirpath.replace(archive_dir, "")[1:], folder)
                 for dirpath, dirnames, _ in os.walk(archive_logs)
                 for folder in dirnames
                 if folder in session_names]
        logger.debug("log folders to restore: %s", paths)
        return paths
