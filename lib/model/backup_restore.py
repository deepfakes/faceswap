#!/usr/bin/env python3

""" Functions for backing up, restoring and creating model snapshots. """

import logging
import os
from datetime import datetime
from shutil import copyfile, copytree, rmtree

from lib.serializer import get_serializer
from lib.utils import get_folder

logger = logging.getLogger(__name__)


class Backup():
    """ Performs the back up of models at each save iteration, and the restoring of models from
    their back up location.

    Parameters
    ----------
    model_dir: str
        The folder that contains the model to be backed up
    model_name: str
        The name of the model that is to be backed up
    """
    def __init__(self, model_dir, model_name):
        logger.debug("Initializing %s: (model_dir: '%s', model_name: '%s')",
                     self.__class__.__name__, model_dir, model_name)
        self.model_dir = str(model_dir)
        self.model_name = model_name
        logger.debug("Initialized %s", self.__class__.__name__)

    def _check_valid(self, filename, for_restore=False):
        """ Check if the passed in filename is valid for a backup or restore operation.

        Parameters
        ----------
        filename: str
            The filename that is to be checked for backup or restore
        for_restore: bool, optional
            ``True`` if the checks are to be performed for restoring a model, ``False`` if the
            checks are to be performed for backing up a model. Default: ``False``

        Returns
        -------
        bool
            ``True`` if the given file is valid for a backup/restore operation otherwise ``False``
        """
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
    def backup_model(full_path):
        """ Backup a model file.

        The backed up file is saved with the original filename in the original location with `.bk`
        appended to the end of the name.

        Parameters
        ----------
        full_path: str
            The full path to a `.h5` model file or a `.json` state file
        """
        backupfile = full_path + ".bk"
        if os.path.exists(backupfile):
            os.remove(backupfile)
        if os.path.exists(full_path):
            logger.verbose("Backing up: '%s' to '%s'", full_path, backupfile)
            os.rename(full_path, backupfile)

    def snapshot_models(self, iterations):
        """ Take a snapshot of the model at the current state and back it up.

        The snapshot is a copy of the model folder located in the same root location
        as the original model file, with the number of iterations appended to the end
        of the folder name.

        Parameters
        ----------
        iterations: int
            The number of iterations that the model has trained when performing the snapshot.
        """
        print("")  # New line so log message doesn't append to last loss output
        logger.verbose("Saving snapshot")
        snapshot_dir = "{}_snapshot_{}_iters".format(self.model_dir, iterations)

        if os.path.isdir(snapshot_dir):
            logger.debug("Removing previously existing snapshot folder: '%s'", snapshot_dir)
            rmtree(snapshot_dir)

        dst = get_folder(snapshot_dir)
        for filename in os.listdir(self.model_dir):
            if not self._check_valid(filename, for_restore=False):
                logger.debug("Not snapshotting file: '%s'", filename)
                continue
            srcfile = os.path.join(self.model_dir, filename)
            dstfile = os.path.join(dst, filename)
            copyfunc = copytree if os.path.isdir(srcfile) else copyfile
            logger.debug("Saving snapshot: '%s' > '%s'", srcfile, dstfile)
            copyfunc(srcfile, dstfile)
        logger.info("Saved snapshot (%s iterations)", iterations)

    def restore(self):
        """ Restores a model from backup.

        The original model files are migrated into a folder within the original model folder
        named `<model_name>_archived_<timestamp>`. The `.bk` backup files are then moved to
        the location of the previously existing model files. Logs that were generated after the
        the last backup was taken are removed. """
        archive_dir = self._move_archived()
        self._restore_files()
        self._restore_logs(archive_dir)

    def _move_archived(self):
        """ Move archived files to the archived folder.

        Returns
        -------
        str
            The name of the generated archive folder
        """
        logger.info("Archiving existing model files...")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(self.model_dir, "{}_archived_{}".format(self.model_name, now))
        os.mkdir(archive_dir)
        for filename in os.listdir(self.model_dir):
            if not self._check_valid(filename, for_restore=False):
                logger.debug("Not moving file to archived: '%s'", filename)
                continue
            logger.verbose("Moving '%s' to archived model folder: '%s'", filename, archive_dir)
            src = os.path.join(self.model_dir, filename)
            dst = os.path.join(archive_dir, filename)
            os.rename(src, dst)
        logger.verbose("Archived existing model files")
        return archive_dir

    def _restore_files(self):
        """ Restore files from .bk """
        logger.info("Restoring models from backup...")
        for filename in os.listdir(self.model_dir):
            if not self._check_valid(filename, for_restore=True):
                logger.debug("Not restoring file: '%s'", filename)
                continue
            dstfile = os.path.splitext(filename)[0]
            logger.verbose("Restoring '%s' to '%s'", filename, dstfile)
            src = os.path.join(self.model_dir, filename)
            dst = os.path.join(self.model_dir, dstfile)
            copyfile(src, dst)
        logger.verbose("Restored models from backup")

    def _restore_logs(self, archive_dir):
        """ Restores the log files up to and including the last backup.

        Parameters
        ----------
        archive_dir: str
            The full path to the model's archive folder
        """
        logger.info("Restoring Logs...")
        session_names = self._get_session_names()
        log_dirs = self._get_log_dirs(archive_dir, session_names)
        for log_dir in log_dirs:
            src = os.path.join(archive_dir, log_dir)
            dst = os.path.join(self.model_dir, log_dir)
            logger.verbose("Restoring logfile: %s", dst)
            copytree(src, dst)
        logger.verbose("Restored Logs")

    def _get_session_names(self):
        """ Get the existing session names from a state file. """
        serializer = get_serializer("json")
        state_file = os.path.join(self.model_dir,
                                  "{}_state.{}".format(self.model_name, serializer.file_extension))
        state = serializer.load(state_file)
        session_names = ["session_{}".format(key)
                         for key in state["sessions"].keys()]
        logger.debug("Session to restore: %s", session_names)
        return session_names

    def _get_log_dirs(self, archive_dir, session_names):
        """ Get the session log directory paths in the archive folder.

        Parameters
        ----------
        archive_dir: str
            The full path to the model's archive folder
        session_names: list
            The name of the training sessions that exist for the model

        Returns
        -------
        list
            The full paths to the log folders
        """
        archive_logs = os.path.join(archive_dir, "{}_logs".format(self.model_name))
        paths = [os.path.join(dirpath.replace(archive_dir, "")[1:], folder)
                 for dirpath, dirnames, _ in os.walk(archive_logs)
                 for folder in dirnames
                 if folder in session_names]
        logger.debug("log folders to restore: %s", paths)
        return paths
