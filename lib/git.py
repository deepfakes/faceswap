#!/usr/bin python3
""" Handles command line calls to git """
import logging
import os
import sys

from subprocess import PIPE, Popen

logger = logging.getLogger(__name__)


class Git():
    """ Handles calls to github """
    def __init__(self) -> None:
        logger.debug("Initializing: %s", self.__class__.__name__)
        self._working_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
        self._available = self._check_available()
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _from_git(self, command: str) -> tuple[bool, list[str]]:
        """ Execute a git command

        Parameters
        ----------
        command : str
            The command to send to git

        Returns
        -------
        success: bool
            ``True`` if the command succesfully executed otherwise ``False``
        list[str]
            The output lines from stdout if there was no error, otherwise from stderr
        """
        logger.debug("command: '%s'", command)
        cmd = f"git {command}"
        with Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE, cwd=self._working_dir) as proc:
            stdout, stderr = proc.communicate()
        retcode = proc.returncode
        success = retcode == 0
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        if not lines:
            lines = stderr.decode("utf-8", errors="replace").splitlines()
        logger.debug("command: '%s', returncode: %s, success: %s, lines: %s",
                     cmd, retcode, success, lines)
        return success, lines

    def _check_available(self) -> bool:
        """ Check if git is available. Does a call to git status. If the process errors due to
        folder ownership, attempts to add the folder to github safe folders list and tries
        again

        Returns
        -------
        bool
            ``True`` if git is available otherwise ``False``

        """
        success, msg = self._from_git("status")
        if success:
            return True
        config = next((line.strip() for line in msg if "add safe.directory" in line), None)
        if not config:
            return False
        success, _ = self._from_git(config.split("git ", 1)[-1])
        return True

    @property
    def status(self) -> list[str]:
        """ Obtain the output of git status for tracked files only """
        if not self._available:
            return []
        success, status = self._from_git("status -uno")
        if not success or not status:
            return []
        return status

    @property
    def branch(self) -> str:
        """ str: The git branch that is currently being used to execute Faceswap. """
        status = next((line.strip() for line in self.status if "On branch" in line), "Not Found")
        return status.replace("On branch ", "")

    @property
    def branches(self) -> list[str]:
        """ list[str]: List of all available branches. """
        if not self._available:
            return []
        success, branches = self._from_git("branch -a")
        if not success or not branches:
            return []
        return branches

    def update_remote(self) -> bool:
        """ Update all branches to track remote

        Returns
        -------
        bool
            ``True`` if update was succesful otherwise ``False``
        """
        if not self._available:
            return False
        return self._from_git("remote update")[0]

    def pull(self) -> bool:
        """ Pull the current branch

        Returns
        -------
        bool
            ``True`` if pull is successful otherwise ``False``
        """
        if not self._available:
            return False
        return self._from_git("pull")[0]

    def checkout(self, branch: str) -> bool:
        """ Checkout the requested branch

        Parameters
        ----------
        branch : str
            The branch to checkout

        Returns
        -------
        bool
            ``True`` if the branch was succesfully checkout out otherwise ``False``
        """
        if not self._available:
            return False
        return self._from_git(f"checkout {branch}")[0]

    def get_commits(self, count: int) -> list[str]:
        """ Obtain the last commits to the repo

        Parameters
        ----------
        count : int
            The last number of commits to obtain

        Returns
        -------
        list[str]
            list of commits, or empty list if none found
        """
        if not self._available:
            return []
        success, commits = self._from_git(f"log --pretty=oneline --abbrev-commit -n {count}")
        if not success or not commits:
            return []
        return commits


git = Git()
""" :class:`Git`: Handles calls to github """
