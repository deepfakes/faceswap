#!/usr/bin/env python3
""" Installs any required third party libs for faceswap.py

    Checks for installed Conda / Pip packages and updates accordingly
"""

import locale
import os
import re
import sys
import ctypes

from subprocess import CalledProcessError, run, PIPE, Popen

_LOGGER = None


class Environment():
    """ Hold information about the running environment """
    def __init__(self):
        self.is_conda = "conda" in sys.version.lower()
        self.encoding = locale.getpreferredencoding()
        self.is_admin = self.get_admin_status()
        self.is_virtualenv = self.get_virtualenv
        required_packages = self.get_required_packages()
        self.installed_packages = self.get_installed_packages()
        self.get_installed_conda_packages()
        self.packages_to_install = self.get_packages_to_install(required_packages)

    @staticmethod
    def get_admin_status():
        """ Check whether user is admin """
        try:
            retval = os.getuid() == 0
        except AttributeError:
            retval = ctypes.windll.shell32.IsUserAnAdmin() != 0
        return retval

    def get_virtualenv(self):
        """ Check whether this is a virtual environment """
        if not self.is_conda:
            retval = (hasattr(sys, "real_prefix") or
                      (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix))
        else:
            prefix = os.path.dirname(sys.prefix)
            retval = (os.path.basename(prefix) == "envs")
        return retval

    @staticmethod
    def get_required_packages():
        """ Load requirements list """
        packages = list()
        pypath = os.path.dirname(os.path.realpath(__file__))
        requirements_file = os.path.join(pypath, "requirements.txt")
        with open(requirements_file) as req:
            for package in req.readlines():
                package = package.strip()
                if package and (not package.startswith("#")):
                    packages.append(package)
        return packages

    def get_installed_packages(self):
        """ Get currently installed packages """
        installed_packages = dict()
        chk = Popen("\"{}\" -m pip freeze".format(sys.executable),
                    shell=True, stdout=PIPE)
        installed = chk.communicate()[0].decode(self.encoding).splitlines()

        for pkg in installed:
            if "==" not in pkg:
                continue
            item = pkg.split("==")
            installed_packages[item[0]] = item[1]
        return installed_packages

    def get_installed_conda_packages(self):
        """ Get currently installed conda packages """
        if not self.is_conda:
            return
        chk = os.popen("conda list").read()
        installed = [re.sub(" +", " ", line.strip())
                     for line in chk.splitlines() if not line.startswith("#")]
        for pkg in installed:
            item = pkg.split(" ")
            self.installed_packages[item[0]] = item[1]

    def get_packages_to_install(self, required_packages):
        """ Get packages which need installing, upgrading or downgrading """
        to_install = list()
        for pkg in required_packages:
            pkg = self.check_os_requirement(pkg)
            if pkg is None:
                continue
            key = pkg.split("==")[0]
            if key not in self.installed_packages:
                to_install.append(pkg)
            else:
                if (len(pkg.split("==")) > 1 and
                        pkg.split("==")[1] != self.installed_packages.get(key)):
                    to_install.append(pkg)
        return to_install

    @staticmethod
    def check_os_requirement(package):
        """ Check whether this package is required for this OS """
        if ";" not in package and "sys_platform" not in package:
            return package
        package = "".join(package.split())
        pkg, tags = package.split(";")
        tags = tags.split("==")
        sys_platform = tags[tags.index("sys_platform") + 1].replace('"', "").replace("'", "")
        if sys_platform == sys.platform:
            return pkg
        return None


class Installer():
    """ Install packages through Conda or Pip """
    def __init__(self, environment):
        self.packages = environment.packages_to_install
        self.env = environment
        self.install()

    def install(self):
        """ Install required pip packages """
        success = True
        for pkg in self.packages:
            output("Installing {}".format(pkg))
            if self.env.is_conda and self.conda_install(pkg):
                continue
            if not self.pip_install(pkg):
                success = False
        if not success:
            output("There were problems updating one or more dependencies.")
        else:
            output("Dependencies succesfully updated.")

    @staticmethod
    def conda_install(package):
        """ Install a conda package """
        success = True
        condaexe = ["conda", "install", "-y", package]
        try:
            with open(os.devnull, "w") as devnull:
                run(condaexe, stdout=devnull, stderr=devnull, check=True)
        except CalledProcessError:
            output("{} not available in Conda. Installing with pip...".format(package))
            success = False
        return success

    def pip_install(self, package):
        """ Install a pip package """
        success = True
        pipexe = [sys.executable, "-m", "pip"]
        # hide info/warning and fix cache hang
        pipexe.extend(["install", "-qq", "--no-cache-dir"])
        # install as user to solve perm restriction
        if not self.env.is_admin and not self.env.is_virtualenv:
            pipexe.append("--user")
        pipexe.append(package)
        try:
            run(pipexe, check=True)
        except CalledProcessError:
            output("Couldn't install {}. Please install this package manually".format(package))
            success = False
        return success


def output(msg):
    """ Output to print or logger """
    if _LOGGER is not None:
        _LOGGER.info(msg)
    else:
        print(msg)


def main(logger=None):
    """ Check for and update dependencies """
    if logger is not None:
        global _LOGGER  # pylint:disable=global-statement
        _LOGGER = logger
    output("Updating Dependencies...")
    update = Environment()
    packages = update.packages_to_install
    if not packages:
        output("All Dependencies are up to date")
    else:
        Installer(update)


if __name__ == "__main__":
    main()
