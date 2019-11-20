#!/usr/bin/env python3
""" Handling of Faceswap GUI Projects, Tasks and Sessions """

import logging
import os
import tkinter as tk

from lib.serializer import get_serializer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class GuiSession():
    """ Parent class for GUI Session Handlers """
    def __init__(self, config, file_handler=None):
        # NB file_handler has to be passed in to avoid circular imports
        logger.debug("Initializing: %s: (config: %s, file_handler: %s)",
                     self.__class__.__name__, config, file_handler)
        self._serializer = get_serializer("json")
        self._config = config

        self._options = None
        self._file_handler = file_handler
        self._filename = None
        self._saved_tasks = None
        self._modified = False
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _command_notebook(self):
        """ Return the command tab names to indices """
        return self._config.command_notebook

    @property
    def _tools_notebook(self):
        """ Return the command tab names to indices """
        return self._config.tools_notebook

    @property
    def _command_tabs(self):
        """ Return the command tab names to indices """
        return self._config.command_tabs

    @property
    def _tools_tabs(self):
        """ Return the command tab names to indices """
        return self._config.tools_tabs

    @property
    def _active_tab(self):
        """str: The name of the currently selected tab """
        return self._config.get_active_tab_name()

    @property
    def _modified_vars(self):
        """ dict: The command tab modified tkinter variables """
        return self._config.modified_vars

    @property
    def _cleared_options(self):
        cleared_opts = dict()
        for command, option in self._config.cli_opts.get_option_values().items():
            command_opts = dict()
            for key, val in option.items():
                if isinstance(val, bool):
                    command_opts[key] = False
                elif isinstance(val, int):
                    command_opts[key] = 0
                elif isinstance(val, float):
                    command_opts[key] = 0.0
                else:
                    command_opts[key] = ""
            cleared_opts[command] = command_opts
        return cleared_opts

    @property
    def _file_exists(self):
        return self._filename is not None and os.path.isfile(self._filename)

    @property
    def _default_opts(self):
        return self._config.default_options

    @property
    def filename(self):
        """ str: the filename for the session config file """
        return self._filename

    @property
    def cli_options(self):
        """ dict: Just the cli options from :attr:_options with extra fields removed """
        return {key: val for key, val in self._options.items() if isinstance(val, dict)}

    def _set_filename(self, filename=None, sess_type="project"):
        logger.debug("filename: %s, sess_type: %s", filename, sess_type)
        handler = "config_{}".format(sess_type)

        if filename is None:
            cfgfile = self._file_handler("open", handler).retfile
            if not cfgfile:
                return False
            filename = cfgfile.name
            cfgfile.close()

        if not os.path.isfile(filename):
            msg = "File does not exist: '{}'".format(filename)
            logger.error(msg)
            return False
        logger.debug("Setting filename: %s", filename)
        self._filename = filename
        return True

    def _get_command_options(self, command):
        """ return the saved options for the requested
            command, if not loading global options """
        opts = self._options.get(command, None)
        retval = {command: opts}
        if not opts:
            self._config.tk_vars["consoleclear"].set(True)
            print("No {} section found in file".format(command))
            logger.info("No  %s section found in file", command)
            retval = None
        logger.debug(retval)
        return retval

    def set_options(self, command=None):
        """ Set the options """
        opts = self._get_command_options(command) if command else self.cli_options
        for cmd, opt in opts.items():
            self._set_command_args(cmd, opt)
        tab_name = self._options.get("tab_name", None) if command is None else command
        tab_name = tab_name if tab_name is not None else "extract"
        self._config.set_active_tab_by_name(tab_name)

    def _set_command_args(self, command, options):
        """ Pass the saved config items back to the CliOptions """
        if not options:
            return
        for srcopt, srcval in options.items():
            optvar = self._config.cli_opts.get_one_option_variable(command, srcopt)
            if not optvar:
                continue
            optvar.set(srcval)

    def set_modified_callback(self):
        """ Add a callback to indicate that the file has changed from saved """
        for key, tkvar in self._modified_vars.items():
            tkvar.trace("w", lambda name, index, mode, cmd=key: self._modified_callback(cmd))

    def _modified_callback(self, command):
        raise NotImplementedError

    def _reset_modified_var(self, command=None):
        """ Reset all or given commands modified state to False """
        for key, tk_var in self._modified_vars.items():
            if command is None or command == key:
                logger.debug("Reset modified state for: %s", command)
                tk_var.set(False)

    def _add_to_recent(self, command=None):
        """ Add to recent files """
        if self._filename is None:
            logger.debug("No filename for selected file. Not adding to recent.")
            return
        recent_filename = os.path.join(self._config.pathcache, ".recent.json")
        logger.debug("Adding to recent files '%s': (%s, %s)",
                     recent_filename, self._filename, command)
        if not os.path.exists(recent_filename) or os.path.getsize(recent_filename) == 0:
            recent_files = []
        else:
            recent_files = self._serializer.load(recent_filename)
        logger.debug("Initial recent files: %s", recent_files)
        recent_files = self._del_from_recent(self._filename, recent_files)
        ftype = "project" if command is None else command
        recent_files.insert(0, (self._filename, ftype))
        recent_files = recent_files[:20]
        logger.debug("Final recent files: %s", recent_files)
        self._serializer.save(recent_filename, recent_files)

    def _del_from_recent(self, filename, recent_files=None, save=False):
        """ Remove an item from recent files """
        recent_filename = os.path.join(self._config.pathcache, ".recent.json")
        if recent_files is None:
            if not os.path.exists(recent_filename) or os.path.getsize(recent_filename) == 0:
                logger.debug("No recent file list")
                return None
            recent_files = self._serializer.load(recent_filename)
        filenames = [recent[0] for recent in recent_files]
        if filename in filenames:
            idx = filenames.index(filename)
            logger.debug("Removing from recent file list: %s", filename)
            del recent_files[idx]
            if save:
                self._serializer.save(recent_filename, recent_files)
        else:
            logger.debug("Filename '%s' does not appear in recent file list", filename)
        return recent_files

    def _get_lone_task(self):
        command = None
        if len(self.cli_options) == 1:
            command = list(self.cli_options.keys())[0]
        return command

    def _load(self):
        if self._file_exists:
            logger.debug("Loading config")
            self._options = self._serializer.load(self._filename)
            retval = True
        else:
            logger.debug("File doesn't exist. Aborting")
            retval = False
        return retval

    def _save(self, command=None):
        """ Collect the cli options and save to file """
        self._options = self._config.cli_opts.get_option_values(command)
        self._options["tab_name"] = self._active_tab
        self._serializer.save(self._filename, self._options)
        self._reset_modified_var(command)


class Tasks(GuiSession):
    """ Faceswap .fst Task File handling """
    def __init__(self, config, file_handler):
        super().__init__(config, file_handler)
        self._tasks = dict()

    def _modified_callback(self, command):
        pass

    def load(self, *args,  # pylint:disable=unused-argument
             command=None, filename=None, current_tab=True):
        """ Pop up load dialog for a saved task file """
        logger.debug("Loading task config: (command: '%s', filename: %s)", command, filename)
        # Option to load specific task from project files:
        sess_type = "task" if not current_tab else "all"
        filename_set = self._set_filename(filename, sess_type=sess_type)
        if not filename_set:
            return
        loaded = self._load()
        if not loaded:
            return

        if command is None and current_tab:
            command = self._active_tab
        elif command is None:
            command = self._get_lone_task()

        if command is None:
            logger.error("Unable to determine task from the given file: '%s'", filename)
            return
        if command not in self._options:
            logger.error("No '%s' task in '%s'", command, self._filename)
            return

        self.set_options(command)
        self._filename = None if self._filename.endswith(".fsw") else self._filename
        self._add_to_recent(command)
        self._add_task(command)
        logger.debug("Loaded task config: (command: '%s', filename: '%s')", command, filename)

    def clear(self, command):
        """ Reset all task options to default """
        self._config.cli_opts.reset(command)

    def reload(self, command):
        """ Reload task options from last save """
        self._filename, self._options = self._get_task(command)

        if self._options is None:
            logger.info("No active task to reload")
            return
        logger.debug("Reloading task")
        self.set_options()
        if self._filename is None:
            self._reset_modified_var(command)

    def save(self, command, save_as=False):
        """ Save the current GUI state for current tasks to a config file in json format """
        logger.debug("Saving config as...")
        self._filename, self._options = self._get_task(command)

        save_as = save_as or self._filename is None
        if save_as:
            cfgfile = self._file_handler("save", "config_task").retfile
            if not cfgfile:
                return
            self._filename = cfgfile.name
            cfgfile.close()

        self._save(command=command)
        self._add_task(command)
        self._add_to_recent()
        if not save_as:
            logger.info("Saved project to: '%s'", self._filename)
        else:
            logger.debug("Saved project to: '%s'", self._filename)

    def _add_task(self, command):
        self._tasks[command] = dict(filename=self._filename, options=self._options)

    def add_project_task(self, command, options):
        """ Add a task from a project """
        self._tasks[command] = dict(filename=None, options=options)

    def _get_task(self, command):
        task = self._tasks.get(command, None)
        if task is None:
            filename, options = (None, None)
        else:
            filename, options = (task.get("filename", None), task.get("options", None))
        logger.debug("command: %s, filename: %s, options: %s", command, filename, options)
        return filename, options


class Project(GuiSession):
    """ Faceswap .fsw Project File handling """
    def __init__(self, config, file_handler):
        super().__init__(config, file_handler)
        self._update_root_title()

    @property
    def _project_modified(self):
        """bool: True if the project has been modified otherwise False """
        return any([var.get() for var in self._modified_vars.values()])

    def set_default_opts(self):
        """ Set the default options """
        self._options = self._default_opts

    def _modified_callback(self, command):
        if self._modified == self._project_modified:
            logger.debug("Change state is same as current state: (project: %s, variables: %s)",
                         self._modified, self._project_modified)
            return
        logger.debug("Updating project state from variable: (modified: %s)",
                     self._project_modified)
        self._modified = self._project_modified
        self._update_root_title()

    def load(self, *args, filename=None):  # pylint:disable=unused-argument
        """ Pop up load dialog for a saved config file """
        logger.debug("Loading project config: (filename: '%s')", filename)
        filename_set = self._set_filename(filename, sess_type="project")
        if not filename_set:
            return
        loaded = self._load()
        if not loaded:
            return

        command = self._get_lone_task()
        if command is not None and os.path.splitext(filename)[1] == ".fsw":
            self._update_legacy_task(command)

        self.set_options(command)
        self._update_tasks()
        self._add_to_recent(command)
        self._reset_modified_var(command)
        self._update_root_title()
        logger.debug("Loaded project config: (command: '%s', filename: '%s')", command, filename)

    def _update_tasks(self):
        """ Add the tasks to the :class:`Tasks` class """
        for key, val in self.cli_options.items():
            opts = {key: val}
            opts["tab_name"] = key
            self._config.tasks.add_project_task(key, opts)

    def reload(self, *args):  # pylint:disable=unused-argument
        """ Reload saved options """
        if self._options is None:
            logger.info("No active project to reload")
            return
        logger.debug("Reloading project")
        self.set_options()
        self._update_tasks()
        self._reset_modified_var()
        self._update_root_title()

    def _update_legacy_task(self, command):
        """ Update old tasks saved as projects (.fsw) to tasks (.fst) """
        new_file = "{}.fst".format(os.path.splitext(self._filename)[0])
        self._options["tab_name"] = command
        self._serializer.save(new_file, self._options)
        os.remove(self._filename)
        self._del_from_recent(self._filename, save=True)
        logger.debug("Updated legacy from: '%s' to: '%s", self._filename, new_file)
        self._set_filename(new_file)

    def _update_root_title(self):
        text = "<untitled project>" if self._filename is None else os.path.basename(self._filename)
        text += "*" if self._modified else ""
        self._config.set_root_title(text=text)

    def save(self, *args, save_as=False):  # pylint:disable=unused-argument
        """ Save the current GUI state to a config file in json format """
        logger.debug("Saving config as...")

        save_as = save_as or self._filename is None
        if save_as:
            cfgfile = self._file_handler("save", "config_project").retfile
            if not cfgfile:
                return
            self._filename = cfgfile.name
            cfgfile.close()

        self._save()
        self._update_tasks()
        self._update_root_title()
        self._add_to_recent()
        if not save_as:
            logger.info("Saved project to: '%s'", self._filename)
        else:
            logger.debug("Saved project to: '%s'", self._filename)

    def new(self, *args):  # pylint:disable=unused-argument
        """ Create a new project with default options """
        logger.debug("Creating new project")
        cfgfile = self._file_handler("save", "config_project").retfile
        if not cfgfile:
            return
        self._filename = cfgfile.name
        cfgfile.close()

        self._config.cli_opts.reset()
        self._save()
        self._update_root_title()
        self._add_to_recent()

    def close(self, *args):  # pylint:disable=unused-argument
        """ Clear the current project """
        self._config.cli_opts.reset()
        self._filename = None
        self._reset_modified_var()
        self._update_root_title()
        self._config.set_active_tab_by_name(self._config.user_config_dict["tab"])


class LastSession(GuiSession):
    """ Save and load last session """
    def __init__(self, config):
        super().__init__(config)
        self._filename = os.path.join(self._config.pathcache, ".last_session.json")
        if not self._enabled:
            return

        if self._save_option == "prompt":
            self.ask_load()
        elif self._save_option == "always":
            self.load()

    def _modified_callback(self, command):
        """ No need for a modified callback on LastSession as it is always on or off """
        pass  # pylint: disable=unnecessary-pass

    @property
    def _save_option(self):
        """ str: The user config autosave option """
        return self._config.user_config_dict.get("autosave_last_session", "never")

    @property
    def _enabled(self):
        """ bool: ``True`` if autosave is enabled otherwise ``False`` """
        return self._save_option != "never"

    def ask_load(self):
        """ Load the last saved session """
        if not self._file_exists:
            logger.debug("No last session file found")
        elif tk.messagebox.askyesno("Last Session", "Load last session?"):
            self.load()

    def load(self):
        """ Load last sessions """
        loaded = self._load()
        if not loaded:
            return
        needs_update = self._set_project()
        if needs_update:
            self.set_options()

    def _set_project(self):
        """ Set the project if session is resuming from one

            Returns
            -------
            bool:
                ``True`` If the GUI still needs to be updated from the last session, ``False`` if
                the returned GUI state is the last session
        """
        if self._options.get("project", None) is None:
            logger.debug("No project stored")
            retval = True
        else:
            logger.debug("Loading stored project")
            self._config.project.load(filename=self._options["project"])
            retval = self.cli_options != self._config.project.cli_options
        logger.debug("Needs update: %s", retval)
        return retval

    def collect_options(self):
        """ Collect the cli options for json serializing """
        opts = self._config.cli_opts.get_option_values()
        if not opts or opts == self._default_opts or opts == self._cleared_options:
            logger.debug("Default session, or no opts found. Not saving last session.")
            return None
        opts["tab_name"] = self._active_tab
        opts["project"] = self._config.project.filename
        return opts

    def save(self):
        """ Save snapshot of config options """
        if not self._enabled:
            logger.debug("LastSession not enabled")
            if os.path.exists(self._filename):
                logger.debug("Deleting existing LastSession file")
                os.remove(self._filename)
            return

        opts = self.collect_options()
        if opts is None and os.path.exists(self._filename):
            logger.debug("Last session default or blank. Clearing saved last session.")
            os.remove(self._filename)
        if opts is not None:
            self._serializer.save(self._filename, opts)
            logger.debug("Saved last session. (filename: '%s', opts: %s", self._filename, opts)
