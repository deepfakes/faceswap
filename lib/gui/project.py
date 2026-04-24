#!/usr/bin/env python3
"""Handling of Faceswap GUI Projects, Tasks and Last Session"""
from __future__ import annotations

import logging
import os
import tkinter as tk
from tkinter import messagebox
import typing as T

from lib.serializer import get_serializer
from lib.gui import gui_config as cfg
from lib.logger import parse_class_init
from lib.utils import get_module_objects


if T.TYPE_CHECKING:
    from .utils.config import Config
    from .utils import FileHandler

logger = logging.getLogger(__name__)


class _GuiSession():  # pylint:disable=too-few-public-methods
    """Parent class for GUI Session Handlers.

    Parameters
    ----------
    config
        The master GUI config
    file_handler
        A file handler object

    """
    def __init__(self, config: Config, file_handler: type[FileHandler] | None = None) -> None:
        # NB file_handler has to be passed in to avoid circular imports
        logger.debug(parse_class_init(locals()))
        self._serializer = get_serializer("json")
        self._config = config

        self._options: dict[str, str | dict[str, bool | int | float | str]] | None = None
        self._file_handler = file_handler
        self._filename: str | None = None
        self._saved_tasks = None
        self._modified = False

    @property
    def _active_tab(self) -> str:
        """The name of the currently selected :class:`lib.gui.command.CommandNotebook` tab"""
        notebook = self._config.command_notebook
        assert notebook is not None
        tools_book = self._config.tools_notebook
        command = notebook.tab(notebook.select(), "text").lower()
        if command == "tools":
            command = tools_book.tab(tools_book.select(), "text").lower()
        logger.debug("Active tab: %s", command)
        return command

    @property
    def _modified_vars(self) -> dict[str, tk.BooleanVar]:
        """The tkinter Boolean vars indicating the modified state for each tab."""
        return self._config.modified_vars

    @property
    def _file_exists(self) -> bool:
        """``True`` if :attr:`_filename` exists otherwise ``False``."""
        return self._filename is not None and os.path.isfile(self._filename)

    @property
    def _cli_options(self) -> dict[str, dict[str, bool | int | float | str]]:
        """The raw cli options from :attr:`_options` with project fields removed. """
        assert self._options is not None
        return {key: val for key, val in self._options.items() if isinstance(val, dict)}

    @property
    def _default_options(self) -> dict[str, T.Any]:
        """The default options for all tabs"""
        return self._config.default_options

    @property
    def _dirname(self) -> str | None:
        """The folder name that :attr:`_filename` resides in. Returns ``None`` if filename is
        ``None``."""
        return os.path.dirname(self._filename) if self._filename is not None else None

    @property
    def _basename(self) -> str | None:
        """The base name of :attr:`_filename`. Returns ``None`` if filename is ``None``."""
        return os.path.basename(self._filename) if self._filename is not None else None

    @property
    def _stored_tab_name(self) -> str | None:
        """The tab_name stored in :attr:`_options` or ``None`` if it does not exist"""
        if self._options is None:
            return None
        retval = self._options.get("tab_name", None)
        assert retval is None or isinstance(retval, str)
        return retval

    @property
    def _selected_to_choices(self) -> dict[str, dict[str, dict[str, T.Any]]]:
        """The selected value and valid choices for multi-option, radio or combo options."""
        # TODO do instance check on CliOption. Not done for now due to circular import
        # pylint:disable=line-too-long
        valid_choices = {
            cmd: {
                opt: {
                    "choices": val.panel_option.choices,  # pyright:ignore[reportAttributeAccessIssue]  # noqa[E501]
                    "is_multi": val.panel_option.is_multi_option  # pyright:ignore[reportAttributeAccessIssue]  # noqa[E501]
                    }
                for opt, val in data.items()
                if hasattr(val, "panel_option")  # Filter out helptext
                and val.panel_option.choices is not None  # pyright:ignore[reportAttributeAccessIssue]  # noqa[E501]
                }
            for cmd, data in self._config.cli_opts.opts.items()
            }
        logger.trace("valid_choices: %s", valid_choices)  # type:ignore[attr-defined]
        assert self._options is not None
        retval = {command: {option: {"value": value,
                                     "is_multi": valid_choices[command][option]["is_multi"],
                                     "choices":  valid_choices[command][option]["choices"]}
                            for option, value in options.items()
                            if value and command in valid_choices
                            and option in valid_choices[command]}
                  for command, options in self._options.items()
                  if isinstance(options, dict)}
        logger.trace("returning: %s", retval)  # type:ignore[attr-defined]
        return retval

    def _current_gui_state(self, command: str | None = None
                           ) -> dict[str, dict[str, bool | int | float | str]]:
        """The current state of the GUI.

        Parameters
        ----------
        command
            If provided, returns the state of just the given tab command. If ``None`` returns
            options for all tabs. Default ``None``

        Returns
        -------
        The options currently set in the GUI
        """
        return self._config.cli_opts.get_option_values(command)

    def _set_filename(self,
                      filename: str | None = None,
                      session_type: T.Literal["all", "project", "task"] = "project") -> bool:
        """Set the :attr:`_filename` attribute.

        :attr:`_filename` is set either from a given filename or the result from
        a :attr:`_file_handler`.

        Parameters
        ----------
        filename
            An optional filename. If given then this filename will be used otherwise it will be
            collected by a :attr:`_file_handler`

        session_type
            The session type that the filename is being set for. Dictates the type of file handler
            that is opened. Default: `"Project"`

        Returns
        -------
        ``True`` if filename has been successfully set otherwise ``False``
        """
        logger.debug("filename: '%s', session_type: '%s'", filename, session_type)
        handler = T.cast(T.Literal["config_all", "config_project", "config_task"],
                         f"config_{session_type}")
        if filename is None:
            logger.debug("Popping file handler")
            assert self._file_handler is not None
            cfg_file = self._file_handler("open", handler).return_file
            if not cfg_file:
                logger.debug("No filename given")
                return False
            filename = cfg_file.name
            cfg_file.close()
            assert filename is not None

        if not os.path.isfile(filename):
            msg = f"File does not exist: '{filename}'"
            logger.error(msg)
            return False
        ext = os.path.splitext(filename)[1]
        if (session_type == "project" and ext != ".fsw") or (session_type == "task"
                                                             and ext != ".fst"):
            logger.debug("Invalid file extension for session type: (session_type: '%s', "
                         "extension: '%s')", session_type, ext)
            return False
        logger.debug("Setting filename: '%s'", filename)
        self._filename = filename
        return True

    # GUI STATE SETTING
    def _set_options(self, command: str | None = None) -> None:
        """Set the GUI options based on the currently stored properties of :attr:`_options`
        and sets the active tab.

        Parameters
        ----------
        command
            The tab to set the options for. If None then sets options for all tabs.
            Default: ``None``
        """
        opts = self._get_options_for_command(command) if command else self._cli_options
        logger.debug("command: %s, opts: %s", command, opts)
        if opts is None:
            logger.debug("No options found. Returning")
            return
        for cmd, opt in opts.items():
            self._set_gui_state_for_command(cmd, opt)
        assert self._options is not None
        tab_name = self._options.get("tab_name", None) if command is None else command
        tab_name = tab_name if tab_name is not None else "extract"
        logger.debug("tab_name: %s", tab_name)
        assert isinstance(tab_name, str)
        self._config.set_active_tab_by_name(tab_name)

    def _get_options_for_command(self, command: str
                                 ) -> dict[str, dict[str, bool | int | float | str]] | None:
        """Return a single command's options from :attr:`_options` formatted consistently with
        an all options dict.

        Parameters
        ----------
        command
            The command to return the options for

        Returns
        -------
        dict: The options for a single command in the format {command: options}. If the command
        is not found then returns ``None``
        """
        logger.debug(command)
        assert self._options is not None
        opts = T.cast(dict[str, int | float | bool | str] | None, self._options.get(command, None))
        if opts is None:
            self._config.tk_vars.console_clear.set(True)
            logger.info("No  %s section found in file", command)
            retval = None
        else:
            retval = {command: opts}
        logger.debug(retval)
        return retval

    def _set_gui_state_for_command(self,
                                   command: str,
                                   options: dict[str, bool | int | float | str]
                                   ) -> None:
        """Set the GUI state for the given command.

        Parameters
        ----------
        command
            The tab to set the options for
        options
            The option values to set the GUI to
        """
        logger.debug("command: %s: options: %s", command, options)
        if not options:
            logger.debug("No options provided, not updating GUI")
            return
        for src_opt, src_val in options.items():
            opt_var = self._config.cli_opts.get_one_option_variable(command, src_opt)
            if not opt_var:
                continue
            logger.trace(  # type:ignore[attr-defined]
                "setting option: (src_opt: %s, opt_var: %s, src_val: %s)",
                src_opt, opt_var, src_val)
            opt_var.set(src_val)

    def _reset_modified_var(self, command: str | None = None) -> None:
        """Reset :attr:`_modified_vars` variables back to unmodified (`False`) for all
        commands or for the given command.

        Parameters
        ----------
        command
            The command to reset the modified tkinter variable for. If ``None`` then all tkinter
            modified variables are reset to `False`. Default: ``None``
        """
        for key, tk_var in self._modified_vars.items():
            if (command is None or command == key) and tk_var.get():
                logger.debug("Reset modified state for: (command: %s key: %s)", command, key)
                tk_var.set(False)

    # RECENT FILE HANDLING
    def _add_to_recent(self, command: str | None = None) -> None:
        """Add the file for this session to the recent files list.

        Parameters
        ----------
        command
            The command that this session relates to. If `None` then the whole project is added.
            Default: ``None``
        """
        logger.debug(command)
        if self._filename is None:
            logger.debug("No filename for selected file. Not adding to recent.")
            return
        recent_filename = os.path.join(self._config.path_cache, ".recent.json")
        logger.debug("Adding to recent files '%s': (%s, %s)",
                     recent_filename, self._filename, command)
        if not os.path.exists(recent_filename) or os.path.getsize(recent_filename) == 0:
            logger.debug("Starting with empty recent_files list")
            recent_files: list[tuple[str, str]] | None = []
        else:
            logger.debug("loading recent_files list: %s", recent_filename)
            assert self._serializer is not None
            recent_files = self._serializer.load(  # pyright:ignore[reportCallIssue]
                recent_filename)
        logger.debug("Initial recent files: %s", recent_files)
        recent_files = self._del_from_recent(self._filename, recent_files)
        assert recent_files is not None
        f_type = "project" if command is None else command
        recent_files.insert(0, (self._filename, f_type))
        recent_files = recent_files[:20]
        logger.debug("Final recent files: %s", recent_files)

        assert self._serializer is not None
        self._serializer.save(recent_filename, recent_files)  # pyright:ignore[reportCallIssue]

    def _del_from_recent(self,
                         filename: str,
                         recent_files: list[tuple[str, str]] | None = None,
                         save: bool = False) -> list[tuple[str, str]] | None:
        """Remove an item from the recent files list.

        Parameters
        ----------
        filename
            The filename to be removed from the recent files list
        recent_files
            If the recent files list has already been loaded, it can be passed in to avoid
            loading again. If ``None`` then load the recent files list from disk. Default: ``None``
        save
            Whether the recent files list should be saved after removing the file. ``True`` saves
            the file, ``False`` does not. Default: ``False``

        Returns
        -------
        List of recent files and their filetypes
        """
        recent_filename = os.path.join(self._config.path_cache, ".recent.json")
        if recent_files is None:
            logger.debug("Loading file list from disk: %s", recent_filename)
            if not os.path.exists(recent_filename) or os.path.getsize(recent_filename) == 0:
                logger.debug("No recent file list")
                return None
            assert self._serializer is not None
            recent_files = self._serializer.load(  # pyright:ignore[reportCallIssue]
                recent_filename)
        assert recent_files is not None
        filenames = [recent[0] for recent in recent_files]
        if filename in filenames:
            idx = filenames.index(filename)
            logger.debug("Removing from recent file list: %s", filename)
            del recent_files[idx]
            if save:
                logger.debug("Saving recent files list: %s", recent_filename)
                assert self._serializer is not None
                self._serializer.save(recent_filename,  # pyright:ignore[reportCallIssue]
                                      recent_files)
        else:
            logger.debug("Filename '%s' does not appear in recent file list", filename)
        return recent_files

    def _get_lone_task(self) -> str | None:
        """Get the sole command name from :attr:`_options`.

        Returns
        -------
        The only existing command name in the current :attr:`_options` dict or ``None`` if there
        are multiple commands stored.
        """
        command = None
        if len(self._cli_options) == 1:
            command = list(self._cli_options.keys())[0]
        logger.debug(command)
        return command

    # DISK IO
    def _load(self) -> bool:
        """Load GUI options from :attr:`_filename` location and set to :attr:`_options`.

        Returns
        -------
        ``True`` if successfully loaded otherwise ``False``
        """
        if self._file_exists:
            logger.debug("Loading config")
            assert self._serializer is not None
            self._options = self._serializer.load(  # pyright:ignore[reportCallIssue]
                self._filename)
            self._check_valid_choices()
            retval = True
        else:
            logger.debug("File doesn't exist. Aborting")
            retval = False
        return retval

    def _check_valid_choices(self) -> None:
        """Check whether the loaded file has any selected combo/radio/multi-option values that are
        no longer valid and remove them so that they are not passed into faceswap."""
        assert self._options is not None
        for command, options in self._selected_to_choices.items():
            opts = T.cast(dict[str, bool | int | float | str], self._options[command])
            for option, data in options.items():
                if not data["is_multi"] and data["value"] in data["choices"]:
                    continue
                if (data["is_multi"] and
                        isinstance(data["value"], str) and
                        all(v in data["choices"] for v in data["value"].split())):
                    continue
                if data["is_multi"] and isinstance(data["value"], str):
                    val = " ".join([v for v in data["value"].split() if v in data["choices"]])
                else:
                    val = ""
                val = self._default_options[command][option] if not val else val
                logger.debug("Updating invalid value to default: (command: '%s', option: '%s', "
                             "original value: '%s', new value: '%s')", command, option,
                             opts[option], val)
                opts[option] = val

    def _save_as_to_filename(self, session_type: T.Literal["all", "task", "project"]) -> bool:
        """Set :attr:`_filename` from a save as dialog.

        Parameters
        ----------
        session_type: ['all', 'task', 'project']
            The type of session to pop the save as dialog for. Limits the allowed filetypes

        Returns
        -------
        True if :attr:`filename` successfully set otherwise ``False``
        """
        logger.debug("Popping save as file handler. session_type: '%s'", session_type)
        title = f"Save {f'{session_type.title()} ' if session_type != 'all' else ''}As..."
        assert self._file_handler is not None
        cfg_file = self._file_handler(
            "save",
            T.cast(T.Literal["config_all", "config_project", "config_task"],
                   f"config_{session_type}"),
            title=title,
            initial_folder=self._dirname).return_file
        if not cfg_file:
            logger.debug("No filename provided. session_type: '%s'", session_type)
            return False
        self._filename = cfg_file.name
        logger.debug("Set filename: (session_type: '%s', filename: '%s'",
                     session_type, self._filename)
        cfg_file.close()
        return True

    def _save(self, command: str | None = None) -> None:
        """Collect the options in the current GUI state and save.

        Obtains the current options set in the GUI with the selected tab and applies them to
        :attr:`_options`. Saves :attr:`_options` to :attr:`_filename`. Resets :attr:_modified_vars
        for either the given command or all commands,

        Parameters
        ----------
        command
            The tab to collect the current state for. If ``None`` then collects the current
            state for all tabs. Default: ``None``
        """
        self._options = T.cast(dict[str, str | dict[str, bool | int | float | str]],
                               self._current_gui_state(command))
        self._options["tab_name"] = self._active_tab
        logger.debug("Saving options: (filename: %s, options: %s", self._filename, self._options)
        assert self._serializer is not None
        self._serializer.save(self._filename, self._options)  # pyright:ignore[reportCallIssue]
        self._reset_modified_var(command)
        self._add_to_recent(command)


class Tasks(_GuiSession):
    """Faceswap ``.fst`` Task File handling.

    Faceswap tasks handle the management of each individual task tab in the GUI. Unlike
    :class:`Projects`, Tasks contains all the active tasks currently running, rather than an
    individual task.

    Parameters
    ----------
    config
        The master GUI config
    file_handler
        A file handler object
    """
    def __init__(self, config: Config, file_handler: type[FileHandler]):
        super().__init__(config, file_handler)
        self._tasks: dict[
            str, dict[T.Literal["filename", "options", "is_project"],
                      str | bool | dict[str, str | dict[str,
                                                        bool | int | float | str]] | None]] = {}

    @property
    def _is_project(self) -> bool:
        """``True`` if all tasks are from an overarching session project else ``False``."""
        retval = False if not self._tasks else all(v.get("is_project", False)
                                                   for v in self._tasks.values())
        return retval

    @property
    def _project_filename(self) -> str | None:
        """The overarching session project filename."""
        fname = None
        if not self._is_project:
            return fname

        for val in self._tasks.values():
            fname = val["filename"]
            break
        assert fname is None or isinstance(fname, str)
        return fname

    def load(self,  # pylint:disable=unused-argument
             *args,
             filename: str | None = None,
             current_tab: bool = True) -> None:
        """Load a task into this :class:`Tasks` class.

        Tasks can be loaded from project ``.fsw`` files or task ``.fst`` files, depending on where
        this function is being called from.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        filename
            If a filename is passed in, This will be used, otherwise a file handler will be
            launched to select the relevant file.
        current_tab
            ``True`` if the task to be loaded must be for the currently selected tab. ``False``
            if loading a task into any tab. If current_tab is `True` then tasks can be loaded from
            ``.fsw`` and ``.fst`` files, otherwise they can only be loaded from ``.fst`` files.
            Default: ``True``
        """
        logger.debug("Loading task config: (filename: '%s', current_tab: '%s')",
                     filename, current_tab)

        # Option to load specific task from project files:
        session_type: T.Literal["all", "task"] = "all" if current_tab else "task"

        is_legacy = (not self._is_project and
                     filename is not None and session_type == "task" and
                     os.path.splitext(filename)[1] == ".fsw")
        if is_legacy:
            logger.debug("Legacy task found: '%s'", filename)
            assert filename is not None
            filename = self._update_legacy_task(filename)

        filename_set = self._set_filename(filename, session_type=session_type)
        if not filename_set:
            return
        loaded = self._load()
        if not loaded:
            return

        command = self._active_tab if current_tab else self._stored_tab_name
        command = self._get_lone_task() if command is None else command
        if command is None:
            logger.error("Unable to determine task from the given file: '%s'", filename)
            return
        assert self._options is not None
        if command not in self._options:
            logger.error("No '%s' task in '%s'", command, self._filename)
            return

        self._set_options(command)
        self._add_to_recent(command)

        if self._is_project:
            self._filename = self._project_filename
        elif self._filename is not None and self._filename.endswith(".fsw"):
            self._filename = None

        self._add_task(command)
        if is_legacy:
            self.save()

        logger.debug("Loaded task config: (command: '%s', filename: '%s')", command, filename)

    def _update_legacy_task(self, filename: str) -> str:
        """Update legacy ``.fsw`` tasks to ``.fst`` tasks.

        Tasks loaded from the recent files menu may be passed in with a ``.fsw`` extension.
        This renames the file and removes it from the recent file list.

        Parameters
        ----------
        filename
            The filename of the `.fsw` file that needs converting

        Returns
        -------
        The new filename of the updated tasks file
        """
        # TODO remove this code after a period of time. Implemented November 2019
        logger.debug("original filename: '%s'", filename)
        fname, ext = os.path.splitext(filename)
        if ext != ".fsw":
            logger.debug("Not a .fsw file: '%s'", filename)
            return filename

        new_filename = f"{fname}.fst"
        logger.debug("Renaming '%s' to '%s'", filename, new_filename)
        os.rename(filename, new_filename)
        self._del_from_recent(filename, save=True)
        logger.debug("new filename: '%s'", new_filename)
        return new_filename

    def save(self, save_as: bool = False) -> None:
        """Save the current GUI state for the active tab to a ``.fst`` faceswap task file.

        Parameters
        ----------
        save_as
            Whether to save to the stored filename, or pop open a file handler to ask for a
            location. If there is no stored filename, then a file handler will automatically be
            popped. Default: ``False``
        """
        logger.debug("Saving config...")
        self._set_active_task()
        save_as = save_as or self._is_project or self._filename is None

        if save_as and not self._save_as_to_filename("task"):
            return

        command = self._active_tab
        self._save(command=command)
        self._add_task(command)
        if not save_as:
            logger.info("Saved project to: '%s'", self._filename)
        else:
            logger.debug("Saved project to: '%s'", self._filename)

    def clear(self) -> None:
        """Reset all GUI options to their default values for the active tab."""
        self._config.cli_opts.reset(self._active_tab)

    def reload(self) -> None:
        """Reset currently selected tab GUI options to their last saved state."""
        self._set_active_task()

        if self._options is None:
            logger.info("No active task to reload")
            return
        logger.debug("Reloading task")
        self.load(filename=self._filename, current_tab=True)
        if self._is_project:
            self._reset_modified_var(self._active_tab)

    def _add_task(self, command: str) -> None:
        """Add the currently active task to the internal :attr:`_tasks` dict.

        If the currently stored task is from an overarching session project, then
        only the options are updated. When resetting a tab to saved a project will always
        be preferred to a task loaded into the project, so the original reference file name
        stays with the project.

        Parameters
        ----------
        command
            The tab that pertains to the currently active task
        """
        self._tasks[command] = {"filename": self._filename,
                                "options": self._options,
                                "is_project": self._is_project}

    def clear_tasks(self) -> None:
        """Clears all of the stored tasks.

        This is required when loading a task stored in a legacy project file, and is only to be
        called by :class:`Project` when a project has been loaded which is in fact a task.
        """
        logger.debug("Clearing stored tasks")
        self._tasks = {}

    def add_project_task(self,
                         filename: str,
                         command: str,
                         options: dict[str, str | dict[str, bool | int | float | str]]) -> None:
        """Add an individual task from a loaded :class:`Project` to the internal :attr:`_tasks`
        dict.

        Project tasks take priority over any other tasks, so the individual tasks from a new
        project must be placed in the _tasks dict.

        Parameters
        ----------
        filename
            The filename of the session project file
        command
            The tab that this task's options belong to
        options
            The options for this task loaded from the project
        """
        self._tasks[command] = {"filename": filename, "options": options, "is_project": True}

    def _set_active_task(self, command: str | None = None) -> None:
        """Set the active :attr:`_filename` and :attr:`_options` to currently selected tab's
        options.

        Parameters
        ----------
        command
            If a command is passed in then set the given tab to active, If this is none set the tab
            which currently has focus to active. Default: ``None``
        """
        logger.debug(command)
        command = self._active_tab if command is None else command
        task = self._tasks.get(command, None)
        if task is None:
            self._filename, self._options = (None, None)
        else:
            filename = task.get("filename", None)
            opts = task.get("options", None)
            assert filename is None or isinstance(filename, str)
            assert opts is None or isinstance(opts, dict)
            self._filename = filename
            self._options = opts
        logger.debug("tab: %s, filename: %s, options: %s",
                     self._active_tab, self._filename, self._options)


class Project(_GuiSession):
    """Faceswap ``.fsw`` Project File handling.

    Faceswap projects handle the management of all task tabs in the GUI and updates
    the main Faceswap title bar with the project name and modified state.

    Parameters
    ----------
    config
        The master GUI config
    file_handler
        A file handler object
    """

    def __init__(self, config: Config, file_handler: type[FileHandler]) -> None:
        super().__init__(config, file_handler)
        self._update_root_title()

    @property
    def filename(self) -> str | None:
        """The currently active project filename."""
        return self._filename

    @property
    def cli_options(self) -> dict[str, dict[str, bool | int | float | str]]:
        """The raw cli options from :attr:`_options` with project fields removed."""
        return self._cli_options

    @property
    def _project_modified(self) -> bool:
        """``True`` if the project has been modified otherwise ``False``. """
        return any(var.get() for var in self._modified_vars.values())

    @property
    def _tasks(self) -> Tasks:
        """The current session's :class:``Tasks``."""
        return self._config.tasks

    def set_default_options(self) -> None:
        """Set the default options. The Default GUI options are stored on Faceswap startup.

        Exposed as the :attr:`_default_options` for a project cannot be set until after the main
        Command Tabs have been loaded.
        """
        logger.debug("Setting options to default")
        self._options = self._default_options

    # MODIFIED STATE CALLBACK
    def set_modified_callback(self) -> None:
        """Adds a callback to each of the :attr:`_modified_vars` tkinter variables
        When one of these variables is changed, triggers :func:`_modified_callback`
        with the command that was changed.

        This is exposed as the callback can only be added after the main Command Tabs have
        been drawn, and their options' initial values have been set."""
        for key, tk_var in self._modified_vars.items():
            logger.debug("Adding callback for tab: %s", key)
            tk_var.trace("w", self._modified_callback)

    def _modified_callback(self, *args) -> None:  # pylint:disable=unused-argument
        """Update the project modified state on a GUI modification change and
        update the Faceswap title bar. """
        if self._project_modified and self._current_gui_state() == self._cli_options:
            logger.debug("Project is same as stored. Setting modified to False")
            self._reset_modified_var()

        if self._modified != self._project_modified:
            logger.debug("Updating project state from variable: (modified: %s)",
                         self._project_modified)
            self._modified = self._project_modified
            self._update_root_title()

    def load(self,  # pylint:disable=unused-argument
             *args,
             filename: str | None = None,
             last_session: bool = False) -> None:
        """Load a project from a saved ``.fsw`` project file.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        filename
            If a filename is passed in, This will be used, otherwise a file handler will be
            launched to select the relevant file.
        last_session
            ``True`` if the project is being loaded from the last opened session ``False`` if the
            project is being loaded directly from disk. Default: ``False``
        """
        logger.debug("Loading project config: (filename: '%s', last_session: %s)",
                     filename, last_session)
        filename_set = self._set_filename(filename, session_type="project")

        if not filename_set:
            logger.debug("No filename set")
            return
        loaded = self._load()
        if not loaded:
            logger.debug("Options not loaded")
            return

        # Legacy .fsw files could store projects or tasks. Check if this is a legacy file
        # and hand off file to Tasks if necessary
        command = self._get_lone_task()
        legacy = command is not None
        if legacy:
            self._handoff_legacy_task()
            return

        if not last_session:
            self._set_options()  # Options will be set by last session. Don't set now
        self._update_tasks()
        self._add_to_recent()
        self._reset_modified_var()
        self._update_root_title()
        logger.debug("Loaded project config: (command: '%s', filename: '%s')", command, filename)

    def _handoff_legacy_task(self) -> None:
        """Update legacy tasks saved with the old file extension ``.fsw`` to tasks ``.fst``.

        Hands off file handling to :class:`Tasks` and resets project to default."""
        logger.debug("Updating legacy task '%s", self._filename)
        filename = self._filename
        self._filename = None
        self.set_default_options()
        self._tasks.clear_tasks()
        self._tasks.load(filename=filename, current_tab=False)
        logger.debug("Updated legacy task and reset project")

    def _update_tasks(self) -> None:
        """Add the tasks from the loaded project to the :class:`Tasks` class."""
        assert self._filename is not None
        for key, val in self._cli_options.items():
            opts: dict[str, str | dict[str, bool | int | float | str]] = {key: val}
            opts["tab_name"] = key
            self._tasks.add_project_task(self._filename, key, opts)

    def reload(self, *args) -> None:  # pylint:disable=unused-argument
        """Reset all GUI's option tabs to their last saved state.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        """
        if self._options is None:
            logger.info("No active project to reload")
            return
        logger.debug("Reloading project")
        self._set_options()
        self._update_tasks()
        self._reset_modified_var()
        self._update_root_title()

    def _update_root_title(self) -> None:
        """Update the root Window title with the project name. Add a asterisk if the file is
        modified."""
        text = "<untitled project>" if self._basename is None else self._basename
        text += "*" if self._modified else ""
        self._config.set_root_title(text=text)

    def save(self, *args, save_as: bool = False) -> None:  # pylint:disable=unused-argument
        """Save the current GUI state to a ``.fsw`` project file.

        Parameters
        ----------
        *args: tuple
            Unused, but needs to be present for arguments passed by tkinter event handling
        save_as: bool, optional
            Whether to save to the stored filename, or pop open a file handler to ask for a
            location. If there is no stored filename, then a file handler will automatically be
            popped.
        """
        logger.debug("Saving config as...")

        save_as = save_as or self._filename is None
        if save_as and not self._save_as_to_filename("project"):
            return
        self._save()
        self._update_tasks()
        self._update_root_title()
        if not save_as:
            logger.info("Saved project to: '%s'", self._filename)
        else:
            logger.debug("Saved project to: '%s'", self._filename)

    def new(self, *args) -> None:  # pylint:disable=unused-argument
        """Create a new project with default options.

        Pops a file handler to select location.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        """
        logger.debug("Creating new project")
        if not self.confirm_close():
            logger.debug("Creating new project cancelled")
            return
        assert self._file_handler is not None
        cfg_file = self._file_handler("save",
                                      "config_project",
                                      title="New Project...",
                                      initial_folder=self._basename).return_file
        if not cfg_file:
            logger.debug("No filename selected")
            return
        self._filename = cfg_file.name
        cfg_file.close()

        self.set_default_options()
        self._config.cli_opts.reset()
        self._save()
        self._update_root_title()

    def close(self, *args) -> None:  # pylint:disable=unused-argument
        """Clear the current project and set all options to default.

        Parameters
        ----------
        *args
            Unused, but needs to be present for arguments passed by tkinter event handling
        """
        logger.debug("Close requested")
        if not self.confirm_close():
            logger.debug("Close cancelled")
            return
        self._config.cli_opts.reset()
        self._filename = None
        self.set_default_options()
        self._reset_modified_var()
        self._update_root_title()
        self._config.set_active_tab_by_name(cfg.tab())

    def confirm_close(self) -> bool:
        """Pop a message box to get confirmation that an unsaved project should be closed

        Returns
        -------
        ``True`` if user confirms close, ``False`` if user cancels close
        """
        if not self._modified:
            logger.debug("Project is not modified")
            return True
        confirm_txt = "You have unsaved changes.\n\nAre you sure you want to close the project?"
        if messagebox.askokcancel("Close", confirm_txt, default="cancel", icon="warning"):
            logger.debug("Close Cancelled")
            return True
        logger.debug("Close confirmed")
        return False


class LastSession(_GuiSession):
    """Faceswap Last Session handling.

    Faceswap :class:`LastSession` handles saving the state of the Faceswap GUI at close and
    reloading the state  at launch.

    Last Session behavior can be configured in :file:`config.gui.ini`.

    Parameters
    ----------
    config
        The master GUI config
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._filename = os.path.join(self._config.path_cache, ".last_session.json")
        if not self._enabled:
            return

        if cfg.autosave_last_session() == "prompt":
            self.ask_load()
        elif cfg.autosave_last_session() == "always":
            self.load()

    @property
    def _enabled(self) -> bool:
        """``True`` if autosave is enabled otherwise ``False``."""
        return cfg.autosave_last_session() != "never"

    def from_dict(self, options: dict[str, str | dict[str, bool | int | float | str]]) -> None:
        """Set the :attr:`_options` property based on the given options dictionary
        and update the GUI to use these values.

        This function is required for reloading the GUI state when the GUI has been force
        refreshed on a config change.

        Parameters
        ----------
        options
            The options to set. Should be the output of :func:`to_dict`
        """
        logger.debug("Setting options from dict: %s", options)
        self._options = options
        self._set_options()

    def to_dict(self) -> dict[str, str | dict[str, bool | int | float | str]] | None:
        """Collect the current GUI options and place them in a dict for retrieval or storage.

        This function is required for reloading the GUI state when the GUI has been force
        refreshed on a config change.

        Returns
        -------
        The current cli options ready for saving or retrieval by :func:`from_dict`
        """
        opts = T.cast(dict[str, str | dict[str, bool | int | float | str]],
                      self._current_gui_state())
        logger.debug("Collected opts: %s", opts)
        if not opts or opts == self._default_options:
            logger.debug("Default session, or no opts found. Not saving last session.")
            return None
        opts["tab_name"] = self._active_tab
        fname = self._config.project.filename
        assert fname is not None
        opts["project"] = fname
        logger.debug("Added project items: %s", {k: v for k, v in opts.items()
                                                 if k in ("tab_name", "project")})
        return opts

    def ask_load(self) -> None:
        """Pop a message box to ask the user if they wish to load their last session."""
        if not self._file_exists:
            logger.debug("No last session file found")
        elif messagebox.askyesno("Last Session", "Load last session?"):
            logger.debug("Loading last session at user request")
            self.load()
        else:
            logger.debug("Not loading last session at user request")
            logger.debug("Deleting LastSession file")
            assert self._filename is not None
            os.remove(self._filename)

    def load(self) -> None:
        """Load the last session.

        Loads the last saved session options. Checks if a previous project was loaded
        and whether there have been changes since the last saved version of the project.
        Sets the display and :class:`Project` and :class:`Task` objects accordingly."""
        loaded = self._load()
        if not loaded:
            return
        self._set_project()
        self._set_options()

    def _set_project(self) -> None:
        """Set the :class:`Project` if session is resuming from one. """
        assert self._options is not None
        if self._options.get("project", None) is None:
            logger.debug("No project stored")
        else:
            logger.debug("Loading stored project")
            fname = self._options["project"]
            assert isinstance(fname, str)
            self._config.project.load(filename=fname, last_session=True)

    def save(self) -> None:
        """Save a snapshot of currently set GUI config options.

        Called on Faceswap shutdown.
        """
        assert self._filename is not None
        if not self._enabled:
            logger.debug("LastSession not enabled")
            if os.path.exists(self._filename):
                logger.debug("Deleting existing LastSession file")
                os.remove(self._filename)
            return

        opts = self.to_dict()
        if opts is None and os.path.exists(self._filename):
            logger.debug("Last session default or blank. Clearing saved last session.")
            os.remove(self._filename)
        if opts is not None:
            assert self._serializer is not None
            self._serializer.save(self._filename, opts)  # pyright:ignore[reportCallIssue]
            logger.debug("Saved last session. (filename: '%s', opts: %s", self._filename, opts)


__all__ = get_module_objects(__name__)
