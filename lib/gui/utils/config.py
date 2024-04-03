#!/usr/bin python3
""" Global configuration optiopns for the Faceswap GUI """
from __future__ import annotations
import logging
import os
import sys
import tkinter as tk
import typing as T

from dataclasses import dataclass, field

from lib.gui._config import Config as UserConfig
from lib.gui.project import Project, Tasks
from lib.gui.theme import Style
from .file_handler import FileHandler

if T.TYPE_CHECKING:
    from lib.gui.options import CliOptions
    from lib.gui.custom_widgets import StatusBar
    from lib.gui.command import CommandNotebook
    from lib.gui.command import ToolsNotebook

logger = logging.getLogger(__name__)

PATHCACHE = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), "lib", "gui", ".cache")
_CONFIG: Config | None = None


def initialize_config(root: tk.Tk,
                      cli_opts: CliOptions | None,
                      statusbar: StatusBar | None) -> Config | None:
    """ Initialize the GUI Master :class:`Config` and add to global constant.

    This should only be called once on first GUI startup. Future access to :class:`Config`
    should only be executed through :func:`get_config`.

    Parameters
    ----------
    root: :class:`tkinter.Tk`
        The root Tkinter object
    cli_opts: :class:`lib.gui.options.CliOptions` or ``None``
        The command line options object. Must be provided for main GUI. Must be ``None`` for tools
    statusbar: :class:`lib.gui.custom_widgets.StatusBar` or ``None``
        The GUI Status bar. Must be provided for main GUI. Must be ``None`` for tools

    Returns
    -------
    :class:`Config` or ``None``
        ``None`` if the config has already been initialized otherwise the global configuration
        options
    """
    global _CONFIG  # pylint:disable=global-statement
    if _CONFIG is not None:
        return None
    logger.debug("Initializing config: (root: %s, cli_opts: %s, "
                 "statusbar: %s)", root, cli_opts, statusbar)
    _CONFIG = Config(root, cli_opts, statusbar)
    return _CONFIG


def get_config() -> "Config":
    """ Get the Master GUI configuration.

    Returns
    -------
    :class:`Config`
        The Master GUI Config
    """
    assert _CONFIG is not None
    return _CONFIG


class GlobalVariables():
    """ Global tkinter variables accessible from all parts of the GUI. Should only be accessed from
    :attr:`get_config().tk_vars` """
    def __init__(self) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._display = tk.StringVar()
        self._running_task = tk.BooleanVar()
        self._is_training = tk.BooleanVar()
        self._action_command = tk.StringVar()
        self._generate_command = tk.StringVar()
        self._console_clear = tk.BooleanVar()
        self._refresh_graph = tk.BooleanVar()
        self._analysis_folder = tk.StringVar()

        self._initialize_variables()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def display(self) -> tk.StringVar:
        """ :class:`tkinter.StringVar`: The current Faceswap command running """
        return self._display

    @property
    def running_task(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: ``True`` if a Faceswap task is running otherwise
        ``False`` """
        return self._running_task

    @property
    def is_training(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: ``True`` if Faceswap is currently training otherwise
        ``False`` """
        return self._is_training

    @property
    def action_command(self) -> tk.StringVar:
        """ :class:`tkinter.StringVar`: The command line action to perform """
        return self._action_command

    @property
    def generate_command(self) -> tk.StringVar:
        """ :class:`tkinter.StringVar`: The command line action to generate """
        return self._generate_command

    @property
    def console_clear(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`: ``True`` if the console should be cleared otherwise
        ``False`` """
        return self._console_clear

    @property
    def refresh_graph(self) -> tk.BooleanVar:
        """ :class:`tkinter.BooleanVar`:  ``True`` if the training graph should be refreshed
        otherwise ``False`` """
        return self._refresh_graph

    @property
    def analysis_folder(self) -> tk.StringVar:
        """ :class:`tkinter.StringVar`: Full path the analysis folder"""
        return self._analysis_folder

    def _initialize_variables(self) -> None:
        """ Initialize the default variable values"""
        self._display.set("")
        self._running_task.set(False)
        self._is_training.set(False)
        self._action_command.set("")
        self._generate_command.set("")
        self._console_clear.set(False)
        self._refresh_graph.set(False)
        self._analysis_folder.set("")


@dataclass
class _GuiObjects:
    """ Data class for commonly accessed GUI Objects """
    cli_opts: CliOptions | None
    tk_vars: GlobalVariables
    project: Project
    tasks: Tasks
    status_bar: StatusBar | None
    default_options: dict[str, dict[str, T.Any]] = field(default_factory=dict)
    command_notebook: CommandNotebook | None = None


class Config():
    """ The centralized configuration class for holding items that should be made available to all
    parts of the GUI.

    This class should be initialized on GUI startup through :func:`initialize_config`. Any further
    access to this class should be through :func:`get_config`.

    Parameters
    ----------
    root: :class:`tkinter.Tk`
        The root Tkinter object
    cli_opts: :class:`lib.gui.options.CliOptions` or ``None``
        The command line options object. Must be provided for main GUI. Must be ``None`` for tools
    statusbar: :class:`lib.gui.custom_widgets.StatusBar` or ``None``
        The GUI Status bar. Must be provided for main GUI. Must be ``None`` for tools
    """
    def __init__(self,
                 root: tk.Tk,
                 cli_opts: CliOptions | None,
                 statusbar: StatusBar | None) -> None:
        logger.debug("Initializing %s: (root %s, cli_opts: %s, statusbar: %s)",
                     self.__class__.__name__, root, cli_opts, statusbar)
        self._default_font = T.cast(dict,
                                    tk.font.nametofont("TkDefaultFont").configure())["family"]
        self._constants = {"root": root,
                           "scaling_factor": self._get_scaling(root),
                           "default_font": self._default_font}
        self._gui_objects = _GuiObjects(
            cli_opts=cli_opts,
            tk_vars=GlobalVariables(),
            project=Project(self, FileHandler),
            tasks=Tasks(self, FileHandler),
            status_bar=statusbar)

        self._user_config = UserConfig(None)
        self._style = Style(self.default_font, root, PATHCACHE)
        self._user_theme = self._style.user_theme
        logger.debug("Initialized %s", self.__class__.__name__)

    # Constants
    @property
    def root(self) -> tk.Tk:
        """ :class:`tkinter.Tk`: The root tkinter window. """
        return self._constants["root"]

    @property
    def scaling_factor(self) -> float:
        """ float: The scaling factor for current display. """
        return self._constants["scaling_factor"]

    @property
    def pathcache(self) -> str:
        """ str: The path to the GUI cache folder """
        return PATHCACHE

    # GUI Objects
    @property
    def cli_opts(self) -> CliOptions:
        """ :class:`lib.gui.options.CliOptions`: The command line options for this GUI Session. """
        # This should only be None when a separate tool (not main GUI) is used, at which point
        # cli_opts do not exist
        assert self._gui_objects.cli_opts is not None
        return self._gui_objects.cli_opts

    @property
    def tk_vars(self) -> GlobalVariables:
        """ dict: The global tkinter variables. """
        return self._gui_objects.tk_vars

    @property
    def project(self) -> Project:
        """ :class:`lib.gui.project.Project`: The project session handler. """
        return self._gui_objects.project

    @property
    def tasks(self) -> Tasks:
        """ :class:`lib.gui.project.Tasks`: The session tasks handler. """
        return self._gui_objects.tasks

    @property
    def default_options(self) -> dict[str, dict[str, T.Any]]:
        """ dict: The default options for all tabs """
        return self._gui_objects.default_options

    @property
    def statusbar(self) -> StatusBar:
        """ :class:`lib.gui.custom_widgets.StatusBar`: The GUI StatusBar
        :class:`tkinter.ttk.Frame`. """
        # This should only be None when a separate tool (not main GUI) is used, at which point
        # this statusbar does not exist
        assert self._gui_objects.status_bar is not None
        return self._gui_objects.status_bar

    @property
    def command_notebook(self) -> CommandNotebook | None:
        """ :class:`lib.gui.command.CommandNotebook`: The main Faceswap Command Notebook. """
        return self._gui_objects.command_notebook

    # Convenience GUI Objects
    @property
    def tools_notebook(self) -> ToolsNotebook:
        """ :class:`lib.gui.command.ToolsNotebook`: The Faceswap Tools sub-Notebook. """
        assert self.command_notebook is not None
        return self.command_notebook.tools_notebook

    @property
    def modified_vars(self) -> dict[str, tk.BooleanVar]:
        """ dict: The command notebook modified tkinter variables. """
        assert self.command_notebook is not None
        return self.command_notebook.modified_vars

    @property
    def _command_tabs(self) -> dict[str, int]:
        """ dict: Command tab titles with their IDs. """
        assert self.command_notebook is not None
        return self.command_notebook.tab_names

    @property
    def _tools_tabs(self) -> dict[str, int]:
        """ dict: Tools command tab titles with their IDs. """
        assert self.command_notebook is not None
        return self.command_notebook.tools_tab_names

    # Config
    @property
    def user_config(self) -> UserConfig:
        """ dict: The GUI config in dict form. """
        return self._user_config

    @property
    def user_config_dict(self) -> dict[str, T.Any]:  # TODO Dataclass
        """ dict: The GUI config in dict form. """
        return self._user_config.config_dict

    @property
    def user_theme(self) -> dict[str, T.Any]:  # TODO Dataclass
        """ dict: The GUI theme selection options. """
        return self._user_theme

    @property
    def default_font(self) -> tuple[str, int]:
        """ tuple: The selected font as configured in user settings. First item is the font (`str`)
        second item the font size (`int`). """
        font = self.user_config_dict["font"]
        font = self._default_font if font == "default" else font
        return (font, self.user_config_dict["font_size"])

    @staticmethod
    def _get_scaling(root) -> float:
        """ Get the display DPI.

        Returns
        -------
        float:
            The scaling factor
        """
        dpi = root.winfo_fpixels("1i")
        scaling = dpi / 72.0
        logger.debug("dpi: %s, scaling: %s'", dpi, scaling)
        return scaling

    def set_default_options(self) -> None:
        """ Set the default options for :mod:`lib.gui.projects`

        The Default GUI options are stored on Faceswap startup.

        Exposed as the :attr:`_default_opts` for a project cannot be set until after the main
        Command Tabs have been loaded.
        """
        default = self.cli_opts.get_option_values()
        logger.debug(default)
        self._gui_objects.default_options = default
        self.project.set_default_options()

    def set_command_notebook(self, notebook: CommandNotebook) -> None:
        """ Set the command notebook to the :attr:`command_notebook` attribute
        and enable the modified callback for :attr:`project`.

        Parameters
        ----------
        notebook: :class:`lib.gui.command.CommandNotebook`
            The main command notebook for the Faceswap GUI
        """
        logger.debug("Setting commane notebook: %s", notebook)
        self._gui_objects.command_notebook = notebook
        self.project.set_modified_callback()

    def set_active_tab_by_name(self, name: str) -> None:
        """ Sets the :attr:`command_notebook` or :attr:`tools_notebook` to active based on given
        name.

        Parameters
        ----------
        name: str
            The name of the tab to set active
        """
        assert self.command_notebook is not None
        name = name.lower()
        if name in self._command_tabs:
            tab_id = self._command_tabs[name]
            logger.debug("Setting active tab to: (name: %s, id: %s)", name, tab_id)
            self.command_notebook.select(tab_id)
        elif name in self._tools_tabs:
            self.command_notebook.select(self._command_tabs["tools"])
            tab_id = self._tools_tabs[name]
            logger.debug("Setting active Tools tab to: (name: %s, id: %s)", name, tab_id)
            self.tools_notebook.select()
        else:
            logger.debug("Name couldn't be found. Setting to id 0: %s", name)
            self.command_notebook.select(0)

    def set_modified_true(self, command: str) -> None:
        """ Set the modified variable to ``True`` for the given command in :attr:`modified_vars`.

        Parameters
        ----------
        command: str
            The command to set the modified state to ``True``

        """
        tkvar = self.modified_vars.get(command, None)
        if tkvar is None:
            logger.debug("No tkvar for command: '%s'", command)
            return
        tkvar.set(True)
        logger.debug("Set modified var to True for: '%s'", command)

    def refresh_config(self) -> None:
        """ Reload the user config from file. """
        self._user_config = UserConfig(None)

    def set_cursor_busy(self, widget: tk.Widget | None = None) -> None:
        """ Set the root or widget cursor to busy.

        Parameters
        ----------
        widget: tkinter object, optional
            The widget to set busy cursor for. If the provided value is ``None`` then sets the
            cursor busy for the whole of the GUI. Default: ``None``.
        """
        logger.debug("Setting cursor to busy. widget: %s", widget)
        component = self.root if widget is None else widget
        component.config(cursor="watch")  # type: ignore
        component.update_idletasks()

    def set_cursor_default(self, widget: tk.Widget | None = None) -> None:
        """ Set the root or widget cursor to default.

        Parameters
        ----------
        widget: tkinter object, optional
            The widget to set default cursor for. If the provided value is ``None`` then sets the
            cursor busy for the whole of the GUI. Default: ``None``
        """
        logger.debug("Setting cursor to default. widget: %s", widget)
        component = self.root if widget is None else widget
        component.config(cursor="")  # type: ignore
        component.update_idletasks()

    def set_root_title(self, text: str | None = None) -> None:
        """ Set the main title text for Faceswap.

        The title will always begin with 'Faceswap.py'. Additional text can be appended.

        Parameters
        ----------
        text: str, optional
            Additional text to be appended to the GUI title bar. Default: ``None``
        """
        title = "Faceswap.py"
        title += f" - {text}" if text is not None and text else ""
        self.root.title(title)

    def set_geometry(self, width: int, height: int, fullscreen: bool = False) -> None:
        """ Set the geometry for the root tkinter object.

        Parameters
        ----------
        width: int
            The width to set the window to (prior to scaling)
        height: int
            The height to set the window to (prior to scaling)
        fullscreen: bool, optional
            Whether to set the window to full-screen mode. If ``True`` then :attr:`width` and
            :attr:`height` are ignored. Default: ``False``
        """
        self.root.tk.call("tk", "scaling", self.scaling_factor)
        if fullscreen:
            initial_dimensions = (self.root.winfo_screenwidth(), self.root.winfo_screenheight())
        else:
            initial_dimensions = (round(width * self.scaling_factor),
                                  round(height * self.scaling_factor))

        if fullscreen and sys.platform in ("win32", "darwin"):
            self.root.state('zoomed')
        elif fullscreen:
            self.root.attributes('-zoomed', True)
        else:
            self.root.geometry(f"{str(initial_dimensions[0])}x{str(initial_dimensions[1])}+80+80")
        logger.debug("Geometry: %sx%s", *initial_dimensions)
