#!/usr/bin/env python3
"""Manages the widgets that hold the bottom 'control' area of the preview tool."""
from __future__ import annotations
import gettext
import logging
import typing as T

import tkinter as tk

from tkinter import ttk

from lib.gui.custom_widgets import Tooltip
from lib.gui.control_helper import ControlPanel, ControlPanelOption
from lib.logger import parse_class_init
from lib.gui.utils import get_images
from lib.utils import get_module_objects
from plugins.plugin_loader import PluginLoader
from plugins.convert import convert_config

if T.TYPE_CHECKING:
    from collections.abc import Callable
    from .preview import Preview

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("tools.preview", localedir="locales", fallback=True)
_ = _LANG.gettext


class ConfigTools():
    """Tools for loading, saving, setting and retrieving configuration file values.

    Parameters
    ----------
    config_file
        Path to a custom config .ini file or ``None`` to load the default config file

    Attributes
    ----------
    tk_vars
        Global tkinter variables. `Refresh` and `Busy` :class:`tkinter.BooleanVar`
    """
    def __init__(self, config_file: str | None) -> None:
        logger.debug(parse_class_init(locals()))
        self._config = convert_config.load_config(config_file=config_file)
        self.tk_vars: dict[str, dict[str, tk.Variable]] = {}
        self._config_dicts = self._get_config_dicts()  # Holds currently saved config

    @property
    def config_dicts(self) -> dict[str, dict[str, ControlPanelOption]]:
        """The convert configuration options in dictionary form."""
        return self._config_dicts

    @property
    def sections(self) -> list[str]:
        """The sorted section names that exist within the convert Configuration options."""
        return sorted(set(sect.split(".")[0] for sect in self._config.sections
                          if sect.split(".")[0] != "writer"))

    @property
    def plugins_dict(self) -> dict[str, list[str]]:
        """Dictionary of configuration option sections as key with a list of containing plugin
        names as the value"""
        return {section: sorted([sect.split(".")[1] for sect in self._config.sections
                                 if sect.split(".")[0] == section])
                for section in self.sections}

    def _get_config_dicts(self) -> dict[str, dict[str, ControlPanelOption]]:
        """Obtain a custom configuration dictionary for convert configuration items in use
        by the preview tool formatted for control helper.

        Returns
        -------
        Each configuration section as keys, with the values as a dict of option_name to
        :class:`lib.gui.control_helper.ControlOption`."""
        logger.debug("Formatting Config for GUI")
        config_dicts: dict[str, dict[str, ControlPanelOption]] = {}
        for section_name, section in self._config.sections.items():
            if section_name.startswith("writer."):
                continue
            cp_options: dict[str, ControlPanelOption] = {}
            for option_name, option in section.options.items():
                cp_option = ControlPanelOption.from_config_object(option_name, option)
                cp_options[option_name] = cp_option
                self.tk_vars.setdefault(section_name, {})[option_name] = cp_option.tk_var
            config_dicts[section_name] = cp_options
        logger.debug("Formatted Config for GUI: %s", config_dicts)
        return config_dicts

    def update_config(self) -> None:
        """Update :attr:`config` with the currently selected values from the GUI."""
        for section, options in self.tk_vars.items():
            for option_name, tk_option in options.items():
                try:
                    new_value = tk_option.get()
                except tk.TclError as err:
                    # When manually filling in text fields, blank values will
                    # raise an error on numeric data types so return 0
                    logger.trace(  # type:ignore[attr-defined]
                        "Error getting value. Defaulting to 0. Error: %s", str(err))
                    new_value = "" if isinstance(tk_option, tk.StringVar) else 0
                option = self._config.sections[section].options[option_name]
                old_value = option.value
                if new_value == old_value or (isinstance(old_value, list) and
                                              set(str(new_value).split()) == set(old_value)):
                    logger.trace("Skipping unchanged option '%s'",  # type:ignore[attr-defined]
                                 option_name)
                logger.debug("Updating config: '%s', '%s' from %s to %s",
                             section, option_name, repr(old_value), repr(new_value))
                option.set(new_value)

    def reset_config_to_saved(self, section: str | None = None) -> None:
        """Reset the GUI parameters to their saved values within the configuration file.

        Parameters
        ----------
        section
            The configuration section to reset the values for, If ``None`` provided then all
            sections are reset. Default: ``None``
        """
        logger.debug("Resetting to saved config: %s", section)
        sections = [section] if section is not None else list(self.tk_vars.keys())
        for section_name in sections:
            for option_name, tk_option in self._config_dicts[section_name].items():
                val = tk_option.value
                if val != self.tk_vars[section_name][option_name].get():
                    self.tk_vars[section_name][option_name].set(val)
                    logger.debug("Setting '%s' - '%s' to saved value %s",
                                 section_name, option_name, repr(val))
        logger.debug("Reset to saved config: %s", section)

    def reset_config_to_default(self, section: str | None = None) -> None:
        """Reset the GUI parameters to their default configuration values.

        Parameters
        ----------
        section
            The configuration section to reset the values for, If ``None`` provided then all
            sections are reset. Default: ``None``
        """
        logger.debug("Resetting to default: %s", section)
        sections = [section] if section is not None else list(self.tk_vars.keys())
        for section_name in sections:
            for option_name, options in self._config_dicts[section_name].items():
                default = options.default
                if default != self.tk_vars[section_name][option_name].get():
                    self.tk_vars[section_name][option_name].set(default)
                    logger.debug("Setting '%s' - '%s' to default value %s",
                                 section_name, option_name, repr(default))
        logger.debug("Reset to default: %s", section)

    def save_config(self, section: str | None = None) -> None:
        """Save the configuration ``.ini`` file with the currently stored values.

        Parameters
        ----------
        section
            The configuration section to save, If ``None`` provided then all sections are saved.
            Default: ``None``
        """
        logger.debug("Saving %s config", section)

        for section_name, sect in self._config.sections.items():
            if section_name not in self._config_dicts:
                logger.debug("[%s] Skipping section not in local config", section_name)
                continue
            if section is not None and section_name != section:
                logger.debug("[%s] Skipping section not selected for saving", section_name)
                continue
            for option_name, option in sect.options.items():
                new_opt = self.tk_vars[section_name][option_name].get()
                fmt_opt = str(new_opt).split() if isinstance(option.value, list) else new_opt
                logger.debug("[%s] Setting '%s' to %s", section_name, option_name, repr(fmt_opt))
                option.set(new_opt)

        self._config.save_config()


class BusyProgressBar():
    """An infinite progress bar for when a thread is running to swap/patch a group of samples"""
    def __init__(self, parent: ttk.Frame) -> None:
        self._progress_bar = self._add_busy_indicator(parent)

    def _add_busy_indicator(self, parent: ttk.Frame) -> ttk.Progressbar:
        """Place progress bar into bottom bar to indicate when processing.

        Parameters
        ----------
        parent
            The tkinter object that holds the busy indicator

        Returns
        -------
        A Progress bar to indicate that the Preview tool is busy
        """
        logger.debug("Placing busy indicator")
        pbar = ttk.Progressbar(parent, mode="indeterminate")
        pbar.pack(side=tk.LEFT)
        pbar.pack_forget()
        return pbar

    def stop(self) -> None:
        """Stop and hide progress bar"""
        logger.debug("Stopping busy indicator")
        if not self._progress_bar.winfo_ismapped():
            logger.debug("busy indicator already hidden")
            return
        self._progress_bar.stop()
        self._progress_bar.pack_forget()

    def start(self) -> None:
        """Start and display progress bar"""
        logger.debug("Starting busy indicator")
        if self._progress_bar.winfo_ismapped():
            logger.debug("busy indicator already started")
            return

        self._progress_bar.pack(side=tk.LEFT, padx=5, pady=(5, 10), fill=tk.X, expand=True)
        self._progress_bar.start(25)


class ActionFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """Frame that holds the left hand side options panel containing the command line options.

    Parameters
    ----------
    app
        The main tkinter Preview app
    parent
        The parent tkinter object that holds the Action Frame
    """
    def __init__(self, app: Preview, parent: ttk.Frame) -> None:
        logger.debug("Initializing %s: (app: %s, parent: %s)",
                     self.__class__.__name__, app, parent)
        self._app = app

        super().__init__(parent)
        self.pack(side=tk.LEFT, anchor=tk.N, fill=tk.Y)
        self._tk_vars: dict[str, tk.Variable] = {}

        self._options = {
            "color": app._patch.converter.cli_arguments.color_adjustment.replace("-", "_"),
            "mask_type": app._patch.converter.cli_arguments.mask_type.replace("-", "_"),
            "face_scale": app._patch.converter.cli_arguments.face_scale}
        defaults = {opt: self._format_to_display(val) if opt != "face_scale" else val
                    for opt, val in self._options.items()}
        self._busy_bar = self._build_frame(defaults,
                                           app._samples.generate,
                                           app._refresh,
                                           app._samples.available_masks,
                                           app._samples.predictor.has_predicted_mask)

    @property
    def convert_args(self) -> dict[str, T.Any]:
        """Currently selected Command line arguments from the :class:`ActionFrame`."""
        retval = {opt if opt != "color" else "color_adjustment":
                  self._format_from_display(self._tk_vars[opt].get())
                  for opt in self._options if opt != "face_scale"}
        retval["face_scale"] = self._tk_vars["face_scale"].get()
        return retval

    @property
    def busy_progress_bar(self) -> BusyProgressBar:
        """The progress bar on the left hand side whilst a swap/patch is being applied."""
        return self._busy_bar

    @staticmethod
    def _format_from_display(var: str) -> str:
        """Format a variable from the display version to the command line action version.

        Parameters
        ----------
        var
            The variable name to format

        Returns
        -------
        The formatted variable name
        """
        return var.replace(" ", "_").lower()

    @staticmethod
    def _format_to_display(var: str) -> str:
        """Format a variable from the command line action version to the display version.

        Parameters
        ----------
        var
            The variable name to format

        Returns
        -------
        The formatted variable name
        """
        return var.replace("_", " ").replace("-", " ").title()

    def _build_frame(self,
                     defaults: dict[str, T.Any],
                     refresh_callback: Callable[[], None],
                     patch_callback: Callable[[], None],
                     available_masks: list[str],
                     has_predicted_mask: bool) -> BusyProgressBar:
        """Build the :class:`ActionFrame`.

        Parameters
        ----------
        defaults
            The default command line options
        patch_callback
            The function to execute when a patch callback is received
        refresh_callback
            The function to execute when a refresh callback is received
        available_masks
            The available masks that exist within the alignments file
        has_predicted_mask
            Whether the model was trained with a mask

        Returns
        -------
        A Progress bar to indicate that the Preview tool is busy
        """
        logger.debug("Building Action frame")

        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, anchor=tk.S)
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, anchor=tk.N, expand=True)

        self._add_cli_choices(top_frame, defaults, available_masks, has_predicted_mask)

        busy_indicator = BusyProgressBar(bottom_frame)
        self._add_refresh_button(bottom_frame, refresh_callback)
        self._add_patch_callback(patch_callback)
        self._add_actions(bottom_frame)
        logger.debug("Built Action frame")
        return busy_indicator

    def _add_cli_choices(self,
                         parent: ttk.Frame,
                         defaults: dict[str, T.Any],
                         available_masks: list[str],
                         has_predicted_mask: bool) -> None:
        """Create :class:`lib.gui.control_helper.ControlPanel` object for the command line options.

        parent
            The frame to hold the command line choices
        defaults
            The default command line options
        available_masks
            The available masks that exist within the alignments file
        has_predicted_mask
            Whether the model was trained with a mask
        """
        cp_options = self._get_control_panel_options(defaults, available_masks, has_predicted_mask)
        panel_kwargs = {"blank_nones": False, "label_width": 10, "style": "CPanel"}
        ControlPanel(parent, cp_options, header_text=None, **panel_kwargs)

    def _get_control_panel_options(self,
                                   defaults: dict[str, T.Any],
                                   available_masks: list[str],
                                   has_predicted_mask: bool) -> list[ControlPanelOption]:
        """Create :class:`lib.gui.control_helper.ControlPanelOption` objects for the cli options.

        defaults
            The default command line options
        available_masks
            The available masks that exist within the alignments file
        has_predicted_mask
            Whether the model was trained with a mask

        Returns
        -------
        The list of `lib.gui.control_helper.ControlPanelOption` objects for the Action Frame
        """
        cp_options: list[ControlPanelOption] = []
        for opt in self._options:
            if opt == "face_scale":
                cp_option = ControlPanelOption(title=opt,
                                               dtype=float,
                                               default=0.0,
                                               rounding=2,
                                               min_max=(-10., 10.),
                                               group="Command Line Choices")
            else:
                if opt == "mask_type":
                    choices = self._create_mask_choices(defaults,
                                                        available_masks,
                                                        has_predicted_mask)
                else:
                    choices = PluginLoader.get_available_convert_plugins(opt, True)
                cp_option = ControlPanelOption(title=opt,
                                               dtype=str,
                                               default=defaults[opt],
                                               initial_value=defaults[opt],
                                               choices=choices,
                                               group="Command Line Choices",
                                               is_radio=False)
            self._tk_vars[opt] = cp_option.tk_var
            cp_options.append(cp_option)
        return cp_options

    def _create_mask_choices(self,
                             defaults: dict[str, T.Any],
                             available_masks: list[str],
                             has_predicted_mask: bool) -> list[str]:
        """Set the mask choices and default mask based on available masks.

        Parameters
        ----------
        defaults
            The default command line options
        available_masks
            The available masks that exist within the alignments file
        has_predicted_mask
            Whether the model was trained with a mask

        Returns
        -------
        list
            The masks that are available to use from the alignments file
        """
        logger.debug("Initial mask choices: %s", available_masks)
        available_masks += ["components", "extended"]
        if has_predicted_mask:
            available_masks += ["predicted"]
        if "none" not in available_masks:
            available_masks += ["none"]
        if self._format_from_display(defaults["mask_type"]) not in available_masks:
            logger.debug("Setting default mask to first available: %s", available_masks[0])
            defaults["mask_type"] = available_masks[0]
        logger.debug("Final mask choices: %s", available_masks)
        return available_masks

    @classmethod
    def _add_refresh_button(cls,
                            parent: ttk.Frame,
                            refresh_callback: Callable[[], None]) -> None:
        """Add a button to refresh the images.

        Parameters
        ----------
        refresh_callback
            The function to execute when the refresh button is pressed
        """
        btn = ttk.Button(parent, text="Update Samples", command=refresh_callback)
        btn.pack(padx=5, pady=5, side=tk.TOP, fill=tk.X, anchor=tk.N)

    def _add_patch_callback(self, patch_callback: Callable[[], None]) -> None:
        """Add callback to re-patch images on action option change.

        Parameters
        ----------
        patch_callback
            The function to execute when the images require patching
        """
        for tk_var in self._tk_vars.values():
            tk_var.trace("w", patch_callback)

    def _add_actions(self, parent: ttk.Frame) -> None:
        """Add Action Buttons to the :class:`ActionFrame`.

        Parameters
        ----------
        parent
            The tkinter object that holds the action buttons
        """
        logger.debug("Adding util buttons")
        frame = ttk.Frame(parent)
        frame.pack(padx=5, pady=(5, 10), side=tk.RIGHT, fill=tk.X, anchor=tk.E)
        text = ""
        action: T.Callable[[], T.Any] | None = None
        for utl in ("save", "clear", "reload"):
            logger.debug("Adding button: '%s'", utl)
            img = get_images().icons[utl]
            if utl == "save":
                text = _("Save full config")
                action = self._app.config_tools.save_config
            elif utl == "clear":
                text = _("Reset full config to default values")
                action = self._app.config_tools.reset_config_to_default
            elif utl == "reload":
                text = _("Reset full config to saved values")
                action = self._app.config_tools.reset_config_to_saved

            assert action is not None
            btnutl = ttk.Button(frame,
                                image=img,  # type:ignore[arg-type]
                                command=action)
            btnutl.pack(padx=2, side=tk.RIGHT)
            Tooltip(btnutl, text=text, wrap_length=200)
        logger.debug("Added util buttons")


class OptionsBook(ttk.Notebook):  # pylint:disable=too-many-ancestors

    """The notebook that holds the Convert configuration options.

    Parameters
    ----------
    parent
        The parent tkinter object that holds the Options book
    config_tools
        Tools for loading and saving configuration files
    patch_callback
        The function to execute when a patch callback is received

    Attributes
    ----------
    config_tools
        Tools for loading and saving configuration files
    """
    def __init__(self,
                 parent: ttk.Frame,
                 config_tools: ConfigTools,
                 patch_callback: Callable[[], None]) -> None:
        logger.debug("Initializing %s: (parent: %s, config: %s)",
                     self.__class__.__name__, parent, config_tools)
        super().__init__(parent)
        self.pack(side=tk.RIGHT, anchor=tk.N, fill=tk.BOTH, expand=True)
        self.config_tools = config_tools

        self._tabs: dict[str, dict[str, ttk.Notebook | ConfigFrame]] = {}
        self._build_tabs()
        self._build_sub_tabs()
        self._add_patch_callback(patch_callback)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _build_tabs(self) -> None:
        """Build the notebook tabs for the each configuration section."""
        logger.debug("Build Tabs")
        for section in self.config_tools.sections:
            tab = ttk.Notebook(self)
            self._tabs[section] = {"tab": tab}
            self.add(tab, text=section.replace("_", " ").title())

    def _build_sub_tabs(self) -> None:
        """Build the notebook sub tabs for each convert section's plugin."""
        for section, plugins in self.config_tools.plugins_dict.items():
            for plugin in plugins:
                config_key = ".".join((section, plugin))
                config_dict = self.config_tools.config_dicts[config_key]
                tab = ConfigFrame(self, config_key, config_dict)
                self._tabs[section][plugin] = tab
                text = plugin.replace("_", " ").title()
                T.cast(ttk.Notebook, self._tabs[section]["tab"]).add(tab, text=text)

    def _add_patch_callback(self, patch_callback: Callable[[], None]) -> None:
        """Add callback to re-patch images on configuration option change.

        Parameters
        ----------
        patch_callback
            The function to execute when the images require patching
        """
        for plugins in self.config_tools.tk_vars.values():
            for tk_var in plugins.values():
                tk_var.trace("w", patch_callback)


class ConfigFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """Holds the configuration options for a convert plugin inside the :class:`OptionsBook`.

    Parameters
    ----------
    parent
        The tkinter object that will hold this configuration frame
    config_key
        The section/plugin key for these configuration options
    options
        The options for this section/plugin
    """

    def __init__(self,
                 parent: OptionsBook,
                 config_key: str,
                 options: dict[str, T.Any]):
        logger.debug("Initializing %s", self.__class__.__name__)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._options = options

        self._action_frame = ttk.Frame(self)
        self._action_frame.pack(padx=0, pady=(0, 5), side=tk.BOTTOM, fill=tk.X, anchor=tk.E)
        self._add_frame_separator()

        self._build_frame(parent, config_key)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _build_frame(self, parent: OptionsBook, config_key: str) -> None:
        """Build the options frame for this command.

        Parameters
        ----------
        parent
            The tkinter object that will hold this configuration frame
        config_key
            The section/plugin key for these configuration options
        """
        logger.debug("Add Config Frame")
        panel_kwargs = {"columns": 2, "option_columns": 2, "blank_nones": False, "style": "CPanel"}
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        cp_options = [opt for key, opt in self._options.items() if key != "helptext"]
        ControlPanel(frame, cp_options, header_text=None, **panel_kwargs)
        self._add_actions(parent, config_key)
        logger.debug("Added Config Frame")

    def _add_frame_separator(self) -> None:
        """Add a separator between top and bottom frames."""
        logger.debug("Add frame separator")
        sep = ttk.Frame(self._action_frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=5, side=tk.TOP)
        logger.debug("Added frame separator")

    def _add_actions(self, parent: OptionsBook, config_key: str) -> None:
        """Add Action Buttons.

        Parameters
        ----------
        parent
            The tkinter object that will hold this configuration frame
        config_key
            The section/plugin key for these configuration options
        """
        logger.debug("Adding util buttons")

        title = config_key.split(".")[1].replace("_", " ").title()
        btn_frame = ttk.Frame(self._action_frame)
        btn_frame.pack(padx=5, side=tk.BOTTOM, fill=tk.X)
        text = ""
        action = None
        for utl in ("save", "clear", "reload"):
            logger.debug("Adding button: '%s'", utl)
            img = get_images().icons[utl]
            if utl == "save":
                text = _(f"Save {title} config")
                action = parent.config_tools.save_config
            elif utl == "clear":
                text = _(f"Reset {title} config to default values")
                action = parent.config_tools.reset_config_to_default
            elif utl == "reload":
                text = _(f"Reset {title} config to saved values")
                action = parent.config_tools.reset_config_to_saved

            btnutl = ttk.Button(btn_frame,
                                image=img,  # type:ignore[arg-type]
                                command=lambda cmd=action: cmd(config_key))  # type:ignore[misc]
            btnutl.pack(padx=2, side=tk.RIGHT)
            Tooltip(btnutl, text=text, wrap_length=200)
        logger.debug("Added util buttons")


__all__ = get_module_objects(__name__)
