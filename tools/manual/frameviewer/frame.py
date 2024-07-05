#!/usr/bin/env python3
""" The frame viewer section of the manual tool GUI """
import gettext
import logging
import tkinter as tk
from tkinter import ttk, TclError

from functools import partial
from time import time

from lib.gui.control_helper import set_slider_rounding
from lib.gui.custom_widgets import Tooltip
from lib.gui.utils import get_images

from .control import Navigation, BackgroundImage
from .editor import (BoundingBox, ExtractBox, Landmarks, Mask,  # noqa pylint:disable=unused-import
                     Mesh, View)

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext


class DisplayFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The main video display frame (top left section of GUI).

    Parameters
    ----------
    parent: :class:`ttk.PanedWindow`
        The paned window that the display frame resides in
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    detected_faces: :class:`tools.manual.detected_faces.DetectedFaces`
        The detected faces stored in the alignments file
    """
    def __init__(self, parent, tk_globals, detected_faces):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, detected_faces: %s)",
                     self.__class__.__name__, parent, tk_globals, detected_faces)
        super().__init__(parent)

        self._globals = tk_globals
        self._det_faces = detected_faces
        self._optional_widgets = {}

        self._actions_frame = ActionsFrame(self)
        main_frame = ttk.Frame(self)

        self._transport_frame = ttk.Frame(main_frame)
        self._nav = self._add_nav()
        self._navigation = Navigation(self)
        self._buttons = self._add_transport()
        self._add_transport_tk_trace()

        video_frame = ttk.Frame(main_frame)
        video_frame.bind("<Configure>", self._resize)

        self._canvas = FrameViewer(video_frame,
                                   self._globals,
                                   self._det_faces,
                                   self._actions_frame.actions,
                                   self._actions_frame.tk_selected_action)

        self._actions_frame.add_optional_buttons(self.editors)

        self._transport_frame.pack(side=tk.BOTTOM, padx=5, fill=tk.X)
        video_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        main_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.pack(side=tk.LEFT, anchor=tk.NW, expand=True, fill=tk.BOTH)

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _helptext(self):
        """ dict: {`name`: `help text`} Helptext lookup for navigation buttons """
        return {
            "play": _("Play/Pause (SPACE)"),
            "beginning": _("Go to First Frame (HOME)"),
            "prev": _("Go to Previous Frame (Z)"),
            "next": _("Go to Next Frame (X)"),
            "end": _("Go to Last Frame (END)"),
            "extract": _("Extract the faces to a folder... (Ctrl+E)"),
            "save": _("Save the Alignments file (Ctrl+S)"),
            "mode": _("Filter Frames to only those Containing the Selected Item (F)"),
            "distance": _("Set the distance from an 'average face' to be considered misaligned. "
                          "Higher distances are more restrictive")}

    @property
    def _btn_action(self):
        """ dict: {`name`: `action`} Command lookup for navigation buttons """
        actions = {"play": self._navigation.handle_play_button,
                   "beginning": self._navigation.goto_first_frame,
                   "prev": self._navigation.decrement_frame,
                   "next": self._navigation.increment_frame,
                   "end": self._navigation.goto_last_frame,
                   "extract": self._det_faces.extract,
                   "save": self._det_faces.save}
        return actions

    @property
    def tk_selected_action(self):
        """ :class:`tkinter.StringVar`: The variable holding the currently selected action """
        return self._actions_frame.tk_selected_action

    @property
    def active_editor(self):
        """ :class:`Editor`: The current editor in use based on :attr:`selected_action`. """
        return self._canvas.active_editor

    @property
    def editors(self):
        """ dict: All of the :class:`Editor` that the canvas holds """
        return self._canvas.editors

    @property
    def navigation(self):
        """ :class:`~tools.manual.frameviewer.control.Navigation`: Class that handles frame
        Navigation and transport. """
        return self._navigation

    @property
    def tk_control_colors(self):
        """ :dict: Editor key with :class:`tkinter.StringVar` containing the selected color hex
        code for each annotation """
        return {key: val["color"].tk_var for key, val in self._canvas.annotation_formats.items()}

    @property
    def tk_selected_mask(self):
        """ :dict: Editor key with :class:`tkinter.StringVar` containing the selected color hex
        code for each annotation """
        return self._canvas.control_tk_vars["Mask"]["display"]["MaskType"]

    @property
    def _filter_modes(self):
        """ list: The filter modes combo box values """
        return ["All Frames", "Has Face(s)", "No Faces", "Multiple Faces", "Misaligned Faces"]

    def _add_nav(self):
        """ Add the slider to navigate through frames """
        max_frame = self._globals.frame_count - 1
        frame = ttk.Frame(self._transport_frame)

        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        lbl_frame = ttk.Frame(frame)
        lbl_frame.pack(side=tk.RIGHT)
        tbox = ttk.Entry(lbl_frame,
                         width=7,
                         textvariable=self._globals.var_transport_index,
                         justify=tk.RIGHT)
        tbox.pack(padx=0, side=tk.LEFT)
        lbl = ttk.Label(lbl_frame, text=f"/{max_frame}")
        lbl.pack(side=tk.RIGHT)

        cmd = partial(set_slider_rounding,
                      var=self._globals.var_transport_index,
                      d_type=int,
                      round_to=1,
                      min_max=(0, max_frame))

        nav = ttk.Scale(frame,
                        variable=self._globals.var_transport_index,
                        from_=0,
                        to=max_frame,
                        command=cmd)
        nav.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._globals.var_transport_index.trace_add("write", self._set_frame_index)
        return {"entry": tbox, "scale": nav, "label": lbl}

    def _set_frame_index(self, *args):  # pylint:disable=unused-argument
        """ Set the actual frame index based on current slider position and filter mode. """
        try:
            slider_position = self._globals.var_transport_index.get()
        except TclError:
            # don't update the slider when the entry box has been cleared of any value
            return
        frames = self._det_faces.filter.frames_list
        actual_position = max(0, min(len(frames) - 1, slider_position))
        if actual_position != slider_position:
            self._globals.var_transport_index.set(actual_position)
        frame_idx = frames[actual_position] if frames else -1
        logger.trace("slider_position: %s, frame_idx: %s", actual_position, frame_idx)
        self._globals.var_frame_index.set(frame_idx)

    def _add_transport(self):
        """ Add video transport controls """
        frame = ttk.Frame(self._transport_frame)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        icons = get_images().icons
        buttons = {}
        for action in ("play", "beginning", "prev", "next", "end", "save", "extract", "mode"):
            padx = (0, 6) if action in ("play", "prev", "mode") else (0, 0)
            side = tk.RIGHT if action in ("extract", "save", "mode") else tk.LEFT
            state = ["!disabled"] if action != "save" else ["disabled"]
            if action != "mode":
                icon = action if action != "extract" else "folder"
                wgt = ttk.Button(frame, image=icons[icon], command=self._btn_action[action])
                wgt.state(state)
            else:
                wgt = self._add_filter_section(frame)
            wgt.pack(side=side, padx=padx)
            if action != "mode":
                Tooltip(wgt, text=self._helptext[action])
            buttons[action] = wgt
        logger.debug("Transport buttons: %s", buttons)
        return buttons

    def _add_transport_tk_trace(self):
        """ Add the tkinter variable traces to buttons """
        self._navigation.tk_is_playing.trace("w", self._play)
        self._det_faces.tk_unsaved.trace("w", self._toggle_save_state)

    def _add_filter_section(self, frame):
        """ Add the section that holds the filter mode combo and any optional filter widgets

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The Frame that holds the filter section

        Returns
        -------
        :class:`tkinter.ttk.Frame`
            The filter section frame
        """
        filter_frame = ttk.Frame(frame)
        self._add_filter_mode_combo(filter_frame)
        self._add_filter_threshold_slider(filter_frame)
        filter_frame.pack(side=tk.RIGHT)
        return filter_frame

    def _add_filter_mode_combo(self, frame):
        """ Add the navigation mode combo box to the filter frame.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The Filter Frame that holds the filter combo box
        """
        self._globals.var_filter_mode.set("All Frames")
        self._globals.var_filter_mode.trace("w", self._navigation.nav_scale_callback)
        nav_frame = ttk.Frame(frame)
        lbl = ttk.Label(nav_frame, text="Filter:")
        lbl.pack(side=tk.LEFT, padx=(0, 5))
        combo = ttk.Combobox(
            nav_frame,
            textvariable=self._globals.var_filter_mode,
            state="readonly",
            values=self._filter_modes)
        combo.pack(side=tk.RIGHT)
        Tooltip(nav_frame, text=self._helptext["mode"])
        nav_frame.pack(side=tk.RIGHT)

    def _add_filter_threshold_slider(self, frame):
        """ Add the optional filter threshold slider for misaligned filter to the filter frame.

        Parameters
        ----------
        frame: :class:`tkinter.ttk.Frame`
            The Filter Frame that holds the filter threshold slider
        """
        slider_frame = ttk.Frame(frame)
        tk_var = self._globals.var_filter_distance

        min_max = (5, 20)
        ctl_frame = ttk.Frame(slider_frame)
        ctl_frame.pack(padx=2, side=tk.RIGHT)

        lbl = ttk.Label(ctl_frame, text="Distance:", anchor=tk.W)
        lbl.pack(side=tk.LEFT, anchor=tk.N, expand=True)

        tbox = ttk.Entry(ctl_frame, width=6, textvariable=tk_var, justify=tk.RIGHT)
        tbox.pack(padx=(0, 5), side=tk.RIGHT)

        ctl = ttk.Scale(
            ctl_frame,
            variable=tk_var,
            command=lambda val, var=tk_var, dt=int, rn=1, mm=min_max:
            set_slider_rounding(val, var, dt, rn, mm))
        ctl["from_"] = min_max[0]
        ctl["to"] = min_max[1]
        ctl.pack(padx=5, fill=tk.X, expand=True)
        for item in (tbox, ctl):
            Tooltip(item,
                    text=self._helptext["distance"],
                    wrap_length=200)
        tk_var.trace_add("write", self._navigation.nav_scale_callback)
        self._optional_widgets["distance_slider"] = slider_frame

    def pack_threshold_slider(self):
        """ Display or hide the threshold slider depending on the current filter mode. For
        misaligned faces filter, display the slider. Hide for all other filters. """
        if self._globals.var_filter_mode.get() == "Misaligned Faces":
            self._optional_widgets["distance_slider"].pack(side=tk.LEFT)
        else:
            self._optional_widgets["distance_slider"].pack_forget()

    def cycle_filter_mode(self):
        """ Cycle the navigation mode combo entry """
        current_mode = self._globals.var_filter_mode.get()
        idx = (self._filter_modes.index(current_mode) + 1) % len(self._filter_modes)
        self._globals.var_filter_mode.set(self._filter_modes[idx])

    def set_action(self, key):
        """ Set the current action based on keyboard shortcut

        Parameters
        ----------
        key: str
            The pressed key
        """
        # Allow key pad keys for numeric presses
        key = key.replace("KP_", "") if key.startswith("KP_") else key
        self._actions_frame.on_click(self._actions_frame.key_bindings[key])

    def _resize(self, event):
        """  Resize the image to fit the frame, maintaining aspect ratio """
        framesize = (event.width, event.height)
        logger.trace("Resizing video frame. Framesize: %s", framesize)
        self._globals.set_frame_display_dims(*framesize)
        self._globals.var_full_update.set(True)

    # << TRANSPORT >> #
    def _play(self, *args, frame_count=None):  # pylint:disable=unused-argument
        """ Play the video file. """
        start = time()
        is_playing = self._navigation.tk_is_playing.get()
        icon = "pause" if is_playing else "play"
        self._buttons["play"].config(image=get_images().icons[icon])

        if not is_playing:
            logger.debug("Pause detected. Stopping.")
            return

        # Populate the filtered frames count on first frame
        frame_count = self._det_faces.filter.count if frame_count is None else frame_count
        self._navigation.increment_frame(frame_count=frame_count, is_playing=True)
        delay = 16  # Cap speed at approx 60fps max. Unlikely to hit, but just in case
        duration = int((time() - start) * 1000)
        delay = max(1, delay - duration)
        self.after(delay, lambda f=frame_count: self._play(f))

    def _toggle_save_state(self, *args):  # pylint:disable=unused-argument
        """ Toggle the state of the save button when alignments are updated. """
        state = ["!disabled"] if self._det_faces.tk_unsaved.get() else ["disabled"]
        self._buttons["save"].state(state)


class ActionsFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The left hand action frame holding the action buttons.

    Parameters
    ----------
    parent: :class:`DisplayFrame`
        The Display frame that the Actions reside in
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 4), pady=2)
        self._globals = parent._globals
        self._det_faces = parent._det_faces

        self._configure_styles()
        self._actions = ("View", "BoundingBox", "ExtractBox", "Landmarks", "Mask")
        self._initial_action = "View"
        self._buttons = self._add_buttons()
        self._static_buttons = self._add_static_buttons()
        self._selected_action = self._set_selected_action_tkvar()
        self._optional_buttons = {}  # Has to be set from parent after canvas is initialized

    @property
    def actions(self):
        """ tuple: The available action names as a tuple of strings. """
        return self._actions

    @property
    def tk_selected_action(self):
        """ :class:`tkinter.StringVar`: The variable holding the currently selected action """
        return self._selected_action

    @property
    def key_bindings(self):
        """ dict: {`key`: `action`}. The mapping of key presses to actions. Keyboard shortcut is
        the first letter of each action. """
        return {f"F{idx + 1}": action for idx, action in enumerate(self._actions)}

    @property
    def _helptext(self):
        """ dict: `button key`: `button helptext`. The help text to display for each button. """
        inverse_keybindings = {val: key for key, val in self.key_bindings.items()}
        retval = {"View": _('View alignments'),
                  "BoundingBox": _('Bounding box editor'),
                  "ExtractBox": _("Location editor"),
                  "Mask": _("Mask editor"),
                  "Landmarks": _("Landmark point editor")}
        for item in retval:
            retval[item] += f" ({inverse_keybindings[item]})"
        return retval

    def _configure_styles(self):
        """ Configure background color for Actions widget """
        style = ttk.Style()
        style.configure("actions.TFrame", background='#d3d3d3')
        style.configure("actions_selected.TButton", relief="flat", background="#bedaf1")
        style.configure("actions_deselected.TButton", relief="flat")
        self.config(style="actions.TFrame")

    def _add_buttons(self):
        """ Add the action buttons to the Display window.

        Returns
        -------
        dict:
            The action name and its associated button.
        """
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.Y)
        buttons = {}
        for action in self.key_bindings.values():
            if action == self._initial_action:
                btn_style = "actions_selected.TButton"
                state = (["pressed", "focus"])
            else:
                btn_style = "actions_deselected.TButton"
                state = (["!pressed", "!focus"])

            button = ttk.Button(frame,
                                image=get_images().icons[action.lower()],
                                command=lambda t=action: self.on_click(t),
                                style=btn_style)
            button.state(state)
            button.pack()
            Tooltip(button, text=self._helptext[action])
            buttons[action] = button
        return buttons

    def on_click(self, action):
        """ Click event for all of the main buttons.

        Parameters
        ----------
        action: str
            The action name for the button that has called this event as exists in :attr:`_buttons`
        """
        for title, button in self._buttons.items():
            if action == title:
                button.configure(style="actions_selected.TButton")
                button.state(["pressed", "focus"])
            else:
                button.configure(style="actions_deselected.TButton")
                button.state(["!pressed", "!focus"])
        self._selected_action.set(action)

    def _set_selected_action_tkvar(self):
        """ Set the tkinter string variable that holds the currently selected editor action.
        Add traceback to display or hide editor specific optional buttons.

        Returns
        -------
        :class:`tkinter.StringVar
            The variable that holds the currently selected action
        """
        var = tk.StringVar()
        var.set(self._initial_action)
        var.trace("w", self._display_optional_buttons)
        return var

    def _add_static_buttons(self):
        """ Add the buttons to copy alignments from previous and next frames """
        lookup = {"copy_prev": (_("Previous"), "C"),
                  "copy_next": (_("Next"), "V"),
                  "reload": ("", "R")}
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.Y)
        sep = ttk.Frame(frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=5, side=tk.TOP)
        buttons = {}
        for action in ("copy_prev", "copy_next", "reload"):
            if action == "reload":
                icon = "reload3"
                cmd = lambda f=self._globals: self._det_faces.revert_to_saved(f.frame_index)  # noqa:E731,E501  # pylint:disable=line-too-long,unnecessary-lambda-assignment
                helptext = _("Revert to saved Alignments ({})").format(lookup[action][1])
            else:
                icon = action
                direction = action.replace("copy_", "")
                cmd = lambda f=self._globals, d=direction: self._det_faces.update.copy(  # noqa:E731,E501  # pylint:disable=line-too-long,unnecessary-lambda-assignment
                    f.frame_index, d)
                helptext = _("Copy {} Alignments ({})").format(*lookup[action])
            state = ["!disabled"] if action == "copy_next" else ["disabled"]
            button = ttk.Button(frame,
                                image=get_images().icons[icon],
                                command=cmd,
                                style="actions_deselected.TButton")
            button.state(state)
            button.pack()
            Tooltip(button, text=helptext)
            buttons[action] = button
        self._globals.var_frame_index.trace_add("write", self._disable_enable_copy_buttons)
        self._globals.var_full_update.trace_add("write", self._disable_enable_reload_button)
        return buttons

    def _disable_enable_copy_buttons(self, *args):  # pylint:disable=unused-argument
        """ Disable or enable the static buttons """
        position = self._globals.frame_index
        face_count_per_index = self._det_faces.face_count_per_index
        prev_exists = position != -1 and any(count != 0
                                             for count in face_count_per_index[:position])
        next_exists = position != -1 and any(count != 0
                                             for count in face_count_per_index[position + 1:])
        states = {"prev": ["!disabled"] if prev_exists else ["disabled"],
                  "next": ["!disabled"] if next_exists else ["disabled"]}
        for direction in ("prev", "next"):
            self._static_buttons[f"copy_{direction}"].state(states[direction])

    def _disable_enable_reload_button(self, *args):  # pylint:disable=unused-argument
        """ Disable or enable the static buttons """
        position = self._globals.frame_index
        state = ["!disabled"] if (position != -1 and
                                  self._det_faces.is_frame_updated(position)) else ["disabled"]
        self._static_buttons["reload"].state(state)

    def add_optional_buttons(self, editors):
        """ Add the optional editor specific action buttons """
        for name, editor in editors.items():
            actions = editor.actions
            if not actions:
                self._optional_buttons[name] = None
                continue
            frame = ttk.Frame(self)
            sep = ttk.Frame(frame, height=2, relief=tk.RIDGE)
            sep.pack(fill=tk.X, pady=5, side=tk.TOP)
            seen_groups = set()
            for action in actions.values():
                group = action["group"]
                if group is not None and group not in seen_groups:
                    btn_style = "actions_selected.TButton"
                    state = (["pressed", "focus"])
                    action["tk_var"].set(True)
                    seen_groups.add(group)
                else:
                    btn_style = "actions_deselected.TButton"
                    state = (["!pressed", "!focus"])
                    action["tk_var"].set(False)
                button = ttk.Button(frame,
                                    image=get_images().icons[action["icon"]],
                                    style=btn_style)
                button.config(command=lambda b=button: self._on_optional_click(b))
                button.state(state)
                button.pack()

                helptext = action["helptext"]
                hotkey = action["hotkey"]
                helptext += "" if hotkey is None else f" ({hotkey.upper()})"
                Tooltip(button, text=helptext)
                self._optional_buttons.setdefault(
                    name, {})[button] = {"hotkey": hotkey,
                                         "group": group,
                                         "tk_var": action["tk_var"]}
            self._optional_buttons[name]["frame"] = frame
        self._display_optional_buttons()

    def _on_optional_click(self, button):
        """ Click event for all of the optional buttons.

        Parameters
        ----------
        button: str
            The action name for the button that has called this event as exists in :attr:`_buttons`
        """
        options = self._optional_buttons[self._selected_action.get()]
        group = options[button]["group"]
        for child in options["frame"].winfo_children():
            if child.winfo_class() != "TButton":
                continue
            child_group = options[child]["group"]
            if child == button and group is not None:
                child.configure(style="actions_selected.TButton")
                child.state(["pressed", "focus"])
                options[child]["tk_var"].set(True)
            elif child != button and group is not None and child_group == group:
                child.configure(style="actions_deselected.TButton")
                child.state(["!pressed", "!focus"])
                options[child]["tk_var"].set(False)
            elif group is None and child_group is None:
                if child.cget("style") == "actions_selected.TButton":
                    child.configure(style="actions_deselected.TButton")
                    child.state(["!pressed", "!focus"])
                    options[child]["tk_var"].set(False)
                else:
                    child.configure(style="actions_selected.TButton")
                    child.state(["pressed", "focus"])
                    options[child]["tk_var"].set(True)

    def _display_optional_buttons(self, *args):  # pylint:disable=unused-argument
        """ Pack or forget the optional buttons depending on active editor """
        self._unbind_optional_hotkeys()
        for editor, option in self._optional_buttons.items():
            if option is None:
                continue
            if editor == self._selected_action.get():
                logger.debug("Displaying optional buttons for '%s'", editor)
                option["frame"].pack(side=tk.TOP, fill=tk.Y)
                for child in option["frame"].winfo_children():
                    if child.winfo_class() != "TButton":
                        continue
                    hotkey = option[child]["hotkey"]
                    if hotkey is not None:
                        logger.debug("Binding optional hotkey for editor '%s': %s", editor, hotkey)
                        self.winfo_toplevel().bind(hotkey.lower(),
                                                   lambda e, b=child: self._on_optional_click(b))
            elif option["frame"].winfo_ismapped():
                logger.debug("Hiding optional buttons for '%s'", editor)
                option["frame"].pack_forget()

    def _unbind_optional_hotkeys(self):
        """ Unbind all mapped optional button hotkeys """
        for editor, option in self._optional_buttons.items():
            if option is None or not option["frame"].winfo_ismapped():
                continue
            for child in option["frame"].winfo_children():
                if child.winfo_class() != "TButton":
                    continue
                hotkey = option[child]["hotkey"]
                if hotkey is not None:
                    logger.debug("Unbinding optional hotkey for editor '%s': %s", editor, hotkey)
                    self.winfo_toplevel().unbind(hotkey.lower())


class FrameViewer(tk.Canvas):  # pylint:disable=too-many-ancestors
    """ Annotation onto tkInter Canvas.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the canvas
    tk_globals: :class:`~tools.manual.manual.TkGlobals`
        The tkinter variables that apply to the whole of the GUI
    detected_faces: :class:`AlignmentsData`
        The alignments data for this manual session
    actions: tuple
        The available actions from :attr:`ActionFrame.actions`
    tk_action_var: :class:`tkinter.StringVar`
        The variable holding the currently selected action
    """
    def __init__(self, parent, tk_globals, detected_faces, actions, tk_action_var):
        logger.debug("Initializing %s: (parent: %s, tk_globals: %s, detected_faces: %s, "
                     "actions: %s, tk_action_var: %s)", self.__class__.__name__,
                     parent, tk_globals, detected_faces, actions, tk_action_var)
        super().__init__(parent, bd=0, highlightthickness=0, background="black")
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.E)
        self._globals = tk_globals
        self._det_faces = detected_faces
        self._actions = actions
        self._tk_action_var = tk_action_var
        self._image = BackgroundImage(self)
        self._editor_globals = {"control_tk_vars": {},
                                "annotation_formats": {},
                                "key_bindings": {}}
        self._max_face_count = 0
        self._editors = self._get_editors()
        self._add_callbacks()
        self._change_active_editor()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def selected_action(self):
        """str: The name of the currently selected Editor action """
        return self._tk_action_var.get()

    @property
    def control_tk_vars(self):
        """ dict: dictionary of tkinter variables as populated by the right hand control panel.
        Tracking for all control panel variables, for access from all editors. """
        return self._editor_globals["control_tk_vars"]

    @property
    def key_bindings(self):
        """ dict: dictionary of key bindings for each editor for access from all editors. """
        return self._editor_globals["key_bindings"]

    @property
    def annotation_formats(self):
        """ dict: The selected formatting options for each annotation """
        return self._editor_globals["annotation_formats"]

    @property
    def active_editor(self):
        """ :class:`Editor`: The current editor in use based on :attr:`selected_action`. """
        return self._editors[self.selected_action]

    @property
    def editors(self):
        """ dict: All of the :class:`Editor` objects that exist """
        return self._editors

    @property
    def editor_display(self):
        """ dict: List of editors and any additional annotations they should display. """
        return {"View": ["BoundingBox", "ExtractBox", "Landmarks", "Mesh"],
                "BoundingBox": ["Mesh"],
                "ExtractBox": ["Mesh"],
                "Landmarks": ["ExtractBox", "Mesh"],
                "Mask": []}

    @property
    def offset(self):
        """ tuple: The (`width`, `height`) offset of the canvas based on the size of the currently
        displayed image """
        frame_dims = self._globals.current_frame.display_dims
        offset_x = (self._globals.frame_display_dims[0] - frame_dims[0]) / 2
        offset_y = (self._globals.frame_display_dims[1] - frame_dims[1]) / 2
        logger.trace("offset_x: %s, offset_y: %s", offset_x, offset_y)
        return offset_x, offset_y

    def _get_editors(self):
        """ Get the object editors for the canvas.

        Returns
        ------
        dict
            The {`action`: :class:`Editor`} dictionary of editors for :attr:`_actions` name.
        """
        editors = {}
        for editor_name in self._actions + ("Mesh", ):
            editor = eval(editor_name)(self,  # pylint:disable=eval-used
                                       self._det_faces)
            editors[editor_name] = editor
        logger.debug(editors)
        return editors

    def _add_callbacks(self):
        """ Add the callback trace functions to the :class:`tkinter.Variable` s

        Adds callbacks for:
            :attr:`_globals.var_full_update` Update the display for the current image
            :attr:`__tk_action_var` Update the mouse display tracking for current action
        """
        self._globals.var_full_update.trace_add("write", self._update_display)
        self._tk_action_var.trace_add("write", self._change_active_editor)

    def _change_active_editor(self, *args):  # pylint:disable=unused-argument
        """ Update the display for the active editor.

        Hide the annotations that are not relevant for the selected editor.
        Set the selected editor's cursor tracking.

        Parameters
        ----------
        args: tuple, unused
            Required for tkinter callback but unused
        """
        to_display = [self.selected_action] + self.editor_display[self.selected_action]
        to_hide = [editor for editor in self._editors if editor not in to_display]
        for editor in to_hide:
            self._editors[editor].hide_annotation()

        self.active_editor.bind_mouse_motion()
        self.active_editor.set_mouse_click_actions()
        self._globals.var_full_update.set(True)

    def _update_display(self, *args):  # pylint:disable=unused-argument
        """ Update the display on frame cache update

        Notes
        -----
        A little hacky, but the editors to display or hide are processed in alphabetical
        order, so that they are always processed in the same order (for tag lowering and raising)
        """
        if not self._globals.var_full_update.get():
            return
        zoomed_centering = self.active_editor.zoomed_centering
        self._image.refresh(self.active_editor.view_mode)
        to_display = sorted([self.selected_action] + self.editor_display[self.selected_action])
        self._hide_additional_faces()
        for editor in to_display:
            self._editors[editor].update_annotation()
        self._bind_unbind_keys()
        if zoomed_centering != self.active_editor.zoomed_centering:
            # Refresh the image if editor annotation has changed the zoom centering of the image
            self._image.refresh(self.active_editor.view_mode)
        self._globals.var_full_update.set(False)
        self.update_idletasks()

    def _hide_additional_faces(self):
        """ Hide additional faces if the number of faces on the canvas reduces on a frame
        change. """
        if self._globals.is_zoomed:
            current_face_count = 1
        elif self._globals.frame_index == -1:
            current_face_count = 0
        else:
            current_face_count = len(self._det_faces.current_faces[self._globals.frame_index])

        if current_face_count > self._max_face_count:
            # Most faces seen to date so nothing to hide. Update max count and return
            logger.debug("Incrementing max face count from: %s to: %s",
                         self._max_face_count, current_face_count)
            self._max_face_count = current_face_count
            return
        for idx in range(current_face_count, self._max_face_count):
            tag = f"face_{idx}"
            if any(self.itemcget(item_id, "state") != "hidden"
                   for item_id in self.find_withtag(tag)):
                logger.debug("Hiding face tag '%s'", tag)
                self.itemconfig(tag, state="hidden")

    def _bind_unbind_keys(self):
        """ Bind or unbind this editor's hotkeys depending on whether it is active. """
        unbind_keys = [key for key, binding in self.key_bindings.items()
                       if binding["bound_to"] is not None
                       and binding["bound_to"] != self.selected_action]
        for key in unbind_keys:
            logger.debug("Unbinding key '%s'", key)
            self.winfo_toplevel().unbind(key)
            self.key_bindings[key]["bound_to"] = None

        bind_keys = {key: binding[self.selected_action]
                     for key, binding in self.key_bindings.items()
                     if self.selected_action in binding
                     and binding["bound_to"] != self.selected_action}
        for key, method in bind_keys.items():
            logger.debug("Binding key '%s' to method %s", key, method)
            self.winfo_toplevel().bind(key, method)
            self.key_bindings[key]["bound_to"] = self.selected_action
