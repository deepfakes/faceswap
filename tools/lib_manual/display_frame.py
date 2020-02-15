#!/usr/bin/env python3
""" The frame viewer section of the manual tool GUI """
import logging
import tkinter as tk
from tkinter import ttk

from functools import partial
from time import time

from lib.gui.control_helper import set_slider_rounding
from lib.gui.custom_widgets import Tooltip
from lib.gui.utils import get_images

from .editor import BoundingBox, ExtractBox, Landmarks, Mask, Mesh, View

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DisplayFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The main video display frame (top left section of GUI).

    Parameters
    ----------
    parent: :class:`tkinter.PanedWindow`
        The paned window that the display frame resides in
    frames: :class:`FrameNavigation`
        The object that holds the cache of frames.
    alignments: dict
        Dictionary of :class:`lib.faces_detect.DetectedFace` objects
    """
    def __init__(self, parent, frames, alignments):
        logger.debug("Initializing %s: (parent: %s, frames: %s)",
                     self.__class__.__name__, parent, frames)
        super().__init__(parent)
        self.pack(side=tk.LEFT, anchor=tk.NW)

        self._frames = frames
        self._alignments = alignments

        self._actions_frame = ActionsFrame(self, self._alignments)
        main_frame = ttk.Frame(self)
        main_frame.pack(side=tk.RIGHT)
        self._video_frame = ttk.Frame(main_frame,
                                      width=self._frames.display_dims[0],
                                      height=self._frames.display_dims[1])

        self._video_frame.pack(side=tk.TOP, expand=True)
        self._video_frame.pack_propagate(False)

        self._canvas = FrameViewer(self._video_frame,
                                   self._alignments,
                                   self._frames,
                                   self._actions_frame.actions,
                                   self._actions_frame.tk_selected_action)

        self._transport_frame = ttk.Frame(main_frame)
        self._transport_frame.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)

        self._nav = self._add_nav()
        self._play_button = self._add_transport()
        self._actions_frame.add_optional_buttons(self.editors)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _helptext(self):
        """ dict: {`name`: `help text`} Helptext lookup for navigation buttons """
        return dict(
            play="Play/Pause (SPACE)",
            beginning="Go to First Frame (HOME)",
            prev="Go to Previous Frame (LEFT)",
            next="Go to Next Frame (RIGHT)",
            end="Go to Last Frame (END)",
            mode="Filter the frames to those containing the selected item.")

    @property
    def _btn_action(self):
        """ dict: {`name`: `action`} Command lookup for navigation buttons """
        actions = dict(play=self.handle_play_button,
                       beginning=self.goto_first_frame,
                       prev=self.decrement_frame,
                       next=self.increment_frame,
                       end=self.goto_last_frame)
        return actions

    @property
    def tk_selected_action(self):
        """ :class:`tkinter.StringVar`: The variable holding the currently selected action """
        return self._actions_frame.tk_selected_action

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: The frames display update flag. """
        return self._frames.tk_update

    @property
    def active_editor(self):
        """ :class:`Editor`: The current editor in use based on :attr:`selected_action`. """
        return self._canvas.active_editor

    @property
    def editors(self):
        """ dict: All of the :class:`Editor` that the canvas holds """
        return self._canvas.editors

    @property
    def _frames_count(self):
        """ int: The number of frames based on the current navigation mode """
        nav_mode = self._frames.tk_navigation_mode.get()
        if nav_mode == "No Faces":
            retval = self._alignments.no_face_count
        elif nav_mode == "Multiple Faces":
            retval = self._alignments.multi_face_count
        elif nav_mode == "Has Face(s)":
            retval = self._alignments.with_face_count
        else:
            retval = self._frames.frame_count
        logger.trace("nav_mode: %s, number_frames: %s", nav_mode, retval)
        return retval

    def _add_nav(self):
        """ Add the slider to navigate through frames """
        var = self._frames.tk_transport_position
        var.trace("w", self._set_frame_index)
        max_frame = self._frames.frame_count - 1

        frame = ttk.Frame(self._transport_frame)

        frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        lbl_frame = ttk.Frame(frame)
        lbl_frame.pack(side=tk.RIGHT)
        tbox = ttk.Entry(lbl_frame,
                         width=7,
                         textvariable=var,
                         justify=tk.RIGHT)
        tbox.pack(padx=0, side=tk.LEFT)
        lbl = ttk.Label(lbl_frame, text="/{}".format(max_frame))
        lbl.pack(side=tk.RIGHT)

        cmd = partial(set_slider_rounding,
                      var=var,
                      d_type=int,
                      round_to=1,
                      min_max=(0, max_frame))

        nav = ttk.Scale(frame, variable=var, from_=0, to=max_frame, command=cmd)
        nav.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        return dict(entry=tbox, scale=nav, label=lbl)

    def _set_frame_index(self, *args):  # pylint:disable=unused-argument
        """ Set the actual frame index based on current slider position and filter mode. """
        # TODO Update total/slider position when adding/removing faces for single/multi face frames
        slider_position = self._frames.tk_transport_position.get()
        frames = self._alignments.get_filtered_frames_list()
        frame_idx = frames[slider_position] if frames else 0
        logger.trace("slider_position: %s, frame_idx: %s", slider_position, frame_idx)
        self._frames.tk_position.set(frame_idx)

    def _add_transport(self):
        """ Add video transport controls """
        frame = ttk.Frame(self._transport_frame)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        icons = get_images().icons

        for action in ("play", "beginning", "prev", "next", "end", "mode"):
            padx = (0, 6) if action in ("play", "prev") else (0, 0)
            if action != "mode":
                wgt = ttk.Button(frame, image=icons[action], command=self._btn_action[action])
                side = tk.LEFT
            else:
                wgt = self._add_navigation_mode_combo(frame)
                side = tk.RIGHT
            wgt.pack(side=side, padx=padx)
            if action == "play":
                play_btn = wgt
                self._frames.tk_is_playing.trace("w", self._play)
            Tooltip(wgt, text=self._helptext[action])
        return play_btn

    def _add_navigation_mode_combo(self, frame):
        """ Add the navigation mode combo box to the transport frame """
        tk_var = self._frames.tk_navigation_mode
        tk_var.set("All Frames")
        tk_var.trace("w", self._nav_scale_callback)
        nav_frame = ttk.Frame(frame)
        lbl = ttk.Label(nav_frame, text="Filter:")
        lbl.pack(side=tk.LEFT, padx=(0, 5))
        combo = ttk.Combobox(
            nav_frame,
            textvariable=tk_var,
            state="readonly",
            values=["All Frames", "Has Face(s)", "No Faces", "Multiple Faces"])
        combo.pack(side=tk.RIGHT)
        return nav_frame

    def _nav_scale_callback(self, *args):  # pylint:disable=unused-argument
        """ Adjust transport slider scale for different filters """
        self._frames.stop_playback()
        frame_count = self._frames_count
        max_frame = frame_count if frame_count == 0 else frame_count - 1
        self._nav["scale"].config(to=max_frame)
        self._nav["label"].config(text="/{}".format(max_frame))
        state = "disabled" if max_frame == 0 else "normal"
        self._nav["entry"].config(state=state)
        self._frames.tk_transport_position.set(0)

    def handle_play_button(self):
        """ Handle the play button.

        Switches the :attr:`_frames.is_playing` variable.
        """
        is_playing = self._frames.tk_is_playing.get()
        self._frames.tk_is_playing.set(not is_playing)

    def set_action(self, key):
        """ Set the current action based on keyboard shortcut

        Parameters
        ----------
        key: str
            The pressed key
        """
        # Allow key pad keys for numeric presses
        key = key.replace("KP_", "") if key.startswith("KP_") else key
        self._actions_frame.on_click(self._actions_frame.key_bindings[key.lower()])

    # << TRANSPORT >> #
    def _play(self, *args, frame_count=None):  # pylint:disable=unused-argument
        """ Play the video file. """
        start = time()
        is_playing = self._frames.tk_is_playing.get()
        icon = "pause" if is_playing else "play"
        self._play_button.config(image=get_images().icons[icon])

        if not is_playing:
            logger.debug("Pause detected. Stopping.")
            return

        # Populate the filtered frames count on first frame
        frame_count = self._frames_count if frame_count is None else frame_count
        self.increment_frame(frame_count=frame_count, is_playing=True)
        delay = 16  # Cap speed at approx 60fps max. Unlikely to hit, but just in case
        duration = int((time() - start) * 1000)
        delay = max(1, delay - duration)
        self.after(delay, lambda f=frame_count: self._play(f))

    def increment_frame(self, frame_count=None, is_playing=False):
        """ Update The frame navigation position to the next frame based on filter. """
        if not is_playing:
            self._frames.stop_playback()
        position = self._frames.tk_transport_position.get()
        frame_count = self._frames_count if frame_count is None else frame_count
        if position == frame_count - 1:
            logger.trace("End of stream. Not incrementing")
            self._frames.stop_playback()
            return
        self._frames.tk_transport_position.set(position + 1)

    def decrement_frame(self):
        """ Update The frame navigation position to the previous frame based on filter. """
        self._frames.stop_playback()
        position = self._frames.tk_transport_position.get()
        if position == 0:
            logger.trace("Beginning of stream. Not decrementing")
            return
        self._frames.tk_transport_position.set(position - 1)

    def goto_first_frame(self):
        """ Go to the first frame that meets the filter criteria. """
        self._frames.stop_playback()
        position = self._frames.tk_transport_position.get()
        if position == 0:
            return
        self._frames.tk_transport_position.set(0)

    def goto_last_frame(self):
        """ Go to the last frame that meets the filter criteria. """
        self._frames.stop_playback()
        position = self._frames.tk_transport_position.get()
        frame_count = self._frames_count
        if position == frame_count - 1:
            return
        self._frames.tk_transport_position.set(frame_count - 1)


class ActionsFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The left hand action frame holding the action buttons.

    Parameters
    ----------
    parent: :class:`DisplayFrame`
        The Display frame that the Actions reside in
    alignments: dict
        Dictionary of :class:`lib.faces_detect.DetectedFace` objects
    """
    def __init__(self, parent, alignments):
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.Y, padx=(2, 4), pady=2)

        self._configure_styles()
        self._actions = ("view", "boundingbox", "extractbox", "mask", "landmarks")
        self._initial_action = "view"
        self._buttons = self._add_buttons()
        self._add_copy_buttons(alignments)
        self._selected_action = self._set_selected_action_tkvar()
        self._optional_buttons = dict()  # Has to be set from parent after canvas is initialized

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
        return {str(idx + 1): action for idx, action in enumerate(self._actions)}

    @property
    def _helptext(self):
        """ dict: `button key`: `button helptext`. The help text to display for each button. """
        inverse_keybindings = {val: key for key, val in self.key_bindings.items()}
        retval = dict(view="View alignments",
                      boundingbox="Bounding box editor",
                      extractbox="Edit the size and orientation of the existing alignments",
                      mask="Mask editor",
                      landmarks="Individual landmark point editor")
        for item in retval:
            retval[item] += " ({})".format(inverse_keybindings[item])
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
        buttons = dict()
        for action in self.key_bindings.values():
            if action == self._initial_action:
                btn_style = "actions_selected.TButton"
                state = (["pressed", "focus"])
            else:
                btn_style = "actions_deselected.TButton"
                state = (["!pressed", "!focus"])

            button = ttk.Button(frame,
                                image=get_images().icons[action],
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

    def _add_copy_buttons(self, alignments):
        """ Add the buttons to copy alignments from previous and next frames """
        lookup = dict(copy_prev=("Previous", "C"), copy_next=("Next", "V"))
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, fill=tk.Y)
        sep = ttk.Frame(frame, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=5, side=tk.TOP)
        for action in ("copy_prev", "copy_next"):
            direction = action.replace("copy_", "")
            button = ttk.Button(frame,
                                image=get_images().icons[action],
                                command=lambda d=direction: alignments.copy_alignments(d),
                                style="actions_deselected.TButton")
            button.pack()
            helptext = "Copy {} Alignments to this Frame ({})".format(*lookup[action])
            Tooltip(button, text=helptext)

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
            for idx, action in enumerate(actions.values()):
                if idx == 0:
                    btn_style = "actions_selected.TButton"
                    state = (["pressed", "focus"])
                    action["tk_var"].set(True)
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
                helptext += "" if hotkey is None else " ({})".format(hotkey.upper())
                Tooltip(button, text=helptext)
                self._optional_buttons.setdefault(
                    name, dict())[button] = dict(hotkey=hotkey, tk_var=action["tk_var"])
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
        for child in options["frame"].winfo_children():
            if child.winfo_class() != "TButton":
                continue
            if child == button:
                child.configure(style="actions_selected.TButton")
                child.state(["pressed", "focus"])
                options[child]["tk_var"].set(True)
            else:
                child.configure(style="actions_deselected.TButton")
                child.state(["!pressed", "!focus"])
                options[child]["tk_var"].set(False)

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
    alignments: :class:`AlignmentsData`
        The alignments data for this manual session
    frames: :class:`FrameNavigation`
        The frames navigator for this manual session
    actions: tuple
        The available actions from :attr:`ActionFrame.actions`
    tk_action_var: :class:`tkinter.StringVar`
        The variable holding the currently selected action
    """
    def __init__(self, parent, alignments, frames, actions, tk_action_var):
        logger.debug("Initializing %s: (parent: %s, alignments: %s, frames: %s, actions: %s, "
                     "tk_action_var: %s)",
                     self.__class__.__name__, parent, alignments, frames, actions, tk_action_var)
        super().__init__(parent, bd=0, highlightthickness=0, background="black")
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.E)

        self._alignments = alignments
        self._frames = frames
        self._image_is_hidden = False
        self._actions = actions
        self._tk_action_var = tk_action_var
        self._image = self.create_image(self._frames.display_dims[0] / 2,
                                        self._frames.display_dims[1] / 2,
                                        image=self._frames.current_display_frame,
                                        anchor=tk.CENTER)
        self._control_tk_vars = dict()
        self._annotation_formats = dict()
        self._editors = self._get_editors()
        self._add_callbacks()
        self._update_active_display()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def selected_action(self):
        """str: The name of the currently selected Editor action """
        return self._tk_action_var.get()

    @property
    def control_tk_vars(self):
        """ dict: dictionary of tkinter variables as populated by the right hand control panel.
        Tracking for all control panel variables, for access from all editors. """
        return self._control_tk_vars

    @property
    def annotation_formats(self):
        """ dict: The selected formatting options for each annotation """
        return self._annotation_formats

    @property
    def active_editor(self):
        """ :class:`Editor`: The current editor in use based on :attr:`selected_action`. """
        return self._editors[self.selected_action]

    @property
    def editors(self):
        """ dict: All of the :class:`Editor` objects that exist """
        return self._editors

    @property
    def offset(self):
        """ tuple: The (`width`, `height`) offset of the canvas based on the size of the currently
        displayed image """
        frame_dims = self._frames.current_meta_data["display_dims"]
        offset_x = (self._frames.display_dims[0] - frame_dims[0]) / 2
        offset_y = (self._frames.display_dims[1] - frame_dims[1]) / 2
        logger.trace("offset_x: %s, offset_y: %s", offset_x, offset_y)
        return offset_x, offset_y

    @property
    def image_is_hidden(self):
        """ bool: ``True`` if the background frame image is hidden otherwise ``False``. """
        return self._image_is_hidden

    def send_frame_to_bottom(self):
        """ Sent the background frame to the bottom of the stack """
        self.tag_lower(self._image)

    def toggle_image_display(self):
        """ Toggle the background frame between displayed and hidden. """
        state = "normal" if self._image_is_hidden else "hidden"
        self.itemconfig(self._image, state=state)
        self._image_is_hidden = not self._image_is_hidden

    def _get_editors(self):
        """ Get the object editors for the canvas.

        Returns
        ------
        dict
            The {`action`: :class:`Editor`} dictionary of editors for :attr:`_actions` name.
        """
        name_mapping = dict(boundingbox=BoundingBox,
                            extractbox=ExtractBox,
                            landmarks=Landmarks,
                            mask=Mask,
                            mesh=Mesh,
                            view=View)
        editors = dict()
        for action in self._actions + ("mesh", ):
            editor = name_mapping[action]
            editor = editor(self, self._alignments, self._frames)
            editors[action] = editor
        logger.debug(editors)
        return editors

    def _add_callbacks(self):
        """ Add the callback trace functions to the :class:`tkinter.Variable` s

        Adds callbacks for:
            :attr:`_frames.tk_update` Update the display for the current image
            :attr:`__tk_action_var` Update the mouse display tracking for current action
        """
        self._frames.tk_update.trace("w", self._update_display)
        self._tk_action_var.trace("w", self._update_active_display)

    def _update_active_display(self, *args):  # pylint:disable=unused-argument
        """ Update the display for the active editor.

        Sets the editor's cursor tracking and annotation display based on which editor is active.

        Parameters
        ----------
        args: tuple, unused
            Required for tkinter callback but unused
        """
        self.active_editor.bind_mouse_motion()
        self.active_editor.set_mouse_click_actions()
        self._frames.tk_update.set(True)

    def _update_display(self, *args):  # pylint:disable=unused-argument
        """ Update the display on frame cache update """
        if not self._frames.tk_update.get():
            return
        self.refresh_display_image()
        for editor in self._editors.values():
            editor.update_annotation()
            editor.hide_additional_annotations()
        self._frames.tk_update.set(False)
        self.update_idletasks()

    def refresh_display_image(self):
        """ Update the displayed frame """
        if not self._frames.needs_update:
            logger.trace("Background frame not updated. Returning")
            return
        logger.trace("Updating background frame")
        self.itemconfig(self._image, image=self._frames.current_display_frame)
        if self._image_is_hidden:
            logger.trace("Unhiding background frame")
            self.toggle_image_display()
        self._frames.clear_update_flag()
