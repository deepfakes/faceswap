#!/usr/bin/env python3
""" Tool to manually interact with the alignments file using visual tools """
import logging

import tkinter as tk
from tkinter import ttk
from functools import partial
from time import time, sleep

from lib.gui.control_helper import set_slider_rounding, ControlPanel
from lib.gui.custom_widgets import Tooltip, StatusBar
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.multithreading import MultiThread
from plugins.extract.pipeline import Extractor, ExtractMedia

from .lib_manual.media import AlignmentsData, FrameNavigation, FaceCache
from .lib_manual.editor import BoundingBox, ExtractBox, Landmarks, Mask, Mesh, View

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Manual(tk.Tk):
    """ This tool is part of the Faceswap Tools suite and should be called from
    ``python tools.py manual`` command.

    Allows for visual interaction with frames, faces and alignments file to perform various
    adjustments to the alignments file.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The :mod:`argparse` arguments as passed in from :mod:`tools.py`
    """

    def __init__(self, arguments):
        logger.debug("Initializing %s: (arguments: '%s')", self.__class__.__name__, arguments)
        super().__init__()
        self._initialize_tkinter()

        extractor = Aligner()
        self._frames = FrameNavigation(arguments.frames, get_config().scaling_factor)
        self._wait_for_extractor(extractor)

        self._alignments = AlignmentsData(arguments.alignments_path, self._frames, extractor)

        self._containers = self._create_containers()

        pbar = StatusBar(self._containers["bottom"], hide_status=True)
        self._faces = FaceCache(self._alignments, pbar)

        self._display = DisplayFrame(self._containers["top"], self._frames, self._alignments)
        self._faces_frame = FacesFrame(self._containers["bottom"], self._faces)

        self._options = Options(self._containers["top"], self._display)
        self._display.tk_selected_action.set("view")

        self.bind("<Key>", self._handle_key_press)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def _wait_for_extractor(extractor):
        """ The :class:`Aligner` is launched in a background thread. Wait for it to be initialized
        prior to proceeding """
        while True:
            if extractor.is_initialized:
                logger.debug("Aligner inialized")
                return
            logger.debug("Aligner not initialized. Waiting...")
            sleep(1)

    def _initialize_tkinter(self):
        """ Initialize a standalone tkinter instance. """
        logger.debug("Initializing tkinter")
        initialize_config(self, None, None, None)
        initialize_images()
        get_config().set_geometry(940, 600, fullscreen=True)
        self.title("Faceswap.py - Visual Alignments")
        self.tk.call(
            "wm",
            "iconphoto",
            self._w, get_images().icons["favicon"])  # pylint:disable=protected-access
        logger.debug("Initialized tkinter")

    def _create_containers(self):
        """ Create the paned window containers for various GUI elements

        Returns
        -------
        dict:
            The main containers of the manual tool.
        """
        logger.debug("Creating containers")
        main = tk.PanedWindow(self,
                              sashrelief=tk.RIDGE,
                              sashwidth=2,
                              sashpad=4,
                              orient=tk.VERTICAL,
                              name="pw_main")
        main.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(main, name="frame_top")
        main.add(top)

        bottom = ttk.Frame(main, name="frame_bottom")
        main.add(bottom)
        logger.debug("Created containers")
        return dict(main=main, top=top, bottom=bottom)

    def _handle_key_press(self, event):
        """ Keyboard shortcuts

        Parameters
        ----------
        event: :class:`tkinter.Event()`
            The tkinter key press event

        Notes
        -----
        The following keys are reserved for the :mod:`tools.lib_manual.editor` classes
            * Delete - Used for deleting faces
            * [] - decrease / increase brush size
        """
        # TODO - Alt modifier doesn't seem to work on windows. Check
        modifiers = {0x0001: 'shift',
                     0x0004: 'ctrl',
                     0x0008: 'alt',
                     0x0080: 'alt'}

        bindings = dict(
            left=self._frames.decrement_frame,
            shift_left=lambda d="prev", f="single": self._alignments.set_next_frame(d, f),
            ctrl_left=lambda d="prev", f="multi": self._alignments.set_next_frame(d, f),
            alt_left=lambda d="prev", f="no": self._alignments.set_next_frame(d, f),
            right=self._frames.increment_frame,
            shift_right=lambda d="next", f="single": self._alignments.set_next_frame(d, f),
            ctrl_right=lambda d="next", f="multi": self._alignments.set_next_frame(d, f),
            alt_right=lambda d="next", f="no": self._alignments.set_next_frame(d, f),
            space=self._display.handle_play_button,
            home=self._frames.set_first_frame,
            end=self._frames.set_last_frame,
            v=lambda k=event.keysym: self._display.set_action(k),
            b=lambda k=event.keysym: self._display.set_action(k),
            e=lambda k=event.keysym: self._display.set_action(k),
            m=lambda k=event.keysym: self._display.set_action(k),
            l=lambda k=event.keysym: self._display.set_action(k))  # noqa

        modifier = "_".join(val for key, val in modifiers.items() if event.state & key != 0)
        key_press = "_".join([modifier, event.keysym]) if modifier else event.keysym
        if key_press.lower() in bindings:
            self.focus_set()
            bindings[key_press.lower()]()

    def process(self):
        """ The entry point for the Visual Alignments tool from :file:`lib.tools.cli`.

        Launch the tkinter Visual Alignments Window and run main loop.
        """
        self.mainloop()


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

        self._actions_frame = ActionsFrame(self)

        self._video_frame = ttk.Frame(self,
                                      width=self._frames.display_dims[0],
                                      height=self._frames.display_dims[1])

        self._video_frame.pack(side=tk.RIGHT, expand=True)
        self._video_frame.pack_propagate(False)

        self._canvas = FrameViewer(self._video_frame,
                                   self._alignments,
                                   self._frames,
                                   self._actions_frame.actions,
                                   self._actions_frame.tk_selected_action)

        self._transport_frame = ttk.Frame(self._video_frame)
        self._transport_frame.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)

        self._add_nav()
        self._play_button = self._add_transport()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _helptext(self):
        """ dict: {`name`: `help text`} Helptext lookup for navigation buttons """
        return dict(
            play="Play/Pause (SPACE)",
            beginning="Go to First Frame (HOME)",
            prev="Go to Previous Frame (LEFT)",
            prev_single_face="Go to Previous Frame that contains a Single Face (SHIFT+LEFT)",
            prev_multi_face="Go to Previous Frame that contains Multiple Faces (CTRL+LEFT)",
            prev_no_face="Go to Previous Frame that contains No Faces (ALT+LEFT)",
            next="Go to Next Frame (RIGHT)",
            next_single_face="Go to Next Frame that contains a Single Face (SHIFT+RIGHT)",
            next_multi_face="Go to Next Frame that contains Multiple Faces (CTRL+RIGHT)",
            next_no_face="Go to Next Frame that contains No Faces (ALT+RIGHT)",
            end="Go to Last Frame (END)")

    @property
    def _btn_action(self):
        """ dict: {`name`: `action`} Command lookup for navigation buttons """
        actions = dict(play=self.handle_play_button,
                       beginning=self._frames.set_first_frame,
                       prev=self._frames.decrement_frame,
                       next=self._frames.increment_frame,
                       end=self._frames.set_last_frame)
        for drn in ("prev", "next"):
            for flt in ("no", "multi", "single"):
                actions["{}_{}_face".format(drn, flt)] = (lambda d=drn, f=flt:
                                                          self._alignments.set_next_frame(d, f))
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

    def _add_nav(self):
        """ Add the slider to navigate through frames """
        var = self._frames.tk_position
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

    def _add_transport(self):
        """ Add video transport controls """
        # TODO Disable buttons when no frames meet filter criteria
        frame = ttk.Frame(self._transport_frame)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        icons = get_images().icons

        for action in ("play", "beginning", "prev", "prev_single_face", "prev_multi_face",
                       "prev_no_face", "next_no_face", "next_multi_face", "next_single_face",
                       "next", "end"):
            padx = (0, 6) if action in ("play", "prev", "next_single_face") else (0, 0)
            wgt = ttk.Button(frame, image=icons[action], command=self._btn_action[action])
            wgt.pack(side=tk.LEFT, padx=padx)
            if action == "play":
                play_btn = wgt
                self._frames.tk_is_playing.trace("w", self._play)
            Tooltip(wgt, text=self._helptext[action])
        return play_btn

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
        self._actions_frame.on_click(self._actions_frame.key_bindings[key.lower()])

    def _play(self, *args):  # pylint:disable=unused-argument
        """ Play the video file. """
        start = time()
        is_playing = self._frames.tk_is_playing.get()
        icon = "pause" if is_playing else "play"
        self._play_button.config(image=get_images().icons[icon])

        if not is_playing:
            logger.debug("Pause detected. Stopping.")
            return

        self._frames.increment_frame(is_playing=True)
        delay = 16  # Cap speed at approx 60fps max. Unlikely to hit, but just in case
        duration = int((time() - start) * 1000)
        delay = max(1, delay - duration)
        self.after(delay, self._play)


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

        self._configure_style()
        self._actions = ("view", "boundingbox", "extractbox", "mask", "landmarks")
        self._buttons = self._add_buttons()
        self._selected_action = tk.StringVar()
        self._selected_action.set("view")

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
        return {action[0]: action for action in self._actions}

    @property
    def _helptext(self):
        """ dict: `button key`: `button helptext`. The help text to display for each button. """
        return dict(view="View alignments (V)",
                    boundingbox="Bounding box editor (B)",
                    extractbox="Edit the size and orientation of the existing alignments (E)",
                    mask="Mask editor (M)",
                    landmarks="Individual landmark point editor (L)")

    def _configure_style(self):
        """ Configure background color for Actions widget """
        style = ttk.Style()
        style.configure("actions.TFrame", background='#d3d3d3')
        self.config(style="actions.TFrame")

    def _add_buttons(self):
        """ Add the action buttons to the Display window.

        Returns
        -------
        dict:
            The action name and its associated button.
        """
        style = ttk.Style()
        style.configure("actions_selected.TButton", relief="flat", background="#bedaf1")
        style.configure("actions_deselected.TButton", relief="flat")

        buttons = dict()
        for action in self.key_bindings.values():
            if action == "view":
                btn_style = "actions_selected.TButton"
                state = (["pressed", "focus"])
            else:
                btn_style = "actions_deselected.TButton"
                state = (["!pressed", "!focus"])

            button = ttk.Button(self,
                                image=get_images().icons[action],
                                command=lambda t=action: self.on_click(t),
                                style=btn_style)
            button.state(state)
            button.pack()
            Tooltip(button, text=self._helptext[action])
            buttons[action] = button
        return buttons

    def on_click(self, action):
        """ Click event for all of the buttons.

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
        self.update_idletasks()
        self._selected_action.set(action)


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
        self._actions = actions
        self._tk_action_var = tk_action_var
        self._image = self.create_image(self._frames.display_dims[0] / 2,
                                        self._frames.display_dims[1] / 2,
                                        image=self._frames.current_display_frame,
                                        anchor=tk.CENTER)
        self._editors = self._get_editors()
        self._add_callbacks()
        self._update_active_display()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def selected_action(self):
        """str: The name of the currently selected Editor action """
        return self._tk_action_var.get()

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

    def send_frame_to_bottom(self):
        """ Sent the background frame to the bottom of the stack """
        self.tag_lower(self._image)

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
        needs_update = self._frames.tk_update
        needs_update.trace("w", self._update_display)
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
        self._frames.tk_update.set(False)
        self.update_idletasks()

    def refresh_display_image(self):
        """ Update the displayed frame """
        self.itemconfig(self._image, image=self._frames.current_display_frame)


class Options(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Control panel options for currently displayed Editor.

    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the control panel options
    display_frame: :class:`DisplayFrame`
        The frame that holds the editors
    """
    def __init__(self, parent, display_frame):
        super().__init__(parent)
        self.pack(side=tk.LEFT, fill=tk.BOTH)
        self._display_frame = display_frame
        self._control_panels = self._initialize()
        self._set_tk_callbacks()
        self._update_options()

    def _initialize(self):
        """ Initialize all of the control panels, then display the default panel.

        Adds the control panel to :attr:`_control_panels` and sets the traceback to update
        display when a panel option has been changed.

        Notes
        -----
        All panels must be initialized at the beginning so that the global format options are not
        reset to default when the editor is first selected.

        The Traceback must be set after the panel has first been packed as otherwise it interferes
        with the loading of the faces pane.
        """
        panels = dict()
        for name, editor in self._display_frame.editors.items():
            logger.debug("Initializing control panel for '%s' editor", name)
            controls = editor.controls
            panel = ControlPanel(self, controls["controls"],
                                 option_columns=2,
                                 header_text=controls["header"],
                                 blank_nones=False,
                                 label_width=18)
            panel.pack_forget()
            panels[name] = panel
        return panels

    def _set_tk_callbacks(self):
        """ Sets the callback to change to the relevant control panel options when the selected
        editor is changed, and the display update on panel option change."""
        self._display_frame.tk_selected_action.trace("w", self._update_options)
        seen_controls = set()
        for name, editor in self._display_frame.editors.items():
            for ctl in editor.controls["controls"]:
                if ctl in seen_controls:
                    # Some controls are re-used (annotation format), so skip if trace has already
                    # been set
                    continue
                logger.debug("Adding control update callback: (editor: %s, control: %s)",
                             name, ctl.title)
                seen_controls.add(ctl)
                ctl.tk_var.trace("w", lambda *e: self._display_frame.tk_update.set(True))

    def _update_options(self, *args):  # pylint:disable=unused-argument
        """ Update the control panel display for the current editor.

        If the options have not already been set, then adds the control panel to
        :attr:`_control_panels`. Displays the current editor's control panel

        Parameters
        ----------
        args: tuple
            Unused but required for tkinter variable callback
        """
        self._clear_options_frame()
        editor = self._display_frame.tk_selected_action.get()
        logger.debug("Displaying control panel for editor: '%s'", editor)
        self._control_panels[editor].pack(expand=True, fill=tk.BOTH)

    def _clear_options_frame(self):
        """ Hides the currently displayed control panel """
        for editor, panel in self._control_panels.items():
            if panel.winfo_ismapped():
                logger.debug("Hiding control panel for: %s", editor)
                panel.pack_forget()


class FacesFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The faces display frame (bottom section of GUI).

    Parameters
    ----------
    parent: :class:`tkinter.PanedWindow`
        The paned window that the faces frame resides in
    faces: :class:`tools.manual.lib_manual.FaceCache`
        The faces cache that holds the aligned faces
    """
    def __init__(self, parent, faces):
        logger.debug("Initializing %s: (parent: %s, faces: %s)",
                     self.__class__.__name__, parent, faces)
        super().__init__(parent)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._faces = faces

        self._faces_frame = ttk.Frame(self)
        self._faces_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas = FacesViewer(self._faces_frame, self._faces)
        scrollbar_width = self._add_scrollbar()
        self._canvas.load_faces(self.winfo_width() - scrollbar_width)

        logger.debug("Initialized %s", self.__class__.__name__)

    def _add_scrollbar(self):
        """ Add a scrollbar to the faces frame """
        logger.debug("Add Config Scrollbar")
        scrollbar = ttk.Scrollbar(self._faces_frame, command=self._canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.config(yscrollcommand=scrollbar.set)
        self.bind("<Configure>", self._update_scrollbar)
        logger.debug("Added Config Scrollbar")
        self.update_idletasks()  # Update so scrollbar width is correct
        return scrollbar.winfo_width()

    def _update_scrollbar(self, event):  # pylint: disable=unused-argument
        """ Update the faces frame scrollbar """
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))


class FacesViewer(tk.Canvas):   # pylint:disable=too-many-ancestors
    """ Annotation onto tkInter Canvas.

    Parameters
    ----------
    parent: :class:`tkinter.ttk.Frame`
        The parent frame for the canvas
    faces: :class:`tools.manual.lib_manual.FaceCache`
        The faces cache that holds the aligned faces
    """
    def __init__(self, parent, faces):
        logger.debug("Initializing %s: (parent: %s, faces: %s)", self.__class__.__name__, parent,
                     faces)
        super().__init__(parent, bd=0, highlightthickness=0)
        self.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, anchor=tk.E)
        self.parent = parent
        self._faces = faces
        logger.debug("Initialized %s", self.__class__.__name__)

    def load_faces(self, frame_width):
        """ Load the faces into the Faces Canvas in a background thread.

        Parameters
        ----------
        frame_width: int
            The width of the :class:`tkinter.ttk.Frame` that holds this canvas """
        self._faces.load_faces(self, frame_width)


class Aligner():
    """ Handles the extraction pipeline for retrieving the alignment landmarks

    Parameters
    ----------
    alignments: :class:`Aligner`
        The alignments cache object for the manual tool
    """
    def __init__(self):
        self._alignments = None
        self._aligner = None
        self._init_thread = self._background_init_aligner()

    @property
    def _in_queue(self):
        """ :class:`queue.Queue` - The input queue to the aligner. """
        return self._aligner.input_queue

    @property
    def _feed_face(self):
        """ :class:`plugins.extract.pipeline.ExtractMedia`: The current face for feeding into the
        aligner, formatted for the pipeline """
        return ExtractMedia(self._alignments.frames.current_meta_data["filename"],
                            self._alignments.frames.current_frame,
                            detected_faces=[self._alignments.current_face])

    @property
    def is_initialized(self):
        """ bool: ``True`` if the aligner has completed initialization otherwise ``False``. """
        thread_is_alive = self._init_thread.is_alive()
        if thread_is_alive:
            self._init_thread.check_and_raise_error()
        else:
            self._init_thread.join()
        return not thread_is_alive

    def _background_init_aligner(self):
        """ Launch the aligner in a background thread so we can run other tasks whilst
        waiting for initialization """
        thread = MultiThread(self._init_aligner,
                             thread_count=1,
                             name="{}.init_aligner".format(self.__class__.__name__))
        thread.start()
        return thread

    def _init_aligner(self):
        """ Initialize Aligner in a background thread, and set it to :attr:`_aligner`. """
        logger.debug("Initialize Aligner")
        # TODO FAN
        # TODO Normalization option
        aligner = Extractor(None, "cv2-dnn", None, multiprocess=True, normalize_method="hist")
        # Set the batchsize to 1
        aligner.set_batchsize("align", 1)
        aligner.launch()
        logger.debug("Initialized Extractor")
        self._aligner = aligner

    def link_alignments(self, alignments):
        """ Add the :class:`AlignmentsData` object as a property of the aligner.

        Parameters
        ----------
        alignments: :class:`AlignmentsData`
            The alignments cache object for the manual tool
        """
        self._alignments = alignments

    def get_landmarks(self):
        """ Feed the detected face into the alignment pipeline and retrieve the landmarks

        Returns
        -------
        :class:`numpy.ndarray`
            The 68 point landmark alignments
        """
        self._in_queue.put(self._feed_face)
        detected_face = next(self._aligner.detected_faces()).detected_faces[0]
        return detected_face.landmarks_xy
