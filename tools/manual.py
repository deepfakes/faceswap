#!/usr/bin/env python3
""" Tool to manually interact with the alignments file using visual tools """
import logging
import os

import tkinter as tk
from tkinter import ttk
from functools import partial
from time import time, sleep

import cv2
from PIL import Image, ImageTk

from lib.alignments import Alignments
from lib.faces_detect import DetectedFace
from lib.gui.control_helper import set_slider_rounding
from lib.gui.custom_widgets import Tooltip
from lib.gui.utils import get_images, get_config, initialize_config, initialize_images
from lib.image import ImagesLoader
from lib.multithreading import MultiThread
from plugins.extract.pipeline import Extractor, ExtractMedia

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
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        super().__init__()
        extractor = Aligner()
        self._frames = FrameNavigation(arguments.frames)
        self._wait_for_extractor(extractor)

        alignments = AlignmentsCache(arguments.alignments_path, self._frames, extractor)

        self._initialize_tkinter()
        self._containers = self._create_containers()

        self._display = DisplayFrame(self._containers["top"], self._frames, alignments)

        lbl = ttk.Label(self._containers["top"], text="Top Right")
        self._containers["top"].add(lbl)

        self._set_layout()
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

        top = tk.PanedWindow(main,
                             sashrelief=tk.RIDGE,
                             sashwidth=2,
                             sashpad=4,
                             orient=tk.HORIZONTAL,
                             name="pw_top")
        main.add(top)

        bottom = ttk.Frame(main, name="frame_bottom")
        main.add(bottom)
        logger.debug("Created containers")
        return dict(main=main, top=top, bottom=bottom)

    def _set_layout(self):
        """ Place the sashes of the paned window """
        self.update_idletasks()
        self._containers["top"].sash_place(0, (self._frames.display_dims[0]) + 8, 1)
        self._containers["main"].sash_place(0, 1, self._frames.display_dims[1] + 72)

    def _handle_key_press(self, event):
        """ Keyboard shortcuts """
        bindings = dict(left=self._frames.decrement_frame,
                        right=self._frames.increment_frame,
                        space=self._display.handle_play_button,
                        home=self._frames.set_first_frame,
                        end=self._frames.set_last_frame)
        key = event.keysym
        if key.lower() in bindings:
            self.focus_set()
            bindings[key.lower()]()

    def process(self):
        """ The entry point for the Visual Alignments tool from :file:`lib.tools.cli`.

        Launch the tkinter Visual Alignments Window and run main loop.
        """
        lbl = ttk.Label(self._containers["bottom"], text="Bottom")
        lbl.pack()
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
        parent.add(self)
        self._frames = frames
        self._canvas = Viewer(self, alignments, self._frames)

        self._transport_frame = ttk.Frame(self)
        self._transport_frame.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)

        self._add_nav()
        self._play_button = self._add_transport()
        logger.debug("Initialized %s", self.__class__.__name__)

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
        frame = ttk.Frame(self._transport_frame)
        frame.pack(side=tk.BOTTOM, fill=tk.X)
        icons = get_images().icons

        for action in ("play", "beginning", "prev", "next", "end", "speed"):
            if action == "play":
                play_button = ttk.Button(frame,
                                         image=icons[action],
                                         width=14,
                                         command=self.handle_play_button)
                play_button.pack(side=tk.LEFT, padx=(0, 6))
                Tooltip(play_button, text="Play/Pause (SPACE)")
                self._frames.tk_is_playing.trace("w", self._play)
            elif action in ("prev", "next"):
                cmd_action = "decrement" if action == "prev" else "increment"
                cmd = getattr(self._frames, "{}_frame".format(cmd_action))
                if action == "prev":
                    helptext = "Go to Previous Frame (LEFT)"
                else:
                    helptext = "Go to Next Frame (RIGHT)"
                btn = ttk.Button(frame, image=icons[action], width=14, command=cmd)
                btn.pack(side=tk.LEFT)
                Tooltip(btn, text=helptext)
            elif action in ("beginning", "end"):
                lookup = ("First", "HOME") if action == "beginning" else ("Last", "END")
                helptext = "Go to {} Frame ({})".format(*lookup)
                cmd = getattr(self._frames, "set_{}_frame".format(lookup[0].lower()))
                btn = ttk.Button(frame, image=icons[action], width=14, command=cmd)
                btn.pack(side=tk.LEFT)
                Tooltip(btn, text=helptext)
            elif action == "speed":
                self._add_speed_combo(frame)
        return play_button

    def handle_play_button(self):
        """ Handle the play button.

        Switches the :attr:`_frames.is_playing` variable.
        """
        is_playing = self._frames.tk_is_playing.get()
        self._frames.tk_is_playing.set(not is_playing)

    def _add_speed_combo(self, frame):
        """ Adds the speed control Combo box and links to
        :attr:`_frames.tk_playback_speed`. """
        tk_var = self._frames.tk_playback_speed
        tk_var.set("Standard")
        sframe = ttk.Frame(frame)
        sframe.pack(side=tk.RIGHT)
        lbl = ttk.Label(sframe, text="Playback Speed")
        lbl.pack(side=tk.LEFT, padx=(0, 5))
        combo = ttk.Combobox(sframe,
                             textvariable=tk_var,
                             state="readonly",
                             values=["Standard", "Max"],
                             width=8)
        combo.pack(side=tk.RIGHT)
        Tooltip(combo, text="Set Playback Speed")

    def _play(self, *args):  # pylint:disable=unused-argument
        """ Play the video file at the selected speed """
        start = time()
        is_playing = self._frames.tk_is_playing.get()
        icon = "pause" if is_playing else "play"
        self._play_button.config(image=get_images().icons[icon])

        if not is_playing:
            logger.debug("Pause detected. Stopping.")
            return

        self._frames.increment_frame(is_playing=True)
        if self._frames.tk_playback_speed.get() == "Standard":
            delay = self._frames.delay
            duration = int((time() - start) * 1000)
            delay = max(1, delay - duration)
        else:
            delay = 1
        self.after(delay, self._play)


class Viewer(tk.Canvas):  # pylint:disable=too-many-ancestors
    """ Annotation onto tkInter Canvas """
    def __init__(self, parent, alignments, frames):
        logger.debug("Initializing %s: (parent: %s, alignments: %s, frames: %s)",
                     self.__class__.__name__, parent, alignments, frames)
        super().__init__(parent, bd=0, highlightthickness=0)
        self.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor=tk.E)

        self._alignments = alignments
        self._frames = frames
        self._colors = dict(red="#ff0000",
                            green="#00ff00",
                            blue="#0000ff",
                            cyan="#00ffff",
                            yellow="#ffff00",
                            magenta="#ff00ff")
        self._image = None
        self._annotations = dict()

        self._bounding_box = None
        self._extract_box = None
        self._landmarks = None
        self._mesh = None
        self._drag_data = dict()
        self._add_initial_frame()
        self._add_callback()
        self._add_mouse_tracking()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _scaling(self):
        """ float: The scaling factor for the currently displayed frame """
        return self._frames.current_scale

    @property
    def _bounding_box_layout(self):
        """ tuple: The layout order of tkinter canvas bounding box points """
        return ("left", "top", "right", "bottom")

    @property
    def _bounding_box_corner_order(self):
        """ dict: The position index of bounding box corners """
        return {0: ("top", "left"),
                1: ("bottom", "left"),
                2: ("top", "right"),
                3: ("bottom", "right")}

    @property
    def _bounding_boxes(self):
        """ list: List of (`Left`, `Top`, `Right`, `Bottom`) tuples for each displayed face's
        bounding box. """
        return [self.coords(face[0]) for face in self._annotations["bounding_box"]]

    @property
    def _bounding_box_anchors(self):
        """ list: List of bounding box anchors for the corners of each face's bounding box. """
        return [[self.coords(obj) for obj in face[1:]]
                for face in self._annotations["bounding_box"]]

    @property
    def _bounding_box_points(self):
        """ list: List of bounding box tuples for each face's bounding box """
        return [((self.coords(obj[0])[0], self.coords(obj[0])[1]),
                 (self.coords(obj[0])[0], self.coords(obj[0])[3]),
                 (self.coords(obj[0])[2], self.coords(obj[0])[1]),
                 (self.coords(obj[0])[2], self.coords(obj[0])[3]))
                for obj in self._annotations["bounding_box"]]

    def _add_initial_frame(self):
        """ Adds the initial items to the canvas. """
#        self._image = self.create_image(self._frames.display_dims[0] // 2,
#                                        self._frames.display_dims[1] // 2,
#                                        image=self._frames.current_frame,
#                                        anchor=tk.CENTER)
        self._image = self.create_image(0, 0,
                                        image=self._frames.current_display_frame,
                                        anchor=tk.NW)
        self._annotations["mesh"] = self._update_mesh()
        self._annotations["extract_box"] = self._update_extract_box()
        self._annotations["landmarks"] = self._update_landmarks()
        self._annotations["bounding_box"] = self._update_bounding_box()

    def _add_callback(self):
        needs_update = self._frames.tk_update
        needs_update.trace("w", self._update_display)

    def _add_mouse_tracking(self):
        self.bind("<Motion>", self._update_cursor)
        self.bind("<ButtonPress-1>", self._drag_start)
        self.bind("<ButtonRelease-1>", self._drag_stop)
        self.bind("<B1-Motion>", self._drag)

    def _update_display(self, *args):  # pylint:disable=unused-argument
        """ Update the display on frame cache update """
        if not self._frames.tk_update.get():
            return
        self._clear_annotations()
        self.itemconfig(self._image, image=self._frames.current_display_frame)
        self._annotations["mesh"] = self._update_mesh()
        self._annotations["extract_box"] = self._update_extract_box()
        self._annotations["landmarks"] = self._update_landmarks()
        if not self._drag_data:
            self._annotations["bounding_box"] = self._update_bounding_box()
        self._frames.tk_update.set(False)
        self.update_idletasks()

    def _clear_annotations(self):
        """ Removes all currently drawn annotations """
        for title, annotation in self._annotations.items():
            # TODO this check will need to be more robust when adding new elements
            if title == "bounding_box" and self._drag_data:
                continue
            for face in annotation:
                for instance in face:
                    self.delete(instance)

    def _update_bounding_box(self):
        """ Draw the bounding box around faces """
        color = self._colors["blue"]
        thickness = 1
        faces = []
        for face in self._alignments.current_faces:
            bbox = []
            box = (face.left * self._scaling,
                   face.top * self._scaling,
                   face.right * self._scaling,
                   face.bottom * self._scaling)
            corners = ((box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3]))
            bbox.append(self.create_rectangle(*box, outline=color, width=thickness))
            radius = thickness * 5
            for cnr in corners:
                anc = (cnr[0] - radius, cnr[1] - radius, cnr[0] + radius, cnr[1] + radius)
                bbox.append(self.create_oval(*anc,
                                             outline=color,
                                             fill="gray",
                                             width=thickness,
                                             activefill="white"))
            faces.append(bbox)
        return faces

    def _update_extract_box(self):
        """ Draw the extracted face box """
        color = self._colors["green"]
        thickness = 1
        faces = []
        # TODO FIX THIS TEST
        #  if not all(face.original_roi for face in self._alignments.current_faces):
        #      return extract_box
        for idx, face in enumerate(self._alignments.current_faces):
            extract_box = []
            logger.trace("Drawing Extract Box: (idx: %s, roi: %s)", idx, face.original_roi)
            box = face.original_roi.flatten() * self._scaling
            top_left = box[:2] - 10
            extract_box.append(self.create_text(*top_left,
                                                fill=color,
                                                font=("Default", 20, "bold"),
                                                text=str(idx)))
            extract_box.append(self.create_polygon(*box, fill="", outline=color, width=thickness))
            faces.append(extract_box)
        return faces

    def _update_landmarks(self):
        """ Draw the facial landmarks """
        color = self._colors["red"]
        radius = 1
        faces = []
        for face in self._alignments.current_faces:
            landmarks = []
            for landmark in face.landmarks_xy:
                box = (landmark * self._scaling).astype("int32")
                bbox = (box[0] - radius, box[1] - radius, box[0] + radius, box[1] + radius)
                landmarks.append(self.create_oval(*bbox, outline=color, fill=color, width=radius))
            faces.append(landmarks)
        return faces

    def _update_mesh(self):
        """ Draw the facial landmarks """
        color = self._colors["cyan"]
        thickness = 1
        facial_landmarks_idxs = dict(mouth=(48, 68),
                                     right_eyebrow=(17, 22),
                                     left_eyebrow=(22, 27),
                                     right_eye=(36, 42),
                                     left_eye=(42, 48),
                                     nose=(27, 36),
                                     jaw=(0, 17),
                                     chin=(8, 11))
        faces = []
        for face in self._alignments.current_faces:
            mesh = []
            landmarks = face.landmarks_xy
            logger.trace("Drawing Landmarks Mesh: (landmarks: %s, color: %s, thickness: %s)",
                         landmarks, color, thickness)
            for key, val in facial_landmarks_idxs.items():
                pts = (landmarks[val[0]:val[1]] * self._scaling).astype("int32").flatten()
                if key in ("right_eye", "left_eye", "mouth"):
                    mesh.append(self.create_polygon(*pts, fill="", outline=color, width=thickness))
                else:
                    mesh.append(self.create_line(*pts, fill=color, width=thickness))
            faces.append(mesh)
        return faces

    # Mouse Callbacks
    def _update_cursor(self, event):
        if any(bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]
               for face in self._bounding_box_anchors for bbox in face):
            # Bounding box anchors
            idx = [idx for face in self._bounding_box_anchors
                   for idx, bbox in enumerate(face)
                   if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]][0]
            self.config(cursor="{}_{}_corner".format(*self._bounding_box_corner_order[idx]))
        elif any(bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]
                 for bbox in self._bounding_boxes):
            self.config(cursor="fleur")
        else:
            self.config(cursor="")

    def _drag_start(self, event):
        """ Collect information on start of drag """
        click_object = self._get_click_object(event)
        if click_object is None:
            self._drag_data = dict()
            return
        object_type, self._drag_data["index"] = click_object
        if object_type == "bounding_box_anchor":
            indices = [(face_idx, pnt_idx)
                       for face_idx, face in enumerate(self._bounding_box_anchors)
                       for pnt_idx, bbox in enumerate(face)
                       if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]][0]
            self._drag_data["objects"] = self._annotations["bounding_box"][indices[0]]
            self._drag_data["corner"] = self._bounding_box_corner_order[indices[1]]
            self._drag_data["callback"] = self._resize_bounding_box
        elif object_type == "bounding_box":
            face_idx = [idx for idx, bbox in enumerate(self._bounding_boxes)
                        if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]][0]
            self._drag_data["objects"] = self._annotations["bounding_box"][face_idx]
            self._drag_data["current_location"] = (event.x, event.y)
            self._drag_data["callback"] = self._move_bounding_box

    def _get_click_object(self, event):
        """ Return the object type and index that has been clicked on.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event

        Returns
        -------
        tuple
            (`type`, `index`) The type of object being clicked on and the index of the face.
            If no object clicked on then return value is ``None``
        """
        retval = None
        for idx, face in enumerate(self._bounding_box_anchors):
            if any(bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]
                   for bbox in face):
                retval = "bounding_box_anchor", idx
        if retval is not None:
            return retval

        for idx, bbox in enumerate(self._bounding_boxes):
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                retval = "bounding_box", idx

        return retval

    def _drag_stop(self, event):  # pylint:disable=unused-argument
        """ Reset the :attr:`_drag_data` dict

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused but required
        """
        self._drag_data = dict()

    def _drag(self, event):
        """ Drag the bounding box and its anchors to current mouse position.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if not self._drag_data:
            return
        self._drag_data["callback"](event)

    def _resize_bounding_box(self, event):
        """ Resizes a bounding box on an anchor drag event

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        radius = 4  # TODO Variable
        rect = self._drag_data["objects"][0]
        box = list(self.coords(rect))
        # Switch top/bottom and left/right and set partial so indices match and we don't
        # need branching logic for min/max.
        limits = (partial(min, box[2] - 20),
                  partial(min, box[3] - 20),
                  partial(max, box[0] + 20),
                  partial(max, box[1] + 20))
        rect_xy_indices = [self._bounding_box_layout.index(pnt)
                           for pnt in self._drag_data["corner"]]
        box[rect_xy_indices[1]] = limits[rect_xy_indices[1]](event.x)
        box[rect_xy_indices[0]] = limits[rect_xy_indices[0]](event.y)
        self.coords(rect, *box)
        corners = ((box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3]))
        for idx, cnr in enumerate(corners):
            anc = (cnr[0] - radius, cnr[1] - radius, cnr[0] + radius, cnr[1] + radius)
            self.coords(self._drag_data["objects"][idx + 1], *anc)
        self._alignments.set_current_bounding_box(self._drag_data["index"],
                                                  *self._coords_to_bounding_box(box))

    def _move_bounding_box(self, event):
        """ Moves the bounding box on a bounding box drag event """
        shift_x = event.x - self._drag_data["current_location"][0]
        shift_y = event.y - self._drag_data["current_location"][1]
        selected_objects = self._drag_data["objects"]
        for obj in selected_objects:
            self.move(obj, shift_x, shift_y)
        box = self.coords(selected_objects[0])
        self._alignments.set_current_bounding_box(self._drag_data["index"],
                                                  *self._coords_to_bounding_box(box))
        self._drag_data["current_location"] = (event.x, event.y)

    def _coords_to_bounding_box(self, coords):
        """ Converts tkinter coordinates to :class:`lib.faces_detect.DetectedFace` bounding
        box format, scaled up for feeding the model.

        Returns
        -------
        tuple
            The (`x`, `width`, `y`, `height`) integer points of the bounding box.

        """
        return (int(round(coords[0] / self._scaling)),
                int(round((coords[2] - coords[0]) / self._scaling)),
                int(round(coords[1] / self._scaling)),
                int(round((coords[3] - coords[1]) / self._scaling)))


class FrameNavigation():
    """Handles the return of the correct frame for the GUI.

    Parameters
    ----------
    frames_location: str
        The path to the input frames
    """
    def __init__(self, frames_location):
        logger.debug("Initializing %s: (frames_location: '%s')",
                     self.__class__.__name__, frames_location)
        self._loader = ImagesLoader(frames_location, fast_count=False, queue_size=1)
        self._delay = int(round(1000 / self._loader.fps))
        self._meta = dict()
        self._current_idx = 0
        self._current_scale = 1.0
        self._tk_vars = self._set_tk_vars()
        self._current_frame = None
        self._current_display_frame = None
        self._display_dims = (960, 540)
        self._set_current_frame(initialize=True)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_video(self):
        """ bool: 'True' if input is a video 'False' if it is a folder. """
        return self._loader.is_video

    @property
    def location(self):
        """ str: The input folder or video location. """
        return self._loader.location

    @property
    def filename_list(self):
        """ list: List of filenames in correct frame order. """
        return self._loader.file_list

    @property
    def frame_count(self):
        """ int: The total number of frames """
        return self._loader.count

    @property
    def current_meta_data(self):
        """ dict: The current cache item for the current location. Keys are `filename`,
        `display_dims`, `scale` and `interpolation`. """
        return self._meta[self.tk_position.get()]

    @property
    def current_scale(self):
        """ float: The scaling factor for the currently displayed frame """
        return self._current_scale

    @property
    def current_frame(self):
        """ :class:`numpy.ndarray`: The currently loaded, full frame. """
        return self._current_frame

    @property
    def current_display_frame(self):
        """ :class:`ImageTk.PhotoImage`: The currently loaded frame, formatted and sized
        for display. """
        return self._current_display_frame

    @property
    def delay(self):
        """ int: The number of milliseconds between updates to playback the video at the
        correct fps """
        return self._delay

    @property
    def display_dims(self):
        """ tuple: The (`width`, `height`) of the display image. """
        return self._display_dims

    @property
    def tk_playback_speed(self):
        """ :class:`tkinter.IntVar`: Multiplier to playback speed. """
        return self._tk_vars["speed"]

    @property
    def tk_position(self):
        """ :class:`tkinter.IntVar`: The current frame position. """
        return self._tk_vars["position"]

    @property
    def tk_is_playing(self):
        """ :class:`tkinter.BooleanVar`: Whether the stream is currently playing. """
        return self._tk_vars["is_playing"]

    @property
    def tk_update(self):
        """ :class:`tkinter.BooleanVar`: Whether the display needs to be updated. """
        return self._tk_vars["updated"]

    def _set_tk_vars(self):
        """ Set the initial tkinter variables and add traces. """
        logger.debug("Setting tkinter variables")
        position = tk.IntVar()
        position.set(self._current_idx)
        position.trace("w", self._set_current_frame)

        speed = tk.StringVar()

        is_playing = tk.BooleanVar()
        is_playing.set(False)

        updated = tk.BooleanVar()
        updated.set(False)

        retval = dict(position=position, is_playing=is_playing, speed=speed, updated=updated)
        logger.debug("Set tkinter variables: %s", retval)
        return retval

    def _set_current_frame(self, *args,  # pylint:disable=unused-argument
                           initialize=False):
        """ Set the currently loaded frame to :attr:`_current_frame`

        Parameters
        ----------
        args: tuple
            Required for event callback. Unused.
        initialize: bool, optional
            ``True`` if initializing for the first frame to be displayed otherwise ``False``.
            Default: ``False``
        """
        position = self.tk_position.get()
        if not initialize and position == self._current_idx:
            return
        filename, frame = self._loader.image_from_index(position)
        self._add_meta_data(position, frame, filename)
        self._current_frame = frame
        display = cv2.resize(self._current_frame,
                             self.current_meta_data["display_dims"],
                             interpolation=self.current_meta_data["interpolation"])[..., 2::-1]
        self._current_display_frame = ImageTk.PhotoImage(Image.fromarray(display))
        self._current_idx = position
        self._current_scale = self.current_meta_data["scale"]
        self.tk_update.set(True)

    def _add_meta_data(self, position, frame, filename):
        """ Adds the metadata for the current frame to :attr:`meta`.

        Parameters
        ----------
        position: int
            The current frame index
        frame: :class:`numpy.ndarray`
            The current frame
        filename: str
            The filename for the current frame

        """
        if position in self._meta:
            return
        scale = min(self._display_dims[0] / frame.shape[1],
                    self._display_dims[1] / frame.shape[0])
        self._meta[position] = dict(
            scale=scale,
            interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
            display_dims=(int(round(frame.shape[1] * scale)),
                          int(round(frame.shape[0] * scale))),
            filename=filename)

    def increment_frame(self, is_playing=False):
        """ Update :attr:`self.current_frame` to the next frame.

        Parameters
        ----------
        is_playing: bool, optional
            ``True`` if the frame is being incremented because the Play button has been pressed.
            ``False`` if incremented for other reasons. Default: ``False``
        """
        position = self.tk_position.get()
        if position == self.frame_count - 1:
            logger.trace("End of stream. Not incrementing")
            if self.tk_is_playing.get():
                self.tk_is_playing.set(False)
            return
        if not is_playing and self.tk_is_playing.get():
            self.tk_is_playing.set(False)
        self.tk_position.set(position + 1)

    def decrement_frame(self):
        """ Update :attr:`self.current_frame` to the previous frame """
        position = self.tk_position.get()
        if self.tk_is_playing.get():
            # Stop playback
            self.tk_is_playing.set(False)
        if position == 0:
            logger.trace("Beginning of stream. Not decrementing")
            return
        self.tk_position.set(position - 1)

    def set_first_frame(self):
        """ Load the first frame """
        if self.tk_is_playing.get():
            self.tk_is_playing.set(False)
        self.tk_position.set(0)

    def set_last_frame(self):
        """ Load the last frame """
        if self.tk_is_playing.get():
            self.tk_is_playing.set(False)
        self.tk_position.set(self.frame_count - 1)


class AlignmentsCache():
    """ Holds the alignments and annotations.

    Parameters
    ----------
    alignments_path: str
        Full path to the alignments file. If empty string is passed then location is calculated
        from the source folder
    frames: :class:`FrameNavigation`
        The object that holds the cache of frames.
    """
    def __init__(self, alignments_path, frames, extractor):
        logger.debug("Initializing %s: (alignments_path: '%s')",
                     self.__class__.__name__, alignments_path)
        self.frames = frames
        self._alignments = self._get_alignments(alignments_path)
        self._tk_position = frames.tk_position
        self._face_index = 0
        self._extractor = extractor
        self._extractor.link_alignments(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def current_faces(self):
        """ list: list of the current :class:`lib.faces_detect.DetectedFace` objects. Returns
        modified alignments if they are modified, otherwise original saved alignments. """
        alignments = self._alignments[self.frames.current_meta_data["filename"]]
        # TODO use get and return a default for when alignments don't exist
        retval = alignments.get("new", alignments["saved"])
        return retval

    @property
    def current_face(self):
        """ :class:`lib.faces_detect.DetectedFace` The currently selected face """
        return self.current_faces[self._face_index]

    def set_current_bounding_box(self, index, pnt_x, width, pnt_y, height):
        """ Update the bounding box for the current alignments.

        Parameters
        ----------
        index: int
            The face index to set this bounding box for
        pnt_x: int
            The left point of the bounding box
        width: int
            The width of the bounding box
        pnt_y: int
            The top point of the bounding box
        height: int
            The height of the bounding box
        """
        self._face_index = index
        filename = self.frames.current_meta_data["filename"]
        if self._alignments[filename].get("new", None) is None:
            # Copy over saved alignments to new alignments
            self._alignments[filename]["new"] = self._alignments[filename]["saved"].copy()
        face = self.current_face
        face.x = pnt_x
        face.w = width
        face.y = pnt_y
        face.h = height
        face.mask = dict()
        face.landmarks_xy = self._extractor.get_landmarks()
        face.load_aligned(None, size=128, force=True)
        self.frames.tk_update.set(True)

    def _get_alignments(self, alignments_path):
        """ Get the alignments object.

        Parameters
        ----------
        alignments_path: str
            Full path to the alignments file. If empty string is passed then location is calculated
            from the source folder

        Returns
        -------
        dict
            `frame name`: list of :class:`lib.faces_detect.DetectedFace` for the current frame
        """
        if alignments_path:
            folder, filename = os.path.split(alignments_path, self.frames)
        else:
            filename = "alignments.fsa"
            if self.frames.is_video:
                folder, vid = os.path.split(os.path.splitext(self.frames.location)[0])
                filename = "{}_{}".format(vid, filename)
            else:
                folder = self.frames.location
        alignments = Alignments(folder, filename)
        faces = dict()
        for framename, items in alignments.data.items():
            faces[framename] = []
            this_frame_faces = []
            for item in items:
                face = DetectedFace()
                face.from_alignment(item)
                face.load_aligned(None, size=128)
                this_frame_faces.append(face)
            faces[framename] = dict(saved=this_frame_faces)
        return faces


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
        aligner = Extractor(None, "FAN", None, multiprocess=True, normalize_method="hist")
        # Set the batchsize to 1
        aligner.set_batchsize("align", 1)
        aligner.launch()
        logger.debug("Initialized Extractor")
        self._aligner = aligner

    def link_alignments(self, alignments):
        """ Add the :class:`AlignmentsCache` object as a property of the aligner.

        Parameters
        ----------
        alignments: :class:`AlignmentsCache`
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
