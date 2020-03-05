#!/usr/bin/env python3
""" Face viewer for the manual adjustments tool """
import logging
import platform
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

from lib.gui.custom_widgets import RightClickMenu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO Make it so user can't save until faces are loaded (so alignments dict doesn't change)


class FacesViewerLoader():
    """ Loads the faces into the Faces Viewer as they are read from the input source. """
    def __init__(self, canvas, faces_cache, frame_count, progress_bar, enable_buttons_callback):
        self._canvas = canvas
        self._faces_cache = faces_cache
        self._frame_count = frame_count
        self._progress_bar = progress_bar
        self._enable_buttons_callback = enable_buttons_callback
        self._progress_bar.start(mode="determinate")
        self._load_faces(0, 0)

    def _load_faces(self, faces_seen, frame_index):
        """ Set the number of columns based on the holding frame width and face size.
        Load the faces into the Faces Canvas in a background thread.

        Parameters
        ----------
        frame_width: int
            The width of the :class:`tkinter.ttk.Frame` that holds this canvas """
        self._update_progress(frame_index)
        tk_faces, frames_count = self._convert_faces_to_photoimage(frame_index)
        frame_landmarks = self._faces_cache.mesh_landmarks[frame_index:frames_count + frame_index]

        for faces, mesh_landmarks in zip(tk_faces, frame_landmarks):
            for face, landmarks in zip(faces, mesh_landmarks):
                coords = self._canvas.coords_from_index(faces_seen)
                self._canvas.new_objects.create(coords, face, landmarks, frame_index,
                                                is_multi=len(faces) > 1)
                if coords[0] == 0:  # Resize canvas on new line
                    self._canvas.configure(scrollregion=self._canvas.bbox("all"))
                faces_seen += 1
            frame_index += 1

        if self._faces_cache.is_initialized and frame_index == self._frame_count:
            self._on_load_complete()
        else:
            self._canvas.after(1000, self._load_faces, faces_seen, frame_index)

    def _update_progress(self, frame_index):
        """ Update the progress on load. """
        position = frame_index + 1
        progress = int(round((position / self._frame_count) * 100))
        msg = "Loading Faces: {}/{} - {}%".format(position, self._frame_count, progress)
        self._progress_bar.progress_update(msg, progress)

    def _convert_faces_to_photoimage(self, frame_index):
        """ Retrieve latest loaded faces and convert to :class:`PIL.ImakeTk.PhotoImage`. """
        update_faces = self._faces_cache.tk_faces[frame_index:]
        frames_count = len(update_faces)
        tk_faces = [[ImageTk.PhotoImage(Image.fromarray(img)) for img in faces]
                    for faces in update_faces]
        self._faces_cache.tk_faces[frame_index:frames_count + frame_index] = tk_faces
        return tk_faces, frames_count

    def _on_load_complete(self):
        """ Actions to perform once the faces have finished loading into the canvas """
        # TODO Enable saving
        self._canvas._filters["current_display"] = "AllFrames"
        self._canvas.active_display.initialize()
        for frame_idx, faces in enumerate(self._canvas._alignments.updated_alignments):
            if faces is None:
                continue
            image_ids = self._canvas.find_withtag("image_{}".format(frame_idx))
            existing_count = len(image_ids)
            new_count = len(faces)
            self._on_load_remove_faces(existing_count, new_count, frame_idx)

            for face_idx, face in enumerate(faces):
                objects = self._canvas.get_tk_face_and_landmarks(detected_face=face)
                if face_idx + 1 > existing_count:
                    self._on_load_add_face(frame_idx, objects[0], objects[1])
                else:
                    self._on_load_update_face(image_ids[face_idx],
                                              frame_idx,
                                              face_idx,
                                              objects[1],
                                              objects[1])
        self._enable_buttons_callback()
        # TODO Move this to canvas, as it checks whether it's still loading anyway?
        self._canvas.tk_control_colors["Mesh"].trace("w", self._canvas.update_mesh_color)
        self._progress_bar.stop()
        self._canvas.switch_filter()
        self._canvas.set_selected()

    def _on_load_remove_faces(self, existing_count, new_count, frame_index):
        """ Remove any faces that have been deleted whilst face viewer was loading. """
        if existing_count <= new_count:
            return
        for face_idx in range(new_count, existing_count):
            logger.debug("Deleting face at index %s for frame %s", face_idx, frame_index)
            self._canvas.delete_face_at_index_by_frame(frame_index, face_idx)

    def _on_load_add_face(self, frame_index, tk_face, mesh_landmarks):
        """ Add a face that has been been added face viewer was loading. """
        logger.debug("Adding new face for frame %s", frame_index)
        self._faces_cache.tk_faces[frame_index].append(tk_face)
        self._faces_cache.mesh_landmarks[frame_index].append(mesh_landmarks)
        next_frame_idx = self._canvas.get_next_frame_idx(frame_index)
        self._canvas.active_display.add_face(tk_face, frame_index, next_frame_idx)

    def _on_load_update_face(self, image_id, frame_index, face_index, tk_face, mesh_landmarks):
        """ Add a face that has been been added face viewer was loading. """
        logger.debug("Updating face id %s for frame %s", face_index, frame_index)
        self._faces_cache.tk_faces[frame_index][face_index] = tk_face
        self._faces_cache.mesh_landmarks[frame_index][face_index] = mesh_landmarks
        self._canvas.itemconfig(image_id, image=tk_face)
        coords = self._canvas.coords(image_id)
        mesh_tag = self._canvas.find_withtag("mesh_{}".format(frame_index))
        mesh_ids = self._canvas.mesh_ids_for_face(face_index, mesh_tag)
        for points, item_id in zip(mesh_landmarks["landmarks"], mesh_ids):
            self._canvas.coords(item_id, *(points + coords).flatten())


class ObjectCreator():
    """ Creates objects and annotations for the Faces Viewer. """
    def __init__(self, canvas):
        self._canvas = canvas
        self._object_types = ("image", "mesh")
        self._current_face_id = 0

    def create(self, coordinates, tk_face, mesh_landmarks, frame_index, is_multi=False):
        """ Create all if the annotations for a single Face Viewer face.

        Parameters
        ----------
        coordinates: tuple
            The top left (x, y) coordinates for the annotations' position in the Faces Viewer
        tk_face: :class:`PIL.ImageTk.PhotoImage`
            The face to be used for the image annotation
        mesh_landmarks: dict
            A dictionary containing the keys `landmarks` holding a `list` of :class:`numpy.ndarray`
            objects and `is_poly` containing a `list` of `bool` types corresponding to the
            `landmarks`
            indicating whether a line or polygon should be created for each mesh annotation.
        mesh_color: str
            The hex code holding the color that the mesh should be displayed as
        frame_index: int
            The frame index that this object appears in
        is_multi: bool
            ``True`` if there are multiple faces in the given frame, otherwise ``False``.
            Default: ``False``

        Returns
        -------
        image_id: int
            The item id of the newly created face
        mesh_ids: list
            List of item ids for the newly created mesh

        """
        tags = {obj: self._get_viewer_tags(obj, frame_index, is_multi)
                for obj in self._object_types}
        image_id = self._canvas.create_image(*coordinates,
                                             image=tk_face,
                                             anchor=tk.NW,
                                             tags=tags["image"])
        mesh_ids = self.create_mesh_annotations(self._canvas.get_muted_color("Mesh"),
                                                mesh_landmarks,
                                                coordinates,
                                                tags["mesh"])
        self._current_face_id += 1
        return image_id, mesh_ids

    def _get_viewer_tags(self, object_type, frame_index, is_multi):
        """ Obtain the tags for a Faces Viewer object.

        Parameters
        ----------
        object_type: str
            The type of object that these tags will be associated with
        frame_index: int
            The frame index that this object appears in
        is_multi: bool
            ``True`` if there are multiple faces in the given frame, otherwise ``False``

        Returns
        -------
        list
            The list of tags for the Faces Viewer object
        """
        tags = ["viewer",
                "viewer_{}".format(object_type),
                "frame_id_{}".format(frame_index),
                "face_id_{}".format(self._current_face_id),
                "{}_{}".format(object_type, frame_index),
                "{}_face_id_{}".format(object_type, self._current_face_id)]
        if is_multi:
            tags.extend(["multi", "multi_{}".format(object_type)])
        else:
            tags.append("not_multi")
        logger.trace(tags)
        return tags

    def create_mesh_annotations(self, color, mesh_landmarks, offset, tag):
        """ Create the coordinates for the face mesh. """
        retval = []
        state = "normal" if self._canvas.optional_annotation == "landmarks" else "hidden"
        kwargs = dict(polygon=dict(fill="", outline=color), line=dict(fill=color))
        for is_poly, landmarks in zip(mesh_landmarks["is_poly"], mesh_landmarks["landmarks"]):
            key = "polygon" if is_poly else "line"
            tags = tag + ["mesh_{}".format(key)]
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            coords = (landmarks + offset).flatten()
            retval.append(obj(*coords, state=state, width=1, tags=tags, **obj_kwargs))
        return retval


class HoverBox():
    """ Hover box for FacesViewer """
    def __init__(self, canvas, face_size):
        self._canvas = canvas
        self._frames = canvas._frames
        self._face_size = face_size
        self._box = self._canvas.create_rectangle(0, 0, face_size, face_size,
                                                  outline="#FFFF00",
                                                  width=2,
                                                  state="hidden")
        self._canvas.bind("<Leave>", lambda e: self._clear())
        self._canvas.bind("<Motion>", self.on_hover)
        self._canvas.bind("<ButtonPress-1>", lambda e: self._select_frame())

    def on_hover(self, event):  # pylint: disable=unused-argument
        """ The mouse cursor display as bound to the mouses <Motion> event.
        The canvas only displays faces, so if the mouse is over an object change the cursor
        otherwise use default.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused for default tracking, but available for specific editor
            tracking.
        """
        coords = (self._canvas.canvasx(event.x), self._canvas.canvasy(event.y))
        item_id = next((idx for idx in self._canvas.find_overlapping(*coords, *coords)
                        if self._canvas.type(idx) == "image"), None)
        if item_id is None:
            self._clear()
            self._canvas.config(cursor="")
            return
        frame_id = self._canvas.frame_index_from_object(item_id)
        if item_id is None or frame_id == self._frames.tk_position.get():
            self._clear()
            self._canvas.config(cursor="")
            return
        self._canvas.config(cursor="hand1")
        self._highlight(item_id)

    def _clear(self):
        """ Hide the hovered box and clear the :attr:`_hovered` attribute """
        if self._canvas.itemcget(self._box, "state") != "hidden":
            self._canvas.itemconfig(self._box, state="hidden")

    def _highlight(self, item_id):
        """ Display the box around the face the mouse is over

        Parameters
        ----------
        item_id: int
            The tkinter canvas object id
        """
        top_left = np.array(self._canvas.coords(item_id))
        coords = (*top_left, *top_left + self._face_size)
        self._canvas.coords(self._box, *coords)
        self._canvas.itemconfig(self._box, state="normal")
        self._canvas.tag_raise(self._box)

    def _select_frame(self):
        """ Go to the frame corresponding to the mouse click location in the faces window. """
        item_id = next((idx for idx in self._canvas.find_withtag("current")), None)
        if item_id is None:
            return
        frame_id = self._canvas.frame_index_from_object(item_id)
        if frame_id is None or frame_id == self._frames.tk_position.get():
            return
        transport_id = self._canvas.transport_index_from_frame_index(frame_id)
        if transport_id is None:
            return
        self._frames.stop_playback()
        self._frames.tk_transport_position.set(transport_id)


class ContextMenu():
    """ Right click menu for the Faces Viewer """
    def __init__(self, canvas):
        self._canvas = canvas
        self._faces_cache = canvas._faces_cache
        self._menu = RightClickMenu(["Delete Face"], [self._delete_face])
        self._face_id = None
        self._canvas.bind("<Button-2>" if platform.system() == "Darwin" else "<Button-3>",
                          self._pop_menu)

    def _pop_menu(self, event):
        """ Pop up the context menu"""
        if not self._faces_cache.is_initialized:
            return
        coords = (self._canvas.canvasx(event.x), self._canvas.canvasy(event.y))
        self._face_id = next((idx for idx in self._canvas.find_overlapping(*coords, *coords)
                              if self._canvas.type(idx) == "image"), None)
        if self._face_id is None:
            logger.trace("No valid item under mouse")
            return
        logger.trace("Popping right click menu")
        self._menu.popup(event)

    def _delete_face(self):
        """ Delete the selected face on a right click mouse delete action. """
        # TODO Fix for new layout
        self._canvas.delete_face_from_viewer(self._face_id)
        self._face_id = None


class ActiveFrame():
    """ Holds the objects and handles faces for the currently selected frame. """
    def __init__(self, canvas):
        self._canvas = canvas
        self._alignments = canvas._alignments
        self._size = canvas._faces_cache.size
        self._highlighter = Highlighter(canvas)
        self._tk_faces = None
        self._mesh_landmarks = None
        self._image_ids = None
        self._mesh_ids = None
        self._frame_index = 0
        self._face_count = 0

    @property
    def _face_index(self):
        """ int: The currently selected face index """
        return self._alignments.face_index

    @property
    def face_count(self):
        """ int: The count of faces in the currently selected frame. """
        return self._face_count

    @property
    def frame_index(self):
        """ int: The current frame index. """
        return self._frame_index

    def set_selected(self, tk_faces, mesh_landmarks, frame_index):
        """ Set the currently selected frame's objects """
        self._tk_faces = tk_faces
        self._mesh_landmarks = mesh_landmarks
        self._image_ids = list(self._canvas.find_withtag("image_{}".format(frame_index)))
        self._mesh_ids = list(self._canvas.find_withtag("mesh_{}".format(frame_index)))
        self._frame_index = frame_index
        self._face_count = len(self._image_ids)
        self._highlighter.highlight_selected(self._image_ids, self._mesh_landmarks)

    def reload_annotations(self):
        """ Reload the currently selected annotations on an add/remove face. """
        self._image_ids = list(self._canvas.find_withtag("image_{}".format(self._frame_index)))
        self._mesh_ids = list(self._canvas.find_withtag("mesh_{}".format(self._frame_index)))
        self._face_count = len(self._image_ids)
        self._highlighter.highlight_selected(self._image_ids, self._mesh_landmarks)

    def update(self, tk_face, mesh_landmarks):
        """ Update the currently selected face on editor update """
        if self._face_count != 0:
            self._update_face(tk_face)
            self._update_mesh(mesh_landmarks)
        self._highlighter.highlight_selected(self._image_ids, self._mesh_landmarks)

    def _update_face(self, tk_face):
        """ Update the face photo image and the face object id """
        self._tk_faces[self._face_index] = tk_face
        self._canvas.itemconfig(self._image_ids[self._face_index], image=tk_face)

    def _update_mesh(self, mesh_landmarks):
        """ Update the optional mesh annotation """
        coords = self._canvas.coords(self._image_ids[self._face_index])
        mesh_ids = self._canvas.mesh_ids_for_face(self._face_index, self._mesh_ids)
        for points, item_id in zip(mesh_landmarks["landmarks"], mesh_ids):
            self._canvas.coords(item_id, *(points + coords).flatten())
        self._mesh_landmarks[self._face_index] = mesh_landmarks

    def add_face(self, tk_face, mesh_landmarks):
        """ Add a face to the currently selected frame, """
        logger.debug("Adding face to frame(frame_id: %s new face_count: %s)",
                     self._frame_index, self._face_count + 1)
        self._tk_faces.append(tk_face)
        self._mesh_landmarks.append(mesh_landmarks)

    def remove_face(self):
        """ Remove a face from the currently selected frame. """
        face_idx = self._alignments.get_removal_index()
        logger.debug("Removing face for frame %s at index: %s", self._frame_index, face_idx)
        del self._tk_faces[face_idx]
        del self._mesh_landmarks[face_idx]
        return face_idx


class Highlighter():
    """ Highlights the currently active frame's faces """
    def __init__(self, canvas):
        logger.debug("Initializing: %s: (canvas: %s)",
                     self.__class__.__name__, canvas)
        self._size = canvas._faces_cache.size
        self._canvas = canvas

        self._face_count = 0
        self._hidden_highlighters_count = 0
        self._boxes = []
        self._meshes = []
        logger.debug("Initialized: %s", self.__class__.__name__,)

    @property
    def _highlighter_count(self):
        """ int: The number of highlighter objects currently available. """
        return len(self._boxes)

    def highlight_selected(self, image_ids, mesh_landmarks):
        """ Highlight the currently selected faces """
        self._face_count = len(image_ids)
        self._create_new_highlighters(mesh_landmarks)
        self._hide_unused_highlighters()
        if self._face_count == 0:
            return

        boxes = self._boxes[:self._face_count]
        meshes = self._meshes[:self._face_count]
        for image_id, landmarks, box, mesh in zip(image_ids, mesh_landmarks, boxes, meshes):
            top_left = np.array(self._canvas.coords(image_id))
            un_hide = self._highlight_box(box, top_left)
            self._highlight_mesh(mesh, landmarks["landmarks"], top_left, un_hide)
            self._hidden_highlighters_count -= 1 if un_hide else 0

        top = self._canvas.coords(self._boxes[0])[1] / self._canvas.bbox("all")[3]
        if top != self._canvas.yview()[0]:
            self._canvas.yview_moveto(top)

    # << Add new highlighters >> #
    def _create_new_highlighters(self, landmarks):
        """ Add new highlight annotations if there are more faces in the frame than
        current highlighters. """
        new_highlighter_count = max(0, self._face_count - self._highlighter_count)
        logger.trace("new_highlighter_count: %s", new_highlighter_count)
        if new_highlighter_count == 0:
            return
        for idx in range(new_highlighter_count):
            self._create_highlight_box()
            self._create_highlight_mesh(landmarks[idx])
            self._hidden_highlighters_count += 1

    def _create_highlight_box(self):
        """ Create a new highlight box and append to :attr:`_boxes`. """
        box = self._canvas.create_rectangle(0, 0, 1, 1, outline="#00FF00", width=2, state="hidden")
        logger.trace("Created new highlight_box: %s", box)
        self._boxes.append(box)

    def _create_highlight_mesh(self, landmarks):
        """ Create new highlight mesh annotations and append to :attr:`_meshes`. """
        mesh_color = self._canvas.control_colors["Mesh"]
        kwargs = dict(polygon=dict(fill="", outline=mesh_color),
                      line=dict(fill=mesh_color))
        mesh_ids = []
        for is_poly, pts in zip(landmarks["is_poly"], landmarks["landmarks"]):
            key = "polygon" if is_poly else "line"
            tag = ["highlight_mesh_{}".format(key)]
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            mesh_ids.append(obj(*pts.flatten(), state="hidden", width=1, tags=tag, **obj_kwargs))
        logger.trace("Created new highlight_mesh: %s", mesh_ids)
        self._meshes.append(mesh_ids)

    # << Hide unused highlighters >> #
    def _hide_unused_highlighters(self):
        """ Hide any highlighters that are not required for the current frame """
        hide_count = self._highlighter_count - self._face_count - self._hidden_highlighters_count
        hide_count = max(0, hide_count)
        logger.trace("hide_highlighter_count: %s", hide_count)
        if hide_count == 0:
            return
        hide_slice = slice(self._face_count, self._face_count + hide_count)
        for box, mesh in zip(self._boxes[hide_slice], self._meshes[hide_slice]):
            logger.trace("Hiding highlight box: %s", box)
            self._canvas.itemconfig(box, state="hidden")
            logger.trace("Hiding highlight mesh: %s", mesh)
            for mesh_id in mesh:
                self._canvas.itemconfig(mesh_id, state="hidden")
            self._hidden_highlighters_count += 1

    # << Highlight current faces >> #

    def _highlight_box(self, box, top_left):
        """ Locate and display the given highlight box """
        un_hide = False
        coords = (*top_left, *top_left + self._size)
        logger.trace("Highlighting box (id: %s, coords: %s)", box, coords)
        self._canvas.coords(box, *coords)
        if self._canvas.itemcget(box, "state") == "hidden":
            un_hide = True
            self._canvas.itemconfig(box, state="normal")
        return un_hide

    def _highlight_mesh(self, mesh_ids, landmarks, top_left, un_hide):
        """ Locate and display the given mesh annotations """
        logger.trace("Highlighting mesh (id: %s, top_left: %s, unhide: %s)",
                     mesh_ids, top_left, un_hide)
        for points, mesh_id in zip(landmarks, mesh_ids):
            self._canvas.coords(mesh_id, *(points + top_left).flatten())
            if un_hide:
                self._canvas.itemconfig(mesh_id, state="normal")


class FaceFilter():
    """ Base class for different faces view filters.

    All Filter views inherit from this class. Handles the layout of faces and annotations in the
    face viewer.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    toggle_tags: dict or ``None``, optional
        Tags for toggling optional annotations. Set to ``None`` if there are no annotations to
        be toggled for the selected filter. Default: ``None``
    """
    def __init__(self, canvas, toggle_tags=None):
        logger.debug("Initializing: %s: (canvas: %s, toggle_tags: %s)",
                     self.__class__.__name__, canvas, toggle_tags)
        self._toggle_tags = toggle_tags
        self._canvas = canvas
        self._tk_position = canvas._frames.tk_position
        self._size = canvas._faces_cache.size
        self._item_ids = dict(image_ids=[], mesh_ids=[])
        self._removed_image_index = -1
        self._frame_faces_change = 0

        self._mesh_landmarks = canvas._faces_cache.mesh_landmarks

        # Set and unset during :func:`initialize` and :func:`de-initialize`
        self._updated_frames = []
        self._tk_position_callback = None
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def _image_ids(self):
        """ list: The canvas item ids for the face images. """
        return self._item_ids["image_ids"]

    @property
    def _mesh_ids(self):
        """ list: The canvas item ids for the optional mesh annotations. """
        return self._item_ids["mesh_ids"]

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        Override for filter specific hiding criteria.
        """
        raise NotImplementedError

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        Override for filter specific hiding criteria.
        """
        raise NotImplementedError

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        Override for filter specific actions
        """
        raise NotImplementedError

    def initialize(self):
        """ Initialize the viewer for the selected filter type.

        Hides annotations and faces that should not be displayed for the current filter.
        Displays and moves the faces to the correct position on the canvas based on which faces
        should be displayed.
        """
        self._set_object_display_state()
        self._set_display_objects()
        for idx, image_id in enumerate(self._image_ids):
            old_position = np.array(self._canvas.coords(image_id), dtype="int")
            new_position = self._canvas.coords_from_index(idx)
            offset = new_position - old_position
            if not offset.any():
                continue
            self._canvas.move(self._canvas.face_id_from_object(image_id), *offset)
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._tk_position_callback = self._tk_position.trace("w", self._on_frame_change)
        self._updated_frames = [self._tk_position.get()]

    def add_face(self, tk_face, frame_index, next_frame_index):
        """ Display a new face in the correct location and move subsequent faces to their new
        location.

        Parameters
        ----------
        tk_face: :class:`PIL.ImageTk.PhotoImage`
            The new face that is to be added to the canvas
        frame_index: int
            The frame index that the face is to be added to
        """
        if not self._image_ids:
            display_idx = 0
        else:
            image_ids = self._canvas.find_withtag("image_{}".format(frame_index))
            if image_ids:
                display_idx = self._image_ids.index(image_ids[-1]) + 1
            else:
                display_idx = self._removed_image_index
        # Update layout - Layout must be updated first so space is made for the new face
        self._update_layout(display_idx, is_insert=True)

        coords = self._canvas.coords_from_index(display_idx)
        logger.debug("display_idx: %s, coords: %s", display_idx, coords)
        # Create new annotations
        image_id, mesh_ids = self._canvas.new_objects.create(
            coords,
            tk_face,
            self._mesh_landmarks[frame_index][-1],
            frame_index)
        # Insert annotations into correct position in object tracking
        self._image_ids.insert(display_idx, image_id)
        mesh_idx_offset = display_idx * self._canvas.items_per_mesh
        self._mesh_ids[mesh_idx_offset:mesh_idx_offset] = mesh_ids
        # Update multi tags
        self._update_multi_tags_on_add(frame_index, image_id, mesh_ids)
        # Place faces in correct position in stack
        if next_frame_index is not None:
            self._canvas.tag_lower("frame_id_{}".format(frame_index),
                                   "frame_id_{}".format(next_frame_index))
        self._frame_faces_change += 1

    def _update_multi_tags_on_add(self, frame_index, image_id, mesh_ids):
        """ Update the tags indicating whether this frame contains multiple faces.

        Parameters
        ----------
        frame_index: int
            The frame index that the tags are to be updated for
        image_id: int
            The item id of the newly added face
        mesh_ids: list
            The item ids of the newly added mesh annotations
        """
        lookup_tag = "frame_id_{}".format(frame_index)
        num_faces = len(self._canvas.find_withtag(lookup_tag))
        if num_faces == 2:
            self._canvas.dtag(lookup_tag, tag="not_multi")
        if num_faces == 1:
            self._canvas.addtag_withtag(lookup_tag, "not_multi")
        else:
            self._canvas.addtag_withtag(lookup_tag, "multi")
            self._canvas.addtag_withtag(image_id, "multi_image")
            for mesh_id in mesh_ids:
                self._canvas.addtag_withtag(mesh_id, "multi_mesh")
        return lookup_tag

    def remove_face(self, frame_index, face_index):
        """ Remove a face at the given location and update subsequent faces to the
        correct location.

        """
        display_idx = self._delete_annotations_and_update(frame_index, face_index)
        # Track the last removal index in case of adding faces back in after removing all faces
        # TODO Check this logic
        self._removed_image_index = display_idx
        self._frame_faces_change -= 1

    def _delete_annotations_and_update(self, frame_index, face_index):
        """ Delete the face annotations for the given frame and face indices and update the display
        to reflect the existing faces.

        Parameters
        ----------
        frame_index: int
            The frame index that the face has been removed from
        face_index: int
            The index of the face within the given frame that is to be removed
        """
        image_id = self._canvas.find_withtag("image_{}".format(frame_index))[face_index]
        display_idx = self._image_ids.index(image_id)
        logger.debug("image_id: %s, display_idx: %s", image_id, display_idx)

        self._canvas.delete(self._canvas.face_id_from_object(image_id))
        del self._image_ids[display_idx]
        mesh_idx_offset = display_idx * self._canvas.items_per_mesh
        self._mesh_ids[mesh_idx_offset:mesh_idx_offset + self._canvas.items_per_mesh] = []
        self._update_multi_tags_on_remove(frame_index)
        self._update_layout(display_idx, is_insert=False)
        return display_idx

    def _update_multi_tags_on_remove(self, frame_index):
        """ Update the tags indicating whether this frame contains multiple faces on face removal

        Parameters
        ----------
        frame_index: int
            The frame index that the tags are to be updated for
        """
        lookup_tag = "frame_id_{}".format(frame_index)
        num_faces = len(self._canvas.find_withtag(lookup_tag))
        if num_faces == 1:
            self._canvas.dtag(lookup_tag, tag="multi")
            self._canvas.dtag("image_{}".format(frame_index), tag="multi_image")
            self._canvas.dtag("mesh_{}".format(frame_index), tag="multi_mesh")
            self._canvas.addtag_withtag(lookup_tag, "not_multi")

    def _update_layout(self, starting_object_index, is_insert=True):
        """ Reposition faces and annotations on the canvas after a face has been added or removed.

        The faces are moved in bulk in 3 groups: The row that the face is being added to or removed
        from is shifted left or right. The first or last column is shifted to the other side of the
        viewer and 1 row up or down. Finally the block of faces that have not been covered by the
        previous shifts are moved left or right one face.

        Parameters
        ----------
        starting_object_index: int
            The starting absolute face index that new locations should be calculated for
        is_insert: bool, optional
            ``True`` if adjusting display for an added face, ``False`` if adjusting for a removed
            face. Default: ``True``
        """
        # TODO addtag_enclosed vs addtag_overlapping - The mesh going out of face box problem
        top_bulk_delta = np.array((self._size, 0))
        col_delta = np.array((-(self._canvas.column_count - 1) * self._size, self._size))
        if not is_insert:
            top_bulk_delta *= -1
            col_delta *= -1
        self._tag_objects_to_move(starting_object_index, is_insert)
        self._canvas.move("move_top", *top_bulk_delta)
        self._canvas.move("move_col", *col_delta)
        self._canvas.move("move_bulk", *top_bulk_delta)

        self._canvas.dtag("move_top", "move_top")
        self._canvas.dtag("move_col", "move_col")
        self._canvas.dtag("move_bulk", "move_bulk")

        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    # TODO Adding faces to last frame breaks this
    def _tag_objects_to_move(self, start_index, is_insert):
        """ Tag the 3 object groups that require moving.

        Any hidden annotations are set to normal so that the :func:`tkinter.Canvas.find_enclosed`
        function picks them up for moving. They are immediately hidden again.

        Parameters
        ----------
        starting_index: int
            The starting absolute face index that new locations should be calculated for
        is_insert: bool
            ``True`` if adjusting display for an added face, ``False`` if adjusting for a removed
            face.
         """
        # Display hidden annotations so they get tagged
        hidden_tags = [val for key, val in self._toggle_tags.items()
                       if key != self._canvas.optional_annotation]
        for tag in hidden_tags:
            self._canvas.itemconfig(tag, state="normal")
        new_row_start = start_index == self._canvas.column_count - 1
        first_face_xy = self._canvas.coords(self._image_ids[start_index])
        first_row_y = first_face_xy[1] - self._size if new_row_start else first_face_xy[1]
        last_col_x = (self._canvas.column_count - 1) * self._size
        logger.debug("is_insert: %s, start_index: %s, new_row_start: %s, first_face_xy: %s, "
                     "first_row_y: %s, last_col_x: %s", is_insert, start_index, new_row_start,
                     first_face_xy, first_row_y, last_col_x)
        # Top Row
        if not new_row_start:
            # Skip top row shift if starting index is a new row
            br_top_xy = (last_col_x if is_insert else last_col_x + self._size,
                         first_row_y + self._size)
            logger.debug("first row: (top left: %s, bottom right:%s)", first_face_xy, br_top_xy)
            self._canvas.addtag_enclosed("move_top", *first_face_xy, *br_top_xy)
        # First or last column (depending on delete or insert)
        tl_col_xy = (last_col_x if is_insert else 0,
                     first_row_y if is_insert else first_row_y + self._size)
        br_col_xy = (tl_col_xy[0] + self._size, self._canvas.bbox("all")[3])
        logger.debug("end column: (top left: %s, bottom right:%s)", tl_col_xy, br_col_xy)
        self._canvas.addtag_enclosed("move_col", *tl_col_xy, *br_col_xy)
        # Bulk faces
        tl_bulk_xy = (0 if is_insert else self._size, first_row_y + self._size)
        br_bulk_xy = (last_col_x if is_insert else last_col_x + self._size,
                      self._canvas.bbox("all")[3])
        logger.debug("bulk: (top left: %s, bottom right:%s)", tl_bulk_xy, br_bulk_xy)
        self._canvas.addtag_enclosed("move_bulk", *tl_bulk_xy, *br_bulk_xy)
        # Re-hide hidden annotations
        for tag in hidden_tags:
            self._canvas.itemconfig(tag, state="hidden")

    def delete_face_from_viewer(self, frame_index, face_index):
        """ Delete a face when called from the face viewer.

        It is highly likely that this is called when not displaying the frame that the face is
        deleted from, so updates are run immediately rather than on a frame change.

        Parameters
        ----------
        frame_index: int
            The frame index that the face should be deleted for
        face_index: int
            The face index to remove the face for
        """
        self._delete_annotations_and_update(frame_index, face_index)
        self._updated_frames.append(frame_index)

    def toggle_annotation(self):
        """ Toggle additional object annotations on or off. """
        if self._toggle_tags is None:
            return
        display = self._canvas.optional_annotation
        if display is not None:
            self._canvas.itemconfig(self._toggle_tags[display], state="normal")
        else:
            for tag in self._toggle_tags.values():
                self._canvas.itemconfig(tag, state="hidden")

    @staticmethod
    def _get_toggle_item_ids(objects, display):
        """ Return the item_ids in a single list for items that are to be toggled.

        Parameters
        ----------
        objects: dict
            The objects dictionary for the current face
        display: str or ``None``
            The key for the object annotation that is to be toggled or ``None`` if annotations
            are to be hidden

        Returns
        -------
        list
            The list of canvas object ids that are to be toggled
        """
        if display is not None:
            retval = objects[display] if isinstance(objects[display], list) else [objects[display]]
        else:
            retval = []
            for key, val in objects.items():
                if key == "image_id":
                    continue
                retval.extend(val if isinstance(val, list) else [val])
        logger.trace(retval)
        return retval

    def de_initialize(self):
        """ Unloads the Face Filter on filter type change. """
        self._tk_position.trace_vdelete("w", self._tk_position_callback)
        self._tk_position_callback = None
        self._updated_frames = []


class FilterAllFrames(FaceFilter):
    """ The Frames that have Faces viewer

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    def __init__(self, canvas):
        super().__init__(canvas, dict(landmarks="viewer_mesh"))

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        All viewer annotations should be show
        """
        annotation = self._canvas.optional_annotation
        if annotation == "landmarks":
            self._canvas.itemconfig("viewer", state="normal")
        else:
            self._canvas.itemconfig("viewer_image", state="normal")

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        All image and mesh annotations should be loaded
        """
        self._item_ids["image_ids"] = list(self._canvas.find_withtag("viewer_image"))
        self._item_ids["mesh_ids"] = list(self._canvas.find_withtag("viewer_mesh"))

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        All Frames has no frame change specific actions.
        """
        self._updated_frames = [self._tk_position.get()]


class FilterNoFaces(FaceFilter):
    """ The Frames with No Faces viewer.

    Extends the base filter to track when faces have been added to a frame, so that the display
    can be updated accordingly.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    def __init__(self, canvas):
        self._added_objects = []
        super().__init__(canvas)

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        All viewer annotations should be show
        """
        self._canvas.itemconfig("viewer", state="hidden")

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        Empty lists are set as no annotations will meet criteria
        """
        self._item_ids["image_ids"] = []
        self._item_ids["mesh_ids"] = []

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        If faces have been added to the current frame, hide the new faces and clear
        the viewer objects.
        """
        if self._frame_faces_change != 0:
            self._canvas.itemconfig("frame_id_{}".format(self._updated_frames[0]), state="hidden")
            self._item_ids["image_ids"] = []
            self._item_ids["mesh_ids"] = []
        self._frame_faces_change = 0
        self._updated_frames = [self._tk_position.get()]


class FilterMultipleFaces(FaceFilter):
    """ The Frames with Multiple Faces viewer.

    Parameters
    ----------
    face_cache: :class:`FaceCache`
        The main Face Viewers face cache, that holds all of the display items and annotations
    """
    def __init__(self, canvas):
        super().__init__(canvas, dict(landmarks="multi_mesh"))

    def _set_object_display_state(self):
        """ Hide annotations that are not relevant for this filter type and show those that are

        All viewer annotations should be show
        """
        self._canvas.itemconfig("not_multi", state="hidden")
        annotation = self._canvas.optional_annotation
        if annotation == "landmarks":
            self._canvas.itemconfig("multi", state="normal")
        else:
            self._canvas.itemconfig("multi_image", state="normal")

    def _set_display_objects(self):
        """ Set the :attr:`_image_ids` and :attr:`_mesh_ids` for the current filter

        All image and mesh annotations should be loaded
        """
        self._item_ids["image_ids"] = list(self._canvas.find_withtag("multi_image"))
        self._item_ids["mesh_ids"] = list(self._canvas.find_withtag("multi_mesh"))

    def _on_frame_change(self, *args):  # pylint:disable=unused-argument
        """ Callback to be executed whenever the frame is changed.

        Multiple faces display can be impacted by either deleting faces in the frame or by deleting
        faces from the frame viewer. Frames which have had faces deleted from the face viewer will
        have been added to :attr:`update_frames`, so an additional check has to be made here when
        changing frames to ensure that the display is updated accordingly.

        If the face count for any of the impacted frames is no longer 2 or more, remove the frame
        from :attr:`_object_indices`, hide any existing sole faces and update the layout.
        """
        logger.trace("frame_faces_change: %s, updated_frames: %s",
                     self._frame_faces_change, self._updated_frames)
        if self._frame_faces_change == 0 and len(self._updated_frames) == 1:
            self._updated_frames = [self._tk_position.get()]
            return
        for frame_index in self._updated_frames:
            image_ids = self._canvas.find_withtag("image_{}".format(frame_index))
            if len(image_ids) < 2:
                self._canvas.itemconfig("frame_id_{}".format(frame_index, state="hidden"))
            if len(image_ids) == 1:
                # Remove the final face from the display and update
                display_idx = self._image_ids.index(image_ids[0])
                del self._image_ids[display_idx]
                mesh_idx_offset = display_idx * self._canvas.items_per_mesh
                self._mesh_ids[mesh_idx_offset:mesh_idx_offset + self._canvas.items_per_mesh] = []
                self._update_layout(display_idx, is_insert=False)
        self._updated_frames = [self._tk_position.get()]
        self._frame_faces_change = 0
