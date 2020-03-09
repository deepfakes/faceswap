#!/usr/bin/env python3
""" Face viewer for the manual adjustments tool """
import logging
import platform
import tkinter as tk

import numpy as np

from lib.gui.custom_widgets import RightClickMenu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO Make it so user can't save until faces are loaded (so alignments dict doesn't change)
# TODO Adding lots of faces during load leads to faces duplicating


class FacesViewerLoader():
    """ Loads the faces into the Faces Viewer as they are read from the input source. """
    def __init__(self, canvas, faces_cache, frame_count, progress_bar, enable_buttons_callback):
        self._canvas = canvas
        self._faces_cache = faces_cache
        self._alignments = canvas._alignments
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
        if frame_index == self._frame_count:
            self._on_load_complete()
        else:
            logger.trace("Refreshing...")
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
        tk_faces = [[tk.PhotoImage(data=strbyte) for strbyte in faces] for faces in update_faces]
        self._faces_cache.tk_faces[frame_index:frames_count + frame_index] = tk_faces
        return tk_faces, frames_count

    def _on_load_complete(self):
        """ Actions to perform once the faces have finished loading into the canvas """
        # TODO Enable saving
        for frame_idx, faces in enumerate(self._alignments.updated_alignments):
            if faces is None:
                continue
            image_ids = self._canvas.find_withtag("image_{}".format(frame_idx))
            existing_count = len(image_ids)
            new_count = len(faces)
            self._on_load_remove_faces(existing_count, new_count, frame_idx)
            for face_idx in range(new_count):
                if face_idx + 1 > existing_count:
                    self._canvas.update_face.add(frame_idx)
                else:
                    self._canvas.update_face.update(frame_idx, face_idx)
        self._alignments.tk_edited.set(False)
        self._enable_buttons_callback()
        self._canvas.update_mesh_color()
        self._progress_bar.stop()
        self._canvas.switch_filter()
        self._canvas.active_frame.reload_annotations()
        self._faces_cache.set_load_complete()

    def _on_load_remove_faces(self, existing_count, new_count, frame_index):
        """ Remove any faces that have been deleted whilst face viewer was loading. """
        if existing_count <= new_count:
            return
        for face_idx in range(new_count, existing_count):
            logger.debug("Deleting face at index %s for frame %s", face_idx, frame_index)
            self._canvas.update_face.remove(frame_index, face_idx)


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
        tk_face: :class:`tkinter.PhotoImage`
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
        logger.trace("coordinates: %s, tk_face: %s, mesh_landmarks: %s, frame_index: %s, "
                     "is_multi: %s", coordinates, tk_face, mesh_landmarks, frame_index, is_multi)
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
        state = "normal" if self._canvas.optional_annotations["mesh"] else "hidden"
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
        self._alignments = canvas._alignments
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
        transport_id = self._transport_index_from_frame_index(frame_id)
        if transport_id is None:
            return
        self._frames.stop_playback()
        self._frames.tk_transport_position.set(transport_id)

    def _transport_index_from_frame_index(self, frame_index):
        """ Retrieve the index in the filtered frame list for the given frame index. """
        frames_list = self._alignments.get_filtered_frames_list()
        retval = frames_list.index(frame_index) if frame_index in frames_list else None
        logger.trace("frame_index: %s, transport_index: %s", frame_index, retval)
        return retval


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
        self._canvas.update_face.remove_face_from_viewer(self._face_id)
        self._face_id = None


class ActiveFrame():
    """ Holds the objects and handles faces for the currently selected frame. """
    def __init__(self, canvas):
        self._canvas = canvas
        self._alignments = canvas._alignments
        self._faces_cache = canvas._faces_cache
        self._size = canvas._faces_cache.size
        self._tk_position = canvas._frames.tk_position
        self._highlighter = Highlighter(canvas)
        self._tk_position.trace("w", lambda *e: self.reload_annotations())
        self._alignments.tk_edited.trace("w", lambda *e: self._update())

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

    @property
    def _image_ids(self):
        """ tuple: The tkinter canvas image ids for the current frame's faces. """
        return self._canvas.find_withtag("image_{}".format(self._frame_index))

    @property
    def _mesh_ids(self):
        """ tuple: The tkinter canvas mesh ids for the current frame's faces. """
        return self._canvas.find_withtag("mesh_{}".format(self._frame_index))

    @property
    def _face_count(self):
        """ int: The number of faces in the current frame """
        return len(self._image_ids)

    @property
    def _tk_faces(self):
        """ list: The :class:`tkinter.PhotoImage` faces for the current frame """
        return self._faces_cache.tk_faces[self._frame_index]

    @property
    def _mesh_landmarks(self):
        """ TODO check if we can just get the landmarks here """
        return self._faces_cache.mesh_landmarks[self._frame_index]

    @property
    def _frame_index(self):
        """ int: The current frame position """
        return self._tk_position.get()

    def reload_annotations(self):
        """ Reload the currently selected annotations on an add/remove face. """
        if not self._faces_cache.is_initialized:
            return
        self._highlighter.highlight_selected(self._image_ids, self._mesh_ids, self._frame_index)

    def _update(self):
        """ Update the currently selected face in the active frame """
        if not self._alignments.tk_edited.get() or not self._faces_cache.is_initialized:
            return
        if self._add_remove_face():
            self.reload_annotations()
            return
        self._canvas.update_face.update(self._frame_index, self._face_index)
        self._highlighter.highlight_selected(self._image_ids, self._mesh_ids, self._frame_index)
        self._alignments.tk_edited.set(False)

    def _add_remove_face(self):
        """ add or remove a face for the current frame """
        alignment_faces = len(self._alignments.current_faces)
        if alignment_faces > self._canvas.active_frame.face_count:
            self._canvas.update_face.add(self._canvas.active_frame.frame_index)
            retval = True
        elif alignment_faces < self._canvas.active_frame.face_count:
            self._canvas.update_face.remove(self._canvas.active_frame.frame_index,
                                            self._alignments.get_removal_index())
            retval = True
        else:
            retval = False
        return retval


class Highlighter():
    """ Highlights the currently active frame's faces """
    def __init__(self, canvas):
        logger.debug("Initializing: %s: (canvas: %s)",
                     self.__class__.__name__, canvas)
        self._size = canvas._faces_cache.size
        self._canvas = canvas
        self._faces_cache = canvas._faces_cache
        self._tk_selected_editor = canvas._display_frame.tk_selected_action
        self._tk_selected_mask = canvas._display_frame.tk_selected_mask
        self._tk_optional_annotations = canvas._tk_optional_annotations
        self._face_count = 0
        self._frame_index = 0
        self._hidden_boxes_count = 0
        self._boxes = []
        self._prev_objects = dict()
        logger.debug("Initialized: %s", self.__class__.__name__,)

    @property
    def _boxes_count(self):
        """ int: The number of highlighter objects currently available. """
        return len(self._boxes)

    def highlight_selected(self, image_ids, mesh_ids, frame_index):
        """ Highlight the currently selected faces """
        self._face_count = len(image_ids)
        self._frame_index = frame_index
        self._create_new_boxes()
        self._revert_last_mask()
        self._hide_unused_boxes()
        self._revert_last_mesh()
        if self._face_count == 0:
            return

        self._highlight_mask()
        for image_id, box in zip(image_ids, self._boxes[:self._face_count]):
            top_left = np.array(self._canvas.coords(image_id))
            self._highlight_box(box, top_left)
        self._highlight_mesh(mesh_ids)

        top = self._canvas.coords(self._boxes[0])[1] / self._canvas.bbox("all")[3]
        if top != self._canvas.yview()[0]:
            self._canvas.yview_moveto(top)

    # << Add new highlighters >> #
    def _create_new_boxes(self):
        """ Add new highlight annotations if there are more faces in the frame than
        current highlighters. """
        new_boxes_count = max(0, self._face_count - self._boxes_count)
        logger.trace("new_boxes_count: %s", new_boxes_count)
        if new_boxes_count == 0:
            return
        for _ in range(new_boxes_count):
            box = self._canvas.create_rectangle(0, 0, 1, 1,
                                                outline="#00FF00", width=2, state="hidden")
            logger.trace("Created new highlight_box: %s", box)
            self._boxes.append(box)
            self._hidden_boxes_count += 1

    # << Hide unused highlighters >> #
    def _hide_unused_boxes(self):
        """ Hide any highlighters that are not required for the current frame """
        hide_count = self._boxes_count - self._face_count - self._hidden_boxes_count
        hide_count = max(0, hide_count)
        logger.trace("hide_boxes_count: %s", hide_count)
        if hide_count == 0:
            return
        hide_slice = slice(self._face_count, self._face_count + hide_count)
        for box in self._boxes[hide_slice]:
            logger.trace("Hiding highlight box: %s", box)
            self._canvas.itemconfig(box, state="hidden")
            self._hidden_boxes_count += 1

    def _revert_last_mesh(self):
        if self._prev_objects.get("mesh", None) is None:
            return
        color = self._canvas.get_muted_color("Mesh")
        kwargs = dict(polygon=dict(fill="", outline=color), line=dict(fill=color))
        state = "normal" if self._tk_optional_annotations["mesh"].get() else "hidden"
        for mesh_id in self._prev_objects["mesh"]:
            self._canvas.itemconfig(mesh_id, state=state, **kwargs[self._canvas.type(mesh_id)])
        self._prev_objects["mesh"] = None

    def _revert_last_mask(self):
        if (self._prev_objects.get("mask", None) is None
                or self._tk_optional_annotations["mask"].get()):
            return
        self._faces_cache.update_selected(self._prev_objects["mask"], None)
        self._prev_objects["mask"] = None

    # << Highlight current faces >> #
    def _highlight_box(self, box, top_left):
        """ Locate and display the given highlight box """
        coords = (*top_left, *top_left + self._size)
        logger.trace("Highlighting box (id: %s, coords: %s)", box, coords)
        self._canvas.coords(box, *coords)
        if self._canvas.itemcget(box, "state") == "hidden":
            self._hidden_boxes_count -= 1
            self._canvas.itemconfig(box, state="normal")

    def _highlight_mesh(self, mesh_ids):
        if self._tk_selected_editor.get() == "Mask":
            return
        color = self._canvas.control_colors["Mesh"]
        kwargs = dict(polygon=dict(fill="", outline=color),
                      line=dict(fill=color))
        for mesh_id in mesh_ids:
            self._canvas.itemconfig(mesh_id, **kwargs[self._canvas.type(mesh_id)], state="normal")
        self._prev_objects["mesh"] = mesh_ids

    def _highlight_mask(self):
        if self._tk_selected_editor.get() != "Mask" or self._tk_optional_annotations["mask"].get():
            return
        self._faces_cache.update_selected(self._frame_index, self._tk_selected_mask.get())
        self._prev_objects["mask"] = self._frame_index


class UpdateFace():
    """ Handles all adding, removing and updating of faces in a frame. """
    def __init__(self, canvas):
        self._canvas = canvas
        self._alignments = canvas._alignments
        self._faces_cache = canvas._faces_cache
        self._frames = canvas._frames

    # TODO moving to new frame and adding faces seems to mess up the tk_face of the existing face

    # << ADD FACE METHODS >> #
    def add(self, frame_index):
        """ Add a face to the faces_viewer """
        face_idx = self._alignments.face_count_per_index[frame_index] - 1
        logger.debug("Adding face to frame: (frame_index: %s, face_index: %s)",
                     frame_index, face_idx)
        # Add objects to cache
        tk_face, mesh_landmarks = self._canvas.get_tk_face_and_landmarks(frame_index, face_idx)
        self._faces_cache.add(frame_index, tk_face, mesh_landmarks)
        # Create new annotations
        image_id, mesh_ids = self._canvas.new_objects.create((0, 0),
                                                             tk_face,
                                                             mesh_landmarks,
                                                             frame_index)
        # Place in stack
        frame_tag = self._update_multi_tags(frame_index)
        self._place_in_stack(frame_index, frame_tag)
        # Update viewer
        self._canvas.active_filter.add_face(image_id, mesh_ids, mesh_landmarks["landmarks"])

    def _place_in_stack(self, frame_index, frame_tag):
        """ Place newly added faces in the correct location in the stack """
        next_frame_idx = self._get_next_frame_idx(frame_index)
        if next_frame_idx is not None:
            next_tag = "frame_id_{}".format(next_frame_idx)
            logger.debug("Lowering annotations for frame %s below frame %s", frame_tag, next_tag)
            self._canvas.tag_lower(frame_tag, next_tag)

    def _get_next_frame_idx(self, frame_index):
        """ Get the index of the next frame that has faces for placing newly added faces
        in the stack. """
        next_frame_idx = next((
            idx for idx, f_count in enumerate(self._alignments.face_count_per_index[frame_index:])
            if f_count > 0), None)
        if next_frame_idx is None:
            return None
        next_frame_idx += frame_index + 1
        logger.debug("Returning next frame with faces: %s for frame index: %s",
                     next_frame_idx, frame_index)
        return next_frame_idx

    # << REMOVE FACE METHODS >> #
    def remove_face_from_viewer(self, item_id):
        """ Delete a face from alignments and the Face Viewer for the given item id. """
        frame_idx = self._canvas.frame_index_from_object(item_id)
        face_idx = self._canvas.find_withtag("image_{}".format(frame_idx)).index(item_id)
        logger.debug("item_id: %s, frame_index: %s, face_index: %s", item_id, frame_idx, face_idx)
        self._alignments.delete_face_at_index_by_frame(frame_idx, face_idx)
        self.remove(frame_idx, face_idx)
        if frame_idx == self._frames.tk_position.get():
            self._frames.tk_update.set(True)
            self._canvas.active_frame.reload_annotations()

    def remove(self, frame_index, face_index):
        """ Remove a face. """
        logger.debug("Removing face for frame %s at index: %s", frame_index, face_index)
        self._faces_cache.remove(frame_index, face_index)
        image_id = self._canvas.find_withtag("image_{}".format(frame_index))[face_index]
        # Retrieve the position of the face in the current viewer prior to deletion
        display_index = self._canvas.active_filter.image_ids.index(image_id)
        self._canvas.delete(self._canvas.face_id_from_object(image_id))
        self._update_multi_tags(frame_index)
        self._canvas.active_filter.remove_face(frame_index, face_index, display_index)

    # << ADD AND REMOVE METHODS >> #
    def _update_multi_tags(self, frame_index):
        """ Update the tags indicating whether this frame contains multiple faces.

        Parameters
        ----------
        frame_index: int
            The frame index that the tags are to be updated for
        """
        image_tag = "image_{}".format(frame_index)
        mesh_tag = "mesh_{}".format(frame_index)
        frame_tag = "frame_id_{}".format(frame_index)
        num_faces = len(self._canvas.find_withtag(image_tag))
        logger.info("image_tag: %s, frame_tag: %s, faces_count: %s",
                    image_tag, frame_tag, num_faces)
        if num_faces == 0:
            return None
        self._canvas.dtag(frame_tag, "not_multi")
        self._canvas.dtag(frame_tag, "multi")
        self._canvas.dtag(frame_tag, "multi_image")
        self._canvas.dtag(frame_tag, "multi_mesh")
        if num_faces > 1:
            self._canvas.addtag_withtag("multi", frame_tag)
            self._canvas.addtag_withtag("multi_image", image_tag)
            self._canvas.addtag_withtag("multi_mesh", mesh_tag)
        else:
            self._canvas.addtag_withtag("not_multi", frame_tag)
        return frame_tag

    # << UPDATE METHODS >> #
    def update(self, frame_index, face_index):
        """ Update the currently selected face on editor update """
        # TODO Decide what to update based on current edit mode
        tk_face, mesh_landmarks = self._canvas.get_tk_face_and_landmarks(frame_index, face_index)
        self._faces_cache.update(frame_index, face_index, tk_face, mesh_landmarks)
        image_id = self._canvas.find_withtag("image_{}".format(frame_index))[face_index]
        self._canvas.itemconfig(image_id, image=tk_face)
        coords = self._canvas.coords(image_id)
        mesh_ids = self._canvas.mesh_ids_for_face_id(image_id)
        logger.trace("frame_index: %s, face_index: %s, image_id: %s, coords: %s, mesh_ids: %s, "
                     "tk_face: %s, mesh_landmarks: %s", frame_index, face_index, image_id, coords,
                     mesh_ids, tk_face, mesh_landmarks)
        for points, item_id in zip(mesh_landmarks["landmarks"], mesh_ids):
            self._canvas.coords(item_id, *(points + coords).flatten())
