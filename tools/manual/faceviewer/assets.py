#!/usr/bin/env python3
""" Handles loading, updating and creation of objects for the Face viewer of
the manual adjustments tool """
import logging
import tkinter as tk

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FacesViewerLoader():  # pylint:disable=too-few-public-methods
    """ Loads the faces into the :class:`tools.manual.FacesViewer` as they become available
    in the :class:`tools.lib_manual.media.FacesCache`.

    Faces are loaded into the Face Cache in a background thread. This class checks for the
    availability of loaded faces (every 0.5 seconds until the faces are loaded) and updates
    the display with the latest loaded faces.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    alignments: :class:`~tool.manual.media.AlignmentsData`
        The alignments data for the currently loaded frames
    """
    def __init__(self, canvas, alignments):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._alignments = alignments
        self._faces_cache = canvas._faces_cache
        self._frame_count = canvas._frames.frame_count
        self._progress_bar = canvas._progress_bar
        self._progress_bar.start(mode="determinate")
        face_count = self._alignments.face_count_per_index
        frame_faces_loaded = [False for _ in range(self._frame_count)]
        self._load_faces(0, face_count, frame_faces_loaded)
        self._faces_cache.loader.launch()
        logger.debug("Initialized: %s ", self.__class__.__name__,)

    def _load_faces(self, faces_seen, faces_count, frame_faces_loaded):
        """ Load the currently available faces from :class:`tools.lib_manual.media.FacesCache`
        into the Faces Viewer.

        Obtains the indices of all faces that have currently been loaded, and displays them in the
        Faces Viewer. This process re-runs every 0.5 seconds until all faces have been loaded.

        Parameters
        ----------
        faces_seen: int
            The number of faces that have already been loaded. For tracking progress
        faces_count: list
            The number of faces that appear in each frame. List is of length :attr:`_frame_count`
            with each value being the number of faces that appear for the given index.
        frame_faces_loaded: list
            List of length :attr:`_frame_count` containing `bool` values indicating whether
            annotations have been created for each frame or not.
        """
        self._update_progress(faces_seen)
        update_indices = self._faces_cache.loader.loaded_frame_indices[faces_seen:]
        logger.debug("faces_seen: %s, update count: %s", faces_seen, len(update_indices))
        tk_faces = self._faces_cache.tk_faces[update_indices]
        for frame_idx, faces in zip(update_indices, tk_faces):
            starting_idx = sum(faces_count[:frame_idx])
            for idx, tk_face in enumerate(faces):
                coords = self._canvas.coords_from_index(starting_idx + idx)
                self._canvas.new_objects.create(coords, tk_face, frame_idx,
                                                is_multi=len(faces) > 1)
                self._reshape_canvas(coords)
            if faces:
                self._place_in_stack(frame_idx, frame_faces_loaded)

        faces_seen += len(update_indices)
        if faces_seen == self._frame_count:
            logger.debug("Load complete")
            self._on_load_complete()
        else:
            logger.debug("Refreshing... (faces_seen: %s, frame_count: %s",
                         faces_seen, self._frame_count)
            self._canvas.after(500, self._load_faces, faces_seen, faces_count, frame_faces_loaded)

    def _reshape_canvas(self, coordinates):
        """ Scroll the canvas to the first row, when the first row has been received from the
        Faces Cache (sometimes the first row doesn't load first).

        Update the canvas scroll region to include any newly added objects

        Parameters
        ----------
        coordinates: tuple
            The (x, y) coordinates of the newly created object
        """
        if coordinates[1] == 0:  # Set the top of canvas when first row seen
            logger.debug("Scrolling to top row")
            self._canvas.yview_moveto(0.0)
        if coordinates[0] == 0:  # Resize canvas on new line
            logger.trace("Extending canvas")
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _place_in_stack(self, frame_index, frame_faces_loaded):
        """ Place any out of order objects into their correct position in the stack.
        As the images are loaded in parallel, the faces are not created in order on the stack.
        For the viewer to work correctly, out of order items are placed back in the correct
        position.

        Parameters
        ----------
        frame_index: int
            The index that the currently loading objects belong to
        frame_faces_loaded: list
            List of length :attr:`_frame_count` containing `bool` values indicating whether
            annotations have been created for each frame or not.
        """
        frame_faces_loaded[frame_index] = True
        offset = frame_index + 1
        higher_frames = frame_faces_loaded[offset:]
        if not any(higher_frames):
            return
        below_frame = next(idx for idx, loaded in enumerate(higher_frames) if loaded) + offset
        logger.trace("Placing frame %s in stack below frame %s", frame_index, below_frame)
        self._canvas.tag_lower("frame_id_{}".format(frame_index),
                               "frame_id_{}".format(below_frame))

    def _update_progress(self, faces_seen):
        """ Update the progress bar prior to loading the latest faces.

        Parameters
        ----------
        faces_seen: int
            The number of faces that have already been loaded
        """
        position = faces_seen + 1
        progress = int(round((position / self._frame_count) * 100))
        msg = "Loading Faces: {}/{} - {}%".format(position, self._frame_count, progress)
        logger.debug("Progress update: %s", msg)
        self._progress_bar.progress_update(msg, progress)

    def _on_load_complete(self):
        """ Final actions to perform once the faces have finished loading into the Faces Viewer.

        Updates any faces where edits have been made whilst the faces were loading.
        Updates any color settings that were changed during load.
        Sets the display to the currently selected filter.
        Sets the load complete variable to ``True``
        Highlights the active face.
        Enables saving of an alignments file
        """
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
        self._canvas.update_mesh_color()
        self._canvas.switch_filter()
        self._progress_bar.stop()
        self._faces_cache.loader.set_load_complete()
        self._alignments.enable_save()
        self._canvas.active_frame.reload_annotations()

    def _on_load_remove_faces(self, existing_count, new_count, frame_index):
        """ Remove any faces from the viewer for the given frame index of any faces
        that have been deleted whilst face viewer was loading.

        Parameters
        ----------
        existing_count: int
            The number of faces that currently appear in the Faces Viewer for the given frame
        new_count: int
            The number of faces that should appear in the Faces Viewer for the given frame
        frame_index: int
            The frame index to remove faces for
        """
        logger.debug("existing_count: %s. new_count: %s, frame_index: %s",
                     existing_count, new_count, frame_index)
        if existing_count <= new_count:
            return
        for face_idx in range(new_count, existing_count):
            logger.debug("Deleting face at index %s for frame %s", face_idx, frame_index)
            self._canvas.update_face.remove(frame_index, face_idx)


class ObjectCreator():
    """ Creates the objects and annotations that are to be displayed in the
    :class:`tools.manual.FacesViewer.`

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._object_types = ("image", "mesh")
        self._current_face_id = 0
        logger.debug("Initialized: %s", self.__class__.__name__)

    def create(self, coordinates, tk_face, frame_index, is_multi=False):
        """ Create all of the annotations for a single Face Viewer face.

        Parameters
        ----------
        coordinates: tuple
            The top left (x, y) coordinates for the annotations' position in the Faces Viewer
        tk_face: :class:`~manual.facesviewer.cache.TKFace`
            The tk face object containing the face to be used for the image annotation and the
            mesh landmarks
        mesh_color: str
            The hex code holding the color that the mesh should be displayed as
        frame_index: int
            The frame index that this object appears in
        is_multi: bool
            ``True`` if there are multiple faces in the given frame, otherwise ``False``.
            Default: ``False``. Used for creating multi-face tags

        Returns
        -------
        image_id: int
            The item id of the newly created face
        mesh_ids: list
            List of item ids for the newly created mesh
        """
        logger.trace("coordinates: %s, tk_face: %s, frame_index: %s, "
                     "is_multi: %s", coordinates, tk_face, frame_index, is_multi)
        tags = {obj: self._get_viewer_tags(obj, frame_index, is_multi)
                for obj in self._object_types}
        image_id = self._canvas.create_image(*coordinates,
                                             image=tk_face.face,
                                             anchor=tk.NW,
                                             tags=tags["image"])
        mesh_ids = self.create_mesh_annotations(self._canvas.get_muted_color("Mesh"),
                                                tk_face,
                                                coordinates,
                                                tags["mesh"])
        self._current_face_id += 1
        logger.trace("image_id: %s, mesh_ids: %s", image_id, mesh_ids)
        return image_id, mesh_ids

    def _get_viewer_tags(self, object_type, frame_index, is_multi):
        """ Generates tags for the given object based on the frame index, the object type,
        the current face identifier and whether multiple faces appear in the given frame.

        Parameters
        ----------
        object_type: ["image" or "mesh"]
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
        logger.trace("object_type: %s, frame_index: %s, is_multi: %s",
                     object_type, frame_index, is_multi)
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
        logger.trace("tags: %s", tags)
        return tags

    def create_mesh_annotations(self, color, tk_face, offset, tag):
        """ Creates the mesh annotation for the landmarks. This is made up of a series
        of polygons or lines, depending on which part of the face is being annotated.

        Parameters
        ----------
        color: str
            The hex code for the color that the mesh should be displayed as
        tk_face: :class:`~manual.facesviewer.cache.TKFace`
            The tk face object containing the face to be used for the image annotation and the
            mesh landmarks
        offset: :class:`numpy.ndarray`
            The top left co-ordinates of the face that corresponds to the given landmarks.
            The mesh annotations will be offset by this amount, to place them in the correct
            place on the canvas
        tag: list
            The list of tags, as generated in :func:`_get_viewer_tags` that are to applied to these
            mesh annotations

        Returns
        -------
        list
            The canvas object ids for the created mesh annotations
        """
        retval = []
        state = "normal" if self._canvas.optional_annotations["mesh"] else "hidden"
        kwargs = dict(polygon=dict(fill="", outline=color), line=dict(fill=color))
        logger.trace("color: %s, offset: %s, tag: %s, state: %s, kwargs: %s",
                     color, offset, tag, state, kwargs)
        for is_poly, landmarks in zip(tk_face.mesh_is_poly, tk_face.mesh_points):
            key = "polygon" if is_poly else "line"
            tags = tag + ["mesh_{}".format(key)]
            obj = getattr(self._canvas, "create_{}".format(key))
            obj_kwargs = kwargs[key]
            coords = (landmarks + offset).flatten()
            retval.append(obj(*coords, state=state, width=1, tags=tags, **obj_kwargs))
        return retval


class UpdateFace():
    """ Handles all adding, removing and updating of faces in the
        :class:`~tools.manual.FacesViewer` canvas when a user performs an edit.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    alignments: :class:`~tool.manual.media.AlignmentsData`
        The alignments data for the currently loaded frames
    """

    def __init__(self, canvas, alignments):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._alignments = alignments
        self._faces_cache = canvas._faces_cache
        self._frames = canvas._frames
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _get_aligned_face(self, frame_index, face_index):
        """ Obtain the resized photo image face and scaled landmarks for the requested face.

        Parameters
        ----------
        frame_index: int
            The frame to obtain the face and landmarks for
        face_index: int
            The index of the face within the requested frame

        Returns
        -------
        :class:`tkinter.PhotoImage`
            The masked or unmasked face formatted for canvas placement
        dict
            A dictionary containing the keys `landmarks` holding a `list` of :class:`numpy.ndarray`
            objects and `is_poly` containing a `list` of `bool` types corresponding to the
            `landmarks` indicating whether a line or polygon should be created for each mesh
            annotation
         """
        logger.trace("frame_index: %s, face_index: %s", frame_index, face_index)
        face, landmarks, mask = self._alignments.get_aligned_face_at_index(
            face_index,
            frame_index=frame_index,
            size=self._faces_cache.size,
            with_landmarks=True,
            with_mask=True)

        mask = mask.get(self._canvas.selected_mask,
                        None) if self._canvas.optional_annotations["mask"] else None
        mask = mask if mask is None else mask.mask.squeeze()
        return face, landmarks, mask

    # << ADD FACE METHODS >> #
    def add(self, frame_index):
        """ Add a face to the :class:`~tools.manual.FacesViewer` canvas for the given frame.

        Generates the face image and mesh annotations for a newly added face, creates the relevant
        tags into the correct location in the object stack.

        Parameters
        ----------
        frame_index: int
            The frame index to add the face for
        """
        face_idx = len(self._canvas.find_withtag("image_{}".format(frame_index)))
        logger.debug("Adding face to frame: (frame_index: %s, face_index: %s)",
                     frame_index, face_idx)
        # Add objects to cache
        face, landmarks, mask = self._get_aligned_face(frame_index, face_idx)
        tk_face = self._faces_cache.add(frame_index, face, landmarks, mask)

        # Create new annotations
        image_id, mesh_ids = self._canvas.new_objects.create((0, 0), tk_face, frame_index)
        # Place in stack
        frame_tag = self._update_multi_tags(frame_index)
        self._place_in_stack(frame_index, frame_tag)
        # Update viewer
        self._canvas.active_filter.add_face(image_id, mesh_ids, tk_face.mesh_points["landmarks"])

    def _place_in_stack(self, frame_index, frame_tag):
        """ Place newly added faces in the correct location in the object stack.

        Parameters
        ----------
        frame_index: int
            The frame index that the face(s) need to be placed for
        frame_tag: str
            The tag of the canvas objects that need to be placed
        """
        next_frame_idx = self._get_next_frame_index(frame_index)
        if next_frame_idx is not None:
            next_tag = "frame_id_{}".format(next_frame_idx)
            logger.debug("Lowering annotations for frame %s below frame %s", frame_tag, next_tag)
            self._canvas.tag_lower(frame_tag, next_tag)

    def _get_next_frame_index(self, frame_index):
        """ Get the index of the next frame that contains faces.

        Used for calculating the correct location to place the newly created objects in the stack.

        Parameters
        ----------
        frame_index: int
            The frame index for the objects that require placing in the stack

        Returns
        -------
        int or ``None``
            The frame index for the next frame that contains faces. ``None`` is returned if the
            given frame index is already at the end of the stack
        """
        offset = frame_index + 1
        next_frame_idx = next((
            idx for idx, f_count in enumerate(self._alignments.face_count_per_index[offset:])
            if f_count > 0), None)
        if next_frame_idx is None:
            return None
        next_frame_idx += offset
        logger.debug("Returning next frame with faces: %s for frame index: %s",
                     next_frame_idx, frame_index)
        return next_frame_idx

    # << REMOVE FACE METHODS >> #
    def remove_face_from_viewer(self, item_id):
        """ Remove a face and it's annotations from the :class:`~tools.manual.FacesViewer` canvas
        for the given item identifier. Also removes the alignments data from the alignments file,
        and the cached face data from :class:`~tools.lib_manual.media.FacesCache`.

        This action is specifically called when a face is deleted from the viewer through a right
        click menu action.

        parameters
        ----------
        item_id: int
            The face group item identifier stored in the face's object tags
        """
        frame_idx = self._canvas.frame_index_from_object(item_id)
        face_idx = self._canvas.find_withtag("image_{}".format(frame_idx)).index(item_id)
        logger.debug("item_id: %s, frame_index: %s, face_index: %s", item_id, frame_idx, face_idx)
        self._alignments.delete_face_at_index_by_frame(frame_idx, face_idx)
        self.remove(frame_idx, face_idx)
        if frame_idx == self._frames.tk_position.get():
            self._frames.tk_update.set(True)
            self._canvas.active_frame.reload_annotations()

    def remove(self, frame_index, face_index):
        """ Remove a face and it's annotations from the :class:`~tools.manual.FacesViewer` canvas
        for the given face index at the given frame index. Also removes the cached face data from
        :class:`~tools.lib_manual.media.FacesCache`.

        Parameters
        ----------
        frame_index: int
            The frame index that contains the face and objects that are to be removed
        face_index: int
            The index of the face within the given frame that is to have its objects removed
        """
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
        """ Update the tags indicating whether this frame contains multiple faces or not.

        Parameters
        ----------
        frame_index: int
            The frame index that the tags are to be updated for
        """
        image_tag = "image_{}".format(frame_index)
        mesh_tag = "mesh_{}".format(frame_index)
        frame_tag = "frame_id_{}".format(frame_index)
        num_faces = len(self._canvas.find_withtag(image_tag))
        logger.debug("image_tag: %s, frame_tag: %s, faces_count: %s",
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
        """  Update the face and annotations for the given face index at the given frame index.

        This method is called when an editor update is made. It updates the displayed annotations
        as well as the meta information stored in :class:`~tools.lib_manual.media.FacesCache`.

        Parameters
        ----------
        frame_index: int
            The frame index that contains the face and objects that are to be updated
        face_index: int
            The index of the face within the given frame that is to have its objects updated
        """
        tk_face = self._faces_cache.tk_faces[frame_index][face_index]
        face, landmarks, mask = self._get_aligned_face(frame_index, face_index)
        tk_face.update_face(face, landmarks, mask=mask)

        image_id = self._canvas.find_withtag("image_{}".format(frame_index))[face_index]
        self._canvas.itemconfig(image_id, image=tk_face.face)
        coords = self._canvas.coords(image_id)
        mesh_ids = self._mesh_ids_for_face_id(image_id)
        logger.trace("frame_index: %s, face_index: %s, image_id: %s, coords: %s, mesh_ids: %s, "
                     "tk_face: %s", frame_index, face_index, image_id, coords, mesh_ids, tk_face)
        for points, item_id in zip(tk_face.mesh_points, mesh_ids):
            self._canvas.coords(item_id, *(points + coords).flatten())

    def _mesh_ids_for_face_id(self, item_id):
        """ Obtain all the item ids for a given face index's mesh annotation.

        Parameters
        ----------
        face_index: int
            The face index to retrieve the mesh ids for

        Returns
        -------
        list
            The list of item ids for the mesh annotation pertaining to the given face index
        """
        face_id = next((tag for tag in self._canvas.gettags(item_id)
                        if tag.startswith("face_id_")), None)
        if face_id is None:
            return None
        retval = self._canvas.find_withtag("mesh_{}".format(face_id))
        logger.trace("item_id: %s, face_id: %s, mesh ids: %s", item_id, face_id, retval)
        return retval
