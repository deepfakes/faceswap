#!/usr/bin/env python3
""" Bounding Box Editor for the manual adjustments tool """

import gettext
import platform
from functools import partial

import numpy as np

from lib.gui.custom_widgets import RightClickMenu
from ._base import ControlPanelOption, Editor, logger


# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext


class BoundingBox(Editor):
    """ The Bounding Box Editor.

    Adjusting the bounding box feeds the aligner to generate new 68 point landmarks.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The _detected_faces data for this manual session
    """
    def __init__(self, canvas, detected_faces):
        self._tk_aligner = None
        self._right_click_menu = RightClickMenu([_("Delete Face")],
                                                [self._delete_current_face],
                                                ["Del"])
        control_text = _("Bounding Box Editor\nEdit the bounding box being fed into the aligner "
                         "to recalculate the landmarks.\n\n"
                         " - Grab the corner anchors to resize the bounding box.\n"
                         " - Click and drag the bounding box to relocate.\n"
                         " - Click in empty space to create a new bounding box.\n"
                         " - Right click a bounding box to delete a face.")
        key_bindings = {"<Delete>": self._delete_current_face}
        super().__init__(canvas, detected_faces,
                         control_text=control_text, key_bindings=key_bindings)

    @property
    def _corner_order(self):
        """ dict: The position index of bounding box corners """
        return {0: ("top", "left"),
                1: ("top", "right"),
                2: ("bottom", "right"),
                3: ("bottom", "left")}

    @property
    def _bounding_boxes(self):
        """ list: The :func:`tkinter.Canvas.coords` for all displayed bounding boxes. """
        item_ids = self._canvas.find_withtag("bb_box")
        return [self._canvas.coords(item_id) for item_id in item_ids
                if self._canvas.itemcget(item_id, "state") != "hidden"]

    def _add_controls(self):
        """ Controls for feeding the Aligner. Exposes Normalization Method as a parameter. """
        align_ctl = ControlPanelOption(
            "Aligner",
            str,
            group="Aligner",
            choices=["cv2-dnn", "FAN"],
            default="FAN",
            is_radio=True,
            helptext=_("Aligner to use. FAN will obtain better alignments, but cv2-dnn can be "
                       "useful if FAN cannot get decent alignments and you want to set a base to "
                       "edit from."))
        self._tk_aligner = align_ctl.tk_var
        self._add_control(align_ctl)

        norm_ctl = ControlPanelOption(
            "Normalization method",
            str,
            group="Aligner",
            choices=["none", "clahe", "hist", "mean"],
            default="hist",
            is_radio=True,
            helptext=_("Normalization method to use for feeding faces to the aligner. This can "
                       "help the aligner better align faces with difficult lighting conditions. "
                       "Different methods will yield different results on different sets. NB: "
                       "This does not impact the output face, just the input to the aligner."
                       "\n\tnone: Don't perform normalization on the face."
                       "\n\tclahe: Perform Contrast Limited Adaptive Histogram Equalization on "
                       "the face."
                       "\n\thist: Equalize the histograms on the RGB channels."
                       "\n\tmean: Normalize the face colors to the mean."))
        var = norm_ctl.tk_var
        var.trace("w",
                  lambda *e, v=var: self._det_faces.extractor.set_normalization_method(v.get()))
        self._add_control(norm_ctl)

    def update_annotation(self):
        """ Get the latest bounding box data from alignments and update. """
        if self._globals.is_zoomed:
            logger.trace("Image is zoomed. Hiding Bounding Box.")
            self.hide_annotation()
            return
        key = "bb_box"
        color = self._control_color
        for idx, face in enumerate(self._face_iterator):
            box = np.array([(face.left, face.top), (face.right, face.bottom)])
            box = self._scale_to_display(box).astype("int32").flatten()
            kwargs = {"outline": color, "width": 1}
            logger.trace("frame_index: %s, face_index: %s, box: %s, kwargs: %s",
                         self._globals.frame_index, idx, box, kwargs)
            self._object_tracker(key, "rectangle", idx, box, kwargs)
            self._update_anchor_annotation(idx, box, color)
        logger.trace("Updated bounding box annotations")

    def _update_anchor_annotation(self, face_index, bounding_box, color):
        """ Update the anchor annotations for each corner of the bounding box.

        The anchors only display when the bounding box editor is active.

        Parameters
        ----------
        face_index: int
            The index of the face being annotated
        bounding_box: :class:`numpy.ndarray`
            The scaled bounding box to get the corner anchors for
        color: str
            The hex color of the bounding box line
        """
        if not self._is_active:
            self.hide_annotation("bb_anc_dsp")
            self.hide_annotation("bb_anc_grb")
            return
        fill_color = "gray"
        activefill_color = "white" if self._is_active else ""
        anchor_points = self._get_anchor_points(((bounding_box[0], bounding_box[1]),
                                                 (bounding_box[2], bounding_box[1]),
                                                 (bounding_box[2], bounding_box[3]),
                                                 (bounding_box[0], bounding_box[3])))
        for idx, (anc_dsp, anc_grb) in enumerate(zip(*anchor_points)):
            dsp_kwargs = {"outline": color, "fill": fill_color, "width": 1}
            grb_kwargs = {"outline": '', "fill": '', "width": 1, "activefill": activefill_color}
            dsp_key = f"bb_anc_dsp_{idx}"
            grb_key = f"bb_anc_grb_{idx}"
            self._object_tracker(dsp_key, "oval", face_index, anc_dsp, dsp_kwargs)
            self._object_tracker(grb_key, "oval", face_index, anc_grb, grb_kwargs)
        logger.trace("Updated bounding box anchor annotations")

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Set the cursor action.

        Update :attr:`_mouse_location` with the current cursor position and display appropriate
        icon.

        If the cursor is over a corner anchor, then pop resize icon.
        If the cursor is over a bounding box, then pop move icon.
        If the cursor is over the image, then pop add icon.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
        if self._check_cursor_anchors():
            return
        if self._check_cursor_bounding_box(event):
            return
        if self._check_cursor_image(event):
            return

        self._canvas.config(cursor="")
        self._mouse_location = None

    def _check_cursor_anchors(self):
        """ Check whether the cursor is over a corner anchor.

        If it is, set the appropriate cursor type and set :attr:`_mouse_location` to
        ("anchor", (`face index`, `anchor index`)

        Returns
        -------
        bool
            ``True`` if cursor is over an anchor point otherwise ``False``
        """
        anchors = set(self._canvas.find_withtag("bb_anc_grb"))
        item_ids = set(self._canvas.find_withtag("current")).intersection(anchors)
        if not item_ids:
            return False
        item_id = list(item_ids)[0]
        tags = self._canvas.gettags(item_id)
        face_idx = int(next(tag for tag in tags if tag.startswith("face_")).split("_")[-1])
        corner_idx = int(next(tag for tag in tags
                              if tag.startswith("bb_anc_grb_")
                              and "face_" not in tag).split("_")[-1])
        pos_x, pos_y = self._corner_order[corner_idx]
        self._canvas.config(cursor=f"{pos_x}_{pos_y}_corner")
        self._mouse_location = ("anchor", f"{face_idx}_{corner_idx}")
        return True

    def _check_cursor_bounding_box(self, event):
        """ Check whether the cursor is over a bounding box.

        If it is, set the appropriate cursor type and set :attr:`_mouse_location` to:
        ("box", `face index`)

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event

        Returns
        -------
        bool
            ``True`` if cursor is over a bounding box otherwise ``False``

        Notes
        -----
        We can't use tags on unfilled rectangles as the interior of the rectangle is not tagged.
        """
        for face_idx, bbox in enumerate(self._bounding_boxes):
            if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
                self._canvas.config(cursor="fleur")
                self._mouse_location = ("box", str(face_idx))
                return True
        return False

    def _check_cursor_image(self, event):
        """ Check whether the cursor is over the image.

        If it is, set the appropriate cursor type and set :attr:`_mouse_location` to:
        ("image", )

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event

        Returns
        -------
        bool
            ``True`` if cursor is over a bounding box otherwise ``False``
        """
        if self._globals.frame_index == -1:
            return False
        display_dims = self._globals.current_frame.display_dims
        if (self._canvas.offset[0] <= event.x <= display_dims[0] + self._canvas.offset[0] and
                self._canvas.offset[1] <= event.y <= display_dims[1] + self._canvas.offset[1]):
            self._canvas.config(cursor="plus")
            self._mouse_location = ("image", )
            return True
        return False

    # Mouse Actions
    def set_mouse_click_actions(self):
        """ Add context menu to OS specific right click action. """
        super().set_mouse_click_actions()
        self._canvas.bind("<Button-2>" if platform.system() == "Darwin" else "<Button-3>",
                          self._context_menu)

    def _drag_start(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse.

        If :attr:`_mouse_location` indicates a corner anchor, then the bounding box is resized
        based on the adjusted corner, and the alignments re-generated.

        If :attr:`_mouse_location` indicates a bounding box, then the bounding box is moved, and
        the alignments re-generated.

        If :attr:`_mouse_location` indicates being over the main image, then a new bounding box is
        created, and alignments generated.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        if self._mouse_location is None:
            self._drag_data = {}
            self._drag_callback = None
            return
        if self._mouse_location[0] == "anchor":
            corner_idx = int(self._mouse_location[1].split("_")[-1])
            self._drag_data["corner"] = self._corner_order[corner_idx]
            self._drag_callback = self._resize
        elif self._mouse_location[0] == "box":
            self._drag_data["current_location"] = (event.x, event.y)
            self._drag_callback = self._move
        elif self._mouse_location[0] == "image":
            self._create_new_bounding_box(event)
            # Refresh cursor and _mouse_location for new bounding box and reset _drag_start
            self._update_cursor(event)
            self._drag_start(event)

    def _drag_stop(self, event):  # pylint:disable=unused-argument
        """ Trigger a viewport thumbnail update on click + drag release

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Required but unused.
        """
        if self._mouse_location is None:
            return
        face_idx = int(self._mouse_location[1].split("_")[0])
        self._det_faces.update.post_edit_trigger(self._globals.frame_index, face_idx)

    def _create_new_bounding_box(self, event):
        """ Create a new bounding box when user clicks on image, outside of existing boxes.

        The bounding box is created as a square located around the click location, with dimensions
        1 quarter the size of the frame's shortest side

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event
        """
        size = min(self._globals.current_frame.display_dims) // 8
        box = (event.x - size, event.y - size, event.x + size, event.y + size)
        logger.debug("Creating new bounding box: %s ", box)
        self._det_faces.update.add(self._globals.frame_index, *self._coords_to_bounding_box(box))

    def _resize(self, event):
        """ Resizes a bounding box on a corner anchor drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        face_idx = int(self._mouse_location[1].split("_")[0])
        face_tag = f"bb_box_face_{face_idx}"
        box = self._canvas.coords(face_tag)
        logger.trace("Face Index: %s, Corner Index: %s. Original ROI: %s",
                     face_idx, self._drag_data["corner"], box)
        # Switch top/bottom and left/right and set partial so indices match and we don't
        # need branching logic for min/max.
        limits = (partial(min, box[2] - 20),
                  partial(min, box[3] - 20),
                  partial(max, box[0] + 20),
                  partial(max, box[1] + 20))
        rect_xy_indices = [("left", "top", "right", "bottom").index(pnt)
                           for pnt in self._drag_data["corner"]]
        box[rect_xy_indices[1]] = limits[rect_xy_indices[1]](event.x)
        box[rect_xy_indices[0]] = limits[rect_xy_indices[0]](event.y)
        logger.trace("New ROI: %s", box)
        self._det_faces.update.bounding_box(self._globals.frame_index,
                                            face_idx,
                                            *self._coords_to_bounding_box(box),
                                            aligner=self._tk_aligner.get())

    def _move(self, event):
        """ Moves the bounding box on a bounding box drag event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        logger.trace("event: %s, mouse_location: %s", event, self._mouse_location)
        face_idx = int(self._mouse_location[1])
        shift = (event.x - self._drag_data["current_location"][0],
                 event.y - self._drag_data["current_location"][1])
        face_tag = f"bb_box_face_{face_idx}"
        coords = np.array(self._canvas.coords(face_tag)) + (*shift, *shift)
        logger.trace("face_tag: %s, shift: %s, new co-ords: %s", face_tag, shift, coords)
        self._det_faces.update.bounding_box(self._globals.frame_index,
                                            face_idx,
                                            *self._coords_to_bounding_box(coords),
                                            aligner=self._tk_aligner.get())
        self._drag_data["current_location"] = (event.x, event.y)

    def _coords_to_bounding_box(self, coords):
        """ Converts tkinter coordinates to :class:`lib.align.DetectedFace` bounding
        box format, scaled up and offset for feeding the model.

        Returns
        -------
        tuple
            The (`x`, `width`, `y`, `height`) integer points of the bounding box.
        """
        logger.trace("in: %s", coords)
        coords = self.scale_from_display(
            np.array(coords).reshape((2, 2))).flatten().astype("int32")
        logger.trace("out: %s", coords)
        return (coords[0], coords[2] - coords[0], coords[1], coords[3] - coords[1])

    def _context_menu(self, event):
        """ Create a right click context menu to delete the alignment that is being
        hovered over. """
        if self._mouse_location is None or self._mouse_location[0] != "box":
            return
        self._right_click_menu.popup(event)

    def _delete_current_face(self, *args):  # pylint:disable=unused-argument
        """ Called by the right click delete event. Deletes the face that the mouse is currently
        over.

        Parameters
        ----------
        args: tuple (unused)
            The event parameter is passed in by the hot key binding, so args is required
        """
        if self._mouse_location is None or self._mouse_location[0] != "box":
            logger.debug("Delete called without valid location. _mouse_location: %s",
                         self._mouse_location)
            return
        logger.debug("Deleting face. _mouse_location: %s", self._mouse_location)
        self._det_faces.update.delete(self._globals.frame_index, int(self._mouse_location[1]))
