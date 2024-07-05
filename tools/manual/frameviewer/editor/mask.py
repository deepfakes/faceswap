#!/usr/bin/env python3
""" Mask Editor for the manual adjustments tool """
import gettext
import tkinter as tk

import numpy as np
import cv2
from PIL import Image, ImageTk

from ._base import ControlPanelOption, Editor, logger

# LOCALES
_LANG = gettext.translation("tools.manual", localedir="locales", fallback=True)
_ = _LANG.gettext


class Mask(Editor):
    """ The mask Editor.

    Edit a mask in the alignments file.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The canvas that holds the image and annotations
    detected_faces: :class:`~tools.manual.detected_faces.DetectedFaces`
        The _detected_faces data for this manual session
    """
    def __init__(self, canvas, detected_faces):
        self._meta = []
        self._tk_faces = []
        self._internal_size = 512
        control_text = _("Mask Editor\nEdit the mask."
                         "\n - NB: For Landmark based masks (e.g. components/extended) it is "
                         "better to make sure the landmarks are correct rather than editing the "
                         "mask directly. Any change to the landmarks after editing the mask will "
                         "override your manual edits.")
        key_bindings = {"[": lambda *e, i=False: self._adjust_brush_radius(increase=i),
                        "]": lambda *e, i=True: self._adjust_brush_radius(increase=i)}
        super().__init__(canvas, detected_faces,
                         control_text=control_text, key_bindings=key_bindings)
        # Bind control click for reverse painting
        self._canvas.bind("<Control-ButtonPress-1>", self._control_click)
        self._mask_type = self._set_tk_mask_change_callback()
        self._cursor_shape = self._set_tk_cursor_shape_change_callback()
        self._mouse_location = [
            self._get_cursor_shape(), False]

    @property
    def _opacity(self):
        """ float: The mask opacity setting from the control panel from 0.0 - 1.0. """
        annotation = self.__class__.__name__
        return self._annotation_formats[annotation]["mask_opacity"].get() / 100.0

    @property
    def _brush_radius(self):
        """ int: The radius of the brush to use as set in control panel options """
        return self._control_vars["brush"]["BrushSize"].get()

    @property
    def _edit_mode(self):
        """ str: The currently selected edit mode based on optional action button.
        One of "draw" or "erase" """
        action = [name for name, option in self._actions.items()
                  if option["group"] == "paint" and option["tk_var"].get()]
        return "draw" if not action else action[0]

    @property
    def _cursor_color(self):
        """ str: The hex code for the selected cursor color """
        return self._control_vars["brush"]["CursorColor"].get()

    @property
    def _cursor_shape_name(self):
        """ str: The selected cursor shape """
        return self._control_vars["display"]["CursorShape"].get()

    def _add_actions(self):
        """ Add the optional action buttons to the viewer. Current actions are Draw, Erase
        and Zoom. """
        self._add_action("magnify", "zoom", _("Magnify/Demagnify the View"),
                         group=None, hotkey="M")
        self._add_action("draw", "draw", _("Draw Tool"), group="paint", hotkey="D")
        self._add_action("erase", "erase", _("Erase Tool"), group="paint", hotkey="E")
        self._actions["magnify"]["tk_var"].trace(
            "w",
            lambda *e: self._globals.var_full_update.set(True))

    def _add_controls(self):
        """ Add the mask specific control panel controls.

        Current controls are:
          - the mask type to edit
          - the size of brush to use
          - the cursor display color
        """
        masks = sorted(msk.title() for msk in list(self._det_faces.available_masks) + ["None"])
        default = masks[0] if len(masks) == 1 else [mask for mask in masks if mask != "None"][0]
        self._add_control(ControlPanelOption("Mask type",
                                             str,
                                             group="Display",
                                             choices=masks,
                                             default=default,
                                             is_radio=True,
                                             helptext=_("Select which mask to edit")))
        self._add_control(ControlPanelOption("Brush Size",
                                             int,
                                             group="Brush",
                                             min_max=(1, 100),
                                             default=10,
                                             rounding=1,
                                             helptext=_("Set the brush size. ([ - decrease, "
                                                        "] - increase)")))
        self._add_control(ControlPanelOption("Cursor Color",
                                             str,
                                             group="Brush",
                                             choices="colorchooser",
                                             default="#ffffff",
                                             helptext=_("Select the brush cursor color.")))
        self._add_control(ControlPanelOption("Cursor Shape",
                                             str,
                                             group="Display",
                                             choices=["Circle", "Rectangle"],
                                             default="Circle",
                                             is_radio=True,
                                             helptext=_("Select a shape for masking cursor.")))

    def _set_tk_mask_change_callback(self):
        """ Add a trace to change the displayed mask on a mask type change. """
        var = self._control_vars["display"]["MaskType"]
        var.trace("w", lambda *e: self._on_mask_type_change())
        return var.get()

    def _set_tk_cursor_shape_change_callback(self):
        """ Add a trace to change the displayed cursor on a cursor shape type change. """
        var = self._control_vars["display"]["CursorShape"]
        var.trace("w", lambda *e: self._on_cursor_shape_change())
        return var.get()

    def _on_cursor_shape_change(self):
        self._mouse_location[0] = self._get_cursor_shape()

    def _on_mask_type_change(self):
        """ Update the displayed mask on a mask type change """
        mask_type = self._control_vars["display"]["MaskType"].get()
        if mask_type == self._mask_type:
            return
        self._meta = {"position": self._globals.frame_index}
        self._mask_type = mask_type
        self._globals.var_full_update.set(True)

    def hide_annotation(self, tag=None):
        """ Clear the mask :attr:`_meta` dict when hiding the annotation. """
        super().hide_annotation()
        self._meta = {}

    def update_annotation(self):
        """ Update the mask annotation with the latest mask. """
        position = self._globals.frame_index
        if position != self._meta.get("position", -1):
            # Reset meta information when moving to a new frame
            self._meta = {"position": position}
        key = self.__class__.__name__
        mask_type = self._control_vars["display"]["MaskType"].get().lower()
        color = self._control_color[1:]
        rgb_color = np.array(tuple(int(color[i:i + 2], 16) for i in (0, 2, 4)))
        roi_color = self._annotation_formats["ExtractBox"]["color"].get()
        opacity = self._opacity
        for idx, face in enumerate(self._face_iterator):
            face_idx = self._globals.face_index if self._globals.is_zoomed else idx
            mask = face.mask.get(mask_type, None)
            if mask is None:
                continue
            self._set_face_meta_data(mask, face_idx)
            self._update_mask_image(key.lower(), face_idx, rgb_color, opacity)
            self._update_roi_box(mask, face_idx, roi_color)

        self._canvas.tag_raise(self._mouse_location[0])  # Always keep brush cursor on top
        logger.trace("Updated mask annotation")

    def _set_face_meta_data(self, mask, face_index):
        """ Set the metadata for the current face if it has changed or is new.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The one channel mask cropped to the ROI
        face_index: int
            The index pertaining to the current face
        """
        masks = self._meta.get("mask", None)
        if masks is not None and len(masks) - 1 == face_index:
            logger.trace("Meta information already defined for face: %s", face_index)
            return

        logger.debug("Defining meta information for face: %s", face_index)
        scale = self._internal_size / mask.stored_size
        self._set_full_frame_meta(mask, scale)
        dims = (self._internal_size, self._internal_size)
        self._meta.setdefault("mask", []).append(cv2.resize(mask.stored_mask,
                                                            dims,
                                                            interpolation=cv2.INTER_CUBIC))
        if self.zoomed_centering != mask.stored_centering:
            self.zoomed_centering = mask.stored_centering

    def _set_full_frame_meta(self, mask, mask_scale):
        """ Sets the meta information for displaying the mask in full frame mode.

        Parameters
        ----------
        mask: :class:`lib.align.Mask`
            The mask object
        mask_scale: float
            The scaling factor from the stored mask size to the internal mask size

        Sets the following parameters to :attr:`_meta`:
            - roi_mask: the rectangular ROI box from the full frame that contains the original ROI
            for the full frame mask
            - top_left: The location that the roi_mask should be placed in the display frame
            - affine_matrix: The matrix for transposing the mask to a full frame
            - interpolator: The cv2 interpolation method to use for transposing mask to a
            full frame
            - slices: The (`x`, `y`) slice objects required to extract the mask ROI
            from the full frame
        """
        frame_dims = self._globals.current_frame.display_dims
        scaled_mask_roi = np.rint(mask.original_roi *
                                  self._globals.current_frame.scale).astype("int32")

        # Scale and clip the ROI to fit within display frame boundaries
        clipped_roi = scaled_mask_roi.clip(min=(0, 0), max=frame_dims)

        # Obtain min and max points to get ROI as a rectangle
        min_max = {"min": clipped_roi.min(axis=0), "max": clipped_roi.max(axis=0)}

        # Create a bounding box rectangle ROI
        roi_dims = np.rint((min_max["max"][1] - min_max["min"][1],
                            min_max["max"][0] - min_max["min"][0])).astype("uint16")
        roi = {"mask": np.zeros(roi_dims, dtype="uint8")[..., None],
               "corners": np.expand_dims(scaled_mask_roi - min_max["min"], axis=0)}
        # Block out areas outside of the actual mask ROI polygon
        cv2.fillPoly(roi["mask"], roi["corners"], 255)
        logger.trace("Setting Full Frame mask ROI. shape: %s", roi["mask"].shape)

        # obtain the slices for cropping mask from full frame
        xy_slices = (slice(int(round(min_max["min"][1])), int(round(min_max["max"][1]))),
                     slice(int(round(min_max["min"][0])), int(round(min_max["max"][0]))))

        # Adjust affine matrix for internal mask size and display dimensions
        adjustments = (np.array([[mask_scale, 0., 0.], [0., mask_scale, 0.]]),
                       np.array([[1 / self._globals.current_frame.scale, 0., 0.],
                                 [0., 1 / self._globals.current_frame.scale, 0.],
                                 [0., 0., 1.]]))
        in_matrix = np.dot(adjustments[0],
                           np.concatenate((mask.affine_matrix, np.array([[0., 0., 1.]]))))
        affine_matrix = np.dot(in_matrix, adjustments[1])

        # Get the size of the mask roi box in the frame
        side_sizes = (scaled_mask_roi[1][0] - scaled_mask_roi[0][0],
                      scaled_mask_roi[1][1] - scaled_mask_roi[0][1])
        mask_roi_size = (side_sizes[0] ** 2 + side_sizes[1] ** 2) ** 0.5

        self._meta.setdefault("roi_mask", []).append(roi["mask"])
        self._meta.setdefault("affine_matrix", []).append(affine_matrix)
        self._meta.setdefault("interpolator", []).append(mask.interpolator)
        self._meta.setdefault("slices", []).append(xy_slices)
        self._meta.setdefault("top_left", []).append(min_max["min"] + self._canvas.offset)
        self._meta.setdefault("mask_roi_size", []).append(mask_roi_size)

    def _update_mask_image(self, key, face_index, rgb_color, opacity):
        """ Obtain a mask, overlay over image and add to canvas or update.

        Parameters
        ----------
        key: str
            The base annotation name for creating tags
        face_index: int
            The index of the face within the current frame
        rgb_color: tuple
            The color that the mask should be displayed as
        opacity: float
            The opacity to apply to the mask
        """
        mask = (self._meta["mask"][face_index] * opacity).astype("uint8")
        if self._globals.is_zoomed:
            display_image = self._update_mask_image_zoomed(mask, rgb_color)
            top_left = self._zoomed_roi[:2]
            # Hide all masks and only display selected
            self._canvas.itemconfig("Mask", state="hidden")
            self._canvas.itemconfig(f"Mask_face_{face_index}", state="normal")
        else:
            display_image = self._update_mask_image_full_frame(mask, rgb_color, face_index)
            top_left = self._meta["top_left"][face_index]

        if len(self._tk_faces) < face_index + 1:
            logger.trace("Adding new Photo Image for face index: %s", face_index)
            self._tk_faces.append(ImageTk.PhotoImage(display_image))
        elif self._tk_faces[face_index].width() != display_image.width:
            logger.trace("Replacing existing Photo Image on width change for face index: %s",
                         face_index)
            self._tk_faces[face_index] = ImageTk.PhotoImage(display_image)
        else:
            logger.trace("Updating existing image")
            self._tk_faces[face_index].paste(display_image)

        self._object_tracker(key,
                             "image",
                             face_index,
                             top_left,
                             {"image": self._tk_faces[face_index], "anchor": tk.NW})

    def _update_mask_image_zoomed(self, mask, rgb_color):
        """ Update the mask image when zoomed in.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The raw mask
        rgb_color: tuple
            The rgb color selected for the mask

        Returns
        -------
        :class: `PIL.Image`
            The zoomed mask image formatted for display
        """
        rgb = np.tile(rgb_color, self._zoomed_dims + (1, )).astype("uint8")
        mask = cv2.resize(mask,
                          tuple(reversed(self._zoomed_dims)),
                          interpolation=cv2.INTER_CUBIC)[..., None]
        rgba = np.concatenate((rgb, mask), axis=2)
        return Image.fromarray(rgba)

    def _update_mask_image_full_frame(self, mask, rgb_color, face_index):
        """ Update the mask image when in full frame view.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The raw mask
        rgb_color: tuple
            The rgb color selected for the mask
        face_index: int
            The index of the face being displayed

        Returns
        -------
        :class: `PIL.Image`
            The full frame mask image formatted for display
        """
        frame_dims = self._globals.current_frame.display_dims
        frame = np.zeros(frame_dims + (1, ), dtype="uint8")
        interpolator = self._meta["interpolator"][face_index]
        slices = self._meta["slices"][face_index]
        mask = cv2.warpAffine(mask,
                              self._meta["affine_matrix"][face_index],
                              frame_dims,
                              frame,
                              flags=cv2.WARP_INVERSE_MAP | interpolator,
                              borderMode=cv2.BORDER_CONSTANT)[slices[0], slices[1]]
        mask = mask[..., None] if mask.ndim == 2 else mask
        rgb = np.tile(rgb_color, mask.shape).astype("uint8")
        rgba = np.concatenate((rgb, np.minimum(mask, self._meta["roi_mask"][face_index])), axis=2)
        return Image.fromarray(rgba)

    def _update_roi_box(self, mask, face_index, color):
        """ Update the region of interest box for the current mask.

        mask: :class:`~lib.align.Mask`
            The current mask object to create an ROI box for
        face_index: int
            The index of the face within the current frame
        color: str
            The hex color code that the mask should be displayed as
        """
        if self._globals.is_zoomed:
            roi = self._zoomed_roi
            box = np.array((roi[0], roi[1], roi[2], roi[1], roi[2], roi[3], roi[0], roi[3]))
        else:
            box = self._scale_to_display(mask.original_roi).flatten()
        top_left = box[:2] - 10
        kwargs = {"fill": color, "font": ("Default", 20, "bold"), "text": str(face_index)}
        self._object_tracker("mask_text", "text", face_index, top_left, kwargs)
        kwargs = {"fill": "", "outline": color, "width": 1}
        self._object_tracker("mask_roi", "polygon", face_index, box, kwargs)
        if self._globals.is_zoomed:
            # Raise box above zoomed image
            self._canvas.tag_raise(f"mask_roi_face_{face_index}")

    # << MOUSE HANDLING >>
    # Mouse cursor display
    def _update_cursor(self, event):
        """ Set the cursor action.

        Update :attr:`_mouse_location` with the current cursor position and display appropriate
        icon.

        Checks whether the mouse is over a mask ROI box and pops the paint icon.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The current tkinter mouse event
        """
        roi_boxes = self._canvas.find_withtag("mask_roi")
        item_ids = set(self._canvas.find_withtag("current")).intersection(roi_boxes)
        if not item_ids:
            self._canvas.config(cursor="")
            self._canvas.itemconfig(self._mouse_location[0], state="hidden")
            self._mouse_location[1] = None
            return
        item_id = list(item_ids)[0]
        tags = self._canvas.gettags(item_id)
        face_idx = int(next(tag for tag in tags if tag.startswith("face_")).split("_")[-1])

        radius = self._brush_radius
        coords = (event.x - radius, event.y - radius, event.x + radius, event.y + radius)
        self._canvas.config(cursor="none")
        self._canvas.coords(self._mouse_location[0], *coords)
        self._canvas.itemconfig(self._mouse_location[0],
                                state="normal",
                                outline=self._cursor_color)
        self._mouse_location[1] = face_idx
        self._canvas.update_idletasks()

    def _control_click(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse whilst
        pressing the control button.

        For editing the mask this will activate the opposite action than what is currently selected
        (e.g. it will erase if draw is set and it will draw if erase is set)

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        self._drag_start(event, control_click=True)

    def _drag_start(self, event, control_click=False):  # pylint:disable=arguments-differ
        """ The action to perform when the user starts clicking and dragging the mouse.

        Paints on the mask with the appropriate draw or erase action.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        control_click: bool, optional
            Indicates whether the control button is depressed when drag has commenced. If ``True``
            then the opposite of the selected action is performed. Default: ``False``
        """
        face_idx = self._mouse_location[1]
        if face_idx is None:
            self._drag_data = {}
            self._drag_callback = None
        else:
            self._drag_data["starting_location"] = np.array((event.x, event.y))
            self._drag_data["control_click"] = control_click
            self._drag_data["color"] = np.array(tuple(int(self._control_color[1:][i:i + 2], 16)
                                                      for i in (0, 2, 4)))
            self._drag_data["opacity"] = self._opacity
            self._get_cursor_shape_mark(
                self._meta["mask"][face_idx],
                np.array(((event.x, event.y), )),
                face_idx)
            self._drag_callback = self._paint

    def _paint(self, event):
        """ Paint or erase from Mask and update cursor on click and drag.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        face_idx = self._mouse_location[1]
        line = np.array((self._drag_data["starting_location"], (event.x, event.y)))
        line, scale = self._transform_points(face_idx, line)
        brush_radius = int(round(self._brush_radius * scale))
        color = 0 if self._edit_mode == "erase" else 255
        # Reverse action on control click
        color = abs(color - 255) if self._drag_data["control_click"] else color
        cv2.line(self._meta["mask"][face_idx],
                 tuple(line[0]),
                 tuple(line[1]),
                 color,
                 brush_radius * 2)
        self._update_mask_image("mask",
                                face_idx,
                                self._drag_data["color"],
                                self._drag_data["opacity"])
        self._drag_data["starting_location"] = np.array((event.x, event.y))
        self._update_cursor(event)

    def _transform_points(self, face_index, points):
        """ Transform the edit points from a full frame or zoomed view back to the mask.

        Parameters
        ----------
        face_index: int
            The index of the face within the current frame
        points: :class:`numpy.ndarray`
            The points that are to be translated from the viewer to the underlying
            Detected Face
        """
        if self._globals.is_zoomed:
            offset = self._zoomed_roi[:2]
            scale = self._internal_size / self._zoomed_dims[0]
            t_points = np.rint((points - offset) * scale).astype("int32").squeeze()
        else:
            scale = self._internal_size / self._meta["mask_roi_size"][face_index]
            t_points = np.expand_dims(points - self._canvas.offset, axis=0)
            t_points = cv2.transform(t_points, self._meta["affine_matrix"][face_index]).squeeze()
            t_points = np.rint(t_points).astype("int32")
        logger.trace("original points: %s, transformed points: %s, scale: %s",
                     points, t_points, scale)
        return t_points, scale

    def _drag_stop(self, event):
        """ The action to perform when the user stops clicking and dragging the mouse.

        If a line hasn't been drawn then draw a circle. Update alignments.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event. Unused but required
        """
        if not self._drag_data:
            return
        face_idx = self._mouse_location[1]
        location = np.array(((event.x, event.y), ))
        if np.array_equal(self._drag_data["starting_location"], location[0]):
            self._get_cursor_shape_mark(self._meta["mask"][face_idx], location, face_idx)
        self._mask_to_alignments(face_idx)
        self._drag_data = {}
        self._update_cursor(event)

    def _get_cursor_shape_mark(self, img, location, face_idx):
        """ Draw object depending on the cursor shape selection. Defaults to circle.

        Parameters
        ----------
        img: Image to draw on (mask)
        location: Cursor location coordinates that will be transformed to correct
            coordinates
        face_index: int
            The index of the face within the current frame
        """
        points, scale = self._transform_points(face_idx, location)
        radius = int(round(self._brush_radius * scale))
        color = 0 if self._edit_mode == "erase" else 255
        # Reverse action on control click
        color = abs(color - 255) if self._drag_data["control_click"] else color

        if self._cursor_shape_name == "Rectangle":
            point2 = points.copy()
            points[0] = points[0] - radius
            points[1] = points[1] - radius
            point2[0] = point2[0] + radius
            point2[1] = point2[1] + radius
            cv2.rectangle(img, tuple(points), tuple(point2), color, -1)
        else:
            cv2.circle(img, tuple(points), radius, color, thickness=-1)

    def _get_cursor_shape(self, x_1=0, y_1=0, x_2=0, y_2=0, outline="black", state="hidden"):
        if self._cursor_shape_name == "Rectangle":
            return self._canvas.create_rectangle(x_1, y_1, x_2, y_2, outline=outline, state=state)
        return self._canvas.create_oval(x_1, y_1, x_2, y_2, outline=outline, state=state)

    def _mask_to_alignments(self, face_index):
        """ Update the annotated mask to alignments.

        Parameters
        ----------
        face_index: int
            The index of the face in the current frame
        """
        mask_type = self._control_vars["display"]["MaskType"].get().lower()
        mask = self._meta["mask"][face_index].astype("float32") / 255.0
        self._det_faces.update.mask(self._globals.frame_index, face_index, mask, mask_type)

    def _adjust_brush_radius(self, increase=True):  # pylint:disable=unused-argument
        """ Adjust the brush radius up or down by 2px.

        Sets the control panel option for brush radius to 2 less or 2 more than its current value

        Parameters
        ----------
        increase: bool, optional
            ``True`` to increment brush radius, ``False`` to decrement. Default: ``True``
        """
        radius_var = self._control_vars["brush"]["BrushSize"]
        current_val = radius_var.get()
        new_val = min(100, current_val + 2) if increase else max(1, current_val - 2)
        logger.trace("Adjusting brush radius from %s to %s", current_val, new_val)
        radius_var.set(new_val)

        delta = new_val - current_val
        if delta == 0:
            return
        current_coords = self._canvas.coords(self._mouse_location[0])
        new_coords = tuple(coord - delta if idx < 2 else coord + delta
                           for idx, coord in enumerate(current_coords))
        logger.trace("Adjusting brush coordinates from %s to %s", current_coords, new_coords)
        self._canvas.coords(self._mouse_location[0], new_coords)
