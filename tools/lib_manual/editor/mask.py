#!/usr/bin/env python3
""" Mask Editor for the manual adjustments tool """
import tkinter as tk

import numpy as np
import cv2
from PIL import Image, ImageTk

from ._base import ControlPanelOption, Editor, logger


class Mask(Editor):
    """ The mask Editor """
    def __init__(self, canvas, alignments, frames):
        self._meta = []
        self._internal_size = 512
        control_text = ("Mask Editor\nEdit the mask."
                        "\n - NB: For Landmark based masks (e.g. components/extended) it is "
                        "better to make sure the landmarks are correct rather than editing the "
                        "mask directly. Any change to the landmarks after editing the mask will "
                        "override your manual edits.")
        key_bindings = {"[": lambda *e, i=False: self._adjust_brush_radius(increase=i),
                        "]": lambda *e, i=True: self._adjust_brush_radius(increase=i)}
        super().__init__(canvas, alignments, frames,
                         control_text=control_text, key_bindings=key_bindings)
        self._mouse_location = [
            self._canvas.create_oval(0, 0, 0, 0, outline="black", state="hidden"), False]

    @property
    def _opacity(self):
        """ float: The mask opacity setting from the control panel from 0.0 - 1.0. """
        annotation = self.__class__.__name__
        return self._annotation_formats[annotation]["mask_opacity"].get() / 100.0

    @property
    def _brush_radius(self):
        """ int: The radius of the brush to use as set in control panel options """
        return self._control_vars[self.__class__.__name__]["brush"]["BrushSize"].get()

    @property
    def _edit_mode(self):
        """ str: The currently selected edit mode based on optional action button.
        One of "draw", "erase" or "zoom" """
        action = [name for name, option in self._actions.items() if option["tk_var"].get()]
        return "draw" if not action else action[0]

    @property
    def _cursor_color(self):
        """ str: The hex code for the selected cursor color """
        color = self._control_vars[self.__class__.__name__]["brush"]["CursorColor"].get()
        return self._canvas.colors[color.lower()]

    def _add_actions(self):
        self._add_action("zoom", "zoom", "Zoom Tool", hotkey="Z")
        self._add_action("draw", "draw", "Draw Tool", hotkey="D")
        self._add_action("erase", "erase", "Erase Tool", hotkey="E")

    def _add_controls(self):
        masks = sorted(msk.title() for msk in list(self._alignments.available_masks) + ["None"])
        default = masks[0] if len(masks) == 1 else [mask for mask in masks if mask != "None"][0]
        self._add_control(ControlPanelOption("Mask type",
                                             str,
                                             group="Display",
                                             choices=masks,
                                             default=default,
                                             is_radio=True,
                                             helptext="Select which mask to edit"))
        self._add_control(ControlPanelOption("Brush Size",
                                             int,
                                             group="Brush",
                                             min_max=(1, 100),
                                             default=10,
                                             rounding=1,
                                             helptext="Set the brush size. ([ - decrease, "
                                                      "] - increase)"))
        self._add_control(ControlPanelOption("Cursor Color",
                                             str,
                                             group="Brush",
                                             choices=sorted(self._canvas.colors),
                                             default="White",
                                             helptext="Select the brush cursor color."))

    def _update_meta(self, key, item, face_index):
        """ Update the meta information for the given object.

        Parameters
        ----------
        key: str
            The object key in the meta dictionary
        item: tkinter object
            The object to be stored
        index:
            The face index that this object pertains to.
        """
        logger.trace("Updating meta dict: (key: %s, object: %s, face_index: %s)",
                     key, item, face_index)
        if key not in self._meta or len(self._meta[key]) - 1 < face_index:
            logger.trace("Creating new item list")
            self._meta.setdefault(key, []).append(item)
        else:
            logger.trace("Appending to existing item list")
            self._meta[key][face_index] = item

    def hide_annotation(self):
        """ Clear the mask :attr:`_meta` dict when hiding the annotation. """
        super().hide_annotation()
        self._meta = dict()

    def update_annotation(self):
        """ Draw the Landmarks and set the objects to :attr:`_object`"""
        position = self._frames.tk_position.get()
        if position != self._meta.get("position", -1):
            # Reset meta information when moving to a new frame
            self._meta = dict(position=position)
        key = self.__class__.__name__
        mask_type = self._control_vars[key]["display"]["MaskType"].get().lower()
        color = self._control_color[1:]
        rgb_color = np.array(tuple(int(color[i:i + 2], 16) for i in (0, 2, 4)))
        roi_color = self._canvas.colors[self._annotation_formats["ExtractBox"]["color"].get()]
        opacity = self._opacity
        for idx, face in enumerate(self._alignments.current_faces):
            mask = face.mask.get(mask_type, None)
            if mask is None:
                continue
            self._set_face_meta_data(mask, idx)
            self._update_mask_image(key.lower(), idx, rgb_color, opacity)
            self._update_roi_box(mask, idx, roi_color)

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
        # TODO Update when mask type changes
        masks = self._meta.get("mask", None)
        if masks is not None and len(masks) - 1 == face_index:
            logger.trace("Meta information already defined for face: %s", face_index)
            return

        logger.debug("Defining meta information for face: %s", face_index)
        scale = self._internal_size / mask.mask.shape[0]
        self._set_full_frame_meta(mask, scale)
        dims = (self._internal_size, self._internal_size)
        self._meta.setdefault("mask", []).append(cv2.resize(mask.mask,
                                                            dims,
                                                            interpolation=cv2.INTER_CUBIC))

    def _set_full_frame_meta(self, mask, mask_scale):
        """ Sets the meta information for displaying the mask in full frame mode.

        Parameters
        ----------
        mask: :class:`lib.faces_detect.Mask`
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
        frame_dims = self._frames.current_meta_data["display_dims"]
        scaled_mask_roi = np.rint(mask.original_roi * self._frames.current_scale).astype("int32")

        # Scale and clip the ROI to fit within display frame boundaries
        clipped_roi = scaled_mask_roi.clip(min=(0, 0), max=frame_dims)

        # Obtain min and max points to get ROI as a rectangle
        min_max = dict(min=clipped_roi.min(axis=0), max=clipped_roi.max(axis=0))

        # Create a bounding box rectangle ROI
        roi_dims = np.rint((min_max["max"][1] - min_max["min"][1],
                            min_max["max"][0] - min_max["min"][0])).astype("uint16")
        roi_mask = np.zeros(roi_dims, dtype="uint8")[..., None]

        # Block out areas outside of the actual mask ROI polygon
        roi_corners = np.expand_dims(scaled_mask_roi - min_max["min"], axis=0)
        cv2.fillPoly(roi_mask, roi_corners, 255)
        logger.trace("Setting Full Frame mask ROI. shape: %s", roi_mask.shape)

        # obtain the slices for cropping mask from full frame
        xslice = slice(int(round(min_max["min"][1])), int(round(min_max["max"][1])))
        yslice = slice(int(round(min_max["min"][0])), int(round(min_max["max"][0])))

        # Adjust affine matrix for internal mask size and display dimensions
        in_adjustment = np.array([[mask_scale, 0., 0.], [0., mask_scale, 0.]])
        out_adjustment = np.array([[1 / self._frames.current_scale, 0., 0.],
                                   [0., 1 / self._frames.current_scale, 0.],
                                   [0., 0., 1.]])

        in_matrix = np.dot(in_adjustment,
                           np.concatenate((mask.affine_matrix, np.array([[0., 0., 1.]]))))
        affine_matrix = np.dot(in_matrix, out_adjustment)

        # Get the size of the mask roi box in the frame
        side_a = scaled_mask_roi[1][0] - scaled_mask_roi[0][0]
        side_b = scaled_mask_roi[1][1] - scaled_mask_roi[0][1]
        mask_roi_size = (side_a ** 2 + side_b ** 2) ** 0.5

        self._meta.setdefault("roi_mask", []).append(roi_mask)
        self._meta.setdefault("affine_matrix", []).append(affine_matrix)
        self._meta.setdefault("interpolator", []).append(mask.interpolator)
        self._meta.setdefault("slices", []).append((xslice, yslice))
        self._meta.setdefault("top_left", []).append(min_max["min"] + self._canvas.offset)
        self._meta.setdefault("mask_roi_size", []).append(mask_roi_size)

    def _update_mask_image(self, key, face_index, rgb_color, opacity):
        """ Obtain a full frame mask, overlay over image. """
        mask = (self._meta["mask"][face_index] * opacity).astype("uint8")
        if self._is_zoomed:
            display_image = self._update_mask_image_zoomed(mask, rgb_color)
            top_left = self._zoomed_roi[:2]
        else:
            display_image = self._update_mask_image_full_frame(mask, rgb_color, face_index)
            top_left = self._meta["top_left"][face_index]
        self._update_meta("image", display_image, face_index)
        self._object_tracker(key,
                             "image",
                             face_index,
                             0,
                             top_left,
                             dict(image=display_image, anchor=tk.NW))

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
        :class: `ImageTk.PhotoImage`
            The zoomed mask image formatted for display
        """
        rgb = np.tile(rgb_color, self._zoomed_dims + (1, )).astype("uint8")
        mask = cv2.resize(mask, self._zoomed_dims, interpolation=cv2.INTER_CUBIC)[..., None]
        rgba = np.concatenate((rgb, mask), axis=2)
        display = ImageTk.PhotoImage(Image.fromarray(rgba))
        return display

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
        :class: `ImageTk.PhotoImage`
            The full frame mask image formatted for display
        """
        frame_dims = self._frames.current_meta_data["display_dims"]
        frame = np.zeros(frame_dims + (1, ), dtype="uint8")
        interpolator = self._meta["interpolator"][face_index]
        slices = self._meta["slices"][face_index]
        mask = cv2.warpAffine(mask,
                              self._meta["affine_matrix"][face_index],
                              frame_dims,
                              frame,
                              flags=cv2.WARP_INVERSE_MAP | interpolator,
                              borderMode=cv2.BORDER_CONSTANT)[slices[0], slices[1]][..., None]
        rgb = np.tile(rgb_color, mask.shape).astype("uint8")
        rgba = np.concatenate((rgb, np.minimum(mask, self._meta["roi_mask"][face_index])), axis=2)
        display = ImageTk.PhotoImage(Image.fromarray(rgba))
        return display

    def _update_roi_box(self, mask, face_index, color):
        """ Update the region of interest box for the current mask """
        keys = ("text", "roibox")
        if self._is_zoomed:
            box = np.array((self._zoomed_roi[0], self._zoomed_roi[1],
                            self._zoomed_roi[2], self._zoomed_roi[1],
                            self._zoomed_roi[2], self._zoomed_roi[3],
                            self._zoomed_roi[0], self._zoomed_roi[3]))
        else:
            box = self._scale_to_display(mask.original_roi).flatten()
        top_left = box[:2] - 10
        kwargs = dict(fill=color, font=("Default", 20, "bold"), text=str(face_index))
        self._object_tracker(keys[0], "text", face_index, 0, top_left, kwargs)
        kwargs = dict(fill="", outline=color, width=1)
        self._object_tracker(keys[1], "polygon", face_index, 0, box, kwargs)
        if self._is_zoomed:
            # Raise box above zoomed image
            self._canvas.tag_raise(self._objects[keys[1]][face_index][0])

    # << MOUSE HANDLING >>
    # Mouse cursor display

    def _update_cursor(self, event):
        """ Update the cursor for brush painting and set :attr:`_mouse_location`. """
        roi_boxes = self._flatten_list(self._objects.get("roibox", []))
        item_ids = set(self._canvas.find_withtag("current")).intersection(roi_boxes)
        if not item_ids:
            self._canvas.config(cursor="")
            self._canvas.itemconfig(self._mouse_location[0], state="hidden")
            self._mouse_location[1] = None
            return
        item_id = list(item_ids)[0]
        obj_idx = [face_idx
                   for face_idx, face in enumerate(self._objects["roibox"])
                   if item_id in face][0]
        if self._edit_mode == "zoom":
            self._canvas.config(cursor="sizing")
        else:
            radius = self._brush_radius
            coords = (event.x - radius, event.y - radius, event.x + radius, event.y + radius)
            self._canvas.config(cursor="none")
            self._canvas.coords(self._mouse_location[0], *coords)
            self._canvas.itemconfig(self._mouse_location[0],
                                    state="normal",
                                    outline=self._cursor_color)
        self._mouse_location[1] = obj_idx
        self._canvas.update_idletasks()

    def _drag_start(self, event):
        """ The action to perform when the user starts clicking and dragging the mouse.

        Collect information about the object being clicked on and add to :attr:`_drag_data`

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The tkinter mouse event.
        """
        face_idx = self._mouse_location[1]
        if face_idx is None:
            self._drag_data = dict()
            self._drag_callback = None
        elif self._edit_mode == "zoom":
            self._drag_data = dict()
            self._drag_callback = None
            self._zoom_face(face_idx)
        else:
            self._drag_data["starting_location"] = np.array((event.x, event.y))
            self._drag_callback = self._paint

    def _zoom_face(self, face_index):
        self._canvas.toggle_image_display()
        coords = (self._frames.display_dims[0] / 2, self._frames.display_dims[1] / 2)
        if self._is_zoomed:
            face = self._alignments.get_aligned_face_at_index(face_index)[..., 2::-1]
            display = ImageTk.PhotoImage(Image.fromarray(face))
            self._update_meta("zoomed", display, face_index)
            kwargs = dict(image=display, anchor=tk.CENTER)
        else:
            kwargs = dict(state="hidden")
        self._object_tracker("zoom", "image", face_index, 0, coords, kwargs)
        self._canvas.tag_lower(self._objects["zoom"][face_index][0])
        self._frames.tk_update.set(True)

    def _paint(self, event):
        """ Paint or erase from Mask and update cursor on click and drag """
        face_idx = self._mouse_location[1]
        mask = self._meta["mask"][face_idx]
        line = np.array((self._drag_data["starting_location"], (event.x, event.y)))

        if self._is_zoomed:
            offset = self._zoomed_roi[:2]
            scale = mask.shape[0] / self._zoomed_dims[0]
            line = np.rint((line - offset) * scale).astype("int32")
        else:
            scale = mask.shape[0] / self._meta["mask_roi_size"][face_idx]
            line = np.expand_dims(line - self._canvas.offset, axis=0)
            line = cv2.transform(line, self._meta["affine_matrix"][face_idx]).squeeze()
            line = np.rint(line).astype("int32")

        brush_radius = int(round(self._brush_radius * scale))
        cv2.line(self._meta["mask"][face_idx],
                 tuple(line[0]),
                 tuple(line[1]),
                 0 if self._edit_mode == "erase" else 255,
                 brush_radius * 2)

        self._drag_data["starting_location"] = np.array((event.x, event.y))
        self._frames.tk_update.set(True)
        self._update_cursor(event)

    def _adjust_brush_radius(self, increase=True):  # pylint:disable=unused-argument
        """ Adjust the brush radius up or down by 1px.

        Sets the control panel option for brush radius to 1 less or 1 more than its current value

        Parameters
        ----------
        increase: bool, optional
            ``True`` to increment brush radius, ``False`` to decrement. Default: ``True``
        """
        radius_var = self._control_vars[self.__class__.__name__]["brush"]["BrushSize"]
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
