#!/usr/bin/env python3
""" functions for implementing themes in Faceswap's GUI """
import logging
import os
import tkinter as tk
from tkinter import ttk

import numpy as np

from lib.serializer import get_serializer
from lib.utils import FaceswapError


logger = logging.getLogger(__name__)


class Style():
    """ Set the overarching theme and customize widgets.

    Parameters
    ----------
    default_font: tuple
        The name and size of the default font
    root: :class:`tkinter.Tk`
        The root tkinter object
    path_cache: str
        The path to the GUI's cache
    """
    def __init__(self, default_font, root, path_cache):
        self._root = root
        self._font = default_font
        default = os.path.join(path_cache, "themes", "default.json")
        self._user_theme = get_serializer("json").load(default)
        self._style = ttk.Style()
        self._widgets = _Widgets(self._style)
        self._set_styles()

    @property
    def user_theme(self):
        """ dict: The currently selected user theme. """
        return self._user_theme

    def _set_styles(self):
        """ Configure widget theme and styles """
        self._config_settings_group()
        # Command page
        theme = self._user_theme["command_tabs"]
        self._widgets.notebook("CPanel",
                               theme["frame_border"],
                               theme["tab_color"],
                               theme["tab_selected"],
                               theme["tab_hover"])

        # Settings Popup
        self._style.configure("SPanel.Header1.TLabel",
                              font=(self._font[0], self._font[1] + 4, "bold"))
        self._style.configure("SPanel.Header2.TLabel",
                              font=(self._font[0], self._font[1] + 2, "bold"))
        # Console
        theme = self._user_theme["console"]
        console_sbar = tuple(tuple(theme[f"scrollbar_{area}_{state}"]
                                   for state in ("normal", "disabled", "active"))
                             for area in ("background", "foreground", "border"))
        self._widgets.scrollbar("Console",
                                theme["scrollbar_trough"],
                                theme["scrollbar_border"],
                                *console_sbar)
        self._widgets.frame("Console",
                            theme["background_color"],
                            theme["border_color"],
                            borderwidth=1)

    def _config_settings_group(self):
        """ Configures the style of the control panel entry boxes. Used for inputting Faceswap
        options or controlling plugin settings. """
        theme = self._user_theme["group_panel"]
        for panel_type in ("CPanel", "SPanel"):
            if panel_type == "SPanel":  # Merge in Settings Panel overrides
                theme = {**theme, **self._user_theme["group_settings"]}
            self._style.configure(f"{panel_type}.Holder.TFrame",
                                  background=theme["panel_background"])
            # Header Colors on option/group controls
            self._style.configure(f"{panel_type}.Group.TLabelframe.Label",
                                  foreground=theme["header_color"])
            self._style.configure(f"{panel_type}.Groupheader.TLabel",
                                  background=theme["header_color"],
                                  foreground=theme["header_font"],
                                  font=(self._font[0], self._font[1], "bold"))
            # Widgets and specific areas
            self._group_panel_widgets(panel_type, theme)
            self._group_panel_infoheader(panel_type, theme)
            self._widgets.slider(panel_type,
                                 theme["control_color"],
                                 theme["control_active"],
                                 self._user_theme["group_panel"]["group_background"])
            backgrounds = (theme["control_color"],
                           theme["control_disabled"],
                           theme["control_active"])
            foregrounds = (theme["control_disabled"],
                           theme["control_color"],
                           theme["control_disabled"])
            borders = (theme["header_color"], theme["control_color"], theme["header_color"])
            self._widgets.scrollbar(panel_type,
                                    theme["scrollbar_trough"],
                                    theme["scrollbar_border"],
                                    backgrounds,
                                    foregrounds,
                                    borders)
            self._widgets.combobox(panel_type,
                                   theme["control_color"],
                                   theme["control_active"],
                                   theme["control_disabled"],
                                   theme["header_color"],
                                   theme["group_background"],
                                   theme["group_font"])

    def _group_panel_infoheader(self, key, theme):
        """ Set the theme for the information header box that appears at the top of each group
        panel

        Parameters
        ----------
        key: str
            The section that the slider will belong to
        theme: dict
            The user configuration theme options
        """
        self._widgets.frame(f"{key}.InfoHeader",
                            theme["info_color"],
                            theme["info_border"],
                            borderwidth=1)

        self._style.configure(f"{key}.InfoHeader.TLabel",
                              background=theme["info_color"],
                              foreground=theme["info_font"],
                              font=(self._font[0], self._font[1], "bold"))
        self._style.configure(f"{key}.InfoBody.TLabel",
                              background=theme["info_color"],
                              foreground=theme["info_font"])

    def _group_panel_widgets(self, key, theme):
        """ Configure the foreground and background colors of common widgets.

        Parameters
        ----------
        key: str
            The section that the slider will belong to
        theme: dict
            The user configuration theme options
        """
        # Put a border on a group's sub-frame
        self._widgets.frame(f"{key}.Subframe.Group",
                            theme["group_background"],
                            theme["group_border"],
                            borderwidth=1)

        # Background and Foreground of widgets and labels
        for lbl in ["TLabel", "TFrame", "TLabelframe", "TCheckbutton", "TRadiobutton",
                    "TLabelframe.Label"]:
            self._style.configure(f"{key}.Group.{lbl}",
                                  background=theme["group_background"],
                                  foreground=theme["group_font"])


class _Widgets():
    """ Create custom ttk widget layouts for themed widgets.

    Parameters
    ----------
    style: :class:`ttk.Style`
        The master style object
    """
    def __init__(self, style):
        self._images = _TkImage()
        self._style = style

    def combobox(self, key, control_color, active_color, arrow_color, control_border, field_color,
                 field_border):
        """ Combo-boxes are fairly complex to style.

        Parameters
        ----------
        key: str
            The section that the slider will belong to
        control_color: str
            The color of inactive combo pull down button
        active_color: str
            The color of combo pull down button when it is hovered or pressed
        arrow_color: str
            The color of the combo pull down arrow
        control_border: str
            The color of the combo pull down button border
        field_color: str
            The color of the input field's background
        field_border: str
            The color of the input field's border
        """
        # All the stock down arrow images are bad
        images = {}
        for state in ("active", "normal"):
            images[f"arrow_{state}"] = self._images.get_image(
                (20, 20),
                control_color if state == "normal" else active_color,
                foreground=arrow_color,
                pattern="arrow",
                thickness=2,
                border_width=1,
                border_color=control_border)

        self._style.element_create(f"{key}.Combobox.downarrow",
                                   "image",
                                   images["arrow_normal"],
                                   ("active", images["arrow_active"]),
                                   ("pressed", images["arrow_active"]),
                                   sticky="e",
                                   width=20)

        # None of the themes give us the border control we need, so create an image
        box = self._images.get_image((16, 16),
                                     field_color,
                                     border_width=1,
                                     border_color=field_border)
        self._style.element_create(f"{key}.Combobox.field",
                                   "image",
                                   box,
                                   border=1,
                                   padding=(6, 0, 0, 0))

        # Set a layout so we can access required params
        self._style.layout(f"{key}.TCombobox", [
            (f"{key}.Combobox.field", {
                "children": [
                    (f"{key}.Combobox.downarrow", {"side": "right", "sticky": "ns"}),
                    (f"{key}.Combobox.padding", {
                        "expand": "1",
                        "sticky": "nswe",
                        "children": [(f"{key}.Combobox.focus", {
                            "expand": "1",
                            "sticky": "nswe",
                            "children": [(f"{key}.Combobox.textarea", {"sticky": "nswe"})]})]})],
                "sticky": "nswe"})])

    def frame(self, key, background, border, borderwidth=1):
        """ Create a custom frame widget for controlling background and border colors.

        Parameters
        ----------
        key: str
            The section that the Frame will belong to
        background: str
            The hex code for the background of the frame
        border: str
            The hex code for the border of the frame
        """
        self._style.element_create(f"{key}.Frame.border", "from", "alt")
        self._style.layout(f"{key}.TFrame",
                           [(f"{key}.Frame.border", {"sticky": "nswe"})])
        self._style.configure(f"{key}.TFrame",
                              background=background,
                              relief=tk.SOLID,
                              borderwidth=borderwidth,
                              bordercolor=border)

    def notebook(self, key, frame_border, tab_color, tab_selected, tab_hover):
        """ Create a custom notebook widget so we can control the colors.

        Parameters
        ----------
        key: str
            The section that the scrollbar will belong to
        frame_border: str
            The border color around the tab's contents
        tab_color: str
            The color of non selected tabs
        tab_selected: str
            The color of selected tabs
        tab_hover: str
            The color of hovered tabs
        """
        # TODO This lags out the GUI, so need to test where this is failing prior to implementing
        client = self._images.get_image((8, 8), frame_border)
        self._style.element_create(f"{key}.Notebook.client", "image", client, border=1)

        tabs = [self._images.get_image((8, 8), color)
                for color in (tab_color, tab_selected, tab_hover)]

        self._style.element_create(f"{key}.Notebook.tab",
                                   "image",
                                   tabs[0],
                                   ("selected", tabs[1]),
                                   ("active", tabs[2]),
                                   padding=(0, 2, 0, 0),
                                   border=3)

        self._style.layout(f"{key}.TNotebook", [(f"{key}.Notebook.client", {"sticky": "nswe"})])
        self._style.layout(f"{key}.TNotebook.Tab", [
            (f"{key}.Notebook.tab", {
                "sticky": "nswe",
                "children": [
                    ("Notebook.padding", {
                        "side": "top",
                        "sticky": "nswe",
                        "children": [
                            ("Notebook.focus", {
                                "side": "top",
                                "sticky": "nswe",
                                "children": [("Notebook.label", {"side": "top", "sticky": ""})]
                            })]
                    })]
            })])

        self._style.configure(f"{key}.TNotebook", tabmargins=(0, 2, 0, 0))
        self._style.configure(f"{key}.TNotebook.Tab", padding=(6, 2, 6, 2), expand=(0, 0, 2))
        self._style.configure(f"{key}.TNotebook.Tab", expand=("selected", (1, 2, 4, 2)))

    def scrollbar(self, key, trough_color, border_color, control_backgrounds, control_foregrounds,
                  control_borders):
        """ Create a custom scroll bar widget so we can control the colors.

        Parameters
        ----------
        key: str
            The section that the scrollbar will belong to
        theme: dict
            The theme options for a scroll bar. The dict should contain the keys: `background`,
            `foreground`, `border`, with each item containing a tuple of the colors for the states
            `normal`, `disabled` and `active` respectively
        trough_color: str
            The hex code for the scrollbar trough color
        border_color: str
            The hex code for the scrollbar border color
        control_backgrounds: tuple
            Tuple of length 3 for the button and slider colors for the states `normal`,
            `disabled`, `active`
        control_foregrounds: tuple
            Tuple of length 3 for the button arrow colors for the states `normal`,
            `disabled`, `active`
        control_borders: tuple
            Tuple of length 3 for the borders of the buttons and slider for the states `normal`,
            `disabled`, `active`
        """
        logger.debug("Creating scrollbar: (key: %s, trough_color: %s, border_color: %s, "
                     "control_backgrounds: %s, control_foregrounds: %s, control_borders: %s)",
                     key, trough_color, border_color, control_backgrounds, control_foregrounds,
                     control_borders)
        images = {}
        for idx, state in enumerate(("normal", "disabled", "active")):
            # Create arrow and slider widgets for each state
            img_args = ((16, 16), control_backgrounds[idx])
            for dir_ in ("up", "down"):
                images[f"img_{dir_}_{state}"] = self._images.get_image(
                    *img_args,
                    foreground=control_foregrounds[idx],
                    pattern="arrow",
                    direction=dir_,
                    thickness=4,
                    border_width=1,
                    border_color=control_borders[idx])
            images[f"img_thumb_{state}"] = self._images.get_image(
                *img_args,
                border_width=1,
                border_color=control_borders[idx])

        for element in ("thumb", "uparrow", "downarrow"):
            # Create the elements with the new images
            lookup = element.replace("arrow", "")
            args = (f"{key}.Vertical.Scrollbar.{element}",
                    "image",
                    images[f"img_{lookup}_normal"],
                    ("disabled", images[f"img_{lookup}_disabled"]),
                    ("pressed !disabled", images[f"img_{lookup}_active"]),
                    ("active !disabled", images[f"img_{lookup}_active"]))
            kwargs = {"border": 1, "sticky": "ns"} if element == "thumb" else {}
            self._style.element_create(*args, **kwargs)

        # Get a configurable trough
        self._style.element_create(f"{key}.Vertical.Scrollbar.trough", "from", "clam")

        self._style.layout(
            f"{key}.Vertical.TScrollbar",
            [(f"{key}.Vertical.Scrollbar.trough", {
                "sticky": "ns",
                "children": [
                    (f"{key}.Vertical.Scrollbar.uparrow", {"side": "top", "sticky": ""}),
                    (f"{key}.Vertical.Scrollbar.downarrow", {"side": "bottom", "sticky": ""}),
                    (f"{key}.Vertical.Scrollbar.thumb", {"expand": "1", "sticky": "nswe"})
                ]
            })])
        self._style.configure(f"{key}.Vertical.TScrollbar",
                              troughcolor=trough_color,
                              bordercolor=border_color,
                              troughrelief=tk.SOLID,
                              troughborderwidth=1)

    def slider(self, key, control_color, active_color, trough_color):
        """ Take a copy of the default ttk.Scale widget and replace the slider element with a
        version we can control the color and shape of.

        Parameters
        ----------
        key: str
            The section that the slider will belong to
        control_color: str
            The color of inactive slider and up down buttons
        active_color: str
            The color of slider and up down buttons when they are hovered or pressed
        trough_color: str
            The color of the scroll bar's trough
        """
        img_slider = self._images.get_image((10, 25), control_color)
        img_slider_alt = self._images.get_image((10, 25), active_color)

        self._style.element_create(f"{key}.Horizontal.Scale.trough", "from", "alt")
        self._style.element_create(f"{key}.Horizontal.Scale.slider",
                                   "image",
                                   img_slider,
                                   ("active", img_slider_alt))

        self._style.layout(
            f"{key}.Horizontal.TScale",
            [(f"{key}.Scale.focus", {
                "expand": "1",
                "sticky": "nswe",
                "children": [
                    (f"{key}.Horizontal.Scale.trough", {
                        "expand": "1",
                        "sticky": "nswe",
                        "children": [
                            (f"{key}.Horizontal.Scale.track", {"sticky": "we"}),
                            (f"{key}.Horizontal.Scale.slider", {"side": "left", "sticky": ""})
                            ]
                        })
                ]
            })])

        self._style.configure(f"{key}.Horizontal.TScale",
                              background=trough_color,
                              groovewidth=4,
                              troughcolor=trough_color)


class _TkImage():
    """ Create a tk image for a given pattern and shape.
    """
    def __init__(self):
        self._cache = []  # We need to keep a reference to every image created

    # Numpy array patterns
    @classmethod
    def _get_solid(cls, dimensions):
        """ Return a solid background color pattern.

        Parameters
        ----------
        dimensions: tuple
            The (`width`, `height`) of the desired tk image

        Returns
        -------
        :class:`numpy.ndarray`
            A 2D, UINT8 array of shape (height, width) of all zeros
        """
        return np.zeros((dimensions[1], dimensions[0]), dtype="uint8")

    @classmethod
    def _get_arrow(cls, dimensions, thickness, direction):
        """ Return a background color with a "v" arrow in foreground color

        Parameters
        ----------
        dimensions: tuple
            The (`width`, `height`) of the desired tk image
        thickness: int
            The thickness of the pattern to be drawn
        direction: ["left", "up", "right", "down"]
            The direction that the pattern should be facing

        Returns
        -------
        :class:`numpy.ndarray`
            A 2D, UINT8 array of shape (height, width) of all zeros
        """
        square_size = min(dimensions[1], dimensions[0])
        if square_size < 16 or any(dim % 2 != 0 for dim in dimensions):
            raise FaceswapError("For arrow image, the minimum size across any axis must be 8 and "
                                "dimensions must all be divisible by 2")
        crop_size = (square_size // 16) * 16
        draw_rows = int(6 * crop_size / 16)
        start_row = dimensions[1] // 2 - draw_rows // 2
        initial_indent = 2 * (crop_size // 16) + (dimensions[0] - crop_size) // 2

        retval = np.zeros((dimensions[1], dimensions[0]), dtype="uint8")
        for i in range(start_row, start_row + draw_rows):
            indent = initial_indent + i - start_row
            join = (min(indent + thickness, dimensions[0] // 2),
                    max(dimensions[0] - indent - thickness, dimensions[0] // 2))
            retval[i, np.r_[indent:join[0], join[1]:dimensions[0] - indent]] = 1
        if direction in ("right", "left"):
            retval = np.rot90(retval)
        if direction in ("up", "left"):
            retval = np.flip(retval)
        return retval

    def get_image(self,
                  dimensions,
                  background,
                  foreground=None,
                  pattern="solid",
                  border_width=0,
                  border_color=None,
                  thickness=2,
                  direction="down"):
        """ Obtain a tk image.

        Generates the requested image and stores in cache.

        Parameters
        ----------
        dimensions: tuple
            The (`width`, `height`) of the desired tk image
        background: str
            The hex code for the background (main) color
        foreground: str, optional
            The hex code for the background (secondary) color. If ``None`` is provided then a
            solid background color image will be returned. Default: ``None``
        pattern: ["solid", "arrow"], optional
            The pattern to generate for the tk image. Default: `"solid"`
        border_width: int, optional
            The thickness of foreground border to apply. Default: 0
        border_color: int, optional
            The color of the border, if one is to be created. Default: ``None`` (use foreground
            color)
        thickness: int, optional
            The thickness of the pattern to be drawn. Default: `2`
        direction: ["left", "up", "right", "down"], optional
            The direction that the pattern should be facing. Default: `"down"`
        """
        foreground = foreground if foreground else background
        border_color = border_color if border_color else foreground

        args = [dimensions]
        if pattern.lower() == "arrow":
            args.extend([thickness, direction])
        if pattern.lower() == "border":
            args.extend([thickness])
        pattern = getattr(self, f"_get_{pattern.lower()}")(*args)

        if border_width > 0:
            border = np.ones_like(pattern) + 1
            border[border_width:-border_width,
                   border_width:-border_width] = pattern[border_width:-border_width,
                                                         border_width:-border_width]
            pattern = border

        return self._create_photoimage(background, foreground, border_color, pattern)

    def _create_photoimage(self, background, foreground, border, pattern):
        """ Create a tkinter PhotoImage and populate it with the requested color pattern.

        Parameters
        ----------
        background: str
            The hex code for the background (main) color
        foreground: str
            The hex code for the foreground (secondary) color
        border: str
            The hex code for the border color
        pattern: class:`numpy.ndarray`
            The pattern for the final image with background colors marked as 0 and foreground
            colors marked as 1
        """
        image = tk.PhotoImage(width=pattern.shape[1], height=pattern.shape[0])
        self._cache.append(image)

        pixels = "} {".join(" ".join(foreground
                                     if pxl == 1 else border if pxl == 2 else background
                                     for pxl in row)
                            for row in pattern)
        image.put("{" + pixels + "}")
        return image
