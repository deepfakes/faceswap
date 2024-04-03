#!/usr/bin/python
""" The pop up preview window for Faceswap.

If Tkinter is installed, then this will be used to manage the preview image, otherwise we
fallback to opencv's imshow
"""
from __future__ import annotations
import logging
import os
import sys
import tkinter as tk
import typing as T

from datetime import datetime
from platform import system
from tkinter import ttk
from math import ceil, floor

from PIL import Image, ImageTk

import cv2

from .preview_cv import PreviewBase, TriggerKeysType

if T.TYPE_CHECKING:
    import numpy as np
    from .preview_cv import PreviewBuffer, TriggerType

logger = logging.getLogger(__name__)


class _Taskbar():
    """ Taskbar at bottom of Preview window

    Parameters
    ----------
    parent: :class:`tkinter.Frame`
        The parent frame that holds the canvas and taskbar
    taskbar: :class:`tkinter.ttk.Frame` or ``None``
        None if preview is a pop-up window otherwise ttk.Frame if taskbar is managed by the GUI
    """
    def __init__(self, parent: tk.Frame, taskbar: ttk.Frame | None) -> None:
        logger.debug("Initializing %s (parent: '%s', taskbar: %s)",
                     self.__class__.__name__, parent, taskbar)
        self._is_standalone = taskbar is None
        self._gui_mapped: list[tk.Widget] = []
        self._frame = tk.Frame(parent) if taskbar is None else taskbar

        self._min_max_scales = (20, 400)
        self._vars = {"save": tk.BooleanVar(),
                      "scale": tk.StringVar(),
                      "slider": tk.IntVar(),
                      "interpolator": tk.IntVar()}
        self._interpolators = [("nearest_neighbour", cv2.INTER_NEAREST),
                               ("bicubic", cv2.INTER_CUBIC)]
        self._scale = self._add_scale_combo()
        self._slider = self._add_scale_slider()
        self._add_interpolator_radio()

        if self._is_standalone:
            self._add_save_button()
            self._frame.pack(side=tk.BOTTOM, fill=tk.X, padx=2, pady=2)

        logger.debug("Initialized %s ('%s')", self.__class__.__name__, self)

    @property
    def min_scale(self) -> int:
        """ int: The minimum allowed scale """
        return self._min_max_scales[0]

    @property
    def max_scale(self) -> int:
        """ int: The maximum allowed scale """
        return self._min_max_scales[1]

    @property
    def save_var(self) -> tk.BooleanVar:
        """:class:`tkinter.IntVar`: Variable which is set to ``True`` when the save button has
        been. pressed """
        retval = self._vars["save"]
        assert isinstance(retval, tk.BooleanVar)
        return retval

    @property
    def scale_var(self) -> tk.StringVar:
        """:class:`tkinter.StringVar`: The variable holding the currently selected "##%" formatted
        percentage scaling amount displayed in the Combobox. """
        retval = self._vars["scale"]
        assert isinstance(retval, tk.StringVar)
        return retval

    @property
    def slider_var(self) -> tk.IntVar:
        """:class:`tkinter.IntVar`: The variable holding the currently selected percentage scaling
        amount in the slider. """
        retval = self._vars["slider"]
        assert isinstance(retval, tk.IntVar)
        return retval

    @property
    def interpolator_var(self) -> tk.IntVar:
        """:class:`tkinter.IntVar`: The variable holding the CV2 Interpolator Enum. """
        retval = self._vars["interpolator"]
        assert isinstance(retval, tk.IntVar)
        return retval

    def _track_widget(self, widget: tk.Widget) -> None:
        """ If running embedded in the GUI track the widgets so that they can be destroyed if
        the preview is disabled """
        if self._is_standalone:
            return
        logger.debug("Tracking option bar widget for GUI: %s", widget)
        self._gui_mapped.append(widget)

    def _add_scale_combo(self) -> ttk.Combobox:
        """ Add a scale combo for selecting zoom amount.

        Returns
        -------
        :class:`tkinter.ttk.Combobox`
            The Combobox widget
        """
        logger.debug("Adding scale combo")
        self.scale_var.set("100%")
        scale = ttk.Combobox(self._frame,
                             textvariable=self.scale_var,
                             values=["Fit"],
                             state="readonly",
                             width=10)
        scale.pack(side=tk.RIGHT)
        scale.bind("<FocusIn>", self._clear_combo_focus)  # Remove auto-focus on widget text box
        self._track_widget(scale)
        logger.debug("Added scale combo: '%s'", scale)
        return scale

    def _clear_combo_focus(self, *args) -> None:  # pylint:disable=unused-argument
        """ Remove the highlighting and stealing of focus that the combobox annoyingly
        implements. """
        logger.debug("Clearing scale combo focus")
        self._scale.selection_clear()
        self._scale.winfo_toplevel().focus_set()
        logger.debug("Cleared scale combo focus")

    def _add_scale_slider(self) -> tk.Scale:
        """ Add a scale slider for zooming the image.

        Returns
        -------
        :class:`tkinter.Scale`
            The scale widget
        """
        logger.debug("Adding scale slider")
        self.slider_var.set(100)
        slider = tk.Scale(self._frame,
                          orient=tk.HORIZONTAL,
                          to=self.max_scale,
                          showvalue=False,
                          variable=self.slider_var,
                          command=self._on_slider_update)
        slider.pack(side=tk.RIGHT)
        self._track_widget(slider)
        logger.debug("Added scale slider: '%s'", slider)
        return slider

    def _add_interpolator_radio(self) -> None:
        """ Add a radio box to choose interpolator """
        frame = tk.Frame(self._frame)
        for text, mode in self._interpolators:
            logger.debug("Adding %s radio button", text)
            radio = tk.Radiobutton(frame, text=text, value=mode, variable=self.interpolator_var)
            radio.pack(side=tk.LEFT, anchor=tk.W)
            self._track_widget(radio)

            logger.debug("Added %s radio button", radio)
        self.interpolator_var.set(cv2.INTER_NEAREST)
        frame.pack(side=tk.RIGHT)
        self._track_widget(frame)

    def _add_save_button(self) -> None:
        """ Add a save button for saving out original preview """
        logger.debug("Adding save button")
        button = tk.Button(self._frame,
                           text="Save",
                           cursor="hand2",
                           command=lambda: self.save_var.set(True))
        button.pack(side=tk.LEFT)
        logger.debug("Added save burron: '%s'", button)

    def _on_slider_update(self, value) -> None:
        """ Callback for when the scale slider is adjusted. Adjusts the combo box display to the
        current slider value.

        Parameters
        ----------
        value: int
            The value that the slider has been set to
         """
        self.scale_var.set(f"{value}%")

    def set_min_max_scale(self, min_scale: int, max_scale: int) -> None:
        """ Set the minimum and maximum value that we allow an image to be scaled down to. This
        impacts the slider and combo box min/max values:

        Parameters
        ----------
        min_scale: int
            The minimum percentage scale that is permitted
        max_scale: int
            The maximum percentage scale that is permitted
        """
        logger.debug("Setting min/max scales: (min: %s, max: %s)", min_scale, max_scale)
        self._min_max_scales = (min_scale, max_scale)
        self._slider.config(from_=self.min_scale, to=max_scale)
        scales = [10, 25, 50, 75, 100, 200, 300, 400, 800]
        if min_scale not in scales:
            scales.insert(0, min_scale)
        if max_scale not in scales:
            scales.append(max_scale)
        choices = ["Fit", *[f"{x}%" for x in scales if self.max_scale >= x >= self.min_scale]]
        self._scale.config(values=choices)
        logger.debug("Set min/max scale. min_max_scales: %s, scale combo choices: %s",
                     self._min_max_scales, choices)

    def cycle_interpolators(self, *args) -> None:  # pylint:disable=unused-argument
        """ Cycle interpolators on a keypress callback """
        current = next(i for i in self._interpolators if i[1] == self.interpolator_var.get())
        next_idx = self._interpolators.index(current) + 1
        next_idx = 0 if next_idx == len(self._interpolators) else next_idx
        self.interpolator_var.set(self._interpolators[next_idx][1])

    def destroy_widgets(self) -> None:
        """ Remove the taskbar widgets when the preview within the GUI has been disabled """
        if self._is_standalone:
            return

        for widget in self._gui_mapped:
            if widget.winfo_ismapped():
                logger.debug("Removing widget: %s", widget)
                widget.pack_forget()
                widget.destroy()
                del widget

        for var in list(self._vars):
            logger.debug("Deleting tk variable: %s", var)
            del self._vars[var]


class _PreviewCanvas(tk.Canvas):  # pylint:disable=too-many-ancestors
    """ The canvas that holds the preview image

    Parameters
    ----------
    parent: :class:`tkinter.Frame`
        The parent frame that will hold the Canvas and taskbar
    scale_var: :class:`tkinter.StringVar`
        The variable that holds the value from the scale combo box
    screen_dimensions: tuple
        The (`width`, `height`) of the displaying monitor
    is_standalone: bool
        ``True`` if the preview is standalone, ``False`` if it is in the GUI
    """
    def __init__(self,
                 parent: tk.Frame,
                 scale_var: tk.StringVar,
                 screen_dimensions: tuple[int, int],
                 is_standalone: bool) -> None:
        logger.debug("Initializing %s (parent: '%s', scale_var: %s, screen_dimensions: %s)",
                     self.__class__.__name__, parent, scale_var, screen_dimensions)
        frame = tk.Frame(parent)
        super().__init__(frame)

        self._is_standalone = is_standalone
        self._screen_dimensions = screen_dimensions
        self._var_scale = scale_var
        self._configure_scrollbars(frame)
        self._image: ImageTk.PhotoImage | None = None
        self._image_id = self.create_image(self.width / 2,
                                           self.height / 2,
                                           anchor=tk.CENTER,
                                           image=self._image)
        self.pack(fill=tk.BOTH, expand=True)
        self.bind("<Configure>", self._resize)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        logger.debug("Initialized %s ('%s')", self.__class__.__name__, self)

    @property
    def image_id(self) -> int:
        """ int: The ID of the preview image item within the canvas """
        return self._image_id

    @property
    def width(self) -> int:
        """int: The pixel width of canvas"""
        return self.winfo_width()

    @property
    def height(self) -> int:
        """int: The pixel width of the canvas"""
        return self.winfo_height()

    def _configure_scrollbars(self, frame: tk.Frame) -> None:
        """ Add X and Y scrollbars to the frame and set to scroll the canvas.

        Parameters
        ----------
        frame: :class:`tkinter.Frame`
            The parent frame to the canvas
        """
        logger.debug("Configuring scrollbars")
        x_scrollbar = tk.Scrollbar(frame, orient="horizontal", command=self.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        y_scrollbar = tk.Scrollbar(frame, command=self.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.configure(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        logger.debug("Configured scrollbars. x: '%s', y: '%s'", x_scrollbar, y_scrollbar)

    def _resize(self, event: tk.Event) -> None:  # pylint:disable=unused-argument
        """ Place the image in center of canvas on resize event and move to top left

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The canvas resize event. Unused.
        """
        if self._var_scale.get() == "Fit":  # Trigger an update to resize image
            logger.debug("Triggering redraw for 'Fit' Scaling")
            self._var_scale.set("Fit")
            return

        self.configure(scrollregion=self.bbox("all"))
        self.update_idletasks()

        assert self._image is not None
        self._center_image(self.width / 2, self.height / 2)

        # Move to top left when resizing into screen dimensions (initial startup)
        if self.width > self._screen_dimensions[0]:
            logger.debug("Moving image to left edge")
            self.xview_moveto(0.0)
        if self.height > self._screen_dimensions[1]:
            logger.debug("Moving image to top edge")
            self.yview_moveto(0.0)

    def _center_image(self, point_x: float, point_y: float) -> None:
        """ Center the image on the canvas on a resize or image update.

        Parameters
        ----------
        point_x: int
            The x point to center on
        point_y: int
            The y point to center on
        """
        canvas_location = (self.canvasx(point_x), self.canvasy(point_y))
        logger.debug("Centering canvas for size (%s, %s). New image coordinates: %s",
                     point_x, point_y, canvas_location)
        self.coords(self.image_id, canvas_location)

    def set_image(self,
                  image: ImageTk.PhotoImage,
                  center_image: bool = False) -> None:
        """ Update the canvas with the given image and update area/scrollbars accordingly

        Parameters
        ----------
        image: :class:`ImageTK.PhotoImage`
            The preview image to display in the canvas
        bool, optional
            ``True`` if the image should be re-centered. Default ``True``
        """
        logger.debug("Setting canvas image. ID: %s, size: %s for canvas size: %s (recenter: %s)",
                     self.image_id, (image.width(), image.height()), (self.width, self.height),
                     center_image)
        self._image = image
        self.itemconfig(self.image_id, image=self._image)

        if self._is_standalone:  # canvas size should not be updated inside GUI
            self.config(width=self._image.width(), height=self._image.height())

        self.update_idletasks()
        if center_image:
            self._center_image(self.width / 2, self.height / 2)
        self.configure(scrollregion=self.bbox("all"))
        logger.debug("set canvas image. Canvas size: %s", (self.width, self.height))


class _Image():
    """ Holds the source image and the resized display image for the canvas

    Parameters
    ----------
    save_variable: :class:`tkinter.BooleanVar`
        Variable that indicates a save preview has been requested in standalone mode
    is_standalone: bool
        ``True`` if the preview is running in standalone mode. ``False`` if it is running in the
        GUI
    """
    def __init__(self, save_variable: tk.BooleanVar, is_standalone: bool) -> None:
        logger.debug("Initializing %s: (save_variable: %s, is_standalone: %s)",
                     self.__class__.__name__, save_variable, is_standalone)
        self._is_standalone = is_standalone
        self._source: np.ndarray | None = None
        self._display: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._interpolation = cv2.INTER_NEAREST

        self._save_var = save_variable
        self._save_var.trace("w", self.save_preview)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def display_image(self) -> ImageTk.PhotoImage:
        """ :class:`PIL.ImageTk.PhotoImage`: The current display image """
        assert self._display is not None
        return self._display

    @property
    def source(self) -> np.ndarray:
        """ :class:`PIL.Image.Image`: The current source preview image """
        assert self._source is not None
        return self._source

    @property
    def scale(self) -> int:
        """int: The current display scale as a percentage of original image size """
        return int(self._scale * 100)

    def set_source_image(self, name: str, image: np.ndarray) -> None:
        """ Set the source image to :attr:`source`

        Parameters
        ----------
        name: str
            The name of the preview image to load
        image: :class:`numpy.ndarray`
            The image to use in RGB format
        """
        logger.debug("Setting source image. name: '%s', shape: %s", name, image.shape)
        self._source = image

    def set_display_image(self) -> None:
        """ Obtain the scaled image and set to :attr:`display_image` """
        logger.debug("Setting display image. Scale: %s", self._scale)
        image = self.source[..., 2::-1]  # TO RGB
        if self._scale not in (0.0, 1.0):  # Scale will be 0,0 on initial load in GUI
            interp = self._interpolation if self._scale > 1.0 else cv2.INTER_NEAREST
            dims = (int(round(self.source.shape[1] * self._scale, 0)),
                    int(round(self.source.shape[0] * self._scale, 0)))
            image = cv2.resize(image, dims, interpolation=interp)
        self._display = ImageTk.PhotoImage(Image.fromarray(image))
        logger.debug("Set display image. Size: %s",
                     (self._display.width(), self._display.height()))

    def set_scale(self, scale: float) -> bool:
        """ Set the display scale to the given value.

        Parameters
        ----------
        scale: float
            The value to set scaling to

        Returns
        -------
        bool
            ``True`` if the scale has been changed otherwise ``False``
        """
        if self._scale == scale:
            return False
        logger.debug("Setting scale: %s", scale)
        self._scale = scale
        return True

    def set_interpolation(self, interpolation: int) -> bool:
        """ Set the interpolation enum to the given value.

        Parameters
        ----------
        interpolation: int
            The value to set interpolation to

        Returns
        -------
        bool
            ``True`` if the interpolation has been changed otherwise ``False``
        """
        if self._interpolation == interpolation:
            return False
        logger.debug("Setting interpolation: %s")
        self._interpolation = interpolation
        return True

    def save_preview(self, *args) -> None:
        """ Save out the full size preview to the faceswap folder on a save button press

        Parameters
        ----------
        args: tuple
            Tuple containing either the key press event (Ctrl+s shortcut), the tk variable
            arguments (standalone save button press) or the folder location (GUI save button press)
        """
        if self._is_standalone and not self._save_var.get() and not isinstance(args[0], tk.Event):
            return

        if self._is_standalone:
            root_path = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])))
        else:
            root_path = args[0]

        now = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        filename = os.path.join(root_path, f"preview_{now}.png")
        cv2.imwrite(filename, self.source)
        print("")
        logger.info("Saved preview to: '%s'", filename)

        if self._is_standalone:
            self._save_var.set(False)


class _Bindings():  # pylint:disable=too-few-public-methods
    """ Handle Mouse and Keyboard bindings for the canvas.

    Parameters
    ----------
    canvas: :class:`_PreviewCanvas`
        The canvas that holds the preview image
    taskbar: :class:`_Taskbar`
        The taskbar widget which holds the scaling variables
    image: :class:`_Image`
        The object which holds the source and display version of the preview image
    is_standalone: bool
        ``True`` if the preview is standalone, ``False`` if it is embedded in the GUI
    """
    def __init__(self,
                 canvas: _PreviewCanvas,
                 taskbar: _Taskbar,
                 image: _Image,
                 is_standalone: bool) -> None:
        logger.debug("Initializing %s (canvas: '%s', taskbar: '%s', image: '%s')",
                     self.__class__.__name__, canvas, taskbar, image)
        self._canvas = canvas
        self._taskbar = taskbar
        self._image = image

        self._drag_data: list[float] = [0., 0.]
        self._set_mouse_bindings()
        self._set_key_bindings(is_standalone)
        logger.debug("Initialized %s", self.__class__.__name__,)

    def _on_bound_zoom(self, event: tk.Event) -> None:
        """ Action to perform on a valid zoom key press or mouse wheel action

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The key press or mouse wheel event
        """
        if event.keysym in ("KP_Add", "plus") or event.num == 4 or event.delta > 0:
            scale = min(self._taskbar.max_scale, self._image.scale + 25)
        else:
            scale = max(self._taskbar.min_scale, self._image.scale - 25)
        logger.trace("Bound zoom action: (event: %s, scale: %s)", event, scale)  # type: ignore
        self._taskbar.scale_var.set(f"{scale}%")

    def _on_mouse_click(self, event: tk.Event) -> None:
        """ log initial click coordinates for mouse click + drag action

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The mouse event
        """
        self._drag_data = [event.x / self._image.display_image.width(),
                           event.y / self._image.display_image.height()]
        logger.trace("Mouse click action: (event: %s, drag_data: %s)",  # type: ignore
                     event, self._drag_data)

    def _on_mouse_drag(self, event: tk.Event) -> None:
        """ Drag image left, right, up or down

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The mouse event
        """
        location_x = event.x / self._image.display_image.width()
        location_y = event.y / self._image.display_image.height()

        if self._canvas.xview() != (0.0, 1.0):
            to_x = min(1.0, max(0.0, self._drag_data[0] - location_x + self._canvas.xview()[0]))
            self._canvas.xview_moveto(to_x)
        if self._canvas.yview() != (0.0, 1.0):
            to_y = min(1.0, max(0.0, self._drag_data[1] - location_y + self._canvas.yview()[0]))
            self._canvas.yview_moveto(to_y)

        self._drag_data = [location_x, location_y]

    def _on_key_move(self, event: tk.Event) -> None:
        """ Action to perform on a valid move key press

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The key press event
        """
        move_axis = self._canvas.xview if event.keysym in ("Left", "Right") else self._canvas.yview
        visible = move_axis()[1] - move_axis()[0]
        amount = -visible / 25 if event.keysym in ("Up", "Left") else visible / 25
        logger.trace("Key move event: (event: %s, move_axis: %s, visible: %s, "  # type: ignore
                     "amount: %s)", move_axis, visible, amount)
        move_axis(tk.MOVETO, min(1.0, max(0.0, move_axis()[0] + amount)))

    def _set_mouse_bindings(self) -> None:
        """ Set the mouse bindings for interacting with the preview image

        Mousewheel: Zoom in and out
        Mouse click: Move image
        """
        logger.debug("Binding mouse events")
        if system() == "Linux":
            self._canvas.tag_bind(self._canvas.image_id, "<Button-4>", self._on_bound_zoom)
            self._canvas.tag_bind(self._canvas.image_id, "<Button-5>", self._on_bound_zoom)
        else:
            self._canvas.bind("<MouseWheel>", self._on_bound_zoom)

        self._canvas.tag_bind(self._canvas.image_id, "<Button-1>", self._on_mouse_click)
        self._canvas.tag_bind(self._canvas.image_id, "<B1-Motion>", self._on_mouse_drag)
        logger.debug("Bound mouse events")

    def _set_key_bindings(self, is_standalone: bool) -> None:
        """ Set the keyboard bindings.

        Up/Down/Left/Right: Moves image
        +/-: Zooms image
        ctrl+s: Save
        i: Cycle interpolators

        Parameters
        ----------
        ``True`` if the preview is standalone, ``False`` if it is embedded in the GUI
        """
        if not is_standalone:
            # Don't bind keys for GUI as it adds complication
            return
        logger.debug("Binding key events")
        root = self._canvas.winfo_toplevel()
        for key in ("Left", "Right", "Up", "Down"):
            root.bind(f"<{key}>", self._on_key_move)
        for key in ("Key-plus", "Key-minus", "Key-KP_Add", "Key-KP_Subtract"):
            root.bind(f"<{key}>", self._on_bound_zoom)
        root.bind("<Control-s>", self._image.save_preview)
        root.bind("<i>", self._taskbar.cycle_interpolators)
        logger.debug("Bound key events")


class PreviewTk(PreviewBase):
    """ Holds a preview window for displaying the pop out preview.

    Parameters
    ----------
    preview_buffer: :class:`PreviewBuffer`
        The thread safe object holding the preview images
    parent: tkinter widget, optional
        If this viewer is being called from the GUI the parent widget should be passed in here.
        If this is a standalone pop-up window then pass ``None``. Default: ``None``
    taskbar: :class:`tkinter.ttk.Frame`, optional
        If this viewer is being called from the GUI the parent's option frame should be passed in
        here. If this is a standalone pop-up window then pass ``None``. Default: ``None``
    triggers: dict, optional
        Dictionary of event triggers for pop-up preview. Not required when running inside the GUI.
        Default: `None`
    """
    def __init__(self,
                 preview_buffer: PreviewBuffer,
                 parent: tk.Widget | None = None,
                 taskbar: ttk.Frame | None = None,
                 triggers: TriggerType | None = None) -> None:
        logger.debug("Initializing %s (parent: '%s')", self.__class__.__name__, parent)
        super().__init__(preview_buffer, triggers=triggers)
        self._is_standalone = parent is None
        self._initialized = False
        self._root = parent if parent is not None else tk.Tk()
        self._master_frame = tk.Frame(self._root)

        self._taskbar = _Taskbar(self._master_frame, taskbar)

        self._screen_dimensions = self._get_geometry()
        self._canvas = _PreviewCanvas(self._master_frame,
                                      self._taskbar.scale_var,
                                      self._screen_dimensions,
                                      self._is_standalone)

        self._image = _Image(self._taskbar.save_var, self._is_standalone)

        _Bindings(self._canvas, self._taskbar, self._image, self._is_standalone)

        self._taskbar.scale_var.trace("w", self._set_scale)
        self._taskbar.interpolator_var.trace("w", self._set_interpolation)

        self._process_triggers()

        if self._is_standalone:
            self.pack(fill=tk.BOTH, expand=True)

        self._output_helptext()

        logger.debug("Initialized %s", self.__class__.__name__)

        self._launch()

    @property
    def master_frame(self) -> tk.Frame:
        """ :class:`tkinter.Frame`: The master frame that holds the preview window """
        return self._master_frame

    def pack(self, *args, **kwargs):
        """ Redirect calls to pack the widget to pack the actual :attr:`_master_frame`.

        Takes standard :class:`tkinter.Frame` pack arguments
        """
        logger.debug("Packing master frame: (args: %s, kwargs: %s)", args, kwargs)
        self._master_frame.pack(*args, **kwargs)

    def save(self, location: str) -> None:
        """ Save action to be performed when save button pressed from the GUI.

        location: str
            Full path to the folder to save the preview image to
        """
        self._image.save_preview(location)

    def remove_option_controls(self) -> None:
        """ Remove the taskbar options controls when the preview is disabled in the GUI """
        self._taskbar.destroy_widgets()

    def _output_helptext(self) -> None:
        """ Output the keybindings to Console. """
        if not self._is_standalone:
            return
        logger.info("---------------------------------------------------")
        logger.info("  Preview key bindings:")
        logger.info("    Zoom:              +/-")
        logger.info("    Toggle Zoom Mode:  i")
        logger.info("    Move:              arrow keys")
        logger.info("    Save Preview:      Ctrl+s")
        logger.info("---------------------------------------------------")

    def _get_geometry(self) -> tuple[int, int]:
        """ Obtain the geometry of the current screen (standalone) or the dimensions of the widget
        holding the preview window (GUI).

        Just pulling screen width and height does not account for multiple monitors, so dummy in a
        window to pull actual dimensions before hiding it again.

        Returns
        -------
        Tuple
            The (`width`, `height`) of the current monitor's display
        """
        if not self._is_standalone:
            root = self._root.winfo_toplevel()  # Get dims of whole GUI
            retval = root.winfo_width(), root.winfo_height()
            logger.debug("Obtained frame geometry: %s", retval)
            return retval

        assert isinstance(self._root, tk.Tk)
        logger.debug("Obtaining screen geometry")
        self._root.update_idletasks()
        self._root.attributes("-fullscreen", True)
        self._root.state("iconic")
        retval = self._root.winfo_width(), self._root.winfo_height()
        self._root.attributes("-fullscreen", False)
        self._root.state("withdraw")
        logger.debug("Obtained screen geometry: %s", retval)
        return retval

    def _set_min_max_scales(self) -> None:
        """ Set the minimum and maximum area that we allow to scale image to. """
        logger.debug("Calculating minimum scale for screen dimensions %s", self._screen_dimensions)
        half_screen = tuple(x // 2 for x in self._screen_dimensions)
        min_scales = (half_screen[0] / self._image.source.shape[1],
                      half_screen[1] / self._image.source.shape[0])
        min_scale = min(1.0, *min_scales)
        min_scale = (ceil(min_scale * 10)) * 10

        eight_screen = tuple(x * 8 for x in self._screen_dimensions)
        max_scales = (eight_screen[0] / self._image.source.shape[1],
                      eight_screen[1] / self._image.source.shape[0])
        max_scale = min(8.0, max(1.0, min(max_scales)))
        max_scale = (floor(max_scale * 10)) * 10

        logger.debug("Calculated minimum scale: %s, maximum_scale: %s", min_scale, max_scale)
        self._taskbar.set_min_max_scale(min_scale, max_scale)

    def _initialize_window(self) -> None:
        """ Initialize the window to fit into the current screen """
        logger.debug("Initializing window")
        assert isinstance(self._root, tk.Tk)
        width = min(self._master_frame.winfo_reqwidth(), self._screen_dimensions[0])
        height = min(self._master_frame.winfo_reqheight(), self._screen_dimensions[1])
        self._set_min_max_scales()
        self._root.state("normal")
        self._root.geometry(f"{width}x{height}")
        self._root.protocol("WM_DELETE_WINDOW", lambda: None)  # Intercept close window
        self._initialized = True
        logger.debug("Initialized window: (width: %s, height: %s)", width, height)

    def _update_image(self, center_image: bool = False) -> None:
        """ Update the image displayed in the canvas and set the canvas size and scroll region
        accordingly

        center_image: bool = ``True``
            ``True`` if the image in the canvas should be recentered. Defaul:``True``
        """
        logger.debug("Updating image (center_image: %s)", center_image)
        self._image.set_display_image()
        self._canvas.set_image(self._image.display_image, center_image)
        logger.debug("Updated image")

    def _convert_fit_scale(self) -> str:
        """ Convert "Fit" scale to the actual scaling amount

        Returns
        -------
        str
            The fit scaling in '##%' format
         """
        logger.debug("Converting 'Fit' scaling")
        width_scale = self._canvas.width / self._image.source.shape[1]
        height_scale = self._canvas.height / self._image.source.shape[0]
        scale = min(width_scale, height_scale) * 100
        retval = f"{floor(scale)}%"
        logger.debug("Converted 'Fit' scaling: (width_scale: %s, height_scale: %s, scale: %s, "
                     "retval: '%s'", width_scale, height_scale, scale, retval)
        return retval

    def _set_scale(self, *args) -> None:  # pylint:disable=unused-argument
        """ Update the image on a scale request """
        txtscale = self._taskbar.scale_var.get()
        logger.debug("Setting scale: '%s'", txtscale)
        txtscale = self._convert_fit_scale() if txtscale == "Fit" else txtscale
        scale = int(txtscale[:-1])  # Strip percentage and convert to int
        logger.debug("Got scale: %s", scale)

        if self._image.set_scale(scale / 100):
            logger.debug("Updating for new scale")
            self._taskbar.slider_var.set(scale)
            self._update_image(center_image=True)

    def _set_interpolation(self, *args) -> None:  # pylint:disable=unused-argument
        """ Callback for when the interpolator is change"""
        interp = self._taskbar.interpolator_var.get()
        if not self._image.set_interpolation(interp) or self._image.scale <= 1.0:
            return
        self._update_image(center_image=False)

    def _process_triggers(self) -> None:
        """ Process the standard faceswap key press triggers:

        m = toggle_mask
        r = refresh
        s = save
        enter = quit
        """
        if self._triggers is None:  # Don't need triggers for GUI
            return
        logger.debug("Processing triggers")
        root = self._canvas.winfo_toplevel()
        for key in self._keymaps:
            bindkey = "Return" if key == "enter" else key
            logger.debug("Adding trigger for key: '%s'", bindkey)

            root.bind(f"<{bindkey}>", self._on_keypress)
        logger.debug("Processed triggers")

    def _on_keypress(self, event: tk.Event) -> None:
        """ Update the triggers on a keypress event for picking up by main faceswap process.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The valid preview trigger keypress
        """
        if self._triggers is None:  # Don't need triggers for GUI
            return
        keypress = "enter" if event.keysym == "Return" else event.keysym
        key = T.cast(TriggerKeysType, keypress)
        logger.debug("Processing keypress '%s'", key)
        if key == "r":
            print("")  # Let log print on different line from loss output
            logger.info("Refresh preview requested...")

        self._triggers[self._keymaps[key]].set()
        logger.debug("Processed keypress '%s'. Set event for '%s'", key, self._keymaps[key])

    def _display_preview(self) -> None:
        """ Handle the displaying of the images currently in :attr:`_preview_buffer`"""
        if self._should_shutdown:
            self._root.destroy()

        if not self._buffer.is_updated:
            self._root.after(1000, self._display_preview)
            return

        for name, image in self._buffer.get_images():
            logger.debug("Updating image: (name: '%s', shape: %s)", name, image.shape)
            if self._is_standalone and not self._title:
                assert isinstance(self._root, tk.Tk)
                self._title = name
                logger.debug("Setting title: '%s;", self._title)
                self._root.title(self._title)
            self._image.set_source_image(name, image)
            self._update_image(center_image=not self._initialized)

        self._root.after(1000, self._display_preview)

        if not self._initialized and self._is_standalone:
            self._initialize_window()
            self._root.mainloop()
        if not self._initialized:  # Set initialized to True for GUI
            self._set_min_max_scales()
            self._taskbar.scale_var.set("Fit")
            self._initialized = True


def main():
    """ Load image from first given argument and display

    python -m lib.training.preview_tk <filename>
    """
    from lib.logger import log_setup  # pylint:disable=import-outside-toplevel
    from .preview_cv import PreviewBuffer  # pylint:disable=import-outside-toplevel
    log_setup("DEBUG", "faceswap_preview.log", "Test", False)

    img = cv2.imread(sys.argv[-1], cv2.IMREAD_UNCHANGED)
    buff = PreviewBuffer()  # pylint:disable=used-before-assignment
    buff.add_image("test_image", img)
    PreviewTk(buff)


if __name__ == "__main__":
    main()
