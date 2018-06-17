#!/usr/bin python3
""" Tooltip. Pops up help messages for the GUI """
import platform
import tkinter as tk


class Tooltip:
    """
    Create a tooltip for a given widget as the mouse goes on it.

    Adapted from StackOverflow:

    http://stackoverflow.com/questions/3221956/
           what-is-the-simplest-way-to-make-tooltips-
           in-tkinter/36221216#36221216

    http://www.daniweb.com/programming/software-development/
           code/484591/a-tooltip-class-for-tkinter

    - Originally written by vegaseat on 2014.09.09.

    - Modified to include a delay time by Victor Zaccardo on 2016.03.25.

    - Modified
        - to correct extreme right and extreme bottom behavior,
        - to stay inside the screen whenever the tooltip might go out on
          the top but still the screen is higher than the tooltip,
        - to use the more flexible mouse positioning,
        - to add customizable background color, padding, waittime and
          wraplength on creation
      by Alberto Vassena on 2016.11.05.

      Tested on Ubuntu 16.04/16.10, running Python 3.5.2

    """

    def __init__(self, widget,
                 *,
                 background="#FFFFEA",
                 pad=(5, 3, 5, 3),
                 text="widget info",
                 waittime=400,
                 wraplength=250):

        self.waittime = waittime  # in miliseconds, originally 500
        self.wraplength = wraplength  # in pixels, originally 180
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)
        self.background = background
        self.pad = pad
        self.ident = None
        self.topwidget = None

    def on_enter(self, event=None):
        """ Schedule on an enter event """
        self.schedule()

    def on_leave(self, event=None):
        """ Unschedule on a leave event """
        self.unschedule()
        self.hide()

    def schedule(self):
        """ Show the tooltip after wait period """
        self.unschedule()
        self.ident = self.widget.after(self.waittime, self.show)

    def unschedule(self):
        """ Hide the tooltip """
        id_ = self.ident
        self.ident = None
        if id_:
            self.widget.after_cancel(id_)

    def show(self):
        """ Show the tooltip """
        def tip_pos_calculator(widget, label,
                               *,
                               tip_delta=(10, 5), pad=(5, 3, 5, 3)):
            """ Calculate the tooltip position """

            s_width, s_height = widget.winfo_screenwidth(), widget.winfo_screenheight()

            width, height = (pad[0] + label.winfo_reqwidth() + pad[2],
                             pad[1] + label.winfo_reqheight() + pad[3])

            mouse_x, mouse_y = widget.winfo_pointerxy()

            x_1, y_1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
            x_2, y_2 = x_1 + width, y_1 + height

            x_delta = x_2 - s_width
            if x_delta < 0:
                x_delta = 0
            y_delta = y_2 - s_height
            if y_delta < 0:
                y_delta = 0

            offscreen = (x_delta, y_delta) != (0, 0)

            if offscreen:

                if x_delta:
                    x_1 = mouse_x - tip_delta[0] - width

                if y_delta:
                    y_1 = mouse_y - tip_delta[1] - height

            offscreen_again = y_1 < 0  # out on the top

            if offscreen_again:
                # No further checks will be done.

                # TIP:
                # A further mod might automagically augment the
                # wraplength when the tooltip is too high to be
                # kept inside the screen.
                y_1 = 0

            return x_1, y_1

        background = self.background
        pad = self.pad
        widget = self.widget

        # creates a toplevel window
        self.topwidget = tk.Toplevel(widget)
        if platform.system() == "Darwin":
            # For Mac OS
            self.topwidget.tk.call("::tk::unsupported::MacWindowStyle",
                                   "style", self.topwidget._w,
                                   "help", "none")

        # Leaves only the label and removes the app window
        self.topwidget.wm_overrideredirect(True)

        win = tk.Frame(self.topwidget,
                       background=background,
                       borderwidth=0)
        label = tk.Label(win,
                         text=self.text,
                         justify=tk.LEFT,
                         background=background,
                         relief=tk.SOLID,
                         borderwidth=0,
                         wraplength=self.wraplength)

        label.grid(padx=(pad[0], pad[2]),
                   pady=(pad[1], pad[3]),
                   sticky=tk.NSEW)
        win.grid()

        xpos, ypos = tip_pos_calculator(widget, label)

        self.topwidget.wm_geometry("+%d+%d" % (xpos, ypos))

    def hide(self):
        """ Hide the tooltip """
        topwidget = self.topwidget
        if topwidget:
            topwidget.destroy()
        self.topwidget = None
