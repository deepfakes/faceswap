#!/usr/bin python3
""" Utility functions for the GUI

    Items that are needed across multiple sections are stored here
    for easy reference """
import os

from lib.Serializer import JSONSerializer
from .faceswap_loader import FaceswapControl

# An error will be thrown when importing tkinter for users without tkinter
# distribution packages or without an X-Console. This error is handled in
# gui.py but import errors still need to be captured here
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None

class Utils(object):
    """ Inter-class object holding items that are required across classes """

    def __init__(self, options, pathscript, calling_file="faceswap.py"):
        self.opts = options
        self.pathscript = pathscript

        self.icons = dict()
        self.guitext = dict()
        self.actionbtns = dict()

        self.console = None
        self.debugconsole = False
        self.progress = dict()

        self.serializer = JSONSerializer
        self.filetypes = (('Faceswap files', '*.fsw'), ('All files', '*.*'))

        self.task = FaceswapControl(self, pathscript, calling_file=calling_file)
        self.runningtask = False

        self.previewloc = os.path.join(pathscript, 'lib', 'gui', 'cache', '.gui_preview.png')

        self.lossdict = dict()

    def init_tk(self):
        """ TK System must be on prior to setting tk variables,
        so initialised from GUI """
        pathicons = os.path.join(self.pathscript, 'lib', 'gui', 'icons')
        self.icons['folder'] = tk.PhotoImage(
            file=os.path.join(pathicons, 'open_folder.png'))
        self.icons['load'] = tk.PhotoImage(
            file=os.path.join(pathicons, 'open_file.png'))
        self.icons['save'] = tk.PhotoImage(
            file=os.path.join(pathicons, 'save.png'))
        self.icons['reset'] = tk.PhotoImage(
            file=os.path.join(pathicons, 'reset.png'))
        self.icons['clear'] = tk.PhotoImage(
            file=os.path.join(pathicons, 'clear.png'))

        self.guitext['help'] = tk.StringVar()
        self.guitext['status'] = tk.StringVar()

        self.progress['message'] = tk.StringVar()
        self.progress['position'] = tk.IntVar()
        self.progress['bar'] = None

    def set_progress_bar_type(self, mode):
        """ Set the progress bar mode """
        self.progress['bar'].config(mode=mode)
        if mode == 'indeterminate':
            self.progress['bar'].config(maximum=100)
            self.progress['bar'].start()
        else:
            self.progress['bar'].stop()
            self.progress['bar'].config(maximum=1000)

    def update_progress(self, message, position, update_position=True):
        """ Update the GUIs progress bar and position """
        self.progress['message'].set(message)
        if update_position:
            self.progress['position'].set(position)

    def action_command(self, command):
        """ The action to perform when the action button is pressed """
        if self.runningtask:
            self.action_terminate()
        else:
            self.action_execute(command)

    def action_execute(self, command):
        """ Execute the task in Faceswap.py """
        self.clear_console()
        self.task.prepare(self.opts, command)
        self.task.execute_script()

    def action_terminate(self):
        """ Terminate the subprocess Faceswap.py task """
        self.task.terminate()
        self.runningtask = False
        self.progress['message'].set('')
        self.progress['position'].set(0)
        self.set_progress_bar_type('determinate')
        self.clear_display_panel()
        self.change_action_button()

    def clear_display_panel(self):
        ''' Clear the preview window and graph '''
        self.delete_preview()
        self.lossdict = dict()

    def change_action_button(self):
        """ Change the action button to relevant control """
        for cmd in self.actionbtns.keys():
            btnact = self.actionbtns[cmd]
            if self.runningtask:
                ttl = 'Terminate'
                hlp = 'Exit the running process'
            else:
                ttl = cmd.title()
                hlp = 'Run the {} script'.format(cmd.title())
            btnact.config(text=ttl)
            Tooltip(btnact, text=hlp, wraplength=200)

    def clear_console(self):
        """ Clear the console output screen """
        self.console.delete(1.0, tk.END)

    def load_config(self, command=None):
        """ Load a saved config file """
        cfgfile = filedialog.askopenfile(mode='r', filetypes=self.filetypes)
        if not cfgfile:
            return
        cfg = self.serializer.unmarshal(cfgfile.read())
        if command is None:
            for cmd, opts in cfg.items():
                self.set_command_args(cmd, opts)
        else:
            opts = cfg.get(command, None)
            if opts:
                self.set_command_args(command, opts)
            else:
                self.clear_console()
                print('No ' + command + ' section found in file')

    def set_command_args(self, command, options):
        """ Pass the saved config items back to the GUI """
        for srcopt, srcval in options.items():
            for dstopts in self.opts[command]:
                if dstopts['control_title'] == srcopt:
                    dstopts['value'].set(srcval)
                    break

    def save_config(self, command=None):
        """ Save the current GUI state to a config file in json format """
        cfgfile = filedialog.asksaveasfile(mode='w',
                                           filetypes=self.filetypes,
                                           defaultextension='.fsw')
        if not cfgfile:
            return
        if command is None:
            cfg = {cmd: {opt['control_title']: opt['value'].get() for opt in opts}
                   for cmd, opts in self.opts.items()}
        else:
            cfg = {command: {opt['control_title']: opt['value'].get()
                             for opt in self.opts[command]}}
        cfgfile.write(self.serializer.marshal(cfg))
        cfgfile.close()

    def reset_config(self, command=None):
        """ Reset the GUI to the default values """
        if command is None:
            options = [opt for opts in self.opts.values() for opt in opts]
        else:
            options = [opt for opt in self.opts[command]]
        for option in options:
            default = option.get('default', '')
            default = '' if default is None else default
            option['value'].set(default)

    def clear_config(self, command=None):
        """ Clear all values from the GUI """
        if command is None:
            options = [opt for opts in self.opts.values() for opt in opts]
        else:
            options = [opt for opt in self.opts[command]]
        for option in options:
            if isinstance(option['value'].get(), bool):
                option['value'].set(False)
            elif isinstance(option['value'].get(), int):
                option['value'].set(0)
            else:
                option['value'].set('')

    def delete_preview(self):
        """ Delete the preview file """
        if os.path.exists(self.previewloc):
            os.remove(self.previewloc)

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
                 background='#FFFFEA',
                 pad=(5, 3, 5, 3),
                 text='widget info',
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
