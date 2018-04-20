#!/usr/bin python3
""" The optional GUI for faceswap """

import os
import signal
import re
import subprocess
from subprocess import PIPE, Popen, TimeoutExpired
import sys

from argparse import SUPPRESS
from math import ceil, floor
from threading import Thread
from time import time

import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from lib.cli import FullPaths, ComboFullPaths, DirFullPaths, FileFullPaths
from lib.Serializer import JSONSerializer

PATHSCRIPT = os.path.realpath(os.path.dirname(sys.argv[0]))

# An error will be thrown when importing tkinter for users without tkinter
# distribution packages or without an X-Console. Therefore if importing fails
# no attempt will be made to instantiate the gui.
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog
    from tkinter import messagebox
    from tkinter import TclError
except ImportError:
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    TclError = None


class Utils(object):
    """ Inter-class object holding items that are required across classes """

    def __init__(self, options, calling_file="faceswap.py"):
        self.opts = options

        self.icons = dict()
        self.guitext = dict()
        self.actionbtns = dict()

        self.console = None
        self.debugconsole = False

        self.serializer = JSONSerializer
        self.filetypes = (('Faceswap files', '*.fsw'), ('All files', '*.*'))

        self.task = FaceswapControl(self, calling_file=calling_file)
        self.runningtask = False

        self.previewloc = os.path.join(PATHSCRIPT, '.gui_preview.png')

        self.lossdict = dict()

    def init_tk(self):
        """ TK System must be on prior to setting tk variables,
        so initialised from GUI """
        pathicons = os.path.join(PATHSCRIPT, 'icons')
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
        self.clear_display_panel()
        self.change_action_button()

    def clear_display_panel(self):
        """ Clear the preview window and graph """
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

    def get_chosen_action(self, task_name):
        return self.opts[task_name][0]['value'].get()


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


class FaceswapGui(object):
    """ The Graphical User Interface """

    def __init__(self, utils, calling_file='faceswap.py'):
        self.gui = tk.Tk()
        self.utils = utils
        self.calling_file = calling_file
        self.utils.delete_preview()
        self.utils.init_tk()
        self.gui.protocol('WM_DELETE_WINDOW', self.close_app)

    def build_gui(self):
        """ Build the GUI """
        self.gui.title(self.calling_file)
        self.menu()

        container = tk.PanedWindow(self.gui,
                                   sashrelief=tk.RAISED,
                                   orient=tk.VERTICAL)
        container.pack(fill=tk.BOTH, expand=True)

        topcontainer = tk.PanedWindow(container,
                                      sashrelief=tk.RAISED,
                                      orient=tk.HORIZONTAL)
        container.add(topcontainer)

        bottomcontainer = ttk.Frame(container, height=150)
        container.add(bottomcontainer)

        optsnotebook = ttk.Notebook(topcontainer, width=400, height=500)
        topcontainer.add(optsnotebook)

        if self.calling_file == 'faceswap.py':
            # Commands explicitly stated to ensure consistent ordering
            cmdlist = ('extract', 'train', 'convert')
        else:
            cmdlist = self.utils.opts.keys()

        for command in cmdlist:
            commandtab = CommandTab(self.utils, optsnotebook, command)
            commandtab.build_tab()

        dspnotebook = ttk.Notebook(topcontainer, width=780)
        topcontainer.add(dspnotebook)

        for display in ('graph', 'preview'):
            displaytab = DisplayTab(self.utils, dspnotebook, display)
            displaytab.build_tab()

        self.add_console(bottomcontainer)
        self.add_status_bar(bottomcontainer)

    def menu(self):
        """ Menu bar for loading and saving configs """
        menubar = tk.Menu(self.gui)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Load full config...',
                             command=self.utils.load_config)
        filemenu.add_command(label='Save full config...',
                             command=self.utils.save_config)
        filemenu.add_separator()
        filemenu.add_command(label='Reset all to default',
                             command=self.utils.reset_config)
        filemenu.add_command(label='Clear all',
                             command=self.utils.clear_config)
        filemenu.add_separator()
        filemenu.add_command(label='Quit', command=self.close_app)
        menubar.add_cascade(label="File", menu=filemenu)
        self.gui.config(menu=menubar)

    def add_console(self, frame):
        """ Build the output console """
        consoleframe = ttk.Frame(frame)
        consoleframe.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(2, 0),
                          fill=tk.BOTH, expand=True)
        console = ConsoleOut(consoleframe, self.utils)
        console.build_console()

    def add_status_bar(self, frame):
        """ Build the info text section page """
        statusframe = ttk.Frame(frame)
        statusframe.pack(side=tk.BOTTOM, anchor=tk.W, padx=10, pady=2,
                         fill=tk.X, expand=False)

        lbltitle = ttk.Label(statusframe, text='Status:', width=6, anchor=tk.W)
        lbltitle.pack(side=tk.LEFT, expand=False)
        self.utils.guitext['status'].set('Ready')
        lblstatus = ttk.Label(statusframe,
                              width=20,
                              textvariable=self.utils.guitext['status'],
                              anchor=tk.W)
        lblstatus.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)

    def close_app(self):
        """ Close Python. This is here because the graph animation function
        continues to
            run even when tkinter has gone away """
        confirm = messagebox.askokcancel
        confirmtxt = 'Processes are still running. Are you sure...?'
        if self.utils.runningtask and not confirm('Close', confirmtxt):
            return
        if self.utils.runningtask:
            self.utils.task.terminate()
        self.utils.delete_preview()
        self.gui.quit()
        exit()


class ConsoleOut(object):
    """ The Console out tab of the Display section """

    def __init__(self, frame, utils):
        self.frame = frame
        utils.console = tk.Text(self.frame)
        self.console = utils.console
        self.debug = utils.debugconsole

    def build_console(self):
        """ Build and place the console """
        self.console.config(width=100, height=6, bg='gray90', fg='black')
        self.console.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.frame, command=self.console.yview)
        scrollbar.pack(side=tk.LEFT, fill='y')
        self.console.configure(yscrollcommand=scrollbar.set)

        if self.debug:
            print('Console debug activated. Outputting to main terminal')
        else:
            sys.stdout = SysOutRouter(console=self.console, out_type="stdout")
            sys.stderr = SysOutRouter(console=self.console, out_type="stderr")


class SysOutRouter(object):
    """ Route stdout/stderr to the console window """

    def __init__(self, console=None, out_type=None):
        self.console = console
        self.out_type = out_type
        self.color = ("black" if out_type == "stdout" else "red")

    def write(self, string):
        """ Capture stdout/stderr """
        self.console.insert(tk.END, string, self.out_type)
        self.console.tag_config(self.out_type, foreground=self.color)
        self.console.see(tk.END)

    @staticmethod
    def flush():
        """ If flush is forced, send it to normal terminal """
        sys.__stdout__.flush()


class CommandTab(object):
    """ Tabs to hold the command options """

    def __init__(self, utils, notebook, command):
        self.utils = utils
        self.notebook = notebook
        self.page = ttk.Frame(self.notebook)
        self.command = command
        self.title = command.title()

    def build_tab(self):
        """ Build the tab """
        actionframe = ActionFrame(self.utils, self.page, self.command)
        actionframe.build_frame()

        self.add_frame_separator()
        opts_frame = OptionsFrame(self.utils, self.page, self.command)
        opts_frame.build_frame()

        self.notebook.add(self.page, text=self.title)

    def add_frame_separator(self):
        """ Add a separator between left and right frames """
        sep = ttk.Frame(self.page, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)


class OptionsFrame(object):
    """ Options Frame - Holds the Options for each command """

    def __init__(self, utils, page, command):
        self.utils = utils
        self.page = page
        self.command = command

        self.canvas = tk.Canvas(self.page, bd=0, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.optsframe = tk.Frame(self.canvas)
        self.optscanvas = self.canvas.create_window((0, 0), window=self.optsframe, anchor=tk.NW)

    def build_frame(self):
        """ Build the options frame for this command """
        self.add_scrollbar()
        self.canvas.bind('<Configure>', self.resize_frame)

        for option in self.utils.opts[self.command]:
            optioncontrol = OptionControl(self.utils, option, self.optsframe)
            optioncontrol.build_full_control()

    def add_scrollbar(self):
        """ Add a scrollbar to the options frame """
        scrollbar = ttk.Scrollbar(self.page, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.config(yscrollcommand=scrollbar.set)
        self.optsframe.bind("<Configure>", self.update_scrollbar)

    def update_scrollbar(self, event):
        """ Update the options frame scrollbar """
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def resize_frame(self, event):
        """ Resize the options frame to fit the canvas """
        canvas_width = event.width
        self.canvas.itemconfig(self.optscanvas, width=canvas_width)


class OptionControl(object):
    """ Build the correct control for the option parsed and place it on the
    frame """

    def __init__(self, utils, option, option_frame):
        self.utils = utils
        self.option = option
        self.option_frame = option_frame

    def build_full_control(self):
        """ Build the correct control type for the option passed through """
        ctl = self.option['control']
        ctltitle = self.option['control_title']
        sysbrowser = self.option['filesystem_browser']
        ctlhelp = ' '.join(self.option.get('help', '').split())
        ctlhelp = '. '.join(i.capitalize() for i in ctlhelp.split('. '))
        ctlhelp = ctltitle + ' - ' + ctlhelp
        ctlframe = self.build_one_control_frame()
        dflt = self.option.get('default', '')
        dflt = self.option.get('default', False) if ctl == ttk.Checkbutton else dflt
        choices = self.option['choices'] if ctl == ttk.Combobox else None

        self.build_one_control_label(ctlframe, ctltitle)
        self.option['value'] = self.build_one_control(ctlframe,
                                                      ctl,
                                                      dflt,
                                                      ctlhelp,
                                                      choices,
                                                      sysbrowser)

    def build_one_control_frame(self):
        """ Build the frame to hold the control """
        frame = ttk.Frame(self.option_frame)
        frame.pack(fill=tk.X, expand=True)
        return frame

    @staticmethod
    def build_one_control_label(frame, control_title):
        """ Build and place the control label """
        lbl = ttk.Label(frame, text=control_title, width=18, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

    def build_one_control(self, frame, control, default, helptext, choices,
                          sysbrowser):
        """ Build and place the option controls """
        default = default if default is not None else ''

        var = tk.BooleanVar(
            frame) if control == ttk.Checkbutton else tk.StringVar(frame)
        var.set(default)

        if sysbrowser is not None:
            # if sysbrowser in "load file":
            self.add_browser_buttons(frame, sysbrowser, var)
            # elif sysbrowser == "combo":
            #    self.add_browser_combo_button(frame, sysbrowser, var)

        ctlkwargs = {'variable': var} if control == ttk.Checkbutton else {
            'textvariable': var}
        packkwargs = {'anchor': tk.W} if control == ttk.Checkbutton else {
            'fill': tk.X, 'expand': True}
        ctl = control(frame, **ctlkwargs)

        if control == ttk.Combobox:
            ctl['values'] = [choice for choice in choices]

        ctl.pack(padx=5, pady=5, **packkwargs)
        Tooltip(ctl, text=helptext, wraplength=200)
        return var

    def add_browser_buttons(self, frame, sysbrowser, filepath):
        """ Add correct file browser button for control """
        if sysbrowser == "combo":
            img = self.utils.icons['load']
        else:
            img = self.utils.icons[sysbrowser]
        action = getattr(self, 'ask_' + sysbrowser)
        filetypes = self.option['filetypes']
        fileopn = ttk.Button(frame, image=img,
                             command=lambda cmd=action: cmd(filepath,
                                                            filetypes))
        fileopn.pack(padx=(0, 5), side=tk.RIGHT)

    @staticmethod
    def ask_folder(filepath, filetypes=None):
        """
        Pop-up to get path to a directory
        :param filepath: tkinter StringVar object that will store the path to a
        directory.
        :param filetypes: Unused argument to allow filetypes to be given in
        ask_load().
        """
        dirname = filedialog.askdirectory()
        if dirname:
            filepath.set(dirname)

    @staticmethod
    def ask_load(filepath, filetypes=None):
        """ Pop-up to get path to a file """
        if filetypes is None:
            filename = filedialog.askopenfilename()
        else:
            # In case filetypes were not configured properly in the
            # arguments_list
            try:
                filename = filedialog.askopenfilename(filetypes=filetypes)
            except TclError as te1:
                filetypes = FileFullPaths.prep_filetypes(filetypes)
                filename = filedialog.askopenfilename(filetypes=filetypes)
            except TclError as te2:
                filename = filedialog.askopenfilename()
        if filename:
            filepath.set(filename)

    def ask_combo(self, filepath, filetypes):
        actions_open_type = self.option['actions_open_type']
        task_name = actions_open_type['task_name']
        chosen_action = self.utils.get_chosen_action(task_name)
        action = getattr(self, "ask_" + actions_open_type[chosen_action])
        filetypes = filetypes[chosen_action]
        action(filepath, filetypes)


class ActionFrame(object):
    """Action Frame - Displays information and action controls """

    def __init__(self, utils, page, command):
        self.utils = utils
        self.page = page
        self.command = command
        self.title = command.title()

    def build_frame(self):
        """ Add help display and Action buttons to the left frame of each
        page """
        frame = ttk.Frame(self.page)
        frame.pack(fill=tk.BOTH, padx=(10, 5), side=tk.BOTTOM, anchor=tk.N)

        self.add_action_button(frame)
        self.add_util_buttons(frame)

    def add_action_button(self, frame):
        """ Add the action buttons for page """
        actframe = ttk.Frame(frame)
        actframe.pack(fill=tk.X, side=tk.LEFT, padx=5, pady=5)

        btnact = ttk.Button(actframe,
                            text=self.title,
                            width=12,
                            command=lambda: self.utils.action_command(
                                self.command))
        btnact.pack(side=tk.TOP)
        Tooltip(btnact, text='Run the {} script'.format(self.title), wraplength=200)
        self.utils.actionbtns[self.command] = btnact

    def add_util_buttons(self, frame):
        """ Add the section utility buttons """
        utlframe = ttk.Frame(frame)
        utlframe.pack(side=tk.RIGHT, padx=(5, 10), pady=5)

        for utl in ('load', 'save', 'clear', 'reset'):
            img = self.utils.icons[utl]
            action = getattr(self.utils, utl + '_config')
            btnutl = ttk.Button(utlframe,
                                image=img,
                                command=lambda cmd=action: cmd(self.command))
            btnutl.pack(padx=2, side=tk.LEFT)
            Tooltip(btnutl, text=utl.capitalize() + ' ' + self.title + ' config', wraplength=200)


class DisplayTab(object):
    """ The display tabs """

    def __init__(self, utils, notebook, display):
        self.utils = utils
        self.notebook = notebook
        self.page = ttk.Frame(self.notebook)
        self.display = display
        self.title = self.display.title()

    def build_tab(self):
        """ Build the tab """
        frame = ttk.Frame(self.page)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        if self.display == 'graph':
            graphframe = GraphDisplay(frame, self.utils)
            graphframe.create_graphs()
        elif self.display == 'preview':
            preview = PreviewDisplay(frame, self.utils.previewloc)
            preview.update_preview()
        else:  # Dummy in a placeholder
            lbl = ttk.Label(frame, text=self.display, width=15, anchor=tk.NW)
            lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

        self.notebook.add(self.page, text=self.title)


class GraphDisplay(object):
    """ The Graph Tab of the Display section """

    def __init__(self, frame, utils):
        self.frame = frame
        self.utils = utils
        self.losskeys = None

        self.graphpane = tk.PanedWindow(self.frame, sashrelief=tk.RAISED, orient=tk.VERTICAL)
        self.graphpane.pack(fill=tk.BOTH, expand=True)

        self.graphs = list()

    def create_graphs(self):
        """ create the graph frames when there are loss values to graph """
        if not self.utils.lossdict:
            self.frame.after(1000, self.create_graphs)
            return

        self.losskeys = sorted([key for key in self.utils.lossdict.keys()])

        framecount = int(len(self.utils.lossdict) / 2)
        for i in range(framecount):
            self.add_graph(i)

        self.monitor_state()

    def add_graph(self, index):
        """ Add a single graph to the graph window """
        graphframe = ttk.Frame(self.graphpane)
        self.graphpane.add(graphframe)

        selectedkeys = self.losskeys[index * 2:(index + 1) * 2]
        selectedloss = {key: self.utils.lossdict[key] for key in selectedkeys}

        graph = Graph(graphframe, selectedloss, selectedkeys)
        self.graphs.append(graph)
        graph.build_graph()

    def monitor_state(self):
        """ Check there is a task still running. If not, destroy graphs
            and reset graph display to waiting state """
        if self.utils.lossdict:
            self.frame.after(5000, self.monitor_state)
            return
        self.destroy_graphs()
        self.create_graphs()

    def destroy_graphs(self):
        """ Destroy graphs when the process has stopped """
        for graph in self.graphs:
            del graph
        self.graphs = list()
        for child in self.graphpane.panes():
            self.graphpane.remove(child)


class Graph(object):
    """ Each graph to be displayed. Until training is run it is not known
        how many graphs will be required, so they sit in their own class
        ready to be created when requested """

    def __init__(self, frame, loss, losskeys):
        self.frame = frame
        self.loss = loss
        self.losskeys = losskeys

        self.ylim = (100, 0)

        style.use('ggplot')

        self.fig = plt.figure(figsize=(4, 4), dpi=75)
        self.ax1 = self.fig.add_subplot(1, 1, 1)
        self.losslines = list()
        self.trndlines = list()

    def build_graph(self):
        """ Update the plot area with loss values and cycle through to
        animate """
        self.ax1.set_xlabel('Iterations')
        self.ax1.set_ylabel('Loss')
        self.ax1.set_ylim(0.00, 0.01)
        self.ax1.set_xlim(0, 1)

        losslbls = [lbl.replace('_', ' ').title() for lbl in self.losskeys]
        for idx, linecol in enumerate(['blue', 'red']):
            self.losslines.extend(self.ax1.plot(0, 0,
                                                color=linecol,
                                                linewidth=1,
                                                label=losslbls[idx]))
        for idx, linecol in enumerate(['navy', 'firebrick']):
            lbl = losslbls[idx]
            lbl = 'Trend{}'.format(lbl[lbl.rfind(' '):])
            self.trndlines.extend(self.ax1.plot(0, 0,
                                                color=linecol,
                                                linewidth=2,
                                                label=lbl))

        self.ax1.legend(loc='upper right')

        plt.subplots_adjust(left=0.075, bottom=0.075, right=0.95, top=0.95,
                            wspace=0.2, hspace=0.2)

        plotcanvas = FigureCanvasTkAgg(self.fig, self.frame)
        plotcanvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ani = animation.FuncAnimation(self.fig, self.animate, interval=2000, blit=False)
        plotcanvas.draw()

    def animate(self, i):
        """ Read loss data and apply to graph """
        loss = [self.loss[key][:] for key in self.losskeys]

        xlim = self.recalculate_axes(loss)

        xrng = [x for x in range(xlim)]

        self.raw_plot(xrng, loss)

        if xlim > 10:
            self.trend_plot(xrng, loss)

    def recalculate_axes(self, loss):
        """ Recalculate the latest x and y axes limits from latest data """
        ymin = floor(min([min(lossvals) for lossvals in loss]) * 100) / 100
        ymax = ceil(max([max(lossvals) for lossvals in loss]) * 100) / 100

        if ymin < self.ylim[0] or ymax > self.ylim[1]:
            self.ylim = (ymin, ymax)
            self.ax1.set_ylim(self.ylim[0], self.ylim[1])

        xlim = len(loss[0])
        xlim = 2 if xlim == 1 else xlim
        self.ax1.set_xlim(0, xlim - 1)

        return xlim

    def raw_plot(self, x_range, loss):
        """ Raw value plotting """
        for idx, lossvals in enumerate(loss):
            self.losslines[idx].set_data(x_range, lossvals)

    def trend_plot(self, x_range, loss):
        """ Trend value plotting """
        for idx, lossvals in enumerate(loss):
            fit = numpy.polyfit(x_range, lossvals, 3)
            poly = numpy.poly1d(fit)
            self.trndlines[idx].set_data(x_range, poly(x_range))


class PreviewDisplay(object):
    """ The Preview tab of the Display section """

    def __init__(self, frame, previewloc):
        self.frame = frame
        self.previewimg = None
        self.errcount = 0
        self.previewloc = previewloc

        self.previewlbl = ttk.Label(self.frame, image=None, anchor=tk.NW)
        self.previewlbl.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_preview(self):
        """ Display the image if it exists or a place holder if it doesn't """
        self.load_preview()
        if self.previewimg is None:
            self.previewlbl.config(image=None)
        else:
            self.previewlbl.config(image=self.previewimg)
        self.previewlbl.after(1000, self.update_preview)

    def load_preview(self):
        """ Load the preview image into tk PhotoImage """
        if os.path.exists(self.previewloc):
            try:
                self.previewimg = tk.PhotoImage(file=self.previewloc)
                self.errcount = 0
            except TclError:
                # This is probably an error reading the file whilst it's
                # being saved
                # so ignore it for now and only pick up if there have been
                # multiple
                # consecutive fails
                if self.errcount < 10:
                    self.errcount += 1
                    self.previewimg = None
                else:
                    print('Error reading the preview file')
        else:
            self.previewimg = None


class FaceswapControl(object):
    """ Control the underlying Faceswap tasks """

    def __init__(self, utils, calling_file="faceswap.py"):
        self.pathexecscript = os.path.join(PATHSCRIPT, calling_file)
        self.utils = utils

        self.command = None
        self.args = None
        self.process = None
        self.lenloss = 0

    def prepare(self, options, command):
        """ Prepare for running the subprocess """
        self.command = command
        self.utils.runningtask = True
        self.utils.change_action_button()
        self.utils.guitext['status'].set('Executing - ' + self.command + '.py')
        print('Loading...')
        self.args = ['python', '-u', self.pathexecscript, self.command]
        self.build_args(options)

    def build_args(self, options):
        """ Build the faceswap command and arguments list """
        for item in options[self.command]:
            optval = str(item.get('value', '').get())
            opt = item['opts'][0]
            if optval == 'False' or optval == '':
                continue
            elif optval == 'True':
                if self.command == 'train' and opt == '-p':  # Embed the preview pane
                    self.args.append('-gui')
                else:
                    self.args.append(opt)
            else:
                self.args.extend((opt, optval))

    def execute_script(self):
        """ Execute the requested Faceswap Script """
        kwargs = {'stdout': PIPE,
                  'stderr': PIPE,
                  'bufsize': 1,
                  'universal_newlines': True}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = Popen(self.args, **kwargs)
        self.thread_stdout()
        self.thread_stderr()

    def read_stdout(self):
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        while True:
            output = self.process.stdout.readline()
            if output == '' and self.process.poll() is not None:
                break
            if output:
                if self.command == 'train' and str.startswith(output, '['):
                    self.capture_loss(output)
                print(output.strip())
        returncode = self.process.poll()
        self.utils.runningtask = False
        self.utils.change_action_button()
        self.set_final_status(returncode)
        print('Process exited.')

    def read_stderr(self):
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        while True:
            output = self.process.stderr.readline()
            if output == '' and self.process.poll() is not None:
                break
            print(output.strip(), file=sys.stderr)

    def thread_stdout(self):
        """ Put the subprocess stdout so that it can be read without
        blocking """
        thread = Thread(target=self.read_stdout)
        thread.daemon = True
        thread.start()

    def thread_stderr(self):
        """ Put the subprocess stderr so that it can be read without
        blocking """
        thread = Thread(target=self.read_stderr)
        thread.daemon = True
        thread.start()

    def capture_loss(self, string):
        """ Capture loss values from stdout """
        #TODO: Remove this hideous hacky fix. When the subprocess is terminated and
        # the loss dictionary is reset, 1 set of loss values ALWAYS slips through
        # and appends to the lossdict AFTER the subprocess has closed meaning that
        # checks on whether the dictionary is empty fail.
        # Therefore if the size of current loss dictionary is smaller than the
        # previous loss dictionary, assume that the process has been terminated
        # and reset it.
        # I have tried and failed to empty the subprocess stdout with:
        #   sys.exit() on the stdout/err threads (no effect)
        #   sys.stdout/stderr.flush (no effect)
        #   thread.join (locks the whole process up, because the stdout thread
        #       stubbonly refuses to release it's last line)

        currentlenloss = len(self.utils.lossdict)
        if self.lenloss > currentlenloss:
            self.utils.lossdict = dict()
            self.lenloss = 0
            return
        self.lenloss = currentlenloss

        loss = re.findall(r'([a-zA-Z_]+):.*?(\d+\.\d+)', string)

        if len(loss) < 2:
            return

        if not self.utils.lossdict:
            self.utils.lossdict.update((item[0], []) for item in loss)

        for item in loss:
            self.utils.lossdict[item[0]].append(float(item[1]))

    def terminate(self):
        """ Terminate the subprocess """
        if self.command == 'train':
            print('Sending Exit Signal', flush=True)
            try:
                now = time()
                if os.name == 'nt':
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    self.process.send_signal(signal.SIGINT)
                while True:
                    timeelapsed = time() - now
                    if self.process.poll() is not None:
                        break
                    if timeelapsed > 30:
                        raise ValueError('Timeout reached sending Exit Signal')
                return
            except ValueError as err:
                print(err)
        print('Terminating Process...')
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
            print('Terminated')
        except TimeoutExpired:
            print('Termination timed out. Killing Process...')
            self.process.kill()
            print('Killed')

    def set_final_status(self, returncode):
        """ Set the status bar output based on subprocess return code """
        if returncode == 0 or returncode == 3221225786:
            status = 'Ready'
        elif returncode == -15:
            status = 'Terminated - {}.py'.format(self.command)
        elif returncode == -9:
            status = 'Killed - {}.py'.format(self.command)
        elif returncode == -6:
            status = 'Aborted - {}.py'.format(self.command)
        else:
            status = 'Failed - {}.py. Return Code: {}'.format(self.command, returncode)
        self.utils.guitext['status'].set(status)


class TKGui(object):
    """ Main GUI Control """

    def __init__(self, subparser, subparsers, command, description='default'):
        # Don't try to load the GUI if there is no display or there are
        # problems importing tkinter
        cmd = sys.argv
        if not self.check_display(cmd) or not self.check_tkinter_available(cmd):
            return

        # If not running in gui mode return before starting to create a window
        if 'gui' not in cmd:
            return

        self.arguments = None
        self.opts = self.extract_options(subparsers)
        self.utils = Utils(self.opts, calling_file=cmd[0])
        self.root = FaceswapGui(self.utils, calling_file=cmd[0])
        self.parse_arguments(description, subparser, command)

    @staticmethod
    def check_display(command):
        """ Check whether there is a display to output the GUI. If running on
            Windows then assume not running in headless mode """
        if not os.environ.get('DISPLAY', None) and os.name != 'nt':
            if 'gui' in command:
                print('Could not detect a display. The GUI has been disabled.')
                if os.name == 'posix':
                    print('macOS users need to install XQuartz. '
                          'See https://support.apple.com/en-gb/HT201341')
            return False
        return True

    @staticmethod
    def check_tkinter_available(command):
        """ Check whether TkInter is installed on user's machine """
        tkinter_vars = [tk, ttk, filedialog, messagebox, TclError]
        if any(var is None for var in tkinter_vars):
            if "gui" in command:
                print(
                    "It looks like TkInter isn't installed for your OS, so "
                    "the GUI has been "
                    "disabled. To enable the GUI please install the TkInter "
                    "application.\n"
                    "You can try:\n"
                    "  Windows/macOS:      Install ActiveTcl Community "
                    "Edition from "
                    "www.activestate.com\n"
                    "  Ubuntu/Mint/Debian: sudo apt install python3-tk\n"
                    "  Arch:               sudo pacman -S tk\n"
                    "  CentOS/Redhat:      sudo yum install tkinter\n"
                    "  Fedora:             sudo dnf install python3-tkinter\n",
                    file=sys.stderr)
            return False
        return True

    def extract_options(self, subparsers):
        """ Extract the existing ArgParse Options """
        opts = {cmd: subparsers[cmd].argument_list + subparsers[cmd].optional_arguments
                for cmd in subparsers.keys()}
        for command in opts.values():
            for opt in command:
                if opt.get('help', '') == SUPPRESS:
                    command.remove(opt)
                ctl, sysbrowser, filetypes, actions_open_types = self.set_control(opt)
                opt['control_title'] = self.set_control_title(
                    opt.get('opts', ''))
                opt['control'] = ctl
                opt['filesystem_browser'] = sysbrowser
                opt['filetypes'] = filetypes
                opt['actions_open_types'] = actions_open_types
        return opts

    @staticmethod
    def set_control_title(opts):
        """ Take the option switch and format it nicely """
        ctltitle = opts[1] if len(opts) == 2 else opts[0]
        ctltitle = ctltitle.replace('-', ' ').replace('_', ' ').strip().title()
        return ctltitle

    @staticmethod
    def set_control(option):
        """ Set the control and filesystem browser to use for each option """
        sysbrowser = None
        filetypes = None
        actions_open_type = None
        ctl = ttk.Entry
        if option.get('action', '') == FullPaths:
            sysbrowser = 'folder'
        elif option.get('action', '') == DirFullPaths:
            sysbrowser = 'folder'
        elif option.get('action', '') == FileFullPaths:
            sysbrowser = 'load'
            filetypes = option.get('filetypes', None)
        elif option.get('action', '') == ComboFullPaths:
            sysbrowser = 'combo'
            actions_open_type = option['actions_open_type']
            filetypes = option.get('filetypes', None)
        elif option.get('choices', '') != '':
            ctl = ttk.Combobox
        elif option.get('action', '') == 'store_true':
            ctl = ttk.Checkbutton
        return ctl, sysbrowser, filetypes, actions_open_type

    def parse_arguments(self, description, subparser, command):
        """ Parse the command line arguments for the GUI """
        parser = subparser.add_parser(
            command,
            help="This Launches a GUI for Faceswap.",
            description=description,
            epilog="Questions and feedback: \
                    https://github.com/deepfakes/faceswap-playground")

        parser.add_argument('-d', '--debug',
                            action='store_true',
                            dest='debug',
                            default=False,
                            help='Output to Shell console instead of GUI console')
        parser.set_defaults(func=self.process)

    def process(self, arguments):
        """ Builds the GUI """
        self.arguments = arguments
        self.utils.debugconsole = self.arguments.debug
        self.root.build_gui()
        self.root.gui.mainloop()
