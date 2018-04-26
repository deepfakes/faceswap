#!/usr/bin python3
""" The command frame for Faceswap GUI """

from .utils import Tooltip

# An error will be thrown when importing tkinter for users without tkinter
# distribution packages or without an X-Console. This error is handled in
# gui.py but import errors still need to be captured here
try:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import ttk
except ImportError:
    tk = None
    filedialog = None
    ttk = None

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
        self.chkbtns = self.checkbuttons_frame()

    def build_frame(self):
        """ Build the options frame for this command """
        self.add_scrollbar()
        self.canvas.bind('<Configure>', self.resize_frame)

        for option in self.utils.opts[self.command]:
            optioncontrol = OptionControl(self.utils, option, self.optsframe, self.chkbtns[1])
            optioncontrol.build_full_control()

        if self.chkbtns[1].winfo_children():
            self.chkbtns[0].pack(side=tk.BOTTOM, fill=tk.X, expand=True)

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

    def checkbuttons_frame(self):
        """ Build and format frame for holding the check buttons """
        container = ttk.Frame(self.optsframe)

        lbl = ttk.Label(container, text="Options", width=16, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

        chkframe = ttk.Frame(container)
        chkframe.pack(side=tk.BOTTOM, expand=True)

        chkleft = ttk.Frame(chkframe, name="leftFrame")
        chkleft.pack(side=tk.LEFT, anchor=tk.N, expand=True)

        chkright = ttk.Frame(chkframe, name="rightFrame")
        chkright.pack(side=tk.RIGHT, anchor=tk.N, expand=True)

        return container, chkframe

class OptionControl(object):
    """ Build the correct control for the option parsed and place it on the
    frame """

    def __init__(self, utils, option, option_frame, checkbuttons_frame):
        self.utils = utils
        self.option = option
        self.option_frame = option_frame
        self.chkbtns = checkbuttons_frame

    def build_full_control(self):
        """ Build the correct control type for the option passed through """
        ctl = self.option['control']
        ctltitle = self.option['control_title']
        sysbrowser = self.option['filesystem_browser']
        ctlhelp = ' '.join(self.option.get('help', '').split())
        ctlhelp = '. '.join(i.capitalize() for i in ctlhelp.split('. '))
        ctlhelp = ctltitle + ' - ' + ctlhelp
        dflt = self.option.get('default', '')
        dflt = self.option.get('default', False) if ctl == ttk.Checkbutton else dflt
        choices = self.option['choices'] if ctl == ttk.Combobox else None

        ctlframe = self.build_one_control_frame()

        if ctl != ttk.Checkbutton:
            self.build_one_control_label(ctlframe, ctltitle)

        ctlvars = (ctl, ctltitle, dflt, ctlhelp)
        self.option['value'] = self.build_one_control(ctlframe,
                                                      ctlvars,
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
        lbl = ttk.Label(frame, text=control_title, width=16, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

    def build_one_control(self, frame, controlvars, choices, sysbrowser):
        """ Build and place the option controls """
        control, control_title, default, helptext = controlvars
        default = default if default is not None else ''

        var = tk.BooleanVar(
            frame) if control == ttk.Checkbutton else tk.StringVar(frame)
        var.set(default)

        if sysbrowser is not None:
            self.add_browser_buttons(frame, sysbrowser, var)

        if control == ttk.Checkbutton:
            self.checkbutton_to_checkframe(control, control_title, var, helptext)
        else:
            self.control_to_optionsframe(control, frame, var, choices, helptext)
        return var

    def checkbutton_to_checkframe(self, control, control_title, var, helptext):
        """ Add checkbuttons to the checkbutton frame """
        leftframe = self.chkbtns.children['leftFrame']
        rightframe = self.chkbtns.children['rightFrame']
        chkbtn_count = len({**leftframe.children, **rightframe.children})

        frame = leftframe if chkbtn_count % 2 == 0 else rightframe

        ctl = control(frame, variable=var, text=control_title)
        ctl.pack(side=tk.TOP, padx=5, pady=5, anchor=tk.W)

        Tooltip(ctl, text=helptext, wraplength=200)

    @staticmethod
    def control_to_optionsframe(control, frame, var, choices, helptext):
        """ Standard non-check buttons sit in the main options frame """
        ctl = control(frame, textvariable=var)
        ctl.pack(padx=5, pady=5, fill=tk.X, expand=True)

        if control == ttk.Combobox:
            ctl['values'] = [choice for choice in choices]

        Tooltip(ctl, text=helptext, wraplength=200)

    def add_browser_buttons(self, frame, sysbrowser, filepath):
        """ Add correct file browser button for control """
        img = self.utils.icons[sysbrowser]
        action = getattr(self, 'ask_' + sysbrowser)
        fileopn = ttk.Button(frame, image=img,
                             command=lambda cmd=action: cmd(filepath))
        fileopn.pack(padx=(0, 5), side=tk.RIGHT)

    @staticmethod
    def ask_folder(filepath):
        """ Pop-up to get path to a folder """
        dirname = filedialog.askdirectory()
        if dirname:
            filepath.set(dirname)

    @staticmethod
    def ask_load(filepath):
        """ Pop-up to get path to a file """
        filename = filedialog.askopenfilename()
        if filename:
            filepath.set(filename)

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
