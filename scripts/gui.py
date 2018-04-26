#!/usr/bin python3
""" The optional GUI for faceswap """

import os
import sys

from argparse import SUPPRESS

from lib.cli import FullPaths
from lib.gui import CommandTab, ConsoleOut, DisplayTab, Utils

# An error will be thrown when importing tkinter for users without tkinter
# distribution packages or without an X-Console. Therefore if importing fails
# no attempt will be made to instantiate the gui.
try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import messagebox
except ImportError:
    tk = None
    ttk = None
    messagebox = None

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

        statuscontainer = ttk.Frame(self.gui)
        statuscontainer.pack(side=tk.BOTTOM, padx=10, pady=2, fill=tk.X, expand=True)

        optsnotebook = ttk.Notebook(topcontainer, width=420, height=500)
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
        self.add_status_bar(statuscontainer)
        self.add_progress_bar(statuscontainer)

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
        """ Place Status into bottom bar """
        statusframe = ttk.Frame(frame)
        statusframe.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=False)

        lbltitle = ttk.Label(statusframe, text='Status:', width=6, anchor=tk.W)
        lbltitle.pack(side=tk.LEFT, expand=False)
        self.utils.guitext['status'].set('Ready')
        lblstatus = ttk.Label(statusframe,
                              width=20,
                              textvariable=self.utils.guitext['status'],
                              anchor=tk.W)
        lblstatus.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)

    def add_progress_bar(self, frame):
        """ Place progress bar into bottom bar """
        progressframe = ttk.Frame(frame)
        progressframe.pack(side=tk.RIGHT, anchor=tk.E, fill=tk.X)

        lblmessage = ttk.Label(progressframe, textvariable=self.utils.progress['message'])
        lblmessage.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        progressbar = ttk.Progressbar(progressframe,
                                      length=200,
                                      variable=self.utils.progress['position'],
                                      maximum=1000,
                                      mode='determinate')
        progressbar.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        self.utils.progress['bar'] = progressbar

    def close_app(self):
        """ Close Python. This is here because the graph
            animation function continues to run even when
            tkinter has gone away """
        confirm = messagebox.askokcancel
        confirmtxt = 'Processes are still running. Are you sure...?'
        if self.utils.runningtask and not confirm('Close', confirmtxt):
            return
        if self.utils.runningtask:
            self.utils.task.terminate()
        self.utils.delete_preview()
        self.gui.quit()
        exit()

class Gui(object):
    """ The GUI process. """
    def __init__(self, arguments, subparsers):
        # Don't try to load the GUI if there is no display or there are
        # problems importing tkinter
        if not self.check_display() or not self.check_tkinter_available():
            return

        cmd = sys.argv
        pathscript = os.path.realpath(os.path.dirname(cmd[0]))

        self.args = arguments
        self.opts = self.extract_options(subparsers)
        self.utils = Utils(self.opts, pathscript, calling_file=cmd[0])
        self.root = FaceswapGui(self.utils, calling_file=cmd[0])

    @staticmethod
    def check_display():
        """ Check whether there is a display to output the GUI. If running on
            Windows then assume not running in headless mode """
        if not os.environ.get('DISPLAY', None) and os.name != 'nt':
            if os.name == 'posix':
                print('macOS users need to install XQuartz. '
                      'See https://support.apple.com/en-gb/HT201341')
            return False
        return True

    @staticmethod
    def check_tkinter_available():
        """ Check whether TkInter is installed on user's machine """
        tkinter_vars = [tk, ttk, messagebox]
        if any(var is None for var in tkinter_vars):
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
                ctl, sysbrowser = self.set_control(opt)
                opt['control_title'] = self.set_control_title(
                    opt.get('opts', ''))
                opt['control'] = ctl
                opt['filesystem_browser'] = sysbrowser
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
        ctl = ttk.Entry
        if option.get('dest', '') == 'alignments_path':
            sysbrowser = 'load'
        elif option.get('action', '') == FullPaths:
            sysbrowser = 'folder'
        elif option.get('choices', '') != '':
            ctl = ttk.Combobox
        elif option.get('action', '') == 'store_true':
            ctl = ttk.Checkbutton
        return ctl, sysbrowser

    def process(self):
        """ Builds the GUI """
        self.utils.debugconsole = self.args.debug
        self.root.build_gui()
        self.root.gui.mainloop()
