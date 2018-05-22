#!/usr/bin python3
""" The optional GUI for faceswap """

import os
import sys
import tkinter as tk

from tkinter import messagebox, ttk
from argparse import SUPPRESS

import lib.cli as cli
from lib.gui import CurrentSession, CommandNotebook, Config, ConsoleOut
from lib.gui import  DisplayNotebook, Images, ProcessWrapper, StatusBar

class FaceswapGui(tk.Tk):
    """ The Graphical User Interface """

    def __init__(self, opts, pathscript, calling_file="faceswap.py"):
        tk.Tk.__init__(self)
        self.geometry('1200x640+80+80')
        pathcache = os.path.join(pathscript, "lib", "gui", ".cache")
        #TODO Remove DisplayNotebook from wrapper and handle internally
        #TODO Fix circular imports:
        #   singletion/console
        #TODO Saving session stats currently overwrites last session. Fix
        self.images = Images(pathcache)
        self.opts = opts
        self.calling_file = calling_file
        self.session = CurrentSession()
        self.wrapper = ProcessWrapper(self.session, pathscript, calling_file)

        StatusBar(self)
        self.images.delete_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_gui(self, debug_console):
        """ Build the GUI """
        self.title(self.calling_file)
        self.menu()

        topcontainer, bottomcontainer = self.add_containers()

        console = ConsoleOut(bottomcontainer, debug_console)
        console.build_console()

        CommandNotebook(topcontainer, self.opts, self.calling_file)
        self.wrapper.displaybook = DisplayNotebook(topcontainer, self.session)

    def menu(self):
        """ Menu bar for loading and saving configs """
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)

        config = Config(self.opts)

        filemenu.add_command(label="Load full config...",
                             underline=0,
                             command=config.load)
        filemenu.add_command(label="Save full config...",
                             underline=0,
                             command=config.save)
        filemenu.add_separator()
        filemenu.add_command(label="Reset all to default",
                             underline=0,
                             command=config.reset)
        filemenu.add_command(label="Clear all",
                             underline=0,
                             command=config.clear)
        filemenu.add_separator()
        filemenu.add_command(label="Quit",
                             underline=0,
                             command=self.close_app)

        menubar.add_cascade(label="File", menu=filemenu, underline=0)
        self.config(menu=menubar)

    def add_containers(self):
        """ Add the paned window containers that hold each main area of the gui """
        maincontainer = tk.PanedWindow(self,
                                       sashrelief=tk.RAISED,
                                       orient=tk.VERTICAL)
        maincontainer.pack(fill=tk.BOTH, expand=True)

        topcontainer = tk.PanedWindow(maincontainer,
                                      sashrelief=tk.RAISED,
                                      orient=tk.HORIZONTAL)
        maincontainer.add(topcontainer)

        bottomcontainer = ttk.Frame(maincontainer, height=150)
        maincontainer.add(bottomcontainer)

        return topcontainer, bottomcontainer

    def close_app(self):
        """ Close Python. This is here because the graph
            animation function continues to run even when
            tkinter has gone away """
        confirm = messagebox.askokcancel
        confirmtxt = "Processes are still running. Are you sure...?"
        if self.wrapper.runningtask and not confirm("Close", confirmtxt):
            return
        if self.wrapper.runningtask:
            self.wrapper.task.terminate()
        self.images.delete_preview()
        self.quit()
        exit()

class Gui(object):
    """ The GUI process. """
    def __init__(self, arguments, subparsers):
        cmd = sys.argv[0]
        self.pathscript = os.path.realpath(os.path.dirname(cmd))
        self.args = arguments
        self.opts = self.extract_options(subparsers)
        self.root = FaceswapGui(self.opts, self.pathscript, calling_file=cmd)

    def extract_options(self, subparsers):
        """ Extract the existing ArgParse Options """
        opts = {cmd: subparsers[cmd].argument_list + subparsers[cmd].optional_arguments
                for cmd in subparsers.keys()}
        for command in opts.values():
            for opt in command:
                if opt.get("help", "") == SUPPRESS:
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
        ctltitle = ctltitle.replace("-", " ").replace("_", " ").strip().title()
        return ctltitle

    @staticmethod
    def set_control(option):
        """ Set the control and filesystem browser to use for each option """
        sysbrowser = None
        filetypes = None
        actions_open_type = None
        ctl = ttk.Entry
        if option.get('action', '') == cli.FullPaths:
            sysbrowser = 'folder'
        elif option.get('action', '') == cli.DirFullPaths:
            sysbrowser = 'folder'
        elif option.get('action', '') == cli.FileFullPaths:
            sysbrowser = 'load'
            filetypes = option.get('filetypes', None)
        elif option.get('action', '') == cli.ComboFullPaths:
            sysbrowser = 'combo'
            actions_open_type = option['actions_open_type']
            filetypes = option.get('filetypes', None)
        elif option.get('choices', '') != '':
            ctl = ttk.Combobox
        elif option.get("action", "") == "store_true":
            ctl = ttk.Checkbutton
        return ctl, sysbrowser, filetypes, actions_open_type

    def process(self):
        """ Builds the GUI """
        self.root.build_gui(self.args.debug)
        self.root.mainloop()
