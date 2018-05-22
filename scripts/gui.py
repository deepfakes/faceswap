#!/usr/bin python3
""" The optional GUI for faceswap """

import os
import sys
import tkinter as tk

from tkinter import messagebox, ttk

from lib.gui import CurrentSession, CommandNotebook, Config, ConsoleOut
from lib.gui import  DisplayNotebook, Images, Options, ProcessWrapper, StatusBar

class FaceswapGui(tk.Tk):
    """ The Graphical User Interface """

    def __init__(self, opts, pathscript):
        tk.Tk.__init__(self)
        self.geometry('1200x640+80+80')
        pathcache = os.path.join(pathscript, "lib", "gui", ".cache")
        #TODO Remove DisplayNotebook from wrapper and handle internally
        #TODO Saving session stats currently overwrites last session. Fix
        self.images = Images(pathcache)
        self.opts = opts
        self.session = CurrentSession()
        self.wrapper = ProcessWrapper(self.session, pathscript)

        StatusBar(self)
        self.images.delete_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_gui(self, debug_console):
        """ Build the GUI """
        self.title('Faceswap.py')
        self.menu()

        topcontainer, bottomcontainer = self.add_containers()

        console = ConsoleOut(bottomcontainer, debug_console)
        console.build_console()

        CommandNotebook(topcontainer, self.opts)
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
    def __init__(self, arguments):
        cmd = sys.argv[0]
        self.pathscript = os.path.realpath(os.path.dirname(cmd))
        self.args = arguments
        self.opts = Options().opts
        self.root = FaceswapGui(self.opts, self.pathscript)

    def process(self):
        """ Builds the GUI """
        self.root.build_gui(self.args.debug)
        self.root.mainloop()
