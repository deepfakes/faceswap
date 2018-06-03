#!/usr/bin python3
""" The optional GUI for faceswap """

import ctypes
import os
import sys
import tkinter as tk

from tkinter import messagebox, ttk

from lib.gui import CurrentSession, CommandNotebook, Config, ConsoleOut
from lib.gui import  DisplayNotebook, Images, CliOptions, ProcessWrapper, StatusBar


class FaceswapGui(tk.Tk):
    """ The Graphical User Interface """

    def __init__(self, pathscript):
        tk.Tk.__init__(self)
        self.geometry("1200x640+80+80")
        pathcache = os.path.join(pathscript, "lib", "gui", ".cache")
        self.images = Images(pathcache)
        self.cliopts = CliOptions()
        self.session = CurrentSession()
        self.wrapper = ProcessWrapper(self.session, pathscript, self.cliopts)

        StatusBar(self)
        self.images.delete_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    def build_gui(self, debug_console):
        """ Build the GUI """
        self.title("Faceswap.py")
        self.menu()

        topcontainer, bottomcontainer = self.add_containers()

        console = ConsoleOut(bottomcontainer, debug_console)
        console.build_console()

        CommandNotebook(topcontainer, self.cliopts, self.wrapper.tk_vars)

        DisplayNotebook(topcontainer, self.session, self.wrapper.tk_vars)

    def menu(self):
        """ Menu bar for loading and saving configs """
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)

        config = Config(self.cliopts)

        filemenu.add_command(label="Load full config...",
                             underline=0,
                             command=config.load)
        filemenu.add_command(label="Save full config...",
                             underline=0,
                             command=config.save)
        filemenu.add_separator()
        filemenu.add_command(label="Reset all to default",
                             underline=0,
                             command=self.cliopts.reset)
        filemenu.add_command(label="Clear all",
                             underline=0,
                             command=self.cliopts.clear)
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
        if self.wrapper.tk_vars["runningtask"].get() and not confirm("Close", confirmtxt):
            return
        if self.wrapper.tk_vars["runningtask"].get():
            self.wrapper.task.terminate()
        self.images.delete_preview()
        self.quit()
        exit()

class Gui(object):
    """ The GUI process. """
    def __init__(self, arguments):
        cmd = sys.argv[0]
        pathscript = os.path.realpath(os.path.dirname(cmd))
        self.args = arguments
        self.root = FaceswapGui(pathscript)

    @staticmethod
    def set_windows_font_scaling():
        """ Set process to be dpi aware for windows users
            to fix blurry scaled fonts """
        if os.name == "nt":
            user32 = ctypes.WinDLL("user32")
            user32.SetProcessDPIAware(True)

    def process(self):
        """ Builds the GUI """
        self.set_windows_font_scaling()
        self.root.build_gui(self.args.debug)
        self.root.mainloop()
