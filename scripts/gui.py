#!/usr/bin python3
""" The optional GUI for faceswap """

import os
import sys
import tkinter as tk

from tkinter import messagebox, ttk

from lib.gui import (CliOptions, CurrentSession, CommandNotebook, Config,
                     ConsoleOut, DisplayNotebook, Images, ProcessWrapper,
                     StatusBar)


class FaceswapGui(tk.Tk):
    """ The Graphical User Interface """

    def __init__(self, pathscript):
        tk.Tk.__init__(self)
        self.scaling_factor = self.get_scaling()
        self.set_geometry()

        pathcache = os.path.join(pathscript, "lib", "gui", ".cache")
        self.images = Images(pathcache)
        self.cliopts = CliOptions()
        self.session = CurrentSession()
        statusbar = StatusBar(self)
        self.wrapper = ProcessWrapper(statusbar,
                                      self.session,
                                      pathscript,
                                      self.cliopts)

        self.images.delete_preview()
        self.protocol("WM_DELETE_WINDOW", self.close_app)

    def get_scaling(self):
        """ Get the display DPI """
        dpi = self.winfo_fpixels("1i")
        return dpi / 72.0

    def set_geometry(self):
        """ Set GUI geometry """
        self.tk.call("tk", "scaling", self.scaling_factor)
        width = int(1200 * self.scaling_factor)
        height = int(640 * self.scaling_factor)
        self.geometry("{}x{}+80+80".format(str(width), str(height)))

    def build_gui(self, debug_console):
        """ Build the GUI """
        self.title("Faceswap.py")
        self.menu()

        topcontainer, bottomcontainer = self.add_containers()

        CommandNotebook(topcontainer,
                        self.cliopts,
                        self.wrapper.tk_vars,
                        self.scaling_factor)
        DisplayNotebook(topcontainer,
                        self.session,
                        self.wrapper.tk_vars,
                        self.scaling_factor)
        ConsoleOut(bottomcontainer, debug_console, self.wrapper.tk_vars)

    def menu(self):
        """ Menu bar for loading and saving configs """
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)

        config = Config(self.cliopts, self.wrapper.tk_vars)

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
        """ Add the paned window containers that
            hold each main area of the gui """
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
        if (self.wrapper.tk_vars["runningtask"].get()
                and not confirm("Close", confirmtxt)):
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

    def process(self):
        """ Builds the GUI """
        self.root.build_gui(self.args.debug)
        self.root.mainloop()
