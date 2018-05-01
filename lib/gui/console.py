#!/usr/bin python3
""" Console section of the GUI """

import sys
import tkinter as tk
from tkinter import ttk

from .utils import Singleton

class ConsoleOut(ttk.Frame, metaclass=Singleton):
    """ The Console out section of the GUI """

    def __init__(self, parent=None, debug=None):
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(2, 0),
                  fill=tk.BOTH, expand=True)
        self.console = tk.Text(self)
        self.debug = debug

    def build_console(self):
        """ Build and place the console """
        self.console.config(width=100, height=6, bg='gray90', fg='black')
        self.console.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self, command=self.console.yview)
        scrollbar.pack(side=tk.LEFT, fill='y')
        self.console.configure(yscrollcommand=scrollbar.set)

        self.redirect_console()

    def redirect_console(self):
        """ Redirect stdout/stderr to console frame """
        if self.debug:
            print('Console debug activated. Outputting to main terminal')
        else:
            sys.stdout = SysOutRouter(console=self.console, out_type="stdout")
            sys.stderr = SysOutRouter(console=self.console, out_type="stderr")

    def clear(self):
        """ Clear the console output screen """
        self.console.delete(1.0, tk.END)

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
