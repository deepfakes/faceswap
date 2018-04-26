#!/usr/bin python3
""" Console section of the GUI """

import sys

# An error will be thrown when importing tkinter for users without tkinter
# distribution packages or without an X-Console. This error is handled in
# gui.py but import errors still need to be captured here
try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None

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

        self.redirect_console()

    def redirect_console(self):
        """ Redirect stdout/stderr to console frame """
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
