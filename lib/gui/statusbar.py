#!/usr/bin python3
""" Status bar for the GUI """

import tkinter as tk
from tkinter import ttk


class StatusBar(ttk.Frame):
    """ Status Bar for displaying the Status Message and
        Progress Bar """

    def __init__(self, parent):
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.BOTTOM, padx=10, pady=2, fill=tk.X, expand=False)

        self.status_message = tk.StringVar()
        self.pbar_message = tk.StringVar()
        self.pbar_position = tk.IntVar()

        self.status_message.set("Ready")

        self.status()
        self.pbar = self.progress_bar()

    def status(self):
        """ Place Status into bottom bar """
        statusframe = ttk.Frame(self)
        statusframe.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=False)

        lbltitle = ttk.Label(statusframe, text="Status:", width=6, anchor=tk.W)
        lbltitle.pack(side=tk.LEFT, expand=False)

        lblstatus = ttk.Label(statusframe,
                              width=20,
                              textvariable=self.status_message,
                              anchor=tk.W)
        lblstatus.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)

    def progress_bar(self):
        """ Place progress bar into bottom bar """
        progressframe = ttk.Frame(self)
        progressframe.pack(side=tk.RIGHT, anchor=tk.E, fill=tk.X)

        lblmessage = ttk.Label(progressframe, textvariable=self.pbar_message)
        lblmessage.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        pbar = ttk.Progressbar(progressframe,
                               length=200,
                               variable=self.pbar_position,
                               maximum=1000,
                               mode="determinate")
        pbar.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        pbar.pack_forget()
        return pbar

    def progress_start(self, mode):
        """ Set progress bar mode and display """
        self.progress_set_mode(mode)
        self.pbar.pack()

    def progress_stop(self):
        """ Reset progress bar and hide """
        self.pbar_message.set("")
        self.pbar_position.set(0)
        self.progress_set_mode("determinate")
        self.pbar.pack_forget()

    def progress_set_mode(self, mode):
        """ Set the progress bar mode """
        self.pbar.config(mode=mode)
        if mode == "indeterminate":
            self.pbar.config(maximum=100)
            self.pbar.start()
        else:
            self.pbar.stop()
            self.pbar.config(maximum=1000)

    def progress_update(self, message, position, update_position=True):
        """ Update the GUIs progress bar and position """
        self.pbar_message.set(message)
        if update_position:
            self.pbar_position.set(position)
