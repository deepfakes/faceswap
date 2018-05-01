#!/usr/bin python3
""" Utility functions for the GUI """

import os
import tkinter as tk

class Singleton(type):
    """ Instigate a singleton.
    From: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

    Singletons are often frowned upon. Feel free to instigate a better solution """

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Images(object, metaclass=Singleton):
    """ Holds locations of images and actual images """
    def __init__(self, pathcache=None):
        self.pathicons = os.path.join(pathcache, "icons")
        self.pathpreview = os.path.join(pathcache, "preview")

        self.icons = dict()
        self.icons["folder"] = tk.PhotoImage(file=os.path.join(self.pathicons, "open_folder.png"))
        self.icons["load"] = tk.PhotoImage(file=os.path.join(self.pathicons, "open_file.png"))
        self.icons["save"] = tk.PhotoImage(file=os.path.join(self.pathicons, "save.png"))
        self.icons["reset"] = tk.PhotoImage(file=os.path.join(self.pathicons, "reset.png"))
        self.icons["clear"] = tk.PhotoImage(file=os.path.join(self.pathicons, "clear.png"))

    def delete_preview(self):
        """ Delete the preview files """
        for item in os.listdir(self.pathpreview):
            if item.startswith(".gui_preview_") and item.endswith(".png"):
                fullitem = os.path.join(self.pathpreview, item)
                os.remove(fullitem)
