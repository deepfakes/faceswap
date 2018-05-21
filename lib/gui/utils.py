#!/usr/bin python3
""" Utility functions for the GUI """

import os
import tkinter as tk

from tkinter import filedialog
from PIL import Image, ImageTk

class Singleton(type):
    """ Instigate a singleton.
    From: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

    Singletons are often frowned upon. Feel free to instigate a better solution """

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class FileHandler(object):
    """ Raise a filedialog box and capture input """
    def __init__(self, handletype, filetype=None):

        self.filetypes = {'config': (('Faceswap config files', '*.fsw'), ('All files', '*.*')),
                          'session': (('Faceswap session files', '*.fss'), ('All files', '*.*')),
                          'csv':(('Comma separated values', '*.csv'), ('All files', '*.*'))}
        self.retfile = getattr(self, handletype.lower())(filetype)

    def open(self, filetype):
        """ Open a file """
        return filedialog.askopenfile(mode='r', filetypes=self.filetypes[filetype])

    def save(self, filetype):
        """ Save a file """
        default = self.filetypes[filetype][0][1].replace('*', '')
        return filedialog.asksaveasfile(mode='w',
                                        filetypes=self.filetypes[filetype],
                                        defaultextension=default)

    @staticmethod
    def dir(filetype):
        """ Get a directory location """
        return filedialog.askdirectory()

    @staticmethod
    def filename(filetype):
        """ Get an existing file location """
        return filedialog.askopenfilename()


class Images(object, metaclass=Singleton):
    """ Holds locations of images and actual images """
    def __init__(self, pathcache=None):
        self.pathicons = os.path.join(pathcache, "icons")
        self.pathpreview = os.path.join(pathcache, "preview")
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()
        self.errcount = 0

        self.icons = dict()
        self.icons["folder"] = tk.PhotoImage(file=os.path.join(self.pathicons, "open_folder.png"))
        self.icons["load"] = tk.PhotoImage(file=os.path.join(self.pathicons, "open_file.png"))
        self.icons["save"] = tk.PhotoImage(file=os.path.join(self.pathicons, "save.png"))
        self.icons["reset"] = tk.PhotoImage(file=os.path.join(self.pathicons, "reset.png"))
        self.icons["clear"] = tk.PhotoImage(file=os.path.join(self.pathicons, "clear.png"))
        self.icons["graph"] = tk.PhotoImage(file=os.path.join(self.pathicons, "graph.png"))
        self.icons["zoom"] = tk.PhotoImage(file=os.path.join(self.pathicons, "zoom.png"))
        self.icons["move"] = tk.PhotoImage(file=os.path.join(self.pathicons, "move.png"))

    def delete_preview(self):
        """ Delete the preview files """
        for item in os.listdir(self.pathpreview):
            if item.startswith(".gui_preview_") and item.endswith(".jpg"):
                fullitem = os.path.join(self.pathpreview, item)
                os.remove(fullitem)
        self.clear_image_cache()

    def clear_image_cache(self):
        """ Clear all cached images """
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()

    @staticmethod
    def get_images(imgpath):
        """ Get the images stored within the given directory """
        if not os.path.isdir(imgpath):
            return None
        files = [os.path.join(imgpath, f)
                 for f in os.listdir(imgpath) if f.endswith(('.png', '.jpg'))]
        return files

    def load_latest_preview(self):
        """ Load the latest preview image for extract and convert """
        imagefiles = self.get_images(self.pathoutput)
        if not imagefiles or len(imagefiles) == 1:
            self.previewoutput = None
            return
        # Get penultimate file so we don't accidently load a file that is being saved
        show_file = sorted(imagefiles, key=os.path.getctime)[-2]
        img = Image.open(show_file)
        img.thumbnail((768, 432))
        self.previewoutput = (img, ImageTk.PhotoImage(img))

    def load_training_preview(self):
        """ Load the training preview images """
        imagefiles = self.get_images(self.pathpreview)
        modified = None
        if not imagefiles:
            self.previewtrain = dict()
            return
        for img in imagefiles:
            modified = os.path.getmtime(img) if modified is None else modified
            name = os.path.basename(img)
            name = os.path.splitext(name)[0]
            name = name[name.rfind("_") + 1:].title()
            try:
                size = self.get_current_size(name)
                self.previewtrain[name] = [Image.open(img), None, modified]
                self.resize_image(name, size)
                self.errcount = 0
            except ValueError:
                # This is probably an error reading the file whilst it's
                # being saved  so ignore it for now and only pick up if
                # there have been multiple consecutive fails
                if self.errcount < 10:
                    self.errcount += 1
                else:
                    print('Error reading the preview file for {}'.format(name))
                    self.previewtrain[name] = None

    def get_current_size(self, name):
        """ Return the size of the currently displayed image """
        if not self.previewtrain.get(name, None):
            return None
        img = self.previewtrain[name][1]
        if not img:
            return None
        return img.width(), img.height()

    def resize_image(self, name, framesize):
        """ Resize the training preview image based on the passed in frame size """
        displayimg = self.previewtrain[name][0]
        if framesize:
            frameratio = float(framesize[0]) / float(framesize[1])
            imgratio = float(displayimg.size[0]) / float(displayimg.size[1])

            if frameratio <= imgratio:
                scale = framesize[0] / float(displayimg.size[0])
                size = (framesize[0], int(displayimg.size[1] * scale))
            else:
                scale = framesize[1] / float(displayimg.size[1])
                size = (int(displayimg.size[0] * scale), framesize[1])

            # Hacky fix to force a reload if it happens to find corrupted
            # data, probably due to reading the image whilst it is partially
            # saved. If it continues to fail, then eventually raise.
            for i in range(0, 1000):
                try:
                    displayimg = displayimg.resize(size, Image.ANTIALIAS)
                except OSError:
                    if i == 999:
                        raise
                    else:
                        continue
                break

        self.previewtrain[name][1] = ImageTk.PhotoImage(displayimg)
