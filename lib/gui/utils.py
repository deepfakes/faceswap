#!/usr/bin/env python3
""" Utility functions for the GUI """
import os
import sys
import tkinter as tk

from tkinter import filedialog, ttk
from PIL import Image, ImageTk


class Singleton(type):
    """ Instigate a singleton.
    From: https://stackoverflow.com/questions/6760685

    Singletons are often frowned upon.
    Feel free to instigate a better solution """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton,
                                        cls).__call__(*args,
                                                      **kwargs)
        return cls._instances[cls]


class FileHandler(object):
    """ Raise a filedialog box and capture input """

    def __init__(self, handletype, filetype, command=None, action=None,
                 variable=None):

        all_files = ("All files", "*.*")
        self.filetypes = {"default": (all_files,),
                          "alignments": (("JSON", "*.json"),
                                         ("Pickle", "*.p"),
                                         ("YAML", "*.yaml"),
                                         all_files),
                          "config": (("Faceswap config files", "*.fsw"),
                                     all_files),
                          "csv": (("Comma separated values", "*.csv"),
                                  all_files),
                          "image": (("Bitmap", "*.bmp"),
                                    ("JPG", "*.jpeg", "*.jpg"),
                                    ("PNG", "*.png"),
                                    ("TIFF", "*.tif", "*.tiff"),
                                    all_files),
                          "session": (("Faceswap session files", "*.fss"),
                                      all_files),
                          "video": (("Audio Video Interleave", "*.avi"),
                                    ("Flash Video", "*.flv"),
                                    ("Matroska", "*.mkv"),
                                    ("MOV", "*.mov"),
                                    ("MP4", "*.mp4"),
                                    ("MPEG", "*.mpeg"),
                                    ("WebM", "*.webm"),
                                    all_files)}
        self.contexts = {
            "effmpeg": {
                "input": {"extract": "open",
                          "gen-vid": "dir",
                          "get-fps": "open",
                          "get-info": "open",
                          "mux-audio": "open",
                          "rescale": "open",
                          "rotate": "open",
                          "slice": "open"},
                "output": {"extract": "dir",
                           "gen-vid": "save",
                           "get-fps": "nothing",
                           "get-info": "nothing",
                           "mux-audio": "save",
                           "rescale": "save",
                           "rotate": "save",
                           "slice": "save"}
            }
        }
        self.defaults = self.set_defaults()
        self.kwargs = self.set_kwargs(handletype, filetype, command, action,
                                      variable)
        self.retfile = getattr(self, handletype.lower())()

    def set_defaults(self):
        """ Set the default filetype to be first in list of filetypes,
            or set a custom filetype if the first is not correct """
        defaults = {key: val[0][1].replace("*", "")
                    for key, val in self.filetypes.items()}
        defaults["default"] = None
        defaults["video"] = ".mp4"
        defaults["image"] = ".png"
        return defaults

    def set_kwargs(self, handletype, filetype, command, action, variable=None):
        """ Generate the required kwargs for the requested browser """
        kwargs = dict()
        if handletype.lower() in ("open", "save", "filename", "savefilename"):
            kwargs["filetypes"] = self.filetypes[filetype]
            if self.defaults.get(filetype, None):
                kwargs['defaultextension'] = self.defaults[filetype]
        if handletype.lower() == "save":
            kwargs["mode"] = "w"
        if handletype.lower() == "open":
            kwargs["mode"] = "r"
        if handletype.lower() == "context":
            kwargs["filetype"] = filetype
            kwargs["command"] = command
            kwargs["action"] = action
            kwargs["variable"] = variable
        return kwargs

    def open(self):
        """ Open a file """
        return filedialog.askopenfile(**self.kwargs)

    def save(self):
        """ Save a file """
        return filedialog.asksaveasfile(**self.kwargs)

    def dir(self):
        """ Get a directory location """
        return filedialog.askdirectory(**self.kwargs)

    def savedir(self):
        """ Get a save dir location """
        return filedialog.askdirectory(**self.kwargs)

    def filename(self):
        """ Get an existing file location """
        return filedialog.askopenfilename(**self.kwargs)

    def savefilename(self):
        """ Get a save file location """
        return filedialog.asksaveasfilename(**self.kwargs)

    def nothing(self):
        """ Method that does nothing, used for disabling open/save pop up  """
        return

    def context(self):
        """ Choose the correct file browser action based on context """
        command = self.kwargs["command"]
        action = self.kwargs["action"]
        filetype = self.kwargs["filetype"]
        variable = self.kwargs["variable"]
        if self.contexts[command].get(variable, None) is not None:
            handletype = self.contexts[command][variable][action]
        else:
            handletype = self.contexts[command][action]
        self.kwargs = self.set_kwargs(handletype, filetype, command, action,
                                      variable)
        self.retfile = getattr(self, handletype.lower())()


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
        self.icons["folder"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                               "open_folder.png"))
        self.icons["load"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                             "open_file.png"))
        self.icons["context"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                                "open_file.png"))
        self.icons["save"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                             "save.png"))
        self.icons["reset"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                              "reset.png"))
        self.icons["clear"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                              "clear.png"))
        self.icons["graph"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                              "graph.png"))
        self.icons["zoom"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                             "zoom.png"))
        self.icons["move"] = tk.PhotoImage(file=os.path.join(self.pathicons,
                                                             "move.png"))

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
                 for f in os.listdir(imgpath) if f.endswith((".png", ".jpg"))]
        return files

    def load_latest_preview(self):
        """ Load the latest preview image for extract and convert """
        imagefiles = self.get_images(self.pathoutput)
        if not imagefiles or len(imagefiles) == 1:
            self.previewoutput = None
            return
        # Get penultimate file so we don't accidently
        # load a file that is being saved
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
                    print("Error reading the preview file for {}".format(name))
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
        """ Resize the training preview image
            based on the passed in frame size """
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


class ConsoleOut(ttk.Frame):
    """ The Console out section of the GUI """

    def __init__(self, parent, debug, tk_vars):
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(2, 0),
                  fill=tk.BOTH, expand=True)
        self.console = tk.Text(self)
        self.console_clear = tk_vars['consoleclear']
        self.set_console_clear_var_trace()
        self.debug = debug
        self.build_console()

    def set_console_clear_var_trace(self):
        """ Set the trigger actions for the clear console var
            when it has been triggered from elsewhere """
        self.console_clear.trace("w", self.clear)

    def build_console(self):
        """ Build and place the console """
        self.console.config(width=100, height=6, bg="gray90", fg="black")
        self.console.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self, command=self.console.yview)
        scrollbar.pack(side=tk.LEFT, fill="y")
        self.console.configure(yscrollcommand=scrollbar.set)

        self.redirect_console()

    def redirect_console(self):
        """ Redirect stdout/stderr to console frame """
        if self.debug:
            print("Console debug activated. Outputting to main terminal")
        else:
            sys.stdout = SysOutRouter(console=self.console, out_type="stdout")
            sys.stderr = SysOutRouter(console=self.console, out_type="stderr")

    def clear(self, *args):
        """ Clear the console output screen """
        if not self.console_clear.get():
            return
        self.console.delete(1.0, tk.END)
        self.console_clear.set(False)


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
