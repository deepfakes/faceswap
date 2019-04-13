#!/usr/bin/env python3
""" Utility functions for the GUI """
import logging
import os
import platform
import sys
import tkinter as tk

from tkinter import filedialog, ttk
from PIL import Image, ImageTk

from lib.Serializer import JSONSerializer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = None
_IMAGES = None


def initialize_config(cli_opts, scaling_factor, pathcache, statusbar, session):
    """ Initialize the config and add to global constant """
    global _CONFIG  # pylint: disable=global-statement
    if _CONFIG is not None:
        return
    logger.debug("Initializing config: (cli_opts: %s, tk_vars: %s, pathcache: %s, statusbar: %s, "
                 "session: %s)", cli_opts, scaling_factor, pathcache, statusbar, session)
    _CONFIG = Config(cli_opts, scaling_factor, pathcache, statusbar, session)


def get_config():
    """ return the _CONFIG constant """
    return _CONFIG


def initialize_images():
    """ Initialize the config and add to global constant """
    global _IMAGES  # pylint: disable=global-statement
    if _IMAGES is not None:
        return
    logger.debug("Initializing images")
    _IMAGES = Images()


def get_images():
    """ return the _CONFIG constant """
    return _IMAGES


def set_slider_rounding(value, var, d_type, round_to, min_max):
    """ Set the underlying variable to correct number based on slider rounding """
    if d_type == float:
        var.set(round(float(value), round_to))
    else:
        steps = range(min_max[0], min_max[1] + round_to, round_to)
        value = min(steps, key=lambda x: abs(x - int(float(value))))
        var.set(value)


class FileHandler():
    """ Raise a filedialog box and capture input """

    def __init__(self, handletype, filetype, command=None, action=None,
                 variable=None):
        logger.debug("Initializing %s: (Handletype: '%s', filetype: '%s', command: '%s', action: "
                     "'%s', variable: %s)", self.__class__.__name__, handletype, filetype, command,
                     action, variable)
        self.handletype = handletype
        self.contexts = {
            "effmpeg": {
                "input": {"extract": "filename",
                          "gen-vid": "dir",
                          "get-fps": "filename",
                          "get-info": "filename",
                          "mux-audio": "filename",
                          "rescale": "filename",
                          "rotate": "filename",
                          "slice": "filename"},
                "output": {"extract": "dir",
                           "gen-vid": "savefilename",
                           "get-fps": "nothing",
                           "get-info": "nothing",
                           "mux-audio": "savefilename",
                           "rescale": "savefilename",
                           "rotate": "savefilename",
                           "slice": "savefilename"}
            }
        }
        self.defaults = self.set_defaults()
        self.kwargs = self.set_kwargs(filetype, command, action, variable)
        self.retfile = getattr(self, self.handletype.lower())()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def filetypes(self):
        """ Set the filetypes for opening/saving """
        all_files = ("All files", "*.*")
        filetypes = {"default": (all_files,),
                     "alignments": [("JSON", "*.json"),
                                    ("Pickle", "*.p"),
                                    ("YAML", "*.yaml" "*.yml"),  # pylint: disable=W1403
                                    all_files],
                     "config": [("Faceswap config files", "*.fsw"), all_files],
                     "csv": [("Comma separated values", "*.csv"), all_files],
                     "image": [("Bitmap", "*.bmp"),
                               ("JPG", "*.jpeg" "*.jpg"),  # pylint: disable=W1403
                               ("PNG", "*.png"),
                               ("TIFF", "*.tif" "*.tiff"),  # pylint: disable=W1403
                               all_files],
                     "state": [("State files", "*.json"), all_files],
                     "log": [("Log files", "*.log"), all_files],
                     "video": [("Audio Video Interleave", "*.avi"),
                               ("Flash Video", "*.flv"),
                               ("Matroska", "*.mkv"),
                               ("MOV", "*.mov"),
                               ("MP4", "*.mp4"),
                               ("MPEG", "*.mpeg"),
                               ("WebM", "*.webm"),
                               all_files]}
        # Add in multi-select options
        for key, val in filetypes.items():
            if len(val) < 3:
                continue
            multi = ["{} Files".format(key.title())]
            multi.append(" ".join([ftype[1] for ftype in val if ftype[0] != "All files"]))
            val.insert(0, tuple(multi))
        return filetypes

    def set_defaults(self):
        """ Set the default filetype to be first in list of filetypes,
            or set a custom filetype if the first is not correct """
        defaults = {key: val[0][1].replace("*", "")
                    for key, val in self.filetypes.items()}
        defaults["default"] = None
        defaults["video"] = ".mp4"
        defaults["image"] = ".png"
        logger.debug(defaults)
        return defaults

    def set_kwargs(self, filetype, command, action, variable=None):
        """ Generate the required kwargs for the requested browser """
        logger.debug("Setting Kwargs: (filetype: '%s', command: '%s': action: '%s', "
                     "variable: '%s')", filetype, command, action, variable)
        kwargs = dict()
        if self.handletype.lower() == "context":
            self.set_context_handletype(command, action, variable)

        if self.handletype.lower() in (
                "open", "save", "filename", "filename_multi", "savefilename"):
            kwargs["filetypes"] = self.filetypes[filetype]
            if self.defaults.get(filetype, None):
                kwargs['defaultextension'] = self.defaults[filetype]
        if self.handletype.lower() == "save":
            kwargs["mode"] = "w"
        if self.handletype.lower() == "open":
            kwargs["mode"] = "r"
        logger.debug("Set Kwargs: %s", kwargs)
        return kwargs

    def set_context_handletype(self, command, action, variable):
        """ Choose the correct file browser action based on context """
        if self.contexts[command].get(variable, None) is not None:
            handletype = self.contexts[command][variable][action]
        else:
            handletype = self.contexts[command][action]
        logger.debug(handletype)
        self.handletype = handletype

    def open(self):
        """ Open a file """
        logger.debug("Popping Open browser")
        return filedialog.askopenfile(**self.kwargs)

    def save(self):
        """ Save a file """
        logger.debug("Popping Save browser")
        return filedialog.asksaveasfile(**self.kwargs)

    def dir(self):
        """ Get a directory location """
        logger.debug("Popping Dir browser")
        return filedialog.askdirectory(**self.kwargs)

    def savedir(self):
        """ Get a save dir location """
        logger.debug("Popping SaveDir browser")
        return filedialog.askdirectory(**self.kwargs)

    def filename(self):
        """ Get an existing file location """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilename(**self.kwargs)

    def filename_multi(self):
        """ Get multiple existing file locations """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilenames(**self.kwargs)

    def savefilename(self):
        """ Get a save file location """
        logger.debug("Popping SaveFilename browser")
        return filedialog.asksaveasfilename(**self.kwargs)

    @staticmethod
    def nothing():  # pylint: disable=useless-return
        """ Method that does nothing, used for disabling open/save pop up  """
        logger.debug("Popping Nothing browser")
        return


class Images():
    """ Holds locations of images and actual images

        Don't call directly. Call get_images()
    """

    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        pathcache = get_config().pathcache
        self.pathicons = os.path.join(pathcache, "icons")
        self.pathpreview = os.path.join(pathcache, "preview")
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()
        self.errcount = 0
        self.icons = dict()
        self.icons["folder"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_folder.png"))
        self.icons["load"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_file.png"))
        self.icons["load_multi"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_file.png"))
        self.icons["context"] = ImageTk.PhotoImage(file=os.path.join(
            self.pathicons, "open_file.png"))
        self.icons["save"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "save.png"))
        self.icons["reset"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "reset.png"))
        self.icons["clear"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "clear.png"))
        self.icons["graph"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "graph.png"))
        self.icons["zoom"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "zoom.png"))
        self.icons["move"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "move.png"))
        self.icons["favicon"] = ImageTk.PhotoImage(file=os.path.join(self.pathicons, "logo.png"))
        logger.debug("Initialized %s: (icons: %s)", self.__class__.__name__, self.icons)

    def delete_preview(self):
        """ Delete the preview files """
        logger.debug("Deleting previews")
        for item in os.listdir(self.pathpreview):
            if item.startswith(".gui_training_preview") and item.endswith(".jpg"):
                fullitem = os.path.join(self.pathpreview, item)
                logger.debug("Deleting: '%s'", fullitem)
                os.remove(fullitem)
        self.clear_image_cache()

    def clear_image_cache(self):
        """ Clear all cached images """
        logger.debug("Clearing image cache")
        self.pathoutput = None
        self.previewoutput = None
        self.previewtrain = dict()

    @staticmethod
    def get_images(imgpath):
        """ Get the images stored within the given directory """
        logger.trace("Getting images: '%s'", imgpath)
        if not os.path.isdir(imgpath):
            logger.debug("Folder does not exist")
            return None
        files = [os.path.join(imgpath, f)
                 for f in os.listdir(imgpath) if f.endswith((".png", ".jpg"))]
        logger.trace("Image files: %s", files)
        return files

    def load_latest_preview(self):
        """ Load the latest preview image for extract and convert """
        logger.trace("Loading preview image")
        imagefiles = self.get_images(self.pathoutput)
        if not imagefiles or len(imagefiles) == 1:
            logger.debug("No preview to display")
            self.previewoutput = None
            return
        # Get penultimate file so we don't accidentally
        # load a file that is being saved
        show_file = sorted(imagefiles, key=os.path.getctime)[-2]
        img = Image.open(show_file)
        img.thumbnail((768, 432))
        logger.trace("Displaying preview: '%s'", show_file)
        self.previewoutput = (img, ImageTk.PhotoImage(img))

    def load_training_preview(self):
        """ Load the training preview images """
        logger.trace("Loading Training preview images")
        imagefiles = self.get_images(self.pathpreview)
        modified = None
        if not imagefiles:
            logger.debug("No preview to display")
            self.previewtrain = dict()
            return
        for img in imagefiles:
            modified = os.path.getmtime(img) if modified is None else modified
            name = os.path.basename(img)
            name = os.path.splitext(name)[0]
            name = name[name.rfind("_") + 1:].title()
            try:
                logger.trace("Displaying preview: '%s'", img)
                size = self.get_current_size(name)
                self.previewtrain[name] = [Image.open(img), None, modified]
                self.resize_image(name, size)
                self.errcount = 0
            except ValueError:
                # This is probably an error reading the file whilst it's
                # being saved  so ignore it for now and only pick up if
                # there have been multiple consecutive fails
                logger.warning("Unable to display preview: (image: '%s', attempt: %s)",
                               img, self.errcount)
                if self.errcount < 10:
                    self.errcount += 1
                else:
                    logger.error("Error reading the preview file for '%s'", img)
                    print("Error reading the preview file for {}".format(name))
                    self.previewtrain[name] = None

    def get_current_size(self, name):
        """ Return the size of the currently displayed image """
        logger.trace("Getting size: '%s'", name)
        if not self.previewtrain.get(name, None):
            return None
        img = self.previewtrain[name][1]
        if not img:
            return None
        logger.trace("Got size: (name: '%s', width: '%s', height: '%s')",
                     name, img.width(), img.height())
        return img.width(), img.height()

    def resize_image(self, name, framesize):
        """ Resize the training preview image
            based on the passed in frame size """
        logger.trace("Resizing image: (name: '%s', framesize: %s", name, framesize)
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
            logger.trace("Scaling: (scale: %s, size: %s", scale, size)

            # Hacky fix to force a reload if it happens to find corrupted
            # data, probably due to reading the image whilst it is partially
            # saved. If it continues to fail, then eventually raise.
            for i in range(0, 1000):
                try:
                    displayimg = displayimg.resize(size, Image.ANTIALIAS)
                except OSError:
                    if i == 999:
                        raise
                    continue
                break

        self.previewtrain[name][1] = ImageTk.PhotoImage(displayimg)


class ContextMenu(tk.Menu):  # pylint: disable=too-many-ancestors
    """ Pop up menu """
    def __init__(self, widget):
        logger.debug("Initializing %s: (widget_class: '%s')",
                     self.__class__.__name__, widget.winfo_class())
        super().__init__(tearoff=0)
        self.widget = widget
        self.standard_actions()
        logger.debug("Initialized %s", self.__class__.__name__)

    def standard_actions(self):
        """ Standard menu actions """
        self.add_command(label="Cut", command=lambda: self.widget.event_generate("<<Cut>>"))
        self.add_command(label="Copy", command=lambda: self.widget.event_generate("<<Copy>>"))
        self.add_command(label="Paste", command=lambda: self.widget.event_generate("<<Paste>>"))
        self.add_separator()
        self.add_command(label="Select all", command=self.select_all)

    def cm_bind(self):
        """ Bind the menu to the widget's Right Click event """
        button = "<Button-2>" if platform.system() == "Darwin" else "<Button-3>"
        logger.debug("Binding '%s' to '%s'", button, self.widget.winfo_class())
        x_offset = int(34 * get_config().scaling_factor)
        self.widget.bind(button,
                         lambda event: self.tk_popup(event.x_root + x_offset, event.y_root, 0))

    def select_all(self):
        """ Select all for Text or Entry widgets """
        logger.debug("Selecting all for '%s'", self.widget.winfo_class())
        if self.widget.winfo_class() == "Text":
            self.widget.focus_force()
            self.widget.tag_add("sel", "1.0", "end")
        else:
            self.widget.focus_force()
            self.widget.select_range(0, tk.END)


class ConsoleOut(ttk.Frame):  # pylint: disable=too-many-ancestors
    """ The Console out section of the GUI """

    def __init__(self, parent, debug):
        logger.debug("Initializing %s: (parent: %s, debug: %s)",
                     self.__class__.__name__, parent, debug)
        ttk.Frame.__init__(self, parent)
        self.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(2, 0),
                  fill=tk.BOTH, expand=True)
        self.console = tk.Text(self)
        rc_menu = ContextMenu(self.console)
        rc_menu.cm_bind()
        self.console_clear = get_config().tk_vars['consoleclear']
        self.set_console_clear_var_trace()
        self.debug = debug
        self.build_console()
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_console_clear_var_trace(self):
        """ Set the trigger actions for the clear console var
            when it has been triggered from elsewhere """
        logger.debug("Set clear trace")
        self.console_clear.trace("w", self.clear)

    def build_console(self):
        """ Build and place the console """
        logger.debug("Build console")
        self.console.config(width=100, height=6, bg="gray90", fg="black")
        self.console.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self, command=self.console.yview)
        scrollbar.pack(side=tk.LEFT, fill="y")
        self.console.configure(yscrollcommand=scrollbar.set)

        self.redirect_console()
        logger.debug("Built console")

    def redirect_console(self):
        """ Redirect stdout/stderr to console frame """
        logger.debug("Redirect console")
        if self.debug:
            logger.info("Console debug activated. Outputting to main terminal")
        else:
            sys.stdout = SysOutRouter(console=self.console, out_type="stdout")
            sys.stderr = SysOutRouter(console=self.console, out_type="stderr")
        logger.debug("Redirected console")

    def clear(self, *args):  # pylint: disable=unused-argument
        """ Clear the console output screen """
        logger.debug("Clear console")
        if not self.console_clear.get():
            logger.debug("Console not set for clearing. Skipping")
            return
        self.console.delete(1.0, tk.END)
        self.console_clear.set(False)
        logger.debug("Cleared console")


class SysOutRouter():
    """ Route stdout/stderr to the console window """

    def __init__(self, console=None, out_type=None):
        logger.debug("Initializing %s: (console: %s, out_type: '%s')",
                     self.__class__.__name__, console, out_type)
        self.console = console
        self.out_type = out_type
        self.color = ("black" if out_type == "stdout" else "red")
        logger.debug("Initialized %s", self.__class__.__name__)

    def write(self, string):
        """ Capture stdout/stderr """
        self.console.insert(tk.END, string, self.out_type)
        self.console.tag_config(self.out_type, foreground=self.color)
        self.console.see(tk.END)

    @staticmethod
    def flush():
        """ If flush is forced, send it to normal terminal """
        sys.__stdout__.flush()


class Config():
    """ Global configuration settings

        Don't call directly. Call get_config()
    """

    def __init__(self, cli_opts, scaling_factor, pathcache, statusbar, session):
        logger.debug("Initializing %s: (cli_opts: %s, scaling_factor: %s, pathcache: %s, "
                     "statusbar: %s, session: %s)", self.__class__.__name__, cli_opts,
                     scaling_factor, pathcache, statusbar, session)
        self.cli_opts = cli_opts
        self.scaling_factor = scaling_factor
        self.pathcache = pathcache
        self.statusbar = statusbar
        self.serializer = JSONSerializer
        self.tk_vars = self.set_tk_vars()
        self.command_notebook = None  # set in command.py
        self.session = session
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def command_tabs(self):
        """ Return dict of command tab titles with their IDs """
        return {self.command_notebook.tab(tab_id, "text").lower(): tab_id
                for tab_id in range(0, self.command_notebook.index("end"))}

    @staticmethod
    def set_tk_vars():
        """ TK Variables to be triggered by to indicate
            what state various parts of the GUI should be in """
        display = tk.StringVar()
        display.set(None)

        runningtask = tk.BooleanVar()
        runningtask.set(False)

        actioncommand = tk.StringVar()
        actioncommand.set(None)

        generatecommand = tk.StringVar()
        generatecommand.set(None)

        consoleclear = tk.BooleanVar()
        consoleclear.set(False)

        refreshgraph = tk.BooleanVar()
        refreshgraph.set(False)

        updatepreview = tk.BooleanVar()
        updatepreview.set(False)

        traintimeout = tk.IntVar()
        traintimeout.set(120)

        tk_vars = {"display": display,
                   "runningtask": runningtask,
                   "action": actioncommand,
                   "generate": generatecommand,
                   "consoleclear": consoleclear,
                   "refreshgraph": refreshgraph,
                   "updatepreview": updatepreview,
                   "traintimeout": traintimeout}
        logger.debug(tk_vars)
        return tk_vars

    def load(self, command=None, filename=None):
        """ Pop up load dialog for a saved config file """
        logger.debug("Loading config: (command: '%s')", command)
        if filename:
            with open(filename, "r") as cfgfile:
                cfg = self.serializer.unmarshal(cfgfile.read())
        else:
            cfgfile = FileHandler("open", "config").retfile
            if not cfgfile:
                return
            cfg = self.serializer.unmarshal(cfgfile.read())

        if not command and len(cfg.keys()) == 1:
            command = list(cfg.keys())[0]

        opts = self.get_command_options(cfg, command) if command else cfg
        if not opts:
            return

        for cmd, opts in opts.items():
            self.set_command_args(cmd, opts)

        if command:
            self.command_notebook.select(self.command_tabs[command])

        self.add_to_recent(cfgfile.name, command)
        logger.debug("Loaded config: (command: '%s', cfgfile: '%s')", command, cfgfile)

    def get_command_options(self, cfg, command):
        """ return the saved options for the requested
            command, if not loading global options """
        opts = cfg.get(command, None)
        retval = {command: opts}
        if not opts:
            self.tk_vars["consoleclear"].set(True)
            print("No {} section found in file".format(command))
            logger.info("No  %s section found in file", command)
            retval = None
        logger.debug(retval)
        return retval

    def set_command_args(self, command, options):
        """ Pass the saved config items back to the CliOptions """
        if not options:
            return
        for srcopt, srcval in options.items():
            optvar = self.cli_opts.get_one_option_variable(command, srcopt)
            if not optvar:
                continue
            optvar.set(srcval)

    def save(self, command=None):
        """ Save the current GUI state to a config file in json format """
        logger.debug("Saving config: (command: '%s')", command)
        cfgfile = FileHandler("save", "config").retfile
        if not cfgfile:
            return
        cfg = self.cli_opts.get_option_values(command)
        cfgfile.write(self.serializer.marshal(cfg))
        cfgfile.close()
        self.add_to_recent(cfgfile.name, command)
        logger.debug("Saved config: (command: '%s', cfgfile: '%s')", command, cfgfile)

    def add_to_recent(self, filename, command):
        """ Add to recent files """
        recent_filename = os.path.join(self.pathcache, ".recent.json")
        logger.debug("Adding to recent files '%s': (%s, %s)", recent_filename, filename, command)
        if not os.path.exists(recent_filename) or os.path.getsize(recent_filename) == 0:
            recent_files = list()
        else:
            with open(recent_filename, "rb") as inp:
                recent_files = self.serializer.unmarshal(inp.read().decode("utf-8"))
        logger.debug("Initial recent files: %s", recent_files)
        filenames = [recent[0] for recent in recent_files]
        if filename in filenames:
            idx = filenames.index(filename)
            del recent_files[idx]
        recent_files.insert(0, (filename, command))
        recent_files = recent_files[:20]
        logger.debug("Final recent files: %s", recent_files)
        recent_json = self.serializer.marshal(recent_files)
        with open(recent_filename, "wb") as out:
            out.write(recent_json.encode("utf-8"))
