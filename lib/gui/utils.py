#!/usr/bin python3
""" Utility functions for the GUI """

import inspect
import os
import sys
import tkinter as tk

from tkinter import filedialog, ttk
from argparse import SUPPRESS
from PIL import Image, ImageTk

import lib.cli as cli
from lib.Serializer import JSONSerializer
import tools.cli as ToolsCli


class Singleton(type):
    """ Instigate a singleton.
    From: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

    Singletons are often frowned upon. Feel free to instigate a better solution """

    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Options(object):
    """ Class and methods for the command line options """
    def __init__(self):
        self.categories = ('faceswap', 'tools')
        self.commands = dict()
        self.opts = self.extract_options()

    def extract_options(self):
        """ Get the commands that belong to each category """
        opts = dict()
        for category in self.categories:
            src = ToolsCli if category == 'tools' else cli
            mod_classes = self.cli_classes(src)
            self.commands[category] = self.commands_sorted(category, mod_classes)
            opts[category] = self.extract_command_options(src, mod_classes)
        return opts

    @staticmethod
    def cli_classes(cli_source):
        """ Parse the cli scripts for the arg classes """
        mod_classes = list()
        for name, obj in inspect.getmembers(cli_source):
            if inspect.isclass(obj) and name.lower().endswith('args') \
                    and name.lower() not in (('faceswapargs',
                                              'extractconvertargs',
                                              'guiargs')):
                mod_classes.append(name)
        return mod_classes

    def commands_sorted(self, category, classes):
        """ Format classes into command names and sort:
            Specific workflow order for faceswap.
            Alphabetical for all others """
        commands = sorted(self.format_command(command) for command in classes)
        if category == 'faceswap':
            ordered = ['extract', 'train', 'convert']
            command = ordered + [command for command in commands if command not in ordered]
        return commands

    @staticmethod
    def format_command(classname):
        """ Format args class name to command """
        return classname.lower()[:-4]

    def extract_command_options(self, cli_source, mod_classes):
        """ Extract the existing ArgParse Options into master options Dictionary """
        subopts = dict()
        for classname in mod_classes:
            command = self.format_command(classname)
            options = self.get_cli_arguments(cli_source, classname, command)
            self.process_options(options)
            subopts[command] = options
        return subopts

    @staticmethod
    def get_cli_arguments(cli_source, classname, command):
        """ Extract the options from the main and tools cli files """
        meth = getattr(cli_source, classname)(None, command)
        return meth.argument_list + meth.optional_arguments

    def process_options(self, command_options):
        """ Process the options for a single command """
        for opt in command_options:
            if opt.get("help", "") == SUPPRESS:
                command_options.remove(opt)
            ctl, sysbrowser, filetypes, actions_open_types = self.set_control(opt)
            opt['control_title'] = self.set_control_title(
                opt.get('opts', ''))
            opt['control'] = ctl
            opt['filesystem_browser'] = sysbrowser
            opt['filetypes'] = filetypes
            opt['actions_open_types'] = actions_open_types

    @staticmethod
    def set_control_title(opts):
        """ Take the option switch and format it nicely """
        ctltitle = opts[1] if len(opts) == 2 else opts[0]
        ctltitle = ctltitle.replace("-", " ").replace("_", " ").strip().title()
        return ctltitle

    @staticmethod
    def set_control(option):
        """ Set the control and filesystem browser to use for each option """
        sysbrowser = None
        filetypes = None
        actions_open_type = None
        ctl = ttk.Entry
        if option.get('action', '') == cli.FullPaths:
            sysbrowser = 'folder'
        elif option.get('action', '') == cli.DirFullPaths:
            sysbrowser = 'folder'
        elif option.get('action', '') == cli.FileFullPaths:
            sysbrowser = 'load'
            filetypes = option.get('filetypes', None)
        elif option.get('action', '') == cli.ComboFullPaths:
            sysbrowser = 'combo'
            actions_open_type = option['actions_open_type']
            filetypes = option.get('filetypes', None)
        elif option.get('choices', '') != '':
            ctl = ttk.Combobox
        elif option.get("action", "") == "store_true":
            ctl = ttk.Checkbutton
        return ctl, sysbrowser, filetypes, actions_open_type


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

class Config(object):
    """ Actions for controlling Faceswap GUI command configurations """

    def __init__(self, opts):
        self.opts = self.flatten_opts(opts)
        self.serializer = JSONSerializer
        self.filetypes = (('Faceswap files', '*.fsw'), ('All files', '*.*'))

    @staticmethod
    def flatten_opts(full_opts):
        """ This is to maintain backwards compatibility.
            when saving/loading full configs.
            It is somewhat hacky, as it assumes that no
            commands between tools and faceswap will ever
            be named the same """
        if not any(key in full_opts for key in ('tools', 'faceswap')):
            return full_opts
        flat_opts = {cmd: opts for value in full_opts.values() for cmd, opts in value.items()}
        return flat_opts

    def set_command_args(self, command, options):
        """ Pass the saved config items back to the GUI """
        for srcopt, srcval in options.items():
            for dstopts in self.opts[command]:
                if dstopts['control_title'] == srcopt:
                    dstopts['value'].set(srcval)
                    break

    def load(self, command=None):
        """ Load a saved config file """
        cfgfile = FileHandler('open', 'config').retfile
        if not cfgfile:
            return
        cfg = self.serializer.unmarshal(cfgfile.read())
        if command is None:
            for cmd, opts in cfg.items():
                self.set_command_args(cmd, opts)
        else:
            opts = cfg.get(command, None)
            if opts:
                self.set_command_args(command, opts)
            else:
                ConsoleOut().clear()
                print('No ' + command + ' section found in file')

    def save(self, command=None):
        """ Save the current GUI state to a config file in json format """
        cfgfile = FileHandler('save', 'config').retfile
        if not cfgfile:
            return
        if command is None:
            cfg = {cmd: {opt['control_title']: opt['value'].get() for opt in opts}
                   for cmd, opts in self.opts.items()}
        else:
            cfg = {command: {opt['control_title']: opt['value'].get()
                             for opt in self.opts[command]}}
        cfgfile.write(self.serializer.marshal(cfg))
        cfgfile.close()

    def reset(self, command=None):
        """ Reset the GUI to the default values """
        if command is None:
            options = [opt for opts in self.opts.values() for opt in opts]
        else:
            options = [opt for opt in self.opts[command]]
        for option in options:
            default = option.get('default', '')
            default = '' if default is None else default
            option['value'].set(default)

    def clear(self, command=None):
        """ Clear all values from the GUI """
        if command is None:
            options = [opt for opts in self.opts.values() for opt in opts]
        else:
            options = [opt for opt in self.opts[command]]
        for option in options:
            if isinstance(option['value'].get(), bool):
                option['value'].set(False)
            elif isinstance(option['value'].get(), int):
                option['value'].set(0)
            else:
                option['value'].set('')
