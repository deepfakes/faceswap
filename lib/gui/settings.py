#!/usr/bin python3
""" Settings and config functions for the GUI """
import os
import tkinter as tk
from tkinter import filedialog, ttk

from lib.Serializer import JSONSerializer

from .console import ConsoleOut

class Settings(object):
    """ Holds the faceswap GUI settings """

    def __init__(self, pathcache):
        self.serializer = JSONSerializer
        self.settingsfile = os.path.join(pathcache, 'settings.json')

        self.settings = self.load()

    def load(self):
        """ Load settings from disk """
        if not os.path.exists(self.settingsfile):
            self.generate_default()

        with open(self.settingsfile, self.serializer.roptions) as opts:
            return self.serializer.unmarshal(opts.read())

    def save(self, settings):
        """ Save settings to disk """
        with open(self.settingsfile, self.serializer.woptions) as opts:
            opts.write(self.serializer.marshal(settings))

    def generate_default(self):
        """ Generate the default options """
        extractconvert = [{'title': 'Preview output images',
                           'control': 'Checkbutton',
                           'value': True}]
        train = [{'title': 'Show live graph',
                  'control': 'Checkbutton',
                  'value': True},
                 {'title': 'Preview training progress',
                  'control': 'Checkbutton',
                  'value': True}]
        settings = [{'section': 'extract',
                     'title': 'Extract Options',
                     'options': extractconvert},
                    {'section': 'train',
                     'title': 'Training Options',
                     'options': train},
                    {'section': 'convert',
                     'title': 'Convert Options',
                     'options': extractconvert}]

        self.save(settings)

    def popup(self):
        """ Faceswap GUI pop-up settings window """
        settings = tk.Toplevel()
        settings.title("Settings")

        optscontainer = ttk.Frame(settings)
        optscontainer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        actcontainer = ttk.Frame(settings)
        actcontainer.pack(side=tk.BOTTOM, pady=5, fill=tk.X, expand=True)

        self.options(optscontainer)
        self.actions(actcontainer, settings)

    def options(self, frame):
        """ Options for settings """
        for section in self.settings:
            lbl = ttk.LabelFrame(frame, text=section['title'])
            lbl.pack(side=tk.TOP, padx=5, pady=5, fill=tk.BOTH, expand=True)

            for opts in section['options']:
                var = tk.BooleanVar(lbl)
                var.set(opts['value'])
                opts['var'] = var

                control = getattr(ttk, opts['control'])(lbl,
                                                        text=opts['title'],
                                                        variable=opts['var'])
                control.pack(anchor=tk.W)

    def actions(self, frame, settings):
        """ Actions for settings """
        btnapply = ttk.Button(frame,
                              text='Apply',
                              width=10,
                              command=lambda window=settings: self.compile(window, True))
        btnapply.pack(side=tk.RIGHT, padx=5)

        btncancel = ttk.Button(frame,
                               text='Cancel',
                               width=10,
                               command=lambda window=settings: self.compile(window, False))
        btncancel.pack(side=tk.RIGHT, padx=5)

    def compile(self, window, save=False):
        """ Set values for tk var objects, save and close window """
        for section in self.settings:
            for opts in section['options']:
                opts['value'] = opts['var'].get() if save else opts['value']
                del opts['var']

        if save:
            self.save(self.settings)

        window.destroy()

class Config(object):
    """ Actions for controlling Faceswap GUI command configurations """

    def __init__(self, opts):
        self.opts = opts
        self.serializer = JSONSerializer
        self.filetypes = (('Faceswap files', '*.fsw'), ('All files', '*.*'))

    def set_command_args(self, command, options):
        """ Pass the saved config items back to the GUI """
        for srcopt, srcval in options.items():
            for dstopts in self.opts[command]:
                if dstopts['control_title'] == srcopt:
                    dstopts['value'].set(srcval)
                    break

    def load(self, command=None):
        """ Load a saved config file """
        cfgfile = filedialog.askopenfile(mode='r', filetypes=self.filetypes)
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
        cfgfile = filedialog.asksaveasfile(mode='w',
                                           filetypes=self.filetypes,
                                           defaultextension='.fsw')
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
