#!/usr/bin python3
""" Config functions for the GUI """


from lib.Serializer import JSONSerializer

from .console import ConsoleOut
from .utils import FileHandler

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
