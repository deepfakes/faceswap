import sys
from threading import Thread
import tkinter as tk

from tkinter import ttk
from tkinter import filedialog

from lib.cli import FullPaths
from lib.Serializer import JSONSerializer

class FaceswapGui(tk.Tk):
    ''' The Graphical User Interface '''
    def __init__(self, options, parser, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        self.opts = options
        self.parser = parser
        self.icofolder = tk.PhotoImage(file='icons/open_folder.png')
        self.icoload = tk.PhotoImage(file='icons/open_file.png')
        self.icosave = tk.PhotoImage(file='icons/save.png')
        self.icoreset = tk.PhotoImage(file='icons/reset.png')
        self.icoclear = tk.PhotoImage(file='icons/clear.png')
        self.helptext = tk.StringVar()
        self.statustext = tk.StringVar()
        self.serializer = JSONSerializer
        self.filetypes=(('Faceswap files', '*.fsw'),  ('All files', '*.*'))
        self.task = FaceswapControl()

    def build_gui(self):
        ''' Build the GUI '''
        self.title('faceswap.py')
        self.menu()
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True)

        # Commands explicitly stated to ensure consistent ordering
        for command in ('extract', 'train', 'convert'):
            page = ttk.Frame(notebook)
            
            self.add_left_frame(page, command)
            self.add_frame_seperator(page)
            opt_frame = self.add_right_frame(page)
            
            for option in self.opts[command]:
                self.build_tabs(option, opt_frame)
            
            notebook.add(page, text=command.title())

# All pages stuff
    def menu(self):
        ''' Menu bar for loading and saving configs '''
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Load full config...', command=self.load_config)
        filemenu.add_command(label='Save full config...', command=self.save_config)
        filemenu.add_separator()
        filemenu.add_command(label='Reset all to default', command=self.reset_config)
        filemenu.add_command(label='Clear all', command=self.clear_config)
        filemenu.add_separator()
        filemenu.add_command(label='Quit', command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

    def load_config(self, command=None):
        ''' Load a saved config file '''
        cfgfile = filedialog.askopenfile(mode='r', filetypes=self.filetypes)
        if not cfgfile:
            return
        cfg = self.serializer.unmarshal(cfgfile.read())
        if command is None:
            for cmd, opts in cfg.items():
                self.set_command_args(cmd, opts)
        else:
            opts = cfg[command]
            self.set_command_args(command, opts)
                
    def set_command_args(self, command, options):
        ''' Pass the saved config items back to the GUI '''
        for srcopt, srcval in options.items():
            for dstopts in self.opts[command]:
                if dstopts['control_title'] == srcopt:
                    dstopts['value'].set(srcval)
                    break
        
    def save_config(self, command=None):
        ''' Save the current GUI state to a config file in json format '''
        cfgfile = filedialog.asksaveasfile( mode='w',
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
        cfgfile.close

    def reset_config(self, command=None):
        ''' Reset the GUI to the default values '''
        if command is None:
            options = [opt for opts in self.opts.values() for opt in opts]
        else:
            options = [opt for opt in self.opts[command]]
        for option in options:
            default = option.get('default', '')
            default = '' if default is None else default
            option['value'].set(default)

    def clear_config(self, command=None):
        ''' Clear all values from the GUI '''
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

    @staticmethod
    def add_frame_seperator(page):
        ''' Add a seperator between left and right frames '''
        sep = tk.Frame(page, width=2, bd=1, relief=tk.SUNKEN)
        sep.pack(fill=tk.Y, padx=5, side=tk.LEFT)

# Left Frame stuff
    def add_left_frame(self, page, command):
        ''' Add help display and execute button to the left frame of each page '''
        frame = tk.Frame(page)
        frame.pack(fill=tk.X, padx=(10,5), side=tk.LEFT, anchor=tk.N)

        self.add_info_section(frame)
        self.add_action_buttons(frame, command)
        self.add_util_buttons(frame, command)
        self.add_status_section(frame)
        
    def add_info_section(self, frame):
        ''' Build the info text section page '''
        hlpframe=tk.Frame(frame)
        hlpframe.pack(fill=tk.X, side=tk.TOP, pady=5)
        lbltitle = tk.Label(hlpframe, text='Info', width=15, anchor=tk.SW)
        lbltitle.pack(side=tk.TOP)
        self.helptext.set('')
        lblhelp = tk.Label( hlpframe,
                            height=20,
                            width=15,
                            textvariable=self.helptext,
                            wraplength=120, 
                            justify=tk.LEFT, 
                            anchor=tk.NW,
                            bg="gray90")
        lblhelp.pack(side=tk.TOP, anchor=tk.N)

    def bind_help(self, control, helptext):
        ''' Controls the help text displayed on mouse hover '''
        for action in ('<Enter>', '<FocusIn>', '<Leave>', '<FocusOut>'):
            helptext = helptext if action in ('<Enter>', '<FocusIn>') else ''
            control.bind(action, lambda event, txt=helptext: self.helptext.set(txt))

    def add_action_buttons(self, frame, command):
        ''' Add the action buttons for page '''
        title = command.capitalize()

        actframe = tk.Frame(frame)
        actframe.pack(fill=tk.X, side=tk.TOP, pady=(15, 0))

        btnexecute = tk.Button( actframe,
                                text=title,
                                height=2,
                                width=12,
                                command=lambda: self.execute_task(command))
        btnexecute.pack(side=tk.TOP)
        self.bind_help(btnexecute, 'Run the {} script'.format(title))

    def execute_task(self, command):
        ''' Execute the task in Faceswap.py '''
        self.task.execute_script(self.opts, command, self.parser, self.statustext)

    def add_util_buttons(self, frame, command):
        ''' Add the section utility buttons '''
        utlframe = tk.Frame(frame)
        utlframe.pack(side=tk.TOP, pady=(5,0))

        for utl in ('load', 'save', 'clear', 'reset'):
            img = getattr(self, 'ico' + utl)
            action = getattr(self, utl + '_config')
            btnutl = tk.Button( utlframe,
                                height=16,
                                width=16,
                                image=img,
                                command=lambda cmd=action: cmd(command))
            btnutl.pack(padx=2, pady=2, side=tk.LEFT)
            self.bind_help(btnutl, utl.capitalize() + ' ' + command.capitalize() + ' config')

    def add_status_section(self, frame):
        ''' Build the info text section page '''
        statusframe = tk.Frame(frame)
        statusframe.pack(side=tk.TOP, pady=(5,0))
        
        lbltitle = tk.Label(statusframe, text='Status', width=15, anchor=tk.SW)
        lbltitle.pack(side=tk.TOP)
        self.statustext.set('Idle')
        lblstatus = tk.Label(   statusframe,
                                height=1,
                                width=15,
                                textvariable=self.statustext,
                                wraplength=120,
                                justify=tk.LEFT,
                                anchor=tk.NW,
                                bg="gray90")
        lblstatus.pack(side=tk.BOTTOM, anchor=tk.N)

# Right Frame setup    
    def add_right_frame(self, page):
        ''' Add the options panel to the right frame of each page '''
        frame = tk.Frame(page)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0,5))

        canvas = tk.Canvas(frame, width=490, height=450, bd=0, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.add_scrollbar(frame, canvas)

        optsframe = tk.Frame(canvas)
        canvas.create_window((0,0), window=optsframe, anchor=tk.NW)

        return optsframe

    def add_scrollbar(self, frame, canvas):
        ''' Add a scrollbar to the options frame '''
        scrollbar = tk.Scrollbar(frame, command=canvas.yview)
        scrollbar.pack(side=tk.LEFT, fill='y')
        canvas.configure(yscrollcommand = scrollbar.set)
        canvas.bind('<Configure>',lambda event, cvs=canvas: self.update_scrollbar(event, cvs))

    @staticmethod
    def update_scrollbar(event, canvas):
        canvas.configure(scrollregion=canvas.bbox('all'))

# Build the Right Frame Options
    def build_tabs(self, option, page):
        ''' Build the correct control type for the option passed through '''
        ctl = option['control']
        ctltitle = option['control_title']
        sysbrowser = option['filesystem_browser']
        ctlhelp = ' '.join(option.get('help', '').split())
        ctlhelp = '. '.join(i.capitalize() for i in ctlhelp.split('. '))
        ctlhelp = ctltitle + ' - ' + ctlhelp
        ctlframe = self.build_control_frame(page)
        
        dflt = option.get('default', False) if ctl == tk.Checkbutton else option.get('default', '')
        choices = option['choices'] if ctl == ttk.Combobox else None

        self.build_control_label(ctlframe, ctltitle)
        option['value'] = self.build_control(ctlframe, ctl, dflt, ctlhelp, choices, sysbrowser)

    @staticmethod
    def build_control_frame(page):
        ''' Build the frame to hold the control '''
        frame = tk.Frame(page)
        frame.pack(fill=tk.X)
        return frame
    
    @staticmethod
    def build_control_label(frame, title):
        ''' Build and place the control label '''
        lbl = tk.Label(frame, text=title, width=15, anchor=tk.W)
        lbl.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

    def build_control(self, frame, control, default, helptext, choices, sysbrowser):
        ''' Build and place the option controls '''
        default = default if default is not None else ''

        var = tk.BooleanVar(frame) if control == tk.Checkbutton else tk.StringVar(frame)
        var.set(default)

        if sysbrowser is not None:
            self.add_browser_buttons(frame, sysbrowser, var)

        ctlkwargs = {'variable': var} if control == tk.Checkbutton else {'textvariable': var}
        packkwargs = {'anchor': tk.W} if control == tk.Checkbutton else {'fill': tk.X}

        if control == ttk.Combobox: #TODO: Remove this hacky fix to force the width of the frame
            ctlkwargs['width'] = 40

        ctl = control(frame, **ctlkwargs)
        
        if control == ttk.Combobox:
            ctl['values'] = [choice for choice in choices]
        
        ctl.pack(padx=5, pady=5, **packkwargs)

        self.bind_help(ctl, helptext)
        return(var)

    def add_browser_buttons(self, frame, sysbrowser, filepath):
        ''' Add correct file browser button for control '''
        img = getattr(self, 'ico' + sysbrowser)
        action = getattr(self, 'ask_' + sysbrowser)
        fileopn = tk.Button(frame, image=img, command=lambda cmd=action: cmd(filepath))
        fileopn.pack(side=tk.RIGHT)

    @staticmethod
    def ask_folder(filepath):
        ''' Pop-up to get path to a folder '''
        dirname = filedialog.askdirectory()
        if dirname:
            filepath.set(dirname)
   
    @staticmethod
    def ask_load(filepath):
        ''' Pop-up to get path to a file '''
        filename = filedialog.askopenfilename()
        if filename:
            filepath.set(filename)

class TKGui(object):
    ''' Main GUI Control '''
    def __init__ (self, subparser, subparsers, parser, command, description='default'):
        self.parser = parser
        self.opts = self.extract_options(subparsers)
        self.root = FaceswapGui(self.opts, self.parser)
        self.parse_arguments(description, subparser, command)

    def extract_options(self, subparsers):
        ''' Extract the existing ArgParse Options '''
        opts = {cmd: subparsers[cmd].argument_list + 
                subparsers[cmd].optional_arguments for cmd in subparsers.keys()}
        for command in opts.values():
            for opt in command:
                ctl, sysbrowser = self.set_control(opt)
                opt['control_title'] = self.set_control_title(opt.get('opts',''))
                opt['control'] = ctl
                opt['filesystem_browser'] = sysbrowser
        return opts

    @staticmethod
    def set_control_title(opts):
        ''' Take the option switch and format it nicely '''
        ctltitle = opts[1] if len(opts) == 2 else opts[0]
        ctltitle = ctltitle.replace('-',' ').replace('_',' ').strip().title()
        return ctltitle
 
    @staticmethod
    def set_control(option):
        ''' Set the control and filesystem browser to use for each option '''
        sysbrowser = None
        ctl = tk.Entry
        if option.get('dest', '') == 'alignments_path':
            sysbrowser = 'load'
        elif option.get('action', '') == FullPaths:
            sysbrowser = 'folder'
        elif option.get('choices', '') != '':
            ctl = ttk.Combobox
        elif option.get('action', '') == 'store_true':
            ctl = tk.Checkbutton
        return ctl, sysbrowser

    def parse_arguments(self, description, subparser, command):
        parser = subparser.add_parser(
            command,
            help="This Launches a GUI for Faceswap.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )
        parser.set_defaults(func=self.process)        

    def process(self, arguments):
        ''' Builds the GUI '''
        self.arguments = arguments
        self.root.build_gui()
        self.root.mainloop()

class FaceswapControl(object):
    ''' Control the underlying Faceswap tasks '''

    def __init__(self):
        self.opts = None
        self.command = None
        self.parser = None
        self.statustext = None

    def bad_args(self, args):
        self.parser.print_help()
        exit(0)

    def execute_script(self, options, command, parser, statustext):
        self.opts = options
        self.command = command
        self.parser = parser
        self.statustext = statustext
        
        optlist = ['faceswap.py', self.command]
        for item in self.opts[self.command]:
            optval = str(item.get('value','').get())
            opt = item['opts'][0]
            if optval == 'False' or optval == '':
                continue
            elif optval == 'True':
                optlist.append(opt)
            else:
                optlist.extend((opt, optval))
        sys.argv = optlist
        process = Thread(target=self.launch_thread, args=(self.command,))
        process.start()

    def launch_thread(self, command):
        ''' Launch the script inside a subprocess to keep the GUI active '''
        self.statustext.set('Running - ' + command.title())
        self.parser.set_defaults(func=self.bad_args)
        arguments = self.parser.parse_args()
        arguments.func(arguments)
        self.statustext.set('Idle')        
