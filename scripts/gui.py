import sys
from threading import Thread
import tkinter as tk

from tkinter import ttk
from tkinter import filedialog

from lib.cli import FullPaths
from lib.Serializer import JSONSerializer

class TKGui(object):
    ''' The Graphical User Interface '''
    def __init__(self, subparser, subparsers, parser, command, description='default'):
        self.root = tk.Tk()
        self.parser = parser
        self.opts = self.extract_options(subparsers)
        self.icofolder = tk.PhotoImage(file='icons/open_folder.png')
        self.icofile = tk.PhotoImage(file='icons/open_file.png')
        self.icosave = tk.PhotoImage(file='icons/save.png')
        self.icoreset = tk.PhotoImage(file='icons/reset.png')
        self.icoclear = tk.PhotoImage(file='icons/clear.png')
        self.helptext = tk.StringVar()
        self.statustext = tk.StringVar()
        self.serializer = JSONSerializer
        self.filetypes=(('Faceswap files', '*.fsw'),  ('All files', '*.*'))
       
        self.parse_arguments(description, subparser, command)

    def bad_args(self, args):
        self.parser.print_help()
        exit(0)

    def extract_options(self, subparsers):
        ''' Extract the existing ArgParse Options '''
        options = {command: subparsers[command].argument_list + subparsers[command].optional_arguments for command in subparsers.keys()}
        for command in options.values():
            for option in command:
                option['control_title'] = self.set_control_title(option.get('opts',''))
                option['control_type'] = self.set_control_type(option)
        return options

    @staticmethod
    def set_control_title(opts):
        ''' Take the option switch and format it nicely '''
        ctl_title = opts[0]
        if len(opts) == 2:
            ctl_title = opts[1]
        ctl_title = ctl_title.replace('-',' ').replace('_',' ')
        ctl_title = ctl_title.title().strip()
        return ctl_title
 
    @staticmethod
    def set_control_type(option):
        ''' Set what control type we should use for an option based on existence of various variables '''
        if option.get('dest', '') == 'alignments_path':
            ctl_type = 'filechooser'
        elif option.get('action', '') == FullPaths:
            ctl_type = 'folderchooser'
        elif option.get('choices', '') != '':
            ctl_type = 'combobox'
        elif option.get('type', '') in (str, float, int):
            ctl_type = 'entrybox'
        elif option.get('action', '') == 'store_true':
            ctl_type = 'checkbox'
        return ctl_type

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
        self.root.title('faceswap.py')
        self.menu()
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        # extract, train and convert are explicitly stated to ensure they are always displayed in the same order
        for command in ('extract', 'train', 'convert'):
            title = command.title()
            page = ttk.Frame(notebook)
            
            self.add_left_frame(page, command)
            self.add_frame_seperator(page)
            opt_frame = self.add_right_frame(page)
            
            for option in self.opts[command]:
                self.build_tabs(option, opt_frame)
            
            notebook.add(page, text=title)

        self.root.mainloop()

# All pages stuff
    def menu(self):
        ''' Menu bar for loading and saving configs '''
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label='Load full config...', command=self.load_config)
        filemenu.add_command(label='Save full config...', command=self.save_config)
        filemenu.add_separator()
        filemenu.add_command(label='Reset all to default', command=self.reset_config)
        filemenu.add_command(label='Clear all', command=self.clear_config)
        filemenu.add_separator()
        filemenu.add_command(label='Quit', command=self.root.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        self.root.config(menu=menubar)

    def load_config(self, command=None):
        ''' Load a saved config file '''
        config_file = filedialog.askopenfile(mode='r', filetypes=self.filetypes)
        if not config_file:
            return
        config = self.serializer.unmarshal(config_file.read())
        if command is None:
            for command, options in config.items():
                self.set_command_args(command, options)
        else:
            options = config[command]
            self.set_command_args(command, options)
                
    def set_command_args(self, command, options):
        ''' Pass the saved config items back to the GUI '''
        for src_option, src_value in options.items():
            for dst_options in self.opts[command]:
                if dst_options['control_title'] == src_option:
                    dst_options['value'].set(src_value)
                    break
        
    def save_config(self, command=None):
        ''' Save the current GUI state to a config file in json format '''
        config_file = filedialog.asksaveasfile(mode='w', filetypes=self.filetypes, defaultextension='.fsw')
        if not config_file:
            return
        if command is None:
            config = {command: {option['control_title']: option['value'].get() for option in options} for command, options in self.opts.items()}
        else:
            config = {command: {option['control_title']: option['value'].get() for option in self.opts[command]}}
        config_file.write(self.serializer.marshal(config))
        config_file.close

    def reset_config(self, command=None):
        ''' Reset the GUI to the default values '''
        if command is None:
            options = [option for options in self.opts.values() for option in options]
        else:
            options = [option for option in self.opts[command]]
        for option in options:
            default = option.get('default', '')
            default = '' if default is None else default
            option['value'].set(default)

    def clear_config(self, command=None):
        ''' Clear all values from the GUI '''
        if command is None:
            options = [option for options in self.opts.values() for option in options]
        else:
            options = [option for option in self.opts[command]]
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
        separator = tk.Frame(page, width=2, bd=1, relief=tk.SUNKEN)
        separator.pack(fill=tk.Y, padx=5, side=tk.LEFT)

# Left Frame stuff
    def add_left_frame(self, page, title):
        ''' Add help display and execute button to the left frame of each page '''
        frame = tk.Frame(page)
        frame.pack(fill=tk.X, padx=(10,5), side=tk.LEFT, anchor=tk.N)

        topframe = tk.Frame(frame)
        topframe.pack(fill=tk.X, side=tk.TOP)
        bottomframe = tk.Frame(frame)
        bottomframe.pack(fill=tk.X, side=tk.BOTTOM)
        self.add_info_section(topframe)
        self.add_action_buttons(topframe, title)
        self.add_util_buttons(bottomframe, title)
        self.add_status_section(bottomframe)
        
    def add_info_section(self, frame):
        ''' Build the info text section page '''
        helpframe=tk.Frame(frame)
        helpframe.pack(fill=tk.X, side=tk.TOP, pady=5)
        lbl_title = tk.Label(helpframe, text='Info', width=15, anchor=tk.SW)
        lbl_title.pack(side=tk.TOP)
        self.helptext.set('')
        lbl_help = tk.Label(helpframe, height=20, width=15, textvariable=self.helptext,  wraplength=120, justify=tk.LEFT, anchor=tk.NW, bg="gray90")
        lbl_help.pack(side=tk.BOTTOM, anchor=tk.N)

    def bind_help(self, control, helptext):
        ''' Controls the help text displayed on mouse hover '''
        control.bind('<Enter>', lambda event: self.helptext.set(helptext))
        control.bind('<FocusIn>', lambda event: self.helptext.set(helptext))
        control.bind('<Leave>', lambda event: self.helptext.set(''))
        control.bind('<FocusOut>', lambda event: self.helptext.set(''))

    def add_action_buttons(self, frame, title):
        ''' Add the action buttons for page '''
        command = title.lower()
        title = title.capitalize()

        actframe = tk.Frame(frame)
        actframe.pack(fill=tk.X, side=tk.BOTTOM, pady=(15, 0))

        btnexecute = tk.Button(actframe, text=title, height=2, width=12, command=lambda: self.execute_script(command))
        btnexecute.pack()
        self.bind_help(btnexecute, 'Run the {} script'.format(title))

    def add_util_buttons(self, frame, title):
        ''' Add the section utility buttons '''
        command = title.lower()
        title = title.capitalize()

        utilframe = tk.Frame(frame)
        utilframe.pack(side=tk.TOP, pady=(5,0))

        btnload = tk.Button(utilframe, image=self.icofile, height=16, width=16, command=lambda: self.load_config(command))
        btnload.pack(padx=2, pady=2, side=tk.LEFT)
        self.bind_help(btnload, 'Load existing {} config'.format(title))

        btnsave = tk.Button(utilframe, image=self.icosave, height=16, width=16, command=lambda: self.save_config(command))
        btnsave.pack(padx=2, pady=2, side=tk.LEFT)
        self.bind_help(btnsave, 'Save {} config'.format(title))

        btnreset = tk.Button(utilframe, image=self.icoreset, height=16, width=16, command=lambda: self.reset_config(command))
        btnreset.pack(padx=2, pady=2, side=tk.RIGHT)
        self.bind_help(btnreset, 'Reset {} config to default'.format(title))

        btnclear = tk.Button(utilframe, image=self.icoclear, height=16, width=16, command=lambda: self.clear_config(command))
        btnclear.pack(padx=2, pady=2, side=tk.RIGHT)
        self.bind_help(btnclear, 'Clear all entries for {} config'.format(title))

    def add_status_section(self, frame):
        ''' Build the info text section page '''
        statusframe = tk.Frame(frame)
        statusframe.pack(side=tk.BOTTOM, pady=(5,0))
        
        lbl_title = tk.Label(statusframe, text='Status', width=15, anchor=tk.SW)
        lbl_title.pack(side=tk.TOP)
        self.statustext.set('Idle')
        lbl_status = tk.Label(statusframe, height=1, width=15, textvariable=self.statustext,  wraplength=120, justify=tk.LEFT, anchor=tk.NW, bg="gray90")
        lbl_status.pack(side=tk.BOTTOM, anchor=tk.N)

    def execute_script(self, command):
        
        optlist = ['faceswap.py', command]
        for item in self.opts[command]:
            opt_value = str(item.get('value','').get())
            opt = item['opts'][0]
            if opt_value == 'False' or opt_value == '':
                continue
            elif opt_value == 'True':
                optlist.append(opt)
            else:
                optlist.extend((opt, opt_value))
        sys.argv = optlist
        process = Thread(target=self.launch_thread, args=(command,))
        process.start()

    def launch_thread(self, command):
        ''' Launch the script inside a subprocess to keep the GUI active '''
        title = command.capitalize()
        self.statustext.set('Running - {}'.format(title))
        self.parser.set_defaults(func=self.bad_args)
        arguments = self.parser.parse_args()
        arguments.func(arguments)
        self.statustext.set('Idle')
        
# Right Frame setup    
    def add_right_frame(self, page):
        ''' Add the options panel to the right frame of each page '''
        frame = tk.Frame(page)
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0,5))

        canvas = tk.Canvas(frame, width=490, height=450, bd=0, highlightthickness=0)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.add_scrollbar(frame, canvas)

        opts_frame = tk.Frame(canvas)
        canvas.create_window((0,0), window=opts_frame, anchor=tk.NW)

        return opts_frame

    def add_scrollbar(self, frame, canvas):
        ''' Add a scrollbar to the options frame '''
        scrollbar = tk.Scrollbar(frame, command=canvas.yview)
        scrollbar.pack(side=tk.LEFT, fill='y')
        canvas.configure(yscrollcommand = scrollbar.set)
        canvas.bind('<Configure>', lambda event, opt_canvas=canvas: self.update_scrollbar(event, opt_canvas))

    @staticmethod
    def update_scrollbar(event, canvas):
        canvas.configure(scrollregion=canvas.bbox('all'))

# Build the Right Frame Options
    def build_tabs(self, option, page):
        ''' Build the correct control type for the option passed through '''
        ctl_type = option['control_type']
        ctl_title = option['control_title']
        ctl_help = ' '.join(option.get('help', '').split())
        ctl_help = '. '.join(i.capitalize() for i in ctl_help.split('. '))
        ctl_help = ctl_title + ' - ' + ctl_help
        ctl_frame = self.build_control_frame(page)
        self.build_control_label(ctl_frame, ctl_title)
        if ctl_type == 'combobox':
            option['value'] = self.build_combobox(ctl_frame, ctl_help, option.get('default',''), option['choices'])
        if ctl_type == 'checkbox':
            option['value'] = self.build_checkbox(ctl_frame, ctl_help, option.get('default', False), ctl_title)
        if ctl_type == 'entrybox':
            option['value'] = self.build_entrybox(ctl_frame, ctl_help, option.get('default',''), None)
        if ctl_type in ('filechooser', 'folderchooser'):
            option['value'] = self.build_entrybox(ctl_frame, ctl_help, option.get('default',''), ctl_type)

    @staticmethod
    def build_control_frame(page):
        ''' Build the frame to hold the control '''
        frame = tk.Frame(page)
        frame.pack(fill=tk.X)
        return frame
    
    @staticmethod
    def build_control_label(frame, title):
        ''' Build and place the control label '''
        label = tk.Label(frame, text=title, width=15, anchor=tk.W)
        label.pack(padx=5, pady=5, side=tk.LEFT, anchor=tk.N)

    def build_combobox(self, frame, helptext, default, items):
        ''' Build and place combobox controls '''
        default = default if default in items else ''
        cmbvar = tk.StringVar(frame)
        cmbvar.set(default)
                        
        cmbbox = ttk.Combobox(frame, textvariable=cmbvar, width=40)
        cmbbox['values'] = [item for item in items]
        cmbbox.pack(fill=tk.X, padx=5, pady=5)

        self.bind_help(cmbbox, helptext)
        return cmbvar

    def build_checkbox(self, frame, helptext, default, title):
        ''' Build and place checkbox controls '''
        chkvar = tk.BooleanVar(frame)
        chkvar.set(default)

        chkbox = tk.Checkbutton(frame, variable=chkvar)
        chkbox.pack(padx=5, pady=5, anchor=tk.W)
        
        self.bind_help(chkbox, helptext)
        return chkvar

    def build_entrybox(self, frame, helptext, default, opentype = None):
        ''' Build and place entry controls '''
        default = default if default is not None else ''
        etyvar = tk.StringVar(frame)
        etyvar.set(default)

        if opentype is not None:
            self.add_browser_buttons(frame, opentype, etyvar)

        etybox = tk.Entry(frame, textvariable=etyvar)
        etybox.pack(fill=tk.X, padx=5, pady=5)

        self.bind_help(etybox, helptext)
        return etyvar

    def add_browser_buttons(self, frame, opentype, filepath):
        ''' Add correct file browser button for control '''
        if opentype == 'filechooser':
            fileopn = tk.Button(frame, image=self.icofile, command=lambda: self.askfile(filepath), width=16, height=16)
        else:
            fileopn = tk.Button(frame, image=self.icofolder, command=lambda: self.askdirectory(filepath))
        fileopn.pack(side=tk.RIGHT)

    @staticmethod
    def askdirectory(filepath):
        ''' Pop-up to get path to a directory '''
        dirname = filedialog.askdirectory()
        if dirname:
            filepath.set(dirname)
   
    @staticmethod
    def askfile(filepath):
        ''' Pop-up to get path to a file '''
        filename = filedialog.askopenfilename()
        if filename:
            filepath.set(filename)
