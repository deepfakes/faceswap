#!/usr/bin python3
""" Display Page parent classes for display section of the Faceswap GUI """

import tkinter as tk
from tkinter import ttk

from .tooltip import Tooltip
from .utils import Images


class DisplayPage(ttk.Frame):
    """ Parent frame holder for each tab.
        Defines uniform structure for each tab to inherit from """
    def __init__(self, parent, tabname, helptext):
        ttk.Frame.__init__(self, parent)
        self.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW)

        self.session = parent.session
        self.runningtask = parent.runningtask
        self.helptext = helptext
        self.tabname = tabname

        self.vars = {"info": tk.StringVar()}
        self.add_optional_vars(self.set_vars())

        self.subnotebook = self.add_subnotebook()
        self.optsframe = self.add_options_frame()
        self.add_options_info()

        self.add_frame_separator()
        self.set_mainframe_single_tab_style()
        parent.add(self, text=self.tabname.title())

    def add_optional_vars(self, varsdict):
        """ Add page specific variables """
        if isinstance(varsdict, dict):
            for key, val in varsdict.items():
                self.vars[key] = val

    @staticmethod
    def set_vars():
        """ Overide to return a dict of page specific variables """
        return dict()

    def add_subnotebook(self):
        """ Add the main frame notebook """
        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.TOP, anchor=tk.NW, fill=tk.BOTH, expand=True)
        return notebook

    def add_options_frame(self):
        """ Add the display tab options """
        optsframe = ttk.Frame(self)
        optsframe.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)
        return optsframe

    def add_options_info(self):
        """ Add the info bar """
        lblinfo = ttk.Label(self.optsframe,
                            textvariable=self.vars["info"],
                            anchor=tk.W,
                            width=70)
        lblinfo.pack(side=tk.LEFT, padx=5, pady=5, anchor=tk.W)

    def set_info(self, msg):
        """ Set the info message """
        self.vars["info"].set(msg)

    def add_frame_separator(self):
        """ Add a separator between top and bottom frames """
        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)

    @staticmethod
    def set_mainframe_single_tab_style():
        """ Configure ttk notebook style to represent a single frame """
        nbstyle = ttk.Style()
        nbstyle.configure("single.TNotebook", borderwidth=0)
        nbstyle.layout("single.TNotebook.Tab", [])

    def subnotebook_add_page(self, tabtitle, widget=None):
        """ Add a page to the sub notebook """
        frame = widget if widget else ttk.Frame(self.subnotebook)
        frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.subnotebook.add(frame, text=tabtitle)
        self.subnotebook_configure()
        return frame

    def subnotebook_configure(self):
        """ Configure notebook to display or hide tabs """
        if len(self.subnotebook.children) == 1:
            self.subnotebook.configure(style="single.TNotebook")
        else:
            self.subnotebook.configure(style="TNotebook")

    def subnotebook_hide(self):
        """ Hide the subnotebook. Used for hiding
            Optional displays """
        if self.subnotebook.winfo_ismapped():
            self.subnotebook.pack_forget()

    def subnotebook_show(self):
        """ Show subnotebook. Used for displaying
            Optional displays  """
        if not self.subnotebook.winfo_ismapped():
            self.subnotebook.pack(side=tk.TOP,
                                  anchor=tk.NW,
                                  fill=tk.BOTH,
                                  expand=True)

    def subnotebook_get_widgets(self):
        """ Return each widget that sits within each
            subnotebook frame """
        for child in self.subnotebook.winfo_children():
            for widget in child.winfo_children():
                yield widget

    def subnotebook_get_titles_ids(self):
        """ Return tabs ids and titles """
        tabs = dict()
        for tab_id in range(0, self.subnotebook.index("end")):
            tabs[self.subnotebook.tab(tab_id, "text")] = tab_id
        return tabs

    def subnotebook_page_from_id(self, tab_id):
        """ Return subnotebook tab widget from it's ID """
        tab_name = self.subnotebook.tabs()[tab_id].split(".")[-1]
        return self.subnotebook.children[tab_name]


class DisplayOptionalPage(DisplayPage):
    """ Parent Context Sensitive Display Tab """

    def __init__(self, parent, tabname, helptext, waittime):
        DisplayPage.__init__(self, parent, tabname, helptext)

        self.display_item = None

        self.set_info_text()
        self.add_options()
        parent.select(self)

        self.update_idletasks()
        self.update_page(waittime)

    @staticmethod
    def set_vars():
        """ Analysis specific vars """
        enabled = tk.BooleanVar()
        enabled.set(True)

        ready = tk.BooleanVar()
        ready.set(False)

        modified = tk.DoubleVar()
        modified.set(None)

        return {"enabled": enabled,
                "ready": ready,
                "modified": modified}

    # INFO LABEL
    def set_info_text(self):
        """ Set waiting for display text """
        if not self.vars["enabled"].get():
            self.set_info("{} disabled".format(self.tabname.title()))
        elif self.vars["enabled"].get() and not self.vars["ready"].get():
            self.set_info("Waiting for {}...".format(self.tabname))
        else:
            self.set_info("Displaying {}".format(self.tabname))

    # DISPLAY OPTIONS BAR
    def add_options(self):
        """ Add the additional options """
        self.add_option_save()
        self.add_option_enable()

    def add_option_save(self):
        """ Add save button to save page output to file """
        btnsave = ttk.Button(self.optsframe,
                             image=Images().icons["save"],
                             command=self.save_items)
        btnsave.pack(padx=2, side=tk.RIGHT)
        Tooltip(btnsave,
                text="Save {}(s) to file".format(self.tabname),
                wraplength=200)

    def add_option_enable(self):
        """ Add checkbutton to enable/disable page """
        chkenable = ttk.Checkbutton(self.optsframe,
                                    variable=self.vars["enabled"],
                                    text="Enable {}".format(self.tabname),
                                    command=self.on_chkenable_change)
        chkenable.pack(side=tk.RIGHT, padx=5, anchor=tk.W)
        Tooltip(chkenable,
                text="Enable or disable {} display".format(self.tabname),
                wraplength=200)

    def save_items(self):
        """ Save items. Override for display specific saving """
        raise NotImplementedError()

    def on_chkenable_change(self):
        """ Update the display immediately on a checkbutton change """
        if self.vars["enabled"].get():
            self.subnotebook_show()
        else:
            self.subnotebook_hide()
        self.set_info_text()

    def update_page(self, waittime):
        """ Update the latest preview item """
        if not self.runningtask.get():
            return
        if self.vars["enabled"].get():
            self.display_item_set()
            self.load_display()
        self.after(waittime, lambda t=waittime: self.update_page(t))

    def display_item_set(self):
        """ Override for display specific loading """
        raise NotImplementedError()

    def load_display(self):
        """ Load the display """
        if not self.display_item:
            return
        self.display_item_process()
        self.vars["ready"].set(True)
        self.set_info_text()

    def display_item_process(self):
        """ Override for display specific loading """
        raise NotImplementedError()
