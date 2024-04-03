#!/usr/bin python3
""" Display Page parent classes for display section of the Faceswap GUI """

import gettext
import logging
import tkinter as tk
from tkinter import ttk

from .custom_widgets import Tooltip
from .utils import get_images

logger = logging.getLogger(__name__)

# LOCALES
_LANG = gettext.translation("gui.tooltips", localedir="locales", fallback=True)
_ = _LANG.gettext


class DisplayPage(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Parent frame holder for each tab.
        Defines uniform structure for each tab to inherit from """
    def __init__(self, parent, tab_name, helptext):
        super().__init__(parent)

        self._parent = parent
        self.running_task = parent.running_task
        self.helptext = helptext
        self.tabname = tab_name

        self.vars = {"info": tk.StringVar()}
        self.add_optional_vars(self.set_vars())

        self.subnotebook = self.add_subnotebook()
        self.optsframe = self.add_options_frame()
        self.add_options_info()

        self.add_frame_separator()
        self.set_mainframe_single_tab_style()

        self.pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.NW)
        parent.add(self, text=self.tabname.title())

    @property
    def _tab_is_active(self):
        """ bool: ``True`` if the tab currently has focus otherwise ``False`` """
        return self._parent.tab(self._parent.select(), "text").lower() == self.tabname.lower()

    def add_optional_vars(self, varsdict):
        """ Add page specific variables """
        if isinstance(varsdict, dict):
            for key, val in varsdict.items():
                logger.debug("Adding: (%s: %s)", key, val)
                self.vars[key] = val

    def set_vars(self):
        """ Override to return a dict of page specific variables """
        return {}

    def on_tab_select(self):
        """ Override for specific actions when the current tab is selected """
        logger.debug("Returning as 'on_tab_select' not implemented for %s",
                     self.__class__.__name__)

    def add_subnotebook(self):
        """ Add the main frame notebook """
        logger.debug("Adding subnotebook")
        notebook = ttk.Notebook(self)
        notebook.pack(side=tk.TOP, anchor=tk.NW, fill=tk.BOTH, expand=True)
        return notebook

    def add_options_frame(self):
        """ Add the display tab options """
        logger.debug("Adding options frame")
        optsframe = ttk.Frame(self)
        optsframe.pack(side=tk.BOTTOM, padx=5, pady=5, fill=tk.X)
        return optsframe

    def add_options_info(self):
        """ Add the info bar """
        logger.debug("Adding options info")
        lblinfo = ttk.Label(self.optsframe,
                            textvariable=self.vars["info"],
                            anchor=tk.W)
        lblinfo.pack(side=tk.LEFT, expand=True, padx=5, pady=5, anchor=tk.W)

    def set_info(self, msg):
        """ Set the info message """
        logger.debug("Setting info: %s", msg)
        self.vars["info"].set(msg)

    def add_frame_separator(self):
        """ Add a separator between top and bottom frames """
        logger.debug("Adding frame seperator")
        sep = ttk.Frame(self, height=2, relief=tk.RIDGE)
        sep.pack(fill=tk.X, pady=(5, 0), side=tk.BOTTOM)

    @staticmethod
    def set_mainframe_single_tab_style():
        """ Configure ttk notebook style to represent a single frame """
        logger.debug("Setting main frame single tab style")
        nbstyle = ttk.Style()
        nbstyle.configure("single.TNotebook", borderwidth=0)
        nbstyle.layout("single.TNotebook.Tab", [])

    def subnotebook_add_page(self, tabtitle, widget=None):
        """ Add a page to the sub notebook """
        logger.debug("Adding subnotebook page: %s", tabtitle)
        frame = widget if widget else ttk.Frame(self.subnotebook)
        frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.subnotebook.add(frame, text=tabtitle)
        self.subnotebook_configure()
        return frame

    def subnotebook_configure(self):
        """ Configure notebook to display or hide tabs """
        if len(self.subnotebook.children) == 1:
            logger.debug("Setting single page style")
            self.subnotebook.configure(style="single.TNotebook")
        else:
            logger.debug("Setting multi page style")
            self.subnotebook.configure(style="TNotebook")

    def subnotebook_hide(self):
        """ Hide the subnotebook. Used for hiding
            Optional displays """
        if self.subnotebook and self.subnotebook.winfo_ismapped():
            logger.debug("Hiding subnotebook")
            self.subnotebook.pack_forget()
            self.subnotebook.destroy()
            self.subnotebook = None

    def subnotebook_show(self):
        """ Show subnotebook. Used for displaying
            Optional displays  """
        if not self.subnotebook:
            logger.debug("Showing subnotebook")
            self.subnotebook = self.add_subnotebook()

    def subnotebook_get_widgets(self):
        """ Return each widget that sits within each
            subnotebook frame """
        logger.debug("Getting subnotebook widgets")
        for child in self.subnotebook.winfo_children():
            for widget in child.winfo_children():
                yield widget

    def subnotebook_get_titles_ids(self):
        """ Return tabs ids and titles """
        tabs = {}
        for tab_id in range(0, self.subnotebook.index("end")):
            tabs[self.subnotebook.tab(tab_id, "text")] = tab_id
        logger.debug(tabs)
        return tabs

    def subnotebook_page_from_id(self, tab_id):
        """ Return subnotebook tab widget from it's ID """
        tab_name = self.subnotebook.tabs()[tab_id].split(".")[-1]
        logger.debug(tab_name)
        return self.subnotebook.children[tab_name]


class DisplayOptionalPage(DisplayPage):  # pylint:disable=too-many-ancestors
    """ Parent Context Sensitive Display Tab """

    def __init__(self, parent, tab_name, helptext, wait_time, command=None):
        super().__init__(parent, tab_name, helptext)

        self._waittime = wait_time
        self.command = command
        self.display_item = None

        self.set_info_text()
        self.add_options()
        parent.select(self)

        self.update_idletasks()
        self._update_page()

    def set_vars(self):
        """ Analysis specific vars """
        enabled = tk.BooleanVar()
        enabled.set(True)

        ready = tk.BooleanVar()
        ready.set(False)

        tk_vars = {"enabled": enabled,
                   "ready": ready}
        logger.debug(tk_vars)
        return tk_vars

    def on_tab_select(self):
        """ Callback for when the optional tab is selected.

        Run the tab's update code when the tab is selected.
        """
        logger.debug("Callback received for '%s' tab", self.tabname)
        self._update_page()

    # INFO LABEL
    def set_info_text(self):
        """ Set waiting for display text """
        if not self.vars["enabled"].get():
            msg = f"{self.tabname.title()} disabled"
        elif self.vars["enabled"].get() and not self.vars["ready"].get():
            msg = f"Waiting for {self.tabname}..."
        else:
            msg = f"Displaying {self.tabname}"
        logger.debug(msg)
        self.set_info(msg)

    # DISPLAY OPTIONS BAR
    def add_options(self):
        """ Add the additional options """
        self.add_option_save()
        self.add_option_enable()

    def add_option_save(self):
        """ Add save button to save page output to file """
        logger.debug("Adding save option")
        btnsave = ttk.Button(self.optsframe,
                             image=get_images().icons["save"],
                             command=self.save_items)
        btnsave.pack(padx=2, side=tk.RIGHT)
        Tooltip(btnsave,
                text=_(f"Save {self.tabname}(s) to file"),
                wrap_length=200)

    def add_option_enable(self):
        """ Add check-button to enable/disable page """
        logger.debug("Adding enable option")
        chkenable = ttk.Checkbutton(self.optsframe,
                                    variable=self.vars["enabled"],
                                    text=f"Enable {self.tabname}",
                                    command=self.on_chkenable_change)
        chkenable.pack(side=tk.RIGHT, padx=5, anchor=tk.W)
        Tooltip(chkenable,
                text=_(f"Enable or disable {self.tabname} display"),
                wrap_length=200)

    def save_items(self):
        """ Save items. Override for display specific saving """
        raise NotImplementedError()

    def on_chkenable_change(self):
        """ Update the display immediately on a check-button change """
        logger.debug("Enabled checkbox changed")
        if self.vars["enabled"].get():
            self.subnotebook_show()
        else:
            self.subnotebook_hide()
        self.set_info_text()

    def _update_page(self):
        """ Update the latest preview item """
        if not self.running_task.get() or not self._tab_is_active:
            return
        if self.vars["enabled"].get():
            logger.trace("Updating page: %s", self.__class__.__name__)
            self.display_item_set()
            self.load_display()
        self.after(self._waittime, self._update_page)

    def display_item_set(self):
        """ Override for display specific loading """
        raise NotImplementedError()

    def load_display(self):
        """ Load the display """
        if not self.display_item or not self._tab_is_active:
            return
        logger.debug("Loading display for tab: %s", self.tabname)
        self.display_item_process()
        self.vars["ready"].set(True)
        self.set_info_text()

    def display_item_process(self):
        """ Override for display specific loading """
        raise NotImplementedError()

    def close(self):
        """ Called when the parent notebook is shutting down
            Children must be destroyed as forget only hides display
            Override for page specific shutdown """
        for child in self.winfo_children():
            logger.debug("Destroying child: %s", child)
            child.destroy()
