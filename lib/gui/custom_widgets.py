#!/usr/bin/env python3
""" Custom widgets for Faceswap GUI """

import logging
import platform
import re
import sys
import typing as T
import tkinter as tk
from tkinter import ttk, TclError

import numpy as np

from .utils import get_config

logger = logging.getLogger(__name__)


class ContextMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ A Pop up menu to be triggered when right clicking on widgets that this menu has been
    applied to.

    This widget provides a simple right click pop up menu to the widget passed in with `Cut`,
    `Copy`, `Paste` and `Select all` menu items.

    Parameters
    ----------
    widget: tkinter object
        The widget to apply the :class:`ContextMenu` to

    Example
    -------
    >>> text_box = ttk.Entry(parent)
    >>> text_box.pack()
    >>> right_click_menu = ContextMenu(text_box)
    >>> right_click_menu.cm_bind()
    """
    def __init__(self, widget):
        logger.debug("Initializing %s: (widget_class: '%s')",
                     self.__class__.__name__, widget.winfo_class())
        super().__init__(tearoff=0)
        self._widget = widget
        self._standard_actions()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _standard_actions(self):
        """ Standard menu actions """
        self.add_command(label="Cut", command=lambda: self._widget.event_generate("<<Cut>>"))
        self.add_command(label="Copy", command=lambda: self._widget.event_generate("<<Copy>>"))
        self.add_command(label="Paste", command=lambda: self._widget.event_generate("<<Paste>>"))
        self.add_separator()
        self.add_command(label="Select all", command=self._select_all)

    def cm_bind(self):
        """ Bind the menu to the given widgets Right Click event

        After associating a widget with this :class:`ContextMenu` this function should be called
        to bind it to the right click button
        """
        button = "<Button-2>" if platform.system() == "Darwin" else "<Button-3>"
        logger.debug("Binding '%s' to '%s'", button, self._widget.winfo_class())
        self._widget.bind(button, lambda event: self.tk_popup(event.x_root, event.y_root))

    def _select_all(self):
        """ Select all for Text or Entry widgets """
        logger.debug("Selecting all for '%s'", self._widget.winfo_class())
        if self._widget.winfo_class() == "Text":
            self._widget.focus_force()
            self._widget.tag_add("sel", "1.0", "end")
        else:
            self._widget.focus_force()
            self._widget.select_range(0, tk.END)


class RightClickMenu(tk.Menu):  # pylint:disable=too-many-ancestors
    """ A Pop up menu that can be bound to a right click mouse event to bring up a context menu

    Parameters
    ----------
    labels: list
        A list of label titles that will appear in the right click menu
    actions: list
        A list of python functions that are called when the corresponding label is clicked on
    hotkeys: list, optional
        The hotkeys corresponding to the labels. If using hotkeys, then there must be an entry in
        the list for every label even if they don't all use hotkeys. Labels without a hotkey can be
        an empty string or ``None``. Passing ``None`` instead of a list means that no actions will
        be given hotkeys. NB: The hotkey is not bound by this class, that needs to be done in code.
        Giving hotkeys here means that they will be displayed in the menu though. Default: ``None``
    """
    # TODO This should probably be merged with Context Menu
    def __init__(self, labels, actions, hotkeys=None):
        logger.debug("Initializing %s: (labels: %s, actions: %s)", self.__class__.__name__, labels,
                     actions)
        super().__init__(tearoff=0)
        self._labels = labels
        self._actions = actions
        self._hotkeys = hotkeys
        self._create_menu()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _create_menu(self):
        """ Create the menu based on :attr:`_labels` and :attr:`_actions`. """
        for idx, (label, action) in enumerate(zip(self._labels, self._actions)):
            kwargs = {"label": label, "command": action}
            if isinstance(self._hotkeys, (list, tuple)) and self._hotkeys[idx]:
                kwargs["accelerator"] = self._hotkeys[idx]
            self.add_command(**kwargs)

    def popup(self, event):
        """ Pop up the right click menu.

        Parameters
        ----------
        event: class:`tkinter.Event`
            The tkinter mouse event calling this popup
        """
        self.tk_popup(event.x_root, event.y_root)


class ConsoleOut(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ The Console out section of the GUI.

    A Read only text box for displaying the output from stdout/stderr.

    All handling is internal to this method. To clear the console, the stored tkinter variable in
    :attr:`~lib.gui.Config.tk_vars` ``console_clear`` should be triggered.

    Parameters
    ----------
    parent: tkinter object
        The Console's parent widget
    debug: bool
        ``True`` if console output should not be directed to this widget otherwise ``False``
    """

    def __init__(self, parent, debug):
        logger.debug("Initializing %s: (parent: %s, debug: %s)",
                     self.__class__.__name__, parent, debug)
        super().__init__(parent, relief=tk.SOLID, padding=1, style="Console.TFrame")
        self._theme = get_config().user_theme["console"]
        self._console = _ReadOnlyText(self, relief=tk.FLAT)
        rc_menu = ContextMenu(self._console)
        rc_menu.cm_bind()
        self._console_clear = get_config().tk_vars.console_clear
        self._set_console_clear_var_trace()
        self._debug = debug
        self._build_console()
        self._add_tags()
        self.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(2, 0),
                  fill=tk.BOTH, expand=True)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_console_clear_var_trace(self):
        """ Set a trace on the console clear tkinter variable to trigger :func:`_clear` """
        logger.debug("Set clear trace")
        self._console_clear.trace("w", self._clear)

    def _build_console(self):
        """ Build and place the console  and add stdout/stderr redirection """
        logger.debug("Build console")
        self._console.config(width=100,
                             height=6,
                             bg=self._theme["background_color"],
                             fg=self._theme["stdout_color"])

        scrollbar = ttk.Scrollbar(self,
                                  command=self._console.yview,
                                  style="Console.Vertical.TScrollbar")
        self._console.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill="y")
        self._console.pack(side=tk.LEFT, anchor=tk.N, fill=tk.BOTH, expand=True)
        self._redirect_console()
        logger.debug("Built console")

    def _add_tags(self):
        """ Add tags to text widget to color based on output """
        logger.debug("Adding text color tags")
        self._console.tag_config("default", foreground=self._theme["stdout_color"])
        self._console.tag_config("stderr", foreground=self._theme["stderr_color"])
        self._console.tag_config("info", foreground=self._theme["info_color"])
        self._console.tag_config("verbose", foreground=self._theme["verbose_color"])
        self._console.tag_config("warning", foreground=self._theme["warning_color"])
        self._console.tag_config("critical", foreground=self._theme["critical_color"])
        self._console.tag_config("error", foreground=self._theme["error_color"])

    def _redirect_console(self):
        """ Redirect stdout/stderr to console Text Box """
        logger.debug("Redirect console")
        if self._debug:
            logger.info("Console debug activated. Outputting to main terminal")
        else:
            sys.stdout = _SysOutRouter(self._console, "stdout")
            sys.stderr = _SysOutRouter(self._console, "stderr")
        logger.debug("Redirected console")

    def _clear(self, *args):  # pylint:disable=unused-argument
        """ Clear the console output screen """
        logger.debug("Clear console")
        if not self._console_clear.get():
            logger.debug("Console not set for clearing. Skipping")
            return
        self._console.delete(1.0, tk.END)
        self._console_clear.set(False)
        logger.debug("Cleared console")


class _ReadOnlyText(tk.Text):  # pylint:disable=too-many-ancestors
    """ A read only text widget.

    Standard tkinter Text widgets are read/write by default. As we want to make the console
    display writable by the Faceswap process but not the user, we need to redirect its insert and
    delete attributes.

    Source: https://stackoverflow.com/questions/3842155
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.redirector = _WidgetRedirector(self)
        self.insert = self.redirector.register("insert", lambda *args, **kw: "break")
        self.delete = self.redirector.register("delete", lambda *args, **kw: "break")


class _SysOutRouter():
    """ Route stdout/stderr to the given text box.

    Parameters
    ----------
    console: tkinter Object
        The widget that will receive the output from stderr/stdout
    out_type: ['stdout', 'stderr']
        The output type to redirect
    """

    def __init__(self, console, out_type):
        logger.debug("Initializing %s: (console: %s, out_type: '%s')",
                     self.__class__.__name__, console, out_type)
        self._console = console
        self._out_type = out_type
        self._recolor = re.compile(r".+?(\s\d+:\d+:\d+\s)(?P<lvl>[A-Z]+)\s")
        self._ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_tag(self, string):
        """ Set the tag based on regex of log output """
        if self._out_type == "stderr":
            # Output all stderr in red
            return self._out_type

        output = self._recolor.match(string)
        if not output:
            return "default"
        tag = output.groupdict()["lvl"].strip().lower()
        return tag

    def write(self, string):
        """ Capture stdout/stderr """
        string = self._ansi_escape.sub("", string)
        self._console.insert(tk.END, string, self._get_tag(string))
        self._console.see(tk.END)

    @staticmethod
    def flush():
        """ If flush is forced, send it to normal terminal """
        sys.__stdout__.flush()


class _WidgetRedirector:
    """Support for redirecting arbitrary widget sub-commands.

    Some Tk operations don't normally pass through tkinter.  For example, if a
    character is inserted into a Text widget by pressing a key, a default Tk
    binding to the widget's 'insert' operation is activated, and the Tk library
    processes the insert without calling back into tkinter.

    Although a binding to <Key> could be made via tkinter, what we really want
    to do is to hook the Tk 'insert' operation itself.  For one thing, we want
    a text.insert call in idle code to have the same effect as a key press.

    When a widget is instantiated, a Tcl command is created whose name is the
    same as the path name widget._w.  This command is used to invoke the various
    widget operations, e.g. insert (for a Text widget). We are going to hook
    this command and provide a facility ('register') to intercept the widget
    operation.  We will also intercept method calls on the tkinter class
    instance that represents the tk widget.

    In IDLE, WidgetRedirector is used in Percolator to intercept Text
    commands.  The function being registered provides access to the top
    of a Percolator chain.  At the bottom of the chain is a call to the
    original Tk widget operation.

    Attributes
    -----------
    _operations: dict
        Dictionary mapping operation name to new function. widget: the widget whose tcl command
        is to be intercepted.
    tk: widget.tk
        A convenience attribute, probably not needed.
    orig: str
        new name of the original tcl command.

    Notes
    -----
    Since renaming to orig fails with TclError when orig already exists, only one
    WidgetDirector can exist for a given widget.
    """
    def __init__(self, widget):
        self._operations = {}
        self.widget = widget                                # widget instance
        self.tk_ = tk_ = widget.tk                          # widget's root
        wgt = widget._w  # pylint:disable=protected-access  # widget's (full) Tk pathname
        self.orig = wgt + "_orig"
        # Rename the Tcl command within Tcl:
        tk_.call("rename", wgt, self.orig)
        # Create a new Tcl command whose name is the widget's path name, and
        # whose action is to dispatch on the operation passed to the widget:
        tk_.createcommand(wgt, self.dispatch)

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.widget.__class__.__name__}"
                f"<{self.widget._w}>)")  # pylint:disable=protected-access

    def close(self):
        "de-register operations and revert redirection created by .__init__."
        for operation in list(self._operations):
            self.unregister(operation)
        widget = self.widget
        tk_ = widget.tk
        wgt = widget._w  # pylint:disable=protected-access
        # Restore the original widget Tcl command.
        tk_.deletecommand(wgt)
        tk_.call("rename", self.orig, wgt)
        del self.widget, self.tk_  # Should not be needed
        # if instance is deleted after close, as in Percolator.

    def register(self, operation, function):
        """Return _OriginalCommand(operation) after registering function.

        Registration adds an operation: function pair to ._operations.
        It also adds a widget function attribute that masks the tkinter
        class instance method.  Method masking operates independently
        from command dispatch.

        If a second function is registered for the same operation, the
        first function is replaced in both places.
        """
        self._operations[operation] = function
        setattr(self.widget, operation, function)
        return _OriginalCommand(self, operation)

    def unregister(self, operation):
        """Return the function for the operation, or None.

        Deleting the instance attribute unmasks the class attribute.
        """
        if operation in self._operations:
            function = self._operations[operation]
            del self._operations[operation]
            try:
                delattr(self.widget, operation)
            except AttributeError:
                pass
            return function
        return None

    def dispatch(self, operation, *args):
        """Callback from Tcl which runs when the widget is referenced.

        If an operation has been registered in self._operations, apply the
        associated function to the args passed into Tcl. Otherwise, pass the
        operation through to Tk via the original Tcl function.

        Note that if a registered function is called, the operation is not
        passed through to Tk.  Apply the function returned by self.register()
        to *args to accomplish that.

        """
        op_ = self._operations.get(operation)
        try:
            if op_:
                return op_(*args)
            return self.tk_.call((self.orig, operation) + args)
        except TclError:
            return ""


class _OriginalCommand:
    """Callable for original tk command that has been redirected.

    Returned by .register; can be used in the function registered.
    redirect = WidgetRedirector(text)
    def my_insert(*args):
        print("insert", args)
        original_insert(*args)
    original_insert = redirect.register("insert", my_insert)
    """

    def __init__(self, redirect, operation):
        """Create .tk_call and .orig_and_operation for .__call__ method.

        .redirect and .operation store the input args for __repr__.
        .tk and .orig copy attributes of .redirect (probably not needed).
        """
        self.redirect = redirect
        self.operation = operation
        self.tk_ = redirect.tk_  # redundant with self.redirect
        self.orig = redirect.orig  # redundant with self.redirect
        # These two could be deleted after checking recipient code.
        self.tk_call = redirect.tk_.call
        self.orig_and_operation = (redirect.orig, operation)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.redirect}, {self.operation})"

    def __call__(self, *args):
        return self.tk_call(self.orig_and_operation + args)


class StatusBar(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ Status Bar for displaying the Status Message and  Progress Bar at the bottom of the GUI.

    Parameters
    ----------
    parent: tkinter object
        The parent tkinter widget that will hold the status bar
    hide_status: bool, optional
        ``True`` to hide the status message that appears at the far left hand side of the status
        frame otherwise ``False``. Default: ``False``
    """

    def __init__(self, parent: ttk.Frame, hide_status: bool = False) -> None:
        super().__init__(parent)
        self._frame = ttk.Frame(self)
        self._message = tk.StringVar()
        self._pbar_message = tk.StringVar()
        self._pbar_position = tk.IntVar()
        self._mode: T.Literal["indeterminate", "determinate"] = "determinate"

        self._message.set("Ready")

        self._status(hide_status)
        self._pbar = self._progress_bar()
        self.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self._frame.pack(padx=10, pady=2, fill=tk.X, expand=False)

    @property
    def message(self) -> tk.StringVar:
        """:class:`tkinter.StringVar`: The variable to hold the status bar message on the left
        hand side of the status bar. """
        return self._message

    def _status(self, hide_status: bool) -> None:
        """ Place Status label into left of the status bar.

        Parameters
        ----------
        hide_status: bool, optional
            ``True`` to hide the status message that appears at the far left hand side of the
            status frame otherwise ``False``
        """
        if hide_status:
            return

        statusframe = ttk.Frame(self._frame)
        statusframe.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=False)

        lbltitle = ttk.Label(statusframe, text="Status:", width=6, anchor=tk.W)
        lbltitle.pack(side=tk.LEFT, expand=False)

        lblstatus = ttk.Label(statusframe,
                              width=40,
                              textvariable=self._message,
                              anchor=tk.W)
        lblstatus.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)

    def _progress_bar(self) -> ttk.Progressbar:
        """ Place progress bar into right of the status bar.

        Returns
        -------
        :class:`tkinter.ttk.Progressbar`
            The progress bar object
        """
        progressframe = ttk.Frame(self._frame)
        progressframe.pack(side=tk.RIGHT, anchor=tk.E, fill=tk.X)

        lblmessage = ttk.Label(progressframe, textvariable=self._pbar_message)
        lblmessage.pack(side=tk.LEFT, padx=3, fill=tk.X, expand=True)

        pbar = ttk.Progressbar(progressframe,
                               length=200,
                               variable=self._pbar_position,
                               maximum=100,
                               mode=self._mode)
        pbar.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        pbar.pack_forget()
        return pbar

    def start(self, mode: T.Literal["indeterminate", "determinate"]) -> None:
        """ Set progress bar mode and display,

        Parameters
        ----------
        mode: ["indeterminate", "determinate"]
            The mode that the progress bar should be executed in
        """
        self._set_mode(mode)
        self._pbar.pack()

    def stop(self) -> None:
        """ Reset progress bar and hide """
        self._pbar_message.set("")
        self._pbar_position.set(0)
        self._mode = "determinate"
        self._set_mode(self._mode)
        self._pbar.pack_forget()

    def _set_mode(self, mode: T.Literal["indeterminate", "determinate"]) -> None:
        """ Set the progress bar mode

        Parameters
        ----------
        mode: ["indeterminate", "determinate"]
            The mode that the progress bar should be executed in
        """
        self._mode = mode
        self._pbar.config(mode=self._mode)
        if mode == "indeterminate":
            self._pbar.config(maximum=100)
            self._pbar.start()
        else:
            self._pbar.stop()
            self._pbar.config(maximum=100)

    def set_mode(self, mode: T.Literal["indeterminate", "determinate"]) -> None:
        """ Set the mode of a currently displayed progress bar and reset position to 0.

        If the given mode is the same as the currently configured mode, returns without performing
        any action.

        Parameters
        ----------
        mode: ["indeterminate", "determinate"]
            The mode that the progress bar should be set to
        """
        if mode == self._mode:
            return
        self.stop()
        self.start(mode)

    def progress_update(self, message: str, position: int, update_position: bool = True) -> None:
        """ Update the GUIs progress bar and position.

        Parameters
        ----------
        message: str
            The message to display next to the progress bar
        position: int
            The position that the progress bar should be set to
        update_position: bool, optional
            If ``True`` then the progress bar will be updated to the position given in
            :attr:`position`. If ``False`` the progress bar will not be updates. Default: ``True``
        """
        self._pbar_message.set(message)
        if update_position:
            self._pbar_position.set(position)


class Tooltip:  # pylint:disable=too-few-public-methods
    """ Create a tooltip for a given widget as the mouse goes on it.

    Parameters
    ----------
    widget: tkinter object
        The widget to apply the tool-tip to
    pad: tuple, optional
        (left, top, right, bottom) padding for the tool-tip. Default: (5, 3, 5, 3)
    text: str, optional
        The text to be displayed in the tool-tip. Default: 'widget info'
    text_variable: :class:`tkinter.strVar`, optional
        The text variable to use for dynamic help text. Appended after the contents of :attr:`text`
        if provided. Default: ``None``
    wait_time: int, optional
        The time in milliseconds to wait before showing the tool-tip. Default: 400
    wrap_length: int, optional
        The text length for each line before wrapping. Default: 250

    Example
    -------
    >>> button = ttk.Button(parent, text="Exit")
    >>> Tooltip(button, text="Click to exit")
    >>> button.pack()

    Notes
    -----
    Adapted from StackOverflow: http://stackoverflow.com/questions/3221956 and
    http://www.daniweb.com/programming/software-development/code/484591/a-tooltip-class-for-tkinter
    """
    def __init__(self, widget, *, pad=(5, 3, 5, 3), text="widget info",
                 text_variable=None, wait_time=400, wrap_length=250):

        self._waittime = wait_time  # in milliseconds, originally 500
        self.wrap_length = wrap_length  # in pixels, originally 180
        self._widget = widget
        self._text = text
        self._text_variable = text_variable
        self._widget.bind("<Enter>", self._on_enter)
        self._widget.bind("<Leave>", self._on_leave)
        self._widget.bind("<ButtonPress>", self._on_leave)
        self._theme = get_config().user_theme["tooltip"]
        self._pad = pad
        self._ident = None
        self._topwidget = None

    def _on_enter(self, event=None):  # pylint:disable=unused-argument
        """ Schedule on an enter event """
        self._schedule()

    def _on_leave(self, event=None):  # pylint:disable=unused-argument
        """ remove schedule on a leave event """
        self._unschedule()
        self._hide()

    def _schedule(self):
        """ Show the tooltip after wait period """
        self._unschedule()
        self._ident = self._widget.after(self._waittime, self._show)

    def _unschedule(self):
        """ Hide the tooltip """
        id_ = self._ident
        self._ident = None
        if id_:
            self._widget.after_cancel(id_)

    def _show(self):
        """ Show the tooltip """
        def tip_pos_calculator(widget, label,
                               *,
                               tip_delta=(10, 5), pad=(5, 3, 5, 3)):
            """ Calculate the tooltip position """

            s_width, s_height = widget.winfo_screenwidth(), widget.winfo_screenheight()

            width, height = (pad[0] + label.winfo_reqwidth() + pad[2],
                             pad[1] + label.winfo_reqheight() + pad[3])

            mouse_x, mouse_y = widget.winfo_pointerxy()

            x_1, y_1 = mouse_x + tip_delta[0], mouse_y + tip_delta[1]
            x_2, y_2 = x_1 + width, y_1 + height

            x_delta = max(x_2 - s_width, 0)
            y_delta = max(y_2 - s_height, 0)

            offscreen = (x_delta, y_delta) != (0, 0)

            if offscreen:

                if x_delta:
                    x_1 = mouse_x - tip_delta[0] - width

                if y_delta:
                    y_1 = mouse_y - tip_delta[1] - height

            offscreen_again = y_1 < 0  # out on the top

            if offscreen_again:
                # No further checks will be done.
                # TIP:
                # A further mod might auto-magically augment the wrap length when the tooltip is
                # too high to be kept inside the screen.
                y_1 = 0

            return x_1, y_1

        pad = self._pad
        widget = self._widget

        # Creates a top level window
        self._topwidget = tk.Toplevel(widget)
        if platform.system() == "Darwin":
            # For Mac OS
            self._topwidget.tk.call("::tk::unsupported::MacWindowStyle",
                                    "style", self._topwidget._w,  # pylint:disable=protected-access
                                    "help", "none")

        # Leaves only the label and removes the app window
        self._topwidget.wm_overrideredirect(True)

        win = tk.Frame(self._topwidget,
                       background=self._theme["background_color"],
                       highlightbackground=self._theme["border_color"],
                       highlightcolor=self._theme["border_color"],
                       highlightthickness=1,
                       borderwidth=0)

        text = self._text
        if self._text_variable and self._text_variable.get():
            text += f"\n\nCurrent value: '{self._text_variable.get()}'"
        label = tk.Label(win,
                         text=text,
                         justify=tk.LEFT,
                         background=self._theme["background_color"],
                         foreground=self._theme["font_color"],
                         relief=tk.SOLID,
                         borderwidth=0,
                         wraplength=self.wrap_length)

        label.grid(padx=(pad[0], pad[2]),
                   pady=(pad[1], pad[3]),
                   sticky=tk.NSEW)
        win.grid()

        xpos, ypos = tip_pos_calculator(widget, label)

        self._topwidget.wm_geometry(f"+{xpos}+{ypos}")

    def _hide(self):
        """ Hide the tooltip """
        topwidget = self._topwidget
        if topwidget:
            topwidget.destroy()
        self._topwidget = None


class MultiOption(ttk.Checkbutton):  # pylint:disable=too-many-ancestors
    """ Similar to the standard :class:`ttk.Radio` widget, but with the ability to select
    multiple pre-defined options. Selected options are generated as `nargs` for the argument
    parser to consume.

    Parameters
    ----------
    parent: :class:`ttk.Frame`
        The tkinter parent widget for the check button
    value: str
        The raw option value for this check button
    variable: :class:`tkinter.StingVar`
        The master variable for the group of check buttons that this check button will belong to.
        The output of this variable will be a string containing a space separated list of the
        selected check button options
    """
    def __init__(self, parent, value, variable, **kwargs):
        self._tk_var = tk.BooleanVar()
        self._tk_var.set(value in variable.get().split())
        super().__init__(parent, variable=self._tk_var, **kwargs)
        self._value = value
        self._master_variable = variable
        self._tk_var.trace("w", self._on_update)
        self._master_variable.trace("w", self._on_master_update)

    @property
    def _master_list(self):
        """ list: The contents of the check box group's :attr:`_master_variable` in list form.
        Selected check boxes will appear in this list. """
        retval = self._master_variable.get().split()
        logger.trace(retval)
        return retval

    @property
    def _master_needs_update(self):
        """ bool: ``True`` if :attr:`_master_variable` requires updating otherwise ``False``. """
        active = self._tk_var.get()
        retval = ((active and self._value not in self._master_list) or
                  (not active and self._value in self._master_list))
        logger.trace(retval)
        return retval

    def _on_update(self, *args):  # pylint:disable=unused-argument
        """ Update the master variable on a check button change.

        The value for this checked option is added or removed from the :attr:`_master_variable`
        on a ``True``, ``False`` change for this check button.

        Parameters
        ----------
        args: tuple
            Required for variable callback, but unused
        """
        if not self._master_needs_update:
            return
        new_vals = self._master_list + [self._value] if self._tk_var.get() else [
            val
            for val in self._master_list
            if val != self._value]
        val = " ".join(new_vals)
        logger.trace("Setting master variable to: %s", val)
        self._master_variable.set(val)

    def _on_master_update(self, *args):  # pylint:disable=unused-argument
        """ Update the check button on a master variable change (e.g. load .fsw file in the GUI).

        The value for this option is set to ``True`` or ``False`` depending on it's existence in
        the :attr:`_master_variable`

        Parameters
        ----------
        args: tuple
            Required for variable callback, but unused
        """
        if not self._master_needs_update:
            return
        state = self._value in self._master_list
        logger.trace("Setting '%s' to %s", self._value, state)
        self._tk_var.set(state)


class PopupProgress(tk.Toplevel):
    """ A simple pop up progress bar that appears of the center of the root window.

    When this is called, the root will be disabled until the :func:`close` method is called.

    Parameters
    ----------
    title: str
        The title to appear above the progress bar
    total: int or float
        The total count of items for the progress bar

    Example
    -------
    >>> total = 100
    >>> progress = PopupProgress("My title...", total)
    >>> for i in range(total):
    >>>     progress.update(1)
    >>> progress.close()
    """
    def __init__(self, title, total):
        super().__init__()
        self._total = total
        if platform.system() == "Darwin":  # For Mac OS
            self.tk.call("::tk::unsupported::MacWindowStyle",
                         "style", self._w,  # pylint:disable=protected-access
                         "help", "none")
        # Leaves only the label and removes the app window
        self.wm_overrideredirect(True)
        self.attributes('-topmost', 'true')
        self.transient()

        self._lbl_title = self._set_title(title)
        self._progress_bar = self._get_progress_bar()

        offset = np.array((self.master.winfo_rootx(), self.master.winfo_rooty()))
        # TODO find way to get dimensions of the pop up without it flicking onto the screen
        self.update_idletasks()
        center = np.array((
            (self.master.winfo_width() // 2) - (self.winfo_width() // 2),
            (self.master.winfo_height() // 2) - (self.winfo_height() // 2))) + offset
        self.wm_geometry(f"+{center[0]}+{center[1]}")
        get_config().set_cursor_busy()
        self.grab_set()

    @property
    def progress_bar(self):
        """ :class:`tkinter.ttk.Progressbar`: The progress bar object within the pop up window. """
        return self._progress_bar

    def _set_title(self, title):
        """ Set the initial title of the pop up progress bar.

        Parameters
        ----------
        title: str
            The title to appear above the progress bar

        Returns
        -------
        :class:`tkinter.ttk.Label`
            The heading label for the progress bar
        """
        frame = ttk.Frame(self)
        frame.pack(side=tk.TOP, padx=5, pady=5)
        lbl = ttk.Label(frame, text=title)
        lbl.pack(side=tk.TOP, pady=(5, 0), expand=True, fill=tk.X)
        return lbl

    def _get_progress_bar(self):
        """ Set up the progress bar with the supplied total.

        Returns
        -------
        :class:`tkinter.ttk.Progressbar`
            The configured progress bar for the pop up window
        """
        frame = ttk.Frame(self)
        frame.pack(side=tk.BOTTOM, padx=5, pady=(0, 5))
        pbar = ttk.Progressbar(frame,
                               length=400,
                               maximum=self._total,
                               mode="determinate")
        pbar.pack(side=tk.LEFT)
        return pbar

    def step(self, amount):
        """ Increment the progress bar.

        Parameters
        ----------
        amount: int or float
            The amount to increment the progress bar by
        """
        self._progress_bar.step(amount)
        self._progress_bar.update_idletasks()

    def stop(self):
        """ Stop the progress bar, re-enable the root window and destroy the pop up window. """
        self._progress_bar.stop()
        get_config().set_cursor_default()
        self.grab_release()
        self.destroy()

    def update_title(self, title):
        """ Update the title that displays above the progress bar.

        Parameters
        ----------
        title: str
            The title to appear above the progress bar
        """
        self._lbl_title.config(text=title)
        self._lbl_title.update_idletasks()


class ToggledFrame(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ A collapsible and expandable frame.

    The frame contains a header given in the text argument, and adds an expand contract button.
    Clicking on the header will expand and contract the sub-frame below

    Parameters
    ----------
    text: str
        The text to appear in the Toggle Frame header
    theme: str, optional
        The theme to use for the panel header. Default: `"CPanel"`
    subframe_style: str, optional
        The name of the ttk Style to use for the sub frame. Default: ``None``
    toggle_var: :class:`tk.BooleanVar`, optional
        If provided, this variable will control the expanded (``True``) and minimized (``False``)
        state of the widget. Set to None to create the variable internally. Default: ``None``
    """
    def __init__(self, parent, *args, text="", theme="CPanel", toggle_var=None, **kwargs):
        logger.debug("Initializing %s: (parent: %s, text: %s, theme: %s, toggle_var: %s)",
                     self.__class__.__name__, parent, text, theme, toggle_var)

        theme = "CPanel" if not theme else theme
        theme = theme[:-1] if theme[-1] == "." else theme
        super().__init__(parent, *args, style=f"{theme}.Group.TFrame", **kwargs)
        self._text = text

        if toggle_var:
            self._toggle_var = toggle_var
        else:
            self._toggle_var = tk.BooleanVar()
            self._toggle_var.set(1)
        self._icon_var = tk.StringVar()
        self._icon_var.set("-" if self.is_expanded else "+")

        self._build_header(theme)

        self.sub_frame = ttk.Frame(self, style=f"{theme}.Subframe.Group.TFrame", padding=1)

        if self.is_expanded:
            self.sub_frame.pack(fill=tk.X, expand=True)

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_expanded(self):
        """ bool: ``True`` if the Toggle Frame is expanded. ``False`` if it is minimized. """
        return self._toggle_var.get()

    def _build_header(self, theme):
        """ The Header row. Contains the title text and is made clickable to expand and contract
        the sub-frame.

        Parameters
        theme: str
            The theme to use for the panel header
        """
        header_frame = ttk.Frame(self, name="toggledframe_header")

        text_label = ttk.Label(header_frame,
                               name="toggledframe_headerlbl",
                               text=self._text,
                               style=f"{theme}.Groupheader.TLabel",
                               cursor="hand2")
        toggle_button = ttk.Label(header_frame,
                                  name="toggledframe_headerbtn",
                                  textvariable=self._icon_var,
                                  style=f"{theme}.Groupheader.TLabel",
                                  cursor="hand2",
                                  width=2)
        text_label.bind("<Button-1>", self._toggle)
        toggle_button.bind("<Button-1>", self._toggle)

        text_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        toggle_button.pack(side=tk.RIGHT)
        header_frame.pack(fill=tk.X, expand=True)

    def _toggle(self, event):  # pylint:disable=unused-argument
        """ Toggle the sub-frame between contracted or expanded, and update the toggle icon
        appropriately.

        Parameters
        ----------
        event: tkinter event
            Required but unused
         """
        if self.is_expanded:
            self.sub_frame.forget()
            self._icon_var.set("+")
            self._toggle_var.set(0)
        else:
            self.sub_frame.pack(fill=tk.X, expand=True)
            self._icon_var.set("-")
            self._toggle_var.set(1)
