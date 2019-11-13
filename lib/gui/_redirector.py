#!/usr/bin/env python3
""" Widget redirector from IdleLib
https://github.com/python/cpython/blob/master/Lib/idlelib/redirector.py
"""

from tkinter import TclError


class WidgetRedirector:
    """Support for redirecting arbitrary widget subcommands.

    Some Tk operations don't normally pass through tkinter.  For example, if a
    character is inserted into a Text widget by pressing a key, a default Tk
    binding to the widget's 'insert' operation is activated, and the Tk library
    processes the insert without calling back into tkinter.

    Although a binding to <Key> could be made via tkinter, what we really want
    to do is to hook the Tk 'insert' operation itself.  For one thing, we want
    a text.insert call in idle code to have the same effect as a key press.

    When a widget is instantiated, a Tcl command is created whose name is the
    same as the pathname widget._w.  This command is used to invoke the various
    widget operations, e.g. insert (for a Text widget). We are going to hook
    this command and provide a facility ('register') to intercept the widget
    operation.  We will also intercept method calls on the tkinter class
    instance that represents the tk widget.

    In IDLE, WidgetRedirector is used in Percolator to intercept Text
    commands.  The function being registered provides access to the top
    of a Percolator chain.  At the bottom of the chain is a call to the
    original Tk widget operation.
    """
    def __init__(self, widget):
        """Initialize attributes and setup redirection.

        _operations: dict mapping operation name to new function.
        widget: the widget whose tcl command is to be intercepted.
        tk: widget.tk, a convenience attribute, probably not needed.
        orig: new name of the original tcl command.

        Since renaming to orig fails with TclError when orig already
        exists, only one WidgetDirector can exist for a given widget.
        """
        self._operations = {}
        self.widget = widget                                # widget instance
        self.tk_ = tk_ = widget.tk                          # widget's root
        wgt = widget._w  # pylint:disable=protected-access  # widget's (full) Tk pathname
        self.orig = wgt + "_orig"
        # Rename the Tcl command within Tcl:
        tk_.call("rename", wgt, self.orig)
        # Create a new Tcl command whose name is the widget's pathname, and
        # whose action is to dispatch on the operation passed to the widget:
        tk_.createcommand(wgt, self.dispatch)

    def __repr__(self):
        return "%s(%s<%s>)" % (self.__class__.__name__,
                               self.widget.__class__.__name__,
                               self.widget._w)  # pylint:disable=protected-access

    def close(self):
        "Unregister operations and revert redirection created by .__init__."
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
        """Return OriginalCommand(operation) after registering function.

        Registration adds an operation: function pair to ._operations.
        It also adds a widget function attribute that masks the tkinter
        class instance method.  Method masking operates independently
        from command dispatch.

        If a second function is registered for the same operation, the
        first function is replaced in both places.
        """
        self._operations[operation] = function
        setattr(self.widget, operation, function)
        return OriginalCommand(self, operation)

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
        to *args to accomplish that.  For an example, see colorizer.py.

        """
        op_ = self._operations.get(operation)
        try:
            if op_:
                return op_(*args)
            return self.tk_.call((self.orig, operation) + args)
        except TclError:
            return ""


class OriginalCommand:
    """Callable for original tk command that has been redirected.

    Returned by .register; can be used in the function registered.
    redir = WidgetRedirector(text)
    def my_insert(*args):
        print("insert", args)
        original_insert(*args)
    original_insert = redir.register("insert", my_insert)
    """

    def __init__(self, redir, operation):
        """Create .tk_call and .orig_and_operation for .__call__ method.

        .redir and .operation store the input args for __repr__.
        .tk and .orig copy attributes of .redir (probably not needed).
        """
        self.redir = redir
        self.operation = operation
        self.tk_ = redir.tk_  # redundant with self.redir
        self.orig = redir.orig  # redundant with self.redir
        # These two could be deleted after checking recipient code.
        self.tk_call = redir.tk_.call
        self.orig_and_operation = (redir.orig, operation)

    def __repr__(self):
        return "%s(%r, %r)" % (self.__class__.__name__,
                               self.redir, self.operation)

    def __call__(self, *args):
        return self.tk_call(self.orig_and_operation + args)
