#!/usr/bin/env python3
""" Custom :class:`argparse.Action` objects for Faceswap's Command Line Interface.

The custom actions within this module allow for custom manipulation of Command Line Arguments
as well as adding a mechanism for indicating to the GUI how specific options should be rendered.
"""

import argparse
import os


# << FILE HANDLING >>

class _FullPaths(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Parent class for various file type and file path handling classes.

    Expands out given paths to their full absolute paths. This class should not be
    called directly. It is the base class for the various different file handling
    methods.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, (list, tuple)):
            vals = [os.path.abspath(os.path.expanduser(val)) for val in values]
        else:
            vals = os.path.abspath(os.path.expanduser(values))
        setattr(namespace, self.dest, vals)


class DirFullPaths(_FullPaths):
    """ Adds support for a Directory browser in the GUI.

    This is a standard :class:`argparse.Action` (with stock parameters) which indicates to the GUI
    that a dialog box should be opened in order to browse for a folder.

    No additional parameters are required.

    Example
    -------
    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--folder_location"),
    >>>        action=DirFullPaths)),
    """
    # pylint: disable=too-few-public-methods,unnecessary-pass
    pass


class FileFullPaths(_FullPaths):
    """ Adds support for a File browser to select a single file in the GUI.

    This extends the standard :class:`argparse.Action` and adds an additional parameter
    :attr:`filetypes`, indicating to the GUI that it should pop a file browser for opening a file
    and limit the results to the file types listed. As well as the standard parameters, the
    following parameter is required:

    Parameters
    ----------
    filetypes: str
        The accepted file types for this option. This is the key for the GUIs lookup table which
        can be found in :class:`lib.gui.utils.FileHandler`

    Example
    -------
    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--video_location"),
    >>>        action=FileFullPaths,
    >>>        filetypes="video))"
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, *args, filetypes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.filetypes = filetypes

    def _get_kwargs(self):
        names = ["option_strings",
                 "dest",
                 "nargs",
                 "const",
                 "default",
                 "type",
                 "choices",
                 "help",
                 "metavar",
                 "filetypes"]
        return [(name, getattr(self, name)) for name in names]


class FilesFullPaths(FileFullPaths):  # pylint: disable=too-few-public-methods
    """ Adds support for a File browser to select multiple files in the GUI.

    This extends the standard :class:`argparse.Action` and adds an additional parameter
    :attr:`filetypes`, indicating to the GUI that it should pop a file browser, and limit
    the results to the file types listed. Multiple files can be selected for opening, so the
    :attr:`nargs` parameter must be set. As well as the standard parameters, the following
    parameter is required:

    Parameters
    ----------
    filetypes: str
        The accepted file types for this option. This is the key for the GUIs lookup table which
        can be found in :class:`lib.gui.utils.FileHandler`

    Example
    -------
    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--images"),
    >>>        action=FilesFullPaths,
    >>>        filetypes="image",
    >>>        nargs="+"))
    """
    def __init__(self, *args, filetypes=None, **kwargs):
        if kwargs.get("nargs", None) is None:
            opt = kwargs["option_strings"]
            raise ValueError("nargs must be provided for FilesFullPaths: {}".format(opt))
        super().__init__(*args, **kwargs)


class DirOrFileFullPaths(FileFullPaths):  # pylint: disable=too-few-public-methods
    """ Adds support to the GUI to launch either a file browser or a folder browser.

    Some inputs (for example source frames) can come from a folder of images or from a
    video file. This indicates to the GUI that it should place 2 buttons (one for a folder
    browser, one for a file browser) for file/folder browsing.

    The standard :class:`argparse.Action` is extended with the additional parameter
    :attr:`filetypes`, indicating to the GUI that it should pop a file browser, and limit
    the results to the file types listed. As well as the standard parameters, the following
    parameter is required:

    Parameters
    ----------
    filetypes: str
        The accepted file types for this option. This is the key for the GUIs lookup table which
        can be found in :class:`lib.gui.utils.FileHandler`. NB: This parameter is only used for
        the file browser and not the folder browser

    Example
    -------
    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--input_frames"),
    >>>        action=DirOrFileFullPaths,
    >>>        filetypes="video))"
    """
    pass  # pylint: disable=unnecessary-pass


class SaveFileFullPaths(FileFullPaths):
    """ Adds support for a Save File dialog in the GUI.

    This extends the standard :class:`argparse.Action` and adds an additional parameter
    :attr:`filetypes`, indicating to the GUI that it should pop a save file browser, and limit
    the results to the file types listed. As well as the standard parameters, the following
    parameter is required:

    Parameters
    ----------
    filetypes: str
        The accepted file types for this option. This is the key for the GUIs lookup table which
        can be found in :class:`lib.gui.utils.FileHandler`

    Example
    -------
    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--video_out"),
    >>>        action=SaveFileFullPaths,
    >>>        filetypes="video"))
    """
    # pylint: disable=too-few-public-methods,unnecessary-pass
    pass


class ContextFullPaths(FileFullPaths):
    """ Adds support for context sensitive browser dialog opening in the GUI.

    For some tasks, the type of action (file load, folder open, file save etc.) can vary
    depending on the task to be performed (a good example of this is the effmpeg tool).
    Using this action indicates to the GUI that the type of dialog to be launched can change
    depending on another option. As well as the standard parameters, the below parameters are
    required. NB: :attr:`nargs` are explicitly disallowed.

    Parameters
    ----------
    filetypes: str
        The accepted file types for this option. This is the key for the GUIs lookup table which
        can be found in :class:`lib.gui.utils.FileHandler`
    action_option: str
        The command line option that dictates the context of the file dialog to be opened.
        Bespoke actions are set in :class:`lib.gui.utils.FileHandler`

    Example
    -------
    Assuming an argument has already been set with option string `-a` indicating the action to be
    performed, the following will pop a different type of dialog depending on the action selected:

    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--input_video"),
    >>>        action=ContextFullPaths,
    >>>        filetypes="video",
    >>>        action_option="-a"))
    """
    # pylint: disable=too-few-public-methods, too-many-arguments
    def __init__(self, *args, filetypes=None, action_option=None, **kwargs):
        opt = kwargs["option_strings"]
        if kwargs.get("nargs", None) is not None:
            raise ValueError("nargs not allowed for ContextFullPaths: {}".format(opt))
        if filetypes is None:
            raise ValueError("filetypes is required for ContextFullPaths: {}".format(opt))
        if action_option is None:
            raise ValueError("action_option is required for ContextFullPaths: {}".format(opt))
        super().__init__(*args, filetypes=filetypes, **kwargs)
        self.action_option = action_option

    def _get_kwargs(self):
        names = ["option_strings",
                 "dest",
                 "nargs",
                 "const",
                 "default",
                 "type",
                 "choices",
                 "help",
                 "metavar",
                 "filetypes",
                 "action_option"]
        return [(name, getattr(self, name)) for name in names]


# << GUI DISPLAY OBJECTS >>

class Radio(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Adds support for a GUI Radio options box.

    This is a standard :class:`argparse.Action` (with stock parameters) which indicates to the GUI
    that the options passed should be rendered as a group of Radio Buttons rather than a combo box.

    No additional parameters are required, but the :attr:`choices` parameter must be provided as
    these will be the Radio Box options. :attr:`nargs` are explicitly disallowed.

    Example
    -------
    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--foobar"),
    >>>        action=Radio,
    >>>        choices=["foo", "bar"))
    """
    def __init__(self, *args, **kwargs):
        opt = kwargs["option_strings"]
        if kwargs.get("nargs", None) is not None:
            raise ValueError("nargs not allowed for Radio buttons: {}".format(opt))
        if not kwargs.get("choices", []):
            raise ValueError("Choices must be provided for Radio buttons: {}".format(opt))
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class MultiOption(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Adds support for multiple option checkboxes in the GUI.

    This is a standard :class:`argparse.Action` (with stock parameters) which indicates to the GUI
    that the options passed should be rendered as a group of Radio Buttons rather than a combo box.

    The :attr:`choices` parameter must be provided as this provides the valid option choices.

    Example
    -------
    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--foobar"),
    >>>        action=MultiOption,
    >>>        choices=["foo", "bar"))
    """
    def __init__(self, *args, **kwargs):
        opt = kwargs["option_strings"]
        if not kwargs.get("nargs", []):
            raise ValueError("nargs must be provided for MultiOption: {}".format(opt))
        if not kwargs.get("choices", []):
            raise ValueError("Choices must be provided for MultiOption: {}".format(opt))
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class Slider(argparse.Action):  # pylint: disable=too-few-public-methods
    """ Adds support for a slider in the GUI.

    The standard :class:`argparse.Action` is extended with the additional parameters listed below.
    The :attr:`default` value must be supplied and the :attr:`type` must be either :class:`int` or
    :class:`float`. :attr:`nargs` are explicitly disallowed.

    Parameters
    ----------
    min_max: tuple
        The (`min`, `max`) values that the slider's range should be set to. The values should be a
        pair of `float` or `int` data types, depending on the data type of the slider. NB: These
        min/max values are not enforced, they are purely for setting the slider range. Values
        outside of this range can still be explicitly passed in from the cli.
    rounding: int
        If the underlying data type for the option is a `float` then this value is the number of
        decimal places to round the slider values to. If the underlying data type for the option is
        an `int` then this is the step interval between each value for the slider.

    Examples
    --------
    For integer values:

    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--foobar"),
    >>>        action=Slider,
    >>>        min_max=(0, 10)
    >>>        rounding=1
    >>>        type=int,
    >>>        default=5))

    For floating point values:

    >>> argument_list = []
    >>> argument_list.append(dict(
    >>>        opts=("-f", "--foobar"),
    >>>        action=Slider,
    >>>        min_max=(0.00, 1.00)
    >>>        rounding=2
    >>>        type=float,
    >>>        default=5.00))
    """
    def __init__(self, *args, min_max=None, rounding=None, **kwargs):
        opt = kwargs["option_strings"]
        if kwargs.get("nargs", None) is not None:
            raise ValueError("nargs not allowed for Slider: {}".format(opt))
        if kwargs.get("default", None) is None:
            raise ValueError("A default value must be supplied for Slider: {}".format(opt))
        if kwargs.get("type", None) not in (int, float):
            raise ValueError("Sliders only accept int and float data types: {}".format(opt))
        if min_max is None:
            raise ValueError("min_max must be provided for Sliders: {}".format(opt))
        if rounding is None:
            raise ValueError("rounding must be provided for Sliders: {}".format(opt))

        super().__init__(*args, **kwargs)
        self.min_max = min_max
        self.rounding = rounding

    def _get_kwargs(self):
        names = ["option_strings",
                 "dest",
                 "nargs",
                 "const",
                 "default",
                 "type",
                 "choices",
                 "help",
                 "metavar",
                 "min_max",  # Tuple containing min and max values of scale
                 "rounding"]  # Decimal places to round floats to or step interval for ints
        return [(name, getattr(self, name)) for name in names]

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
