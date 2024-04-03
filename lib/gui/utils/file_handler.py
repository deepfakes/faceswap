#!/usr/bin/env python3
""" File browser utility functions for the Faceswap GUI. """
import logging
import platform
import tkinter as tk
from tkinter import filedialog
import typing as T

logger = logging.getLogger(__name__)
_FILETYPE = T.Literal["default", "alignments", "config_project", "config_task",
                      "config_all", "csv", "image", "ini", "state", "log", "video"]
_HANDLETYPE = T.Literal["open", "save", "filename", "filename_multi", "save_filename",
                        "context", "dir"]


class FileHandler():  # pylint:disable=too-few-public-methods
    """ Handles all GUI File Dialog actions and tasks.

    Parameters
    ----------
    handle_type: ['open', 'save', 'filename', 'filename_multi', 'save_filename', 'context', 'dir']
        The type of file dialog to return. `open` and `save` will perform the open and save actions
        and return the file. `filename` returns the filename from an `open` dialog.
        `filename_multi` allows for multi-selection of files and returns a list of files selected.
        `save_filename` returns the filename from a `save as` dialog. `context` is a context
        sensitive parameter that returns a certain dialog based on the current options. `dir` asks
        for a folder location.
    file_type: ['default', 'alignments', 'config_project', 'config_task', 'config_all', 'csv', \
               'image', 'ini', 'state', 'log', 'video'] or ``None``
        The type of file that this dialog is for. `default` allows selection of any files. Other
        options limit the file type selection
    title: str, optional
        The title to display on the file dialog. If `None` then the default title will be used.
        Default: ``None``
    initial_folder: str, optional
        The folder to initially open with the file dialog. If `None` then tkinter will decide.
        Default: ``None``
    initial_file: str, optional
        The filename to set with the file dialog. If `None` then tkinter no initial filename is.
        specified. Default: ``None``
    command: str, optional
        Required for context handling file dialog, otherwise unused. Default: ``None``
    action: str, optional
        Required for context handling file dialog, otherwise unused. Default: ``None``
    variable: str, optional
        Required for context handling file dialog, otherwise unused. The variable to associate
        with this file dialog. Default: ``None``
    parent: :class:`tkinter.Frame`, optional
        The parent that is launching the file dialog. ``None`` sets this to root. Default: ``None``

    Attributes
    ----------
    return_file: str or object
        The return value from the file dialog

    Example
    -------
    >>> handler = FileHandler('filename', 'video', title='Select a video...')
    >>> video_file = handler.return_file
    >>> print(video_file)
    '/path/to/selected/video.mp4'
    """

    def __init__(self,
                 handle_type: _HANDLETYPE,
                 file_type: _FILETYPE | None,
                 title: str | None = None,
                 initial_folder: str | None = None,
                 initial_file: str | None = None,
                 command: str | None = None,
                 action: str | None = None,
                 variable: str | None = None,
                 parent: tk.Frame | None = None) -> None:
        logger.debug("Initializing %s: (handle_type: '%s', file_type: '%s', title: '%s', "
                     "initial_folder: '%s', initial_file: '%s', command: '%s', action: '%s', "
                     "variable: %s, parent: %s)", self.__class__.__name__, handle_type, file_type,
                     title, initial_folder, initial_file, command, action, variable, parent)
        self._handletype = handle_type
        self._dummy_master = self._set_dummy_master()
        self._defaults = self._set_defaults()
        self._kwargs = self._set_kwargs(title,
                                        initial_folder,
                                        initial_file,
                                        file_type,
                                        command,
                                        action,
                                        variable,
                                        parent)
        self.return_file = getattr(self, f"_{self._handletype.lower()}")()
        self._remove_dummy_master()

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _filetypes(self) -> dict[str, list[tuple[str, str]]]:
        """ dict: The accepted extensions for each file type for opening/saving """
        all_files = ("All files", "*.*")
        filetypes = {
            "default": [all_files],
            "alignments": [("Faceswap Alignments", "*.fsa"), all_files],
            "config_project": [("Faceswap Project files", "*.fsw"), all_files],
            "config_task": [("Faceswap Task files", "*.fst"), all_files],
            "config_all": [("Faceswap Project and Task files", "*.fst *.fsw"), all_files],
            "csv": [("Comma separated values", "*.csv"), all_files],
            "image": [("Bitmap", "*.bmp"),
                      ("JPG", "*.jpeg *.jpg"),
                      ("PNG", "*.png"),
                      ("TIFF", "*.tif *.tiff"),
                      all_files],
            "ini": [("Faceswap config files", "*.ini"), all_files],
            "json": [("JSON file", "*.json"), all_files],
            "model": [("Keras model files", "*.h5"), all_files],
            "state": [("State files", "*.json"), all_files],
            "log": [("Log files", "*.log"), all_files],
            "video": [("Audio Video Interleave", "*.avi"),
                      ("Flash Video", "*.flv"),
                      ("Matroska", "*.mkv"),
                      ("MOV", "*.mov"),
                      ("MP4", "*.mp4"),
                      ("MPEG", "*.mpeg *.mpg *.ts *.vob"),
                      ("WebM", "*.webm"),
                      ("Windows Media Video", "*.wmv"),
                      all_files]}

        # Add in multi-select options and upper case extensions for Linux
        for key in filetypes:
            if platform.system() == "Linux":
                filetypes[key] = [item
                                  if item[0] == "All files"
                                  else (item[0], f"{item[1]} {item[1].upper()}")
                                  for item in filetypes[key]]
            if len(filetypes[key]) > 2:
                multi = [f"{key.title()} Files"]
                multi.append(" ".join([ftype[1]
                                       for ftype in filetypes[key] if ftype[0] != "All files"]))
                filetypes[key].insert(0, T.cast(tuple[str, str], tuple(multi)))
        return filetypes

    @property
    def _contexts(self) -> dict[str, dict[str, str | dict[str, str]]]:
        """dict: Mapping of commands, actions and their corresponding file dialog for context
        handle types. """
        return {"effmpeg": {"input": {"extract": "filename",
                                      "gen-vid": "dir",
                                      "get-fps": "filename",
                                      "get-info": "filename",
                                      "mux-audio": "filename",
                                      "rescale": "filename",
                                      "rotate": "filename",
                                      "slice": "filename"},
                            "output": {"extract": "dir",
                                       "gen-vid": "save_filename",
                                       "get-fps": "nothing",
                                       "get-info": "nothing",
                                       "mux-audio": "save_filename",
                                       "rescale": "save_filename",
                                       "rotate": "save_filename",
                                       "slice": "save_filename"}}}

    @classmethod
    def _set_dummy_master(cls) -> tk.Frame | None:
        """ Add an option to force black font on Linux file dialogs KDE issue that displays light
        font on white background).

        This is a pretty hacky solution, but tkinter does not allow direct editing of file dialogs,
        so we create a dummy frame and add the foreground option there, so that the file dialog can
        inherit the foreground.

        Returns
        -------
        tkinter.Frame or ``None``
            The dummy master frame for Linux systems, otherwise ``None``
        """
        if platform.system().lower() == "linux":
            frame = tk.Frame()
            frame.option_add("*foreground", "black")
            retval: tk.Frame | None = frame
        else:
            retval = None
        return retval

    def _remove_dummy_master(self) -> None:
        """ Destroy the dummy master widget on Linux systems. """
        if platform.system().lower() != "linux" or self._dummy_master is None:
            return
        self._dummy_master.destroy()
        del self._dummy_master
        self._dummy_master = None

    def _set_defaults(self) -> dict[str, str | None]:
        """ Set the default file type for the file dialog. Generally the first found file type
        will be used, but this is overridden if it is not appropriate.

        Returns
        -------
        dict:
            The default file extension for each file type
        """
        defaults: dict[str, str | None] = {
            key: next(ext for ext in val[0][1].split(" ")).replace("*", "")
            for key, val in self._filetypes.items()}
        defaults["default"] = None
        defaults["video"] = ".mp4"
        defaults["image"] = ".png"
        logger.debug(defaults)
        return defaults

    def _set_kwargs(self,
                    title: str | None,
                    initial_folder: str | None,
                    initial_file: str | None,
                    file_type: _FILETYPE | None,
                    command: str | None,
                    action: str | None,
                    variable: str | None,
                    parent: tk.Frame | None
                    ) -> dict[str, None | tk.Frame | str | list[tuple[str, str]]]:
        """ Generate the required kwargs for the requested file dialog browser.

        Parameters
        ----------
        title: str
            The title to display on the file dialog. If `None` then the default title will be used.
        initial_folder: str
            The folder to initially open with the file dialog. If `None` then tkinter will decide.
        initial_file: str
            The filename to set with the file dialog. If `None` then tkinter no initial filename
            is.
        file_type: ['default', 'alignments', 'config_project', 'config_task', 'config_all', \
                    'csv',  'image', 'ini', 'state', 'log', 'video'] or ``None``
            The type of file that this dialog is for. `default` allows selection of any files.
            Other options limit the file type selection
        command: str
            Required for context handling file dialog, otherwise unused.
        action: str
            Required for context handling file dialog, otherwise unused.
        variable: str, optional
            Required for context handling file dialog, otherwise unused. The variable to associate
            with this file dialog. Default: ``None``
        parent: :class:`tkinter.Frame`
            The parent that is launching the file dialog. ``None`` sets this to root

        Returns
        -------
        dict:
            The key word arguments for the file dialog to be launched
        """
        logger.debug("Setting Kwargs: (title: %s, initial_folder: %s, initial_file: '%s', "
                     "file_type: '%s', command: '%s': action: '%s', variable: '%s', parent: %s)",
                     title, initial_folder, initial_file, file_type, command, action, variable,
                     parent)

        kwargs: dict[str, None | tk.Frame | str | list[tuple[str, str]]] = {
            "master": self._dummy_master}

        if self._handletype.lower() == "context":
            assert command is not None and action is not None and variable is not None
            self._set_context_handletype(command, action, variable)

        if title is not None:
            kwargs["title"] = title

        if initial_folder is not None:
            kwargs["initialdir"] = initial_folder

        if initial_file is not None:
            kwargs["initialfile"] = initial_file

        if parent is not None:
            kwargs["parent"] = parent

        if self._handletype.lower() in (
                "open", "save", "filename", "filename_multi", "save_filename"):
            assert file_type is not None
            kwargs["filetypes"] = self._filetypes[file_type]
            if self._defaults.get(file_type):
                kwargs['defaultextension'] = self._defaults[file_type]
        if self._handletype.lower() == "save":
            kwargs["mode"] = "w"
        if self._handletype.lower() == "open":
            kwargs["mode"] = "r"
        logger.debug("Set Kwargs: %s", kwargs)
        return kwargs

    def _set_context_handletype(self, command: str, action: str, variable: str) -> None:
        """ Sets the correct handle type  based on context.

        Parameters
        ----------
        command: str
            The command that is being executed. Used to look up the context actions
        action: str
            The action that is being performed. Used to look up the correct file dialog
        variable: str
            The variable associated with this file dialog
        """
        if self._contexts[command].get(variable, None) is not None:
            handletype = T.cast(dict[str, dict[str, dict[str, str]]],
                                self._contexts)[command][variable][action]
        else:
            handletype = T.cast(dict[str, dict[str, str]],
                                self._contexts)[command][action]
        logger.debug(handletype)
        self._handletype = T.cast(_HANDLETYPE, handletype)

    def _open(self) -> T.IO | None:
        """ Open a file. """
        logger.debug("Popping Open browser")
        return filedialog.askopenfile(**self._kwargs)  # type: ignore

    def _save(self) -> T.IO | None:
        """ Save a file. """
        logger.debug("Popping Save browser")
        return filedialog.asksaveasfile(**self._kwargs)  # type: ignore

    def _dir(self) -> str:
        """ Get a directory location. """
        logger.debug("Popping Dir browser")
        return filedialog.askdirectory(**self._kwargs)  # type: ignore

    def _savedir(self) -> str:
        """ Get a save directory location. """
        logger.debug("Popping SaveDir browser")
        return filedialog.askdirectory(**self._kwargs)  # type: ignore

    def _filename(self) -> str:
        """ Get an existing file location. """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilename(**self._kwargs)  # type: ignore

    def _filename_multi(self) -> tuple[str, ...]:
        """ Get multiple existing file locations. """
        logger.debug("Popping Filename browser")
        return filedialog.askopenfilenames(**self._kwargs)  # type: ignore

    def _save_filename(self) -> str:
        """ Get a save file location. """
        logger.debug("Popping Save Filename browser")
        return filedialog.asksaveasfilename(**self._kwargs)  # type: ignore

    @staticmethod
    def _nothing() -> None:  # pylint:disable=useless-return
        """ Method that does nothing, used for disabling open/save pop up.  """
        logger.debug("Popping Nothing browser")
        return
