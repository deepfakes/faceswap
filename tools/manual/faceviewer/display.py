#!/usr/bin/env python3
""" Handle the highlighting, filtering and display characteristics of objects in the Face viewer
for the manual adjustments tool """
import logging
import platform

from lib.gui.custom_widgets import RightClickMenu

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ContextMenu():  # pylint:disable=too-few-public-methods
    """  Enables a right click context menu for the :class:`~tool.manual.FacesViewer`.

    Parameters
    ----------
    canvas: :class:`tkinter.Canvas`
        The :class:`~tools.manual.FacesViewer` canvas
    """
    def __init__(self, canvas):
        logger.debug("Initializing: %s (canvas: %s)", self.__class__.__name__, canvas)
        self._canvas = canvas
        self._menu = RightClickMenu(["Delete Face"], [self._delete_face])
        self._face_id = None
        self._canvas.bind("<Button-2>" if platform.system() == "Darwin" else "<Button-3>",
                          self._pop_menu)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _pop_menu(self, event):
        """ Pop up the context menu on a right click mouse event.

        Parameters
        ----------
        event: :class:`tkinter.Event`
            The mouse event that has triggered the pop up menu
        """
        coords = (self._canvas.canvasx(event.x), self._canvas.canvasy(event.y))
        self._face_id = next((idx for idx in self._canvas.find_overlapping(*coords, *coords)
                              if self._canvas.type(idx) == "image"), None)
        if self._face_id is None:
            logger.trace("No valid item under mouse")
            return
        logger.trace("Popping right click menu")
        self._menu.popup(event)

    def _delete_face(self):
        """ Delete the selected face on a right click mouse delete action. """
        logger.trace("Right click delete received. face_id: %s", self._face_id)
        self._canvas.update_face.remove_face_from_viewer(self._face_id)
        self._face_id = None
