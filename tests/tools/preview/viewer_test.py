#!/usr/bin python3
""" Pytest unit tests for :mod:`tools.preview.viewer` """
from __future__ import annotations
import tkinter as tk
import typing as T

from tkinter import ttk

from unittest.mock import MagicMock

import pytest
import pytest_mock
import numpy as np
from PIL import ImageTk

from lib.logger import log_setup
# Need to setup logging to avoid trace/verbose errors
log_setup("DEBUG", "pytest_viewer.log", "PyTest, False")

from lib.utils import get_backend  # pylint:disable=wrong-import-position  # noqa
from tools.preview.viewer import _Faces, FacesDisplay, ImagesCanvas  # pylint:disable=wrong-import-position  # noqa

if T.TYPE_CHECKING:
    from lib.align.aligned_face import CenteringType


# pylint:disable=protected-access


def test__faces():
    """ Test the :class:`~tools.preview.viewer._Faces dataclass initializes correctly """
    faces = _Faces()
    assert faces.filenames == []
    assert faces.matrix == []
    assert faces.src == []
    assert faces.dst == []


_PARAMS = [(3, 448), (4, 333), (5, 254), (6, 128)]  # columns/face_size
_IDS = [f"cols:{c},size:{s}[{get_backend().upper()}]" for c, s in _PARAMS]


class TestFacesDisplay():
    """ Test :class:`~tools.preview.viewer.FacesDisplay """
    _padding = 64

    def get_faces_display_instance(self, columns: int = 5, face_size: int = 256) -> FacesDisplay:
        """ Obtain an instance of :class:`~tools.preview.viewer.FacesDisplay` with the given column
        and face size layout.

        Parameters
        ----------
        columns: int, optional
            The number of columns to display in the viewer, default: 5
        face_size: int, optional
            The size of each face image to be displayed in the viewer, default: 256

        Returns
        -------
        :class:`~tools.preview.viewer.FacesDisplay`
            An instance of the FacesDisplay class at the given settings
        """
        app = MagicMock()
        retval = FacesDisplay(app, face_size, self._padding)
        retval._faces = _Faces(
            matrix=[np.random.rand(2, 3) for _ in range(columns)],
            src=[np.random.rand(face_size, face_size, 3) for _ in range(columns)],
            dst=[np.random.rand(face_size, face_size, 3) for _ in range(columns)])
        return retval

    def test_init(self) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` __init__ method """
        f_display = self.get_faces_display_instance(face_size=256)
        assert f_display._size == 256
        assert f_display._padding == self._padding
        assert isinstance(f_display._app, MagicMock)

        assert f_display._display_dims == (1, 1)
        assert isinstance(f_display._faces, _Faces)

        assert f_display._centering is None
        assert f_display._faces_source.size == 0
        assert f_display._faces_dest.size == 0
        assert f_display._tk_image is None
        assert f_display.update_source is False
        assert not f_display.source and isinstance(f_display.source, list)
        assert not f_display.destination and isinstance(f_display.destination, list)

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test__total_columns(self, columns: int, face_size: int) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` _total_columns property is correctly
        calculated

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display.source = [None for _ in range(columns)]  # type:ignore
        assert f_display._total_columns == columns

    def test_set_centering(self) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` set_centering method """
        f_display = self.get_faces_display_instance()
        assert f_display._centering is None
        centering: CenteringType = "legacy"
        f_display.set_centering(centering)
        assert f_display._centering == centering

    def test_set_display_dimensions(self) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` set_display_dimensions method """
        f_display = self.get_faces_display_instance()
        assert f_display._display_dims == (1, 1)
        dimensions = (800, 600)
        f_display.set_display_dimensions(dimensions)
        assert f_display._display_dims == dimensions

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test_update_tk_image(self,
                             columns: int,
                             face_size: int,
                             mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` update_tk_image method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking _build_faces_image method called
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display._build_faces_image = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        f_display._get_scale_size = T.cast(MagicMock,  # type:ignore
                                           mocker.MagicMock(return_value=(128, 128)))
        f_display._faces_source = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        f_display._faces_dest = np.zeros((face_size, face_size, 3), dtype=np.uint8)

        tk.Tk()  # tkinter instance needed for image creation
        f_display.update_tk_image()

        f_display._build_faces_image.assert_called_once()
        f_display._get_scale_size.assert_called_once()
        assert isinstance(f_display._tk_image, ImageTk.PhotoImage)
        assert f_display._tk_image.width() == 128
        assert f_display._tk_image.height() == 128
        assert f_display.tk_image == f_display._tk_image  # public property test

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test_get_scale_size(self, columns: int, face_size: int) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` get_scale_size method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display.set_display_dimensions((800, 600))

        img = np.zeros((face_size, face_size, 3), dtype=np.uint8)
        size = f_display._get_scale_size(img)
        assert size == (600, 600)

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test__build_faces_image(self,
                                columns: int,
                                face_size: int,
                                mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` _build_faces_image method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking internal methods called
        """
        header_size = 32

        f_display = self.get_faces_display_instance(columns, face_size)
        f_display._faces_from_frames = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        f_display._header_text = T.cast(  # type:ignore
            MagicMock,
            mocker.MagicMock(return_value=np.random.rand(header_size, face_size * columns, 3)))
        f_display._draw_rect = T.cast(MagicMock,  # type:ignore
                                      mocker.MagicMock(side_effect=lambda x: x))

        # Test full update
        f_display.update_source = True
        f_display._build_faces_image()

        f_display._faces_from_frames.assert_called_once()
        f_display._header_text.assert_called_once()
        assert f_display._draw_rect.call_count == columns * 2  # src + dst
        assert f_display._faces_source.shape == (face_size + header_size, face_size * columns, 3)
        assert f_display._faces_dest.shape == (face_size, face_size * columns, 3)

        f_display._faces_from_frames.reset_mock()
        f_display._header_text.reset_mock()
        f_display._draw_rect.reset_mock()

        # Test dst update only
        f_display.update_source = False
        f_display._build_faces_image()

        f_display._faces_from_frames.assert_called_once()
        assert not f_display._header_text.called
        assert f_display._draw_rect.call_count == columns  # dst only
        assert f_display._faces_dest.shape == (face_size, face_size * columns, 3)

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test_faces__from_frames(self,
                                columns,
                                face_size,
                                mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` _from_frames method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking _build_faces_image method called
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display.source = [mocker.MagicMock() for _ in range(3)]
        f_display.destination = [np.random.rand(face_size, face_size, 3) for _ in range(3)]
        f_display._crop_source_faces = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        f_display._crop_destination_faces = T.cast(MagicMock, mocker.MagicMock())  # type:ignore

        # Both src + dst
        f_display.update_source = True
        f_display._faces_from_frames()
        f_display._crop_source_faces.assert_called_once()
        f_display._crop_destination_faces.assert_called_once()

        f_display._crop_source_faces.reset_mock()
        f_display._crop_destination_faces.reset_mock()

        # Just dst
        f_display.update_source = False
        f_display._faces_from_frames()
        assert not f_display._crop_source_faces.called
        f_display._crop_destination_faces.assert_called_once()

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test__crop_source_faces(self,
                                columns: int,
                                face_size: int,
                                monkeypatch: pytest.MonkeyPatch,
                                mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` _crop_source_faces method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        monkeypatch: :class:`pytest.MonkeyPatch`
            For patching the transform_image function
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for mocking various internal methods
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display._centering = "face"
        f_display.update_source = True
        f_display._faces.src = []

        transform_image_mock = mocker.MagicMock()
        monkeypatch.setattr("tools.preview.viewer.transform_image", transform_image_mock)

        f_display.source = [mocker.MagicMock() for _ in range(columns)]
        for idx, mock in enumerate(f_display.source):
            assert isinstance(mock, MagicMock)
            mock.inbound.detected_faces.__getitem__ = lambda self, x, y=mock: y
            mock.aligned.matrix = f"test_matrix_{idx}"
            mock.inbound.filename = f"test_filename_{idx}.txt"

        f_display._crop_source_faces()

        assert len(f_display._faces.filenames) == columns
        assert len(f_display._faces.matrix) == columns
        assert len(f_display._faces.src) == columns
        assert not f_display.update_source
        assert transform_image_mock.call_count == columns

        for idx in range(columns):
            assert f_display._faces.filenames[idx] == f"test_filename_{idx}"
            assert f_display._faces.matrix[idx] == f"test_matrix_{idx}"

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test__crop_destination_faces(self,
                                     columns: int,
                                     face_size: int,
                                     mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` _crop_destination_faces method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for dummying in full frames
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display._centering = "face"
        f_display._faces.dst = []  # empty object and test populated correctly

        f_display.source = [mocker.MagicMock() for _ in range(columns)]
        for item in f_display.source:  # type ignore
            item.inbound.image = np.random.rand(1280, 720, 3)  # type:ignore

        f_display._crop_destination_faces()
        assert len(f_display._faces.dst) == columns
        assert all(f.shape == (face_size, face_size, 3) for f in f_display._faces.dst)

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test__header_text(self,
                          columns: int,
                          face_size: int,
                          mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` _header_text method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for dummying in cv2 calls
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        f_display.source = [None for _ in range(columns)]  # type:ignore
        f_display._faces.filenames = [f"filename_{idx}.png" for idx in range(columns)]

        cv2_mock = mocker.patch("tools.preview.viewer.cv2")
        text_width, text_height = (100, 32)
        cv2_mock.getTextSize.return_value = [(text_width, text_height), ]

        header_box = f_display._header_text()
        assert cv2_mock.getTextSize.call_count == columns
        assert cv2_mock.putText.call_count == columns
        assert header_box.shape == (face_size // 8, face_size * columns, 3)

    @pytest.mark.parametrize("columns, face_size", _PARAMS, ids=_IDS)
    def test__draw_rect_text(self,
                             columns: int,
                             face_size: int,
                             mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.FacesDisplay` _draw_rect method

        Parameters
        ----------
        columns: int
            The number of columns to display in the viewer
        face_size: int
            The size of each face image to be displayed in the viewer
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for dummying in cv2 calls
        """
        f_display = self.get_faces_display_instance(columns, face_size)
        cv2_mock = mocker.patch("tools.preview.viewer.cv2")

        image = (np.random.rand(face_size, face_size, 3) * 255.0) + 50
        assert image.max() > 255.0
        output = f_display._draw_rect(image)
        cv2_mock.rectangle.assert_called_once()
        assert output.max() == 255.0  # np.clip


class TestImagesCanvas:
    """ Test :class:`~tools.preview.viewer.ImagesCanvas` """

    @pytest.fixture
    def parent(self) -> MagicMock:
        """ Mock object to act as the parent widget to the ImagesCanvas

        Returns
        --------
        :class:`unittest.mock.MagicMock`
            The mocked ttk.PanedWindow widget
        """
        retval = MagicMock(spec=ttk.PanedWindow)
        retval.tk = retval
        retval._w = "mock_ttkPanedWindow"
        retval.children = {}
        retval.call = retval
        retval.createcommand = retval
        retval.preview_display = MagicMock(spec=FacesDisplay)
        return retval

    @pytest.fixture(name="images_canvas_instance")
    def images_canvas_fixture(self, parent) -> ImagesCanvas:
        """ Fixture for creating a testing :class:`~tools.preview.viewer.ImagesCanvas` instance

        Parameters
        ----------
        parent: :class:`unittest.mock.MagicMock`
            The mocked ttk.PanedWindow parent

        Returns
        -------
        :class:`~tools.preview.viewer.ImagesCanvas`
            The class instance for testing
        """
        app = MagicMock()
        return ImagesCanvas(app, parent)

    def test_init(self, images_canvas_instance: ImagesCanvas, parent: MagicMock) -> None:
        """ Test :class:`~tools.preview.viewer.ImagesCanvas` __init__ method

        Parameters
        ----------
        images_canvas_instance: :class:`~tools.preview.viewer.ImagesCanvas`
            The class instance to test
        parent: :class:`unittest.mock.MagicMock`
            The mocked parent ttk.PanedWindow
         """
        assert images_canvas_instance._display == parent.preview_display
        assert isinstance(images_canvas_instance._canvas, tk.Canvas)
        assert images_canvas_instance._canvas.master == images_canvas_instance
        assert images_canvas_instance._canvas.winfo_ismapped()

    def test_resize(self,
                    images_canvas_instance: ImagesCanvas,
                    parent: MagicMock,
                    mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.ImagesCanvas` resize method

        Parameters
        ----------
        images_canvas_instance: :class:`~tools.preview.viewer.ImagesCanvas`
            The class instance to test
        parent: :class:`unittest.mock.MagicMock`
            The mocked parent ttk.PanedWindow
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for dummying in tk calls
        """
        event_mock = mocker.MagicMock(spec=tk.Event, width=100, height=200)
        images_canvas_instance.reload = T.cast(MagicMock, mocker.MagicMock())  # type:ignore

        images_canvas_instance._resize(event_mock)

        parent.preview_display.set_display_dimensions.assert_called_once_with((100, 200))
        images_canvas_instance.reload.assert_called_once()

    def test_reload(self,
                    images_canvas_instance: ImagesCanvas,
                    parent: MagicMock,
                    mocker: pytest_mock.MockerFixture) -> None:
        """ Test :class:`~tools.preview.viewer.ImagesCanvas` reload method

        Parameters
        ----------
        images_canvas_instance: :class:`~tools.preview.viewer.ImagesCanvas`
            The class instance to test
        parent: :class:`unittest.mock.MagicMock`
            The mocked parent ttk.PanedWindow
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for dummying in tk calls
        """
        itemconfig_mock = mocker.patch.object(tk.Canvas, "itemconfig")

        images_canvas_instance.reload()

        parent.preview_display.update_tk_image.assert_called_once()
        itemconfig_mock.assert_called_once()
