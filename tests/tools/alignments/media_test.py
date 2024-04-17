#!/usr/bin python3
""" Pytest unit tests for :mod:`tools.alignments.media` """
from __future__ import annotations
import os
import typing as T

from operator import itemgetter
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import pytest_mock

from lib.logger import log_setup
# Need to setup logging to avoid trace/verbose errors
log_setup("DEBUG", f"{__name__}.log", "PyTest, False")

# pylint:disable=wrong-import-position,protected-access
from lib.utils import FaceswapError  # noqa:E402
from tools.alignments.media import (AlignmentData, Faces, ExtractedFaces,  # noqa:E402
                                    Frames, MediaLoader)

if T.TYPE_CHECKING:
    from collections.abc import Generator


class TestAlignmentData:
    """ Test for :class:`~tools.alignments.media.AlignmentData` """

    @pytest.fixture
    def alignments_file(self, tmp_path: str) -> Generator[str, None, None]:
        """ Fixture for creating dummy alignments files

        Parameters
        ----------
        tmp_path: str
            pytest temporary path to generate folders

        Yields
        ------
        str
            Path to a dummy alignments file
        """
        alignments_file = os.path.join(tmp_path, "alignments.fsa")
        with open(alignments_file, "w", encoding="utf8") as afile:
            afile.write("test")
        yield alignments_file
        os.remove(alignments_file)

    def test_init(self,
                  alignments_file: str,
                  mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.AlignmentData` __init__ method

        Parameters
        ----------
        alignments_file: str
            The temporarily generated alignments file
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking the superclass __init__
        """
        alignments_parent_init = mocker.patch("tools.alignments.media.Alignments.__init__")
        mocker.patch("tools.alignments.media.Alignments.frames_count",
                     new_callable=mocker.PropertyMock(return_value=20))

        AlignmentData(alignments_file)
        folder, filename = os.path.split(alignments_file)
        alignments_parent_init.assert_called_once_with(folder, filename=filename)

    def test_check_file_exists(self, alignments_file: str) -> None:
        """ Test for :class:`~tools.alignments.media.AlignmentData` _check_file_exists method

        Parameters
        ----------
        alignments_file: str
            The temporarily generated alignments file
        """
        assert AlignmentData.check_file_exists(alignments_file) == os.path.split(alignments_file)
        fake_file = "/not/possibly/a/real/path/alignments.fsa"
        with pytest.raises(SystemExit):
            AlignmentData.check_file_exists(fake_file)

    def test_save(self,
                  alignments_file: str,
                  mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.AlignmentData`save method

        Parameters
        ----------
        alignments_file: str
            The temporarily generated alignments file
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking the superclass calls
        """
        mocker.patch("tools.alignments.media.Alignments.__init__")
        mocker.patch("tools.alignments.media.Alignments.frames_count",
                     new_callable=mocker.PropertyMock(return_value=20))
        alignments_parent_backup = mocker.patch("tools.alignments.media.Alignments.backup")
        alignments_parent_save = mocker.patch("tools.alignments.media.Alignments.save")
        align_data = AlignmentData(alignments_file)
        align_data.save()
        alignments_parent_backup.assert_called_once()
        alignments_parent_save.assert_called_once()


@pytest.fixture(name="folder")
def folder_fixture(tmp_path: str) -> Generator[str, None, None]:
    """ Fixture for creating dummy folders

    Parameters
    ----------
    tmp_path: str
        pytest temporary path to generate folders

    Yields
    ------
    str
        Path to a dummy folder
    """
    folder = os.path.join(tmp_path, "images")
    os.mkdir(folder)
    for fname in (["a.png", "b.png"]):
        with open(os.path.join(folder, fname), "wb"):
            pass
    yield folder
    for fname in (["a.png", "b.png"]):
        os.remove(os.path.join(folder, fname))
    os.rmdir(folder)


class TestMediaLoader:
    """ Test for :class:`~tools.alignments.media.MediaLoader` """

    @pytest.fixture(name="media_loader_instance")
    def media_loader_fixture(self,
                             folder: str,
                             mocker: pytest_mock.MockerFixture) -> MediaLoader:
        """ An instance of :class:`~tools.alignments.media.MediaLoader` with unimplemented
        child methods patched out of __init__ and initialized with a dummy folder containing
        2 images

        Parameters
        ----------
        folder : str
            Dummy media folder
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking subclass calls

        Returns
        -------
        :class:`~tools.alignments.media.MediaLoader`
            Initialized instance for testing
        """
        mocker.patch("tools.alignments.media.MediaLoader.sorted_items",
                     return_value=os.listdir(folder))
        mocker.patch("tools.alignments.media.MediaLoader.load_items")
        loader = MediaLoader(folder)
        return loader

    def test_init(self,
                  folder: str,
                  mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.MediaLoader`__init__ method

        Parameters
        ----------
        folder : str
            Dummy media folder
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking subclass calls
        """
        sort_patch = mocker.patch("tools.alignments.media.MediaLoader.sorted_items",
                                  return_value=os.listdir(folder))
        load_patch = mocker.patch("tools.alignments.media.MediaLoader.load_items")
        loader = MediaLoader(folder)
        sort_patch.assert_called_once()
        load_patch.assert_called_once()
        assert loader.folder == folder
        assert loader._count == 2
        assert loader.count == 2
        assert not loader.is_video

    def test_check_input_folder(self, media_loader_instance: MediaLoader) -> None:
        """ Test for :class:`~tools.alignments.media.MediaLoader` check_input_folder method

        Parameters
        ----------
        media_loader_instance: :class:`~tools.alignments.media.MediaLoader`
            The class instance for testing
        """
        media_loader = media_loader_instance
        assert media_loader.check_input_folder() is None
        media_loader.folder = ""
        with pytest.raises(SystemExit):
            media_loader.check_input_folder()
        media_loader.folder = "/this/path/does/not/exist"
        with pytest.raises(SystemExit):
            media_loader.check_input_folder()

    def test_valid_extension(self, media_loader_instance: MediaLoader) -> None:
        """ Test for :class:`~tools.alignments.media.MediaLoader` valid_extension method

        Parameters
        ----------
        media_loader_instance: :class:`~tools.alignments.media.MediaLoader`
            The class instance for testing
        """
        media_loader = media_loader_instance
        assert media_loader.valid_extension("test.png")
        assert media_loader.valid_extension("test.PNG")
        assert media_loader.valid_extension("test.jpg")
        assert media_loader.valid_extension("test.JPG")
        assert not media_loader.valid_extension("test.doc")
        assert not media_loader.valid_extension("test.txt")
        assert not media_loader.valid_extension("test.mp4")

    def test_load_image(self,
                        media_loader_instance: MediaLoader,
                        mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.MediaLoader` load_image method

        Parameters
        ----------
        media_loader_instance: :class:`~tools.alignments.media.MediaLoader`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking loader specific calls
        """
        media_loader = media_loader_instance
        expected = np.random.rand(256, 256, 3)
        media_loader.load_video_frame = T.cast(MagicMock,  # type:ignore
                                               mocker.MagicMock(return_value=expected))
        read_image_patch = mocker.patch("tools.alignments.media.read_image", return_value=expected)
        filename = "test.png"
        output = media_loader.load_image(filename)
        np.testing.assert_equal(expected, output)
        read_image_patch.assert_called_once_with(os.path.join(media_loader.folder, filename),
                                                 raise_error=True)

        mocker.patch("tools.alignments.media.MediaLoader.is_video",
                     new_callable=mocker.PropertyMock(return_value=True))
        filename = "test.mp4"
        output = media_loader.load_image(filename)
        np.testing.assert_equal(expected, output)
        media_loader.load_video_frame.assert_called_once_with(filename)

    def test_load_video_frame(self,
                              media_loader_instance: MediaLoader,
                              mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.MediaLoader` load_video_frame method

        Parameters
        ----------
        media_loader_instance: :class:`~tools.alignments.media.MediaLoader`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking cv2 calls
        """
        media_loader = media_loader_instance
        filename = "test_0001.png"
        with pytest.raises(AssertionError):
            media_loader.load_video_frame(filename)

        mocker.patch("tools.alignments.media.MediaLoader.is_video",
                     new_callable=mocker.PropertyMock(return_value=True))
        expected = np.random.rand(256, 256, 3)
        vid_cap = mocker.MagicMock(cv2.VideoCapture)
        vid_cap.read.side_effect = ((1, expected), )

        media_loader._vid_reader = T.cast(MagicMock,  vid_cap)  # type:ignore
        output = media_loader.load_video_frame(filename)
        vid_cap.set.assert_called_once()
        np.testing.assert_equal(output, expected)

    def test_stream(self,
                    media_loader_instance: MediaLoader,
                    mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.MediaLoader` stream method

        Parameters
        ----------
        media_loader_instance: :class:`~tools.alignments.media.MediaLoader`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking loader specific calls
        """
        media_loader = media_loader_instance

        loader = mocker.patch("tools.alignments.media.ImagesLoader.load")
        expected = [(fname, np.random.rand(256, 256, 3))
                    for fname in os.listdir(media_loader.folder)]
        loader.side_effect = [expected]
        output = list(media_loader.stream())
        assert output == expected

        loader.reset_mock()

        skip_list = [0]
        expected = [expected[1]]
        loader.side_effect = [expected]
        output = list(media_loader.stream(skip_list))
        assert output == expected
        assert loader.add_skip_list.called_once_with(skip_list)

    def test_save_image(self,
                        media_loader_instance: MediaLoader,
                        mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.MediaLoader` save_image method

        Parameters
        ----------
        media_loader_instance: :class:`~tools.alignments.media.MediaLoader`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking saver specific calls
        """
        media_loader = media_loader_instance
        out_folder = media_loader.folder
        filename = "test_out.jpg"
        expected_filename = os.path.join(media_loader.folder, "test_out.png")
        img = np.random.rand(256, 256, 3)
        metadata = {"test": "data"}

        cv2_write_mock = mocker.patch("cv2.imwrite")
        cv2_encode_mock = mocker.patch("cv2.imencode")
        png_write_meta_mock = mocker.patch("tools.alignments.media.png_write_meta")
        open_mock = mocker.patch("builtins.open")

        media_loader.save_image(out_folder, filename, img, metadata=None)
        cv2_write_mock.assert_called_once_with(expected_filename, img)
        cv2_encode_mock.assert_not_called()
        png_write_meta_mock.assert_not_called()

        cv2_write_mock.reset_mock()

        media_loader.save_image(out_folder, filename, img, metadata=metadata)  # type:ignore
        cv2_write_mock.assert_not_called()
        cv2_encode_mock.assert_called_once_with(".png", img)
        png_write_meta_mock.assert_called_once()
        open_mock.assert_called_once()


class TestFaces:
    """ Test for :class:`~tools.alignments.media.Faces` """

    @pytest.fixture(name="faces_instance")
    def faces_fixture(self,
                      folder: str,
                      mocker: pytest_mock.MockerFixture) -> Faces:
        """ An instance of :class:`~tools.alignments.media.Faces` patching out
        read_image_meta_batch so nothing is loaded

        Parameters
        ----------
        folder : str
            Dummy media folder
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking read_image_meta_batch calls

        Returns
        -------
        :class:`~tools.alignments.media.Faces`
            Initialized instance for testing
        """
        mocker.patch("tools.alignments.media.read_image_meta_batch")
        loader = Faces(folder, None)
        return loader

    def test_init(self,
                  folder: str,
                  mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.Faces`__init__ method

        Parameters
        ----------
        folder : str
            Dummy media folder
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking superclass calls
        """
        parent_mock = mocker.patch("tools.alignments.media.super")
        alignments_mock = mocker.patch("tools.alignments.media.AlignmentData")
        Faces(folder, alignments_mock)
        parent_mock.assert_called_once()

    def test__handle_legacy(self,
                            faces_instance: Faces,
                            mocker: pytest_mock.MockerFixture,
                            caplog: pytest.LogCaptureFixture) -> None:
        """ Test for :class:`~tools.alignments.media.Faces` _handle_legacy method

        Parameters
        ----------
        faces_instance: :class:`~tools.alignments.media.Faces`
            Test class instance
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking various objects
        caplog: :class:`pytest.LogCaptureFixture
            For capturing logging messages
        """
        faces = faces_instance
        folder = faces.folder
        legacy_file = os.path.join(folder, "a.png")

        # No alignments file
        with pytest.raises(FaceswapError):
            faces._handle_legacy(legacy_file)

        # No returned metadata
        alignments_mock = mocker.patch("tools.alignments.media.AlignmentData")
        alignments_mock.version = 2.1
        update_mock = mocker.patch("tools.alignments.media.update_legacy_png_header",
                                   return_value={})
        faces = Faces(folder, alignments_mock)
        faces.folder = folder
        with pytest.raises(FaceswapError):
            faces._handle_legacy(legacy_file)
        update_mock.assert_called_once_with(legacy_file, alignments_mock)

        # Correct data with logging
        caplog.clear()
        update_mock.reset_mock()
        update_mock.return_value = {"test": "data"}
        faces._handle_legacy(legacy_file, log=True)
        assert "Legacy faces discovered" in caplog.text

        # Correct data without logging
        caplog.clear()
        update_mock.reset_mock()
        update_mock.return_value = {"test": "data"}
        faces._handle_legacy(legacy_file, log=False)
        assert "Legacy faces discovered" not in caplog.text

    def test__handle_duplicate(self, faces_instance: Faces) -> None:
        """ Test for :class:`~tools.alignments.media.Faces` _handle_duplicate method

        Parameters
        ----------
        faces_instance: :class:`~tools.alignments.media.Faces`
            The class instance for testing
        """
        faces = faces_instance
        dupe_dir = os.path.join(faces.folder, "_duplicates")
        src_filename = "test_0001.png"
        src_face_idx = 0
        paths = [os.path.join(faces.folder, fname) for fname in os.listdir(faces.folder)]
        data = {"source": {"source_filename": src_filename,
                           "face_index": src_face_idx}}
        seen: dict[str, list[int]] = {}

        # New item
        is_dupe = faces._handle_duplicate(paths[0], data, seen)  # type:ignore
        assert src_filename in seen and seen[src_filename] == [src_face_idx]
        assert not os.path.exists(dupe_dir)
        assert not is_dupe

        # Dupe item
        is_dupe = faces._handle_duplicate(paths[1], data, seen)  # type:ignore
        assert src_filename in seen and seen[src_filename] == [src_face_idx]
        assert len(seen) == 1
        assert os.path.exists(dupe_dir)
        assert not os.path.exists(paths[1])
        assert is_dupe

        # Move everything back for fixture cleanup
        os.rename(os.path.join(dupe_dir, os.path.basename(paths[1])), paths[1])
        os.rmdir(dupe_dir)

    def test_process_folder(self,
                            faces_instance: Faces,
                            mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.Faces` process_folder method

        Parameters
        ----------
        faces_instance: :class:`~tools.alignments.media.Faces`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking various logic calls
        """
        faces = faces_instance
        read_image_meta_mock = mocker.patch("tools.alignments.media.read_image_meta_batch")
        img_sources = [os.path.join(faces.folder, fname) for fname in os.listdir(faces.folder)]
        meta_data = {"itxt": {"source": ({"source_filename": "data.png"})}}
        expected = [(fname, meta_data["itxt"]) for fname in os.listdir(faces.folder)]
        read_image_meta_mock.side_effect = [[(src, meta_data) for src in img_sources]]

        legacy_mock = mocker.patch("tools.alignments.media.Faces._handle_legacy",
                                   return_value=meta_data["itxt"])
        dupe_mock = mocker.patch("tools.alignments.media.Faces._handle_duplicate",
                                 return_value=False)

        # valid itxt
        output = list(faces.process_folder())
        assert read_image_meta_mock.call_count == 1
        assert dupe_mock.call_count == 2
        assert not legacy_mock.called
        assert output == expected

        dupe_mock.reset_mock()
        read_image_meta_mock.reset_mock()

        # valid itxt with alignemnts data
        read_image_meta_mock.side_effect = [[(src, meta_data) for src in img_sources]]
        faces._alignments = mocker.MagicMock(AlignmentData)
        faces._alignments.version = 2.1  # type:ignore
        output = list(faces.process_folder())
        assert faces._alignments.frame_exists.call_count == 2  # type:ignore
        assert read_image_meta_mock.call_count == 1
        assert dupe_mock.call_count == 2

        dupe_mock.reset_mock()
        read_image_meta_mock.reset_mock()
        faces._alignments = None

        # invalid itxt
        read_image_meta_mock.side_effect = [[(src, {}) for src in img_sources]]
        output = list(faces.process_folder())
        assert read_image_meta_mock.call_count == 1
        assert legacy_mock.call_count == 2
        assert dupe_mock.call_count == 2
        assert output == expected

    def test_load_items(self,
                        faces_instance: Faces) -> None:
        """ Test for :class:`~tools.alignments.media.Faces` load_items method

        Parameters
        ----------
        faces_instance: :class:`~tools.alignments.media.Faces`
            The class instance for testing
        """
        faces = faces_instance
        data = [(f"file{idx}.png", {"source": {"source_filename": f"src{idx}.png",
                                               "face_index": 0}})
                for idx in range(4)]
        faces.file_list_sorted = data  # type: ignore
        expected = {"src0.png": [0], "src1.png": [0], "src2.png": [0], "src3.png": [0]}
        result = faces.load_items()
        assert result == expected

        data = [(f"file{idx}.png", {"source": {"source_filename": f"src{idx // 2}.png",
                                               "face_index": 0 if idx % 2 == 0 else 1}})
                for idx in range(4)]
        faces.file_list_sorted = data  # type: ignore
        expected = {"src0.png": [0, 1], "src1.png": [0, 1]}
        result = faces.load_items()
        assert result == expected

    def test_sorted_items(self,
                          faces_instance: Faces,
                          mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.Faces` sorted_items method

        Parameters
        ----------
        faces_instance: :class:`~tools.alignments.media.Faces`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking various logic calls
        """
        faces = faces_instance
        data: list[tuple[str, dict]] = [("file4.png", {}), ("file3.png", {}),
                                        ("file1.png", {}), ("file2.png", {})]
        expected = sorted(data)
        process_folder_mock = mocker.patch("tools.alignments.media.Faces.process_folder",
                                           side_effect=[data])
        result = faces.sorted_items()
        assert process_folder_mock.called
        assert result == expected


class TestFrames:
    """ Test for :class:`~tools.alignments.media.Frames` """

    def test_process_folder(self,
                            folder: str,
                            mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.Frames` process_folder method

        Parameters
        ----------
        folder : str
            Dummy media folder
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking superclass calls
        """
        process_video_mock = mocker.patch("tools.alignments.media.Frames.process_video")
        process_frames_mock = mocker.patch("tools.alignments.media.Frames.process_frames")

        frames = Frames(folder, None)
        frames.process_folder()
        process_frames_mock.assert_called_once()
        process_video_mock.assert_not_called()

        process_frames_mock.reset_mock()
        mocker.patch("tools.alignments.media.Frames.is_video",
                     new_callable=mocker.PropertyMock(return_value=True))
        frames = Frames(folder, None)
        frames.process_folder()
        process_frames_mock.assert_not_called()
        process_video_mock.assert_called_once()

    def test_process_frames(self, folder: str) -> None:
        """ Test for :class:`~tools.alignments.media.Frames` process_frames method

        Parameters
        ----------
        folder : str
            Dummy media folder
        """
        expected = [{"frame_fullname": "a.png", "frame_name": "a", "frame_extension": ".png"},
                    {"frame_fullname": "b.png", "frame_name": "b", "frame_extension": ".png"}]

        frames = Frames(folder, None)
        returned = sorted(list(frames.process_frames()), key=itemgetter("frame_fullname"))
        assert returned == sorted(expected, key=itemgetter("frame_fullname"))

    def test_process_video(self, folder: str) -> None:
        """ Test for :class:`~tools.alignments.media.Frames` process_video method

        Parameters
        ----------
        folder : str
            Dummy media folder
        """
        ext = os.path.splitext(folder)[-1]
        expected = [{"frame_fullname": f"images_000001{ext}",
                     "frame_name": "images_000001",
                     "frame_extension": ext},
                    {"frame_fullname": f"images_000002{ext}",
                     "frame_name": "images_000002",
                     "frame_extension": ext}]

        frames = Frames(folder, None)
        returned = list(frames.process_video())
        assert returned == expected

    def test_load_items(self, folder: str) -> None:
        """ Test for :class:`~tools.alignments.media.Frames` load_items method

        Parameters
        ----------
        folder : str
            Dummy media folder
        """
        expected = {"a.png": ("a", ".png"), "b.png": ("b", ".png")}
        frames = Frames(folder, None)
        result = frames.load_items()
        assert result == expected

    def test_sorted_items(self,
                          folder: str,
                          mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.Frames` sorted_items method

        Parameters
        ----------
        folder : str
            Dummy media folder
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking process_folder call
        """
        frames = Frames(folder, None)
        data = [{"frame_fullname": "c.png", "frame_name": "c", "frame_extension": ".png"},
                {"frame_fullname": "d.png", "frame_name": "d", "frame_extension": ".png"},
                {"frame_fullname": "b.jpg", "frame_name": "b", "frame_extension": ".jpg"},
                {"frame_fullname": "a.png", "frame_name": "a", "frame_extension": ".png"}]
        expected = [{"frame_fullname": "a.png", "frame_name": "a", "frame_extension": ".png"},
                    {"frame_fullname": "b.jpg", "frame_name": "b", "frame_extension": ".jpg"},
                    {"frame_fullname": "c.png", "frame_name": "c", "frame_extension": ".png"},
                    {"frame_fullname": "d.png", "frame_name": "d", "frame_extension": ".png"}]
        process_folder_mock = mocker.patch("tools.alignments.media.Frames.process_folder",
                                           side_effect=[data])
        result = frames.sorted_items()

        assert process_folder_mock.called
        assert result == expected


class TestExtractedFaces:
    """ Test for :class:`~tools.alignments.media.ExtractedFaces` """

    @pytest.fixture(name="extracted_faces_instance")
    def extracted_faces_fixture(self, mocker: pytest_mock.MockerFixture) -> ExtractedFaces:
        """ An instance of :class:`~tools.alignments.media.ExtractedFaces` patching out Frames and
        AlignmentData parameters

        Parameters
        ----------
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking read_image_meta_batch calls

        Returns
        -------
        :class:`~tools.alignments.media.ExtractedFaces`
            Initialized instance for testing
        """
        frames_mock = mocker.MagicMock(Frames)
        alignments_mock = mocker.MagicMock(AlignmentData)
        return ExtractedFaces(frames_mock, alignments_mock, size=512)

    def test_init(self, extracted_faces_instance: ExtractedFaces) -> None:
        """ Test for :class:`~tools.alignments.media.ExtractedFace` __init__ method

        Parameters
        ----------
        extracted_faces_instance: :class:`~tools.alignments.media.ExtractedFace`
            The class instance for testing
        """
        faces = extracted_faces_instance
        assert faces.size == 512
        assert faces.padding == int(512 * 0.1875)
        assert faces.current_frame is None
        assert faces.faces == []

    def test_get_faces(self,
                       extracted_faces_instance: ExtractedFaces,
                       mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.ExtractedFace` get_faces method

        Parameters
        ----------
        extracted_faces_instance: :class:`~tools.alignments.media.ExtractedFace`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking Frames and AlignmentData classes
        """
        extract_face_mock = mocker.patch("tools.alignments.media.ExtractedFaces.extract_one_face")
        faces = extracted_faces_instance

        frame = "test_frame"
        img = np.random.rand(256, 256, 3)

        # No alignment data
        faces.alignments.get_faces_in_frame.return_value = []  # type:ignore
        faces.get_faces(frame, img)
        faces.alignments.get_faces_in_frame.assert_called_once_with(frame)  # type:ignore
        faces.frames.load_image.assert_not_called()  # type:ignore
        extract_face_mock.assert_not_called()
        assert faces.current_frame is None

        faces.alignments.reset_mock()  # type:ignore

        # Alignment data + image
        faces.alignments.get_faces_in_frame.return_value = [1, 2, 3]  # type:ignore
        faces.get_faces(frame, img)
        faces.alignments.get_faces_in_frame.assert_called_once_with(frame)  # type:ignore
        faces.frames.load_image.assert_not_called()  # type:ignore
        assert extract_face_mock.call_count == 3
        assert faces.current_frame == frame

        faces.alignments.reset_mock()  # type:ignore
        extract_face_mock.reset_mock()

        # Alignment data + no image
        faces.alignments.get_faces_in_frame.return_value = ["data1"]  # type:ignore
        faces.get_faces(frame, None)
        faces.alignments.get_faces_in_frame.assert_called_once_with(frame)  # type:ignore
        faces.frames.load_image.assert_called_once_with(frame)  # type:ignore
        assert extract_face_mock.call_count == 1
        assert faces.current_frame == frame

    def test_extract_one_face(self,
                              extracted_faces_instance: ExtractedFaces,
                              mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.ExtractedFace` extract_one_face method

        Parameters
        ----------
        extracted_faces_instance: :class:`~tools.alignments.media.ExtractedFace`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking DetectedFace object
        """
        detected_face = mocker.patch("tools.alignments.media.DetectedFace")
        thumbnail_mock = mocker.patch("tools.alignments.media.generate_thumbnail")
        faces = extracted_faces_instance
        alignment = {"test"}
        img = np.random.rand(256, 256, 3)
        returned = faces.extract_one_face(alignment, img)  # type:ignore
        detected_face.assert_called_once()
        detected_face.return_value.from_alignment.assert_called_once_with(alignment,
                                                                          image=img)
        detected_face.return_value.load_aligned.assert_called_once_with(img,
                                                                        size=512,
                                                                        centering="head")
        thumbnail_mock.assert_called_once()
        assert isinstance(returned, MagicMock)

    def test_get_faces_in_frame(self,
                                extracted_faces_instance: ExtractedFaces,
                                mocker: pytest_mock.MockerFixture) -> None:
        """ Test for :class:`~tools.alignments.media.ExtractedFace` get_faces_in_frame method

        Parameters
        ----------
        extracted_faces_instance: :class:`~tools.alignments.media.ExtractedFace`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking get_faces method
        """
        faces = extracted_faces_instance
        faces.get_faces = T.cast(MagicMock, mocker.MagicMock())  # type:ignore

        frame = "test_frame"
        img = None

        faces.get_faces_in_frame(frame, update=False, image=img)
        faces.get_faces.assert_called_once_with(frame, image=img)

        faces.get_faces.reset_mock()

        faces.current_frame = frame
        faces.get_faces_in_frame(frame, update=False, image=img)
        faces.get_faces.assert_not_called()

        faces.get_faces_in_frame(frame, update=True, image=img)
        faces.get_faces.assert_called_once_with(frame, image=img)

    _params = [(np.array(([[25, 47], [32, 232], [244, 237], [240, 21]])), 216),
               (np.array(([[127, 392], [403, 510], [32, 237], [19, 210]])), 211),
               (np.array(([[26, 1927], [112, 1234], [1683, 1433], [78, 1155]])), 773)]

    @pytest.mark.parametrize("roi,expected", _params)
    def test_get_roi_size_for_frame(self,
                                    extracted_faces_instance: ExtractedFaces,
                                    mocker: pytest_mock.MockerFixture,
                                    roi: np.ndarray,
                                    expected: int) -> None:
        """ Test for :class:`~tools.alignments.media.ExtractedFace` get_roi_size_for_frame method

        Parameters
        ----------
        extracted_faces_instance: :class:`~tools.alignments.media.ExtractedFace`
            The class instance for testing
        mocker: :class:`pytest_mock.MockerFixture`
            Fixture for mocking get_faces method and DetectedFace object
        roi: :class:`numpy.ndarray`
            Test ROI box to feed into the function
        expected: int
            The expected output for the given ROI box
        """
        faces = extracted_faces_instance
        faces.get_faces = T.cast(MagicMock, mocker.MagicMock())  # type:ignore

        frame = "test_frame"
        faces.get_roi_size_for_frame(frame)
        faces.get_faces.assert_called_once_with(frame)

        faces.get_faces.reset_mock()

        faces.current_frame = frame
        faces.get_roi_size_for_frame(frame)
        faces.get_faces.assert_not_called()

        detected_face = mocker.MagicMock("tools.alignments.media.DetectedFace")
        detected_face.aligned = detected_face
        detected_face.original_roi = roi
        faces.faces = [detected_face]
        result = faces.get_roi_size_for_frame(frame)
        assert result == [expected]
