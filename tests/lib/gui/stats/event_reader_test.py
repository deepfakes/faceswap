#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.gui.stats.event_reader` """
# pylint:disable=protected-access
from __future__ import annotations
import json
import os
import typing as T

from shutil import rmtree
from time import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_mock

import tensorflow as tf
from tensorflow.core.util import event_pb2  # pylint:disable=no-name-in-module

from lib.gui.analysis.event_reader import (_Cache, _CacheData, _EventParser,
                                           _LogFiles, EventData, TensorBoardLogs)

if T.TYPE_CHECKING:
    from collections.abc import Iterator


def test__logfiles(tmp_path: str):
    """ Test the _LogFiles class operates correctly

    Parameters
    ----------
    tmp_path: :class:`pathlib.Path`
    """
    # dummy logfiles + junk data
    sess_1 = os.path.join(tmp_path, "session_1", "train")
    sess_2 = os.path.join(tmp_path, "session_2", "train")
    os.makedirs(sess_1)
    os.makedirs(sess_2)

    test_log_1 = os.path.join(sess_1, "events.out.tfevents.123.456.v2")
    test_log_2 = os.path.join(sess_2, "events.out.tfevents.789.012.v2")
    test_log_junk = os.path.join(sess_2, "test_file.txt")

    for fname in (test_log_1, test_log_2, test_log_junk):
        with open(fname, "a", encoding="utf-8"):
            pass

    log_files = _LogFiles(tmp_path)
    # Test all correct
    assert isinstance(log_files._filenames, dict)
    assert len(log_files._filenames) == 2
    assert log_files._filenames == {1: test_log_1, 2: test_log_2}

    assert log_files.session_ids == [1, 2]

    assert log_files.get(1) == test_log_1
    assert log_files.get(2) == test_log_2

    # Remove a file, refresh and check again
    rmtree(sess_1)
    log_files.refresh()
    assert log_files._filenames == {2: test_log_2}
    assert log_files.get(2) == test_log_2
    assert log_files.get(3) == ""


def test__cachedata():
    """ Test the _CacheData class operates correctly """
    labels = ["label_a", "label_b"]
    timestamps = np.array([1.23, 4.56], dtype="float64")
    loss = np.array([[2.34, 5.67], [3.45, 6.78]], dtype="float32")

    # Initial test
    cache = _CacheData(labels, timestamps, loss)
    assert cache.labels == labels
    assert cache._timestamps_shape == timestamps.shape
    assert cache._loss_shape == loss.shape
    np.testing.assert_array_equal(cache.timestamps, timestamps)
    np.testing.assert_array_equal(cache.loss, loss)

    # Add data test
    new_timestamps = np.array([2.34, 6.78], dtype="float64")
    new_loss = np.array([[3.45, 7.89], [8.90, 1.23]], dtype="float32")

    expected_timestamps = np.concatenate([timestamps, new_timestamps])
    expected_loss = np.concatenate([loss, new_loss])

    cache.add_live_data(new_timestamps, new_loss)
    assert cache.labels == labels
    assert cache._timestamps_shape == expected_timestamps.shape
    assert cache._loss_shape == expected_loss.shape
    np.testing.assert_array_equal(cache.timestamps, expected_timestamps)
    np.testing.assert_array_equal(cache.loss, expected_loss)


# _Cache tests
class Test_Cache:  # pylint:disable=invalid-name
    """ Test that :class:`lib.gui.analysis.event_reader._Cache` works correctly """
    @staticmethod
    def test_init() -> None:
        """ Test __init__ """
        cache = _Cache()
        assert isinstance(cache._data, dict)
        assert isinstance(cache._carry_over, dict)
        assert isinstance(cache._loss_labels, list)
        assert not cache._data
        assert not cache._carry_over
        assert not cache._loss_labels

    @staticmethod
    def test_is_cached() -> None:
        """ Test is_cached function works """
        cache = _Cache()

        data = _CacheData(["test_1", "test_2"],
                          np.array([1.23, ], dtype="float64"),
                          np.array([[2.34, ], [4.56]], dtype="float32"))
        cache._data[1] = data
        assert cache.is_cached(1)
        assert not cache.is_cached(2)

    @staticmethod
    def test_cache_data(mocker: pytest_mock.MockerFixture) -> None:
        """ Test cache_data function works

        Parameters
        ----------
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking full_info called from _SysInfo
        """
        cache = _Cache()

        session_id = 1
        data = {1: EventData(4., [1., 2.]), 2: EventData(5., [3., 4.])}
        labels = ['label1', 'label2']
        is_live = False

        cache.cache_data(session_id, data, labels, is_live)
        assert cache._loss_labels == labels
        assert cache.is_cached(session_id)
        np.testing.assert_array_equal(cache._data[session_id].timestamps, np.array([4., 5.]))
        np.testing.assert_array_equal(cache._data[session_id].loss, np.array([[1., 2.], [3., 4.]]))

        add_live = mocker.patch("lib.gui.analysis.event_reader._Cache._add_latest_live")
        is_live = True
        cache.cache_data(session_id, data, labels, is_live)
        assert add_live.called

    @staticmethod
    def test__to_numpy() -> None:
        """ Test _to_numpy function works """
        cache = _Cache()
        cache._loss_labels = ['label1', 'label2']
        data = {1: EventData(4., [1., 2.]), 2: EventData(5., [3., 4.])}

        # Non-live
        is_live = False
        times, loss = cache._to_numpy(data, is_live)
        np.testing.assert_array_equal(times, np.array([4., 5.]))
        np.testing.assert_array_equal(loss, np.array([[1., 2.], [3., 4.]]))

        # Correctly collected live
        is_live = True
        times, loss = cache._to_numpy(data, is_live)
        np.testing.assert_array_equal(times, np.array([4., 5.]))
        np.testing.assert_array_equal(loss, np.array([[1., 2.], [3., 4.]]))

        # Incorrectly collected live
        live_data = {1: EventData(4., [1., 2.]),
                     2: EventData(5., [3.]),
                     3: EventData(6., [4., 5., 6.])}
        times, loss = cache._to_numpy(live_data, is_live)
        np.testing.assert_array_equal(times, np.array([4.]))
        np.testing.assert_array_equal(loss, np.array([[1., 2.]]))

    @staticmethod
    def test__collect_carry_over() -> None:
        """ Test _collect_carry_over function works """
        data = {1: EventData(3., [4., 5.]), 2: EventData(6., [7., 8.])}
        carry_over = {1: EventData(3., [2., 3.])}
        expected = {1: EventData(3., [2., 3., 4., 5.]), 2: EventData(6., [7., 8.])}

        cache = _Cache()
        cache._carry_over = carry_over
        cache._collect_carry_over(data)
        assert data == expected

    @staticmethod
    def test__process_data() -> None:
        """ Test _process_data function works """
        cache = _Cache()
        cache._loss_labels = ['label1', 'label2']

        data = {1: EventData(4., [5., 6.]),
                2: EventData(5., [7., 8.]),
                3: EventData(6., [9.])}
        is_live = False
        expected_timestamps = np.array([4., 5.])
        expected_loss = np.array([[5., 6.], [7., 8.]])
        expected_carry_over = {3: EventData(6., [9.])}

        timestamps, loss = cache._process_data(data, is_live)
        np.testing.assert_array_equal(timestamps, expected_timestamps)
        np.testing.assert_array_equal(loss, expected_loss)
        assert not cache._carry_over

        is_live = True
        timestamps, loss = cache._process_data(data, is_live)
        np.testing.assert_array_equal(timestamps, expected_timestamps)
        np.testing.assert_array_equal(loss, expected_loss)
        assert cache._carry_over == expected_carry_over

    @staticmethod
    def test__add_latest_live() -> None:
        """ Test _add_latest_live function works """
        session_id = 1
        labels = ['label1', 'label2']
        data = {1: EventData(3., [5., 6.]), 2: EventData(4., [7., 8.])}
        new_timestamp = np.array([5.], dtype="float64")
        new_loss = np.array([[8., 9.]], dtype="float32")
        expected_timestamps = np.array([3., 4., 5.])
        expected_loss = np.array([[5., 6.], [7., 8.], [8., 9.]])

        cache = _Cache()
        cache.cache_data(session_id, data, labels)  # Initial data
        cache._add_latest_live(session_id, new_loss, new_timestamp)

        assert cache.is_cached(session_id)
        assert cache._loss_labels == labels
        np.testing.assert_array_equal(cache._data[session_id].timestamps, expected_timestamps)
        np.testing.assert_array_equal(cache._data[session_id].loss, expected_loss)

    @staticmethod
    def test_get_data() -> None:
        """ Test get_data function works """
        session_id = 1

        cache = _Cache()
        assert cache.get_data(session_id, "loss") is None
        assert cache.get_data(session_id, "timestamps") is None

        labels = ['label1', 'label2']
        data = {1: EventData(3., [5., 6.]), 2: EventData(4., [7., 8.])}
        expected_timestamps = np.array([3., 4.])
        expected_loss = np.array([[5., 6.], [7., 8.]])

        cache.cache_data(session_id, data, labels, is_live=False)
        get_timestamps = cache.get_data(session_id, "timestamps")
        get_loss = cache.get_data(session_id, "loss")

        assert isinstance(get_timestamps, dict)
        assert len(get_timestamps) == 1
        assert list(get_timestamps) == [session_id]
        result = get_timestamps[session_id]
        assert list(result) == ["timestamps"]
        np.testing.assert_array_equal(result["timestamps"], expected_timestamps)

        assert isinstance(get_loss, dict)
        assert len(get_loss) == 1
        assert list(get_loss) == [session_id]
        result = get_loss[session_id]
        assert list(result) == ["loss", "labels"]
        np.testing.assert_array_equal(result["loss"], expected_loss)


# TensorBoardLogs
class TestTensorBoardLogs:
    """ Test that :class:`lib.gui.analysis.event_reader.TensorBoardLogs` works correctly """

    @pytest.fixture(name="tensorboardlogs_instance")
    def tensorboardlogs_fixture(self,
                                tmp_path: str,
                                request: pytest.FixtureRequest) -> TensorBoardLogs:
        """ Pytest fixture for :class:`lib.gui.analysis.event_reader.TensorBoardLogs`

        Parameters
        ----------
        tmp_path: :class:`pathlib.Path`
            Temporary folder for dummy data

        Returns
        -------
        :class::class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance for testing
        """
        sess_1 = os.path.join(tmp_path, "session_1", "train")
        sess_2 = os.path.join(tmp_path, "session_2", "train")
        os.makedirs(sess_1)
        os.makedirs(sess_2)

        test_log_1 = os.path.join(sess_1, "events.out.tfevents.123.456.v2")
        test_log_2 = os.path.join(sess_2, "events.out.tfevents.789.012.v2")

        for fname in (test_log_1, test_log_2):
            with open(fname, "a", encoding="utf-8"):
                pass

        tblogs_instance = TensorBoardLogs(tmp_path, False)

        def teardown():
            rmtree(tmp_path)

        request.addfinalizer(teardown)
        return tblogs_instance

    @staticmethod
    def test_init(tensorboardlogs_instance: TensorBoardLogs) -> None:
        """ Test __init__ works correctly

        Parameters
        ----------
        tensorboadlogs_instance: :class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance to test
        """
        tb_logs = tensorboardlogs_instance
        assert isinstance(tb_logs._log_files, _LogFiles)
        assert isinstance(tb_logs._cache, _Cache)
        assert not tb_logs._is_training

        is_training = True
        folder = tb_logs._log_files._logs_folder
        tb_logs = TensorBoardLogs(folder, is_training)
        assert tb_logs._is_training

    @staticmethod
    def test_session_ids(tensorboardlogs_instance: TensorBoardLogs) -> None:
        """ Test session_ids property works correctly

        Parameters
        ----------
        tensorboadlogs_instance: :class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance to test
        """
        tb_logs = tensorboardlogs_instance
        assert tb_logs.session_ids == [1, 2]

    @staticmethod
    def test_set_training(tensorboardlogs_instance: TensorBoardLogs) -> None:
        """ Test set_training works correctly

        Parameters
        ----------
        tensorboadlogs_instance: :class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance to test
        """
        tb_logs = tensorboardlogs_instance
        assert not tb_logs._is_training
        assert tb_logs._training_iterator is None
        tb_logs.set_training(True)
        assert tb_logs._is_training
        assert tb_logs._training_iterator is not None
        tb_logs.set_training(False)
        assert not tb_logs._is_training
        assert tb_logs._training_iterator is None

    @staticmethod
    def test__cache_data(tensorboardlogs_instance: TensorBoardLogs,
                         mocker: pytest_mock.MockerFixture) -> None:
        """ Test _cache_data works correctly

        Parameters
        ----------
        tensorboadlogs_instance: :class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance to test
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking event parser caching is called
        """
        tb_logs = tensorboardlogs_instance
        session_id = 1
        cacher = mocker.patch("lib.gui.analysis.event_reader._EventParser.cache_events")
        tb_logs._cache_data(session_id)
        assert cacher.called
        cacher.reset_mock()

        tb_logs.set_training(True)
        tb_logs._cache_data(session_id)
        assert cacher.called

    @staticmethod
    def test__check_cache(tensorboardlogs_instance: TensorBoardLogs,
                          mocker: pytest_mock.MockerFixture) -> None:
        """ Test _check_cache works correctly

        Parameters
        ----------
        tensorboadlogs_instance: :class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance to test
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking _cache_data is called
        """
        is_cached = mocker.patch("lib.gui.analysis.event_reader._Cache.is_cached")
        cache_data = mocker.patch("lib.gui.analysis.event_reader.TensorBoardLogs._cache_data")
        tb_logs = tensorboardlogs_instance

        # Session ID not training
        is_cached.return_value = False
        tb_logs._check_cache(1)
        assert is_cached.called
        assert cache_data.called
        is_cached.reset_mock()
        cache_data.reset_mock()

        is_cached.return_value = True
        tb_logs._check_cache(1)
        assert is_cached.called
        assert not cache_data.called
        is_cached.reset_mock()
        cache_data.reset_mock()

        # Session ID and training
        tb_logs.set_training(True)
        tb_logs._check_cache(1)
        assert not cache_data.called
        cache_data.reset_mock()

        tb_logs._check_cache(2)
        assert cache_data.called
        cache_data.reset_mock()

        # No session id
        tb_logs.set_training(False)
        is_cached.return_value = False

        tb_logs._check_cache(None)
        assert is_cached.called
        assert cache_data.called
        is_cached.reset_mock()
        cache_data.reset_mock()

        is_cached.return_value = True
        tb_logs._check_cache(None)
        assert is_cached.called
        assert not cache_data.called
        is_cached.reset_mock()
        cache_data.reset_mock()

    @staticmethod
    def test_get_loss(tensorboardlogs_instance: TensorBoardLogs,
                      mocker: pytest_mock.MockerFixture) -> None:
        """ Test get_loss works correctly

        Parameters
        ----------
        tensorboadlogs_instance: :class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance to test
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking _cache_data is called
        """
        tb_logs = tensorboardlogs_instance

        with pytest.raises(tf.errors.NotFoundError):  # Invalid session id
            tb_logs.get_loss(3)

        check_cache = mocker.patch("lib.gui.analysis.event_reader.TensorBoardLogs._check_cache")
        get_data = mocker.patch("lib.gui.analysis.event_reader._Cache.get_data")
        get_data.return_value = None

        assert isinstance(tb_logs.get_loss(None), dict)
        assert check_cache.call_count == 2
        assert get_data.call_count == 2
        check_cache.reset_mock()
        get_data.reset_mock()

        assert isinstance(tb_logs.get_loss(1), dict)
        assert check_cache.call_count == 1
        assert get_data.call_count == 1
        check_cache.reset_mock()
        get_data.reset_mock()

    @staticmethod
    def test_get_timestamps(tensorboardlogs_instance: TensorBoardLogs,
                            mocker: pytest_mock.MockerFixture) -> None:
        """ Test get_timestamps works correctly

        Parameters
        ----------
        tensorboadlogs_instance: :class:`lib.gui.analysis.event_reader.TensorBoardLogs`
            The class instance to test
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for checking _cache_data is called
        """
        tb_logs = tensorboardlogs_instance
        with pytest.raises(tf.errors.NotFoundError):  # invalid session_id
            tb_logs.get_timestamps(3)

        check_cache = mocker.patch("lib.gui.analysis.event_reader.TensorBoardLogs._check_cache")
        get_data = mocker.patch("lib.gui.analysis.event_reader._Cache.get_data")
        get_data.return_value = None

        assert isinstance(tb_logs.get_timestamps(None), dict)
        assert check_cache.call_count == 2
        assert get_data.call_count == 2
        check_cache.reset_mock()
        get_data.reset_mock()

        assert isinstance(tb_logs.get_timestamps(1), dict)
        assert check_cache.call_count == 1
        assert get_data.call_count == 1
        check_cache.reset_mock()
        get_data.reset_mock()


# EventParser
class Test_EventParser:  # pylint:disable=invalid-name
    """ Test that :class:`lib.gui.analysis.event_reader.TensorBoardLogs` works correctly """
    def _create_example_event(self,
                              step: int,
                              loss_value: float,
                              timestamp: float,
                              serialize: bool = True) -> bytes:
        """ Generate a test TensorBoard event

        Parameters
        ----------
        step: int
            The step value to use
        loss_value: float
            The loss value to store
        timestamp: float
            The timestamp to store
        serialize: bool, optional
            ``True`` to serialize the event to bytes, ``False`` to return the Event object
        """
        tags = {0: "keras", 1: "batch_total", 2: "batch_face_a", 3: "batch_face_b"}
        event = event_pb2.Event(step=step)
        event.summary.value.add(tag=tags[step],  # pylint:disable=no-member
                                simple_value=loss_value)
        event.wall_time = timestamp
        retval = event.SerializeToString() if serialize else event
        return retval

    @pytest.fixture(name="mock_iterator")
    def iterator(self) -> Iterator[bytes]:
        """ Dummy iterator for generating test events

        Yields
        ------
        bytes
            A serialized test Tensorboard Event
        """
        return iter([self._create_example_event(i, 1 + (i / 10), time()) for i in range(4)])

    @pytest.fixture(name="mock_cache")
    def mock_cache(self):
        """ Dummy :class:`_Cache` for testing"""
        class _CacheMock:
            def __init__(self):
                self.data = {}
                self._loss_labels = []

            def is_cached(self, session_id):
                """ Dummy is_cached method"""
                return session_id in self.data

            def cache_data(self, session_id, data, labels,
                           is_live=False):  # pylint:disable=unused-argument
                """ Dummy cache_data method"""
                self.data[session_id] = {'data': data, 'labels': labels}

        return _CacheMock()

    @pytest.fixture(name="event_parser_instance")
    def event_parser_fixture(self,
                             mock_iterator: Iterator[bytes],
                             mock_cache: _Cache) -> _EventParser:
        """ Pytest fixture for :class:`lib.gui.analysis.event_reader._EventParser`

        Parameters
        ----------
        mock_iterator: Iterator[bytes]
            Dummy iterator for generating TF Event data
        mock_cache: :class:'_CacheMock'
            Dummy _Cache object

        Returns
        -------
        :class::class:`lib.gui.analysis.event_reader._EventParser`
            The class instance for testing
        """
        event_parser = _EventParser(mock_iterator, mock_cache, live_data=False)
        return event_parser

    def test__init_(self,
                    event_parser_instance: _EventParser,
                    mock_iterator: Iterator[bytes],
                    mock_cache: _Cache) -> None:
        """ Test __init__ works correctly

        Parameters
        ----------
        event_parser_instance: :class:`lib.gui.analysis.event_reader._EventParser`
            The class instance to test
        mock_iterator: Iterator[bytes]
            Dummy iterator for generating TF Event data
        mock_cache: :class:'_CacheMock'
            Dummy _Cache object
        """
        event_parse = event_parser_instance
        assert not hasattr(event_parse._iterator, "__name__")
        evp_live = _EventParser(mock_iterator, mock_cache, live_data=True)
        assert evp_live._iterator.__name__ == "_get_latest_live"  # type:ignore[attr-defined]

    def test__get_latest_live(self, event_parser_instance: _EventParser) -> None:
        """ Test _get_latest_live works correctly

        Parameters
        ----------
        event_parser_instance: :class:`lib.gui.analysis.event_reader._EventParser`
            The class instance to test
        """
        event_parse = event_parser_instance
        test = list(event_parse._get_latest_live(event_parse._iterator))
        assert len(test) == 4

    def test_cache_events(self,
                          event_parser_instance: _EventParser,
                          mocker: pytest_mock.MockerFixture,
                          monkeypatch: pytest.MonkeyPatch) -> None:
        """ Test cache_events works correctly

        Parameters
        ----------
        event_parser_instance: :class:`lib.gui.analysis.event_reader._EventParser`
            The class instance to test
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for capturing method calls
        monkeypatch: :class:`pytest.MonkeyPatch`
            For patching different iterators for testing output
        """
        monkeypatch.setattr("lib.utils._FS_BACKEND", "cpu")

        event_parse = event_parser_instance
        event_parse._parse_outputs = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        event_parse._process_event = T.cast(MagicMock, mocker.MagicMock())  # type:ignore
        event_parse._cache.cache_data = T.cast(MagicMock, mocker.MagicMock())  # type:ignore

        # keras model
        monkeypatch.setattr(event_parse,
                            "_iterator",
                            iter([self._create_example_event(0, 1., time())]))
        event_parse.cache_events(1)
        assert event_parse._parse_outputs.called
        assert not event_parse._process_event.called
        assert event_parse._cache.cache_data.called
        event_parse._parse_outputs.reset_mock()
        event_parse._process_event.reset_mock()
        event_parse._cache.cache_data.reset_mock()

        # Batch item
        monkeypatch.setattr(event_parse,
                            "_iterator",
                            iter([self._create_example_event(1, 1., time())]))
        event_parse.cache_events(1)
        assert not event_parse._parse_outputs.called
        assert event_parse._process_event.called
        assert event_parse._cache.cache_data.called
        event_parse._parse_outputs.reset_mock()
        event_parse._process_event.reset_mock()
        event_parse._cache.cache_data.reset_mock()

        # No summary value
        monkeypatch.setattr(event_parse,
                            "_iterator",
                            iter([event_pb2.Event(step=1).SerializeToString()]))
        assert not event_parse._parse_outputs.called
        assert not event_parse._process_event.called
        assert not event_parse._cache.cache_data.called
        event_parse._parse_outputs.reset_mock()
        event_parse._process_event.reset_mock()
        event_parse._cache.cache_data.reset_mock()

    def test__parse_outputs(self,
                            event_parser_instance: _EventParser,
                            mocker: pytest_mock.MockerFixture) -> None:
        """ Test _parse_outputs works correctly

        Parameters
        ----------
        event_parser_instance: :class:`lib.gui.analysis.event_reader._EventParser`
            The class instance to test
        mocker: :class:`pytest_mock.MockerFixture`
            Mocker for event object
        """
        event_parse = event_parser_instance
        model = {"config": {"layers": [{"name": "decoder_a",
                                        "config": {"output_layers": [["face_out_a", 0, 0]]}},
                                       {"name": "decoder_b",
                                        "config": {"output_layers": [["face_out_b", 0, 0]]}}],
                            "output_layers": [["decoder_a", 1, 0], ["decoder_b", 1, 0]]}}
        data = json.dumps(model).encode("utf-8")

        event = mocker.MagicMock()
        event.summary.value.__getitem__ = lambda self, x: event
        event.tensor.string_val.__getitem__ = lambda self, x: data

        assert not event_parse._loss_labels
        event_parse._parse_outputs(event)
        assert event_parse._loss_labels == ["face_out_a", "face_out_b"]

    def test__get_outputs(self, event_parser_instance: _EventParser) -> None:
        """ Test _get_outputs works correctly

        Parameters
        ----------
        event_parser_instance: :class:`lib.gui.analysis.event_reader._EventParser`
            The class instance to test
        """
        outputs = [["decoder_a", 1, 0], ["decoder_b", 1, 0]]
        model_config = {"output_layers": outputs}

        expected = np.array([[out] for out in outputs])
        actual = event_parser_instance._get_outputs(model_config)
        assert isinstance(actual, np.ndarray)
        assert actual.shape == (2, 1, 3)
        np.testing.assert_equal(expected, actual)

    def test__process_event(self, event_parser_instance: _EventParser) -> None:
        """ Test _process_event works correctly

        Parameters
        ----------
        event_parser_instance: :class:`lib.gui.analysis.event_reader._EventParser`
            The class instance to test
        """
        event_parse = event_parser_instance
        event_data = EventData()
        assert not event_data.timestamp
        assert not event_data.loss
        timestamp = time()
        loss = [1.1, 2.2]
        event = self._create_example_event(1, 1.0, timestamp, serialize=False)  # batch_total
        event_parse._process_event(event, event_data)
        event = self._create_example_event(2, loss[0], time(), serialize=False)  # face A
        event_parse._process_event(event, event_data)
        event = self._create_example_event(3, loss[1], time(), serialize=False)  # face B
        event_parse._process_event(event, event_data)

        # Original timestamp and both loss values collected
        assert event_data.timestamp == timestamp
        np.testing.assert_almost_equal(event_data.loss, loss)  # float rounding
