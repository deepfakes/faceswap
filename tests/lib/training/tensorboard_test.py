#! /usr/env/bin/python3
""" Unit test for :mod:`lib.training.tensorboard` """
import os

import pytest

from keras import layers, Sequential
import numpy as np
from tensorboard.compat.proto import event_pb2
from torch.utils.tensorboard import SummaryWriter

from lib.training import tensorboard as mod_tb

# pylint:disable=protected-access,invalid-name


@pytest.fixture()
def _gen_events_file(tmpdir):
    log_dir = tmpdir.mkdir("logs")

    def _apply(keys=["test1"],  # pylint:disable=dangerous-default-value
               values=[0.42],
               global_steps=[4]):
        writer = SummaryWriter(log_dir)
        for key, val, step in zip(keys, values, global_steps):
            writer.add_scalar(key, val, global_step=step)
        writer.flush()
        return os.path.join(log_dir, os.listdir(log_dir)[0])

    return _apply


@pytest.mark.parametrize("entries", ({"loss1": np.random.rand()},
                                     {f"test{i}": np.random.rand() for i in range(4)},
                                     {f"another_test{i}": np.random.rand() for i in range(10)}))
@pytest.mark.parametrize("batch", [1, 42, 69, 1024, 143432])
@pytest.mark.parametrize("is_live", (True, False), ids=("live", "not_live"))
def test_RecordIterator(entries, batch, is_live, _gen_events_file):
    """ Test that our :class:`lib.training.tensorboard.RecordIterator` returns expected results """
    keys = list(entries)
    vals = list(entries.values())
    batches = [batch + i for i in range(len(keys))]

    file = _gen_events_file(keys, vals, batches)
    iterator = mod_tb.RecordIterator(file, is_live=is_live)

    results = list(event_pb2.Event.FromString(v) for v in iterator)
    valid = [r for r in results if r.summary.value]

    assert len(valid) == len(keys)
    for entry, key, val, btc in zip(valid, keys, vals, batches):
        assert len(entry.summary.value) == 1
        assert entry.step == btc
        assert entry.summary.value[0].tag == key
        assert np.isclose(entry.summary.value[0].simple_value, val)

    if is_live:
        assert iterator._is_live is True
        assert os.path.getsize(file) == iterator._position  # At end of file
    else:
        assert iterator._is_live is False
        assert iterator._position == 0


@pytest.fixture()
def _get_ttb_instance(tmpdir):
    log_dir = tmpdir.mkdir("logs")

    def _apply(write_graph=False, update_freq="batch"):
        instance = mod_tb.TorchTensorBoard(log_dir=log_dir,
                                           write_graph=write_graph,
                                           update_freq=update_freq)
        return log_dir, instance

    return _apply


def _get_logs(temp_path):
    train_logs = os.path.join(temp_path, "train")
    log_files = os.listdir(train_logs)
    assert len(log_files) == 1
    records = [event_pb2.Event.FromString(record)
               for record in mod_tb.RecordIterator(os.path.join(train_logs, log_files[0]))]
    return records


@pytest.mark.parametrize("write_graph", (True, False), ids=("write_graph", "no_write_graph"))
def test_TorchTensorBoard_set_model(write_graph, _get_ttb_instance):
    """ Test that :class:`lib.training.tensorboard.set_model` functions """
    log_dir, instance = _get_ttb_instance(write_graph=write_graph)

    model = Sequential()
    model.add(layers.Input(shape=(8, )))
    model.add(layers.Dense(4))
    model.add(layers.Dense(4))

    assert not os.path.exists(os.path.join(log_dir, "train"))
    instance.set_model(model)
    instance.on_save()

    logs = [x for x in _get_logs(os.path.join(log_dir))
            if x.summary.value]

    if not write_graph:
        assert not logs
        return

    # Only a single logged entry
    assert len(logs) == 1 and len(logs[0].summary.value) == 1
    # Should be our Keras model summary
    assert logs[0].summary.value[0].tag == "keras/text_summary"


def test_TorchTensorBoard_on_train_begin(_get_ttb_instance):
    """ Test that :class:`lib.training.tensorboard.on_train_begin` functions """
    _, instance = _get_ttb_instance()
    instance.on_train_begin()
    assert instance._global_train_batch == 0
    assert instance._previous_epoch_iterations == 0


@pytest.mark.parametrize("batch", (1, 3, 57, 124))
@pytest.mark.parametrize("logs", ({"loss_a": 2.45, "loss_b": 1.56},
                                  {"loss_c": 0.54, "loss_d": 0.51},
                                  {"loss_c": 0.69, "loss_d": 0.42, "loss_g": 2.69}))
def test_TorchTensorBoard_on_train_batch_end(batch, logs, _get_ttb_instance):
    """ Test that :class:`lib.training.tensorboard.on_train_batch_end` functions """
    log_dir, instance = _get_ttb_instance()

    assert not os.path.exists(os.path.join(log_dir, "train"))

    instance.on_train_batch_end(batch, logs)
    instance.on_save()

    tb_logs = [x for x in _get_logs(os.path.join(log_dir))
               if x.summary.value]

    assert len(tb_logs) == len(logs)
    for (k, v), out in zip(logs.items(), tb_logs):
        assert len(out.summary.value) == 1
        assert out.summary.value[0].tag == f"batch_{k}"
        assert np.isclose(out.summary.value[0].simple_value, v)
        assert out.step == batch


def test_TorchTensorBoard_on_save(_get_ttb_instance, mocker):
    """ Test that :class:`lib.training.tensorboard.on_save` functions """
    # Implicitly checked in other tests, so just make sure it calls flush on the writer
    _, instance = _get_ttb_instance()
    instance._train_writer.flush = mocker.MagicMock()

    instance.on_save()
    instance._train_writer.flush.assert_called_once()


def test_TorchTensorBoard_on_train_end(_get_ttb_instance, mocker):
    """ Test that :class:`lib.training.tensorboard.on_train_end` functions """
    # Saving is already implicitly checked in other tests, so just make sure it calls flush and
    # close on the train writer
    _, instance = _get_ttb_instance()
    instance._train_writer.flush = mocker.MagicMock()
    instance._train_writer.close = mocker.MagicMock()

    instance.on_train_end()
    instance._train_writer.flush.assert_called_once()
    instance._train_writer.close.assert_called_once()
