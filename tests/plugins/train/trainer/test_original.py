#!/usr/bin python3
""" Pytest unit tests for :mod:`plugins.train.trainer.original` Trainer plug in """
# pylint:disable=protected-access,invalid-name

import numpy as np
import pytest
import pytest_mock
import torch

from lib.training.data.collate import BatchMeta
from plugins.train.trainer import original as mod_original
from plugins.train.trainer import base as mod_base


class DummyLoss:  # pylint:disable=too-few-public-methods
    """Dummy loss return"""
    total = 1.0


@pytest.fixture
def _trainer_mocked(mocker: pytest_mock.MockFixture):  # noqa:[F811]
    """ Generate a mocked model and feeder object and patch user config items """

    def _apply_patch(batch_size=8):
        model = mocker.MagicMock()
        conf = mod_base.TrainConfig(folders=["x", "y"],
                                    batch_size=batch_size,
                                    augment_color=False,
                                    flip=False,
                                    warp=False,
                                    cache_landmarks=False)
        instance = mod_original.Trainer(model, conf)
        return instance

    return _apply_patch


@pytest.mark.parametrize("batch_size", (4, 8, 16, 32, 64))
def test_Trainer(batch_size, _trainer_mocked):
    """ Test that original trainer creates correctly """
    instance = _trainer_mocked(batch_size=batch_size)
    assert isinstance(instance, mod_base.TrainerBase)
    assert instance.batch_size == batch_size
    assert hasattr(instance, "train_batch")


def test_Trainer_train_batch(_trainer_mocked, mocker):
    """ Test that original trainer calls the forward and backwards methods """
    instance = _trainer_mocked()
    loss_return = [DummyLoss()]
    instance._forward = mocker.MagicMock(return_value=loss_return)
    instance._backwards_and_apply = mocker.MagicMock()
    instance.model.model.zero_grad = mocker.MagicMock()

    ret_val = instance.train_batch("TEST_INPUT", "TEST_TARGET", "TEST_META")

    assert ret_val == loss_return
    instance._forward.assert_called_once_with("TEST_INPUT", "TEST_TARGET", "TEST_META")
    instance._backwards_and_apply.assert_called_once_with(1.0)
    instance.model.model.zero_grad.assert_called_once()


@pytest.mark.parametrize("outputs", (1, 2, 4))
@pytest.mark.parametrize("batch_size", (4, 8, 16, 32, 64))
def test_Trainer_forward(batch_size,  # pylint:disable=too-many-locals
                         outputs,
                         _trainer_mocked,
                         mocker):
    """ Test that original trainer _forward calls the correct model methods """
    instance = _trainer_mocked(batch_size=batch_size)

    loss_returns = [torch.from_numpy(np.random.random((1, ))) for _ in range(outputs * 2)]
    mock_predictions = [torch.from_numpy(np.random.random((batch_size, 16, 16, 3)))
                        for _ in range(outputs * 2)]
    instance.model.model.return_value = mock_predictions
    instance.loss_func = mocker.MagicMock()

    inputs = list(torch.from_numpy(np.random.random((2, batch_size, 16, 16, 3))))
    targets = [torch.from_numpy(np.random.random((2, batch_size, 16, 16, 3)))
               for _ in range(outputs)]

    # Call forwards
    result = instance._forward(inputs, targets, BatchMeta())

    # Output comes from loss functions
    assert (np.allclose(e.numpy(), a.numpy()) for e, a in zip(result, loss_returns))

    # model forward pass called with inputs split
    train_call = instance.model.model

    call_args, call_kwargs = train_call.call_args
    assert call_kwargs == {"training": True}
    expected_inputs = [a.numpy() for a in inputs]
    actual_inputs = [a.numpy() for a in call_args[0]]
    assert (np.allclose(e, a) for e, a in zip(expected_inputs, actual_inputs))

    # losses called with targets split
    loss_calls = instance.model.model.loss
    expected_targets = [t[i].numpy() for i in range(2) for t in targets]
    expected_predictions = [p.numpy() for p in mock_predictions]
    for loss_call, pred, target in zip(loss_calls, expected_predictions, expected_targets):
        loss_call.assert_called_once()
        call_args, call_kwargs = loss_call.call_args
        assert not call_kwargs
        assert len(call_args) == 2

        actual_target = call_args[0].numpy()
        actual_pred = call_args[1].numpy()
        assert np.allclose(pred, actual_pred)
        assert np.allclose(target, actual_target)


def test_Trainer_backwards_and_apply(_trainer_mocked, mocker):
    """ Test that original trainer _backwards_and_apply calls the correct model methods """
    instance = _trainer_mocked()

    mock_loss = mocker.MagicMock()
    instance.model.model.optimizer.scale_loss = mocker.MagicMock(return_value=mock_loss)
    instance.model.model.optimizer.app = mocker.MagicMock(return_value=mock_loss)

    all_loss = np.random.rand()
    instance._backwards_and_apply(all_loss)

    scale_mock = instance.model.model.optimizer.scale_loss
    scale_mock.assert_called_once()
    assert not scale_mock.call_args[1]
    assert len(scale_mock.call_args[0]) == 1
    assert np.isclose(all_loss, scale_mock.call_args[0][0].cpu().numpy())

    mock_loss.backward.assert_called_once()

    instance.model.model.optimizer.apply.assert_called_once()
