#!/usr/bin python3
""" Pytest unit tests for :mod:`lib.training.lr_warmup` """

import pytest
import pytest_mock

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD

from lib.training import LearningRateWarmup


# pylint:disable=protected-access,redefined-outer-name


@pytest.fixture
def model_fixture():
    """ Model fixture for testing LR Warmup """
    inp = Input((4, 4, 3))
    var_x = Dense(8)(inp)
    model = Model(inputs=inp, outputs=var_x)
    model.compile(optimizer=SGD(), loss="mse")
    return model


_LR_STEPS = [(1e-5, 100),
             (3.4e-6, 250),
             (9e-4, 599),
             (6e-5, 1000)]
_LR_STEPS_IDS = [f"lr:{x[0]}|steps:{x[1]}" for x in _LR_STEPS]


@pytest.mark.parametrize(("target_lr", "steps"), _LR_STEPS, ids=_LR_STEPS_IDS)
def test_init(model_fixture: Model, target_lr: float, steps: int) -> None:
    """ Test class initializes correctly """
    instance = LearningRateWarmup(model_fixture, target_lr, steps)

    attrs = ["_model", "_target_lr", "_steps", "_current_lr", "_current_step", "_reporting_points"]
    assert all(a in instance.__dict__ for a in attrs)
    assert all(a in attrs for a in instance.__dict__)
    assert instance._current_lr == 0.0
    assert instance._current_step == 0

    assert isinstance(instance._model, Model)
    assert instance._target_lr == target_lr
    assert instance._steps == steps

    assert len(instance._reporting_points) == 11
    assert all(isinstance(x, int) for x in instance._reporting_points)
    assert instance._reporting_points == [int(steps * i / 10) for i in range(11)]


_NOTATION = [(1e-5, "1.0e-05"),
             (3.45489e-6, "3.5e-06"),
             (0.0004, "4.0e-04"),
             (0.1234, "1.2e-01")]


@pytest.mark.parametrize(("value", "expected"), _NOTATION, ids=[x[1] for x in _NOTATION])
def test_format_notation(value: float, expected: str) -> None:
    """ Test floats format to string correctly """
    result = LearningRateWarmup._format_notation(value)
    assert result == expected


_LR_STEPS_CURRENT = [(1e-5, 100, 79),
                     (3.4e-6, 250, 250),
                     (9e-4, 599, 0),
                     (6e-5, 1000, 12)]
_LR_STEPS_CURRENT_IDS = [f"lr:{x[0]}|steps:{x[1]}|current_step:{x[2]}" for x in _LR_STEPS_CURRENT]


@pytest.mark.parametrize(("target_lr", "steps", "current_step"),
                         _LR_STEPS_CURRENT,
                         ids=_LR_STEPS_CURRENT_IDS)
def test_set_current_learning_rate(model_fixture: Model,
                                   target_lr: float,
                                   steps: int,
                                   current_step: int) -> None:
    """ Test that learning rate is set correctly """
    instance = LearningRateWarmup(model_fixture, target_lr, steps)
    instance._current_step = current_step
    instance._set_learning_rate()

    assert instance._current_lr == instance._current_step / instance._steps * instance._target_lr
    assert instance._model.optimizer.learning_rate.value.cpu().numpy() == instance._current_lr


_STEPS_CURRENT = [(1000, 1, "start"),
                  (250, 250, "end"),
                  (500, 69, "unreported"),
                  (1000, 200, "reported")]
_STEPS_CURRENT_ID = [f"steps:{x[0]}|current_step:{x[1]}|action:{x[2]}" for x in _STEPS_CURRENT]


@pytest.mark.parametrize(("steps", "current_step", "action"),
                         _STEPS_CURRENT,
                         ids=_STEPS_CURRENT_ID)
def test_output_status(model_fixture: Model,
                       steps: int,
                       current_step: int,
                       action: str,
                       mocker: pytest_mock.MockerFixture) -> None:
    """ Test that information is output correctly """
    mock_logger = mocker.patch("lib.training.lr_warmup.logger.info")
    mock_print = mocker.patch("builtins.print")
    instance = LearningRateWarmup(model_fixture, 5e-5, steps)
    instance._current_step = current_step
    instance._format_notation = mocker.MagicMock()  # type:ignore[method-assign]

    instance._output_status()

    if action == "unreported":
        assert current_step not in instance._reporting_points
        mock_logger.assert_not_called()
        instance._format_notation.assert_not_called()  # type:ignore[attr-defined]
        mock_print.assert_not_called()
        return

    mock_logger.assert_called_once()
    log_message: str = mock_logger.call_args.args[0]
    assert log_message.startswith("[Learning Rate Warmup] ")

    instance._format_notation.assert_called()  # type:ignore[attr-defined]
    notation_args = [
        x.args for x in instance._format_notation.call_args_list]  # type:ignore[attr-defined]
    assert all(len(a) == 1 for a in notation_args)
    assert all(isinstance(a[0], float) for a in notation_args)

    if action == "start":
        mock_print.assert_not_called()
        assert all(x in log_message for x in ("Start: ", "Target: ", "Steps: "))
        assert instance._format_notation.call_count == 2  # type:ignore[attr-defined]
        return

    if action == "end":
        mock_print.assert_called()
        assert "Final Learning Rate: " in log_message
        instance._format_notation.assert_called_once()  # type:ignore[attr-defined]
        return

    if action == "reported":
        mock_print.assert_called()
        assert current_step in instance._reporting_points
        assert all(x in log_message for x in ("Step: ", "Current: ", "Target: "))
        assert instance._format_notation.call_count == 2  # type:ignore[attr-defined]


_STEPS_CURRENT_CALL = [(0, 500, "disabled"),
                       (1000, 500, "progress"),
                       (1000, 1000, "completed"),
                       (1000, 1111, "completed2")]
_STEPS_CURRENT_CALL_ID = [f"steps:{x[0]}|current_step:{x[1]}|action:{x[2]}"
                          for x in _STEPS_CURRENT_CALL]


@pytest.mark.parametrize(("steps", "current_step", "action"),
                         _STEPS_CURRENT_CALL,
                         ids=_STEPS_CURRENT_CALL_ID)
def test__call__(model_fixture: Model,
                 steps: int,
                 current_step: int,
                 action: str,
                 mocker: pytest_mock.MockerFixture) -> None:
    """ Test calling the instance works correctly """
    instance = LearningRateWarmup(model_fixture, 5e-5, steps)
    instance._current_step = current_step
    instance._set_learning_rate = mocker.MagicMock()  # type:ignore[method-assign]
    instance._output_status = mocker.MagicMock()  # type:ignore[method-assign]

    instance()

    if action in ("disabled", "completed", "completed2"):
        assert instance._current_step == current_step
        instance._set_learning_rate.assert_not_called()  # type:ignore[attr-defined]
        instance._output_status.assert_not_called()  # type:ignore[attr-defined]
    else:
        assert instance._current_step == current_step + 1
        instance._set_learning_rate.assert_called_once()  # type:ignore[attr-defined]
        instance._output_status.assert_called_once()  # type:ignore[attr-defined]
