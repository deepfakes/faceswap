#! /usr/env/bin/python3
""" Unit tests for Learning Rate Finder. """

import pytest
import pytest_mock

import numpy as np

from lib.training.lr_finder import LearningRateFinder
from plugins.train import train_config as cfg

# pylint:disable=unused-import
from tests.lib.config.helpers import patch_config  # noqa:[F401]

# pylint:disable=protected-access,invalid-name,redefined-outer-name


@pytest.fixture
def _trainer_mock(patch_config, mocker: pytest_mock.MockFixture):  # noqa:[F811]
    """ Generate a mocked model and feeder object and patch user config items """
    def _apply_patch(iters=1000, mode="default", strength="default"):
        patch_config(cfg, {"lr_finder_iterations": iters})
        patch_config(cfg, {"lr_finder_mode": mode})
        patch_config(cfg, {"lr_finder_strength": strength})
        trainer = mocker.MagicMock()
        model = mocker.MagicMock()
        model.name = "TestModel"
        optimizer = mocker.MagicMock()
        trainer._plugin.model = model
        trainer._plugin.model.model.optimizer = optimizer
        return trainer, model, optimizer
    return _apply_patch


_STRENGTH_LOOKUP = {"default": 10, "aggressive": 5, "extreme": 2.5}


_LR_CONF = ((20, "graph_and_set", "default"),
            (500, "set", "aggressive"),
            (1000, "graph_and_exit", "extreme"))
_LR_CONF_PARAMS = ("iters", "mode", "strength")

_LR_CMDS = ((4, 0.98), (8, 0.66), (2, 0.33)
            )
_LR_CMDS_PARAMS = ("stop_factor", "beta")
_LR_CMDS_IDS = [f"stop:{x[0]}|beta:{x[1]}" for x in _LR_CMDS]


@pytest.mark.parametrize(_LR_CONF_PARAMS, _LR_CONF)
@pytest.mark.parametrize(_LR_CMDS_PARAMS, _LR_CMDS, ids=_LR_CMDS_IDS)
def test_LearningRateFinder_init(iters, mode, strength, stop_factor, beta, _trainer_mock):
    """ Test lib.train.LearingRateFinder.__init__ """
    trainer, model, optimizer = _trainer_mock(iters, mode, strength)
    lrf = LearningRateFinder(trainer, stop_factor=stop_factor, beta=beta)
    assert lrf._trainer is trainer
    assert lrf._model is model
    assert lrf._optimizer is optimizer
    assert lrf._start_lr == 1e-10
    assert lrf._stop_factor == stop_factor
    assert lrf._beta == beta


_BATCH_END = ((1, 0.01, 1e-5, 0.5),
              (27, 0.01, 1e-5, 1e-6),
              (42, 0.001, 1e-5, 0.002),)
_BATCH_END_PARAMS = ("iteration", "loss", "learning_rate", "best")
_BATCH_END_IDS = [f"iter:{x[0]}|loss:{x[1]}|lr:{x[2]}" for x in _BATCH_END]


@pytest.mark.parametrize(_LR_CMDS_PARAMS, _LR_CMDS, ids=_LR_CMDS_IDS)
@pytest.mark.parametrize(_BATCH_END_PARAMS, _BATCH_END, ids=_BATCH_END_IDS)
def test_LearningRateFinder_on_batch_end(iteration,
                                         loss,
                                         learning_rate,
                                         best,
                                         stop_factor,
                                         beta,
                                         _trainer_mock,
                                         mocker):
    """ Test lib.train.LearingRateFinder._on_batch_end """
    trainer, model, optimizer = _trainer_mock()
    lrf = LearningRateFinder(trainer, stop_factor=stop_factor, beta=beta)
    optimizer.learning_rate.assign = mocker.MagicMock()
    optimizer.learning_rate.numpy = mocker.MagicMock(return_value=learning_rate)

    initial_avg = lrf._loss["avg"]
    lrf._loss["best"] = best
    lrf._on_batch_end(iteration, loss)

    assert lrf._metrics["learning_rates"][-1] == learning_rate
    assert lrf._loss["avg"] == (lrf._beta * initial_avg) + ((1 - lrf._beta) * loss)
    assert lrf._metrics["losses"][-1] == lrf._loss["avg"] / (1 - (lrf._beta ** iteration))

    if iteration > 1 and lrf._metrics["losses"][-1] > lrf._stop_factor * lrf._loss["best"]:
        assert model.model.stop_training is True
        optimizer.learning_rate.assign.assert_not_called()
        return

    if iteration == 1:
        assert lrf._loss["best"] == lrf._metrics["losses"][-1]

    assert model.model.stop_training is not True
    optimizer.learning_rate.assign.assert_called_with(
        learning_rate * lrf._lr_multiplier)


@pytest.mark.parametrize(_LR_CONF_PARAMS, _LR_CONF)
def test_LearningRateFinder_train(iters,  # pylint:disable=too-many-locals
                                  mode,
                                  strength,
                                  _trainer_mock,
                                  mocker):
    """ Test lib.train.LearingRateFinder._train """
    trainer, _, _ = _trainer_mock(iters, mode, strength)

    mock_loss_return = np.random.rand(2).tolist()
    trainer.train_one_batch = mocker.MagicMock(return_value=mock_loss_return)

    lrf = LearningRateFinder(trainer)

    lrf._on_batch_end = mocker.MagicMock()
    lrf._update_description = mocker.MagicMock()

    lrf._train()

    trainer.train_one_batch.assert_called()
    assert trainer.train_one_batch.call_count == iters

    train_call_args = [mocker.call(x + 1, mock_loss_return[0]) for x in range(iters)]
    assert lrf._on_batch_end.call_args_list == train_call_args

    lrf._update_description.assert_called()
    assert lrf._update_description.call_count == iters

    # NaN break
    mock_loss_return = (np.nan, np.nan)
    trainer.train_one_batch = mocker.MagicMock(return_value=mock_loss_return)

    lrf._train()

    assert trainer.train_one_batch.call_count == 1  # Called once

    assert lrf._update_description.call_count == iters  # Not called
    assert lrf._on_batch_end.call_count == iters  # Not called


def test_LearningRateFinder_rebuild_optimizer(_trainer_mock):
    """ Test lib.train.LearingRateFinder._rebuild_optimizer """
    trainer, _, _ = _trainer_mock()
    lrf = LearningRateFinder(trainer)

    class Dummy:
        """ Dummy Optimizer"""
        name = "test"

        def get_config(self):
            """Dummy get_config"""
            return {}

    opt = Dummy()
    new_opt = lrf._rebuild_optimizer(opt)
    assert isinstance(new_opt, Dummy) and opt is not new_opt


@pytest.mark.parametrize(_LR_CONF_PARAMS, _LR_CONF)
@pytest.mark.parametrize("new_lr", (1e-4, 3.5e-5, 9.3e-6))
def test_LearningRateFinder_reset_model(iters, mode, strength, new_lr, _trainer_mock, mocker):
    """ Test lib.train.LearingRateFinder._reset_model """
    trainer, model, optimizer = _trainer_mock(iters, mode, strength)
    model.state.add_lr_finder = mocker.MagicMock()
    model.state.save = mocker.MagicMock()
    model.model.load_weights = mocker.MagicMock()

    old_optimizer = optimizer
    new_optimizer = mocker.MagicMock()

    def compile_side_effect(*args, **kwargs):  # pylint:disable=unused-argument
        """ Side effect for model.compile"""
        model.model.optimizer = new_optimizer

    model.model.compile.side_effect = compile_side_effect

    lrf = LearningRateFinder(trainer)
    lrf._rebuild_optimizer = mocker.MagicMock()

    lrf._reset_model(1e-5, new_lr)

    model.state.add_lr_finder.assert_called_with(new_lr)
    model.state.save.assert_called_once()

    if mode == "graph_and_exit":
        lrf._rebuild_optimizer.assert_not_called()
        model.model.compile.assert_not_called()
        model.model.load_weights.assert_not_called()
        assert model.model.optimizer is old_optimizer
        new_optimizer.learning_rate.assign.assert_not_called()
    else:
        lrf._rebuild_optimizer.assert_called_once_with(old_optimizer)
        model.model.load_weights.assert_called_once()
        model.model.compile.assert_called_once()
        assert model.model.optimizer is new_optimizer
        new_optimizer.learning_rate.assign.assert_called_once_with(new_lr)


_LR_FIND = (
    (True,  [0.100, 0.050, 0.025], 0.025, [1e-5, 1e-4, 1e-3], "model_exist"),
    (False, [0.100, 0.050, 0.025], 0.025, [1e-5, 1e-4, 1e-3], "no_model"),
    (True, [0.100, 0.050, 0.025], 0.025, [1e-5, 1e-4, 1e-10], "low_lr"),
            )
_LR_PARAMS_FIND = ("exists", "losses", "best", "learning_rates")


@pytest.mark.parametrize(_LR_PARAMS_FIND,
                         [x[:-1] for x in _LR_FIND],
                         ids=[x[-1] for x in _LR_FIND])
@pytest.mark.parametrize(_LR_CONF_PARAMS, _LR_CONF)
@pytest.mark.parametrize(_LR_CMDS_PARAMS, _LR_CMDS[0:1])
def test_LearningRateFinder_find(iters,  # pylint:disable=too-many-arguments,too-many-positional-arguments  # noqa[E501]
                                 mode,
                                 strength,
                                 stop_factor,
                                 beta,
                                 exists,
                                 losses,
                                 best,
                                 learning_rates,
                                 _trainer_mock,
                                 mocker):
    """ Test lib.train.LearingRateFinder.find """
    # pylint:disable=too-many-locals
    trainer, model, optimizer = _trainer_mock(iters, mode, strength)
    model.io.model_exists = exists
    model.io.save = mocker.MagicMock()
    original_lr = float(np.random.rand())
    optimizer.learning_rate.numpy = mocker.MagicMock(return_value=original_lr)
    optimizer.learning_rate.assign = mocker.MagicMock()
    mocker.patch("shutil.rmtree")

    lrf = LearningRateFinder(trainer, stop_factor=stop_factor, beta=beta)

    train_mock = mocker.MagicMock()
    plot_mock = mocker.MagicMock()
    reset_mock = mocker.MagicMock()
    lrf._train = train_mock
    lrf._plot_loss = plot_mock
    lrf._reset_model = reset_mock

    lrf._metrics = {"losses": losses, "learning_rates": learning_rates}
    lrf._loss = {"best": best}

    result = lrf.find()

    if exists:
        model.io.save_assert_not_called()
    else:
        model.io.save.assert_called_once()

    optimizer.learning_rate.assign.assert_called_with(lrf._start_lr)
    train_mock.assert_called_once()

    new_lr = learning_rates[losses.index(best)] / _STRENGTH_LOOKUP[strength]
    if new_lr < 1e-9:
        plot_mock.assert_not_called()
        reset_mock.assert_not_called()
        assert not result
        return

    plot_mock.assert_called_once()
    reset_mock.assert_called_once_with(original_lr, new_lr)
    assert result
