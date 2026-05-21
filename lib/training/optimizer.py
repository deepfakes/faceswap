#!/usr/bin/env python3
"""Wraps the selected Torch optimizer and handles optimizer related functions such as loss scaling,
clipping and gradient accumulation"""
from __future__ import annotations

import logging
import typing as T

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from lib.logger import parse_class_init
from lib.model.autoclip import AutoClipper
from lib.model import optimizers
from lib.utils import get_module_objects

from .lr_finder import LearningRateFinder
from .lr_warmup import WarmupScheduler

if T.TYPE_CHECKING:
    from keras import Model as K_Model, Variable
    from plugins.train.model._base import ModelBase as Model
    from plugins.train.train_config import Optimizer as OptConfig
    from .train import Trainer


logger = logging.getLogger(__name__)

_OPTIMIZERS = {"adabelief": optimizers.AdaBelief,
               "adam": torch.optim.Adam,
               "adamax": torch.optim.Adamax,
               "adamw": torch.optim.AdamW,
               "lion": optimizers.Lion,
               "nadam": torch.optim.NAdam,
               "rms-prop": torch.optim.RMSprop}


def get_parameter_group_ids(trainable_variables: list[Variable]
                            ) -> dict[int, T.Literal["decay", "no_decay"]]:
    """Obtain the index of each item in the keras model's trainable weights that belong to each
    of the optimizer's parameter groups (ie split by weights that take decay and don't take decay)

    Parameters
    ----------
    trainable_variables
        list of trainable variables from keras model

    Returns
    -------
    dictionary of keras model's trainable weight index to the name of the parameter group
    """
    retval: dict[int, T.Literal["decay", "no_decay"]] = {}
    for idx, var in enumerate(trainable_variables):
        retval[idx] = "no_decay" if var.ndim <= 1 or var.name.endswith("bias") else "decay"

    logger.debug("parameter group ids: %s", retval)
    return retval


class GradClip:
    """Handles the clipping of gradients based on user supplied parameters

    Parameters
    ----------
    method
        The clipping method to use
    value
        The clipping value to use. For autoclip this is the percentile to clip at (a value of 1.0
        will clip at the 10th percentile a value of 2.5 will clip at the 25th percentile etc)
    autoclip_history
        The history length for auto clipping. Default: 10000
    """
    def __init__(self,
                 method: T.Literal["autoclip", "global_norm", "norm", "value"],
                 value: float,
                 autoclip_history: int = 10000) -> None:
        logger.debug(parse_class_init(locals()))
        self._value = value
        self._clipper = self._get_clipper(method, autoclip_history)

    @classmethod
    def _clip_norm(cls, parameters: list[nn.Parameter], max_norm: float) -> None:
        """Clip each parameter independently by its own norm

        Parameters
        ----------
        parameters
            The parameters to clip
        max_norm
            The value to clip by
        """
        with torch.no_grad():
            for param in parameters:
                if param.grad is None:
                    continue
                grad = param.grad
                norm = grad.norm(2)
                if norm > max_norm:
                    grad.mul_(max_norm / norm)

    def _get_clipper(self,
                     method: T.Literal["autoclip", "global_norm", "norm", "value"],
                     autoclip_history: int) -> T.Callable[[list[nn.Parameter], float],
                                                          None | torch.Tensor]:
        """Obtain the correct function to clip the gradients based on the selected method

        Parameters
        ----------
        method
            The clipping method to use
        autoclip_history
            The history length for auto clipping

        Returns
        -------
        The function used to clip the gradients
        """
        methods: dict[str, T.Callable[[list[nn.Parameter], float], None | torch.Tensor]] = {
            "autoclip": AutoClipper(int(self._value * 10), history_size=autoclip_history),
            "global_norm": nn.utils.clip_grad_norm_,
            "norm": self._clip_norm,
            "value": nn.utils.clip_grad_value_}
        if method not in methods:
            raise ValueError(f"'{method}' is not a valid clipping method. Select "
                             f"from {list(methods)}")
        retval = methods[method]
        logger.debug("[GradClip] Got clipper '%s': %s", method, retval)
        return retval

    def __call__(self, parameters: list[nn.Parameter]) -> None:
        """Clip the given parameters by the chosen method

        Parameters
        ----------
        parameters
            The parameters to clip
        """
        self._clipper(parameters, self._value)


class Optimizer:
    """Object for managing the selected Torch optimizer

    Parameters
    ----------
    model
        The model that is to be trained
    config
        The optimizer user configuration options
    mixed_precision
        ``True`` to train using mixed precision. Default: ``False``
    warmup_steps
        The number of steps to warmup the learning rate for. Default: 0
    """
    def __init__(self,
                 model: Model,
                 config: type[OptConfig],
                 mixed_precision: bool = False,
                 warmup_steps: int = 0) -> None:
        logger.debug(parse_class_init(locals()))
        self._mixed_precision = mixed_precision
        self._accumulation_steps = config.gradient_accumulation()
        self._scaler = None if not mixed_precision else torch.amp.grad_scaler.GradScaler()
        self._clip = None if config.gradient_clipping() == "none" else GradClip(
            T.cast(T.Literal["autoclip", "global_norm", "norm", "value"],
                   config.gradient_clipping()),
            config.clipping_value(),
            config.autoclip_history())

        self._optimizer = self._get_optimizer(model.model, config)
        self._warmup = None if warmup_steps < 1 else WarmupScheduler(self._optimizer, warmup_steps)
        self._lr_scheduler: ExponentialLR | None = None

        self._load_state(model)

        self._accumulation_count = 0
        self._session_steps = 0

    @classmethod
    def _get_optimizer_kwargs(cls, config: type[OptConfig]) -> dict[str, T.Any]:
        """Obtain the keyword arguments for the requested optimizer from the user configuration

        Parameters
        ----------
        config
            The optimizer user configuration options

        Returns
        -------
        The optimizer keyword arguments
        """
        retval: dict[str, T.Any] = {"weight_decay": config.weight_decay()}
        name = config.optimizer()

        if name != "lion":
            retval["eps"] = 10 ** config.epsilon_exponent()

        if name in ("adabelief", "adam", "adamw", "adamax", "lion", "nadam"):
            retval["betas"] = (config.ada_beta_1(), config.ada_beta_2())

        if name in ("adabelief", "adam", "adamw"):
            retval["amsgrad"] = config.ada_amsgrad()

        logger.debug("[Optimizer] '%s' kwargs: %s", name, retval)
        return retval

    def _get_optimizer(self, model: K_Model, config: type[OptConfig]) -> torch.optim.Optimizer:
        """Obtain the configured optimizer the given configuration file options

        Parameters
        ----------
        model
            The keras model that is to be trained
        config
            The optimizer user configuration options

        Returns
        -------
        The requested configured optimizer
        """
        name = config.optimizer()
        if name not in _OPTIMIZERS:
            raise ValueError(f"'{name}' is not a valid optimizer. Select from {list(_OPTIMIZERS)}")
        optimizer = _OPTIMIZERS[name]

        retval = optimizer(self._get_parameter_groups(model, config.weight_decay()),
                           lr=config.learning_rate(),
                           **self._get_optimizer_kwargs(config))
        logger.debug("[Optimizer] Got optimizer '%s': %s", name, retval)
        return retval

    def _get_parameter_groups(self, model: K_Model, weight_decay: float
                              ) -> tuple[dict[T.Literal["params", "weight_decay"],
                                              list[nn.Parameter] | float],
                                         dict[T.Literal["params", "weight_decay"],
                                              list[nn.Parameter] | float]]:
        """Obtain the parameter groups from within the keras model

        Parameters
        ----------
        model
            The keras model that is to be trained
        weight_decay
            The amount of weight decay to apply

        Returns
        -------
        The parameters that require weight decay in position 0 and no weight decay in position 1
        """
        index_map = get_parameter_group_ids(model.trainable_variables)
        groups: dict[T.Literal["decay", "no_decay"], list[nn.Parameter]] = {"decay": [],
                                                                            "no_decay": []}
        # pylint:disable=protected-access
        for idx, var in enumerate(model.trainable_variables):
            if not hasattr(var, "_value") or not isinstance(var._value, nn.Parameter):
                raise RuntimeError(
                    f"Cannot extract torch parameter from keras.Variable '{var.name}'. "
                    "Keras version may have changed internal structure.")
            groups[index_map[idx]].append(var._value)

        retval: tuple[dict[T.Literal["params", "weight_decay"], list[nn.Parameter] | float],
                      dict[T.Literal["params", "weight_decay"], list[nn.Parameter] | float]] = (
                          {"params": groups["decay"], "weight_decay": weight_decay},
                          {"params": groups["no_decay"], "weight_decay": 0.0}
                          )

        logger.debug("[Optimizer] decay params: %s, no_decay params: %s",
                     {k: len(v) if isinstance(v, list) else v for k, v in retval[0].items()},
                     {k: len(v) if isinstance(v, list) else v for k, v in retval[1].items()})
        return retval

    def _from_legacy(self,
                     state: dict[str, T.Any]) -> dict[str, T.Any] | None:
        """Populate the remaining param_group items for weights from legacy saved keras optimizer
        and validate shapes

        Parameters
        ----------
        state
            The partial state_dict migrated from a keras optimizer

        Returns
        -------
            The final state_dict grouped for torch or ``None`` if weights could not be mapped
        """
        logger.debug("[Optimizer] Loading weights from legacy Keras optimizer")
        imported_params = state["optimizer"]["state"]
        p_groups = self._optimizer.param_groups
        exists = [p for g in p_groups for p in g["params"]]

        if len(imported_params) != len(exists):
            logger.warning("Imported optimizer weights count mismatch. Optimizer will be reset")
            return None

        for idx, exist in enumerate(exists):
            # exp_avg for ada based optimizers, square_avg for rms-prop
            key = "exp_avg" if "exp_avg" in imported_params[idx] else "square_avg"
            if imported_params[idx][key].shape != exist.shape:
                logger.warning("Imported optimizer weights shape mismatch. "
                               "Optimizer will be reset")
                return None

        imported_p_groups = state["optimizer"]["param_groups"]
        if len(p_groups) != len(imported_p_groups):
            logger.warning("Parameter group count mismatch (exists: %s, imported: %s). "
                           "Optimizer will be reset", len(p_groups), len(imported_p_groups))
            return None

        for idx, group in enumerate(p_groups):
            p_group = state["optimizer"]["param_groups"][idx]
            state["optimizer"]["param_groups"][idx] = {k: p_group.get(k, v)
                                                       for k, v in group.items()}

        return state

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None:
        """Load the serialized data from a state dict into this object

        Parameters
        ----------
        state_dict
            The serialized data to load
        """
        logger.debug("[Optimizer] Loading state_dict")
        self._optimizer.load_state_dict(state_dict["optimizer"])
        if self._scaler is not None and state_dict.get("scaler") is not None:
            logger.debug("[Optimizer] Loading scaler state_dict: %s", state_dict["scaler"])
            self._scaler.load_state_dict(state_dict["scaler"])

    def _load_state(self, model: Model) -> None:
        """Load weights if resuming and optimizer weights exist within the model file.

        Also handles migration of legacy Keras optimizer weights to torch optimizer

        Parameters
        ----------
        model
            The model that is to be trained
        """
        if not model.io.model_exists:
            logger.debug("[Optimizer] Model file does not exist. Not loading state")
            return

        state = model.io.load_optimizer()
        if state is None:
            logger.debug("[Optimizer] No optimizer saved in model file")
            return

        if state["version"] == 0.5:  # Migrating from keras optimizer
            state = self._from_legacy(state)
            if state is None:
                return

        self.load_state_dict(state_dict=state)

    def backward(self, loss: torch.Tensor) -> None:
        """Perform the optimizer's backward pass

        Parameters
        ----------
        loss
            The loss scalar from the forward pass
        """
        scaled = loss / self._accumulation_steps
        if self._scaler:
            self._scaler.scale(scaled).backward()
        else:
            scaled.backward()

    def step(self) -> None:
        """Perform the optimizer step if valid and zero the gradients.

        Handles gradient accumulation, scaling for mixed precision and gradient clipping
        """
        self._accumulation_count += 1
        if self._accumulation_count != self._accumulation_steps:
            return

        if self._clip is not None and self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
        if self._clip is not None:
            self._clip([p for g in self._optimizer.param_groups for p in g["params"]])

        if self._scaler is None:
            self._optimizer.step()
        else:
            self._scaler.step(self._optimizer)
            self._scaler.update()

        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        elif self._warmup is not None and self._session_steps < self._warmup.steps:
            self._session_steps += 1
            self._warmup.step()

        self._optimizer.zero_grad(set_to_none=True)
        self._accumulation_count = 0

    def state_dict(self) -> dict[str, T.Any]:
        """Serialized data as a dict for relevant options contained in this class

        Returns
        -------
        The serialized data for this object for saving and loading
        """
        return {"version": 1.0,
                "optimizer": self._optimizer.state_dict(),
                "scaler": None if self._scaler is None else self._scaler.state_dict()}

    def to(self, device: torch.Device) -> None:
        """Place the optimizer onto the given device

        Parameters
        ----------
        device
            The device to place the optimizer on to
        """
        logger.debug("[Optimizer] to: %s", device)
        for state in self._optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def set_lr(self, lr: float) -> None:
        """Manually assign the optimizer's learning rate with the given value

        Parameters
        ----------
        lr
            The learning rate to apply to the optimizer
        """
        logger.debug("[Optimizer] Setting learning rate to: %s", lr)
        for p in self._optimizer.param_groups:
            p["lr"] = lr
            if "initial_lr" in p:
                p["initial_lr"] = lr

    def find_learning_rate(self,
                           trainer: Trainer,
                           steps: int,
                           start_lr: float,
                           end_lr: float,
                           strength: T.Literal["default", "aggressive", "extreme"],
                           mode: T.Literal["set", "graph_and_set", "graph_and_exit"]) -> bool:
        """Use the Learning Rate Finder to discover the optimal learning rate

        Parameters
        ----------
        trainer
            The training loop with the loaded training plugin
        steps
            The number of iterations to run the learning rate finder for
        start_lr
            The learning rate to start scanning from
        end_lr
            The final learning rate to scan until
        strength
            How aggressively to set the optimal learning rate
        mode
            The mode to run the Learning Rate Finder in

        Returns
        -------
        ``True`` if an optimal learning rate was discovered.
        """
        original_lr = self._optimizer.param_groups[0].get("initial_lr",
                                                          self._optimizer.param_groups[0]["lr"])
        self.set_lr(start_lr)
        opt_state = self._optimizer.state_dict()
        scaler_state = None if self._scaler is None else self._scaler.state_dict()

        gamma: float = (end_lr / start_lr) ** (1.0 / steps)
        self._lr_scheduler = ExponentialLR(self._optimizer, gamma=gamma)

        lrf = LearningRateFinder(trainer, self._lr_scheduler, steps, strength, mode)
        lrf.find()

        del self._lr_scheduler
        self._lr_scheduler = None

        if lrf.best_lr is None:
            return False

        logger.debug("[Optimizer] Resetting optimizer for LearningRateFinder: %s", opt_state)
        self._optimizer.load_state_dict(opt_state)
        if self._scaler is not None and scaler_state is not None:
            self._scaler.load_state_dict(scaler_state)

        logger.info("Updating Learning Rate from %s to %s",
                    f"{original_lr:.1e}", f"{lrf.best_lr:.1e}")
        self.set_lr(lrf.best_lr)

        return True


__all__ = get_module_objects(__name__)
