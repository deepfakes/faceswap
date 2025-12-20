#! /usr/env/bin/python3
""" Handles the loading and saving of a model's state file """
from __future__ import annotations

import logging
import os
import time
import typing as T
from importlib import import_module
from inspect import isclass

from lib.logger import parse_class_init
from lib.serializer import get_serializer
from lib.utils import get_module_objects

from lib.config.objects import ConfigItem, GlobalSection
from plugins.train import train_config as cfg

if T.TYPE_CHECKING:
    from lib.config import ConfigValueType


logger = logging.getLogger(__name__)


class State():  # pylint:disable=too-many-instance-attributes
    """ Holds state information relating to the plugin's saved model.

    Parameters
    ----------
    model_dir: str
        The full path to the model save location
    model_name: str
        The name of the model plugin
    no_logs: bool
        ``True`` if Tensorboard logs should not be generated, otherwise ``False``
    """
    def __init__(self,
                 model_dir: str,
                 model_name: str,
                 no_logs: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._serializer = get_serializer("json")
        filename = f"{model_name}_state.{self._serializer.file_extension}"
        self._filename = os.path.join(model_dir, filename)
        self._name = model_name
        self._iterations = 0
        self._mixed_precision_layers: list[str] = []
        self._lr_finder = -1.0
        self._rebuild_model = False
        self._sessions: dict[int, dict] = {}
        self.lowest_avg_loss: float = 0.0
        """float: The lowest average loss seen between save intervals. """

        self._config: dict[str, ConfigValueType] = {}
        self._updateable_options: list[str] = []

        self._load()
        self._session_id = self._new_session_id()
        self._create_new_session(no_logs)
        logger.debug("Initialized %s:", self.__class__.__name__)

    @property
    def filename(self) -> str:
        """ str: Full path to the state filename """
        return self._filename

    @property
    def loss_names(self) -> list[str]:
        """ list: The loss names for the current session """
        return self._sessions[self._session_id]["loss_names"]

    @property
    def current_session(self) -> dict:
        """ dict: The state dictionary for the current :attr:`session_id`. """
        return self._sessions[self._session_id]

    @property
    def iterations(self) -> int:
        """ int: The total number of iterations that the model has trained. """
        return self._iterations

    @property
    def session_id(self) -> int:
        """ int: The current training session id. """
        return self._session_id

    @property
    def sessions(self) -> dict[int, dict[str, T.Any]]:
        """ dict[int, dict[str, Any]]: The session information for each session in the state
        file """
        return {int(k): v for k, v in self._sessions.items()}

    @property
    def mixed_precision_layers(self) -> list[str]:
        """list: Layers that can be switched between mixed-float16 and float32. """
        return self._mixed_precision_layers

    @property
    def lr_finder(self) -> float:
        """ The value discovered from the learning rate finder. -1 if no value stored """
        return self._lr_finder

    @property
    def model_needs_rebuild(self) -> bool:
        """bool: ``True`` if mixed precision policy has changed so model needs to be rebuilt
        otherwise ``False`` """
        return self._rebuild_model

    def _new_session_id(self) -> int:
        """ Generate a new session id. Returns 1 if this is a new model, or the last session id + 1
        if it is a pre-existing model.

        Returns
        -------
        int
            The newly generated session id
        """
        if not self._sessions:
            session_id = 1
        else:
            session_id = max(int(key) for key in self._sessions.keys()) + 1
        logger.debug(session_id)
        return session_id

    def _create_new_session(self, no_logs: bool) -> None:
        """ Initialize a new session, creating the dictionary entry for the session in
        :attr:`_sessions`.

        Parameters
        ----------
        no_logs: bool
            ``True`` if Tensorboard logs should not be generated, otherwise ``False``
        """
        logger.debug("Creating new session. id: %s", self._session_id)
        self._sessions[self._session_id] = {"timestamp": time.time(),
                                            "no_logs": no_logs,
                                            "loss_names": [],
                                            "batchsize": 0,
                                            "iterations": 0,
                                            "config": {k: v for k, v in self._config.items()
                                                       if k in self._updateable_options}}

    def update_session_config(self, key: str, value: T.Any) -> None:
        """ Update a configuration item of the currently loaded session.

        Parameters
        ----------
        key: str
            The configuration item to update for the current session
        value: any
            The value to update to
        """
        old_val = self.current_session["config"][key]
        assert isinstance(value, type(old_val))
        logger.debug("Updating configuration item '%s' from '%s' to '%s'", key, old_val, value)
        self.current_session["config"][key] = value

    def add_session_loss_names(self, loss_names: list[str]) -> None:
        """ Add the session loss names to the sessions dictionary.

        The loss names are used for Tensorboard logging

        Parameters
        ----------
        loss_names: list
            The list of loss names for this session.
        """
        logger.debug("Adding session loss_names: %s", loss_names)
        self._sessions[self._session_id]["loss_names"] = loss_names

    def add_session_batchsize(self, batch_size: int) -> None:
        """ Add the session batch size to the sessions dictionary.

        Parameters
        ----------
        batch_size: int
            The batch size for the current training session
        """
        logger.debug("Adding session batch size: %s", batch_size)
        self._sessions[self._session_id]["batchsize"] = batch_size

    def increment_iterations(self) -> None:
        """ Increment :attr:`iterations` and session iterations by 1. """
        self._iterations += 1
        self._sessions[self._session_id]["iterations"] += 1

    def add_mixed_precision_layers(self, layers: list[str]) -> None:
        """ Add the list of model's layers that are compatible for mixed precision to the
        state dictionary """
        logger.debug("Storing mixed precision layers: %s", layers)
        self._mixed_precision_layers = layers

    def add_lr_finder(self, learning_rate: float) -> None:
        """ Add the optimal discovered learning rate from the learning rate finder

        Parameters
        ----------
        learning_rate : float
            The discovered learning rate
        """
        logger.debug("Storing learning rate from LR Finder: %s", learning_rate)
        self._lr_finder = learning_rate

    def save(self) -> None:
        """ Save the state values to the serialized state file. """
        state = {"name": self._name,
                 "sessions": {k: v for k, v in self._sessions.items()
                              if v.get("iterations", 0) > 0},
                 "lowest_avg_loss": self.lowest_avg_loss,
                 "iterations": self._iterations,
                 "mixed_precision_layers": self._mixed_precision_layers,
                 "lr_finder": self._lr_finder,
                 "config": self._config}
        logger.debug("Saving State: %s", state)
        self._serializer.save(self._filename, state)
        logger.debug("Saved State: '%s'", self._filename)

    def _update_legacy_config(self) -> bool:
        """ Legacy updates for new config additions.

        When new config items are added to the Faceswap code, existing model state files need to be
        updated to handle these new items.

        Current existing legacy update items:

            * loss - If old `dssim_loss` is ``true`` set new `loss_function` to `ssim` otherwise
            set it to `mae`. Remove old `dssim_loss` item

            * l2_reg_term - If this exists, set loss_function_2 to ``mse`` and loss_weight_2 to
            the value held in the old ``l2_reg_term`` item

            * masks - If `learn_mask` does not exist then it is set to ``True`` if `mask_type` is
            not ``None`` otherwise it is set to ``False``.

            * masks type - Replace removed masks 'dfl_full' and 'facehull' with `components` mask

            * clipnorm - Only existed in 2 models (DFL-SAE + Unbalanced). Replaced with global
            option autoclip

            * Clip model - layer names have had to be changed to replace dots with underscores, so
            replace these

        Returns
        -------
        bool
            ``True`` if legacy items exist and state file has been updated, otherwise ``False``
        """
        logger.debug("Checking for legacy state file update")
        priors = ["dssim_loss", "mask_type", "mask_type", "l2_reg_term", "clipnorm", "autoclip"]
        new_items = ["loss_function", "learn_mask", "mask_type", "loss_function_2",
                     "gradient_clipping", "clipping"]
        updated = False
        for old, new in zip(priors, new_items):
            if old not in self._config:
                logger.debug("Legacy item '%s' not in state config. Skipping update", old)
                continue

            # dssim_loss > loss_function
            if old == "dssim_loss":
                self._config[new] = "ssim" if self._config[old] else "mae"
                del self._config[old]
                updated = True
                logger.info("Updated state config from legacy dssim format. New config loss "
                            "function: '%s'", self._config[new])
                continue

            # Add learn mask option and set to True if model has "penalized_mask_loss" specified
            if old == "mask_type" and new == "learn_mask" and new not in self._config:
                self._config[new] = self._config["mask_type"] is not None
                updated = True
                logger.info("Added new 'learn_mask' state config item for this model. Value set "
                            "to: %s", self._config[new])
                continue

            # Replace removed masks with most similar equivalent
            if old == "mask_type" and new == "mask_type" and self._config[old] in ("facehull",
                                                                                   "dfl_full"):
                old_mask = self._config[old]
                self._config[new] = "components"
                updated = True
                logger.info("Updated 'mask_type' from '%s' to '%s' for this model",
                            old_mask, self._config[new])

            # Replace l2_reg_term with the correct loss_2_function and update the value of
            # loss_2_weight
            if old == "l2_reg_term":
                self._config[new] = "mse"
                self._config["loss_weight_2"] = self._config[old]
                del self._config[old]
                updated = True
                logger.info("Updated state config from legacy 'l2_reg_term' to 'loss_function_2'")

            # Replace clipnorm with correct gradient clipping type and value
            if old == "clipnorm":
                self._config[new] = "norm"
                del self._config[old]
                updated = True
                logger.info("Updated state config from legacy '%s' to  '%s: %s'", old, new, old)

            # Replace autoclip with correct gradient clipping type
            if old == "autoclip":
                self._config[new] = old
                del self._config[old]
                updated = True
                logger.info("Updated state config from legacy '%s' to '%s: %s'", old, new, old)

        # Update Clip layer names from dots to underscores
        mixed_precision = self._mixed_precision_layers
        if any("." in name for name in mixed_precision):
            self._mixed_precision_layers = [x.replace(".", "_") for x in mixed_precision]
            updated = True
            logger.info("Updated state config for legacy 'mixed_precision' storage of Clip layers")

        logger.debug("State file updated for legacy config: %s", updated)
        return updated

    def _get_global_options(self) -> dict[str, ConfigItem]:
        """ Obtain all of the current global user config options

        Returns
        -------
        dict[str, :class:`lib.config.objects.ConfigItem`]
            All of the current global user configuration options
        """
        objects = {key: val for key, val in vars(cfg).items()
                   if isinstance(val, ConfigItem)
                   or isclass(val) and issubclass(val, GlobalSection) and val != GlobalSection}

        retval: dict[str, ConfigItem] = {}
        for key, obj in objects.items():
            if isinstance(obj, ConfigItem):
                retval[key] = obj
                continue
            for name, opt in obj.__dict__.items():
                if isinstance(opt, ConfigItem):
                    retval[name] = opt
        logger.debug("Loaded global config options: %s", {k: v.value for k, v in retval.items()})
        return retval

    def _get_model_options(self) -> dict[str, ConfigItem]:
        """ Obtain all of the currently configured model user config options """
        mod_name = f"plugins.train.model.{self._name}_defaults"
        try:
            mod = import_module(mod_name)
        except ModuleNotFoundError:
            logger.debug("No plugin specific defaults file found at '%s'", mod_name)
            return {}

        retval = {k: v for k, v in vars(mod).items() if isinstance(v, ConfigItem)}
        logger.debug("Loaded '%s' config options: %s",
                     self._name, {k: v.value for k, v in retval.items()})
        return retval

    def _update_config(self) -> None:
        """ Update the loaded training config with the one contained within the values loaded
        from the state file.

        Check for any `fixed`=``False`` parameter changes and log info changes.

        Update any legacy config items to their current versions.
        """
        legacy_update = self._update_legacy_config()
        # Add any new items to state config for legacy purposes where the new default may be
        # detrimental to an existing model.
        legacy_defaults: dict[str, str | int | bool | float] = {"centering": "legacy",
                                                                "coverage": 62.5,
                                                                "mask_loss_function": "mse",
                                                                "optimizer": "adam",
                                                                "mixed_precision": False}
        rebuild_tasks = ["mixed_precision"]
        options = self._get_global_options() | self._get_model_options()
        for key, opt in options.items():
            val: ConfigValueType = opt()

            if key not in self._config:
                val = legacy_defaults.get(key, val)
                logger.info("Adding new config item to state file: '%s': %s", key, repr(val))
                self._config[key] = val

            old_val = self._config[key]
            old_val = "none" if old_val is None else old_val  # We used to allow NoneType. No more

            if not opt.fixed:
                self._updateable_options.append(key)

            if not opt.fixed and val != old_val:
                self._config[key] = val
                logger.info("Config item: '%s' has been updated from %s to %s",
                            key, repr(old_val), repr(val))
                self._rebuild_model = self._rebuild_model or key in rebuild_tasks
                continue

            if val != old_val:
                logger.debug("Fixed config item '%s' Updated from %s to %s from state file",
                             key, repr(val), repr(old_val))
                opt.set(old_val)

        if legacy_update:
            self.save()
        logger.info("Using configuration saved in state file")
        logger.debug("Updateable items: %s", self._updateable_options)

    def _generate_config(self) -> None:
        """ Generate an initial state config based on the currently selected user config """
        options = self._get_global_options() | self._get_model_options()
        for key, val in options.items():
            self._config[key] = val.value
            if not val.fixed:
                self._updateable_options.append(key)

        logger.debug("Generated initial state config for '%s': %s", self._name, self._config)
        logger.debug("Updateable items: %s", self._updateable_options)

    def _load(self) -> None:
        """ Load a state file and set the serialized values to the class instance.

        Updates the model's config with the values stored in the state file.
        """
        logger.debug("Loading State")

        if not os.path.exists(self._filename):
            logger.info("No existing state file found. Generating.")
            self._generate_config()
            return

        state = self._serializer.load(self._filename)
        self._name = state.get("name", self._name)
        self._sessions = state.get("sessions", {})

        self.lowest_avg_loss = state.get("lowest_avg_loss", 0.0)
        if isinstance(self.lowest_avg_loss, dict):
            lowest_avg_loss = sum(self.lowest_avg_loss.values())
            logger.debug("Collating legacy lowest_avg_loss from %s to %s",
                         self.lowest_avg_loss, lowest_avg_loss)
            self.lowest_avg_loss = lowest_avg_loss

        self._iterations = state.get("iterations", 0)
        self._mixed_precision_layers = state.get("mixed_precision_layers", [])
        self._lr_finder = state.get("lr_finder", -1.0)
        self._config = state.get("config", {})
        logger.debug("Loaded state: %s", state)
        self._update_config()


__all__ = get_module_objects(__name__)
