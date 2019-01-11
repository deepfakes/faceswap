#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of NNMeta objects.
    See the class for details.
"""
import logging
import os
import sys

from json import JSONDecodeError
from keras import losses
from keras.optimizers import Adam
from keras.utils import multi_gpu_model

from lib import Serializer
from lib.model.losses import DSSIMObjective, PenalizedLoss
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(model_name):
    """ Return the config for the requested model """
    return Config(model_name).config_dict


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self, model_dir, gpus, input_shape=None, encoder_dim=None, trainer="original"):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', gpus: %s, input_shape: %s, "
                     "encoder_dim: %s)", self.__class__.__name__, model_dir, gpus,
                     input_shape, encoder_dim)
        self.config = get_config(".".join(self.__module__.split(".")[-2:]))
        self.model_dir = model_dir
        self.gpus = gpus
        self.input_shape = input_shape
        self.encoder_dim = encoder_dim
        self.trainer = trainer

        self.masks = None  # List of masks to be set if masks are used

        self.name = self.set_model_name()
        self.networks = dict()  # Networks for the model
        self.predictors = dict()  # Predictors for model
        self.loss_names = dict()  # Loss names for model
        self.history = dict()  # Loss history per save iteration)
        self.state = State(self.model_dir, self.base_filename)
        # Training information specific to the model should be placed in this
        # dict for reference by the trainer.
        self.training_opts = self.set_training_data()

        self.add_networks()
        self.initialize()
        self.log_summary()
        self.compile_predictors()
        logger.debug("Initialized ModelBase (%s)", self.__class__.__name__)

    def set_training_data(self):
        """ Override to set model specific training data """
        return dict()

    def initialize(self):
        """ Override for Model Specific Initialization
            Must list of masks if used or None if not """
        raise NotImplementedError

    def add_networks(self):
        """ Override to add neural networks """
        raise NotImplementedError

    def set_model_name(self):
        """ Set the model name based on the subclass """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        retval = os.path.splitext(basename)[0].lower()
        logger.debug("model name: '%s'", retval)
        return retval

    def add_network(self, network_type, side, network):
        """ Add a NNMeta object """
        logger.debug("network_type: '%s', side: '%s', network: '%s'", network_type, side, network)
        filename = "{}_{}".format(self.base_filename, network_type.lower())
        name = network_type.lower()
        if side:
            filename += "_{}".format(side.upper())
            name += "_{}".format(side.lower())
        filename += ".h5"
        logger.debug("name: '%s', filename: '%s'", name, filename)
        self.networks["{}".format(name)] = NNMeta(str(self.model_dir / filename),
                                                  network_type,
                                                  side,
                                                  network)

    def add_predictors(self, model_a, model_b):
        """ Add the predictors to the predictors dictionary """
        logger.debug("Adding predictors: (model_a: %s, model_b: %s)", model_a, model_b)
        self.predictors["a"] = model_a
        self.predictors["b"] = model_b
        logger.debug("Added predictors: %s", self.predictors)

    def convert_multi_gpu(self):
        """ Convert models to multi-gpu if requested """
        if self.gpus > 1:
            for side in self.predictors.keys():
                logger.debug("Converting to multi-gpu: side %s", side.capitalize())
                model = multi_gpu_model(self.predictors[side], self.gpus)
                self.predictors[side] = model
            logger.debug("Converted to multi-gpu: %s", self.predictors)

    def compile_predictors(self):
        """ Compile the predictors """
        logger.debug("Compiling Predictors")
        self.convert_multi_gpu()
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

        for side, model in self.predictors.items():
            loss_funcs = [self.loss_function(side)]
            if self.masks:
                mask = self.masks[0] if side == "a" else self.masks[1]
                loss_funcs.insert(0, self.mask_loss_function(mask, side))
            model.compile(optimizer=optimizer, loss=loss_funcs)

            if len(self.loss_names[side]) > 1:
                self.loss_names[side].insert(0, "total_loss")
            self.history[side] = list()
        logger.debug("Compiled Predictors. Losses: %s", self.loss_names)

    def loss_function(self, side):
        """ Set the loss function """
        if self.config.get("dssim_loss", False):
            if side == "a":
                logger.verbose("Using DSSIM Loss")
            loss_func = DSSIMObjective()
        else:
            if side == "a":
                logger.verbose("Using Mean Absolute Error Loss")
            loss_func = losses.mean_absolute_error
        self.loss_names[side] = ["loss"]
        logger.debug(loss_func)
        return loss_func

    def mask_loss_function(self, mask, side):
        """ Set the loss function for masks
            Side is input so we only log once """
        if self.config.get("dssim_mask_loss", False):
            if side == "a":
                logger.verbose("Using DSSIM Loss for mask")
            mask_loss_func = DSSIMObjective()
        else:
            if side == "a":
                logger.verbose("Using Mean Absolute Error Loss for mask")
            mask_loss_func = losses.mean_absolute_error

        if self.config.get("penalized_mask_loss", False):
            if side == "a":
                logger.verbose("Using Penalized Loss for mask")
            mask_loss_func = PenalizedLoss(mask, mask_loss_func)

        self.loss_names[side].insert(0, "mask_loss")
        logger.debug(mask_loss_func)
        return mask_loss_func

    def converter(self, swap):
        """ Converter for autoencoder models """
        logger.debug("Getting Converter: (swap: %s)", swap)
        if swap:
            retval = self.predictors["a"].predict
        else:
            retval = self.predictors["b"].predict
        logger.debug("Got Converter: %s", retval)
        return retval

    @property
    def iterations(self):
        "Get current training iteration number"
        return self.state.iterations

    @property
    def base_filename(self):
        """ Base filename for model and state files """
        resolution = self.input_shape[0]
        return "{}_{}_dim{}".format(self.name, resolution, self.encoder_dim)

    def map_weights(self, swapped):
        """ Map the weights for A/B side models for swapping """
        logger.debug("Map weights: (swapped: %s)", swapped)
        weights_map = {"A": dict(), "B": dict()}
        side_a, side_b = ("A", "B") if not swapped else ("B", "A")
        for network in self.networks.values():
            if network.side == side_a:
                weights_map["A"][network.type] = network.filename
            if network.side == side_b:
                weights_map["B"][network.type] = network.filename
        logger.debug("Mapped weights: (weights_map: %s)", weights_map)
        return weights_map

    def log_summary(self):
        """ Verbose log the model summaries """
        for name, nnmeta in self.networks.items():
            logger.verbose("%s Summary:", name.title())
            nnmeta.network.summary(print_fn=lambda x: logger.verbose("R|%s", x))

    def load_weights(self, swapped):
        """ Load weights from the weights file """
        logger.debug("Load weights: (swapped: %s)", swapped)
        weights_mapping = self.map_weights(swapped)
        try:
            for network in self.networks.values():
                if not network.side:
                    network.load_weights()
                else:
                    network.load_weights(
                        weights_mapping[network.side][network.type])
            logger.info("loaded model weights")
            return True
        except Exception as err:
            logger.warning("Failed loading existing training data. Generating new weights")
            logger.debug("Exception: %s", str(err))
            return False

    def save_weights(self):
        """ Backup and save the weights files """
        logger.debug("Backing up and saving weights")
        should_backup = self.get_save_averages()
        for network in self.networks.values():
            network.save_weights(should_backup=should_backup)
        self.state.save(should_backup)
        # Put in a line break to avoid jumbled console
        print("\n")
        logger.info("saved model weights")

    def get_save_averages(self):
        """ Return the loss averages since last save and reset historical losses

            This protects against model corruption by only backing up the model
            if any of the loss values have fallen.
            TODO This is not a perfect system. If the model corrupts on save_iteration - 1
            then model may still backup
        """
        logger.debug("Getting Average loss since last save")
        avgs = dict()
        backup = True

        for side in ("a", "b"):
            hist_loss = self.history[side]
            if not hist_loss:
                backup = False
                break

            avgs[side] = sum(hist_loss) / len(hist_loss)
            self.history[side] = list()

            avg_key = "avg_{}".format(side)
            if not self.history.get(avg_key, None):
                logger.debug("Setting initial save iteration loss average for '%s': %s",
                             avg_key, avgs[side])
                self.history[avg_key] = avgs[side]
                continue

            if backup:
                backup = self.check_loss_drop(avg_key, avgs[side])

        logger.debug("Lowest historical save iteration loss average: {avg_a: %s, avg_b: %s)",
                     self.history.get("avg_a", None), self.history.get("avg_b", None))
        logger.debug("Average loss since last save: %s", avgs)

        if backup:  # Update lowest loss values to the history
            for side in ("a", "b"):
                avg_key = "avg_{}".format(side)
                logger.debug("Updating lowest save iteration average for '%s': %s",
                             avg_key, avgs[side])
                self.history[avg_key] = avgs[side]

        logger.debug("Backing up: %s", backup)
        return backup

    def check_loss_drop(self, avg_key, avg):
        """ Check whether total loss has dropped since lowest loss """
        if avg < self.history[avg_key]:
            logger.debug("Loss for '%s' has dropped", avg_key)
            return True
        logger.debug("Loss for '%s' has not dropped", avg_key)
        return False


class NNMeta():
    """ Class to hold a neural network and it's meta data

    filename:   The full path and filename of the weights file for
                this network.
    type:       The type of network. For networks that can be swapped
                The type should be identical for the corresponding
                A and B networks, and should be unique for every A/B pair.
                Otherwise the type should be completely unique.
    side:       A, B or None. Used to identify which networks can
                be swapped.
    network:      Define network to this.
    """

    def __init__(self, filename, network_type, side, network):
        logger.debug("Initializing %s: (filename: '%s', network_type: '%s', side: '%s', "
                     "network: %s", self.__class__.__name__, filename, network_type,
                     side, network)
        self.filename = filename
        self.type = network_type
        self.side = side
        self.network = network
        logger.debug("Initialized %s", self.__class__.__name__)

    def load_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename
        logger.debug("Loading weights: '%s'", fullpath)
        self.network.load_weights(fullpath)

    def save_weights(self, fullpath=None, should_backup=False):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename
        if should_backup:
            self.backup_weights(fullpath=fullpath)
        logger.debug("Saving weights: '%s'", fullpath)
        self.network.save_weights(fullpath)

    def backup_weights(self, fullpath=None):
        """ Backup Model Weights """
        origfile = fullpath if fullpath else self.filename
        backupfile = origfile + ".bk"
        logger.debug("Backing up: '%s' to '%s'", origfile, backupfile)
        if os.path.exists(backupfile):
            os.remove(backupfile)
        if os.path.exists(origfile):
            os.rename(origfile, backupfile)


class State():
    """ Class to hold the model's current state """
    def __init__(self, model_dir, base_filename):
        self.serializer = Serializer.get_serializer("json")
        filename = "{}_state.{}".format(base_filename, self.serializer.ext)
        self.filename = str(model_dir / filename)
        self.iterations = self.load()

    def load(self):
        """ Load epoch number from state file """
        logger.debug("Loading State")
        iterations = 0
        try:
            with open(self.filename, "rb") as inp:
                state = self.serializer.unmarshal(inp.read().decode("utf-8"))
                # TODO Remove this backwards compatibility fix to get iterations from epoch_no
                iterations = state.get("epoch_no", None)
                iterations = state["iterations"] if iterations is None else iterations
        except IOError as err:
            logger.warning("No existing training info found. Generating.")
            logger.debug("IOError: %s", str(err))
            iterations = 0
        except JSONDecodeError as err:
            logger.debug("JSONDecodeError: %s:", str(err))
            iterations = 0
        logger.debug("Loaded State: (iterations: %s)", iterations)
        return iterations

    def save(self, should_backup=False):
        """ Save iteration number to state file """
        logger.debug("Saving State")
        if should_backup:
            self.backup()
        try:
            with open(self.filename, "wb") as out:
                state = {"iterations": self.iterations}
                state_json = self.serializer.marshal(state)
                out.write(state_json.encode("utf-8"))
        except IOError as err:
            logger.error("Unable to save model state: %s", str(err.strerror))
        logger.debug("Saved State")

    def backup(self):
        """ Backup state file """
        origfile = self.filename
        backupfile = origfile + ".bk"
        logger.debug("Backing up: '%s' to '%s'", origfile, backupfile)
        if os.path.exists(backupfile):
            os.remove(backupfile)
        if os.path.exists(origfile):
            os.rename(origfile, backupfile)
