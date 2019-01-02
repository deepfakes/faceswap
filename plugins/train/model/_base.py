#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of NNMeta objects.
    See the class for details.
"""
import logging
import os
import sys

from json import JSONDecodeError
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
    def __init__(self, model_dir, gpus, image_shape=None, encoder_dim=None, trainer="original"):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', gpus: %s, image_shape: %s, "
                     "encoder_dim: %s)", self.__class__.__name__, model_dir, gpus,
                     image_shape, encoder_dim)
        self.config = get_config(self.__module__.split(".")[-1])
        self.model_dir = model_dir
        self.gpus = gpus
        self.image_shape = image_shape
        self.encoder_dim = encoder_dim
        self.trainer = trainer

        self.masks = None  # List of masks to be set if masks are used

        self.name = self.set_model_name()
        self.networks = dict()  # Networks for the model
        self.predictors = dict()  # Predictors for model
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
                logger.debug("Converting to multi-gpu: '%s_%s'",
                             self.predictors[side].network_type, side)
                model = multi_gpu_model(self.predictors[side], self.gpus)
                self.predictors[side] = model
            logger.debug("Converted to multi-gpu: %s", self.predictors)

    def compile_predictors(self):
        """ Compile the predictors """
        logger.debug("Compiling Predictors")
        self.convert_multi_gpu()
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        loss_func = self.loss_function()
        for side, model in self.predictors.items():
            if self.masks:
                mask = self.masks[0] if side == "a" else self.masks[1]
                model.compile(optimizer=optimizer,
                              loss=[PenalizedLoss(mask, loss_func), "mse"])
            else:
                model.compile(optimizer=optimizer, loss=loss_func)
        logger.debug("Compiled Predictors")

    def loss_function(self):
        """ Set the loss function """
        if self.config["dssim_loss"]:
            loss_func = DSSIMObjective()
        else:
            loss_func = "mean_absolute_error"
        logger.debug(loss_func)
        return loss_func

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
        resolution = self.image_shape[0]
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
        for network in self.networks.values():
            network.backup_weights()
            network.save_weights()
        # Put in a line break to avoid jumbled console
        print("\n")
        logger.info("saved model weights")
        self.state.save()

    def log_summary(self):
        """ Verbose log the model summaries """
        for name, nnmeta in self.networks.items():
            logger.verbose("%s Summary:", name.title())
            nnmeta.network.summary(print_fn=lambda x: logger.verbose("R|%s", x))


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

    def save_weights(self, fullpath=None):
        """ Load model weights """
        fullpath = fullpath if fullpath else self.filename
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

    def save(self):
        """ Save epoch number to state file """
        logger.debug("Saving State")
        try:
            with open(self.filename, "wb") as out:
                state = {"iterations": self.iterations}
                state_json = self.serializer.marshal(state)
                out.write(state_json.encode("utf-8"))
        except IOError as err:
            logger.error("Unable to save model state: %s", str(err.strerror))
        logger.debug("Saved State")
