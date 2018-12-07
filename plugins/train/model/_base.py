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
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self, model_dir, gpus, image_shape=None, encoder_dim=None, trainer="original"):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', gpus: %s, image_shape: %s, "
                     "encoder_dim: %s)", self.__class__.__name__, model_dir, gpus,
                     image_shape, encoder_dim)
        self.config = Config().config
        self.model_dir = model_dir
        self.gpus = gpus
        self.image_shape = image_shape
        self.encoder_dim = encoder_dim
        self.trainer = trainer

        # Training information specific to the model should be placed in this
        # dict for reference by the trainer.
        self.training_opts = dict()

        # For autoencoder models, autoencoders should be placed in this dict
        self.autoencoders = dict()

        self.name = self.set_model_name()
        self.serializer = Serializer.get_serializer("json")
        self._epoch_no = self.load_state()
        self.networks = list()

        self.add_networks()
        self.initialize()
        logger.debug("Initialized ModelBase (%s)", self.__class__.__name__)

    def initialize(self):
        """ Override for Model Specific Initialization """
        raise NotImplementedError

    def add_networks(self):
        """ Override to add neural networks """
        raise NotImplementedError

    def add_network(self, network_type, side, network):
        """ Add a NNMeta object to self.models """
        logger.debug("network_type: '%s', side: '%s', network: '%s'", network_type, side, network)
        resolution = self.image_shape[0]
        filename = "{}_{}_{}".format(self.name, resolution, network_type.lower())
        if side:
            filename += "_{}".format(side.upper())
        filename += ".h5"
        logger.debug("filename: '%s'", filename)
        self.networks.append(NNMeta(str(self.model_dir / filename),
                                    network_type,
                                    side,
                                    network))

    def set_model_name(self):
        """ Set the model name based on the subclass """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        retval = os.path.splitext(basename)[0].lower()
        logger.debug("model name: '%s'", retval)
        return retval

    def compile_autoencoders(self):
        """ Compile the autoencoders """
        logger.debug("Compiling Autoencoders")
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)
        if self.gpus > 1:
            for acr in self.autoencoders.keys():
                autoencoder = multi_gpu_model(self.autoencoders[acr],
                                              self.gpus)
                self.autoencoders[acr] = autoencoder

        for autoencoder in self.autoencoders.values():
            autoencoder.compile(optimizer=optimizer,
                                loss="mean_absolute_error")
        logger.debug("Compiled Autoencoders: %s", self.autoencoders)

    def converter(self, swap):
        """ Converter for autoencoder models """
        logger.debug("Getting Converter: (swap: %s)", swap)
        if swap:
            retval = self.autoencoders["a"].predict
        else:
            retval = self.autoencoders["b"].predict
        logger.debug("Got Converter: %s", retval)
        return retval

    @property
    def epoch_no(self):
        "Get current training epoch number"
        return self._epoch_no

    def load_state(self):
        """ Load epoch number from state file """
        logger.debug("Loading State")
        epoch_no = 0
        try:
            with open(self.state_filename(), "rb") as inp:
                state = self.serializer.unmarshal(inp.read().decode("utf-8"))
                epoch_no = state["epoch_no"]
        except IOError as err:
            logger.warning("No existing training info found. Generating.")
            logger.debug("IOError: %s", str(err))
            epoch_no = 0
        except JSONDecodeError as err:
            logger.debug("JSONDecodeError: %s:", str(err))
            epoch_no = 0
        logger.debug("Loaded State: (epoch_no: %s)", epoch_no)
        return epoch_no

    def save_state(self):
        """ Save epoch number to state file """
        logger.debug("Saving State")
        try:
            with open(self.state_filename(), "wb") as out:
                state = {"epoch_no": self.epoch_no}
                state_json = self.serializer.marshal(state)
                out.write(state_json.encode("utf-8"))
        except IOError as err:
            logger.error("Unable to save model state: %s", str(err.strerror))
        logger.debug("Saved State")

    def state_filename(self):
        """ Return full filepath for this models state file """
        filename = "{}_state.{}".format(self.name, self.serializer.ext)
        retval = str(self.model_dir / filename)
        logger.debug(retval)
        return retval

    def map_weights(self, swapped):
        """ Map the weights for A/B side models for swapping """
        logger.debug("Map weights: (swapped: %s)", swapped)
        weights_map = {"A": dict(), "B": dict()}
        side_a, side_b = ("A", "B") if not swapped else ("B", "A")
        for network in self.networks:
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
            for network in self.networks:
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
        """ Save the weights files """
        logger.debug("Saving weights")
        self.backup_weights()
        for network in self.networks:
            network.save_weights()
        # Put in a line break to avoid jumbled console
        print("\n")
        logger.info("saved model weights")
        self.save_state()

    def backup_weights(self):
        """ Backup the weights files by appending .bk to the end """
        logger.debug("Backing up weights")
        for network in self.networks:
            origfile = network.filename
            backupfile = origfile + ".bk"
            logger.debug("Backing up: '%s' to '%s'", origfile, backupfile)
            if os.path.exists(backupfile):
                os.remove(backupfile)
            if os.path.exists(origfile):
                os.rename(origfile, backupfile)
        logger.debug("Backed up weights")

    @staticmethod
    def log_summary(name, model):
        """ Verbose log the passed in model summary """
        logger.debug("%s Summary:", name.title())
        model.summary(print_fn=logger.verbose)


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
