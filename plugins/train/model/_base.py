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
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import get_custom_objects, multi_gpu_model

from lib import Serializer
from lib.model.losses import DSSIMObjective, PenalizedLoss
from lib.model.nn_blocks import NNBlocks
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = None


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self, model_dir, gpus, training_image_size=256,
                 input_shape=None, encoder_dim=None, trainer="original", predict=False):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', gpus: %s, "
                     "training_image_size, %s, input_shape: %s, encoder_dim: %s)",
                     self.__class__.__name__, model_dir, gpus, training_image_size, input_shape,
                     encoder_dim)
        self.predict = predict
        self.model_dir = model_dir
        self.gpus = gpus
        self.blocks = NNBlocks(use_subpixel=self.config["subpixel_upscaling"],
                               use_icnr_init=self.config["use_icnr_init"])
        self.input_shape = input_shape
        self.output_shape = None  # set after model is compiled
        self.encoder_dim = encoder_dim
        self.trainer = trainer
        self.name = self.set_model_name()

        self.state = State(self.model_dir, self.name, training_image_size)
        self.load_state_info()

        self.networks = dict()  # Networks for the model
        self.predictors = dict()  # Predictors for model
        self.loss_names = dict()  # Loss names for model
        self.history = dict()  # Loss history per save iteration)

        # Training information specific to the model should be placed in this
        # dict for reference by the trainer.
        self.training_opts = dict()

        self.build()
        self.set_training_data()
        logger.debug("Initialized ModelBase (%s)", self.__class__.__name__)

    @property
    def config(self):
        """ Return config dict for current plugin """
        global _CONFIG  # pylint: disable=global-statement
        if not _CONFIG:
            model_name = ".".join(self.__module__.split(".")[-2:])
            logger.debug("Loading config for: %s", model_name)
            _CONFIG = Config(model_name).config_dict
        return _CONFIG

    def set_training_data(self):
        """ Override to set model specific training data.

            super() this method for default coverage ratio
            otherwise be sure to add a ratio """
        logger.debug("Setting training data")
        self.training_opts["coverage_ratio"] = 0.625
        if self.output_shape[0] < 128:
            self.training_opts["preview_images"] = 14
        elif self.output_shape[0] < 192:
            self.training_opts["preview_images"] = 10
        elif self.output_shape[0] < 256:
            self.training_opts["preview_images"] = 8
        else:
            self.training_opts["preview_images"] = 6
        logger.debug("Set training data: %s", self.training_opts)

    def build(self):
        """ Build the model. Override for custom build methods """
        self.add_networks()
        self.load_models(swapped=False)
        self.build_autoencoders()
        self.log_summary()
        self.compile_predictors()

    def build_autoencoders(self):
        """ Override for Model Specific autoencoder builds

            NB! ENSURE YOU NAME YOUR INPUTS. At least the following input names
            are expected:
                face (the input for image)
                mask (the input for mask if it is used)

        """
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

    def load_state_info(self):
        """ Load the input shape from state file if it exists """
        logger.debug("Loading Input Shape from State file")
        if not self.state.inputs:
            logger.debug("No input shapes saved. Using model config")
            return
        if not self.state.face_shapes:
            logger.warning("Input shapes stored in State file, but no matches for 'face'."
                           "Using model config")
            return
        input_shape = self.state.face_shapes[0]
        logger.debug("Setting input shape from state file: %s", input_shape)
        self.input_shape = input_shape

    def add_network(self, network_type, side, network):
        """ Add a NNMeta object """
        logger.debug("network_type: '%s', side: '%s', network: '%s'", network_type, side, network)
        filename = "{}_{}".format(self.name, network_type.lower())
        name = network_type.lower()
        if side:
            side = side.lower()
            filename += "_{}".format(side.upper())
            name += "_{}".format(side)
        filename += ".h5"
        logger.debug("name: '%s', filename: '%s'", name, filename)
        self.networks[name] = NNMeta(str(self.model_dir / filename), network_type, side, network)

    def add_predictor(self, side, model):
        """ Add a predictor to the predictors dictionary """
        logger.debug("Adding predictor: (side: '%s', model: %s)", side, model)
        if self.gpus > 1:
            logger.debug("Converting to multi-gpu: side %s", side)
            model = multi_gpu_model(model, self.gpus)
        self.predictors[side] = model
        if not self.state.inputs:
            self.store_input_shapes(model)
        if not self.output_shape:
            self.set_output_shape(model)

    def store_input_shapes(self, model):
        """ Store the input and output shapes to state """
        logger.debug("Adding input shapes to state for model")
        inputs = {tensor.name: tensor.get_shape().as_list()[-3:] for tensor in model.inputs}
        if not any(inp for inp in inputs.keys() if inp.startswith("face")):
            raise ValueError("No input named 'face' was found. Check your input naming. "
                             "Current input names: {}".format(inputs))
        self.state.inputs = inputs
        logger.debug("Added input shapes: %s", self.state.inputs)

    def set_output_shape(self, model):
        """ Set the output shape for use in training and convert """
        logger.debug("Setting output shape")
        out = [tensor.get_shape().as_list()[-3:] for tensor in model.outputs]
        if not out:
            raise ValueError("No outputs found! Check your model.")
        self.output_shape = tuple(out[0])
        logger.debug("Added output shape: %s", self.output_shape)

    def compile_predictors(self):
        """ Compile the predictors """
        logger.debug("Compiling Predictors")
        optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999, clipnorm=1.0)

        for side, model in self.predictors.items():
            loss_funcs = [self.loss_function(side)]
            mask = [inp for inp in model.inputs if inp.name.startswith("mask")]
            if mask:
                loss_funcs.insert(0, self.mask_loss_function(mask[0], side))
            model.compile(optimizer=optimizer, loss=loss_funcs)

            if len(self.loss_names[side]) > 1:
                self.loss_names[side].insert(0, "total_loss")
            self.history[side] = list()
        logger.debug("Compiled Predictors. Losses: %s", self.loss_names)

    def loss_function(self, side):
        """ Set the loss function """
        if self.config.get("dssim_loss", False):
            if side == "a" and not self.predict:
                logger.verbose("Using DSSIM Loss")
            loss_func = DSSIMObjective()
        else:
            if side == "a" and not self.predict:
                logger.verbose("Using Mean Absolute Error Loss")
            loss_func = losses.mean_absolute_error
        self.loss_names[side] = ["loss"]
        logger.debug(loss_func)
        return loss_func

    def mask_loss_function(self, mask, side):
        """ Set the loss function for masks
            Side is input so we only log once """
        if self.config.get("dssim_mask_loss", False):
            if side == "a" and not self.predict:
                logger.verbose("Using DSSIM Loss for mask")
            mask_loss_func = DSSIMObjective()
        else:
            if side == "a" and not self.predict:
                logger.verbose("Using Mean Absolute Error Loss for mask")
            mask_loss_func = losses.mean_absolute_error

        if self.config.get("penalized_mask_loss", False):
            if side == "a" and not self.predict:
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

    def map_models(self, swapped):
        """ Map the models for A/B side for swapping """
        logger.debug("Map models: (swapped: %s)", swapped)
        models_map = {"a": dict(), "b": dict()}
        sides = ("a", "b") if not swapped else ("b", "a")
        for network in self.networks.values():
            if network.side == sides[0]:
                models_map["a"][network.type] = network.filename
            if network.side == sides[1]:
                models_map["b"][network.type] = network.filename
        logger.debug("Mapped models: (models_map: %s)", models_map)
        return models_map

    def log_summary(self):
        """ Verbose log the model summaries """
        if self.predict:
            return
        for side in sorted(list(self.predictors.keys())):
            logger.verbose("[%s %s Summary]:", self.name.title(), side.upper())
            self.predictors[side].summary(print_fn=lambda x: logger.verbose("R|%s", x))
            for name, nnmeta in self.networks.items():
                if nnmeta.side is not None and nnmeta.side != side:
                    continue
                logger.verbose("%s:", name.title())
                nnmeta.network.summary(print_fn=lambda x: logger.verbose("R|%s", x))

    def load_models(self, swapped):
        """ Load models from file """
        logger.debug("Load model: (swapped: %s)", swapped)
        model_mapping = self.map_models(swapped)
        for network in self.networks.values():
            if not network.side:
                is_loaded = network.load(predict=self.predict)
            else:
                is_loaded = network.load(fullpath=model_mapping[network.side][network.type],
                                         predict=self.predict)
            if not is_loaded:
                break
        if is_loaded:
            logger.info("loaded model")
        return is_loaded

    def save_models(self):
        """ Backup and save the models """
        logger.debug("Backing up and saving models")
        should_backup = self.get_save_averages()
        for network in self.networks.values():
            network.save(should_backup=should_backup)
        self.state.save(should_backup)
        # Put in a line break to avoid jumbled console
        print("\n")
        logger.info("saved models")

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

    filename:   The full path and filename of the model file for this network.
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
        self.type = network_type.lower()
        self.side = side
        self.network = network
        self.network.name = self.type
        logger.debug("Initialized %s", self.__class__.__name__)

    def load(self, fullpath=None, predict=False):
        """ Load model """

        fullpath = fullpath if fullpath else self.filename
        logger.debug("Loading model: '%s'", fullpath)
        try:
            network = load_model(self.filename, custom_objects=get_custom_objects())
        except ValueError as err:
            if str(err).lower().startswith("cannot create group in read only mode"):
                self.convert_legacy_weights()
                return True
            if predict:
                raise ValueError("Unable to load training data. Error: {}".format(str(err)))
            logger.warning("Failed loading existing training data. Generating new models")
            logger.debug("Exception: %s", str(err))
            return False

        except Exception as err:  # pylint: disable=broad-except
            if predict:
                raise ValueError("Unable to load training data. Error: {}".format(str(err)))
            logger.warning("Failed loading existing training data. Generating new models")
            logger.debug("Exception: %s", str(err))
            return False
        self.network = network  # Update network with saved model
        self.network.name = self.type
        return True

    def save(self, fullpath=None, should_backup=False):
        """ Save model """
        fullpath = fullpath if fullpath else self.filename
        if should_backup:
            self.backup(fullpath=fullpath)
        logger.debug("Saving model: '%s'", fullpath)
        self.network.save(fullpath)

    def backup(self, fullpath=None):
        """ Backup Model """
        origfile = fullpath if fullpath else self.filename
        backupfile = origfile + ".bk"
        logger.debug("Backing up: '%s' to '%s'", origfile, backupfile)
        if os.path.exists(backupfile):
            os.remove(backupfile)
        if os.path.exists(origfile):
            os.rename(origfile, backupfile)

    def convert_legacy_weights(self):
        """ Convert legacy weights files to hold the model topology """
        logger.info("Adding model topology to legacy weights file: '%s'", self.filename)
        self.network.load_weights(self.filename)
        self.save(should_backup=False)
        self.network.name = self.type


class State():
    """ Class to hold the model's current state and autoencoder structure """
    def __init__(self, model_dir, model_name, training_image_size):
        self.serializer = Serializer.get_serializer("json")
        filename = "{}_state.{}".format(model_name, self.serializer.ext)
        self.filename = str(model_dir / filename)
        self.training_size = training_image_size
        self.iterations = 0
        self.inputs = dict()
        self.config = dict()
        self.load()

    @property
    def face_shapes(self):
        """ Return a list of stored face shape inputs """
        return [tuple(val) for key, val in self.inputs.items() if key.startswith("face")]

    @property
    def mask_shapes(self):
        """ Return a list of stored mask shape inputs """
        return [tuple(val) for key, val in self.inputs.items() if key.startswith("mask")]

    def load(self):
        """ Load epoch number from state file """
        logger.debug("Loading State")
        try:
            with open(self.filename, "rb") as inp:
                state = self.serializer.unmarshal(inp.read().decode("utf-8"))
                self.iterations = state.get("iterations", 0)
                self.inputs = state.get("inputs", dict())
                self.config = state.get("config", dict())
                self.state = state.get("training_size", 256)
                logger.debug("Loaded state: %s", {key: val for key, val in state.items()
                                                  if key != "models"})
        except IOError as err:
            logger.warning("No existing state file found. Generating.")
            logger.debug("IOError: %s", str(err))
        except JSONDecodeError as err:
            logger.debug("JSONDecodeError: %s:", str(err))

    def save(self, should_backup=False):
        """ Save iteration number to state file """
        logger.debug("Saving State")
        if should_backup:
            self.backup()
        try:
            with open(self.filename, "wb") as out:
                state = {"iterations": self.iterations,
                         "inputs": self.inputs,
                         "training_size": self.training_size,
                         "config": _CONFIG}
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
