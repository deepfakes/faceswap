#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of NNMeta objects.
    See the class for details.
"""
import logging
import os
import sys
import time

from json import JSONDecodeError

import keras
from keras import losses
from keras import backend as K
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.utils import get_custom_objects, multi_gpu_model

from lib import Serializer
from lib.model.losses import DSSIMObjective, PenalizedLoss
from lib.model.nn_blocks import NNBlocks
from lib.multithreading import MultiThread
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = None


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self,
                 model_dir,
                 gpus,
                 no_logs=False,
                 warp_to_landmarks=False,
                 no_flip=False,
                 training_image_size=256,
                 alignments_paths=None,
                 preview_scale=100,
                 input_shape=None,
                 encoder_dim=None,
                 trainer="original",
                 pingpong=False,
                 memory_saving_gradients=False,
                 predict=False):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', gpus: %s, no_logs: %s"
                     "training_image_size, %s, alignments_paths: %s, preview_scale: %s, "
                     "input_shape: %s, encoder_dim: %s, trainer: %s, pingpong: %s, "
                     "memory_saving_gradients: %s, predict: %s)",
                     self.__class__.__name__, model_dir, gpus, no_logs, training_image_size,
                     alignments_paths, preview_scale, input_shape, encoder_dim, trainer,
                     pingpong, memory_saving_gradients, predict)

        self.predict = predict
        self.model_dir = model_dir
        self.gpus = gpus
        self.blocks = NNBlocks(use_subpixel=self.config["subpixel_upscaling"],
                               use_icnr_init=self.config["icnr_init"],
                               use_reflect_padding=self.config["reflect_padding"])
        self.input_shape = input_shape
        self.output_shape = None  # set after model is compiled
        self.encoder_dim = encoder_dim
        self.trainer = trainer

        self.state = State(self.model_dir,
                           self.name,
                           self.config_changeable_items,
                           no_logs,
                           pingpong,
                           training_image_size)
        self.is_legacy = False
        self.rename_legacy()
        self.load_state_info()

        self.networks = dict()  # Networks for the model
        self.predictors = dict()  # Predictors for model
        self.history = dict()  # Loss history per save iteration)

        # Training information specific to the model should be placed in this
        # dict for reference by the trainer.
        self.training_opts = {"alignments": alignments_paths,
                              "preview_scaling": preview_scale / 100,
                              "warp_to_landmarks": warp_to_landmarks,
                              "no_flip": no_flip,
                              "pingpong": pingpong}

        self.set_gradient_type(memory_saving_gradients)
        self.build()
        self.set_training_data()
        logger.debug("Initialized ModelBase (%s)", self.__class__.__name__)

    @property
    def config_section(self):
        """ The section name for loading config """
        retval = ".".join(self.__module__.split(".")[-2:])
        logger.debug(retval)
        return retval

    @property
    def config(self):
        """ Return config dict for current plugin """
        global _CONFIG  # pylint: disable=global-statement
        if not _CONFIG:
            model_name = self.config_section
            logger.debug("Loading config for: %s", model_name)
            _CONFIG = Config(model_name).config_dict
        return _CONFIG

    @property
    def config_changeable_items(self):
        """ Return the dict of config items that can be updated after the model
            has been created """
        return Config(self.config_section).changeable_items

    @property
    def name(self):
        """ Set the model name based on the subclass """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        retval = os.path.splitext(basename)[0].lower()
        logger.debug("model name: '%s'", retval)
        return retval

    @property
    def models_exist(self):
        """ Return if all files exist and clear session """
        retval = all([os.path.isfile(model.filename) for model in self.networks.values()])
        logger.debug("Pre-existing models exist: %s", retval)
        return retval

    @staticmethod
    def set_gradient_type(memory_saving_gradients):
        """ Monkeypatch Memory Saving Gradients if requested """
        if not memory_saving_gradients:
            return
        logger.info("Using Memory Saving Gradients")
        from lib.model import memory_saving_gradients
        K.__dict__["gradients"] = memory_saving_gradients.gradients_memory

    def set_training_data(self):
        """ Override to set model specific training data.

            super() this method for defaults otherwise be sure to add """
        logger.debug("Setting training data")
        self.training_opts["training_size"] = self.state.training_size
        self.training_opts["no_logs"] = self.state.current_session["no_logs"]
        self.training_opts["mask_type"] = self.config.get("mask_type", None)
        self.training_opts["coverage_ratio"] = self.calculate_coverage_ratio()
        self.training_opts["preview_images"] = 14
        logger.debug("Set training data: %s", self.training_opts)

    def calculate_coverage_ratio(self):
        """ Coverage must be a ratio, leading to a cropped shape divisible by 2 """
        coverage_ratio = self.config.get("coverage", 62.5) / 100
        logger.debug("Requested coverage_ratio: %s", coverage_ratio)
        cropped_size = (self.state.training_size * coverage_ratio) // 2 * 2
        coverage_ratio = cropped_size / self.state.training_size
        logger.debug("Final coverage_ratio: %s", coverage_ratio)
        return coverage_ratio

    def build(self):
        """ Build the model. Override for custom build methods """
        self.add_networks()
        self.load_models(swapped=False)
        self.build_autoencoders()
        self.log_summary()
        self.compile_predictors(initialize=True)

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
        inputs = {tensor.name: K.int_shape(tensor)[-3:] for tensor in model.inputs}
        if not any(inp for inp in inputs.keys() if inp.startswith("face")):
            raise ValueError("No input named 'face' was found. Check your input naming. "
                             "Current input names: {}".format(inputs))
        self.state.inputs = inputs
        logger.debug("Added input shapes: %s", self.state.inputs)

    def set_output_shape(self, model):
        """ Set the output shape for use in training and convert """
        logger.debug("Setting output shape")
        out = [K.int_shape(tensor)[-3:] for tensor in model.outputs]
        if not out:
            raise ValueError("No outputs found! Check your model.")
        self.output_shape = tuple(out[0])
        logger.debug("Added output shape: %s", self.output_shape)

    def reset_pingpong(self):
        """ Reset the models for pingpong training """
        logger.debug("Resetting models")

        # Clear models and graph
        self.predictors = dict()
        K.clear_session()

        # Load Models for current training run
        for model in self.networks.values():
            model.network = Model.from_config(model.config)
            model.network.set_weights(model.weights)

        self.build_autoencoders()
        self.compile_predictors(initialize=False)
        logger.debug("Reset models")

    def compile_predictors(self, initialize=True):
        """ Compile the predictors """
        logger.debug("Compiling Predictors")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for side, model in self.predictors.items():
            mask = [inp for inp in model.inputs if inp.name.startswith("mask")]
            loss_names = ["loss"]
            loss_funcs = [self.loss_function(mask, side, initialize)]
            if mask:
                loss_names.append("mask_loss")
                loss_funcs.append(self.mask_loss_function(side, initialize))
            model.compile(optimizer=optimizer, loss=loss_funcs)

            if len(loss_names) > 1:
                loss_names.insert(0, "total_loss")
            if initialize:
                self.state.add_session_loss_names(side, loss_names)
                self.history[side] = list()
        logger.debug("Compiled Predictors. Losses: %s", loss_names)

    def get_optimizer(self, lr=5e-5, beta_1=0.5, beta_2=0.999):  # pylint: disable=invalid-name
        """ Build and return Optimizer """
        opt_kwargs = dict(lr=lr, beta_1=beta_1, beta_2=beta_2)
        if (self.config.get("clipnorm", False) and
                keras.backend.backend() != "plaidml.keras.backend"):
            # NB: Clipnorm is ballooning VRAM useage, which is not expected behaviour
            # and may be a bug in Keras/TF.
            # PlaidML has a bug regarding the clipnorm parameter
            # See: https://github.com/plaidml/plaidml/issues/228
            # Workaround by simply removing it.
            # TODO: Remove this as soon it is fixed in PlaidML.
            opt_kwargs["clipnorm"] = 1.0
        logger.debug("Optimizer kwargs: %s", opt_kwargs)
        return Adam(**opt_kwargs)

    def loss_function(self, mask, side, initialize):
        """ Set the loss function
            Side is input so we only log once """
        if self.config.get("dssim_loss", False):
            if side == "a" and not self.predict and initialize:
                logger.verbose("Using DSSIM Loss")
            loss_func = DSSIMObjective()
        else:
            if side == "a" and not self.predict and initialize:
                logger.verbose("Using Mean Absolute Error Loss")
            loss_func = losses.mean_absolute_error

        if mask and self.config.get("penalized_mask_loss", False):
            loss_mask = mask[0]
            if side == "a" and not self.predict and initialize:
                logger.verbose("Penalizing mask for Loss")
            loss_func = PenalizedLoss(loss_mask, loss_func)
        return loss_func

    def mask_loss_function(self, side, initialize):
        """ Set the mask loss function
            Side is input so we only log once """
        if side == "a" and not self.predict and initialize:
            logger.verbose("Using Mean Squared Error Loss for mask")
        mask_loss_func = losses.mean_squared_error
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

        if not self.models_exist and not self.predict:
            logger.info("Creating new '%s' model in folder: '%s'", self.name, self.model_dir)
            return None
        if not self.models_exist and self.predict:
            logger.error("Model could not be found in folder '%s'. Exiting", self.model_dir)
            exit(0)

        if not self.is_legacy:
            K.clear_session()
        model_mapping = self.map_models(swapped)
        for network in self.networks.values():
            if not network.side:
                is_loaded = network.load()
            else:
                is_loaded = network.load(fullpath=model_mapping[network.side][network.type])
            if not is_loaded:
                break
        if is_loaded:
            logger.info("Loaded model from disk: '%s'", self.model_dir)
        return is_loaded

    def save_models(self):
        """ Backup and save the models """
        logger.debug("Backing up and saving models")
        should_backup = self.get_save_averages()
        save_threads = list()
        for network in self.networks.values():
            name = "save_{}".format(network.name)
            save_threads.append(MultiThread(network.save, name=name, should_backup=should_backup))
        save_threads.append(MultiThread(self.state.save,
                                        name="save_state", should_backup=should_backup))
        for thread in save_threads:
            thread.start()
        for thread in save_threads:
            if thread.has_error:
                logger.error(thread.errors[0])
            thread.join()
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

        for side, loss in self.history.items():
            if not loss:
                backup = False
                break

            avgs[side] = sum(loss) / len(loss)
            self.history[side] = list()  # Reset historical loss

            if not self.state.lowest_avg_loss.get(side, None):
                logger.debug("Setting initial save iteration loss average for '%s': %s",
                             side, avgs[side])
                self.state.lowest_avg_loss[side] = avgs[side]
                continue

            if backup:
                # Only run this if backup is true. All losses must have dropped for a valid backup
                backup = self.check_loss_drop(side, avgs[side])

        logger.debug("Lowest historical save iteration loss average: %s",
                     self.state.lowest_avg_loss)
        logger.debug("Average loss since last save: %s", avgs)

        if backup:  # Update lowest loss values to the state
            for side, avg_loss in avgs.items():
                logger.debug("Updating lowest save iteration average for '%s': %s", side, avg_loss)
                self.state.lowest_avg_loss[side] = avg_loss

        logger.debug("Backing up: %s", backup)
        return backup

    def check_loss_drop(self, side, avg):
        """ Check whether total loss has dropped since lowest loss """
        if avg < self.state.lowest_avg_loss[side]:
            logger.debug("Loss for '%s' has dropped", side)
            return True
        logger.debug("Loss for '%s' has not dropped", side)
        return False

    def rename_legacy(self):
        """ Legacy Original, LowMem and IAE models had inconsistent naming conventions
            Rename them if they are found and update """
        legacy_mapping = {"iae": [("IAE_decoder.h5", "iae_decoder.h5"),
                                  ("IAE_encoder.h5", "iae_encoder.h5"),
                                  ("IAE_inter_A.h5", "iae_intermediate_A.h5"),
                                  ("IAE_inter_B.h5", "iae_intermediate_B.h5"),
                                  ("IAE_inter_both.h5", "iae_inter.h5")],
                          "original": [("encoder.h5", "original_encoder.h5"),
                                       ("decoder_A.h5", "original_decoder_A.h5"),
                                       ("decoder_B.h5", "original_decoder_B.h5"),
                                       ("lowmem_encoder.h5", "original_encoder.h5"),
                                       ("lowmem_decoder_A.h5", "original_decoder_A.h5"),
                                       ("lowmem_decoder_B.h5", "original_decoder_B.h5")]}
        if self.name not in legacy_mapping.keys():
            return
        logger.debug("Renaming legacy files")

        set_lowmem = False
        updated = False
        for old_name, new_name in legacy_mapping[self.name]:
            old_path = os.path.join(str(self.model_dir), old_name)
            new_path = os.path.join(str(self.model_dir), new_name)
            if os.path.exists(old_path) and not os.path.exists(new_path):
                logger.info("Updating legacy model name from: '%s' to '%s'", old_name, new_name)
                os.rename(old_path, new_path)
                if old_name.startswith("lowmem"):
                    set_lowmem = True
                updated = True

        if not updated:
            logger.debug("No legacy files to rename")
            return

        self.is_legacy = True
        logger.debug("Creating state file for legacy model")
        self.state.inputs = {"face:0": [64, 64, 3]}
        self.state.training_size = 256
        self.state.config["coverage"] = 62.5
        self.state.config["subpixel_upscaling"] = False
        self.state.config["reflect_padding"] = False
        self.state.config["mask_type"] = None
        self.state.config["lowmem"] = False
        self.encoder_dim = 1024

        if set_lowmem:
            logger.debug("Setting encoder_dim and lowmem flag for legacy lowmem model")
            self.encoder_dim = 512
            self.state.config["lowmem"] = True

        self.state.replace_config(self.config_changeable_items)
        self.state.save()


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
        self.name = self.set_name()
        self.network = network
        self.network.name = self.name
        self.config = network.get_config()  # For pingpong restore
        self.weights = network.get_weights()  # For pingpong restore
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_name(self):
        """ Set the network name """
        name = self.type
        if self.side:
            name += "_{}".format(self.side)
        return name

    def load(self, fullpath=None):
        """ Load model """
        fullpath = fullpath if fullpath else self.filename
        logger.debug("Loading model: '%s'", fullpath)
        try:
            network = load_model(self.filename, custom_objects=get_custom_objects())
        except ValueError as err:
            if str(err).lower().startswith("cannot create group in read only mode"):
                self.convert_legacy_weights()
                return True
            logger.warning("Failed loading existing training data. Generating new models")
            logger.debug("Exception: %s", str(err))
            return False
        except OSError as err:  # pylint: disable=broad-except
            logger.warning("Failed loading existing training data. Generating new models")
            logger.debug("Exception: %s", str(err))
            return False
        self.config = network.get_config()
        self.network = network  # Update network with saved model
        self.network.name = self.type
        return True

    def save(self, fullpath=None, should_backup=False):
        """ Save model """
        fullpath = fullpath if fullpath else self.filename
        if should_backup:
            self.backup(fullpath=fullpath)
        logger.debug("Saving model: '%s'", fullpath)
        self.weights = self.network.get_weights()
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
    def __init__(self, model_dir, model_name, config_changeable_items,
                 no_logs, pingpong, training_image_size):
        logger.debug("Initializing %s: (model_dir: '%s', model_name: '%s', "
                     "config_changeable_items: '%s', no_logs: %s, pingpong: %s, "
                     "training_image_size: '%s'", self.__class__.__name__, model_dir, model_name,
                     config_changeable_items, no_logs, pingpong, training_image_size)
        self.serializer = Serializer.get_serializer("json")
        filename = "{}_state.{}".format(model_name, self.serializer.ext)
        self.filename = str(model_dir / filename)
        self.name = model_name
        self.iterations = 0
        self.session_iterations = 0
        self.training_size = training_image_size
        self.sessions = dict()
        self.lowest_avg_loss = dict()
        self.inputs = dict()
        self.config = dict()
        self.load(config_changeable_items)
        self.session_id = self.new_session_id()
        self.create_new_session(no_logs, pingpong)
        logger.debug("Initialized %s:", self.__class__.__name__)

    @property
    def face_shapes(self):
        """ Return a list of stored face shape inputs """
        return [tuple(val) for key, val in self.inputs.items() if key.startswith("face")]

    @property
    def mask_shapes(self):
        """ Return a list of stored mask shape inputs """
        return [tuple(val) for key, val in self.inputs.items() if key.startswith("mask")]

    @property
    def loss_names(self):
        """ Return the loss names for this session """
        return self.sessions[self.session_id]["loss_names"]

    @property
    def current_session(self):
        """ Return the current session dict """
        return self.sessions[self.session_id]

    def new_session_id(self):
        """ Return new session_id """
        if not self.sessions:
            session_id = 1
        else:
            session_id = max(int(key) for key in self.sessions.keys()) + 1
        logger.debug(session_id)
        return session_id

    def create_new_session(self, no_logs, pingpong):
        """ Create a new session """
        logger.debug("Creating new session. id: %s", self.session_id)
        self.sessions[self.session_id] = {"timestamp": time.time(),
                                          "no_logs": no_logs,
                                          "pingpong": pingpong,
                                          "loss_names": dict(),
                                          "batchsize": 0,
                                          "iterations": 0}

    def add_session_loss_names(self, side, loss_names):
        """ Add the session loss names to the sessions dictionary """
        logger.debug("Adding session loss_names. (side: '%s', loss_names: %s", side, loss_names)
        self.sessions[self.session_id]["loss_names"][side] = loss_names

    def add_session_batchsize(self, batchsize):
        """ Add the session batchsize to the sessions dictionary """
        logger.debug("Adding session batchsize: %s", batchsize)
        self.sessions[self.session_id]["batchsize"] = batchsize

    def increment_iterations(self):
        """ Increment total and session iterations """
        self.iterations += 1
        self.sessions[self.session_id]["iterations"] += 1

    def load(self, config_changeable_items):
        """ Load state file """
        logger.debug("Loading State")
        try:
            with open(self.filename, "rb") as inp:
                state = self.serializer.unmarshal(inp.read().decode("utf-8"))
                self.name = state.get("name", self.name)
                self.sessions = state.get("sessions", dict())
                self.lowest_avg_loss = state.get("lowest_avg_loss", dict())
                self.iterations = state.get("iterations", 0)
                self.training_size = state.get("training_size", 256)
                self.inputs = state.get("inputs", dict())
                self.config = state.get("config", dict())
                logger.debug("Loaded state: %s", state)
                self.replace_config(config_changeable_items)
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
                state = {"name": self.name,
                         "sessions": self.sessions,
                         "lowest_avg_loss": self.lowest_avg_loss,
                         "iterations": self.iterations,
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

    def replace_config(self, config_changeable_items):
        """ Replace the loaded config with the one contained within the state file
            Check for any fixed=False parameters changes and log info changes
        """
        global _CONFIG  # pylint: disable=global-statement
        # Add any new items to state config for legacy purposes
        for key, val in _CONFIG.items():
            if key not in self.config.keys():
                logger.info("Adding new config item to state file: '%s': '%s'", key, val)
                self.config[key] = val
        self.update_changed_config_items(config_changeable_items)
        logger.debug("Replacing config. Old config: %s", _CONFIG)
        _CONFIG = self.config
        logger.debug("Replaced config. New config: %s", _CONFIG)
        logger.info("Using configuration saved in state file")

    def update_changed_config_items(self, config_changeable_items):
        """ Update any parameters which are not fixed and have been changed """
        if not config_changeable_items:
            logger.debug("No changeable parameters have been updated")
            return
        for key, val in config_changeable_items.items():
            old_val = self.config[key]
            if old_val == val:
                continue
            self.config[key] = val
            logger.info("Config item: '%s' has been updated from '%s' to '%s'", key, old_val, val)
