#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of NNMeta objects.
    See the class for details.
"""
import logging
import os
import sys
import time

from concurrent import futures

import keras
from keras import losses
from keras import backend as K
from keras.layers import Input
from keras.models import load_model, Model
from keras.utils import get_custom_objects, multi_gpu_model

from lib.serializer import get_serializer
from lib.model.backup_restore import Backup
from lib.model.losses import (DSSIMObjective, PenalizedLoss, gradient_loss, mask_loss_wrapper,
                              generalized_loss, l_inf_norm, gmsd_loss, gaussian_blur)
from lib.model.nn_blocks import NNBlocks
from lib.model.optimizers import Adam
from lib.utils import deprecation_warning, FaceswapError
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = None


class ModelBase():
    """ Base class that all models should inherit from """
    def __init__(self,
                 model_dir,
                 gpus=1,
                 configfile=None,
                 snapshot_interval=0,
                 no_logs=False,
                 warp_to_landmarks=False,
                 augment_color=True,
                 no_flip=False,
                 training_image_size=256,
                 alignments_paths=None,
                 preview_scale=100,
                 input_shape=None,
                 encoder_dim=None,
                 trainer="original",
                 pingpong=False,
                 memory_saving_gradients=False,
                 optimizer_savings=False,
                 predict=False):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', gpus: %s, configfile: %s, "
                     "snapshot_interval: %s, no_logs: %s, warp_to_landmarks: %s, augment_color: "
                     "%s, no_flip: %s, training_image_size, %s, alignments_paths: %s, "
                     "preview_scale: %s, input_shape: %s, encoder_dim: %s, trainer: %s, "
                     "pingpong: %s, memory_saving_gradients: %s, optimizer_savings: %s, "
                     "predict: %s)",
                     self.__class__.__name__, model_dir, gpus, configfile, snapshot_interval,
                     no_logs, warp_to_landmarks, augment_color, no_flip, training_image_size,
                     alignments_paths, preview_scale, input_shape, encoder_dim, trainer, pingpong,
                     memory_saving_gradients, optimizer_savings, predict)

        self.predict = predict
        self.model_dir = model_dir
        self.vram_savings = VRAMSavings(pingpong, optimizer_savings, memory_saving_gradients)

        self.backup = Backup(self.model_dir, self.name)
        self.gpus = gpus
        self.configfile = configfile
        self.input_shape = input_shape
        self.encoder_dim = encoder_dim
        self.trainer = trainer

        self.load_config()  # Load config if plugin has not already referenced it

        self.state = State(self.model_dir,
                           self.name,
                           self.config_changeable_items,
                           no_logs,
                           self.vram_savings.pingpong,
                           training_image_size)

        self.blocks = NNBlocks(use_icnr_init=self.config["icnr_init"],
                               use_convaware_init=self.config["conv_aware_init"],
                               use_reflect_padding=self.config["reflect_padding"],
                               first_run=self.state.first_run)

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
                              "augment_color": augment_color,
                              "no_flip": no_flip,
                              "pingpong": self.vram_savings.pingpong,
                              "snapshot_interval": snapshot_interval,
                              "training_size": self.state.training_size,
                              "no_logs": self.state.current_session["no_logs"],
                              "coverage_ratio": self.calculate_coverage_ratio(),
                              "mask_type": self.config["mask_type"],
                              "mask_blur_kernel": self.config["mask_blur_kernel"],
                              "mask_threshold": self.config["mask_threshold"],
                              "learn_mask": (self.config["learn_mask"] and
                                             self.config["mask_type"] is not None),
                              "penalized_mask_loss": (self.config["penalized_mask_loss"] and
                                                      self.config["mask_type"] is not None)}
        logger.debug("training_opts: %s", self.training_opts)

        if self.multiple_models_in_folder:
            deprecation_warning("Support for multiple model types within the same folder",
                                additional_info="Please split each model into separate folders to "
                                                "avoid issues in future.")

        self.build()
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
            _CONFIG = Config(model_name, configfile=self.configfile).config_dict
        return _CONFIG

    @property
    def config_changeable_items(self):
        """ Return the dict of config items that can be updated after the model
            has been created """
        return Config(self.config_section, configfile=self.configfile).changeable_items

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

    @property
    def multiple_models_in_folder(self):
        """ Return true if there are multiple model types in the same folder, else false """
        model_files = [fname for fname in os.listdir(str(self.model_dir)) if fname.endswith(".h5")]
        retval = False if not model_files else os.path.commonprefix(model_files) == ""
        logger.debug("model_files: %s, retval: %s", model_files, retval)
        return retval

    @property
    def output_shapes(self):
        """ Return the output shapes from the main AutoEncoder """
        out = list()
        for predictor in self.predictors.values():
            out.extend([K.int_shape(output)[-3:] for output in predictor.outputs])
            break  # Only get output from one autoencoder. Shapes are the same
        return [tuple(shape) for shape in out]

    @property
    def output_shape(self):
        """ The output shape of the model (shape of largest face output) """
        return self.output_shapes[self.largest_face_index]

    @property
    def largest_face_index(self):
        """ Return the index from model.outputs of the largest face
            Required for multi-output model prediction. The largest face
            is assumed to be the final output
        """
        sizes = [shape[1] for shape in self.output_shapes if shape[2] == 3]
        if not sizes:
            return None
        max_face = max(sizes)
        retval = [idx for idx, shape in enumerate(self.output_shapes)
                  if shape[1] == max_face and shape[2] == 3][0]
        logger.debug(retval)
        return retval

    @property
    def largest_mask_index(self):
        """ Return the index from model.outputs of the largest mask
            Required for multi-output model prediction. The largest face
            is assumed to be the final output
        """
        sizes = [shape[1] for shape in self.output_shapes if shape[2] == 1]
        if not sizes:
            return None
        max_mask = max(sizes)
        retval = [idx for idx, shape in enumerate(self.output_shapes)
                  if shape[1] == max_mask and shape[2] == 1][0]
        logger.debug(retval)
        return retval

    @property
    def feed_mask(self):
        """ bool: ``True`` if the model expects a mask to be fed into input otherwise ``False`` """
        return self.config["mask_type"] is not None and (self.config["learn_mask"] or
                                                         self.config["penalized_mask_loss"])

    def load_config(self):
        """ Load the global config for reference in self.config """
        global _CONFIG  # pylint: disable=global-statement
        if not _CONFIG:
            model_name = self.config_section
            logger.debug("Loading config for: %s", model_name)
            _CONFIG = Config(model_name, configfile=self.configfile).config_dict

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
        inputs = self.get_inputs()
        try:
            self.build_autoencoders(inputs)
        except ValueError as err:
            if "must be from the same graph" in str(err).lower():
                msg = ("There was an error loading saved weights. This is most likely due to "
                       "model corruption during a previous save."
                       "\nYou should restore weights from a snapshot or from backup files. "
                       "You can use the 'Restore' Tool to restore from backup.")
                raise FaceswapError(msg) from err
            if "multi_gpu_model" in str(err).lower():
                raise FaceswapError(str(err)) from err
            raise err
        self.log_summary()
        self.compile_predictors(initialize=True)

    def get_inputs(self):
        """ Return the inputs for the model """
        logger.debug("Getting inputs")
        inputs = [Input(shape=self.input_shape, name="face_in")]
        output_network = [network for network in self.networks.values() if network.is_output][0]
        if self.feed_mask:
            # TODO penalized mask doesn't have a mask output, so we can't use output shapes
            # mask should always be last output..this needs to be a rule
            mask_shape = output_network.output_shapes[-1]
            inputs.append(Input(shape=(mask_shape[1:-1] + (1,)), name="mask_in"))
        logger.debug("Got inputs: %s", inputs)
        return inputs

    def build_autoencoders(self, inputs):
        """ Override for Model Specific autoencoder builds

            Inputs is defined in self.get_inputs() and is standardized for all models
                if will generally be in the order:
                [face (the input for image),
                 mask (the input for mask if it is used)]
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

    def add_network(self, network_type, side, network, is_output=False):
        """ Add a NNMeta object """
        logger.debug("network_type: '%s', side: '%s', network: '%s', is_output: %s",
                     network_type, side, network, is_output)
        filename = "{}_{}".format(self.name, network_type.lower())
        name = network_type.lower()
        if side:
            side = side.lower()
            filename += "_{}".format(side.upper())
            name += "_{}".format(side)
        filename += ".h5"
        logger.debug("name: '%s', filename: '%s'", name, filename)
        self.networks[name] = NNMeta(str(self.model_dir / filename),
                                     network_type,
                                     side,
                                     network,
                                     is_output)

    def add_predictor(self, side, model):
        """ Add a predictor to the predictors dictionary """
        logger.debug("Adding predictor: (side: '%s', model: %s)", side, model)
        if self.gpus > 1:
            logger.debug("Converting to multi-gpu: side %s", side)
            model = multi_gpu_model(model, self.gpus)
        self.predictors[side] = model
        if not self.state.inputs:
            self.store_input_shapes(model)

    def store_input_shapes(self, model):
        """ Store the input and output shapes to state """
        logger.debug("Adding input shapes to state for model")
        inputs = {tensor.name: K.int_shape(tensor)[-3:] for tensor in model.inputs}
        if not any(inp for inp in inputs.keys() if inp.startswith("face")):
            raise ValueError("No input named 'face' was found. Check your input naming. "
                             "Current input names: {}".format(inputs))
        # Make sure they are all ints so that it can be json serialized
        inputs = {key: tuple(int(i) for i in val) for key, val in inputs.items()}
        self.state.inputs = inputs
        logger.debug("Added input shapes: %s", self.state.inputs)

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

        inputs = self.get_inputs()
        self.build_autoencoders(inputs)
        self.compile_predictors(initialize=False)
        logger.debug("Reset models")

    def compile_predictors(self, initialize=True):
        """ Compile the predictors """
        logger.debug("Compiling Predictors")
        learning_rate = self.config.get("learning_rate", 5e-5)
        optimizer = self.get_optimizer(lr=learning_rate, beta_1=0.5, beta_2=0.999)

        for side, model in self.predictors.items():
            loss = Loss(model.inputs, model.outputs)
            model.compile(optimizer=optimizer, loss=loss.funcs)
            if initialize:
                self.state.add_session_loss_names(side, loss.names)
                self.history[side] = list()
        logger.debug("Compiled Predictors. Losses: %s", loss.names)

    def get_optimizer(self, lr=5e-5, beta_1=0.5, beta_2=0.999):  # pylint: disable=invalid-name
        """ Build and return Optimizer """
        opt_kwargs = dict(lr=lr, beta_1=beta_1, beta_2=beta_2)
        if (self.config.get("clipnorm", False) and
                keras.backend.backend() != "plaidml.keras.backend"):
            # NB: Clip-norm is ballooning VRAM usage, which is not expected behavior
            # and may be a bug in Keras/Tensorflow.
            # PlaidML has a bug regarding the clip-norm parameter
            # See: https://github.com/plaidml/plaidml/issues/228
            # Workaround by simply removing it.
            # TODO: Remove this as soon it is fixed in PlaidML.
            opt_kwargs["clipnorm"] = 1.0
        logger.debug("Optimizer kwargs: %s", opt_kwargs)
        return Adam(**opt_kwargs, cpu_mode=self.vram_savings.optimizer_savings)

    def converter(self, swap):
        """ Converter for autoencoder models """
        logger.debug("Getting Converter: (swap: %s)", swap)
        side = "a" if swap else "b"
        model = self.predictors[side]
        if self.predict:
            # Must compile the model to be thread safe
            model._make_predict_function()  # pylint: disable=protected-access
        retval = model.predict
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
            self.predictors[side].summary(print_fn=lambda x: logger.verbose("%s", x))
            for name, nnmeta in self.networks.items():
                if nnmeta.side is not None and nnmeta.side != side:
                    continue
                logger.verbose("%s:", name.title())
                nnmeta.network.summary(print_fn=lambda x: logger.verbose("%s", x))

    def do_snapshot(self):
        """ Perform a model snapshot """
        logger.debug("Performing snapshot")
        self.backup.snapshot_models(self.iterations)
        logger.debug("Performed snapshot")

    def load_models(self, swapped):
        """ Load models from file """
        logger.debug("Load model: (swapped: %s)", swapped)

        if not self.models_exist and not self.predict:
            logger.info("Creating new '%s' model in folder: '%s'", self.name, self.model_dir)
            return None
        if not self.models_exist and self.predict:
            logger.error("Model could not be found in folder '%s'. Exiting", self.model_dir)
            exit(0)

        if not self.is_legacy or not self.predict:
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
        # Insert a new line to avoid spamming the same row as loss output
        print("")
        save_averages = self.get_save_averages()
        backup_func = self.backup.backup_model if self.should_backup(save_averages) else None
        if backup_func:
            logger.info("Backing up models...")
        executor = futures.ThreadPoolExecutor()
        save_threads = [executor.submit(network.save, backup_func=backup_func)
                        for network in self.networks.values()]
        save_threads.append(executor.submit(self.state.save, backup_func=backup_func))
        futures.wait(save_threads)
        # call result() to capture errors
        _ = [thread.result() for thread in save_threads]
        msg = "[Saved models]"
        if save_averages:
            lossmsg = ["{}_{}: {:.5f}".format(self.state.loss_names[side][0],
                                              side.capitalize(),
                                              save_averages[side])
                       for side in sorted(list(save_averages.keys()))]
            msg += " - Average since last save: {}".format(", ".join(lossmsg))
        logger.info(msg)

    def get_save_averages(self):
        """ Return the average loss since the last save iteration and reset historical loss """
        logger.debug("Getting save averages")
        avgs = dict()
        for side, loss in self.history.items():
            if not loss:
                logger.debug("No loss in self.history: %s", side)
                break
            avgs[side] = sum(loss) / len(loss)
            self.history[side] = list()  # Reset historical loss
        logger.debug("Average losses since last save: %s", avgs)
        return avgs

    def should_backup(self, save_averages):
        """ Check whether the loss averages for all losses is the lowest that has been seen.

            This protects against model corruption by only backing up the model
            if any of the loss values have fallen.
            TODO This is not a perfect system. If the model corrupts on save_iteration - 1
            then model may still backup
        """
        backup = True

        if not save_averages:
            logger.debug("No save averages. Not backing up")
            return False

        for side, loss in save_averages.items():
            if not self.state.lowest_avg_loss.get(side, None):
                logger.debug("Setting initial save iteration loss average for '%s': %s",
                             side, loss)
                self.state.lowest_avg_loss[side] = loss
                continue
            if backup:
                # Only run this if backup is true. All losses must have dropped for a valid backup
                backup = self.check_loss_drop(side, loss)

        logger.debug("Lowest historical save iteration loss average: %s",
                     self.state.lowest_avg_loss)

        if backup:  # Update lowest loss values to the state
            for side, avg_loss in save_averages.items():
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
        self.state.config["reflect_padding"] = False
        self.state.config["mask_type"] = None
        self.state.config["mask_blur_kernel"] = 3
        self.state.config["mask_threshold"] = 4
        self.state.config["learn_mask"] = False
        self.state.config["lowmem"] = False
        self.encoder_dim = 1024

        if set_lowmem:
            logger.debug("Setting encoder_dim and lowmem flag for legacy lowmem model")
            self.encoder_dim = 512
            self.state.config["lowmem"] = True

        self.state.replace_config(self.config_changeable_items)
        self.state.save()


class VRAMSavings():
    """ VRAM Saving training methods """
    def __init__(self, pingpong, optimizer_savings, memory_saving_gradients):
        logger.debug("Initializing %s: (pingpong: %s, optimizer_savings: %s, "
                     "memory_saving_gradients: %s)", self.__class__.__name__,
                     pingpong, optimizer_savings, memory_saving_gradients)
        self.is_plaidml = keras.backend.backend() == "plaidml.keras.backend"
        self.pingpong = self.set_pingpong(pingpong)
        self.optimizer_savings = self.set_optimizer_savings(optimizer_savings)
        self.memory_saving_gradients = self.set_gradient_type(memory_saving_gradients)
        logger.debug("Initialized: %s", self.__class__.__name__)

    def set_pingpong(self, pingpong):
        """ Disable pingpong for plaidML users """
        if pingpong and self.is_plaidml:
            logger.warning("Pingpong training not supported on plaidML. Disabling")
            pingpong = False
        logger.debug("pingpong: %s", pingpong)
        if pingpong:
            logger.info("Using Pingpong Training")
        return pingpong

    def set_optimizer_savings(self, optimizer_savings):
        """ Disable optimizer savings for plaidML users """
        if optimizer_savings and self.is_plaidml == "plaidml.keras.backend":
            logger.warning("Optimizer Savings not supported on plaidML. Disabling")
            optimizer_savings = False
        logger.debug("optimizer_savings: %s", optimizer_savings)
        if optimizer_savings:
            logger.info("Using Optimizer Savings")
        return optimizer_savings

    def set_gradient_type(self, memory_saving_gradients):
        """ Monkey-patch Memory Saving Gradients if requested """
        if memory_saving_gradients and self.is_plaidml:
            logger.warning("Memory Saving Gradients not supported on plaidML. Disabling")
            memory_saving_gradients = False
        logger.debug("memory_saving_gradients: %s", memory_saving_gradients)
        if memory_saving_gradients:
            logger.info("Using Memory Saving Gradients")
            from lib.model import memory_saving_gradients
            K.__dict__["gradients"] = memory_saving_gradients.gradients_memory
        return memory_saving_gradients


class Loss():
    """ Holds loss names and functions for an Autoencoder """
    def __init__(self, inputs, outputs):
        logger.debug("Initializing %s: (inputs: %s, outputs: %s)",
                     self.__class__.__name__, inputs, outputs)
        self.inputs = inputs
        self.outputs = outputs
        self.names = self.get_loss_names()
        self.funcs = self.get_loss_functions()
        if len(self.names) > 1:
            self.names.insert(0, "total_loss")
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def loss_dict(self):
        """ Return the loss dict """
        loss_dict = dict(mae=losses.mean_absolute_error,
                         mse=losses.mean_squared_error,
                         logcosh=losses.logcosh,
                         smooth_loss=generalized_loss,
                         l_inf_norm=l_inf_norm,
                         ssim=DSSIMObjective(),
                         gmsd=gmsd_loss,
                         pixel_gradient_diff=gradient_loss)
        return loss_dict

    @property
    def config(self):
        """ Return the global _CONFIG variable """
        return _CONFIG

    @property
    def mask_preprocessing_func(self):
        """ The selected pre-processing function for the mask """
        retval = None
        if self.config.get("mask_blur", False):
            retval = gaussian_blur(max(1, self.mask_shape[1] // 32))
        logger.debug(retval)
        return retval

    @property
    def selected_loss(self):
        """ Return the selected loss function """
        retval = self.loss_dict[self.config.get("loss_function", "mae")]
        logger.debug(retval)
        return retval

    @property
    def selected_mask_loss(self):
        """ Return the selected mask loss function. Currently returns mse
            If a processing function has been requested wrap the loss function
            in loss wrapper """
        loss_func = self.loss_dict["mse"]
        func = self.mask_preprocessing_func
        logger.debug("loss_func: %s, func: %s", loss_func, func)
        retval = mask_loss_wrapper(loss_func, preprocessing_func=func)
        return retval

    @property
    def output_shapes(self):
        """ The shapes of the output nodes """
        return [K.int_shape(output)[1:] for output in self.outputs]

    @property
    def mask_input(self):
        """ Return the mask input or None """
        mask_inputs = [inp for inp in self.inputs if inp.name.startswith("mask")]
        if not mask_inputs:
            return None
        return mask_inputs[0]

    @property
    def mask_shape(self):
        """ Return the mask shape """
        if self.mask_input is None:
            return None
        return K.int_shape(self.mask_input)[1:]

    def get_loss_names(self):
        """ Return the loss names based on model output """
        output_names = [output.name for output in self.outputs]
        logger.debug("Model output names: %s", output_names)
        loss_names = [name[name.find("/") + 1:name.rfind("/")].replace("_out", "")
                      for name in output_names]
        if not all(name.startswith("face") or name.startswith("mask") for name in loss_names):
            # Handle incorrectly named/legacy outputs
            logger.debug("Renaming loss names from: %s", loss_names)
            loss_names = self.update_loss_names()
        loss_names = ["{}_loss".format(name) for name in loss_names]
        logger.debug(loss_names)
        return loss_names

    def update_loss_names(self):
        """ Update loss names if named incorrectly or legacy model """
        output_types = ["mask" if shape[-1] == 1 else "face" for shape in self.output_shapes]
        loss_names = ["{}{}".format(name,
                                    "" if output_types.count(name) == 1 else "_{}".format(idx))
                      for idx, name in enumerate(output_types)]
        logger.debug("Renamed loss names to: %s", loss_names)
        return loss_names

    def get_loss_functions(self):
        """ Set the loss function """
        loss_funcs = []
        for idx, loss_name in enumerate(self.names):
            if loss_name.startswith("mask"):
                loss_funcs.append(self.selected_mask_loss)
            elif self.config["penalized_mask_loss"] and self.config["mask_type"] is not None:
                face_size = self.output_shapes[idx][1]
                mask_size = self.mask_shape[1]
                scaling = face_size / mask_size
                logger.debug("face_size: %s mask_size: %s, mask_scaling: %s",
                             face_size, mask_size, scaling)
                loss_funcs.append(PenalizedLoss(self.mask_input, self.selected_loss,
                                                mask_scaling=scaling,
                                                preprocessing_func=self.mask_preprocessing_func))
            else:
                loss_funcs.append(self.selected_loss)
            logger.debug("%s: %s", loss_name, loss_funcs[-1])
        logger.debug(loss_funcs)
        return loss_funcs


class NNMeta():
    """ Class to hold a neural network and it's meta data

    filename:   The full path and filename of the model file for this network.
    type:       The type of network. For networks that can be swapped
                The type should be identical for the corresponding
                A and B networks, and should be unique for every A/B pair.
                Otherwise the type should be completely unique.
    side:       A, B or None. Used to identify which networks can
                be swapped.
    network:    Define network to this.
    is_output:  Set to True to indicate that this network is an output to the Autoencoder
    """

    def __init__(self, filename, network_type, side, network, is_output):
        logger.debug("Initializing %s: (filename: '%s', network_type: '%s', side: '%s', "
                     "network: %s, is_output: %s", self.__class__.__name__, filename,
                     network_type, side, network, is_output)
        self.filename = filename
        self.type = network_type.lower()
        self.side = side
        self.name = self.set_name()
        self.network = network
        self.is_output = is_output
        self.network.name = self.name
        self.config = network.get_config()  # For pingpong restore
        self.weights = network.get_weights()  # For pingpong restore
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def output_shapes(self):
        """ Return the output shapes from the stored network """
        return [K.int_shape(output) for output in self.network.outputs]

    def set_name(self):
        """ Set the network name """
        name = self.type
        if self.side:
            name += "_{}".format(self.side)
        return name

    @property
    def output_names(self):
        """ Return output node names """
        output_names = [output.name for output in self.network.outputs]
        if self.is_output and not any(name.startswith("face_out") for name in output_names):
            # Saved models break if their layer names are changed, so dummy
            # in correct output names for legacy models
            output_names = self.get_output_names()
        return output_names

    def get_output_names(self):
        """ Return the output names based on number of channels and instances """
        output_types = ["mask_out" if K.int_shape(output)[-1] == 1 else "face_out"
                        for output in self.network.outputs]
        output_names = ["{}{}".format(name,
                                      "" if output_types.count(name) == 1 else "_{}".format(idx))
                        for idx, name in enumerate(output_types)]
        logger.debug("Overridden output_names: %s", output_names)
        return output_names

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
        self.network.name = self.name
        return True

    def save(self, fullpath=None, backup_func=None):
        """ Save model """
        fullpath = fullpath if fullpath else self.filename
        if backup_func:
            backup_func(fullpath)
        logger.debug("Saving model: '%s'", fullpath)
        self.weights = self.network.get_weights()
        self.network.save(fullpath)

    def convert_legacy_weights(self):
        """ Convert legacy weights files to hold the model topology """
        logger.info("Adding model topology to legacy weights file: '%s'", self.filename)
        self.network.load_weights(self.filename)
        self.save(backup_func=None)
        self.network.name = self.type


class State():
    """ Class to hold the model's current state and autoencoder structure """
    def __init__(self, model_dir, model_name, config_changeable_items,
                 no_logs, pingpong, training_image_size):
        logger.debug("Initializing %s: (model_dir: '%s', model_name: '%s', "
                     "config_changeable_items: '%s', no_logs: %s, pingpong: %s, "
                     "training_image_size: '%s'", self.__class__.__name__, model_dir, model_name,
                     config_changeable_items, no_logs, pingpong, training_image_size)
        self.serializer = get_serializer("json")
        filename = "{}_state.{}".format(model_name, self.serializer.file_extension)
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
        self.create_new_session(no_logs, pingpong, config_changeable_items)
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

    @property
    def first_run(self):
        """ Return True if this is the first run else False """
        return self.session_id == 1

    def new_session_id(self):
        """ Return new session_id """
        if not self.sessions:
            session_id = 1
        else:
            session_id = max(int(key) for key in self.sessions.keys()) + 1
        logger.debug(session_id)
        return session_id

    def create_new_session(self, no_logs, pingpong, config_changeable_items):
        """ Create a new session """
        logger.debug("Creating new session. id: %s", self.session_id)
        self.sessions[self.session_id] = {"timestamp": time.time(),
                                          "no_logs": no_logs,
                                          "pingpong": pingpong,
                                          "loss_names": dict(),
                                          "batchsize": 0,
                                          "iterations": 0,
                                          "config": config_changeable_items}

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
        if not os.path.exists(self.filename):
            logger.info("No existing state file found. Generating.")
            return
        state = self.serializer.load(self.filename)
        self.name = state.get("name", self.name)
        self.sessions = state.get("sessions", dict())
        self.lowest_avg_loss = state.get("lowest_avg_loss", dict())
        self.iterations = state.get("iterations", 0)
        self.training_size = state.get("training_size", 256)
        self.inputs = state.get("inputs", dict())
        self.config = state.get("config", dict())
        logger.debug("Loaded state: %s", state)
        self.replace_config(config_changeable_items)

    def save(self, backup_func=None):
        """ Save iteration number to state file """
        logger.debug("Saving State")
        if backup_func:
            backup_func(self.filename)
        state = {"name": self.name,
                 "sessions": self.sessions,
                 "lowest_avg_loss": self.lowest_avg_loss,
                 "iterations": self.iterations,
                 "inputs": self.inputs,
                 "training_size": self.training_size,
                 "config": _CONFIG}
        self.serializer.save(self.filename, state)
        logger.debug("Saved State")

    def replace_config(self, config_changeable_items):
        """ Replace the loaded config with the one contained within the state file
            Check for any fixed=False parameters changes and log info changes
        """
        global _CONFIG  # pylint: disable=global-statement
        legacy_update = self._update_legacy_config()
        # Add any new items to state config for legacy purposes
        for key, val in _CONFIG.items():
            if key not in self.config.keys():
                logger.info("Adding new config item to state file: '%s': '%s'", key, val)
                self.config[key] = val
        self.update_changed_config_items(config_changeable_items)
        logger.debug("Replacing config. Old config: %s", _CONFIG)
        _CONFIG = self.config
        if legacy_update:
            self.save()
        logger.debug("Replaced config. New config: %s", _CONFIG)
        logger.info("Using configuration saved in state file")

    def _update_legacy_config(self):
        """ Legacy updates for new config additions.

        When new config items are added to the Faceswap code, existing model state files need to be
        updated to handle these new items.

        Current existing legacy update items:

            * loss - If old `dssim_loss` is ``true`` set new `loss_function` to `ssim` otherwise
            set it to `mae`. Remove old `dssim_loss` item

            * masks - If `learn_mask` does not exist then it is set to ``True`` if `mask_type` is
            not ``None`` otherwise it is set to ``False``.

            * masks type - Replace removed masks 'dfl_full' and 'facehull' with `components` mask

        Returns
        -------
        bool
            ``True`` if legacy items exist and state file has been updated, otherwise ``False``
        """
        logger.debug("Checking for legacy state file update")
        priors = ["dssim_loss", "mask_type", "mask_type"]
        new_items = ["loss_function", "learn_mask", "mask_type"]
        updated = False
        for old, new in zip(priors, new_items):
            if old not in self.config:
                logger.debug("Legacy item '%s' not in config. Skipping update", old)
                continue

            # dssim_loss > loss_function
            if old == "dssim_loss":
                self.config[new] = "ssim" if self.config[old] else "mae"
                del self.config[old]
                updated = True
                logger.info("Updated config from legacy dssim format. New config loss "
                            "function: '%s'", self.config[new])
                continue

            # Add learn mask option and set to True if model has "penalized_mask_loss" specified
            if old == "mask_type" and new == "learn_mask" and new not in self.config:
                self.config[new] = self.config["mask_type"] is not None
                updated = True
                logger.info("Added new 'learn_mask' config item for this model. Value set to: %s",
                            self.config[new])
                continue

            # Replace removed masks with most similar equivalent
            if old == "mask_type" and new == "mask_type" and self.config[old] in ("facehull",
                                                                                  "dfl_full"):
                old_mask = self.config[old]
                self.config[new] = "components"
                updated = True
                logger.info("Updated 'mask_type' from '%s' to '%s' for this model",
                            old_mask, self.config[new])

        logger.debug("State file updated for legacy config: %s", updated)
        return updated

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
