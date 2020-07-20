#!/usr/bin/env python3
""" Base class for Models. ALL Models should at least inherit from this class

    When inheriting model_data should be a list of NNMeta objects.
    See the class for details.
"""
import inspect
import logging
import os
import platform
import sys
import time

from contextlib import nullcontext

import tensorflow as tf

from keras import losses
from keras import backend as K
from keras.layers import Input, Layer
from keras.models import load_model, Model as KerasModel
from keras.optimizers import Adam
from keras.utils import get_custom_objects

from lib.serializer import get_serializer
from lib.model.backup_restore import Backup
from lib.model.losses import (DSSIMObjective, PenalizedLoss, gradient_loss, mask_loss_wrapper,
                              generalized_loss, l_inf_norm, gmsd_loss, gaussian_blur)
from lib.model.nn_blocks import set_config as set_nnblock_config
from lib.utils import deprecation_warning, get_backend, FaceswapError
from plugins.train._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_CONFIG = None

# TODO Legacy is removed. Still check for legacy and give instructions for updating by using TF1.15
# TODO Mask Input
# TODO Load input shape from state file if variable input shapes for saved model, or (preferably)
# read this straight from the model file


class ModelBase():
    """ Base class that all models should inherit from.

    Parameters
    ----------
    model_dir: str
        The full path to the model save location
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the train or convert process as generated from
        Faceswap's command line arguments
    training_image_size: int, optional
        The size of the training images in the training folder. Default: 256
    predict: bool, optional
        ``True`` if the model is being loaded for inference, ``False`` if the model is being loaded
        for training. Default: ``False``
    """
    def __init__(self, model_dir, arguments, training_image_size=256, predict=False):
        logger.debug("Initializing ModelBase (%s): (model_dir: '%s', arguments: %s, "
                     "training_image_size: %s, predict: %s)",
                     self.__class__.__name__, model_dir, arguments, training_image_size, predict)

        self.input_shape = None  # Must be set within the plugin after initializing
        self.output_shape = None  # Must be set within the plugin after initializing
        self.trainer = "original"  # Override for plugin specific trainer

        self._args = arguments
        self._io = IO(self, model_dir, predict)

        self._is_predict = predict
        self._configfile = arguments.configfile if hasattr(arguments, "configfile") else None

        self._model = None

        self._load_config()  # Load config if plugin has not already referenced it
        self._strategy = Strategy(self._args.distribution)

        self.state = State(model_dir,
                           self.name,
                           self._config_changeable_items,
                           self._args.no_logs,
                           training_image_size)

        # The variables holding masks if Penalized Loss is used
        self._mask_variables = dict(a=None, b=None)
        self.predictors = dict()  # Predictors for model
        self.history = [[], []]  # Loss histories per save iteration

        # Training information specific to the model should be placed in this
        # dict for reference by the trainer.
        self.training_opts = {"training_size": self.state.training_size,
                              "coverage_ratio": self._calculate_coverage_ratio(),
                              "mask_type": self.config["mask_type"],
                              "mask_blur_kernel": self.config["mask_blur_kernel"],
                              "mask_threshold": self.config["mask_threshold"],
                              "learn_mask": (self.config["learn_mask"] and
                                             self.config["mask_type"] is not None),
                              "penalized_mask_loss": self.config["penalized_mask_loss"]}
        logger.debug("training_opts: %s", self.training_opts)

        if self._io.multiple_models_in_folder:
            deprecation_warning("Support for multiple model types within the same folder",
                                additional_info="Please split each model into separate folders to "
                                                "avoid issues in future.")
        logger.debug("Initialized ModelBase (%s)", self.__class__.__name__)

    @property
    def model(self):
        """:class:`Keras.models.Model: The compiled model for this plugin. """
        return self._model

    @property
    def command_line_arguments(self):
        """ :class:`argparse.Namespace`: The command line arguments passed to the model plugin from
        either the train or convert script """
        return self._args

    @property
    def model_dir(self):
        """str: The full path to the model folder location. """
        return self._io._model_dir  # pylint:disable=protected-access

    @property
    def config(self):
        """ dict: The configuration dictionary for current plugin. """
        global _CONFIG  # pylint: disable=global-statement
        if not _CONFIG:
            model_name = self._config_section
            logger.debug("Loading config for: %s", model_name)
            _CONFIG = Config(model_name, configfile=self._configfile).config_dict
        return _CONFIG

    @property
    def name(self):
        """ str: The name of this model based on the plugin name. """
        basename = os.path.basename(sys.modules[self.__module__].__file__)
        return os.path.splitext(basename)[0].lower()

    @property
    def output_shapes(self):
        """ list: A list of shape tuples for the output of the model """
        # TODO Currently we're pulling all of the outputs and I'm just extracting from the first
        # side. This is not right, Need to fix this to properly output, especially when masks are
        # involved
        retval = [tuple(K.int_shape(output)[-3:]) for output in self._model.outputs]
        retval = [retval[0]]
        return retval

    # TODO
#    @property
#    def output_shape(self):
#        """ The output shape of the model (shape of largest face output) """
#        return self.output_shapes[self.largest_face_index]

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
        return self.config["mask_type"] is not None and self.config["learn_mask"]

    @property
    def mask_variables(self):
        """ dict: for each side a :class:`keras.backend.variable` or ``None``.
        If Penalized Mask Loss is used then each side will return a Variable of
        (`batch size`, `h`, `w`, 1) corresponding to the size of the model input.
        If Penalized Mask Loss is not used then each side will return ``None``

        Raises
        ------
        FaceswapError:
            If Penalized Mask Loss has been selected, but a mask type has not been specified
        """
        if not self.config["penalized_mask_loss"] or all(val is not None
                                                         for val in self._mask_variables.values()):
            return self._mask_variables

        if self.config["penalized_mask_loss"] and self.config["mask_type"] is None:
            raise FaceswapError("Penalized Mask Loss has been selected but you have not chosen a "
                                "Mask to use. Please select a mask or disable Penalized Mask "
                                "Loss.")

        output_network = [network for network in self.networks.values() if network.is_output][0]
        mask_shape = output_network.output_shapes[-1][:-1] + (1, )
        for side in ("a", "b"):
            var = K.variable(K.ones((self._args.batch_size, ) + mask_shape[1:], dtype="float32"),
                             dtype="float32",
                             name="penalized_mask_variable_{}".format(side))
            if get_backend() != "amd":
                # trainable and shape don't have a setter, so we need to go to private property
                var._trainable = False  # pylint:disable=protected-access
                var._shape = tf.TensorShape(mask_shape)  # pylint:disable=protected-access
            self._mask_variables[side] = var
        logger.debug("Created mask variables: %s", self._mask_variables)
        return self._mask_variables

    @property
    def iterations(self):
        """ int: The total number of iterations that the model has trained. """
        return self.state.iterations

    # Private properties
    @property
    def _config_section(self):
        """ str: The section name for the current plugin for loading configuration options from the
        config file. """
        return ".".join(self.__module__.split(".")[-2:])

    @property
    def _config_changeable_items(self):
        """ dict: The configuration options that can be updated after the model has already been
            created. """
        return Config(self._config_section, configfile=self._configfile).changeable_items

    def _load_config(self):
        """ Load the global config for reference in :attr:`config` and set the faceswap blocks
        configuration options in `lib.model.nn_blocks` """
        global _CONFIG  # pylint: disable=global-statement
        if not _CONFIG:
            model_name = self._config_section
            logger.debug("Loading config for: %s", model_name)
            _CONFIG = Config(model_name, configfile=self._configfile).config_dict

        nn_block_keys = ['icnr_init', 'conv_aware_init', 'reflect_padding']
        set_nnblock_config({key: _CONFIG.pop(key)
                            for key in nn_block_keys})

    def _calculate_coverage_ratio(self):
        """ Coverage must be a ratio, leading to a cropped shape divisible by 2 """
        coverage_ratio = self.config.get("coverage", 62.5) / 100
        logger.debug("Requested coverage_ratio: %s", coverage_ratio)
        cropped_size = (self.state.training_size * coverage_ratio) // 2 * 2
        coverage_ratio = cropped_size / self.state.training_size
        logger.debug("Final coverage_ratio: %s", coverage_ratio)
        return coverage_ratio

    def build(self):
        """ Build the model.

        Within the defined strategy scope, either builds the model from scratch or loads an
        existing model if one exists. The model is then compiled with the optimizer and chosen
        loss function(s), Finally, a model summary is outputted to the logger at verbose level.

        The compiled model is allocated to :attr:`_model`.
        """
        with self._strategy.scope():
            if self._io.model_exists:
                self._model = self._io._load()  # pylint:disable=protected-access
            else:
                inputs = self._get_inputs()
                self._model = self.build_model(inputs)
            self._compile_model()
        if not self._is_predict:
            self._output_summary()

    def _get_inputs(self):
        """ Obtain the standardized inputs for the model.

        The inputs will be returned for the "A" and "B" sides in the shape as defined by
        :attr:`input_shape`.

        Returns
        -------
        list
            A list of :class:`keras.layers.Input` tensors. If just a face is being fed in then this
            will be a list of 2 tensors (one for each side) each of shape :attr:`input_shape`. If a
            mask is to be passed into the model, then the list will contain 2 sub-lists (one for
            each side), with each sub-list containing the input tenor for the face of
            :attr:`input_shape` and the input tensor for the mask, with the same shape as the face,
            but with just a single channel.
        """
        logger.debug("Getting inputs")
        if self.feed_mask:
            mask_shape = self.input_shape[:2] + (1, )
            logger.info("mask_shape: %s", mask_shape)
        inputs = []
        for side in ("a", "b"):
            face_in = Input(shape=self.input_shape, name="face_in_{}".format(side))
            if self.feed_mask:
                mask_in = Input(shape=mask_shape, name="mask_in_{}".format(side))
                inputs.append([face_in, mask_in])
            else:
                inputs.append(face_in)
        logger.debug("inputs: %s", inputs)
        return inputs

    def build_model(self, inputs):
        """ Override for Model Specific autoencoder builds.

        Parameters
        ----------
        inputs: list
            The inputs to the model as returned from the :func:`_get_inputs`. This will be a list
            of :class:`keras.layers.Input` tensors. If just a face is being fed in then this will
            be a list of 2 tensors (one for each side) each of shape :attr:`input_shape`. If a mask
            is to be passed into the model, then the list will contain 2 sub-lists (one for each
            side), with each sub-list containing the input tenor for the face of
            :attr:`input_shape` and the input tensor for the mask, with the same shape as the face,
            but with just a single channel.
        """
        raise NotImplementedError

    def _output_summary(self):
        """ Output the summary of the model and all sub-models to the verbose logger. """
        self._model.summary(print_fn=lambda x: logger.verbose("%s", x))
        for layer in self._model.layers:
            if isinstance(layer, KerasModel):
                layer.summary(print_fn=lambda x: logger.verbose("%s", x))

    def save(self):
        """ Save the model to disk.

        Saves the serialized model, with weights, to the folder location specified when
        initializing the plugin.
        """
        self._io._save()  # pylint:disable=protected-access

    def snapshot(self):
        """ Create a snapshot of the model folder. """
        self._io._snapshot()  # pylint:disable=protected-access

    # TODO (store inputs)
    def add_predictor(self, side, model):
        """ Add a predictor to the predictors dictionary """
        logger.debug("Adding predictor: (side: '%s', model: %s)", side, model)
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

    def _compile_model(self):
        """ Compile the model to include the Optimizer and Loss Function(s). """
        logger.debug("Compiling Model")
        optimizer = self._get_optimizer()
        loss = Loss(self._model.inputs, self._model.outputs, None)
        self._model.compile(optimizer=optimizer, loss=loss.funcs)
        logger.debug("Compiled Model: %s", self._model)

    def _get_optimizer(self):
        """ Return a Keras Adam Optimizer with user selected parameters.

        Returns
        -------
        :class:`keras.optimizers.Adam`
            An Adam Optimizer with the given user settings

        Notes
        -----
        Clip-norm is ballooning VRAM usage, which is not expected behavior and may be a bug in
        Keras/Tensorflow.

        PlaidML has a bug regarding the clip-norm parameter See:
        https://github.com/plaidml/plaidml/issues/228. We workaround by simply not adding this
        parameter for AMD backend users.
        """
        # TODO add clipnorm in for plaidML when it is fixed in the main repository
        clipnorm = get_backend() != "amd" and self.config.get("clipnorm", False)
        if self._strategy.use_strategy and clipnorm:
            logger.warning("Clipnorm has been selected, but is unsupported when using "
                           "distribution strategies, so has been disabled. If you wish to enable "
                           "clipnorm, then you must use the `default` distribution strategy.")
        if self._strategy.use_strategy:
            # Tensorflow checks whether clipnorm is None rather than False when using distribution
            # strategy so we need to explicitly set it to None.
            clipnorm = None
        learning_rate = "lr" if get_backend() == "amd" else "learning_rate"
        kwargs = dict(beta_1=0.5,
                      beta_2=0.99,
                      clipnorm=clipnorm)
        kwargs[learning_rate] = self.config.get("learning_rate", 5e-5)
        retval = Adam(**kwargs)
        logger.debug("Optimizer: %s, kwargs: %s", retval, kwargs)
        return retval

    # TODO
    def converter(self, swap):
        """ Converter for autoencoder models """
        logger.debug("Getting Converter: (swap: %s)", swap)
        side = "a" if swap else "b"
        model = self.predictors[side]
        if self._is_predict:
            # Must compile the model to be thread safe
            model._make_predict_function()  # pylint: disable=protected-access
        retval = model.predict
        logger.debug("Got Converter: %s", retval)
        return retval


class IO():
    """ Model saving and loading functions.

    Handles the loading and saving of the plugin model from disk as well as the model backup and
    snapshot functions.

    Parameters
    ----------
    plugin: :class:`Model`
        The parent plugin class that owns the IO functions.
    model_dir: str
        The full path to the model save location
    is_predict: bool
        ``True`` if the model is being loaded for inference. ``False`` if the model is being loaded
        for training.
    """
    def __init__(self, plugin, model_dir, is_predict):
        self._plugin = plugin
        self._is_predict = is_predict
        self._model_dir = model_dir
        self._backup = Backup(self._model_dir, self._plugin.name)

    @property
    def _filename(self):
        """str: The filename for this model."""
        return os.path.join(self._model_dir, "{}.h5".format(self._plugin.name))

    @property
    def model_exists(self):
        """ bool: ``True`` if a model of the type being loaded exists within the model folder
        location otherwise ``False``.
        """
        return os.path.isfile(self._filename)

    # TODO
    @property
    def multiple_models_in_folder(self):
        """ :bool: ``True`` if there are multiple model types in the same folder otherwise
        ``false``. """
        model_files = [fname for fname in os.listdir(self._model_dir) if fname.endswith(".h5")]
        retval = False if not model_files else os.path.commonprefix(model_files) == ""
        logger.debug("model_files: %s, retval: %s", model_files, retval)
        return retval

    def _load(self):
        """ Loads the model from disk

        If the predict function is to be called and the model cannot be found in the model folder
        then an error is logged and the process exits.

        When loading the model, the plugin model folder is scanned for custom layers which are
        added to Keras' custom objects.

        Returns
        -------
        :class:`keras.models.Model`
            The saved model loaded from disk
        """
        logger.debug("Loading model: %s", self._filename)
        if self._is_predict and not self.model_exists:
            logger.error("Model could not be found in folder '%s'. Exiting", self._model_dir)
            sys.exit(1)

        self._add_custom_objects()
        model = load_model(self._filename)
        logger.info("Loaded model from disk: '%s'", self._filename)
        return model

    def _add_custom_objects(self):
        """ Add the plugin's layers to Keras custom objects """
        custom_objects = {name: obj
                          for name, obj in inspect.getmembers(sys.modules[self._plugin.__module__])
                          if (inspect.isclass(obj)
                              and obj.__module__ == self._plugin.__module__
                              and Layer in obj.__bases__)}
        logger.debug("Adding custom objects: %s", custom_objects)
        get_custom_objects().update(custom_objects)

    def _save(self):
        """ Backup and save the model and state file.

        Notes
        -----
        The backup function actually backups the model from the previous save iteration rather than
        the current save iteration. This is not a bug, but protection against long save times, as
        models can get quite large, so renaming the current model file rather than copying it can
        save substantial amount of time.
        """
        logger.debug("Backing up and saving models")
        print("")  # Insert a new line to avoid spamming the same row as loss output
        save_averages = self._get_save_averages()
        if save_averages and self._should_backup(save_averages):
            self._backup.backup_model(self._filename)
            self._backup.backup_model(self._plugin.state.filename)

        self._plugin.model.save(self._filename, include_optimizer=False)
        self._plugin.state.save()

        msg = "[Saved models]"
        if save_averages:
            lossmsg = ["face_{}: {:.5f}".format(side, avg)
                       for side, avg in zip(("a", "b"), save_averages)]
            msg += " - Average loss since last save: {}".format(", ".join(lossmsg))
        logger.info(msg)

    def _get_save_averages(self):
        """ Return the average loss since the last save iteration and reset historical loss """
        logger.debug("Getting save averages")
        if not all(loss for loss in self._plugin.history):
            logger.debug("No loss in history")
            retval = []
        else:
            retval = [sum(loss) / len(loss) for loss in self._plugin.history]
            self._plugin.history = [[], []]  # Reset historical loss
        logger.debug("Average losses since last save: %s", retval)
        return retval

    def _should_backup(self, save_averages):
        """ Check whether the loss averages for this save iteration is the lowest that has been
        seen.

        This protects against model corruption by only backing up the model if both sides have
        seen a total fall in loss.

        Notes
        -----
        This is by no means a perfect system. If the model corrupts at an iteration close
        to a save iteration, then the averages may still be pushed lower than a previous
        save average, resulting in backing up a corrupted model.

        Parameters
        ----------
        save_averages: list
            The average loss for each side for this save iteration
        """
        backup = True
        for side, loss in zip(("a", "b"), save_averages):
            if not self._plugin.state.lowest_avg_loss.get(side, None):
                logger.debug("Set initial save iteration loss average for '%s': %s", side, loss)
                self._plugin.state.lowest_avg_loss[side] = loss
                continue
            backup = loss < self._plugin.state.lowest_avg_loss[side] if backup else backup

        if backup:  # Update lowest loss values to the state file
            # pylint:disable=unnecessary-comprehension
            old_avgs = {key: val for key, val in self._plugin.state.lowest_avg_loss.items()}
            self._plugin.state.lowest_avg_loss["a"] = save_averages[0]
            self._plugin.state.lowest_avg_loss["b"] = save_averages[1]
            logger.debug("Updated lowest historical save iteration averages from: %s to: %s",
                         old_avgs, self._plugin.state.lowest_avg_loss)

        logger.debug("Should backup: %s", backup)
        return backup

    def _snapshot(self):
        """ Perform a model snapshot.

        Notes
        -----
        Snapshot function is called 1 iteration after the model was saved, so that it is built from
        the latest save, hence iteration being reduced by 1.
        """
        logger.debug("Performing snapshot. Iterations: %s", self._plugin.iterations)
        self._backup.snapshot_models(self._plugin.iterations - 1)
        logger.debug("Performed snapshot")


class Strategy():
    """ Distribution Strategies for Tensorflow.

    Tensorflow 2 uses distribution strategies for multi-GPU/system training. These are context
    managers. To enable the code to be more readable, we handle strategies the same way for Nvidia
    and AMD backends. PlaidML does not support strategies, but we need to still create a context
    manager so that we don't need branching logic.

    Parameters
    ----------
    strategy: ["default", "mirror", "central"]
        The required strategy. `"default"` effectively means 'do not explicitly provide a strategy'
        and let's Tensorflow handle things for itself. `"mirror" is Tensorflow Mirrored Strategy.
        "`central`" is Tensorflow Central Storage Strategy with variables explicitly placed on the
        CPU.
    """
    def __init__(self, strategy):
        logger.debug("Initializing %s: (strategy: %s)", self.__class__.__name__, strategy)
        self._strategy = self._get_strategy(strategy)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def use_strategy(self):
        """ bool: ``True`` if a distribution strategy is to be used otherwise ``False``. """
        return self._strategy is not None

    @staticmethod
    def _get_strategy(strategy):
        """ If we are running on Nvidia backend and the strategy is not `"default"` then return
        the correct tensorflow distribution strategy, otherwise return ``None``.

        Notes
        -----
        By default Tensorflow defaults mirrored strategy to use the Nvidia NCCL method for
        reductions, however this is only available in Linux, so the method used falls back to
        `Hierarchical Copy All Reduce` if the OS is not Linux.

        Parameters
        ----------
        strategy: str
            The request training strategy to use

        Returns
        -------
        :class:`tensorflow.python.distribute.Strategy` or `None`
            The request Tensorflow Strategy if the backend is Nvidia and the strategy is not
            `"Default"` otherwise ``None``
        """
        if get_backend() != "nvidia":
            retval = None
        elif strategy == "mirror":
            if platform.system().lower() == "linux":
                cross_device_ops = tf.distribute.NcclAllReduce()
            else:
                cross_device_ops = tf.distribute.HierarchicalCopyAllReduce()
            logger.debug("cross_device_ops: %s", cross_device_ops)
            retval = tf.distribute.MirroredStrategy(cross_device_ops=cross_device_ops)
        elif strategy == "central":
            retval = tf.distribute.experimental.CentralStorageStrategy(parameter_device="/CPU:0")
        else:
            retval = tf.distribute.get_strategy()
        logger.debug("Using strategy: %s", retval)
        return retval

    def scope(self):
        """ Return the strategy scope if we have set a strategy, otherwise return a null
        context.

        Returns
        -------
        :func:`tensorflow.python.distribute.Strategy.scope` or :func:`contextlib.nullcontext`
            The tensorflow strategy scope if a strategy is valid in the current scenario. A null
            context manager if the strategy is not valid in the current scenario
        """
        retval = nullcontext() if self._strategy is None else self._strategy.scope()
        logger.debug("Using strategy scope: %s", retval)
        return retval


class Loss():
    """ Holds loss names and functions for an Autoencoder """
    def __init__(self, inputs, outputs, mask_variable):
        logger.debug("Initializing %s: (inputs: %s, outputs: %s, mask_variable: %s)",
                     self.__class__.__name__, inputs, outputs, mask_variable)
        self.inputs = inputs
        self._mask_variable = mask_variable
        self.funcs = self._get_loss_functions(outputs)
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
    def mask_input(self):
        """ Return the mask input or None """
        mask_inputs = [inp for inp in self.inputs if inp.name.startswith("mask")]
        if not mask_inputs:
            return None
        return mask_inputs[0]

    @property
    def mask_shape(self):
        """ Return the mask shape """
        if self.mask_input is None and self._mask_variable is None:
            return None
        if self.mask_input:
            retval = K.int_shape(self.mask_input)[1:]
        else:
            retval = K.int_shape(self._mask_variable)
        return retval

    def _get_loss_functions(self, outputs):
        """ Set the loss functions.

        Parameters
        ----------
        outputs: list
            A list of output tensors from the model plugin

        list
            A list of loss functions to apply to the model
        """
        # TODO naming for masks and multi-scale
        loss_funcs = []
        output_shapes = [K.int_shape(output)[1:] for output in outputs]
        output_types = ["mask" if shape[-1] == 1 else "face" for shape in output_shapes]
        for idx, loss_type in enumerate(output_types):
            if loss_type == "mask":
                loss_funcs.append(self.selected_mask_loss)
            elif self.config["penalized_mask_loss"] and self.config["mask_type"] is not None:
                face_size = output_shapes[idx][1]
                mask_size = self.mask_shape[1]
                scaling = face_size / mask_size
                logger.debug("face_size: %s mask_size: %s, mask_scaling: %s",
                             face_size, mask_size, scaling)
                loss_funcs.append(PenalizedLoss(self._mask_variable, self.selected_loss,
                                                mask_scaling=scaling,
                                                preprocessing_func=self.mask_preprocessing_func))
            else:
                loss_funcs.append(self.selected_loss)
            logger.debug("%s: %s", loss_type, loss_funcs[-1])
        logger.debug(loss_funcs)
        return loss_funcs


class State():
    """ Class to hold the model's current state and autoencoder structure """
    def __init__(self, model_dir, model_name, config_changeable_items,
                 no_logs, training_image_size):
        logger.debug("Initializing %s: (model_dir: '%s', model_name: '%s', "
                     "config_changeable_items: '%s', no_logs: %s, "
                     "training_image_size: '%s'", self.__class__.__name__, model_dir, model_name,
                     config_changeable_items, no_logs, training_image_size)
        self.serializer = get_serializer("json")
        filename = "{}_state.{}".format(model_name, self.serializer.file_extension)
        self.filename = os.path.join(model_dir, filename)
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
        self.create_new_session(no_logs, config_changeable_items)
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

    def create_new_session(self, no_logs, config_changeable_items):
        """ Create a new session """
        logger.debug("Creating new session. id: %s", self.session_id)
        self.sessions[self.session_id] = {"timestamp": time.time(),
                                          "no_logs": no_logs,
                                          "batchsize": 0,
                                          "iterations": 0,
                                          "config": config_changeable_items}

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

    def save(self):
        """ Save iteration number to state file """
        logger.debug("Saving State")
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
