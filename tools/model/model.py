#!/usr/bin/env python3
""" Tool to restore models from backup """
from __future__ import annotations
import logging
import os
import sys
import typing as T

import numpy as np
import tensorflow as tf
from tensorflow import keras

from lib.model.backup_restore import Backup

# Import the following libs for custom objects
from lib.model import initializers, layers, normalization  # noqa # pylint:disable=unused-import
from plugins.train.model._base.model import _Inference


if T.TYPE_CHECKING:
    import argparse

logger = logging.getLogger(__name__)


class Model():
    """ Tool to perform actions on a model file.

    Parameters
    ----------
    :class:`argparse.Namespace`
        The command line arguments calling the model tool
    """
    def __init__(self, arguments: argparse.Namespace) -> None:
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self._configure_tensorflow()
        self._model_dir = self._check_folder(arguments.model_dir)
        self._job = self._get_job(arguments)

    @classmethod
    def _configure_tensorflow(cls) -> None:
        """ Disable eager execution and force Tensorflow into CPU mode. """
        tf.config.set_visible_devices([], device_type="GPU")
        tf.compat.v1.disable_eager_execution()

    @classmethod
    def _get_job(cls, arguments: argparse.Namespace) -> T.Any:
        """ Get the correct object that holds the selected job.

        Parameters
        ----------
        arguments: :class:`argparse.Namespace`
            The command line arguments received for the Model tool which will be used to initiate
            the selected job

        Returns
        -------
        class
            The object that will perform the selected job
        """
        jobs = {"inference": Inference,
                "nan-scan": NaNScan,
                "restore": Restore}
        return jobs[arguments.job](arguments)

    @classmethod
    def _check_folder(cls, model_dir: str) -> str:
        """ Check that the passed in model folder exists and contains a valid model.

        If the passed in value fails any checks, process exits.

        Parameters
        ----------
        model_dir: str
            The model folder to be checked

        Returns
        -------
        str
            The confirmed location of the model folder.
        """
        if not os.path.exists(model_dir):
            logger.error("Model folder does not exist: '%s'", model_dir)
            sys.exit(1)

        chkfiles = [fname
                    for fname in os.listdir(model_dir)
                    if fname.endswith(".h5")
                    and not os.path.splitext(fname)[0].endswith("_inference")]

        if not chkfiles:
            logger.error("Could not find a model in the supplied folder: '%s'", model_dir)
            sys.exit(1)

        if len(chkfiles) > 1:
            logger.error("More than one model file found in the model folder: '%s'", model_dir)
            sys.exit(1)

        model_name = os.path.splitext(chkfiles[0])[0].title()
        logger.info("%s Model found", model_name)
        return model_dir

    def process(self) -> None:
        """ Call the selected model job."""
        self._job.process()


class Inference():
    """ Save an inference model from a trained Faceswap model.

    Parameters
    ----------
    :class:`argparse.Namespace`
        The command line arguments calling the model tool
    """
    def __init__(self, arguments: argparse.Namespace) -> None:
        self._switch = arguments.swap_model
        self._format = arguments.format
        self._input_file, self._output_file = self._get_output_file(arguments.model_dir)

    def _get_output_file(self, model_dir: str) -> tuple[str, str]:
        """ Obtain the full path for the output model file/folder

        Parameters
        ----------
        model_dir: str
            The full path to the folder containing the Faceswap trained model .h5 file

        Returns
        -------
        str
            The full path to the source model file
        str
            The full path to the inference model save location
         """
        model_name = next(fname for fname in os.listdir(model_dir) if fname.endswith(".h5"))
        in_path = os.path.join(model_dir, model_name)
        logger.debug("Model input path: '%s'", in_path)

        model_name = f"{os.path.splitext(model_name)[0]}_inference"
        model_name = f"{model_name}.h5" if self._format == "h5" else model_name
        out_path = os.path.join(model_dir, model_name)
        logger.debug("Inference output path: '%s'", out_path)
        return in_path, out_path

    def process(self) -> None:
        """ Run the inference model creation process. """
        logger.info("Loading model '%s'", self._input_file)
        model = keras.models.load_model(self._input_file, compile=False)
        logger.info("Creating inference model...")
        inference = _Inference(model, self._switch).model
        logger.info("Saving to: '%s'", self._output_file)
        inference.save(self._output_file)


class NaNScan():
    """ Tool to scan for NaN and Infs in model weights.

    Parameters
    ----------
    :class:`argparse.Namespace`
        The command line arguments calling the model tool
    """
    def __init__(self, arguments: argparse.Namespace) -> None:
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self._model_file = self._get_model_filename(arguments.model_dir)

    @classmethod
    def _get_model_filename(cls, model_dir: str) -> str:
        """ Obtain the full path the model's .h5 file.

        Parameters
        ----------
        model_dir: str
            The full path to the folder containing the model file

        Returns
        -------
        str
            The full path to the saved model file
        """
        model_file = next(fname for fname in os.listdir(model_dir) if fname.endswith(".h5"))
        return os.path.join(model_dir, model_file)

    def _parse_weights(self,
                       layer: keras.models.Model | keras.layers.Layer) -> dict:
        """ Recursively pass through sub-models to scan layer weights"""
        weights = layer.get_weights()
        logger.debug("Processing weights for layer '%s', length: '%s'",
                     layer.name, len(weights))

        if not weights:
            logger.debug("Skipping layer with no weights: %s", layer.name)
            return {}

        if hasattr(layer, "layers"):  # Must be a submodel
            retval = {}
            for lyr in layer.layers:
                info = self._parse_weights(lyr)
                if not info:
                    continue
                retval[lyr.name] = info
            return retval

        nans = sum(np.count_nonzero(np.isnan(w)) for w in weights)
        infs = sum(np.count_nonzero(np.isinf(w)) for w in weights)

        if nans + infs == 0:
            return {}
        return {"nans": nans, "infs": infs}

    def _parse_output(self, errors: dict, indent: int = 0) -> None:
        """ Parse the output of the errors dictionary and print a pretty summary.

        Parameters
        ----------
        errors: dict
            The nested dictionary of errors found when parsing the weights

        indent: int, optional
            How far should the current printed line be indented. Default: `0`
        """
        for key, val in errors.items():
            logline = f"|{'--' * indent} "
            logline += key.ljust(50 - len(logline))
            if isinstance(val, dict) and "nans" not in val:
                logger.info(logline)
                self._parse_output(val, indent + 1)
            elif isinstance(val, dict) and "nans" in val:
                logline += f"nans: {val['nans']}, infs: {val['infs']}"
                logger.info(logline.ljust(30))

    def process(self) -> None:
        """ Scan the loaded model for NaNs and Infs and output summary. """
        logger.info("Loading model...")
        model = keras.models.load_model(self._model_file, compile=False)
        logger.info("Parsing weights for invalid values...")
        errors = self._parse_weights(model)

        if not errors:
            logger.info("No invalid values found in model: '%s'", self._model_file)
            sys.exit(1)

        logger.info("Invalid values found in model: %s", self._model_file)
        self._parse_output(errors)


class Restore():
    """ Restore a model from backup.

    Parameters
    ----------
    :class:`argparse.Namespace`
        The command line arguments calling the model tool
    """
    def __init__(self, arguments: argparse.Namespace) -> None:
        logger.debug("Initializing %s: (arguments: '%s'", self.__class__.__name__, arguments)
        self._model_dir = arguments.model_dir
        self._model_name = self._get_model_name()

    def process(self) -> None:
        """ Perform the Restore process """
        logger.info("Starting Model Restore...")
        backup = Backup(self._model_dir, self._model_name)
        backup.restore()
        logger.info("Completed Model Restore")

    def _get_model_name(self) -> str:
        """ Additional checks to make sure that a backup exists in the model location. """
        bkfiles = [fname for fname in os.listdir(self._model_dir) if fname.endswith(".bk")]
        if not bkfiles:
            logger.error("Could not find any backup files in the supplied folder: '%s'",
                         self._model_dir)
            sys.exit(1)
        logger.verbose("Backup files: %s)", bkfiles)  # type:ignore

        model_name = next(fname for fname in bkfiles if fname.endswith(".h5.bk"))
        return model_name[:-6]
