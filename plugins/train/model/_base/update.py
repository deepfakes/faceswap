#! /usr/env/bin/python3
""" Updating legacy faceswap models to the current version """
import json
import logging
import os
import typing as T
import zipfile
from shutil import copyfile, copytree

import h5py
import numpy as np
from keras import models as kmodels

from lib.logger import parse_class_init
from lib.model.layers import ScalarOp
from lib.model.networks import TypeModelsViT, ViT
from lib.utils import get_module_objects, FaceswapError

logger = logging.getLogger(__name__)


class Legacy:  # pylint:disable=too-few-public-methods
    """ Handles the updating of Keras 2.x models to Keras 3.x

    Generally Keras 2.x models will open in Keras 3.x. There are a couple of bugs in Keras 3
    legacy loading code which impacts Faceswap models:
    - When a model receives a shared functional model as an inbound node, the node index needs
    reducing by 1 (non-trivial to fix upstream)
    - Keras 3 does not accept nested outputs, so Keras 2 FS models need to have the outputs
    flattened

    Parameters
    ----------
    model_path: str
        Full path to the legacy Keras 2.x model h5 file to upgrade
    """
    def __init__(self, model_path: str):
        logger.debug(parse_class_init(locals()))
        self._old_model_file = model_path
        """str: Full path to the old .h5 model file"""
        self._new_model_file = f"{os.path.splitext(model_path)[0]}.keras"
        """str: Full path to the new .keras model file"""
        self._functionals: set[str] = set()
        """set[str]: The name of any Functional models discovered in the keras 2 model config"""

        self._upgrade_model()
        logger.debug("Initialized %s", self.__class__.__name__)

    def _get_model_config(self) -> dict[str, T.Any]:
        """ Obtain a keras 2.x config from a keras 2.x .h5 file.

        As keras 3.x will error out loading the file, we collect it directly from the .h5 file

        Returns
        -------
        dict[str, Any]
            A keras 2.x model configuration dictionary

        Raises
        ------
        FaceswapError
            If the file is not a valid Faceswap 2 .h5 model file
        """
        h5file = h5py.File(self._old_model_file, "r")
        s_version = T.cast(str | None, h5file.attrs.get("keras_version"))
        s_config = T.cast(str | None, h5file.attrs.get("model_config"))
        if not s_version or not s_config:
            raise FaceswapError(f"'{self._old_model_file}' is not a valid Faceswap 2 model file")

        version = s_version.split(".")[:2]
        if len(version) != 2 or version[0] != "2":
            raise FaceswapError(f"'{self._old_model_file}' is not a valid Faceswap 2 model file")

        retval = json.loads(s_config)
        logger.debug("Loaded keras 2.x model config: %s", retval)
        return retval

    @classmethod
    def _unwrap_outputs(cls, outputs: list[list[T.Any]]) -> list[list[str | int]]:
        """ Unwrap nested output tensors from a config dict to be a single list of output tensor

        Parameters
        ----------
        outputs: list[list[Any]]
            The outputs that exist within the Keras 2 config dict that may be nested

        Returns
        -------
        list[list[str | int]]
            The output configuration formatted to be compatible with Keras 3
        """
        retval = np.array(outputs).reshape(-1, 3).tolist()
        for item in retval:
            item[1] = int(item[1])
            item[2] = int(item[2])
        logger.debug("Unwrapped outputs: %s to: %s", outputs, retval)
        return retval

    def _get_clip_config(self) -> dict[str, T.Any]:
        """ Build a clip model from the configuration information stored in the legacy state file

        Returns
        -------
        dict[str, T.Any]
            The new keras configuration for a Clip model

        Raises
        ------
        FaceswapError
            If the clip model cannot be built
        """
        state_file = f"{os.path.splitext(self._old_model_file)[0]}_state.json"
        if not os.path.isfile(state_file):
            raise FaceswapError(
                f"The state file '{state_file}' does not exist. This model cannot be ported")

        with open(state_file, "r", encoding="utf-8") as ifile:
            config = json.load(ifile)

        logger.debug("Loaded legacy config '%s': %s", state_file, config)
        net_name = config.get("config", {}).get("enc_architecture", "")
        scaling = config.get("config", {}).get("enc_scaling", 0) / 100

        # Import here to prevent circular imports
        from plugins.train.model.phaze_a import _MODEL_MAPPING  # pylint:disable=C0415
        vit_info = _MODEL_MAPPING.get(net_name)

        if not scaling or not vit_info:
            raise FaceswapError(
                f"Clip network could not be found in '{state_file}'. Discovered network is "
                f"'{net_name}' with encoder scaling: {scaling}. This model cannot be ported")

        input_size = int(max(vit_info.min_size, ((vit_info.default_size * scaling) // 16) * 16))
        vit_model = ViT(T.cast(TypeModelsViT, vit_info.keras_name), input_size=input_size)()

        retval = vit_model.get_config()
        del vit_model
        logger.debug("Got new config for '%s' at input size: %s: %s", net_name, input_size, retval)
        return retval

    def _convert_lambda_config(self, layer: dict[str, T.Any]):
        """ Keras 2 TFLambdaOps are not compatible with Keras 3. Scalar operations can be
        relatively easily substituted with a :class:`~lib.model.layers.ScalarOp` layer

        Parameters
        ----------
        layer: dict[str, Any]
            An existing Keras 2 TFLambdaOp layer

        Raises
        ------
        FaceswapError
            If the TFLambdaOp is not currently supported
        """
        name = layer["config"]["name"]
        operation = name.rsplit(".", maxsplit=1)[-1]
        if operation not in ("multiply", "truediv", "add", "subtract"):
            raise FaceswapError(f"The TFLambdaOp '{name}' is not supported")
        value = layer["inbound_nodes"][0][-1]["y"]

        if isinstance(layer["config"]["dtype"], str):
            dtype = layer["config"]["dtype"]
        else:
            dtype = layer["config"]["dtype"]["config"]["name"]
        new_layer = ScalarOp(operation, value, name=name, dtype=dtype)

        logger.debug("Converting legacy TFLambdaOp: %s", layer)

        layer["class_name"] = "ScalarOp"
        layer["config"] = new_layer.get_config()
        for n in layer["inbound_nodes"]:
            n[-1] = {}
        layer["inbound_nodes"] = [layer["inbound_nodes"]]
        logger.debug("Converted legacy TFLambdaOp to %s", layer)

    def _process_deprecations(self, layer: dict[str, T.Any]) -> None:
        """ Some layer kwargs are deprecated between Keras 2 and Keras 3. Some are not mission
        critical, but updating these here prevents Keras from outputting warnings about deprecated
        arguments. Others will fail to load the legacy model (eg Clip) so are replaced with a new
        config. Operation is performed in place

        Parameters
        ----------
        layer: dict[str, T.Any]
            A keras model config item representing a keras layer
        """
        if layer["class_name"] == "LeakyReLU":
            # Non mission-critical, but prevents scary deprecation messages
            config = layer["config"]
            old, new = "alpha", "negative_slope"
            if old in config:
                logger.debug("Updating '%s' kwarg '%s' to '%s'", layer["name"], old, new)
                config[new] = config[old]
                del config[old]

        if layer["name"] == "visual":
            # MultiHeadAttention is not backwards compatible, so get new config for Clip models
            logger.debug("Getting new config for 'visual' model")
            layer["config"] = self._get_clip_config()

        if layer["class_name"] == "TFOpLambda":
            # TFLambdaOp are not supported
            self._convert_lambda_config(layer)

        if layer["class_name"] in ("DepthwiseConv2D",
                                   "Conv2DTranspose") and "groups" in layer["config"]:
            # groups parameter doesn't exist in Keras 3. Hopefully it still works the same
            logger.debug("Removing groups from %s '%s'", layer["class_name"], layer["name"])
            del layer["config"]["groups"]

        if "dtype" in layer["config"]:
            # Incorrectly stored dtypes error when deserializing the new config. May be a Keras bug
            actual_dtype = None
            old_dtype = layer["config"]["dtype"]
            if isinstance(old_dtype, str):
                actual_dtype = layer["config"]["dtype"]
            if isinstance(old_dtype, dict) and old_dtype.get("class_name") == "Policy":
                actual_dtype = old_dtype["config"]["name"]

            if actual_dtype is not None:
                new_dtype = {"module": "keras",
                             "class_name": "DTypePolicy",
                             "config": {"name": actual_dtype},
                             "registered_name": None}
                logger.debug("Updating dtype for '%s' from %s to %s", layer["name"],
                             old_dtype, new_dtype)
                layer["config"]["dtype"] = new_dtype

    def _process_inbounds(self,
                          layer_name: str,
                          inbound_nodes: list[list[list[str | int]]] | list[list[str | int]]
                          ) -> None:
        """ If the inbound nodes are from a shared functional model, decrement the node index by
        one. Operation is performed in place

        Parameters
        ----------
        layer_name: str
            The name of the layer (for logging)
        inbound_nodes: list[list[list[str | int]]] | list[list[str | int]]
            The inbound nodes from a Keras 2 config dict to process
        """
        to_process = T.cast(
            list[list[list[str | int]]],
            inbound_nodes if isinstance(inbound_nodes[0][0], list) else [inbound_nodes])

        for inbound in to_process:
            for node in inbound:
                name, node_index = node[0], node[1]
                assert isinstance(name, str) and isinstance(node_index, int)
                if name in self._functionals and node_index > 0:
                    logger.debug("Updating '%s' inbound node index for '%s' from %s to %s",
                                 layer_name, name, node_index, node_index - 1)
                    node[1] = node_index - 1

    def _update_layers(self, layer_list: list[dict[str, T.Any]]) -> None:
        """ Given a list of keras layers from a keras 2 config dict, increment the indices for
        any inbound nodes that come from a shared Functional model. Flatten any nested output
        tensor lists. Operations are performed in place

        Parameters
        ----------
        layers: list[dict[str, Any]]
            A list of layers that belong to a keras 2 functional model config dictionary
        """
        for layer in layer_list:
            if layer["class_name"] == "Functional":
                logger.debug("Found Functional layer. Keys: %s", list(layer))

                if layer.get("name"):
                    logger.debug("Storing layer: '%s'", layer["name"])
                    self._functionals.add(layer["name"])

                layer["config"]["output_layers"] = self._unwrap_outputs(
                    layer["config"]["output_layers"])

                self._update_layers(layer["config"]["layers"])

            if not layer.get("inbound_nodes"):
                continue

            self._process_deprecations(layer)
            self._process_inbounds(layer["name"], layer["inbound_nodes"])

    def _archive_model(self) -> str:
        """ Archive an existing Keras 2 model to a new archive location

        Raises
        ------
        FaceswapError
            If the destination archive folder exists and is not empty

        Returns
        -------
        str
            The path to the archived keras 2 model folder
        """
        model_dir = os.path.dirname(self._old_model_file)
        dst_path = f"{model_dir}_fs2_backup"
        if os.path.exists(dst_path) and os.listdir(dst_path):
            raise FaceswapError(
                f"The destination archive folder '{dst_path}' already exists. Either delete this "
                "folder, select a different model folder, or remove the legacy model files from "
                f"your model folder '{model_dir}'.")

        if os.path.exists(dst_path):
            logger.info("Removing pre-existing empty folder '%s'", dst_path)
            os.rmdir(dst_path)

        logger.info("Archiving model folder '%s' to '%s'", model_dir, dst_path)
        os.rename(model_dir, dst_path)
        return dst_path

    def _restore_files(self, archive_dir: str) -> None:
        """ Copy the state.json file and the logs folder from the archive folder to the new model
        folder

        Parameters
        ----------
        archive_dir: str
            The full path to the archived Keras 2 model
        """
        model_dir = os.path.dirname(self._new_model_file)
        model_name = os.path.splitext(os.path.basename(self._new_model_file))[0]
        logger.debug("Restoring required '%s 'files from '%s' to '%s'",
                     model_name, archive_dir, model_dir)

        for fname in os.listdir(archive_dir):
            fullpath = os.path.join(archive_dir, fname)
            new_path = os.path.join(model_dir, fname)

            if fname == f"{model_name}_logs" and os.path.isdir(fullpath):
                logger.debug("Restoring '%s' to '%s'", fullpath, new_path)
                copytree(fullpath, new_path)
                continue

            if fname == f"{model_name}_state.json" and os.path.isfile(fullpath):
                logger.debug("Restoring '%s' to '%s'", fullpath, new_path)
                copyfile(fullpath, new_path)
                continue

            logger.debug("Skipping file: '%s'", fname)

    def _upgrade_model(self) -> None:
        """ Get the model configuration of a Faceswap 2 model and upgrade it to Faceswap 3
        compatible """
        logger.info("Upgrading model file from Faceswap 2 to Faceswap 3...")
        config = self._get_model_config()
        self._update_layers([config])

        logger.debug("Migrating data to new model...")
        model = kmodels.Model.from_config(config["config"])
        model.load_weights(self._old_model_file)

        archive_dir = self._archive_model()

        dirname = os.path.dirname(self._new_model_file)
        logger.debug("Saving model '%s'", self._new_model_file)
        os.mkdir(dirname)
        model.save(self._new_model_file)
        logger.debug("Saved model '%s'", self._new_model_file)

        self._restore_files(archive_dir)
        logger.info("Model upgraded: '%s'", dirname)


class PatchKerasConfig:
    """ This class exists to patch breaking changes when moving from older keras 3.x models to
    newer versions

    Parameters
    ----------
    model_path : str
        Full path to the keras model to be patched for the current version
    """
    def __init__(self, model_path: str) -> None:
        logger.debug(parse_class_init(locals()))
        self._model_path = model_path
        self._items, self._config = self._load_model()
        metadata = json.loads(self._items["metadata.json"])
        self._version = tuple(int(x) for x in metadata['keras_version'].split(".")[:2])
        logger.debug("Initialized: %s", self.__class__.__name__)

    def _load_model(self) -> tuple[dict[str, bytes], dict[str, T.Any]]:
        """ Load the objects from the compressed keras model

        Returns
        -------
        items : dict[str, bytes]
            The filename and file objects within the keras 3 model file that are not the model
            config
        config : dict[str, Any]
            The model configuration dictionary from the keras 3 model file
        """
        with zipfile.ZipFile(self._model_path, "r") as zf:
            items = {f.filename: zf.read(f) for f in zf.filelist if f.filename != "config.json"}
            config = json.loads(zf.read("config.json"))

        logger.debug("Loaded legacy existing items %s and 'config.json' from model '%s'",
                     list(items), self._model_path)
        return items, config

    def _update_nn_blocks(self, layer: dict[str, T.Any]):
        """ In older versions of keras our :class:`lib.model.nn_blocks.Conv2D` and
        :class:`lib.model.nn_blocks.DepthwiseConv2D` inherited from their respective Keras layers.
        Sometime between 3.3.3 and 3.12 (during beta testing) this stopped working, raising a
        TypeError. Subsequently we have refactored those classes to no longer inherit, and call the
        underlying keras layer directly instead. The keras config needs to be rewritten to reflect
        this.

        Parameters
        ----------
        layer dict[str, Any]
            A layer config dictionary from a keras 3 model
        """
        if (layer.get("module") == "lib.model.nn_blocks" and
                layer.get("class_name") in ("Conv2D", "DepthwiseConv2D")):
            new_module = "keras.layers"
            logger.debug("Updating Keras %s layer '%s' to '%s': %s",
                         ".".join(str(x) for x in self._version),
                         f"{layer['module']}.{layer['class_name']}",
                         f"{new_module}.{layer['class_name']}",
                         layer["name"])
            layer["module"] = new_module

    def _parse_inbound_args(self, inbound: list | dict[str, T.Any]) -> None:
        """ Recurse through keras inbound node args until we arrive at a dictionary

        Parameters
        ----------
        list[lisr | dict[str, Any]]
            A Keras inbound nodes args entry or the nested dictionary
        """
        if not isinstance(inbound, (list, dict)):
            return

        if isinstance(inbound, list):
            for arg in inbound:
                self._parse_inbound_args(arg)
            return

        arg_conf = inbound["config"]
        if "keras_history" not in arg_conf:
            return

        if "." in arg_conf["keras_history"][0]:
            new_hist = arg_conf["keras_history"][:]
            new_hist[0] = new_hist[0].replace(".", "_")
            logger.debug("Updating Inbound Keras history from '%s' to '%s'",
                         arg_conf["keras_history"], new_hist)
            arg_conf["keras_history"] = new_hist

    def _update_dot_naming(self, layer: dict[str, T.Any]):
        """ Sometime between 3.3.3 and 3.12 (during beta testing) layers with "." in the name
        started generating a KeyError. This is odd as the error comes from Torch, but dot naming is
        standard. To work around this all dots (.) in layer names have been converted to
        underscores (_). The keras config needs to be rewritten to reflect this. This only impacts
        FS models that used the CLiP encoder

        Parameters
        ----------
        layer dict[str, Any]
            A layer config dictionary from a keras 3 model
        """
        if "." in layer["name"]:
            new_name = layer["name"].replace(".", "_")
            logger.debug("Updating Keras layer name from '%s' to '%s'", layer["name"], new_name)
            layer["name"] = new_name

        config = layer["config"]
        if "." in config["name"]:
            new_name = config["name"].replace(".", "_")
            logger.debug("Updating Keras config layer name from '%s' to '%s'",
                         config["name"], new_name)
            config["name"] = new_name

        inbound = layer["inbound_nodes"]
        for in_ in inbound:
            for arg in in_["args"]:
                self._parse_inbound_args(arg)

    def _update_config(self, config: dict[str, T.Any]) -> dict[str, T.Any]:
        """ Recursively update the `config` dictionary from a full keras config in place

        Parameters
        ----------
        config : dict[str, Any]
            A 'config' section of keras config

        Returns
        -------
        dict[str, Any]
            The updated `config` section of a keras config
        """
        layer: dict[str, T.Any]
        for layer in config["layers"]:
            if layer.get("class_name") == "Functional":
                self._update_config(layer["config"])
            if self._version <= (3, 3):
                self._update_nn_blocks(layer)
                self._update_dot_naming(layer)
        return config

    def _save_model(self) -> None:
        """ Save the updated keras model """
        logger.info("Updating Keras model '%s'...", self._model_path)
        with zipfile.ZipFile(self._model_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename, data in self._items.items():
                zf.writestr(filename, data)
            zf.writestr("config.json", json.dumps(self._config).encode("utf-8"))

    def __call__(self) -> None:
        """ Update the keras configuration saved in a keras model file and save over the original
        model """
        logger.debug("Updating saved config for keras version %s", self._version)
        self._config["config"] = self._update_config(self._config["config"])
        self._save_model()


__all__ = get_module_objects(__name__)
