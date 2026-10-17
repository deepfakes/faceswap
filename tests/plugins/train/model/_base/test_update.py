#!/usr/bin python3
""" Pytest unit tests for :mod:`plugins.train.model._base.update` Faceswap 2 model upgrader """
# pylint:disable=protected-access,invalid-name
import typing as T

import pytest
import pytest_mock
from keras import device, models as k_models

from plugins.train.model._base import update as mod_update


def _input_layer(name: str, shape: list[int | None]) -> dict[str, T.Any]:
    """ A Keras 2 InputLayer config entry """
    return {"class_name": "InputLayer",
            "name": name,
            "inbound_nodes": [],
            "config": {"batch_input_shape": shape,
                       "dtype": "float32",
                       "sparse": False,
                       "name": name}}


def _decoder(side: str, node_index: int) -> dict[str, T.Any]:
    """ A Keras 2 'Model' config entry for a decoder that takes its input from the shared encoder
    at the given node index """
    return {"class_name": "Model",
            "name": f"decoder_{side}",
            "inbound_nodes": [[["encoder", node_index, 0, {}]]],
            "config": {"name": f"decoder_{side}",
                       "layers": [_input_layer(f"input_{side}", [None, 8, 8, 3]),
                                  {"class_name": "Activation",
                                   "name": f"face_out_{side}",
                                   "inbound_nodes": [[[f"input_{side}", 0, 0, {}]]],
                                   "config": {"name": f"face_out_{side}",
                                              "trainable": True,
                                              "activation": "sigmoid"}}],
                       "input_layers": [[f"input_{side}", 0, 0]],
                       "output_layers": [[f"face_out_{side}", 0, 0]]}}


def legacy_config() -> dict[str, T.Any]:
    """ A minimal Keras 2 (TF < 2.4) model config as stored in a Faceswap 2 .h5 model file.

    Functional models are serialized with class name 'Model' (rather than 'Functional'), the
    shared 'encoder' sub-model is called once per input (node indices 1 and 2) and the
    SpatialDropout2D layer carries the 'trainable' and 'noise_shape' kwargs that Keras 3 does not
    accept """
    encoder = {"class_name": "Model",
               "name": "encoder",
               "inbound_nodes": [[["face_in_a", 0, 0, {}]], [["face_in_b", 0, 0, {}]]],
               "config": {"name": "encoder",
                          "layers": [_input_layer("input_1", [None, 8, 8, 3]),
                                     {"class_name": "SpatialDropout2D",
                                      "name": "spatial_dropout2d_1",
                                      "inbound_nodes": [[["input_1", 0, 0, {}]]],
                                      "config": {"name": "spatial_dropout2d_1",
                                                 "trainable": True,
                                                 "rate": 0.25,
                                                 "noise_shape": None,
                                                 "seed": None}},
                                     {"class_name": "LeakyReLU",
                                      "name": "leaky_re_lu_1",
                                      "inbound_nodes": [[["spatial_dropout2d_1", 0, 0, {}]]],
                                      "config": {"name": "leaky_re_lu_1",
                                                 "trainable": True,
                                                 "alpha": 0.1}}],
                          "input_layers": [["input_1", 0, 0]],
                          "output_layers": [["leaky_re_lu_1", 0, 0]]}}
    return {"class_name": "Model",
            "config": {"name": "legacy",
                       "layers": [_input_layer("face_in_a", [None, 8, 8, 3]),
                                  _input_layer("face_in_b", [None, 8, 8, 3]),
                                  encoder,
                                  _decoder("a", 1),
                                  _decoder("b", 2)],
                       "input_layers": [["face_in_a", 0, 0], ["face_in_b", 0, 0]],
                       "output_layers": [["decoder_a", 1, 0], ["decoder_b", 1, 0]]}}


def flatten_layers(config: dict[str, T.Any]) -> dict[str, dict[str, T.Any]]:
    """ Collect all (nested) layer config entries from a model config, keyed by layer name """
    retval: dict[str, dict[str, T.Any]] = {}
    for layer in config["config"]["layers"]:
        retval[layer["name"]] = layer
        if layer["class_name"] in ("Model", "Functional"):
            retval |= flatten_layers(layer)
    return retval


@pytest.fixture(name="legacy")
def legacy_fixture(mocker: pytest_mock.MockFixture) -> mod_update.Legacy:
    """ A :class:`~plugins.train.model._base.update.Legacy` instance that has not run an
    upgrade """
    mocker.patch.object(mod_update.Legacy, "_upgrade_model")
    return mod_update.Legacy("legacy.h5")


def test_update_layers_legacy_model_class(legacy):
    """ Keras 2 (TF < 2.4) serializes functional models with class name 'Model'. They must be
    processed the same as 'Functional' models, otherwise none of the legacy fixes get applied """
    config = legacy_config()
    legacy._update_layers([config])
    layers = flatten_layers(config)

    assert legacy._functional == {"encoder", "decoder_a", "decoder_b"}
    # Keras 3 SpatialDropout layers do not accept the generic layer kwargs
    dropout = layers["spatial_dropout2d_1"]["config"]
    assert "trainable" not in dropout
    assert "noise_shape" not in dropout
    assert dropout["rate"] == 0.25 and dropout["seed"] is None
    # Node indices for the shared encoder are decremented for Keras 3
    assert layers["decoder_a"]["inbound_nodes"] == [[["encoder", 0, 0, {}]]]
    assert layers["decoder_b"]["inbound_nodes"] == [[["encoder", 1, 0, {}]]]
    # Deprecated LeakyReLU kwarg is renamed
    assert layers["leaky_re_lu_1"]["config"]["negative_slope"] == 0.1
    assert "alpha" not in layers["leaky_re_lu_1"]["config"]


def test_updated_legacy_config_loads(legacy):
    """ The legacy config does not deserialize in Keras 3 as-is, but does once updated """
    with device("cpu"), pytest.raises(TypeError, match="trainable"):
        k_models.Model.from_config(legacy_config()["config"])

    config = legacy_config()
    legacy._update_layers([config])
    with device("cpu"):
        model = k_models.Model.from_config(config["config"])

    assert len(model.inputs) == 2
    assert len(model.outputs) == 2
    assert model.get_layer("encoder").get_layer("spatial_dropout2d_1").rate == 0.25
