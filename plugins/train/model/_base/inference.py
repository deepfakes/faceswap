#! /usr/env/bin/python3
""" Handles the recompilation of a Faceswap model into a version that can be used for inference """
from __future__ import annotations
import logging
import typing as T

import keras

from lib.logger import parse_class_init
from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from keras.src.ops import node

logger = logging.getLogger(__name__)


class Inference():
    """ Calculates required layers and compiles a saved model for inference.

    Parameters
    ----------
    saved_model: :class:`keras.Model`
        The saved trained Faceswap model
    switch_sides: bool
        ``True`` if the swap should be performed "B" > "A" ``False`` if the swap should be
        "A" > "B"
    """
    def __init__(self, saved_model: keras.Model, switch_sides: bool) -> None:
        logger.debug(parse_class_init(locals()))
        self._name = f"{saved_model.name}_inference"
        self._side_idx = 0 if switch_sides else 1

        self._input = self._get_input(saved_model)
        self._valid_layer_inputs = self._get_valid_layer_inputs(saved_model)
        self._output = self._get_output_layer(saved_model)
        self._filtered_layers = self._backwards_recurse(self._output)

        logger.debug("Initialized: %s", self.__class__.__name__)

    def _get_input(self, model: keras.models.Model) -> keras.KerasTensor:
        """Obtain the input to the model. We select the input for the side of the model we are
        swapping to as this maps correctly within valid_layer_inputs. The actual input does not
        matter, it is the layers within the model and what is input that dictate which weights will
        be loaded and if we are swapping.

        Parameters
        ----------
        layer
            The layer to obtain the inputs for

        Returns
        -------
        the input to the inference model
        """
        assert len(model.input) == 2, f"Unexpected input count: {len(model.input)} ({model.input})"
        input_tensor = model.input[self._side_idx]
        logger.debug("[Inference] '%s' model input for side index %s: '%s'",
                     model.name, self._side_idx, input_tensor.name)
        return input_tensor

    def _get_valid_inputs_for_layer(self, layer) -> list[keras.Layer]:
        """For the given layer obtain the inputs that can be valid for the given swap direction

        Parameters
        ----------
        layer
            The layer to obtain the inputs for

        Returns
        -------
        The list of potentially valid inputs. This will be either the inputs for the correct swap
        direction, if there are 2 potential inputs for the layer, or the sole inputs to the layer
        if it only has one input
        """
        inbound: list[node.Node] = layer._inbound_nodes  # pylint:disable=protected-access
        logger.debug("[Inference] '%s' inbound_nodes: %s", layer.name, inbound)
        tensors = [i.input_tensors for i in inbound]
        logger.debug("[Inference] '%s' input tensors: %s", layer.name, tensors)
        assert len(tensors) in (1, 2), f"Unexpected input tensor count: {len(tensors)}"
        side_tensors = tensors[self._side_idx] if len(tensors) == 2 else tensors[0]
        retval: list[keras.Layer] = [t._keras_history.operation  # pylint:disable=protected-access
                                     for t in side_tensors]
        logger.debug("[Inference] '%s' valid inputs: %s", layer.name, retval)
        return retval

    def _get_valid_layer_inputs(self, model: keras.models.Model
                                ) -> dict[keras.Layer, list[keras.Layer]]:
        """Obtain a dictionary of all layers within the model to a list of inputs that are
        potentially valid for the swap direction that is being performed

        Parameters
        ----------
        model
            The faceswap model that is to be converted for inference

        Returns
        -------
        A dictionary of all layers with in the model to a list of inputs that are potentially valid
        for the swap direction
        """
        retval = {layer: self._get_valid_inputs_for_layer(layer)
                  for layer in T.cast(list[keras.Layer], model.layers)
                  if not isinstance(layer, keras.layers.InputLayer)}
        logger.debug("[Inference] '%s' layer valid inputs for side index %s: %s",
                     model.name,
                     self._side_idx,
                     {k.name: [o.name for o in v] for k, v in retval.items()})
        return retval

    def _get_output_layer(self, model: keras.models.Model) -> keras.Layer:
        """Obtain the layer that acts as the output for the swap direction of the model

        Parameters
        ----------
        model
            The faceswap model that is to be converted for inference

        Returns
        -------
        The layer that acts as output for the model. This will either be a layer unique to the swap
        side, if split decoders, or the shared output layer if shared decoder
        """
        history: list[node.KerasHistory] = [t._keras_history  # pylint:disable=protected-access
                                            for t in model.output]
        logger.debug("[Inference] '%s' output history: %s", model.name, history)

        layers = [h.operation for h in history]
        outputs_count = len(layers)
        layer_count = len(set(o.name for o in layers))
        logger.debug("[Inference] '%s' outputs count: %s, output layer count: %s",
                     model.name, outputs_count, layer_count)
        assert layer_count in (1, 2), f"Unexpected output layers count: {layer_count}"

        if layer_count == 1:
            retval = layers[0]
        else:
            split = outputs_count // 2
            out_layers = layers[:split] if self._side_idx == 0 else layers[split:]
            out_layer_count = len(set(o.name for o in out_layers))
            assert out_layer_count == 1, f"Unexpected output layer count: {out_layer_count}"
            retval = out_layers[0]

        logger.debug("[Inference] '%s' output layer for side index %s: '%s'",
                     model.name, self._side_idx, retval.name)

        return retval

    def _backwards_recurse(self, layer: keras.Layer, seen: set[keras.Layer] | None = None
                           ) -> list[keras.Layer]:
        """Work backwards from the output to filter out layers that are not in the requested swap
        path and update to :attr:`_valid_layers`

        Returns
        -------
        A list of layers that exist within the requested swap path of the training model. Note:
        whilst the order is generally from last to first, due to multiple path splitting, order
        is not guaranteed
        """
        seen = set() if seen is None else seen
        if layer in seen:
            logger.debug("[Inference] Skipping seen layer '%s'.", layer.name)
            return []

        seen.add(layer)
        retval = [layer]

        if layer not in self._valid_layer_inputs:
            logger.debug("[Inference] No inputs for '%s'. Returning", layer.name)
            return retval

        next_layers = self._valid_layer_inputs[layer]
        logger.debug("[Inference] Got inputs for '%s': %s",
                     layer.name, [n.name for n in next_layers])

        for lyr in next_layers:
            retval.extend(self._backwards_recurse(lyr, seen=seen))

        logger.debug("[Inference] Final inputs for '%s': %s", layer.name, retval)
        return retval

    def __call__(self) -> keras.models.Model:
        """Obtain the inference model.

        Returns
        -------
        The built Keras inference model for the requested swap side
        """
        built = {self._input.name: self._input}
        to_build = {k: v for k, v in self._valid_layer_inputs.items()
                    if k in self._filtered_layers}
        logger.debug("[Inference] Building inference model from '%s' with layers %s",
                     self._input.name, [k.name for k in to_build])
        for layer, inputs in to_build.items():
            name = layer.name
            input_names = [i.name for i in inputs]
            logger.debug("[Inference] Building layer '%s' with inputs %s", name, input_names)
            assert all(i in built for i in input_names)
            input_layers = [built[n] for n in input_names]
            built[layer.name] = layer(input_layers if len(input_layers) > 1 else input_layers[0])

        output = built[self._output.name]
        retval = keras.Model(inputs=self._input, outputs=output, name=self._name)
        logger.debug("[Inference] Built model: %s", retval.name)
        return retval


__all__ = get_module_objects(__name__)
