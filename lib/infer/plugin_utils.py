#!/usr/env/bin/python3
"""General utility functions for Faceswap inference"""
from __future__ import annotations

import logging
import typing as T
from collections.abc import Iterable, Mapping
from threading import Event, Lock
from time import sleep

import cv2
import numpy as np
import torch

from lib.utils import get_module_objects

if T.TYPE_CHECKING:
    from plugins.extract.base import ExtractPlugin


logger = logging.getLogger(__name__)


def random_input_from_plugin(plugin: ExtractPlugin,
                             batch_size: int,
                             channels_last: bool) -> np.ndarray:
    """Obtain a random input array from a plugin's information for the given batch size

    Parameters
    ----------
    plugin
        The plugin to obtain the input array for
    batch_size : int
        The batch size for the input array
    channels_last : bool
        ``True`` if the data should be formatted channels last

    Returns
    -------
    A random input array in the correct format for the given plugin at the given batch size
    """
    size = plugin.input_size
    low, high = plugin.scale
    im_range = high - low
    retval = np.random.random((batch_size, 3, size, size)).astype(plugin.dtype) * im_range
    retval += low
    if channels_last:
        retval = retval.transpose(0, 2, 3, 1)
    return retval


def get_torch_modules(obj: T.Any,  # noqa[C901]  # pylint:disable=too-many-branches,too-many-return-statements
                      mod: str | None = None,
                      seen: set[int] | None = None,
                      results: list[torch.nn.Module] | None = None) -> list[torch.nn.Module]:
    """Recursively search a plugin's model attribute to find any parent :class:`torch.nn.Module`s

    Parameters
    ----------
    obj
        The object to check if it is a torch Module. This should be a plugin's `model` attribute
    mod
        The module that the parent model class belongs to. Default: ``None`` (Collected from the
        first object entered into the recursive function)
    seen
        A set of seen object IDs to prevent self-recursion. Default: ``None`` (Created when the
        first object enters the recursive function)
    results
        List of discovered torch modules. Default: ``None`` (Created when the first object enters
        the recursive function)

    Returns
    -------
    The list of discovered torch Modules
    """
    seen = set() if seen is None else seen
    retval: list[torch.nn.Module] = [] if results is None else results
    mod = obj.__class__.__module__ if mod is None else mod

    obj_id = id(obj)
    if obj_id in seen:
        return retval
    seen.add(obj_id)

    if isinstance(obj, torch.nn.Module):
        logger.debug("Torch module found in %s(%s)", obj.__class__.__name__, type(obj))
        retval.append(obj)
        return retval

    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        # Fast exit on primitive
        return retval

    if hasattr(obj, "__class__") and obj.__class__.__module__ not in (mod, "builtins"):
        # Never leave the plugin module
        return retval

    if isinstance(obj, Mapping):
        # Mapping before iterable as a mapping is also an iterable
        for v in obj.values():
            retval = get_torch_modules(v, mod, seen=seen, results=retval)

    if isinstance(obj, Iterable):
        for v in obj:
            retval = get_torch_modules(v, mod, seen=seen, results=retval)

    if hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            retval = get_torch_modules(v, mod, seen=seen, results=retval)
    return retval


def warmup_plugin(plugin: ExtractPlugin,  # noqa[C901]
                  batch_size: int,
                  channels_last: bool | None = None) -> bool | None:
    """Warm up a plugin that contains torch modules. If channels_last is ``None`` then attempt to
    send a channels first batch through. If it fails, send a channels last batch through

    Parameters
    ----------
    plugin
        The plugin to warmup
    batch_size
        The batch size to put through the model
    channels_last
        The expected channel order of the plugin or ``None`` to detect

    Returns
    -------
    bool
        ``True`` if the plugin is detected as channels last, ``False`` for channels first, ``None``
        for could not be detected
    """
    cv2_loglevel = None
    cv2_setlevel = None
    if channels_last is None:
        # cv2 outputs scary warnings when we are testing channels first/last with cv2-dnn plugins
        # so disable logging
        try:  # cv2 arbitrarily moves this based on build options :/
            cv2_loglevel = cv2.getLogLevel()  # type:ignore[attr-defined]
            cv2_setlevel = getattr(cv2, "setLogLevel")
        except AttributeError:
            try:
                cv2_loglevel = cv2.utils.logging.getLogLevel()  # type:ignore[attr-defined]
                cv2_setlevel = getattr(cv2.utils.logging, "setLogLevel")
            except AttributeError:
                pass

    chan_list = [False, True] if channels_last is None else [channels_last]
    is_chan_last = None

    if cv2_setlevel is not None:
        cv2_setlevel(0)

    for chan_last in chan_list:
        try:
            inp = random_input_from_plugin(plugin, batch_size, chan_last)
            plugin.process(inp)
            is_chan_last = chan_last
            break
        except Exception as err:  # pylint:disable=broad-except
            logger.debug("Exception with channels_last=%s: %s", chan_last, str(err).strip())

    if cv2_setlevel is not None:
        cv2_setlevel(cv2_loglevel)
    logger.debug("[%s] Warmed up. channels_last: %s", plugin.name, is_chan_last)
    return is_chan_last


_COMPILE_LOCK = Lock()
_COMPILE_LOGGED = Event()


def compile_models(plugin: ExtractPlugin, modules: list[torch.nn.Module]) -> None:
    """Compile any Torch modules in the plugin's `model` attribute

    Parameters
    ----------
    plugin
        The plugin containing Torch modules to be compiled
    modules
        The list of Torch modules contained within the plugin's `model` attribute
    """
    with _COMPILE_LOCK:
        if not _COMPILE_LOGGED.is_set():
            _COMPILE_LOGGED.set()
            sleep(0.5)  # Let other plugins log their output first
            logger.info("Compiling PyTorch models...")
        channels_last = warmup_plugin(plugin, 1)  # Make sure we don't trace on wrong channel order
        for mod in modules:
            logger.verbose("Compiling %s (%s)...",  # type:ignore[attr-defined]
                           plugin.name, mod.__class__.__name__)
            mod.compile(
                fullgraph=True,
                dynamic=False,  # We handle dynamic BS in code
                options={"triton.cudagraphs": True,  # Required to stop worker speed back to eager
                         "triton.cudagraph_trees": False,  # Optimize for static shapes
                         "triton.cudagraph_support_input_mutation": True,
                         "shape_padding": True,  # Pad tensors for Tensor core usage
                         "epilogue_fusion": True,
                         "coordinate_descent_tuning": True,  # Can sometimes find better kernels
                         "max_autotune": True,
                         "max_autotune_report_choices_stats": False})
        # Send the warmup batch here as we need to keep the lock when tracing
        warmup_plugin(plugin, plugin.batch_size, channels_last=channels_last)
    torch.cuda.empty_cache()  # Need to clear cache or we may run out of VRAM


__all__ = get_module_objects(__name__)
