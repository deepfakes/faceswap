#!/usr/bin/env python3
""" Base class for Faceswap :mod:`~plugins.extract.detect`, :mod:`~plugins.extract.align` and
:mod:`~plugins.extract.mask` Plugins
"""
import logging
import os
import sys

from tensorflow.python import errors_impl as tf_errors  # pylint:disable=no-name-in-module

from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.utils import GetModel, FaceswapError
from ._config import Config
from .pipeline import ExtractMedia

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# TODO CPU mode
# TODO Run with warnings mode


def _get_config(plugin_name, configfile=None):
    """ Return the configuration for the requested model

    Parameters
    ----------
    plugin_name: str
        The module name of the child plugin.
    configfile: str, optional
        Path to a :file:`./config/<plugin_type>.ini` file for this plugin. Default: use system
        configuration.

    Returns
    -------
    config_dict, dict
       A dictionary of configuration items from the configuration file
    """
    return Config(plugin_name, configfile=configfile).config_dict


class Extractor():
    """ Extractor Plugin Object

    All ``_base`` classes for Aligners, Detectors and Maskers inherit from this class.

    This class sets up a pipeline for working with ML plugins.

    Plugins are split into 3 threads, to utilize Numpy and CV2s parallel processing, as well as
    allow the predict function of the model to sit in a dedicated thread.
    A plugin is expected to have 3 core functions, each in their own thread:
    - :func:`process_input()` - Prepare the data for feeding into a model
    - :func:`predict` - Feed the data through the model
    - :func:`process_output()` - Perform any data post-processing

    Parameters
    ----------
    git_model_id: int
        The second digit in the github tag that identifies this model. See
        https://github.com/deepfakes-models/faceswap-models for more information
    model_filename: str
        The name of the model file to be loaded

    Other Parameters
    ----------------
    configfile: str, optional
        Path to a custom configuration ``ini`` file. Default: Use system configfile
    instance: int, optional
        If this plugin is being executed multiple times (i.e. multiple pipelines have been
        launched), the instance of the plugin must be passed in for naming convention reasons.
        Default: 0

    The following attributes should be set in the plugin's :func:`__init__` method after
    initializing the parent.

    Attributes
    ----------
    name: str
        Name of this plugin. Used for display purposes.
    input_size: int
        The input size to the model in pixels across one edge. The input size should always be
        square.
    color_format: str
        Color format for model. Must be ``'BGR'``, ``'RGB'`` or ``'GRAY'``. Defaults to ``'BGR'``
        if not explicitly set.
    vram: int
        Approximate VRAM used by the model at :attr:`input_size`. Used to calculate the
        :attr:`batchsize`. Be conservative to avoid OOM.
    vram_warnings: int
        Approximate VRAM used by the model at :attr:`input_size` that will still run, but generates
        warnings. Used to calculate the :attr:`batchsize`. Be conservative to avoid OOM.
    vram_per_batch: int
        Approximate additional VRAM used by the model for each additional batch. Used to calculate
        the :attr:`batchsize`. Be conservative to avoid OOM.

    See Also
    --------
    plugins.extract.detect._base : Detector parent class for extraction plugins.
    plugins.extract.align._base : Aligner parent class for extraction plugins.
    plugins.extract.mask._base : Masker parent class for extraction plugins.
    plugins.extract.pipeline : The extract pipeline that configures and calls all plugins

    """
    def __init__(self, git_model_id=None, model_filename=None, configfile=None, instance=0):
        logger.debug("Initializing %s: (git_model_id: %s, model_filename: %s, instance: %s, "
                     "configfile: %s, )", self.__class__.__name__, git_model_id, model_filename,
                     instance, configfile)

        self._instance = instance
        self.config = _get_config(".".join(self.__module__.split(".")[-2:]), configfile=configfile)
        """ dict: Config for this plugin, loaded from ``extract.ini`` configfile """

        self.model_path = self._get_model(git_model_id, model_filename)
        """ str or list: Path to the model file(s) (if required). Multiple model files should
        be a list of strings """

        # << SET THE FOLLOWING IN PLUGINS __init__ IF DIFFERENT FROM DEFAULT >> #
        self.name = None
        self.input_size = None
        self.color_format = "BGR"
        self.vram = None
        self.vram_warnings = None  # Will run at this with warnings
        self.vram_per_batch = None

        # << THE FOLLOWING ARE SET IN self.initialize METHOD >> #
        self.queue_size = 1
        """ int: Queue size for all internal queues. Set in :func:`initialize()` """

        self.model = None
        """varies: The model for this plugin.
        Set in the plugin's :func:`init_model()` method """

        # For detectors that support batching, this should be set to  the calculated batch size
        # that the amount of available VRAM will support.
        self.batchsize = 1
        """ int: Batchsize for feeding this model. The number of images the model should
        feed through at once. """

        self._queues = dict()
        """ dict: in + out queues and internal queues for this plugin, """

        self._threads = []
        """ list: Internal threads for this plugin """

        self._extract_media = dict()
        """ dict: The :class:`plugins.extract.pipeline.ExtractMedia` objects currently being
        processed. Stored at input for pairing back up on output of extractor process """

        # << THE FOLLOWING PROTECTED ATTRIBUTES ARE SET IN PLUGIN TYPE _base.py >>> #
        self._plugin_type = None
        """ str: Plugin type. ``detect`` or ``align``
        set in ``<plugin_type>._base`` """

        logger.debug("Initialized _base %s", self.__class__.__name__)

    # <<< OVERIDABLE METHODS >>> #
    def init_model(self):
        """ **Override method**

        Override this method to execute the specific model initialization method """
        raise NotImplementedError

    def process_input(self, batch):
        """ **Override method**

        Override this method for specific extractor pre-processing of image

        Parameters
        ----------
        batch : dict
            Contains the batch that is currently being passed through the plugin process

        Notes
        -----
        When preparing an input to the model a key ``feed`` must be added
        to the :attr:`batch` ``dict`` which contains this input.
        """
        raise NotImplementedError

    def predict(self, batch):
        """ **Override method**

        Override this method for specific extractor model prediction function

        Parameters
        ----------
        batch : dict
            Contains the batch that is currently being passed through the plugin process

        Notes
        -----
        Input for :func:`predict` should have been set in :func:`process_input` with the addition
        of a ``feed`` key to the :attr:`batch` ``dict``.

        Output from the model should add the key ``prediction`` to the :attr:`batch` ``dict``.

        For Detect:
            the expected output for the ``prediction`` key of the :attr:`batch` dict should be a
            ``list`` of :attr:`batchsize` of detected face points. These points should be either
            a ``list``, ``tuple`` or ``numpy.ndarray`` with the first 4 items being the `left`,
            `top`, `right`, `bottom` points, in that order
        """
        raise NotImplementedError

    def process_output(self, batch):
        """ **Override method**

        Override this method for specific extractor model post predict function

        Parameters
        ----------
        batch : dict
            Contains the batch that is currently being passed through the plugin process

        Notes
        -----
        For Align:
            The key ``landmarks`` must be returned in the :attr:`batch` ``dict`` from this method.
            This should be a ``list`` or ``numpy.ndarray`` of :attr:`batchsize` containing a
            ``list``, ``tuple`` or ``numpy.ndarray`` of `(x, y)` coordinates of the 68 point
            landmarks as calculated from the :attr:`model`.
        """
        raise NotImplementedError

    def _predict(self, batch):
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        ``plugins.extract.detect._base`` or ``plugins.extract.align._base``) and should not
        be overridden within plugins themselves.

        It acts as a wrapper for the plugin's ``self.predict`` method and handles any
        predict processing that is consistent for all plugins within the `plugin_type`

        Parameters
        ----------
        batch : dict
            Contains the batch that is currently being passed through the plugin process
        """
        raise NotImplementedError

    def finalize(self, batch):
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        :mod:`plugins.extract.detect._base`, :mod:`plugins.extract.align._base` or
        :mod:`plugins.extract.mask._base`) and should not be overridden within plugins themselves.

        Handles consistent finalization for all plugins that exist within that plugin type. Its
        input is always the output from :func:`process_output()`

        Parameters
        ----------
        batch : dict
            Contains the batch that is currently being passed through the plugin process

        """

    def get_batch(self, queue):
        """ **Override method** (at `<plugin_type>` level)

        This method should be overridden at the `<plugin_type>` level (IE.
        :mod:`plugins.extract.detect._base`, :mod:`plugins.extract.align._base` or
        :mod:`plugins.extract.mask._base`) and should not be overridden within plugins themselves.

        Get :class:`~plugins.extract.pipeline.ExtractMedia` items from the queue in batches of
        :attr:`batchsize`

        Parameters
        ----------
        queue : queue.Queue()
            The ``queue`` that the batch will be fed from. This will be the input to the plugin.
        """
        raise NotImplementedError

    # <<< THREADING METHODS >>> #
    def start(self):
        """ Start all threads

        Exposed for :mod:`~plugins.extract.pipeline` to start plugin's threads
        """
        for thread in self._threads:
            thread.start()

    def join(self):
        """ Join all threads

        Exposed for :mod:`~plugins.extract.pipeline` to join plugin's threads
        """
        for thread in self._threads:
            thread.join()
            del thread

    def check_and_raise_error(self):
        """ Check all threads for errors

        Exposed for :mod:`~plugins.extract.pipeline` to check plugin's threads for errors
        """
        for thread in self._threads:
            err = thread.check_and_raise_error()
            if err is not None:
                logger.debug("thread_error_detected")
                return True
        return False

    # <<< PROTECTED ACCESS METHODS >>> #
    # <<< INIT METHODS >>> #
    def _get_model(self, git_model_id, model_filename):
        """ Check if model is available, if not, download and unzip it """
        if model_filename is None:
            logger.debug("No model_filename specified. Returning None")
            return None
        if git_model_id is None:
            logger.debug("No git_model_id specified. Returning None")
            return None
        plugin_path = os.path.join(*self.__module__.split(".")[:-1])
        if os.path.basename(plugin_path) in ("detect", "align", "mask", "recognition"):
            base_path = os.path.dirname(os.path.realpath(sys.argv[0]))
            cache_path = os.path.join(base_path, plugin_path, ".cache")
        else:
            cache_path = os.path.join(os.path.dirname(__file__), ".cache")
        model = GetModel(model_filename, cache_path, git_model_id)
        return model.model_path

    # <<< PLUGIN INITIALIZATION >>> #
    def initialize(self, *args, **kwargs):
        """ Initialize the extractor plugin

            Should be called from :mod:`~plugins.extract.pipeline`
        """
        logger.debug("initialize %s: (args: %s, kwargs: %s)",
                     self.__class__.__name__, args, kwargs)
        logger.info("Initializing %s (%s)...", self.name, self._plugin_type.title())
        self.queue_size = 1
        name = self.name.replace(" ", "_").lower()
        self._add_queues(kwargs["in_queue"],
                         kwargs["out_queue"],
                         ["predict_{}".format(name), "post_{}".format(name)])
        self._compile_threads()
        try:
            self.init_model()
        except tf_errors.UnknownError as err:
            if "failed to get convolution algorithm" in str(err).lower():
                msg = ("Tensorflow raised an unknown error. This is most likely caused by a "
                       "failure to launch cuDNN which can occur for some GPU/Tensorflow "
                       "combinations. You should enable `allow_growth` to attempt to resolve this "
                       "issue:"
                       "\nGUI: Go to Settings > Extract Plugins > Global and enable the "
                       "`allow_growth` option."
                       "\nCLI: Go to `faceswap/config/extract.ini` and change the `allow_growth "
                       "option to `True`.")
                raise FaceswapError(msg) from err
            raise err
        logger.info("Initialized %s (%s) with batchsize of %s",
                    self.name, self._plugin_type.title(), self.batchsize)

    def _add_queues(self, in_queue, out_queue, queues):
        """ Add the queues
            in_queue and out_queue should be previously created queue manager queues.
            queues should be a list of queue names """
        self._queues["in"] = in_queue
        self._queues["out"] = out_queue
        for q_name in queues:
            self._queues[q_name] = queue_manager.get_queue(
                name="{}{}_{}".format(self._plugin_type, self._instance, q_name),
                maxsize=self.queue_size)

    # <<< THREAD METHODS >>> #
    def _compile_threads(self):
        """ Compile the threads into self._threads list """
        logger.debug("Compiling %s threads", self._plugin_type)
        name = self.name.replace(" ", "_").lower()
        base_name = "{}_{}".format(self._plugin_type, name)
        self._add_thread("{}_input".format(base_name),
                         self.process_input,
                         self._queues["in"],
                         self._queues["predict_{}".format(name)])
        self._add_thread("{}_predict".format(base_name),
                         self._predict,
                         self._queues["predict_{}".format(name)],
                         self._queues["post_{}".format(name)])
        self._add_thread("{}_output".format(base_name),
                         self.process_output,
                         self._queues["post_{}".format(name)],
                         self._queues["out"])
        logger.debug("Compiled %s threads: %s", self._plugin_type, self._threads)

    def _add_thread(self, name, function, in_queue, out_queue):
        """ Add a MultiThread thread to self._threads """
        logger.debug("Adding thread: (name: %s, function: %s, in_queue: %s, out_queue: %s)",
                     name, function, in_queue, out_queue)
        self._threads.append(MultiThread(target=self._thread_process,
                                         name=name,
                                         function=function,
                                         in_queue=in_queue,
                                         out_queue=out_queue))
        logger.debug("Added thread: %s", name)

    def _thread_process(self, function, in_queue, out_queue):
        """ Perform a plugin function in a thread """
        func_name = function.__name__
        logger.debug("threading: (function: '%s')", func_name)
        while True:
            if func_name == "process_input":
                # Process input items to batches
                exhausted, batch = self.get_batch(in_queue)
                if exhausted:
                    if batch:
                        # Put the final batch
                        batch = function(batch)
                        out_queue.put(batch)
                    break
            else:
                batch = self._get_item(in_queue)
                if batch == "EOF":
                    break
            try:
                batch = function(batch)
            except tf_errors.UnknownError as err:
                if "failed to get convolution algorithm" in str(err).lower():
                    msg = ("Tensorflow raised an unknown error. This is most likely caused by a "
                           "failure to launch cuDNN which can occur for some GPU/Tensorflow "
                           "combinations. You should enable `allow_growth` to attempt to resolve "
                           "this issue:"
                           "\nGUI: Go to Settings > Extract Plugins > Global and enable the "
                           "`allow_growth` option."
                           "\nCLI: Go to `faceswap/config/extract.ini` and change the "
                           "`allow_growth option to `True`.")
                    raise FaceswapError(msg) from err
                raise err
            if func_name == "process_output":
                # Process output items to individual items from batch
                for item in self.finalize(batch):
                    out_queue.put(item)
            else:
                out_queue.put(batch)
        logger.debug("Putting EOF")
        out_queue.put("EOF")

    # <<< QUEUE METHODS >>> #
    def _get_item(self, queue):
        """ Yield one item from a queue """
        item = queue.get()
        if isinstance(item, ExtractMedia):
            logger.trace("filename: '%s', image shape: %s, detected_faces: %s, queue: %s, "
                         "item: %s",
                         item.filename, item.image_shape, item.detected_faces, queue, item)
            self._extract_media[item.filename] = item
        else:
            logger.trace("item: %s, queue: %s", item, queue)
        return item

    @staticmethod
    def _dict_lists_to_list_dicts(dictionary):
        """ Convert a dictionary of lists to a list of dictionaries """
        return [dict(zip(dictionary, val)) for val in zip(*dictionary.values())]
