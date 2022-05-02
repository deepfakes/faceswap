#!/usr/bin/env python3
""" Handles the loading and collation of events from Tensorflow event log files. """

import logging
import os
import zlib

import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2  # pylint:disable=no-name-in-module
from tensorflow.python.framework import (  # pylint:disable=no-name-in-module
    errors_impl as tf_errors)

from lib.serializer import get_serializer
from lib.utils import get_backend

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class _LogFiles():
    """ Holds the filenames of the Tensorflow Event logs that require parsing.

    Parameters
    ----------
    logs_folder: str
        The folder that contains the Tensorboard log files
    """
    def __init__(self, logs_folder):
        logger.debug("Initializing: %s: (logs_folder: '%s')", self.__class__.__name__, logs_folder)
        self._logs_folder = logs_folder
        self._filenames = self._get_log_filenames()
        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def session_ids(self):
        """ list: Sorted list of `ints` of available session ids. """
        return list(sorted(self._filenames))

    def _get_log_filenames(self):
        """ Get the Tensorflow event filenames for all existing sessions.

        Returns
        -------
        dict
            The full path of each log file for each training session id that has been run
        """
        logger.debug("Loading log filenames. base_dir: '%s'", self._logs_folder)
        retval = {}
        for dirpath, _, filenames in os.walk(self._logs_folder):
            if not any(filename.startswith("events.out.tfevents") for filename in filenames):
                continue
            session_id = self._get_session_id(dirpath)
            if session_id is None:
                logger.warning("Unable to load session data for model")
                return retval
            retval[session_id] = self._get_log_filename(dirpath, filenames)
        logger.debug("logfiles: %s", retval)
        return retval

    @classmethod
    def _get_session_id(cls, folder):
        """ Obtain the session id for the given folder.

        Parameters
        ----------
        folder: str
            The full path to the folder that contains the session's Tensorflow Event Log

        Returns
        -------
        int
            The session ID for the given folder. If no session id can be determined, return
            ``None``
        """
        session = os.path.split(os.path.split(folder)[0])[1]
        session_id = session[session.rfind("_") + 1:]
        retval = None if not session_id.isdigit() else int(session_id)
        logger.debug("folder: '%s', session_id: %s", folder, retval)
        return retval

    @classmethod
    def _get_log_filename(cls, folder, filenames):
        """ Obtain the session log file for the given folder. If multiple log files exist for the
        given folder, then the most recent log file is used, as earlier files are assumed to be
        obsolete.

        Parameters
        ----------
        folder: str
            The full path to the folder that contains the session's Tensorflow Event Log
        filenames: list
            List of filenames that exist within the given folder

        Returns
        -------
        str
            The full path the the selected log file
        """
        logfiles = [fname for fname in filenames if fname.startswith("events.out.tfevents")]
        retval = os.path.join(folder, sorted(logfiles)[-1])  # Take last item if multi matches
        logger.debug("logfiles: %s, selected: '%s'", logfiles, retval)
        return retval

    def refresh(self):
        """ Refresh the list of log filenames. """
        logger.debug("Refreshing log filenames")
        self._filenames = self._get_log_filenames()

    def get(self, session_id):
        """ Obtain the log filename for the given session id.

        Parameters
        ----------
        session_id: int
            The session id to obtain the log filename for

        Returns
        -------
        str
            The full path to the log file for the requested session id
        """
        retval = self._filenames.get(session_id)
        logger.debug("session_id: %s, log_filename: '%s'", session_id, retval)
        return retval


class _Cache():
    """ Holds parsed Tensorflow log event data in a compressed cache in memory.

    Parameters
    ----------
    session_ids: list
        List of `ints` pertaining to the session ids that exist in the Tensorflow events folder
    """
    def __init__(self, session_ids):
        logger.debug("Initializing: %s: (session_ids: %s)", self.__class__.__name__, session_ids)
        self._data = {idx: None for idx in session_ids}
        self._carry_over = {}
        self._loss_labels = []
        logger.debug("Initialized: %s", self.__class__.__name__)

    def is_cached(self, session_id):
        """ bool: ``True`` if the data already exists in the cache otherwise ``False``. """
        return self._data.get(session_id) is not None

    def cache_data(self, session_id, data, labels, is_live=False):
        """ Add a full session's worth of event data to :attr:`_data`.

        Parameters
        ----------
        session_id: int
            The session id to add the data for
        data: dict
            The extracted event data dictionary generated from :class:`_EventParser`
        labels: list
            List of `str` for the labels of each loss value output
        is_live: bool, optional
            ``True`` if the data to be cached is from a live training session otherwise ``False``.
            Default: ``False``
        """
        logger.debug("Caching event data: (session_id: %s, labels: %s, data points: %s, "
                     "is_live: %s)", session_id, labels, len(data), is_live)

        if labels:
            logger.debug("Setting loss labels: %s", labels)
            self._loss_labels = labels

        if not data:
            logger.debug("No data to cache")
            return

        timestamps, loss = self._to_numpy(data, is_live)

        if not is_live or (is_live and not self._data.get(session_id, None)):
            self._data[session_id] = dict(labels=self._loss_labels,
                                          loss=zlib.compress(loss),
                                          loss_shape=loss.shape,
                                          timestamps=zlib.compress(timestamps),
                                          timestamps_shape=timestamps.shape)
        else:
            self._add_latest_live(session_id, loss, timestamps)

    def _to_numpy(self, data, is_live):
        """ Extract each individual step data into separate numpy arrays for loss and timestamps.

        Timestamps are stored float64 as the extra accuracy is needed for correct timings. Arrays
        are returned at the length of the shortest available data (i.e. truncated records are
        dropped)

        Parameters
        ----------
        data: dict
            The incoming tensorflow event data in dictionary form per step
        is_live: bool, optional
            ``True`` if the data to be cached is from a live training session otherwise ``False``.
            Default: ``False``

        Returns
        -------
        timestamps: :class:`numpy.ndarray`
            float64 array of all iteration's timestamps
        loss: :class:`numpy.ndarray`
            float32 array of all iteration's loss
        """
        if is_live and self._carry_over:
            logger.debug("Processing carry over: %s", self._carry_over)
            self._collect_carry_over(data)

        times, loss = zip(*[(data[idx].get("timestamp"), data[idx].get("loss", []))
                            for idx in sorted(data)])
        times, loss = self._process_data(data, times, loss, is_live)

        if is_live and not all(len(val) == len(self._loss_labels) for val in loss):
            # TODO Many attempts have been made to fix this for live graph logging, and the issue
            # of non-consistent loss record sizes keeps coming up. In the meantime we shall swallow
            # any loss values that are of incorrect length so graph remains functional. This will,
            # most likely, lead to a mismatch on iteration count so a proper fix should be
            # implemented.

            # Timestamps and loss appears to remain consistent with each other, but sometimes loss
            # appears non-consistent. eg (lengths):
            # [2, 2, 2, 2, 2, 2, 2, 0] - last loss collection has zero length
            # [1, 2, 2, 2, 2, 2, 2, 2] - 1st loss collection has 1 length
            # [2, 2, 2, 3, 2, 2, 2] - 4th loss collection has 3 length

            logger.debug("Inconsistent loss found in collection: %s", loss)
            for idx in reversed(range(len(loss))):
                if len(loss[idx]) != len(self._loss_labels):
                    logger.debug("Removing loss/timestamps at position %s", idx)
                    del loss[idx]
                    del times[idx]

        times, loss = (np.array(times, dtype="float64"), np.array(loss, dtype="float32"))
        logger.debug("Converted to numpy: (data points: %s, timestamps shape: %s, loss shape: %s)",
                     len(data), times.shape, loss.shape)

        return times, loss

    def _collect_carry_over(self, data):
        """ For live data, collect carried over data from the previous update and merge into the
        current data dictionary.

        Parameters
        ----------
        data: dict
            The latest raw data dictionary
        """
        logger.debug("Carry over keys: %s, data keys: %s", list(self._carry_over), list(data))
        for key in list(self._carry_over):
            if key not in data:
                logger.debug("Carry over found for item %s which does not exist in current "
                             "data: %s. Skipping.", key, list(data))
                continue
            carry_over = self._carry_over.pop(key)
            update = data[key]
            logger.debug("Merging carry over data: %s in to %s", carry_over, update)
            timestamp = update.get("timestamp")
            update["timestamp"] = carry_over["timestamp"] if timestamp is None else timestamp
            update.setdefault("loss", []).extend(carry_over.get("loss", []))
            logger.debug("Merged carry over data: %s", update)

    def _process_data(self, data, timestamps, loss, is_live):
        """ Process live update data.

        Live data requires different processing as often we will only have partial data for the
        current step, so we need to cache carried over partial data to be picked up at the next
        query. In addition to this, if training is unexpectedly interrupted, there may also be
        partial data which needs to be cleansed prior to creating a numpy array

        Parameters
        ----------
        data: dict
            The incoming tensorflow event data in dictionary form per step
        timestamps: tuple
            The raw timestamps for for the latest live query, including any partial reads
        loss: tuple
            The raw loss for for the latest live query, including any partial reads
        is_live: bool
            ``True`` if the data to be cached is from a live training session otherwise ``False``.

        Returns
        -------
        timestamps: tuple
            Cleaned list of complete timestamps for the latest live query
        loss: list
            Cleaned list of complete loss for the latest live query
        """
        loss = list(loss)
        timestamps = list(timestamps)

        if len(loss[-1]) != len(self._loss_labels):
            logger.debug("Truncated loss found. loss count: %s", len(loss))
            idx = sorted(data)[-1]
            if is_live:
                logger.debug("Setting carried over data: %s", data[idx])
                self._carry_over[idx] = data[idx]
            logger.debug("Removing truncated loss: (timestamp: %s, loss: %s)",
                         timestamps[-1], loss[-1])
            del loss[-1]
            del timestamps[-1]

        return timestamps, loss

    def _add_latest_live(self, session_id, loss, timestamps):
        """ Append the latest received live training data to the cached data.

        Parameters
        ----------
        session_id: int
            The training session ID to update the cache for
        loss: :class:`numpy.ndarray`
            The latest loss values returned from the iterator
        timestamps: :class:`numpy.ndarray`
            The latest time stamps returned from the iterator
        """
        logger.debug("Adding live data to cache: (session_id: %s, loss: %s, timestamps: %s)",
                     session_id, loss.shape, timestamps.shape)
        if not np.any(loss) and not np.any(timestamps):
            return

        cache = self._data[session_id]
        for metric in ("loss", "timestamps"):
            data = locals()[metric]
            old_shape = cache[f"{metric}_shape"]
            dtype = "float32" if metric == "loss" else "float64"
            old = np.frombuffer(zlib.decompress(cache[metric]), dtype=dtype).reshape(old_shape)
            new = np.concatenate((old, data))
            logger.debug("'%s' old_shape: %s new_shape: %s", metric, old_shape, new.shape)
            cache[f"{metric}_shape"] = new.shape
            cache[metric] = zlib.compress(new)
            del old

    def get_data(self, session_id, metric):
        """ Retrieve the decompressed cached data from the cache for the given session id.

        Parameters
        ----------
        session_id: int or ``None``
            If session_id is provided, then the cached data for that session is returned. If
            session_id is ``None`` then the cached data for all sessions is returned
        metric: ['loss', 'timestamps']
            The metric to return the data for.

        Returns
        -------
        dict
            The `session_id` (s) as key, the values are a dictionary containing the requested
            metric information for each session returned
        """
        if session_id is None:
            raw = self._data
        else:
            data = self._data.get(session_id)
            if not data:
                return None
            raw = {session_id: data}

        dtype = "float32" if metric == "loss" else "float64"

        retval = {}
        for idx, data in raw.items():
            val = {metric: np.frombuffer(zlib.decompress(data[metric]),
                                         dtype=dtype).reshape(data[f"{metric}_shape"])}
            if metric == "loss":
                val["labels"] = data["labels"]
            retval[idx] = val

        logger.debug("Obtained cached data: %s",
                     {session_id: {k: v.shape if isinstance(v, np.ndarray) else v
                                   for k, v in data.items()}
                      for session_id, data in retval.items()})
        return retval


class TensorBoardLogs():
    """ Parse data from TensorBoard logs.

    Process the input logs folder and stores the individual filenames per session.

    Caches timestamp and loss data on request and returns this data from the cache.

    Parameters
    ----------
    logs_folder: str
        The folder that contains the Tensorboard log files
    is_training: bool
        ``True`` if the events are being read whilst Faceswap is training otherwise ``False``
    """
    def __init__(self, logs_folder, is_training):
        logger.debug("Initializing: %s: (logs_folder: %s, is_training: %s)",
                     self.__class__.__name__, logs_folder, is_training)
        self._is_training = False
        self._training_iterator = None

        self._log_files = _LogFiles(logs_folder)
        self.set_training(is_training)

        self._cache = _Cache(self.session_ids)

        logger.debug("Initialized: %s", self.__class__.__name__)

    @property
    def session_ids(self):
        """ list: Sorted list of integers of available session ids. """
        return self._log_files.session_ids

    def set_training(self, is_training):
        """ Set the internal training flag to the given `is_training` value.

        If a new training session is being instigated, refresh the log filenames

        Parameters
        ----------
        is_training: bool
            ``True`` to indicate that the logs to be read are from the currently training
            session otherwise ``False``
        """
        if self._is_training == is_training:
            logger.debug("Training flag already set to %s. Returning", is_training)
            return

        logger.debug("Setting is_training to %s", is_training)
        self._is_training = is_training
        if is_training:
            self._log_files.refresh()
            log_file = self._log_files.get(self.session_ids[-1])
            logger.debug("Setting training iterator for log file: '%s'", log_file)
            self._training_iterator = tf.compat.v1.io.tf_record_iterator(log_file)
        else:
            logger.debug("Removing training iterator")
            del self._training_iterator
            self._training_iterator = None

    def _cache_data(self, session_id):
        """ Cache TensorBoard logs for the given session ID on first access.

        Populates :attr:`_cache` with timestamps and loss data.

        If this is a training session and the data is being queried for the training session ID
        then get the latest available data and append to the cache

        Parameters
        -------
        session_id: int
            The session ID to cache the data for
        """
        live_data = self._is_training and session_id == max(self.session_ids)
        iterator = self._training_iterator if live_data else tf.compat.v1.io.tf_record_iterator(
            self._log_files.get(session_id))
        parser = _EventParser(iterator, self._cache, live_data)
        parser.cache_events(session_id)

    def _check_cache(self, session_id=None):
        """ Check if the given session_id has been cached and if not, cache it.

        Parameters
        ----------
        session_id: int, optional
            The Session ID to return the data for. Set to ``None`` to return all session
            data. Default ``None`
        """
        if session_id is not None and not self._cache.is_cached(session_id):
            self._cache_data(session_id)
        elif self._is_training and session_id == self.session_ids[-1]:
            self._cache_data(session_id)
        elif session_id is None:
            for idx in self.session_ids:
                if not self._cache.is_cached(idx):
                    self._cache_data(idx)

    def get_loss(self, session_id=None):
        """ Read the loss from the TensorBoard event logs

        Parameters
        ----------
        session_id: int, optional
            The Session ID to return the loss for. Set to ``None`` to return all session
            losses. Default ``None``

        Returns
        -------
        dict
            The session id(s) as key, with a further dictionary as value containing the loss name
            and list of loss values for each step
        """
        logger.debug("Getting loss: (session_id: %s)", session_id)
        retval = {}
        for idx in [session_id] if session_id else self.session_ids:
            self._check_cache(idx)
            data = self._cache.get_data(idx, "loss")
            if not data:
                continue
            data = data[idx]
            retval[idx] = {title: data["loss"][:, idx] for idx, title in enumerate(data["labels"])}
        logger.debug({key: {k: v.shape for k, v in val.items()}
                      for key, val in retval.items()})
        return retval

    def get_timestamps(self, session_id=None):
        """ Read the timestamps from the TensorBoard logs.

        As loss timestamps are slightly different for each loss, we collect the timestamp from the
        `batch_loss` key.

        Parameters
        ----------
        session_id: int, optional
            The Session ID to return the timestamps for. Set to ``None`` to return all session
            timestamps. Default ``None``

        Returns
        -------
        dict
            The session id(s) as key with list of timestamps per step as value
        """

        logger.debug("Getting timestamps: (session_id: %s, is_training: %s)",
                     session_id, self._is_training)
        retval = {}
        for idx in [session_id] if session_id else self.session_ids:
            self._check_cache(idx)
            data = self._cache.get_data(idx, "timestamps")
            if not data:
                continue
            retval[idx] = data[idx]["timestamps"]
        logger.debug({k: v.shape for k, v in retval.items()})
        return retval


class _EventParser():  # pylint:disable=too-few-public-methods
    """ Parses Tensorflow event and populates data to :class:`_Cache`.

    Parameters
    ----------
    iterator: :func:`tf.compat.v1.io.tf_record_iterator`
        The iterator to use for reading Tensorflow event logs
    cache: :class:`_Cache`
        The cache object to store the collected parsed events to
    live_data: bool
        ``True`` if the iterator to be loaded is a training iterator for reading live data
        otherwise ``False``
    """
    def __init__(self, iterator, cache, live_data):
        logger.debug("Initializing: %s: (iterator: %s, cache: %s, live_data: %s)",
                     self.__class__.__name__, iterator, cache, live_data)
        self._live_data = live_data
        self._cache = cache
        self._iterator = self._get_latest_live(iterator) if live_data else iterator
        self._loss_labels = []
        logger.debug("Initialized: %s", self.__class__.__name__)

    @classmethod
    def _get_latest_live(cls, iterator):
        """ Obtain the latest event logs for live training data.

        The live data iterator remains open so that it can be re-queried

        Parameters
        ----------
        iterator: :func:`tf.compat.v1.io.tf_record_iterator`
            The live training iterator to use for reading Tensorflow event logs

        Yields
        ------
        dict
            A Tensorflow event in dictionary form for a single step
        """
        i = 0
        while True:
            try:
                yield next(iterator)
                i += 1
            except StopIteration:
                logger.debug("End of data reached")
                break
            except tf.errors.DataLossError as err:
                # Truncated records are ignored. The iterator holds the offset, so the record will
                # be completed at the next call.
                logger.debug("Truncated record. Original Error: %s", err)
                break
        logger.debug("Collected %s records from live log file", i)

    def cache_events(self, session_id):
        """ Parse the Tensorflow events logs and add to :attr:`_cache`.

        Parameters
        ----------
        session_id: int
            The session id that the data is being cached for
        """
        data = {}
        try:
            for record in self._iterator:
                event = event_pb2.Event.FromString(record)  # pylint:disable=no-member
                if not event.summary.value:
                    continue
                if event.summary.value[0].tag == "keras":
                    self._parse_outputs(event)
                if get_backend() == "amd":
                    # No model is logged for AMD so need to get loss labels from state file
                    self._add_amd_loss_labels(session_id)
                if event.summary.value[0].tag.startswith("batch_"):
                    data[event.step] = self._process_event(event, data.get(event.step, {}))

        except tf_errors.DataLossError as err:
            logger.warning("The logs for Session %s are corrupted and cannot be displayed. "
                           "The totals do not include this session. Original error message: "
                           "'%s'", session_id, str(err))

        self._cache.cache_data(session_id, data, self._loss_labels, is_live=self._live_data)

    def _parse_outputs(self, event):
        """ Parse the outputs from the stored model structure for mapping loss names to
        model outputs.

        Loss names are added to :attr:`_loss_labels`

        Notes
        -----
        The master model does not actually contain the specified output name, so we dig into the
        sub-model to obtain the name of the output layers

        Parameters
        ----------
        event: :class:`tensorflow.core.util.event_pb2`
            The event data containing the keras model structure to be parsed
        """
        serializer = get_serializer("json")
        struct = event.summary.value[0].tensor.string_val[0]

        config = serializer.unmarshal(struct)["config"]
        model_outputs = self._get_outputs(config)

        for side_outputs, side in zip(model_outputs, ("a", "b")):
            logger.debug("side: '%s', outputs: '%s'", side, side_outputs)
            layer_name = side_outputs[0][0]

            output_config = next(layer for layer in config["layers"]
                                 if layer["name"] == layer_name)["config"]
            layer_outputs = self._get_outputs(output_config)
            for output in layer_outputs:  # Drill into sub-model to get the actual output names
                loss_name = output[0][0]
                if loss_name[-2:] not in ("_a", "_b"):  # Rename losses to reflect the side output
                    new_name = f"{loss_name.replace('_both', '')}_{side}"
                    logger.debug("Renaming loss output from '%s' to '%s'", loss_name, new_name)
                    loss_name = new_name
                if loss_name not in self._loss_labels:
                    logger.debug("Adding loss name: '%s'", loss_name)
                    self._loss_labels.append(loss_name)
        logger.debug("Collated loss labels: %s", self._loss_labels)

    @classmethod
    def _get_outputs(cls, model_config):
        """ Obtain the output names, instance index and output index for the given model.

        If there is only a single output, the shape of the array is expanded to remain consistent
        with multi model outputs

        Parameters
        ----------
        model_config: dict
            The saved Keras model configuration dictionary

        Returns
        -------
        :class:`numpy.ndarray`
            The layer output names, their instance index and their output index
        """
        outputs = np.array(model_config["output_layers"])
        logger.debug("Obtained model outputs: %s, shape: %s", outputs, outputs.shape)
        if outputs.ndim == 2:  # Insert extra dimension for non learn mask models
            outputs = np.expand_dims(outputs, axis=1)
            logger.debug("Expanded dimensions for single output model. outputs: %s, shape: %s",
                         outputs, outputs.shape)
        return outputs

    def _add_amd_loss_labels(self, session_id):
        """ It is not possible to store the model config in the Tensorboard logs for AMD so we
        need to obtain the loss labels from the model's state file. This is called now so we know
        event data is being written, and therefore the most current loss label data is available
        in the state file.

        Loss names are added to :attr:`_loss_labels`

        Parameters
        ----------
        session_id: int
            The session id that the data is being cached for

        """
        if self._cache._loss_labels:  # pylint:disable=protected-access
            return
        # Import global session here to prevent circular import
        from . import Session  # pylint:disable=import-outside-toplevel
        loss_labels = sorted(Session.get_loss_keys(session_id=session_id))
        self._loss_labels = loss_labels
        logger.debug("Collated loss labels: %s", self._loss_labels)

    @classmethod
    def _process_event(cls, event, step):
        """ Process a single Tensorflow event.

        Adds timestamp to the step `dict` if a total loss value is received, process the labels for
        any new loss entries and adds the side loss value to the step `dict`.

        Parameters
        ----------
        event: :class:`tensorflow.core.util.event_pb2`
            The event data to be processed
        step: dict
            The dictionary to populated with the extracted data from the tensorflow event

        Returns
        -------
        dict
            The given step `dict` with the given event data added to it.
        """
        summary = event.summary.value[0]

        if summary.tag in ("batch_loss", "batch_total"):  # Pre tf2.3 totals were "batch_total"
            step["timestamp"] = event.wall_time
            return step

        loss = summary.simple_value
        if not loss:
            # Need to convert a tensor to a float for TF2.8 logged data. This maybe due to change
            # in logging or may be due to work around put in place in FS training function for the
            # following bug in TF 2.8 when writing records:
            #  https://github.com/keras-team/keras/issues/16173
            loss = float(tf.make_ndarray(summary.tensor))

        step.setdefault("loss", []).append(loss)

        return step
