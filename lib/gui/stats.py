#!/usr/bin python3
""" Stats functions for the GUI.

Holds the globally loaded training session. This will either be a user selected session (loaded in
the analysis tab) or the currently training session.

"""

import logging
import time
import os
import warnings
import zlib

from math import ceil
from threading import Event

import numpy as np
import tensorflow as tf
from tensorflow.python import errors_impl as tf_errors  # pylint:disable=no-name-in-module
from tensorflow.core.util import event_pb2
from lib.serializer import get_serializer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class GlobalSession():
    """ Holds information about a loaded or current training session by accessing a model's state
    file and Tensorboard logs. This class should not be accessed directly, rather through
    :attr:`lib.stats.Session`
    """
    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        self._state = None
        self._model_dir = None
        self._model_name = None

        self._tb_logs = None
        self._summary = None

        self._is_training = False
        self._is_querying = Event()

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def is_loaded(self):
        """ bool: ``True`` if session data is loaded otherwise ``False`` """
        return self._model_dir is not None

    @property
    def is_training(self):
        """ bool: ``True`` if the loaded session is the currently training model, otherwise
        ``False`` """
        return self._is_training

    @property
    def model_filename(self):
        """ str: The full model filename """
        return os.path.join(self._model_dir, self._model_name)

    @property
    def batch_sizes(self):
        """ dict: The batch sizes for each session_id for the model. """
        return {int(sess_id): sess["batchsize"]
                for sess_id, sess in self._state["sessions"].items()}

    @property
    def full_summary(self):
        """ list: List of dictionaries containing summary statistics for each session id. """
        return self._summary.get_summary_stats()

    @property
    def logging_disabled(self):
        """ bool: ``True`` if logging is enabled for the currently training session otherwise
        ``False``. """
        return self._state["sessions"][str(self.session_ids[-1])]["no_logs"]

    @property
    def session_ids(self):
        """ list: The sorted list of all existing session ids `int`s in the state file """
        return self._tb_logs.session_ids

    def _load_state_file(self):
        """ Load the current state file to :attr:`_state`. """
        state_file = os.path.join(self._model_dir, "{}_state.json".format(self._model_name))
        logger.debug("Loading State: '%s'", state_file)
        serializer = get_serializer("json")
        self._state = serializer.load(state_file)
        logger.debug("Loaded state: %s", self._state)

    def initialize_session(self, model_folder, model_name, is_training=False):
        """ Initialize a Session.

        Load's the model's state file, and sets the paths to any underlying Tensorboard logs, ready
        for access on request.

        Parameters
        ----------
        model_folder: str, optional
            If loading a session manually (e.g. for the analysis tab), then the path to the model
            folder must be provided. For training sessions, this should be left at ``None``
        model_name: str, optional
            If loading a session manually (e.g. for the analysis tab), then the model filename
            must be provided. For training sessions, this should be left at ``None``
        is_training: bool, optional
            ``True`` if the session is being initialized for a training session, otherwise
            ``False``. Default: ``False``
         """
        logger.debug("Initializing session: (is_training: %s)", is_training)

        if self._model_dir == model_folder and self._model_name == model_name:
            if is_training:
                self._tb_logs.refresh_log_filenames()
                self._load_state_file()
                self._is_training = True
            logger.debug("Requested session is already loaded. Not initializing: (model_folder: "
                         "%s, model_name: %s)", model_folder, model_name)
            return

        self._is_training = is_training
        self._model_dir = model_folder
        self._model_name = model_name
        self._load_state_file()
        self._tb_logs = TensorBoardLogs(os.path.join(self._model_dir,
                                                     "{}_logs".format(self._model_name)))

        self._summary = SessionsSummary(self)
        logger.debug("Initialized session. Session_IDS: %s", self.session_ids)

    def stop_training(self):
        """ Clears the internal training flag. To be called when training completes. """
        self._is_training = False

    def clear(self):
        """ Clear the currently loaded session. """
        self._state = None
        self._model_dir = None
        self._model_name = None

        del self._tb_logs
        self._tb_logs = None

        del self._summary
        self._summary = None

        self._is_training = False

    def get_loss(self, session_id):
        """ Obtain the loss values for the given session_id.

        Parameters
        ----------
        session_id: int or ``None``
            The session ID to return loss for. Pass ``None`` to return loss for all sessions.

        Returns
        -------
        dict
            Loss names as key, :class:`numpy.ndarray` as value. If No session ID was provided
            all session's losses are collated
        """
        self._wait_for_thread()

        if self._is_training:
            self._is_querying.set()

        loss_dict = self._tb_logs.get_loss(session_id=session_id, is_training=self._is_training)
        if session_id is None:
            retval = dict()
            for key in sorted(loss_dict):
                for loss_key, loss in loss_dict[key].items():
                    retval.setdefault(loss_key, []).extend(loss)
            retval = {key: np.array(val, dtype="float32") for key, val in retval.items()}
        else:
            retval = loss_dict[session_id]

        if self._is_training:
            self._is_querying.clear()
        return retval

    def get_timestamps(self, session_id):
        """ Obtain the time stamps keys for the given session_id.

        Parameters
        ----------
        session_id: int or ``None``
            The session ID to return the time stamps for. Pass ``None`` to return time stamps for
            all sessions.

        Returns
        -------
        dict or :class:`numpy.ndarray`
            If a session ID has been given then a single :class:`numpy.ndarray` will be returned
            with the session's time stamps. Otherwise a 'dict' will be returned with the session
            IDs as key with :class:`numpy.ndarray` of timestamps as values
        """
        self._wait_for_thread()

        if self._is_training:
            self._is_querying.set()

        retval = self._tb_logs.get_timestamps(session_id=session_id, is_training=self._is_training)
        if session_id is not None:
            retval = retval[session_id]

        if self._is_training:
            self._is_querying.clear()

        return retval

    def _wait_for_thread(self):
        """ If a thread is querying the log files for live data, then block until task clears. """
        while True:
            if self._is_training and self._is_querying.is_set():
                logger.debug("Waiting for available thread")
                time.sleep(1)
                continue
            break

    def get_loss_keys(self, session_id):
        """ Obtain the loss keys for the given session_id.

        Parameters
        ----------
        session_id: int or ``None``
            The session ID to return the loss keys for. Pass ``None`` to return loss keys for
            all sessions.

        Returns
        -------
        list
            The loss keys for the given session. If ``None`` is passed as session_id then a unique
            list of all loss keys for all sessions is returned
        """
        if session_id is None:
            retval = list(set(loss_key
                              for session in self._state["sessions"].values()
                              for loss_key in session["loss_names"]))
        else:
            retval = self._state["sessions"][str(session_id)]["loss_names"]
        return retval


Session = GlobalSession()


class TensorBoardLogs():
    """ Parse data from TensorBoard logs.

    Process the input logs folder and stores the individual filenames per session.

    Caches timestamp and loss data on request and returns this data from the cache.

    Parameters
    ----------
    logs_folder: str
        The folder that contains the Tensorboard log files
    """
    def __init__(self, logs_folder):
        self._folder_base = logs_folder
        self._log_filenames = self._get_log_filenames()
        self._cache = dict()
        self._training_iterator = None
        self._training_rollover = dict()

    @property
    def session_ids(self):
        """ list: Sorted list of integers of available session ids. """
        return list(sorted(self._log_filenames))

    def _get_log_filenames(self):
        """ Get the TensorBoard log filenames for all existing sessions.

        Returns
        -------
        dict
            The full path of each log file for each training session that has been run
        """
        logger.debug("Loading log filenames. base_dir: '%s'", self._folder_base)
        log_filenames = dict()
        for dirpath, _, filenames in os.walk(self._folder_base):
            if not any(filename.startswith("events.out.tfevents") for filename in filenames):
                continue
            logfiles = [filename for filename in filenames
                        if filename.startswith("events.out.tfevents")]
            # Take the last log file, in case of previous crash
            logfile = os.path.join(dirpath, sorted(logfiles)[-1])
            session = os.path.split(os.path.split(dirpath)[0])[1]
            session = session[session.rfind("_") + 1:]
            if not session.isdigit():
                logger.warning("Unable to load session data for model")
                return log_filenames
            session = int(session)
            log_filenames[session] = logfile
        logger.debug("logfiles: %s", log_filenames)
        return log_filenames

    def _cache_data(self, session_id, is_training=False):
        """ Cache TensorBoard logs for the given session ID on first access.

        Populates :attr:`_cache` with timestamps and loss data.

        If this is a training session and the data is being queried for the training session ID
        then get the latest available data and append to the cache

        Parameters
        -------
        session_id: int
            The session ID to cache the data for
        is_training: bool, optional
            ``True`` if a current training session is running otherwise ``False``.
            Default: ``False``
        """
        labels = []
        step = []
        loss = []
        timestamps = []
        last_step = -1
        live_data = is_training and session_id == max(self._log_filenames)

        if live_data:
            iterator = self._get_latest_live()
        else:
            iterator = tf.compat.v1.io.tf_record_iterator(self._log_filenames[session_id])

        try:
            for record in iterator:
                event = event_pb2.Event.FromString(record)
                if not event.summary.value or not event.summary.value[0].tag.startswith("batch_"):
                    continue

                if last_step == -1:
                    last_step = event.step if live_data else 0

                if event.step != last_step:
                    loss.append(step)
                    step = []
                    last_step = event.step

                summary = event.summary.value[0]
                tag = summary.tag

                if tag == "batch_total":
                    timestamps.append(event.wall_time)
                    continue

                lbl = tag.replace("batch_", "")
                if lbl not in labels:
                    labels.append(lbl)

                step.append(summary.simple_value)

        except tf_errors.DataLossError as err:
            logger.warning("The logs for Session %s are corrupted and cannot be displayed. "
                           "The totals do not include this session. Original error message: "
                           "'%s'", session_id, str(err))

        if step:
            loss.append(step)

        loss = np.array(loss, dtype="float32")
        timestamps = np.array(timestamps, dtype="float64")

        logger.debug("Caching session id: %s, labels: %s, loss: %s, timestamps: %s",
                     session_id, labels, loss.shape, timestamps.shape)

        if live_data and session_id in self._cache:
            self._add_latest_data_to_cache(session_id, loss, timestamps)
        else:
            self._cache[session_id] = dict(labels=labels,
                                           loss=zlib.compress(loss),
                                           loss_shape=loss.shape,
                                           timestamps=zlib.compress(timestamps),
                                           timestamps_shape=timestamps.shape)

    def _get_latest_live(self):
        """ Obtain the latest event logs for live training data and add to the cache """
        if self._training_iterator is None:
            training_session_id = self.session_ids[-1]
            filename = self._log_filenames[training_session_id]
            self._training_iterator = tf.compat.v1.io.tf_record_iterator(filename)
            logger.debug("Set live training iterator %s for session_id: %s",
                         self._training_iterator, training_session_id)

        i = 0
        while True:
            try:
                yield next(self._training_iterator)
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

    def _add_latest_data_to_cache(self, session_id, loss, timestamps):
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
        if not np.any(loss) and not np.any(timestamps):
            logger.debug("No new live data to cache.")
            return

        logger.debug("Adding live data to cache: (loss: %s, timestamps: %s)",
                     loss.shape, timestamps.shape)

        cache = self._cache[session_id]

        past_loss = np.frombuffer(zlib.decompress(cache["loss"]),
                                  dtype="float32").reshape(cache["loss_shape"])
        new_loss = np.concatenate((past_loss, loss))
        cache["loss_shape"] = new_loss.shape
        cache["loss"] = zlib.compress(new_loss)
        del past_loss

        past_timestamps = np.frombuffer(zlib.decompress(cache["timestamps"]),
                                        dtype="float64").reshape(cache["timestamps_shape"])
        new_timestamps = np.concatenate((past_timestamps, timestamps))
        cache["timestamps_shape"] = new_timestamps.shape
        cache["timestamps"] = zlib.compress(new_timestamps)
        del past_timestamps

    def _from_cache(self, session_id=None, is_training=False):
        """ Get the session data from the cache.

        If the request data does not exist in the cache, then populate it.

        Parameters
        ----------
        session_id: int, optional
            The Session ID to return the data for. Set to ``None`` to return all session
            data. Default ``None`
        is_training: bool, optional
            ``True`` if a current training session is running otherwise ``False``.
            Default: ``False``

        Returns
        -------
        dict
            The session id(s) as key, with the event data as value
        """
        if session_id is not None and session_id not in self._cache:
            self._cache_data(session_id)
        elif is_training and session_id == self.session_ids[-1]:
            self._cache_data(session_id, is_training=is_training)
        elif session_id is None and not all(idx in self._cache for idx in self._log_filenames):
            for sess in self._log_filenames:
                if sess not in self._cache:
                    self._cache_data(sess)

        if session_id is None:
            return self._cache
        return {session_id: self._cache[session_id]}

    def get_loss(self, session_id=None, is_training=False):
        """ Read the loss from the TensorBoard event logs

        Parameters
        ----------
        session_id: int, optional
            The Session ID to return the loss for. Set to ``None`` to return all session
            losses. Default ``None``
        is_training: bool, optional
            ``True`` if a current training session is running otherwise ``False``.
            Default: ``False``

        Returns
        -------
        dict
            The session id(s) as key, with a further dictionary as value containing the loss name
            and list of loss values for each step
        """
        logger.debug("Getting loss: (session_id: %s)", session_id)
        retval = dict()
        for sess, info in self._from_cache(session_id, is_training).items():
            arr = np.frombuffer(zlib.decompress(info["loss"]),
                                dtype="float32").reshape(info["loss_shape"])
            for idx, title in enumerate(info["labels"]):
                retval.setdefault(sess, dict())[title] = arr[:, idx]
        logger.debug({key: {k: v.shape for k, v in val.items()}
                      for key, val in retval.items()})
        return retval

    def get_timestamps(self, session_id=None, is_training=False):
        """ Read the timestamps from the TensorBoard logs.

        As loss timestamps are slightly different for each loss, we collect the timestamp from the
        `batch_total` key.

        Parameters
        ----------
        session_id: int, optional
            The Session ID to return the timestamps for. Set to ``None`` to return all session
            timestamps. Default ``None``
        is_training: bool, optional
            ``True`` if a current training session is running otherwise ``False``.
            Default: ``False``

        Returns
        -------
        dict
            The session id(s) as key with list of timestamps per step as value
        """

        logger.debug("Getting timestamps: (session_id: %s, is_training: %s)",
                     session_id, is_training)
        retval = {sess: np.frombuffer(zlib.decompress(info["timestamps"]),
                                      dtype="float64").reshape(info["timestamps_shape"])
                  for sess, info in self._from_cache(session_id, is_training).items()}
        logger.debug({k: v.shape for k, v in retval.items()})
        return retval

    def refresh_log_filenames(self):
        """ Refresh the log file list in :attr:`_log_filenames`.

        Called when a training session is loaded, to add the latest log filename to the list.
        """
        self._log_filenames = self._get_log_filenames()


class SessionsSummary():  # pylint:disable=too-few-public-methods
    """ Performs top level summary calculations for each session ID within the loaded or currently
    training Session for display in the Analysis tree view.

    Parameters
    ----------
    session: :class:`GlobalSession`
        The loaded or currently training session
    """
    def __init__(self, session):
        logger.debug("Initializing %s: (session: %s)", self.__class__.__name__, session)
        self._session = session
        self._state = session._state

        self._time_stats = None
        self._per_session_stats = None
        logger.debug("Initialized %s", self.__class__.__name__)

    def get_summary_stats(self):
        """ Compile the individual session statistics and calculate the total.

        Format the stats for display

        Returns
        -------
        list
            A list of summary statistics dictionaries containing the Session ID, start time, end
            time, elapsed time, rate, batch size and number of iterations for each session id
            within the loaded data as well as the totals.
        """
        logger.debug("Compiling sessions summary data")
        self._get_time_stats()
        self._get_per_session_stats()
        if not self._per_session_stats:
            return self._per_session_stats

        total_stats = self._total_stats()
        retval = self._per_session_stats + [total_stats]
        retval = self._format_stats(retval)
        logger.debug("Final stats: %s", retval)
        return retval

    def _get_time_stats(self):
        """ Populates the attribute :attr:`_time_stats` with the start start time, end time and
        data points for each session id within the loaded session if it has not already been
        calculated.

        If the main Session is currently training, then the training session ID is updated with the
        latest stats.
        """
        if self._time_stats is None:
            logger.debug("Collating summary time stamps")

            self._time_stats = {
                sess_id: dict(start_time=np.min(timestamps) if np.any(timestamps) else 0,
                              end_time=np.max(timestamps) if np.any(timestamps) else 0,
                              iterations=timestamps.shape[0] if np.any(timestamps) else 0)
                for sess_id, timestamps in self._session.get_timestamps(None).items()}

        elif Session.is_training:
            logger.debug("Updating summary time stamps for training session")

            session_id = Session.session_ids[-1]
            latest = self._session.get_timestamps(session_id)

            self._time_stats[session_id] = dict(
                start_time=np.min(latest) if np.any(latest) else 0,
                end_time=np.max(latest) if np.any(latest) else 0,
                iterations=latest.shape[0] if np.any(latest) else 0)

        logger.debug("time_stats: %s", self._time_stats)

    def _get_per_session_stats(self):
        """ Populate the attribute :attr:`_per_session_stats` with a sorted list by session ID
        of each ID in the training/loaded session. Stats contain the session ID, start, end and
        elapsed times, the training rate, batch size and number of iterations for each session.

        If a training session is running, then updates the training sessions stats only.
        """
        if self._per_session_stats is None:
            logger.debug("Collating per session stats")
            compiled = list()
            for session_id, ts_data in self._time_stats.items():
                logger.debug("Compiling session ID: %s", session_id)
                if self._state is None:
                    logger.debug("Session state dict doesn't exist. Most likely task has been "
                                 "terminated during compilation")
                    return
                compiled.append(self._collate_stats(session_id, ts_data))

            self._per_session_stats = list(sorted(compiled, key=lambda k: k["session"]))

        elif self._session.is_training:
            logger.debug("Collating per session stats for latest training data")
            session_id = self._session.session_ids[-1]
            ts_data = self._time_stats[session_id]

            if session_id > len(self._per_session_stats):
                self._per_session_stats.append(self._collate_stats(session_id, ts_data))

            stats = self._per_session_stats[-1]

            stats["start"] = ts_data["start_time"]
            stats["end"] = ts_data["end_time"]
            stats["elapsed"] = int(stats["end"] - stats["start"])
            stats["iterations"] = ts_data["iterations"]
            stats["rate"] = (((stats["batch"] * 2) * stats["iterations"])
                             / stats["elapsed"] if stats["elapsed"] != 0 else 0)
        logger.debug("per_session_stats: %s", self._per_session_stats)

    def _collate_stats(self, session_id, timestamps):
        """ Collate the session summary statistics for the given session ID.

        Parameters
        ----------
        session_id: int
            The session id to compile the stats for
        timestamps:
            The time stamp summary data for the given session id

        Returns
        -------
        dict
            The collated session summary statistics
        """
        timestamps = self._time_stats[session_id]
        elapsed = int(timestamps["end_time"] - timestamps["start_time"])
        batchsize = self._session.batch_sizes.get(session_id, 0)
        retval = dict(
            session=session_id,
            start=timestamps["start_time"],
            end=timestamps["end_time"],
            elapsed=elapsed,
            rate=(((batchsize * 2) * timestamps["iterations"]) / elapsed if elapsed != 0 else 0),
            batch=batchsize,
            iterations=timestamps["iterations"])
        logger.debug(retval)
        return retval

    def _total_stats(self):
        """ Compile the Totals stats.
        Totals are fully calculated each time as they will change on the basis of the training
        session.

        Returns
        -------
        dict:
            The Session name, start time, end time, elapsed time, rate, batch size and number of
            iterations for all session ids within the loaded data.
        """
        logger.debug("Compiling Totals")
        elapsed = 0
        examples = 0
        iterations = 0
        batchset = set()
        total_summaries = len(self._per_session_stats)
        for idx, summary in enumerate(self._per_session_stats):
            if idx == 0:
                starttime = summary["start"]
            if idx == total_summaries - 1:
                endtime = summary["end"]
            elapsed += summary["elapsed"]
            examples += ((summary["batch"] * 2) * summary["iterations"])
            batchset.add(summary["batch"])
            iterations += summary["iterations"]
        batch = ",".join(str(bs) for bs in batchset)
        totals = {"session": "Total",
                  "start": starttime,
                  "end": endtime,
                  "elapsed": elapsed,
                  "rate": examples / elapsed if elapsed != 0 else 0,
                  "batch": batch,
                  "iterations": iterations}
        logger.debug(totals)
        return totals

    def _format_stats(self, compiled_stats):
        """ Format for the incoming list of statistics for display.

        Parameters
        ----------
        compiled_stats: list
            List of summary statistics dictionaries to be formatted for display

        Returns
        -------
        list
            The original statistics formatted for display
        """
        logger.debug("Formatting stats")
        retval = []
        for summary in compiled_stats:
            hrs, mins, secs = self._convert_time(summary["elapsed"])
            stats = dict()
            for key in summary:
                if key not in ("start", "end", "elapsed", "rate"):
                    stats[key] = summary[key]
                    continue
                stats["start"] = time.strftime("%x %X", time.localtime(summary["start"]))
                stats["end"] = time.strftime("%x %X", time.localtime(summary["end"]))
                stats["elapsed"] = "{}:{}:{}".format(hrs, mins, secs)
                stats["rate"] = "{0:.1f}".format(summary["rate"])
            retval.append(stats)
        return retval

    @classmethod
    def _convert_time(cls, timestamp):
        """ Convert time stamp to total hours, minutes and seconds.

        Parameters
        ----------
        timestamp: float
            The Unix timestamp to be converted

        Returns
        -------
        tuple
            (`hours`, `minutes`, `seconds`) as ints
        """
        hrs = int(timestamp // 3600)
        if hrs < 10:
            hrs = "{0:02d}".format(hrs)
        mins = "{0:02d}".format((int(timestamp % 3600) // 60))
        secs = "{0:02d}".format((int(timestamp % 3600) % 60))
        return hrs, mins, secs


class Calculations():
    """ Class that performs calculations on the :class:`GlobalSession` raw data for the given
    session id.

    Parameters
    ----------
    session_id: int or ``None``
        The session id number for the selected session from the Analysis tab. Should be ``None``
        if all sessions are being calculated
    display: {"loss", "rate"}, optional
        Whether to display a graph for loss or training rate. Default: `"loss"`
    loss_keys: list, optional
        The list of loss keys to display on the graph. Default: `["loss"]`
    selections: list, optional
        The selected annotations to display. Default: `["raw"]`
    avg_samples: int, optional
        The number of samples to use for performing moving average calculation. Default: `500`.
    smooth_amount: float, optional
        The amount of smoothing to apply for performing smoothing calculation. Default: `0.9`.
    flatten_outliers: bool, optional
        ``True`` if values significantly away from the average should be excluded, otherwise
        ``False``. Default: ``False``
    """
    def __init__(self, session_id,
                 display="loss",
                 loss_keys="loss",
                 selections="raw",
                 avg_samples=500,
                 smooth_amount=0.90,
                 flatten_outliers=False):
        logger.debug("Initializing %s: (session_id: %s, display: %s, loss_keys: %s, "
                     "selections: %s, avg_samples: %s, smooth_amount: %s, flatten_outliers: %s)",
                     self.__class__.__name__, session_id, display, loss_keys, selections,
                     avg_samples, smooth_amount, flatten_outliers)

        warnings.simplefilter("ignore", np.RankWarning)

        self._session_id = session_id

        self._display = display
        self._loss_keys = loss_keys if isinstance(loss_keys, list) else [loss_keys]
        self._selections = selections if isinstance(selections, list) else [selections]
        self._is_totals = session_id is None
        self._args = dict(avg_samples=avg_samples,
                          smooth_amount=smooth_amount,
                          flatten_outliers=flatten_outliers)
        self._iterations = 0
        self._limit = 0
        self._start_iteration = 0
        self._stats = dict()
        self.refresh()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def iterations(self):
        """ int: The number of iterations in the data set. """
        return self._iterations

    @property
    def start_iteration(self):
        """ int: The starting iteration number of a limit has been set on the amount of data. """
        return self._start_iteration

    @property
    def stats(self):
        """ dict: The final calculated statistics """
        return self._stats

    def refresh(self):
        """ Refresh the stats """
        logger.debug("Refreshing")
        if not Session.is_loaded:
            logger.warning("Session data is not initialized. Not refreshing")
            return None
        self._iterations = 0
        self._get_raw()
        self._get_calculations()
        self._remove_raw()
        logger.debug("Refreshed")
        return self

    def set_smooth_amount(self, amount):
        """ Set the amount of smoothing to apply to smoothed graph.

        Parameters
        ----------
        amount: float
            The amount of smoothing to apply to smoothed graph
        """
        update = max(min(amount, 0.999), 0.001)
        logger.debug("Setting smooth amount to: %s (provided value: %s)", update, amount)
        self._args["smooth_amount"] = update

    def update_selections(self, selection, option):
        """ Update the type of selected data.

        Parameters
        ----------
        selection: str
            The selection to update (as can exist in :attr:`_selections`)
        option: bool
            ``True`` if the selection should be included, ``False`` if it should be removed
        """
        # TODO Somewhat hacky, to ensure values are inserted in the correct order. Fine for
        # now as this is only called from Live Graph and selections can only be "raw" and
        # smoothed.
        if option:
            if selection not in self._selections:
                if selection == "raw":
                    self._selections.insert(0, selection)
                else:
                    self._selections.append(selection)
        else:
            if selection in self._selections:
                self._selections.remove(selection)

    def set_iterations_limit(self, limit):
        """ Set the number of iterations to display in the calculations.

        If a value greater than 0 is passed, then the latest iterations up to the given
        limit will be calculated.

        Parameters
        ----------
        limit: int
            The number of iterations to calculate data for. `0` to calculate for all data
        """
        limit = max(0, limit)
        logger.debug("Setting iteration limit to: %s", limit)
        self._limit = limit

    def _get_raw(self):
        """ Obtain the raw loss values and add them to a new :attr:`stats` dictionary. """
        logger.debug("Getting Raw Data")
        self.stats.clear()
        iterations = set()

        if self._display.lower() == "loss":
            loss_dict = Session.get_loss(self._session_id)
            for loss_name, loss in loss_dict.items():
                if loss_name not in self._loss_keys:
                    continue
                iterations.add(loss.shape[0])

                if self._limit > 0:
                    loss = loss[-self._limit:]

                if self._args["flatten_outliers"]:
                    loss = self._flatten_outliers(loss)

                self.stats["raw_{}".format(loss_name)] = loss

            self._iterations = 0 if not iterations else min(iterations)
            if self._limit > 1:
                self._start_iteration = max(0, self._iterations - self._limit)
                self._iterations = min(self._iterations, self._limit)
            else:
                self._start_iteration = 0

            if len(iterations) > 1:
                # Crop all losses to the same number of items
                if self._iterations == 0:
                    self.stats = {lossname: np.array(list(), dtype=loss.dtype)
                                  for lossname, loss in self.stats.items()}
                else:
                    self.stats = {lossname: loss[:self._iterations]
                                  for lossname, loss in self.stats.items()}

        else:  # Rate calculation
            data = self._calc_rate_total() if self._is_totals else self._calc_rate()
            if self._args["flatten_outliers"]:
                data = self._flatten_outliers(data)
            self._iterations = data.shape[0]
            self.stats["raw_rate"] = data

        logger.debug("Got Raw Data")

    @classmethod
    def _flatten_outliers(cls, data):
        """ Remove the outliers from a provided list.

        Removes data more than 1 Standard Deviation from the mean.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            The data to remove the outliers from

        Returns
        -------
        :class:`numpy.ndarray`
            The data with outliers removed
        """
        logger.debug("Flattening outliers: %s", data.shape)
        mean = np.mean(data)
        limit = np.std(data)
        logger.debug("mean: %s, limit: %s", mean, limit)
        retdata = np.where(abs(data - mean) < limit, data, mean)
        logger.debug("Flattened outliers")
        return retdata

    def _remove_raw(self):
        """ Remove raw values from :attr:`stats` if they are not requested. """
        if "raw" in self._selections:
            return
        logger.debug("Removing Raw Data from output")
        for key in list(self._stats.keys()):
            if key.startswith("raw"):
                del self._stats[key]
        logger.debug("Removed Raw Data from output")

    def _calc_rate(self):
        """ Calculate rate per iteration.

        Returns
        -------
        :class:`numpy.ndarray`
            The training rate for each iteration of the selected session
        """
        logger.debug("Calculating rate")
        retval = (Session.batch_sizes[self._session_id] * 2) / np.diff(Session.get_timestamps(
            self._session_id))
        logger.debug("Calculated rate: Item_count: %s", len(retval))
        return retval

    @classmethod
    def _calc_rate_total(cls):
        """ Calculate rate per iteration for all sessions.

        Returns
        -------
        :class:`numpy.ndarray`
            The training rate for each iteration in all sessions

        Notes
        -----
        For totals, gaps between sessions can be large so the time difference has to be reset for
        each session's rate calculation.
        """
        logger.debug("Calculating totals rate")
        batchsizes = Session.batch_sizes
        total_timestamps = Session.get_timestamps(None)
        rate = list()
        for sess_id in sorted(total_timestamps.keys()):
            batchsize = batchsizes[sess_id]
            timestamps = total_timestamps[sess_id]
            rate.extend((batchsize * 2) / np.diff(timestamps))
        retval = np.array(rate)
        logger.debug("Calculated totals rate: Item_count: %s", len(retval))
        return retval

    def _get_calculations(self):
        """ Perform the required calculations and populate :attr:`stats`. """
        for selection in self._selections:
            if selection == "raw":
                continue
            logger.debug("Calculating: %s", selection)
            method = getattr(self, "_calc_{}".format(selection))
            raw_keys = [key for key in self._stats if key.startswith("raw_")]
            for key in raw_keys:
                selected_key = "{}_{}".format(selection, key.replace("raw_", ""))
                self._stats[selected_key] = method(self._stats[key])

    def _calc_avg(self, data):
        """ Calculate moving average.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            The data to calculate the moving average for

        Returns
        -------
        :class:`numpy.ndarray`
            The moving average for the given data
        """
        logger.debug("Calculating Average")
        window = self._args["avg_samples"]
        pad = ceil(window / 2)
        datapoints = data.shape[0]

        if datapoints <= (self._args["avg_samples"] * 2):
            logger.info("Not enough data to compile rolling average")
            return np.array([], dtype="float64")

        avgs = np.cumsum(data, dtype="float64")
        avgs[window:] = avgs[window:] - avgs[:-window]
        avgs = avgs[window - 1:] / window
        avgs = np.pad(avgs, (pad, datapoints - (avgs.shape[0] + pad)), constant_values=(np.nan,))
        logger.debug("Calculated Average: shape: %s", avgs.shape)
        return avgs

    def _calc_smoothed(self, data):
        """ Smooth the data.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            The data to smoothen

        Returns
        -------
        :class:`numpy.ndarray`
            The smoothed data
        """
        return ExponentialMovingAverage(data, self._args["smooth_amount"])()

    @classmethod
    def _calc_trend(cls, data):
        """ Calculate polynomial trend of the given data.

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            The data to calculate the trend for

        Returns
        -------
        :class:`numpy.ndarray`
            The trend for the given data
        """
        logger.debug("Calculating Trend")
        points = data.shape[0]
        if points < 10:
            dummy = np.empty((points, ), dtype=data.dtype)
            dummy[:] = np.nan
            return dummy
        x_range = range(points)
        trend = np.poly1d(np.polyfit(x_range, data, 3))(x_range)
        logger.debug("Calculated Trend")
        return trend


class ExponentialMovingAverage():  # pylint:disable=too-few-public-methods
    """ Reshapes data before calculating exponential moving average, then iterates once over the
    rows to calculate the offset without precision issues.

    Parameters
    ----------
    data: :class:`numpy.ndarray`
        A 1 dimensional numpy array to obtain smoothed data for
    amount: float
        in the range (0.0, 1.0) The alpha parameter (smoothing amount) for the moving average.

    Notes
    -----
    Adapted from: https://stackoverflow.com/questions/42869495
    """
    def __init__(self, data, amount):
        assert data.ndim == 1
        amount = min(max(amount, 0.001), 0.999)

        self._data = data
        self._alpha = 1. - amount
        self._dtype = "float32" if data.dtype == np.float32 else "float64"
        self._row_size = self._get_max_row_size()
        self._out = np.empty_like(data, dtype=self._dtype)

    def __call__(self):
        """ Perform the exponential moving average calculation.

        Returns
        -------
        :class:`numpy.ndarray`
            The smoothed data
        """
        if self._data.size <= self._row_size:
            self._ewma_vectorized(self._data, self._out)  # Normal function can handle this input
        else:
            self._ewma_vectorized_safe()  # Use the safe version
        return self._out

    def _get_max_row_size(self):
        """ Calculate the maximum row size for the running platform for the given dtype.

        Returns
        -------
        int
            The maximum row size possible on the running platform for the given :attr:`_dtype`

        Notes
        -----
        Might not be the optimal value for speed, which is hard to predict due to numpy
        optimizations.
        """
        # Use :func:`np.finfo(dtype).eps` if you are worried about accuracy and want to be safe.
        epsilon = np.finfo(self._dtype).tiny
        # If this produces an OverflowError, make epsilon larger:
        retval = int(np.log(epsilon) / np.log(1 - self._alpha)) + 1
        logger.debug("row_size: %s", retval)
        return retval

    def _ewma_vectorized_safe(self):
        """ Perform the vectorized exponential moving average in a safe way. """
        num_rows = int(self._data.size // self._row_size)  # the number of rows to use
        leftover = int(self._data.size % self._row_size)  # the amount of data leftover
        first_offset = self._data[0]

        if leftover > 0:
            # set temporary results to slice view of out parameter
            out_main_view = np.reshape(self._out[:-leftover], (num_rows, self._row_size))
            data_main_view = np.reshape(self._data[:-leftover], (num_rows, self._row_size))
        else:
            out_main_view = self._out.reshape(-1, self._row_size)
            data_main_view = self._data.reshape(-1, self._row_size)

        self._ewma_vectorized_2d(data_main_view, out_main_view)  # get the scaled cumulative sums

        scaling_factors = (1 - self._alpha) ** np.arange(1, self._row_size + 1)
        last_scaling_factor = scaling_factors[-1]

        # create offset array
        offsets = np.empty(out_main_view.shape[0], dtype=self._dtype)
        offsets[0] = first_offset
        # iteratively calculate offset for each row

        for i in range(1, out_main_view.shape[0]):
            offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

        # add the offsets to the result
        out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

        if leftover > 0:
            # process trailing data in the 2nd slice of the out parameter
            self._ewma_vectorized(self._data[-leftover:],
                                  self._out[-leftover:],
                                  offset=out_main_view[-1, -1])

    def _ewma_vectorized(self, data, out, offset=None):
        """ Calculates the exponential moving average over a vector. Will fail for large inputs.

        The result is processed in place into the array passed to the `out` parameter

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            A 1 dimensional numpy array to obtain smoothed data for
        out: :class:`numpy.ndarray`
            A location into which the result is stored. It must have the same shape and dtype as
            the input data
        offset: float, optional
            The offset for the moving average, scalar. Default: the value held in data[0].
        """
        if data.size < 1:  # empty input, return empty array
            return

        offset = data[0] if offset is None else offset

        # scaling_factors -> 0 as len(data) gets large. This leads to divide-by-zeros below
        scaling_factors = np.power(1. - self._alpha, np.arange(data.size + 1, dtype=self._dtype),
                                   dtype=self._dtype)
        # create cumulative sum array
        np.multiply(data, (self._alpha * scaling_factors[-2]) / scaling_factors[:-1],
                    dtype=self._dtype, out=out)
        np.cumsum(out, dtype=self._dtype, out=out)

        out /= scaling_factors[-2::-1]  # cumulative sums / scaling

        if offset != 0:
            offset = np.array(offset, copy=False).astype(self._dtype, copy=False)
            out += offset * scaling_factors[1:]

    def _ewma_vectorized_2d(self, data, out):
        """ Calculates the exponential moving average over the last axis.

        The result is processed in place into the array passed to the `out` parameter

        Parameters
        ----------
        data: :class:`numpy.ndarray`
            A 1 or 2 dimensional numpy array to obtain smoothed data for.
        out: :class:`numpy.ndarray`
            A location into which the result is stored. It must have the same shape and dtype as
            the input data
        """
        if data.size < 1:  # empty input, return empty array
            return

        # calculate the moving average
        scaling_factors = np.power(1. - self._alpha, np.arange(data.shape[1] + 1,
                                                               dtype=self._dtype),
                                   dtype=self._dtype)
        # create a scaled cumulative sum array
        np.multiply(data,
                    np.multiply(self._alpha * scaling_factors[-2],
                                np.ones((data.shape[0], 1), dtype=self._dtype),
                                dtype=self._dtype) / scaling_factors[np.newaxis, :-1],
                    dtype=self._dtype, out=out)
        np.cumsum(out, axis=1, dtype=self._dtype, out=out)
        out /= scaling_factors[np.newaxis, -2::-1]
