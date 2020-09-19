#!/usr/bin python3
""" Stats functions for the GUI """

import logging
import time
import os
import warnings
import zlib

from math import ceil

import numpy as np
import tensorflow as tf
from tensorflow.python import errors_impl as tf_errors  # pylint:disable=no-name-in-module
from tensorflow.core.util import event_pb2
from lib.serializer import get_serializer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def convert_time(timestamp):
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

    def _cache_data(self, session):
        """ Cache TensorBoard logs for the given session on first access.

        Populates :attr:`_cache` with timestamps and loss data.

        Parameters
        -------
        session: int
            The session index to cache the data for
        """
        labels = []
        step = []
        loss = []
        timestamps = []
        last_step = 0

        try:
            for record in tf.compat.v1.io.tf_record_iterator(self._log_filenames[session]):
                event = event_pb2.Event.FromString(record)
                if not event.summary.value or not event.summary.value[0].tag.startswith("batch_"):
                    continue

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
                           "'%s'", session, str(err))

        if step:
            loss.append(step)

        loss = np.array(loss, dtype="float32")
        timestamps = np.array(timestamps, dtype="float64")
        logger.debug("Caching session id: %s, labels: %s, loss shape: %s, loss shape: %s",
                     session, labels, loss.shape, timestamps.shape)
        self._cache[session] = dict(labels=labels,
                                    loss=zlib.compress(loss),
                                    loss_shape=loss.shape,
                                    timestamps=zlib.compress(timestamps),
                                    timestamps_shape=timestamps.shape)

    def _from_cache(self, session=None):
        """ Get the session data from the cache.

        If the request data does not exist in the cache, then populate it.

        Parameters
        ----------
        session: int, optional
            The Session ID to return the data for. Set to ``None`` to return all session
            data. Default ``None`

        Returns
        -------
        dict
            The session id(s) as key, with the event data as value
        """
        if session is not None and session not in self._cache:
            self._cache_data(session)
        elif session is None and not all(idx in self._cache for idx in self._log_filenames):
            for sess in self._log_filenames:
                if sess not in self._cache:
                    self._cache_data(sess)

        if session is None:
            return self._cache
        return {session: self._cache[session]}

    def get_loss(self, session=None):
        """ Read the loss from the TensorBoard event logs

        Parameters
        ----------
        session: int, optional
            The Session ID to return the loss for. Set to ``None`` to return all session
            losses. Default ``None``

        Returns
        -------
        dict
            The session id(s) as key, with a further dictionary as value containing the loss name
            and list of loss values for each step
        """
        logger.debug("Getting loss: (session: %s)", session)
        retval = dict()
        for sess, info in self._from_cache(session).items():
            arr = np.frombuffer(zlib.decompress(info["loss"]),
                                dtype="float32").reshape(info["loss_shape"])
            for idx, title in enumerate(info["labels"]):
                retval.setdefault(sess, dict())[title] = arr[:, idx]
        logger.debug({key: {k: v.shape for k, v in val.items()}
                      for key, val in retval.items()})
        return retval

    def get_timestamps(self, session=None):
        """ Read the timestamps from the TensorBoard logs.

        As loss timestamps are slightly different for each loss, we collect the timestamp from the
        `batch_total` key.

        Parameters
        ----------
        session: int, optional
            The Session ID to return the timestamps for. Set to ``None`` to return all session
            timestamps. Default ``None``

        Returns
        -------
        dict
            The session id(s) as key with list of timestamps per step as value
        """

        logger.debug("Getting timestamps")
        retval = {sess: np.frombuffer(zlib.decompress(info["timestamps"]),
                                      dtype="float64").reshape(info["timestamps_shape"])
                  for sess, info in self._from_cache(session).items()}
        logger.debug({k: v.shape for k, v in retval.items()})
        return retval


class Session():
    """ The Loaded or current training session """
    def __init__(self, model_dir=None, model_name=None):
        logger.debug("Initializing %s: (model_dir: %s, model_name: %s)",
                     self.__class__.__name__, model_dir, model_name)
        self.state = None
        self.modeldir = model_dir  # Set and reset by wrapper for training sessions
        self.modelname = model_name  # Set and reset by wrapper for training sessions
        self.tb_logs = None
        self.initialized = False
        self.session_id = None  # Set to specific session_id or current training session
        self.summary = SessionsSummary(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def batchsize(self):
        """ Return the session batchsize """
        return self.session["batchsize"]

    @property
    def config(self):
        """ Return config and other information """
        retval = self.state["config"].copy()
        retval["training_size"] = self.state["training_size"]
        retval["input_size"] = [val[0] for key, val in self.state["inputs"].items()
                                if key.startswith("face")][0]
        return retval

    @property
    def full_summary(self):
        """ Return all sessions summary data"""
        return self.summary.compile_stats()

    @property
    def iterations(self):
        """ Return session iterations """
        return self.session["iterations"]

    @property
    def logging_disabled(self):
        """ Return whether logging is disabled for this session """
        return self.session["no_logs"]

    @property
    def loss(self):
        """ dict: The loss for the current session id for each loss key """
        loss_dict = self.tb_logs.get_loss(session=self.session_id)[self.session_id]
        return loss_dict

    @property
    def loss_keys(self):
        """ list: The loss keys for the current session, or loss keys for all sessions. """
        if self.session_id is None:
            retval = self._total_loss_keys
        else:
            retval = self.session["loss_names"]
        return retval

    @property
    def lowest_loss(self):
        """ Return the lowest average loss per save iteration seen """
        return self.state["lowest_avg_loss"]

    @property
    def session(self):
        """ Return current session dictionary """
        return self.state["sessions"].get(str(self.session_id), dict())

    @property
    def session_ids(self):
        """ Return sorted list of all existing session ids in the state file """
        return sorted([int(key) for key in self.state["sessions"].keys()])

    @property
    def timestamps(self):
        """ Return timestamps from logs for current session """
        ts_dict = self.tb_logs.get_timestamps(session=self.session_id)
        return ts_dict[self.session_id]

    @property
    def total_batchsize(self):
        """ Return all session batch sizes """
        return {int(sess_id): sess["batchsize"]
                for sess_id, sess in self.state["sessions"].items()}

    @property
    def total_iterations(self):
        """ Return session iterations """
        return self.state["iterations"]

    @property
    def total_loss(self):
        """ dict: The collated loss for all sessions for each loss key """
        loss_dict = dict()
        all_loss = self.tb_logs.get_loss()
        for key in sorted(all_loss):
            for loss_key, loss in all_loss[key].items():
                loss_dict.setdefault(loss_key, []).extend(loss)
        retval = {key: np.array(val, dtype="float32") for key, val in loss_dict.items()}
        return retval

    @property
    def _total_loss_keys(self):
        """ list: The loss keys for all sessions. """
        loss_keys = set(loss_key
                        for session in self.state["sessions"].values()
                        for loss_key in session["loss_names"])
        return list(loss_keys)

    @property
    def total_timestamps(self):
        """ Return timestamps from logs separated per session for all sessions """
        return self.tb_logs.get_timestamps()

    def initialize_session(self, is_training=False, session_id=None):
        """ Initialize the training session """
        logger.debug("Initializing session: (is_training: %s, session_id: %s)",
                     is_training, session_id)
        self.load_state_file()
        self.tb_logs = TensorBoardLogs(os.path.join(self.modeldir,
                                                    "{}_logs".format(self.modelname)))
        if is_training:
            self.session_id = max(int(key) for key in self.state["sessions"].keys())
        else:
            self.session_id = session_id
        self.initialized = True
        logger.debug("Initialized session. Session_ID: %s", self.session_id)

    def load_state_file(self):
        """ Load the current state file """
        state_file = os.path.join(self.modeldir, "{}_state.json".format(self.modelname))
        logger.debug("Loading State: '%s'", state_file)
        serializer = get_serializer("json")
        self.state = serializer.load(state_file)
        logger.debug("Loaded state: %s", self.state)

    def get_iterations_for_session(self, session_id):
        """ Return the number of iterations for the given session id """
        session = self.state["sessions"].get(str(session_id), None)
        if session is None:
            logger.warning("No session data found for session id: %s", session_id)
            return 0
        return session["iterations"]


class SessionsSummary():
    """ Calculations for analysis summary stats """

    def __init__(self, session):
        logger.debug("Initializing %s: (session: %s)", self.__class__.__name__, session)
        self.session = session
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def time_stats(self):
        """ Return session time stats """
        ts_data = self.session.tb_logs.get_timestamps()
        time_stats = {sess_id: {"start_time": np.min(timestamps) if np.any(timestamps) else 0,
                                "end_time": np.max(timestamps) if np.any(timestamps) else 0,
                                "datapoints": timestamps.shape[0] if np.any(timestamps) else 0}
                      for sess_id, timestamps in ts_data.items()}
        return time_stats

    @property
    def sessions_stats(self):
        """ Return compiled stats """
        compiled = list()
        for sess_idx, ts_data in self.time_stats.items():
            logger.debug("Compiling session ID: %s", sess_idx)
            if self.session.state is None:
                logger.debug("Session state dict doesn't exist. Most likely task has been "
                             "terminated during compilation")
                return None

            iterations = self.session.get_iterations_for_session(sess_idx)
            elapsed = int(ts_data["end_time"] - ts_data["start_time"])
            batchsize = self.session.total_batchsize.get(sess_idx, 0)
            compiled.append(
                {"session": sess_idx,
                 "start": ts_data["start_time"],
                 "end": ts_data["end_time"],
                 "elapsed": elapsed,
                 "rate": ((batchsize * 2) * iterations) / elapsed if elapsed != 0 else 0,
                 "batch": batchsize,
                 "iterations": iterations})
        compiled = sorted(compiled, key=lambda k: k["session"])
        return compiled

    def compile_stats(self):
        """ Compile sessions stats with totals, format and return """
        logger.debug("Compiling sessions summary data")
        compiled_stats = self.sessions_stats
        if not compiled_stats:
            return compiled_stats
        logger.debug("sessions_stats: %s", compiled_stats)
        total_stats = self.total_stats(compiled_stats)
        compiled_stats.append(total_stats)
        compiled_stats = self.format_stats(compiled_stats)
        logger.debug("Final stats: %s", compiled_stats)
        return compiled_stats

    @classmethod
    def total_stats(cls, sessions_stats):
        """ Return total stats """
        logger.debug("Compiling Totals")
        elapsed = 0
        examples = 0
        iterations = 0
        batchset = set()
        total_summaries = len(sessions_stats)
        for idx, summary in enumerate(sessions_stats):
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

    @classmethod
    def format_stats(cls, compiled_stats):
        """ Format for display """
        logger.debug("Formatting stats")
        for summary in compiled_stats:
            hrs, mins, secs = convert_time(summary["elapsed"])
            summary["start"] = time.strftime("%x %X", time.localtime(summary["start"]))
            summary["end"] = time.strftime("%x %X", time.localtime(summary["end"]))
            summary["elapsed"] = "{}:{}:{}".format(hrs, mins, secs)
            summary["rate"] = "{0:.1f}".format(summary["rate"])
        return compiled_stats


class Calculations():
    """ Class to perform calculations on the raw data for the given session.

    Parameters
    ----------
    session: :class:`Session`
        The session to perform calculations on
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
    def __init__(self, session, session_id,
                 display="loss",
                 loss_keys="loss",
                 selections="raw",
                 avg_samples=500,
                 smooth_amount=0.90,
                 flatten_outliers=False):
        logger.debug("Initializing %s: (session: %s, session_id: %s, display: %s, loss_keys: %s, "
                     "selections: %s, avg_samples: %s, smooth_amount: %s, flatten_outliers: %s)",
                     self.__class__.__name__, session, session_id, display, loss_keys, selections,
                     avg_samples, smooth_amount, flatten_outliers)

        warnings.simplefilter("ignore", np.RankWarning)

        self._session = session
        self._session_id = session_id

        self._display = display
        self._loss_keys = loss_keys if isinstance(loss_keys, list) else [loss_keys]
        self._selections = selections if isinstance(selections, list) else [selections]
        self._is_totals = session_id is None
        self._args = dict(avg_samples=avg_samples,
                          smooth_amount=smooth_amount,
                          flatten_outliers=flatten_outliers)
        self._iterations = 0
        self._stats = None
        self.refresh()
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def iterations(self):
        """ int: The number of iterations in the data set. """
        return self._iterations

    @property
    def stats(self):
        """ dict: The final calculated statistics """
        return self._stats

    def refresh(self):
        """ Refresh the stats """
        logger.debug("Refreshing")
        if not self._session.initialized:
            logger.warning("Session data is not initialized. Not refreshing")
            return None
        old_id = self._session.session_id
        self._session.session_id = self._session_id
        self._iterations = 0
        self._stats = self._get_raw()
        self._get_calculations()
        self._remove_raw()
        self._session.session_id = old_id
        logger.debug("Refreshed")
        return self

    def _get_raw(self):
        """ Obtain the raw loss values.

        Returns
        -------
        dict
            The loss name as key with list of loss values as value
        """
        logger.debug("Getting Raw Data")
        raw = dict()
        iterations = set()
        if self._display.lower() == "loss":
            loss_dict = self._session.total_loss if self._is_totals else self._session.loss
            for loss_name, loss in loss_dict.items():
                if loss_name not in self._loss_keys:
                    continue
                if self._args["flatten_outliers"]:
                    loss = self._flatten_outliers(loss)
                iterations.add(loss.shape[0])
                raw["raw_{}".format(loss_name)] = loss

            self._iterations = 0 if not iterations else min(iterations)
            if len(iterations) > 1:
                # Crop all losses to the same number of items
                if self._iterations == 0:
                    raw = {lossname: np.array(list(), dtype=loss.dtype)
                           for lossname, loss in raw.items()}
                else:
                    raw = {lossname: loss[:self._iterations] for lossname, loss in raw.items()}

        else:  # Rate calculation
            data = self._calc_rate_total() if self._is_totals else self._calc_rate()
            if self._args["flatten_outliers"]:
                data = self._flatten_outliers(data)
            self._iterations = data.shape[0]
            raw = {"raw_rate": data}

        logger.debug("Got Raw Data")
        return raw

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
        retval = (self._session.batchsize * 2) / np.diff(self._session.timestamps)
        logger.debug("Calculated rate: Item_count: %s", len(retval))
        return retval

    def _calc_rate_total(self):
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
        batchsizes = self._session.total_batchsize
        total_timestamps = self._session.total_timestamps
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
            raw_keys = [key for key in self._stats.keys() if key.startswith("raw_")]
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
