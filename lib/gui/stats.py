#!/usr/bin python3
""" Stats functions for the GUI """

import logging
import time
import os
import warnings

from math import ceil, sqrt

import numpy as np
import tensorflow as tf
from tensorflow.python import errors_impl as tf_errors  # pylint:disable=no-name-in-module
from lib.serializer import get_serializer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def convert_time(timestamp):
    """ Convert time stamp to total hours, minutes and seconds """
    hrs = int(timestamp // 3600)
    if hrs < 10:
        hrs = "{0:02d}".format(hrs)
    mins = "{0:02d}".format((int(timestamp % 3600) // 60))
    secs = "{0:02d}".format((int(timestamp % 3600) % 60))
    return hrs, mins, secs


class TensorBoardLogs():
    """ Parse and return data from TensorBoard logs """
    def __init__(self, logs_folder):
        self.folder_base = logs_folder
        self.log_filenames = self.set_log_filenames()

    def set_log_filenames(self):
        """ Set the TensorBoard log filenames for all existing sessions """
        logger.debug("Loading log filenames. base_dir: '%s'", self.folder_base)
        log_filenames = dict()
        for dirpath, _, filenames in os.walk(self.folder_base):
            if not any(filename.startswith("events.out.tfevents") for filename in filenames):
                continue
            logfiles = [filename for filename in filenames
                        if filename.startswith("events.out.tfevents")]
            # Take the last logfile, in case of previous crash
            logfile = os.path.join(dirpath, sorted(logfiles)[-1])
            side, session = os.path.split(dirpath)
            side = os.path.split(side)[1]
            session = int(session[session.rfind("_") + 1:])
            log_filenames.setdefault(session, dict())[side] = logfile
        logger.debug("logfiles: %s", log_filenames)
        return log_filenames

    def get_loss(self, side=None, session=None):
        """ Read the loss from the TensorBoard logs
            Specify a side or a session or leave at None for all
        """
        logger.debug("Getting loss: (side: %s, session: %s)", side, session)
        all_loss = dict()
        for sess, sides in self.log_filenames.items():
            if session is not None and sess != session:
                logger.debug("Skipping session: %s", sess)
                continue
            loss = dict()
            for sde, logfile in sides.items():
                if side is not None and sde != side:
                    logger.debug("Skipping side: %s", sde)
                    continue
                for event in tf.train.summary_iterator(logfile):
                    for summary in event.summary.value:
                        if "loss" not in summary.tag:
                            continue
                        tag = summary.tag.replace("batch_", "")
                        loss.setdefault(tag,
                                        dict()).setdefault(sde,
                                                           list()).append(summary.simple_value)
            all_loss[sess] = loss
        return all_loss

    def get_timestamps(self, session=None):
        """ Read the timestamps from the TensorBoard logs
            Specify a session or leave at None for all
            NB: For all intents and purposes timestamps are the same for
                both sides, so just read from one side """
        logger.debug("Getting timestamps")
        all_timestamps = dict()
        for sess, sides in self.log_filenames.items():
            if session is not None and sess != session:
                logger.debug("Skipping sessions: %s", sess)
                continue
            try:
                for logfile in sides.values():
                    timestamps = [event.wall_time
                                  for event in tf.train.summary_iterator(logfile)
                                  if event.summary.value]
                    logger.debug("Total timestamps for session %s: %s", sess, len(timestamps))
                    all_timestamps[sess] = timestamps
                    break  # break after first file read
            except tf_errors.DataLossError as err:
                logger.warning("The logs for Session %s are corrupted and cannot be displayed. "
                               "The totals do not include this session. Original error message: "
                               "'%s'", sess, str(err))
        return all_timestamps


class Session():
    """ The Loaded or current training session """
    def __init__(self, model_dir=None, model_name=None):
        logger.debug("Initializing %s: (model_dir: %s, model_name: %s)",
                     self.__class__.__name__, model_dir, model_name)
        self.serializer = get_serializer("json")
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
        """ Retun all sessions summary data"""
        return self.summary.compile_stats()

    @property
    def iterations(self):
        """ Return session iterations """
        return self.session["iterations"]

    @property
    def logging_disabled(self):
        """ Return whether logging is disabled for this session """
        return self.session["no_logs"] or self.session["pingpong"]

    @property
    def loss(self):
        """ Return loss from logs for current session """
        loss_dict = self.tb_logs.get_loss(session=self.session_id)[self.session_id]
        return loss_dict

    @property
    def loss_keys(self):
        """ Return list of unique session loss keys """
        if self.session_id is None:
            loss_keys = self.total_loss_keys
        else:
            loss_keys = set(loss_key for side_keys in self.session["loss_names"].values()
                            for loss_key in side_keys)
        return list(loss_keys)

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
        """ Return collated loss for all session """
        loss_dict = dict()
        all_loss = self.tb_logs.get_loss()
        for key in sorted(int(idx) for idx in all_loss):
            for loss_key, side_loss in all_loss[key].items():
                for side, loss in side_loss.items():
                    loss_dict.setdefault(loss_key, dict()).setdefault(side, list()).extend(loss)
        return loss_dict

    @property
    def total_loss_keys(self):
        """ Return list of unique session loss keys across all sessions """
        loss_keys = set(loss_key
                        for session in self.state["sessions"].values()
                        for loss_keys in session["loss_names"].values()
                        for loss_key in loss_keys)
        return list(loss_keys)

    @property
    def total_timestamps(self):
        """ Return timestamps from logs seperated per session for all sessions """
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
        self.state = self.serializer.load(state_file)
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
        time_stats = {sess_id: {"start_time": min(timestamps) if timestamps else 0,
                                "end_time": max(timestamps) if timestamps else 0,
                                "datapoints": len(timestamps) if timestamps else 0}
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
            elapsed = ts_data["end_time"] - ts_data["start_time"]
            batchsize = self.session.total_batchsize.get(sess_idx, 0)
            compiled.append({"session": sess_idx,
                             "start": ts_data["start_time"],
                             "end": ts_data["end_time"],
                             "elapsed": elapsed,
                             "rate": (batchsize * iterations) / elapsed if elapsed != 0 else 0,
                             "batch": batchsize,
                             "iterations": iterations})
        compiled = sorted(compiled, key=lambda k: k["session"])
        return compiled

    def compile_stats(self):
        """ Compile sessions stats with totals, format and return """
        logger.debug("Compiling sessions summary data")
        compiled_stats = self.sessions_stats
        if compiled_stats is None:
            return compiled_stats
        logger.debug("sessions_stats: %s", compiled_stats)
        total_stats = self.total_stats(compiled_stats)
        compiled_stats.append(total_stats)
        compiled_stats = self.format_stats(compiled_stats)
        logger.debug("Final stats: %s", compiled_stats)
        return compiled_stats

    @staticmethod
    def total_stats(sessions_stats):
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
            examples += (summary["batch"] * summary["iterations"])
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

    @staticmethod
    def format_stats(compiled_stats):
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
    """ Class to pull raw data for given session(s) and perform calculations """
    def __init__(self, session, display="loss", loss_keys=["loss"], selections=["raw"],
                 avg_samples=500, smooth_amount=0.90, flatten_outliers=False, is_totals=False):
        logger.debug("Initializing %s: (session: %s, display: %s, loss_keys: %s, selections: %s, "
                     "avg_samples: %s, smooth_amount: %s, flatten_outliers: %s, is_totals: %s",
                     self.__class__.__name__, session, display, loss_keys, selections, avg_samples,
                     smooth_amount, flatten_outliers, is_totals)

        warnings.simplefilter("ignore", np.RankWarning)

        self.session = session
        self.display = display
        self.loss_keys = loss_keys
        self.selections = selections
        self.is_totals = is_totals
        self.args = {"avg_samples": avg_samples,
                     "smooth_amount": smooth_amount,
                     "flatten_outliers": flatten_outliers}
        self.iterations = 0
        self.stats = None
        self.refresh()
        logger.debug("Initialized %s", self.__class__.__name__)

    def refresh(self):
        """ Refresh the stats """
        logger.debug("Refreshing")
        if not self.session.initialized:
            logger.warning("Session data is not initialized. Not refreshing")
            return None
        self.iterations = 0
        self.stats = self.get_raw()
        self.get_calculations()
        self.remove_raw()
        logger.debug("Refreshed")
        return self

    def get_raw(self):
        """ Add raw data to stats dict """
        logger.debug("Getting Raw Data")

        raw = dict()
        iterations = set()
        if self.display.lower() == "loss":
            loss_dict = self.session.total_loss if self.is_totals else self.session.loss
            for loss_name, side_loss in loss_dict.items():
                if loss_name not in self.loss_keys:
                    continue
                for side, loss in side_loss.items():
                    if self.args["flatten_outliers"]:
                        loss = self.flatten_outliers(loss)
                    iterations.add(len(loss))
                    raw["raw_{}_{}".format(loss_name, side)] = loss

            self.iterations = 0 if not iterations else min(iterations)
            if len(iterations) > 1:
                # Crop all losses to the same number of items
                if self.iterations == 0:
                    raw = {lossname: list() for lossname in raw}
                else:
                    raw = {lossname: loss[:self.iterations] for lossname, loss in raw.items()}

        else:  # Rate calulation
            data = self.calc_rate_total() if self.is_totals else self.calc_rate()
            if self.args["flatten_outliers"]:
                data = self.flatten_outliers(data)
            self.iterations = len(data)
            raw = {"raw_rate": data}

        logger.debug("Got Raw Data")
        return raw

    def remove_raw(self):
        """ Remove raw values from stats if not requested """
        if "raw" in self.selections:
            return
        logger.debug("Removing Raw Data from output")
        for key in list(self.stats.keys()):
            if key.startswith("raw"):
                del self.stats[key]
        logger.debug("Removed Raw Data from output")

    def calc_rate(self):
        """ Calculate rate per iteration """
        logger.debug("Calculating rate")
        batchsize = self.session.batchsize
        timestamps = self.session.timestamps
        iterations = range(len(timestamps) - 1)
        rate = [batchsize / (timestamps[i + 1] - timestamps[i]) for i in iterations]
        logger.debug("Calculated rate: Item_count: %s", len(rate))
        return rate

    def calc_rate_total(self):
        """ Calculate rate per iteration
            NB: For totals, gaps between sessions can be large
            so time difference has to be reset for each session's
            rate calculation """
        logger.debug("Calculating totals rate")
        batchsizes = self.session.total_batchsize
        total_timestamps = self.session.total_timestamps
        rate = list()
        for sess_id in sorted(total_timestamps.keys()):
            batchsize = batchsizes[sess_id]
            timestamps = total_timestamps[sess_id]
            iterations = range(len(timestamps) - 1)
            print("===========\n")
            print(timestamps[:100])
            print([batchsize / (timestamps[i + 1] - timestamps[i]) for i in iterations][:100])
            rate.extend([batchsize / (timestamps[i + 1] - timestamps[i]) for i in iterations])
        logger.debug("Calculated totals rate: Item_count: %s", len(rate))
        return rate

    @staticmethod
    def flatten_outliers(data):
        """ Remove the outliers from a provided list """
        logger.debug("Flattening outliers")
        retdata = list()
        samples = len(data)
        mean = (sum(data) / samples)
        limit = sqrt(sum([(item - mean)**2 for item in data]) / samples)
        logger.debug("samples: %s, mean: %s, limit: %s", samples, mean, limit)

        for idx, item in enumerate(data):
            if (mean - limit) <= item <= (mean + limit):
                retdata.append(item)
            else:
                logger.trace("Item idx: %s, value: %s flattened to %s", idx, item, mean)
                retdata.append(mean)
        logger.debug("Flattened outliers")
        return retdata

    def get_calculations(self):
        """ Perform the required calculations """
        for selection in self.selections:
            if selection == "raw":
                continue
            logger.debug("Calculating: %s", selection)
            method = getattr(self, "calc_{}".format(selection))
            raw_keys = [key for key in self.stats.keys() if key.startswith("raw_")]
            for key in raw_keys:
                selected_key = "{}_{}".format(selection, key.replace("raw_", ""))
                self.stats[selected_key] = method(self.stats[key])

    def calc_avg(self, data):
        """ Calculate rolling average """
        logger.debug("Calculating Average")
        avgs = list()
        presample = ceil(self.args["avg_samples"] / 2)
        postsample = self.args["avg_samples"] - presample
        datapoints = len(data)

        if datapoints <= (self.args["avg_samples"] * 2):
            logger.info("Not enough data to compile rolling average")
            return avgs

        for idx in range(0, datapoints):
            if idx < presample or idx >= datapoints - postsample:
                avgs.append(None)
                continue
            avg = sum(data[idx - presample:idx + postsample]) / self.args["avg_samples"]
            avgs.append(avg)
        logger.debug("Calculated Average")
        return avgs

    def calc_smoothed(self, data):
        """ Smooth the data """
        last = data[0]  # First value in the plot (first timestep)
        weight = self.args["smooth_amount"]
        smoothed = list()
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value

        return smoothed

    @staticmethod
    def calc_trend(data):
        """ Compile trend data """
        logger.debug("Calculating Trend")
        points = len(data)
        if points < 10:
            dummy = [None for i in range(points)]
            return dummy
        x_range = range(points)
        fit = np.polyfit(x_range, data, 3)
        poly = np.poly1d(fit)
        trend = poly(x_range)
        logger.debug("Calculated Trend")
        return trend
