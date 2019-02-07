#!/usr/bin python3
""" Stats functions for the GUI """

import logging
import time
import os
import warnings

from math import ceil, sqrt

import numpy as np
import tensorflow as tf
from lib.Serializer import JSONSerializer, PickleSerializer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def convert_time(timestamp):
    """ Convert time stamp to total hours, minutes and seconds """
    hrs = int(timestamp // 3600)
    if hrs < 10:
        hrs = "{0:02d}".format(hrs)
    mins = "{0:02d}".format((int(timestamp % 3600) // 60))
    secs = "{0:02d}".format((int(timestamp % 3600) % 60))
    return hrs, mins, secs


class SavedSessions():
    """ Saved Training Session """
    def __init__(self, sessions_data):
        self.serializer = PickleSerializer
        self.sessions = self.load_sessions(sessions_data)

    def load_sessions(self, filename):
        """ Load previously saved sessions """
        stats = list()
        if os.path.isfile(filename):
            with open(filename, self.serializer.roptions) as sessions:
                stats = self.serializer.unmarshal(sessions.read())
        return stats

    def save_sessions(self, filename):
        """ Save the session file  """
        with open(filename, self.serializer.woptions) as session:
            session.write(self.serializer.marshal(self.sessions))
        logger.info("Saved session stats to: %s", filename)


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
        for sess, sides in self.log_filenames.values():
            if session is not None and sess != session:
                logger.debug("Skipping sessions: %s", sess)
                continue
            timestamps = list()
            for logfile in sides.values():
                timestamps.append([event.wall_time
                                   for event in tf.train.summary_iterator(logfile)])
                logger.debug("Total timestamps for session %s: %s", sess, len(timestamps))
                all_timestamps[sess] = timestamps
            break
        return all_timestamps


class Session():
    """ The Loaded or current training session """
    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.serializer = JSONSerializer
        self.state = None
        self.modeldir = None  # Set and reset by wrapper
        self.modelname = None  # Set and reset by wrapper
        self.logs_disabled = False  # Set and reset by wrapper
        self.tb_logs = None
        self.initialized = False
        self.session_id = None  # Set to specific session_id or current training session
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def batchsize(self):
        """ Return the session batchsize """
        return self.session["batchsize"]

    @property
    def config(self):
        """ Return config and other information """
        retval = {key: val for key, val in self.state["config"]}
        retval["training_size"] = self.state["training_size"]
        retval["input_size"] = [val[0] for key, val in self.state["inputs"].items()
                                if key.startswith("face")][0]
        return retval

    @property
    def iterations(self):
        """ Return session iterations """
        return self.session["iterations"]

    @property
    def total_iterations(self):
        """ Return session iterations """
        return self.state["iterations"]

    @property
    def loss(self):
        """ Return loss from logs for current session """
        loss_dict = self.tb_logs.get_loss(session=self.session_id)[self.session_id]
        return loss_dict

    @property
    def loss_keys(self):
        """ Return list of unique session loss keys """
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
        return self.state["sessions"][str(self.session_id)]

    @property
    def timestamps(self):
        """ Return timestamps from logs for current session """
        ts_dict = self.tb_logs.get_timestamps(session=self.session_id)
        return list(ts_dict.values())

    def initialize_session(self, is_training=False, session_id=None):
        """ Initialize the training session """
        logger.debug("Initializing session: (is_training: %s, session_id: %s",
                     is_training, session_id)
        self.load_state_file()
        self.tb_logs = TensorBoardLogs(os.path.join(self.modeldir,
                                                    "{}_logs".format(self.modelname)))
        if is_training:
            self.session_id = max(int(key) for key in self.state["sessions"].keys())
        else:
            self.session_id = session_id
        self.initialized = True
        logger.debug("Initialized session")

    def load_state_file(self):
        """ Load the current state file """
        state_file = os.path.join(self.modeldir, "{}_state.json".format(self.modelname))
        logger.debug("Loading State: '%s'", state_file)
        try:
            with open(state_file, "rb") as inp:
                state = self.serializer.unmarshal(inp.read().decode("utf-8"))
                self.state = state
                logger.debug("Loaded state: %s", state)
        except IOError as err:
            logger.warning("Unable to load state file. Graphing disabled: %s", str(err))


class SessionsTotals():
    """ The compiled totals of all saved sessions """
    def __init__(self, all_sessions):
        self.stats = {"split": [],
                      "iterations": 0,
                      "batchsize": [],
                      "timestamps": [],
                      "loss": [],
                      "losskeys": []}

        self.initiate(all_sessions)
        self.compile(all_sessions)

    def initiate(self, sessions):
        """ Initiate correct loss key titles and number of loss lists """
        for losskey in sessions[0]["losskeys"]:
            self.stats["losskeys"].append(losskey)
            self.stats["loss"].append(list())

    def compile(self, sessions):
        """ Compile all of the sessions into totals """
        current_split = 0
        for session in sessions:
            iterations = session["iterations"]
            current_split += iterations
            self.stats["split"].append(current_split)
            self.stats["iterations"] += iterations
            self.stats["timestamps"].extend(session["timestamps"])
            self.stats["batchsize"].append(session["batchsize"])
            self.add_loss(session["loss"])

    def add_loss(self, session_loss):
        """ Add loss values to each of their respective lists """
        for idx, loss in enumerate(session_loss):
            self.stats["loss"][idx].extend(loss)


class SessionsSummary():
    """ Calculations for analysis summary stats """

    def __init__(self, raw_data):
        self.summary = list()
        self.summary_stats_compile(raw_data)

    def summary_stats_compile(self, raw_data):
        """ Compile summary stats """
        raw_summaries = list()
        for idx, session in enumerate(raw_data):
            raw_summaries.append(self.summarise_session(idx, session))

        totals_summary = self.summarise_totals(raw_summaries)
        raw_summaries.append(totals_summary)
        self.format_summaries(raw_summaries)

    # Compile Session Summaries
    @staticmethod
    def summarise_session(idx, session):
        """ Compile stats for session passed in """
        starttime = session["timestamps"][0]
        endtime = session["timestamps"][-1]
        elapsed = endtime - starttime
        # Bump elapsed to 0.1s if no time is recorded
        # to hack around div by zero error
        elapsed = 0.1 if elapsed == 0 else elapsed
        rate = (session["batchsize"] * session["iterations"]) / elapsed
        return {"session": idx + 1,
                "start": starttime,
                "end": endtime,
                "elapsed": elapsed,
                "rate": rate,
                "batch": session["batchsize"],
                "iterations": session["iterations"]}

    @staticmethod
    def summarise_totals(raw_summaries):
        """ Compile the stats for all sessions combined """
        elapsed = 0
        rate = 0
        batchset = set()
        iterations = 0
        total_summaries = len(raw_summaries)

        for idx, summary in enumerate(raw_summaries):
            if idx == 0:
                starttime = summary["start"]
            if idx == total_summaries - 1:
                endtime = summary["end"]
            elapsed += summary["elapsed"]
            rate += summary["rate"]
            batchset.add(summary["batch"])
            iterations += summary["iterations"]
        batch = ",".join(str(bs) for bs in batchset)

        return {"session": "Total",
                "start": starttime,
                "end": endtime,
                "elapsed": elapsed,
                "rate": rate / total_summaries,
                "batch": batch,
                "iterations": iterations}

    def format_summaries(self, raw_summaries):
        """ Format the summaries nicely for display """
        for summary in raw_summaries:
            summary["start"] = time.strftime("%x %X",
                                             time.gmtime(summary["start"]))
            summary["end"] = time.strftime("%x %X",
                                           time.gmtime(summary["end"]))
            hrs, mins, secs = convert_time(summary["elapsed"])
            summary["elapsed"] = "{}:{}:{}".format(hrs, mins, secs)
            summary["rate"] = "{0:.1f}".format(summary["rate"])
        self.summary = raw_summaries


class Calculations():
    """ Class to pull raw data for given session(s) and perform calculations """
    def __init__(self, session, display="loss", loss_keys=["loss"], selections=["raw"],
                 avg_samples=10, flatten_outliers=False, is_totals=False):

        warnings.simplefilter("ignore", np.RankWarning)

        self.session = session
        self.display = display
        self.loss_keys = loss_keys
        self.selections = selections
        self.args = {"avg_samples": int(avg_samples),
                     "flatten_outliers": flatten_outliers,
                     "is_totals": is_totals}
        self.iterations = 0
        self.stats = None
        self.refresh()

    def refresh(self):
        """ Refresh the stats """
        self.iterations = 0
        self.stats = self.get_raw()
        self.get_calculations()
        self.remove_raw()

    def get_raw(self):
        """ Add raw data to stats dict """
#        raw = dict()
        raw = dict()
        iterations = set()
        if self.display.lower() == "loss":
            for loss_name, side_loss in self.session.loss.items():
                for side, loss in side_loss.items():
                    iterations.add(len(loss))
                    raw["raw_{}_{}".format(loss_name, side)] = loss

        self.iterations = min(iterations)
        if len(iterations) != 1:
            # Crop all losses to the same number of items
            raw = {lossname: loss[:self.iterations] for lossname, loss in raw}

#        raw = dict()
#        for idx, item in enumerate(self.args["display"]):
#            if item.lower() == "rate":
#                data = self.calc_rate(self.session)
#            else:
#                data = self.session["loss"][idx][:]

#            if self.args["flatten_outliers"]:
#                data = self.flatten_outliers(data)

#            if self.iterations == 0:
#                self.iterations = len(data)

#            raw["raw_{}".format(item)] = data
        return raw

    def remove_raw(self):
        """ Remove raw values from stats if not requested """
        if "raw" in self.selections:
            return
        for key in list(self.stats.keys()):
            if key.startswith("raw"):
                del self.stats[key]

    def calc_rate(self, data):
        """ Calculate rate per iteration
            NB: For totals, gaps between sessions can be large
            so time difference has to be reset for each session's
            rate calculation """
        batchsize = data["batchsize"]
        if self.args["is_totals"]:
            split = data["split"]
        else:
            batchsize = [batchsize]
            split = [len(data["timestamps"])]

        prev_split = 0
        rate = list()

        for idx, current_split in enumerate(split):
            prev_time = data["timestamps"][prev_split]
            timestamp_chunk = data["timestamps"][prev_split:current_split]
            for item in timestamp_chunk:
                current_time = item
                timediff = current_time - prev_time
                iter_rate = 0 if timediff == 0 else batchsize[idx] / timediff
                rate.append(iter_rate)
                prev_time = current_time
            prev_split = current_split

        if self.args["flatten_outliers"]:
            rate = self.flatten_outliers(rate)
        return rate

    @staticmethod
    def flatten_outliers(data):
        """ Remove the outliers from a provided list """
        retdata = list()
        samples = len(data)
        mean = (sum(data) / samples)
        limit = sqrt(sum([(item - mean)**2 for item in data]) / samples)

        for item in data:
            if (mean - limit) <= item <= (mean + limit):
                retdata.append(item)
            else:
                retdata.append(mean)
        return retdata

    def get_calculations(self):
        """ Perform the required calculations """
        for selection in self.selections:
            if selection == "raw":
                continue
            method = getattr(self, "calc_{}".format(selection))
            raw_keys = [key for key in self.stats.keys() if key.startswith("raw_")]
            for key in raw_keys:
                selected_key = "{}_{}".format(selection, key.replace("raw_", ""))
                self.stats[selected_key] = method(self.stats[key])

#    def get_selections(self):
#        """ Compile a list of data to be calculated """
#        if self.display == "loss":
#            process = self.loss_keys
#        else:
#            process = ["rate"]
#        for summary in self.selections:
#            for item in process:
#                yield summary, item

    def calc_avg(self, data):
        """ Calculate rolling average """
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
            else:
                avg = sum(data[idx - presample:idx + postsample]) \
                        / self.args["avg_samples"]
                avgs.append(avg)
        return avgs

    @staticmethod
    def calc_trend(data):
        """ Compile trend data """
        points = len(data)
        if points < 10:
            dummy = [None for i in range(points)]
            return dummy
        x_range = range(points)
        fit = np.polyfit(x_range, data, 3)
        poly = np.poly1d(fit)
        trend = poly(x_range)
        return trend
