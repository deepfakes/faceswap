#!/usr/bin python3
""" Stats functions for the GUI """

import time
import os

from math import ceil, sqrt

import numpy as np

from lib.Serializer import PickleSerializer

class SavedSessions(object):
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
        print('Saved session stats to: {}'.format(filename))

class CurrentSession(object):
    """ The current training session """
    def __init__(self):
        self.stats = {'iterations': 0,
                      'batchsize': None,    # Set and reset by wrapper
                      'timestamps': [],
                      'loss': [],
                      'losskeys': []}
        self.timestats = {'start': None,
                          'elapsed': None}
        self.modeldir = None    # Set and reset by wrapper
        self.filename = None
        self.historical = None

    def initialise_session(self, currentloss):
        """ Initialise the training session """
        self.load_historical()
        for item in currentloss:
            self.stats['losskeys'].append(item[0])
            self.stats['loss'].append(list())
        self.timestats['start'] = time.time()

    def load_historical(self):
        """ Load historical data and add current session to the end """
        self.filename = os.path.join(self.modeldir, 'trainingstats.fss')
        self.historical = SavedSessions(self.filename)
        self.historical.sessions.append(self.stats)

    def add_loss(self, currentloss):
        """ Add a loss item from the training process """
        if self.stats['iterations'] == 0:
            self.initialise_session(currentloss)

        self.stats['iterations'] += 1
        self.add_timestats()

        for idx, item in enumerate(currentloss):
            self.stats['loss'][idx].append(float(item[1]))

    def add_timestats(self):
        """ Add timestats to loss dict and timestats """
        now = time.time()
        self.stats['timestamps'].append(now)
        elapsed_time = now - self.timestats['start']
        self.timestats['elapsed'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    def save_session(self):
        """ Save the session file to the modeldir """
        print('Saving session stats...')
        self.historical.save_sessions(self.filename)

class SessionsTotals(object):
    """ The compiled totals of all saved sessions """
    def __init__(self, all_sessions):
        self.stats = {'split': [],
                      'iterations': 0,
                      'batchsize': [],
                      'timestamps': [],
                      'loss': [],
                      'losskeys': []}

        self.initiate(all_sessions)
        self.compile(all_sessions)

    def initiate(self, sessions):
        """ Initiate correct losskey titles and number of loss lists """
        for losskey in sessions[0]['losskeys']:
            self.stats['losskeys'].append(losskey)
            self.stats['loss'].append(list())

    def compile(self, sessions):
        """ Compile all of the sessions into totals """
        current_split = 0
        for session in sessions:
            iterations = session['iterations']
            current_split += iterations
            self.stats['split'].append(current_split)
            self.stats['iterations'] += iterations
            self.stats['timestamps'].extend(session['timestamps'])
            self.stats['batchsize'].append(session['batchsize'])
            self.add_loss(session['loss'])

    def add_loss(self, session_loss):
        """ Add loss vals to each of their respective lists """
        for idx, loss in enumerate(session_loss):
            self.stats['loss'][idx].extend(loss)


class SessionsSummary(object):
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
        starttime = session['timestamps'][0]
        endtime = session['timestamps'][-1]
        elapsed = endtime - starttime
        rate = (session['batchsize'] * session['iterations']) / elapsed
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
                starttime = summary['start']
            if idx == total_summaries - 1:
                endtime = summary['end']
            elapsed += summary['elapsed']
            rate += summary['rate']
            batchset.add(summary['batch'])
            iterations += summary['iterations']
        batch = ','.join(str(bs) for bs in batchset)

        return {"session": 'Total',
                "start": starttime,
                "end": endtime,
                "elapsed": elapsed,
                "rate": rate / total_summaries,
                "batch": batch,
                "iterations": iterations}

    def format_summaries(self, raw_summaries):
        """ Format the summaries nicely for display """
        for summary in raw_summaries:
            summary['start'] = time.strftime("%x %X", time.gmtime(summary['start']))
            summary['end'] = time.strftime("%x %X", time.gmtime(summary['end']))
            summary["elapsed"] = time.strftime("%H:%M:%S", time.gmtime(summary["elapsed"]))
            summary["rate"] = '{0:.1f}'.format(summary["rate"])
        self.summary = raw_summaries

class Calculations(object):
    """ Class to hold calculations against raw session data """
    def __init__(self,
                 session,
                 display='loss',
                 selections=['raw'],
                 avg_samples=10,
                 remove_outliers=False,
                 is_totals=False):

        display = session['losskeys'] if display.lower() == 'loss' else [display]
        self.iterations = 0
        self.avg_samples = int(avg_samples)
        self.stats = self.get_raw(display, session, is_totals, remove_outliers)

        self.get_calculations(display, selections)
        self.remove_raw(selections)

    def get_raw(self, display, session, is_totals, remove_outliers):
        """ Add raw data to stats dict """
        raw = dict()
        for idx, item in enumerate(display):
            if item.lower() == 'rate':
                data = self.calc_rate(session, is_totals)
            else:
                data = session['loss'][idx]

            if remove_outliers:
                data = self.remove_outliers(data)

            if self.iterations == 0:
                self.iterations = len(data)

            raw['raw_{}'.format(item)] = data
        return raw

    def remove_raw(self, selections):
        """ Remove raw values from stats if not requested """
        if 'raw' in selections:
            return
        for key in self.stats.keys():
            if key.startswith('raw'):
                del self.stats[key]

    def calc_rate(self, data, is_totals):
        """ Calculate rate per iteration
            NB: For totals, gaps between sessions can be large
            so time diffeence has to be reset for each session's
            rate calculation """
        timestamp = data['timestamps']
        batchsize = data['batchsize']
        if is_totals:
            split = data['split']
        else:
            batchsize = [batchsize]
            split = [len(timestamp)]

        total = 0
        rate = list()

        for idx, current_split in enumerate(split):
            prev_time = timestamp[total]
            timestamp_chunk = timestamp[total:total + current_split]
            for item in timestamp_chunk:
                current_time = item
                timediff = current_time - prev_time
                iter_rate = 0 if timediff == 0 else batchsize[idx] / timediff
                rate.append(iter_rate)
                prev_time = current_time
            total += current_split

        if self.remove_outliers != 0:
            rate = self.remove_outliers(rate)
        return rate

    @staticmethod
    def remove_outliers(data):
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

    def get_calculations(self, display, summaries):
        """ Perform the required calculations """
        for selection in self.get_selections(display, summaries):
            if selection[0] == 'raw':
                continue
            method = getattr(self, 'calc_{}'.format(selection[0]))
            key = '{}_{}'.format(selection[0], selection[1])
            raw = self.stats['raw_{}'.format(selection[1])]
            self.stats[key] = method(raw)

    @staticmethod
    def get_selections(display, summaries):
        """ Compile a list of data to be calculated """
        for summary in summaries:
            for item in display:
                yield summary, item

    def calc_avg(self, data):
        """ Calculate rolling average """
        avgs = list()
        presample = ceil(int(self.avg_samples) / 2)
        postsample = int(self.avg_samples) - presample
        datapoints = len(data)

        if datapoints <= (self.avg_samples * 2):
            print("Not enough data to compile rolling average")
            return avgs

        for idx in range(0, datapoints):
            if idx < presample or idx >= datapoints - postsample:
                avgs.append(None)
                continue
            else:
                avg = sum(data[idx - presample:idx + postsample]) / self.avg_samples
                avgs.append(avg)
        return avgs

    @staticmethod
    def calc_trend(data):
        """ Compile trend data """
        x_range = range(len(data))
        fit = np.polyfit(x_range, data, 3)
        poly = np.poly1d(fit)
        trend = poly(x_range)
        return trend
