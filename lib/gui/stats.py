#!/usr/bin python3
""" Stats functions for the GUI """

import time
import os

from math import ceil

import numpy as np

from lib.Serializer import PickleSerializer

def mean(numbers):
    """ Calculate the mean of a given list of numbers """
    return sum(numbers) / max(len(numbers), 1)

class CurrentSession(object):
    """ The current training session """
    def __init__(self):
        self.lossdict = {'timestamp': []}
        self.iterations = 0
        self.timestats = {'start': None, 'elapsed': None}
        self.batchsize = None   # Set and reset by wrapper
        self.modeldir = None    # Set and reset by wrapper
        self.filename = None
        self.serializer = PickleSerializer
        self.historical = None

    def initialise_session(self, keys):
        """ Initialise the training session """
        self.filename = os.path.join(self.modeldir, 'trainingstats.fss')
        self.historical = SavedSessions(self.filename)
        self.lossdict.update((key, []) for key in keys)
        self.timestats['start'] = time.time()

    def add_loss(self, currentloss):
        """ Add a loss item from the training process """
        if self.iterations == 0:
            keys = (item[0] for item in currentloss)
            self.initialise_session(keys)

        self.iterations += 1
        self.add_timestats()

        for item in currentloss:
            self.lossdict[item[0]].append(float(item[1]))

    def add_timestats(self):
        """ Add timestats to loss dict and timestats """
        now = time.time()
        self.lossdict['timestamp'].append(now)
        elapsed_time = now - self.timestats['start']
        self.timestats['elapsed'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    def compile_session(self):
        """ Compile all the session stats """
        sessionid = len(self.historical.sessions) + 1
        starttime = min(self.lossdict['timestamp'])
        endtime = max(self.lossdict['timestamp'])
        session = {sessionid: {'loss': self.lossdict,
                               'batchsize': self.batchsize,
                               'iterations': self.iterations,
                               'starttime': starttime,
                               'endtime': endtime}}
        return session

    def save_session(self):
        """ Save the session file to the modeldir """
        print('Saving session stats...')
        stats = self.compile_session()
        self.historical.sessions.update(stats)
        self.historical.save_sessions(self.filename)

class SavedSessions(object):
    """ Saved Training Session """
    def __init__(self, sessions_data):
        self.serializer = PickleSerializer
        self.sessions = self.load_sessions(sessions_data)

    def load_sessions(self, filename):
        """ Load previously saved sessions """
        stats = dict()
        if os.path.isfile(filename):
            with open(filename, self.serializer.roptions) as sessions:
                stats = self.serializer.unmarshal(sessions.read())
        return stats

    def save_sessions(self, filename):
        """ Save the session file  """
        with open(filename, self.serializer.woptions) as session:
            session.write(self.serializer.marshal(self.sessions))
        print('Saved session stats to: {}'.format(filename))

class SessionsSummary(object):
    """ Calculations for stats """

    def __init__(self, sessions):
        self.sessions = sessions
        self.summary = self.summary_stats_compile()

    # TOP BOX SUMMARY STATS
    def summary_stats_compile(self):
        """ Compile summary stats """
        summary = dict()
        totals = dict()

        for key, val in self.sessions.items():
            self.summary_totals(val, totals)
            summary[key] = self.summary_session(key, val)
        self.summary_add_totals(totals, summary)
        return summary

    # Compile Totals
    def summary_totals(self, values, totals):
        """ Add passed in session to summary totals """
        for key, val in values.items():
            if key == 'batchsize':
                item = self.summary_totals_batchsize(totals.get(key, None), val)
            elif key.endswith('time'):
                item = self.summary_totals_time(key, totals.get(key, None), val)
            elif key != 'loss':
                item = totals.get(key, 0)
                item += val

            if key != 'loss':
                totals[key] = item
        totals['rate'] = self.summary_totals_rate(values['starttime'],
                                                  values['endtime'],
                                                  values['batchsize'],
                                                  values['iterations'],
                                                  totals.get('rate', 0))
        totals['elapsed'] = self.summary_totals_elapsed(values['starttime'],
                                                        values['endtime'],
                                                        totals.get('elapsed', 0))

    @staticmethod
    def summary_totals_rate(starttime, endtime, batchsize, iterations, currentrate):
        """ Update totals batchsize """
        elapsed = endtime - starttime
        rate = (batchsize * iterations) / elapsed
        return currentrate + rate

    @staticmethod
    def summary_totals_elapsed(starttime, endtime, currentelapsed):
        """ Update totals batchsize """
        elapsed = endtime - starttime
        return elapsed + currentelapsed

    @staticmethod
    def summary_totals_batchsize(current, new):
        """ Update totals batchsize """
        retval = new if not current else current
        retval = 'Varies' if retval != new else retval
        return retval

    @staticmethod
    def summary_totals_time(key, current, new):
        """ Update totals start or end time """
        retval = new if not current else current
        retval = new if key == 'starttime' and new < retval else retval
        retval = new if key == 'endtime' and new > retval else retval
        return retval

    # Compile Session Summaries
    @staticmethod
    def summary_session(key, value):
        """ Compile stats for session passed in """
        elapsed = value['endtime'] - value['starttime']
        rate = (value['batchsize'] * value['iterations']) / elapsed
        return {"session": key,
                "start": time.strftime("%x %X", time.gmtime(value['starttime'])),
                "end": time.strftime("%x %X", time.gmtime(value['endtime'])),
                "elapsed": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
                "rate": '{0:.1f}'.format(rate),
                "batch": value["batchsize"],
                "iterations": value["iterations"]}

    @staticmethod
    def summary_add_totals(totals, summary):
        """ Add the total of all sessions for the summary """
        sessions = len(summary)
        total_rate = totals['rate'] / sessions
        start = time.gmtime(totals['starttime'])
        end = time.gmtime(totals['endtime'])
        elapsed = time.gmtime(totals['elapsed'])
        summary[sessions + 1] = {'session': 'Total',
                                 'start': time.strftime("%x %X", start),
                                 'end': time.strftime("%x %X", end),
                                 'elapsed': time.strftime("%H:%M:%S", elapsed),
                                 'batch': totals['batchsize'],
                                 'iterations': totals['iterations'],
                                 'rate': '{0:.1f}'.format(total_rate)}

class QuartersSummary(object):
    """ Calculations for quarters summary for single session """

    def __init__(self, sessions, sessionid):
        self.sessions = sessions
        self.summary = self.compile_summary(sessionid)

    def compile_summary(self, sessionid):
        """ Compile quarterly summary for each session """
        summary = dict()
        sessionid = int(sessionid) if sessionid != 'Total' else sessionid
        process_sessions = self.sessions[int(sessionid)]
        process_sessions = self.get_totals() if sessionid == 'Total' else process_sessions
        for quarterdict in self.compile_quarter_stats(process_sessions):
            summary[quarterdict['quarter']] = quarterdict
        return summary

    def get_totals(self):
        """ Compile all sessions in to total """
        sorted_sessions = [item[1] for item in sorted(self.sessions.items())]
        totals = {'iterations': 0,
                  'batchsize': [],
                  'starttime': 0,
                  'endtime': 0,
                  'loss': dict()}

        for value in sorted_sessions:
            start = value['starttime']
            end = value['endtime']
            loss = value['loss']
            totals['iterations'] += value['iterations']
            totals['batchsize'].append(value['batchsize'])
            totals['starttime'] = start if totals['starttime'] == 0 else totals['starttime']
            totals['endtime'] = end if end > totals['endtime'] else totals['endtime']
            for loss_key, loss_val in loss.items():
                if not totals['loss'].get(loss_key, None):
                    totals['loss'][loss_key] = []
                totals['loss'][loss_key].extend(loss_val)
        return totals

    def compile_quarter_stats(self, session):
        """ Compile the stats for each quarter """
        start_split = 0
        listsplit = self.quarter_list(session['loss']['timestamp'])

        for split in listsplit:
            end_split = start_split + split[1]
            batchsize = session['batchsize']
            batchsize = batchsize[split[0] - 1] if isinstance(batchsize, list) else batchsize
            quarterdict = self.loss_summary(session['loss'],
                                            start_split,
                                            end_split,
                                            batchsize)
            quarterdict['quarter'] = split[0]
            quarterdict['iterations'] = split[1]
            start_split += split[1]
            yield quarterdict

    @staticmethod
    def quarter_list(listsample):
        """ Divide data list into quarters """
        totallen = len(listsample)
        splitlen = int(totallen / 4)
        remainder = totallen % 4
        retval = []
        for quarter in range(1, 5):
            rollover = 1 if remainder else 0
            retval.append((quarter, splitlen + rollover))
            remainder -= 1 if remainder != 0 else 0
        return retval

    @staticmethod
    def loss_summary(lossdict, start_split, end_split, batchsize):
        """ Compile summary stats from loss dict """
        losssummary = dict()
        for key, value in lossdict.items():
            data = value[start_split:end_split]
            if key == 'timestamp':
                starttime = min(data)
                endtime = max(data)
                elapsed = endtime - starttime
                rate = (batchsize * (end_split - start_split)) / elapsed
                losssummary['start'] = time.strftime("%x %X", time.gmtime(starttime))
                losssummary['end'] = time.strftime("%x %X", time.gmtime(endtime))
                losssummary['elapsed'] = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                losssummary['rate'] = '{0:.1f}'.format(rate)
            else:
                losssummary[key] = '{0:.5f}'.format(mean(data))
        return losssummary

class PopUpData(object):
    """ Compiled data for graph display """
    def __init__(self, data, tkvars, session_id):
        self.rawdata = data
        self.session_id = session_id
        self.tkvars = tkvars

        self.iterations = 0
        self.data = self.compile()

    def compile(self):
        """ Load the data based on vars settings """
        dataset = self.tkvars['display'].get().lower()
        raw = self.tkvars['raw'].get()
        avg = self.tkvars['rollingaverage'].get()
        avg_samples = self.tkvars['avgiterations'].get()
        trend = self.tkvars['trend'].get()

        self.iterations = sum(self.rawdata[idx]['iterations'] for idx in self.gen_session_ids())
        compiled_data = [{item[0]: item[1]
                          for item in self.compile_dataset(self.rawdata[idx], dataset)}
                         for idx in self.gen_session_ids()]

        selected_data = self.compiled_data_to_selected_dict(compiled_data)

        if avg:
            self.compile_averages(selected_data, avg_samples)

        if trend:
            self.compile_trend(selected_data)

        if not raw:
            for key in compiled_data[0].keys():
                del selected_data[key]

        return selected_data

    def gen_session_ids(self):
        """ Return the session ids that are to be processed """
        if self.session_id == 'Total':
            session_ids = sorted(int(key) for key in self.rawdata.keys())
        else:
            session_ids = [int(self.session_id)]
        for session_id in session_ids:
            yield session_id

    def compile_dataset(self, data, dataset):
        """ Compile the selected dataset """
        if dataset == 'loss':
            compiled = self.gen_loss(data['loss'])

        if dataset == 'rate':
            batchsize = data['batchsize']
            timestamp = data['loss']['timestamp']
            compiled = self.gen_rate(timestamp, batchsize)
        return compiled

    @staticmethod
    def gen_loss(loss):
        """ Compile loss into the selected data dict """
        for key, val in loss.items():
            if key == 'timestamp':
                continue
            yield (key, val)

    @staticmethod
    def gen_rate(timestamp, batchsize):
        """ Calculate rate per iteration """
        prev_time = timestamp[0]
        rate = list()
        for item in timestamp:
            current_time = item
            timediff = current_time - prev_time
            iter_rate = 0 if timediff == 0 else batchsize / timediff
            rate.append(iter_rate)
            prev_time = current_time
            yield ('rate', rate)

    @staticmethod
    def compiled_data_to_selected_dict(compiled_data):
        """ Add the compiled data to the selected dict """
        selected_data = dict()
        for item in compiled_data:
            for key, val in item.items():
                if not selected_data.get(key, None):
                    selected_data[key] = val[:]
                else:
                    selected_data[key].extend(val)
        return selected_data

    def compile_averages(self, data, samples):
        """ Calculate rolling averages around median point """
        presample = ceil(int(samples) / 2)
        postsample = int(samples) - presample
        avgs = dict()
        for key, val in data.items():
            newkey = 'avg_{}'.format(key)
            avgs[newkey] = self.compile_averages_one_data_set(val,
                                                              presample,
                                                              postsample)
        for key, val in avgs.items():
            data[key] = val

    def compile_trend(self, data):
        """ Compile trend data """
        trend = dict()
        x_range = range(self.iterations)
        for key, val in data.items():
            if key.startswith('avg'):
                continue
            fit = np.polyfit(x_range, val, 3)
            poly = np.poly1d(fit)
            newkey = 'trend_{}'.format(key)
            trend[newkey] = poly(x_range)
        for key, val in trend.items():
            data[key] = val

    @staticmethod
    def compile_averages_one_data_set(data, presample, postsample):
        """ Compile the rolling average for the passed in dataset """
        rollingavg = list()
        datapoints = len(data)
        for idx in range(0, datapoints):
            if idx < presample or idx >= datapoints - postsample:
                rollingavg.append(None)
                continue
            else:
                avg = mean(data[idx - presample:idx + postsample])
                rollingavg.append(avg)
        return rollingavg

    def refresh(self):
        """ Refresh the data """
        self.data = self.compile()
