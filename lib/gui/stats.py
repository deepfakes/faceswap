#!/usr/bin python3
""" Stats functions for the GUI """

import time
import os

from lib.Serializer import PickleSerializer

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
        sessions = len(summary)
        total_rate = totals['rate'] / sessions
        summary[sessions + 1] = {'session': 'Total',
                                 'start': time.strftime("%x %X", time.gmtime(totals['starttime'])),
                                 'end': time.strftime("%x %X", time.gmtime(totals['endtime'])),
                                 'elapsed': time.strftime("%H:%M:%S", time.gmtime(totals['elapsed'])),
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
        process_sessions = self.get_totals() if sessionid == 'Total' else self.sessions[int(sessionid)]
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
            quarterdict['quarter'] =split[0]
            quarterdict['iterations'] = split[1]
            start_split += split[1]
            yield quarterdict

    @staticmethod
    def quarter_list(listsample):
        totallen = len(listsample)
        splitlen = int(totallen / 4)
        remainder = totallen % 4
        retval = []
        for quarter in range(1, 5):
            rollover = 1 if remainder else 0
            retval.append((quarter, splitlen + rollover))
            remainder -= 1 if remainder != 0 else 0
        return retval

    def loss_summary(self, lossdict, start_split, end_split, batchsize):
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
                losssummary[key] = '{0:.5f}'.format(self.mean(data))
        return losssummary

    @staticmethod
    def mean(numbers):
        """ Calculate the mean of a given list of numbers """
        return sum(numbers) / max(len(numbers), 1)

