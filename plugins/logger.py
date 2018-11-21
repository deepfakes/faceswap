#!/usr/bin/python
""" Stripped down version of logger for queuing to main process """
import logging
from logging.handlers import QueueHandler


class MultiProcessingLogger(logging.Logger):
    """ Create custom logger  with custom levels """
    def __init__(self, name):
        for new_level in (('VERBOSE', 15), ('TRACE', 5)):
            level_name, level_num = new_level
            if hasattr(logging, level_name):
                continue
            logging.addLevelName(level_num, level_name)
            setattr(logging, level_name, level_num)
        super().__init__(name)

    def verbose(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'VERBOSE'.
        """
        if self.isEnabledFor(15):
            self._log(15, msg, args, **kwargs)

    def trace(self, msg, *args, **kwargs):
        """
        Log 'msg % args' with severity 'VERBOSE'.
        """
        if self.isEnabledFor(5):
            self._log(5, msg, args, **kwargs)


#logging.setLoggerClass(MultiProcessingLogger)


#def log_setup():
#    """ initial log set up. """
#    rootlogger = logging.getLogger()
#    rootlogger.setLevel(logging.INFO)

#def set_loglevel(log_queue, loglevel):
#    ''' Check valid log level supplied and set log level '''
#    logger = logging.getLogger("test")
#    logger.propagate = False
#    q_handler = QueueHandler(log_queue)
#    logger.addHandler(q_handler)#

#    numeric_level = getattr(logging, loglevel.upper(), None)
#    logger.setLevel(numeric_level)

#log_setup()