#!/usr/bin python3
""" Controls the underlying faceswap python scripts """

import os
import re
import signal
import subprocess
from subprocess import PIPE, Popen, TimeoutExpired
import sys
from threading import Thread
from time import time

class FaceswapControl(object):
    """ Control the underlying Faceswap tasks """

    def __init__(self, utils, pathscript, calling_file="faceswap.py"):
        self.pathexecscript = os.path.join(pathscript, calling_file)
        self.utils = utils

        self.command = None
        self.args = None
        self.process = None
        self.lenloss = 0
        self.consoleregex = {'loss': re.compile(r'([a-zA-Z_]+):.*?(\d+\.\d+)'),
                             'tqdm': re.compile(r'(\d+%|\d+/\d+|\d+:\d+|\d+\.\d+[a-zA-Z/]+)')}

    def prepare(self, options, command):
        """ Prepare for running the subprocess """
        self.command = command
        self.utils.runningtask = True
        self.utils.change_action_button()
        self.utils.guitext['status'].set('Executing - ' + self.command + '.py')
        mode = 'indeterminate' if command == 'train' else 'determinate'
        self.utils.set_progress_bar_type(mode)
        print('Loading...')
        self.args = ['python', '-u', self.pathexecscript, self.command]
        self.build_args(options)

    def build_args(self, options):
        """ Build the faceswap command and arguments list """
        for item in options[self.command]:
            optval = str(item.get('value', '').get())
            opt = item['opts'][0]
            if optval == 'False' or optval == '':
                continue
            elif optval == 'True':
                if self.command == 'train' and opt == '-p':  # Embed the preview pane
                    self.args.append('-gui')
                else:
                    self.args.append(opt)
            else:
                self.args.extend((opt, optval))

    def execute_script(self):
        """ Execute the requested Faceswap Script """
        kwargs = {'stdout': PIPE,
                  'stderr': PIPE,
                  'bufsize': 1,
                  'universal_newlines': True}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = Popen(self.args, **kwargs)
        self.thread_stdout()
        self.thread_stderr()

    def read_stdout(self):
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        while True:
            output = self.process.stdout.readline()
            if output == '' and self.process.poll() is not None:
                break
            if output:
                if self.command == 'train' and self.capture_loss(output):
                    continue
                print(output.strip())
        returncode = self.process.poll()
        self.utils.runningtask = False
        self.utils.change_action_button()
        self.utils.update_progress('', 0, True)
        self.utils.set_progress_bar_type('determinate')
        self.set_final_status(returncode)
        print('Process exited.')

    def read_stderr(self):
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        while True:
            output = self.process.stderr.readline()
            if output == '' and self.process.poll() is not None:
                break
            if output:
                if self.command != 'train' and self.capture_tqdm(output):
                    continue
                print(output.strip(), file=sys.stderr)

    def thread_stdout(self):
        """ Put the subprocess stdout so that it can be read without
        blocking """
        thread = Thread(target=self.read_stdout)
        thread.daemon = True
        thread.start()

    def thread_stderr(self):
        """ Put the subprocess stderr so that it can be read without
        blocking """
        thread = Thread(target=self.read_stderr)
        thread.daemon = True
        thread.start()

    def capture_loss(self, string):
        """ Capture loss values from stdout """

        if not str.startswith(string, '['):
            return False

        loss = self.consoleregex['loss'].findall(string)
        if len(loss) < 2:
            return False

        message = self.update_lossdict(loss)
        if not message:
            return False

        message = 'Iteration: {}  {}'.format(self.lenloss, message)
        self.utils.update_progress(message, 0, False)
        return True

    def update_lossdict(self, loss):
        """ Update the loss dictionary for graphing and stats """
        #TODO: Remove this hideous hacky fix. When the subprocess is terminated and
        # the loss dictionary is reset, 1 set of loss values ALWAYS slips through
        # and appends to the lossdict AFTER the subprocess has closed meaning that
        # checks on whether the dictionary is empty fail.
        # Therefore if the size of current loss dictionary is smaller than the
        # previous loss dictionary, assume that the process has been terminated
        # and reset it.
        # I have tried and failed to empty the subprocess stdout with:
        #   sys.exit() on the stdout/err threads (no effect)
        #   sys.stdout/stderr.flush (no effect)
        #   thread.join (locks the whole process up, because the stdout thread
        #       stubbonly refuses to release it's last line)

        currentlenloss = max(len(lossvals)
                             for lossvals in self.utils.lossdict.values()
                            ) if self.utils.lossdict else 0
        if self.lenloss > currentlenloss:
            self.utils.lossdict = dict()
            self.lenloss = 0
            return False

        self.lenloss = currentlenloss

        if not self.utils.lossdict:
            self.utils.lossdict.update((item[0], []) for item in loss)

        message = ''
        for item in loss:
            self.utils.lossdict[item[0]].append(float(item[1]))
            message += '{}: {}  '.format(item[0], item[1])

        return message

    def capture_tqdm(self, string):
        """ Capture tqdm output for progress bar """
        tqdm = self.consoleregex['tqdm'].findall(string)
        if len(tqdm) != 5:
            return False

        percent = tqdm[0]
        processed = tqdm[1]
        processtime = 'Elapsed: {}  Remaining: {}'.format(tqdm[2], tqdm[3])
        rate = tqdm[4]
        message = '{}  |  {}  |  {}  |  {}'.format(processtime, rate, processed, percent)

        current, total = processed.split('/')
        position = int((float(current) / float(total)) * 1000)

        self.utils.update_progress(message, position, True)
        return True

    def terminate(self):
        """ Terminate the subprocess """
        if self.command == 'train':
            print('Sending Exit Signal', flush=True)
            try:
                now = time()
                if os.name == 'nt':
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    self.process.send_signal(signal.SIGINT)
                while True:
                    timeelapsed = time() - now
                    if self.process.poll() is not None:
                        break
                    if timeelapsed > 30:
                        raise ValueError('Timeout reached sending Exit Signal')
                return
            except ValueError as err:
                print(err)
        print('Terminating Process...')
        try:
            self.process.terminate()
            self.process.wait(timeout=10)
            print('Terminated')
        except TimeoutExpired:
            print('Termination timed out. Killing Process...')
            self.process.kill()
            print('Killed')

    def set_final_status(self, returncode):
        """ Set the status bar output based on subprocess return code """
        if returncode == 0 or returncode == 3221225786:
            status = 'Ready'
        elif returncode == -15:
            status = 'Terminated - {}.py'.format(self.command)
        elif returncode == -9:
            status = 'Killed - {}.py'.format(self.command)
        elif returncode == -6:
            status = 'Aborted - {}.py'.format(self.command)
        else:
            status = 'Failed - {}.py. Return Code: {}'.format(self.command, returncode)
        self.utils.guitext['status'].set(status)
