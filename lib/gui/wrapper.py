#!/usr/bin python3
""" Process wrapper for underlying faceswap commands for the GUI """
import os
import re
import signal
import subprocess
from subprocess import PIPE, Popen, TimeoutExpired
import sys
from threading import Thread
from time import time

from .console import ConsoleOut
from .statusbar import StatusBar
from .tooltip import Tooltip
from .utils import Images, Singleton

class ProcessWrapper(object, metaclass=Singleton):
    """ Builds command, launches and terminates the underlying
        faceswap process. Updates GUI display depending on state """

    def __init__(self, pathscript=None, calling_file='faceswap.py'):
        self.runningtask = False
        self.pathexecscript = os.path.join(pathscript, calling_file)
        self.task = FaceswapControl(self)
        self.actionbtns = dict()
        self.lossdict = dict()
        self.command = None

    def action_command(self, command, opts):
        """ The action to perform when the action button is pressed """
        if self.runningtask:
            self.task.terminate()
            self.command = None
        else:
            self.command = command
            self.prepare()
            args = self.build_args(opts)
            self.task.execute_script(command, args)

    def prepare(self):
        """ Prepare the environment for execution """
        self.runningtask = True
        ConsoleOut().clear()
        print('Loading...')
        self.change_action_button()
        StatusBar().status_message.set('Executing - ' + self.command + '.py')
        mode = 'indeterminate' if self.command == 'train' else 'determinate'
        StatusBar().progress_start(mode)

    def build_args(self, opts):
        """ Build the faceswap command and arguments list """
        args = ['python', '-u', self.pathexecscript, self.command]
        for item in opts[self.command]:
            optval = str(item.get('value', '').get())
            opt = item['opts'][0]
            if optval == 'False' or optval == '':
                continue
            elif optval == 'True':
                args.append(opt)
            else:
                args.extend((opt, optval))
            if self.command == 'train':
                args.append('-gui')  # Embed the preview pane
        return args

    def terminate(self, message):
        """ Finalise wrapper when process has exited """
        self.runningtask = False
        StatusBar().progress_stop()
        StatusBar().status_message.set(message)
        self.change_action_button()
        self.clear_display_panel()
        print('Process exited.')

    def change_action_button(self):
        """ Change the action button to relevant control """
        for cmd in self.actionbtns.keys():
            btnact = self.actionbtns[cmd]
            if self.runningtask:
                ttl = 'Terminate'
                hlp = 'Exit the running process'
            else:
                ttl = cmd.title()
                hlp = 'Run the {} script'.format(cmd.title())
            btnact.config(text=ttl)
            Tooltip(btnact, text=hlp, wraplength=200)

    def clear_display_panel(self):
        ''' Clear the preview window and graph '''
        Images().delete_preview()
        self.lossdict = dict()

class FaceswapControl(object):
    """ Control the underlying Faceswap tasks """

    def __init__(self, wrapper):

        self.wrapper = wrapper
        self.command = None
        self.args = None
        self.process = None
        self.lenloss = 0
        self.consoleregex = {'loss': re.compile(r'([a-zA-Z_]+):.*?(\d+\.\d+)'),
                             'tqdm': re.compile(r'(\d+%|\d+/\d+|\d+:\d+|\d+\.\d+[a-zA-Z/]+)')}

    def execute_script(self, command, args):
        """ Execute the requested Faceswap Script """
        self.command = command
        kwargs = {'stdout': PIPE,
                  'stderr': PIPE,
                  'bufsize': 1,
                  'universal_newlines': True}
        if os.name == 'nt':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = Popen(args, **kwargs)
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
        message = self.set_final_status(returncode)
        self.wrapper.terminate(message)

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
        StatusBar().progress_update(message, 0, False)
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
                             for lossvals in self.wrapper.lossdict.values()
                            ) if self.wrapper.lossdict else 0
        if self.lenloss > currentlenloss:
            self.wrapper.lossdict = dict()
            self.lenloss = 0
            return False

        self.lenloss = currentlenloss

        if not self.wrapper.lossdict:
            self.wrapper.lossdict.update((item[0], []) for item in loss)

        message = ''
        for item in loss:
            self.wrapper.lossdict[item[0]].append(float(item[1]))
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

        StatusBar().progress_update(message, position, True)
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
        return status
