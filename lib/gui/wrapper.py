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

    def __init__(self, session=None, pathscript=None, calling_file='faceswap.py'):
        self.session = session
        self.runningtask = False
        self.pathexecscript = os.path.join(pathscript, calling_file)
        self.task = FaceswapControl(self)
        self.displaybook = None
        self.actionbtns = dict()
        self.command = None

    def action_command(self, command, opts):
        """ The action to perform when the action button is pressed """
        if self.runningtask:
            self.task.terminate()
            self.command = None
        else:
            self.command = command
            args = self.prepare(opts)
            self.task.execute_script(command, args)

    def generate_command(self, command, opts):
        """ Generate the command line arguments and output """
        args = self.build_args(opts, command=command, generate=True)
        ConsoleOut().clear()
        print(' '.join(args))

    def prepare(self, opts):
        """ Prepare the environment for execution """
        self.runningtask = True

        ConsoleOut().clear()
        print('Loading...')

        self.change_action_button()

        StatusBar().status_message.set('Executing - ' + self.command + '.py')
        mode = 'indeterminate' if self.command == 'train' else 'determinate'
        StatusBar().progress_start(mode)

        self.set_display_panel(opts)

        return self.build_args(opts)

    def build_args(self, opts, command=None, generate=False):
        """ Build the faceswap command and arguments list """
        command = self.command if not command else command

        args = ['python'] if generate else ['python', '-u']
        args.extend([self.pathexecscript, command])

        for item in opts[command]:
            optval = str(item.get('value', '').get())
            opt = item['opts'][0]
            if optval == 'False' or optval == '':
                continue
            elif optval == 'True':
                args.append(opt)
            else:
                args.extend((opt, optval))
            if command == 'train' and not generate:
                self.session.batchsize = int(optval) if opt == '-bs' else self.session.batchsize
                self.session.modeldir = optval if opt == '-m' else self.session.modeldir
        if command == 'train' and not generate:
            args.append('-gui')  # Embed the preview pane
        return args

    def terminate(self, message):
        """ Finalise wrapper when process has exited """
        self.runningtask = False
        StatusBar().progress_stop()
        StatusBar().status_message.set(message)
        self.change_action_button()
        self.clear_display_panel()
        if self.command == 'train':
            self.session.save_session()
        self.session.__init__()
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

    def set_display_panel(self, opts):
        """ Set the display tabs based on executing task """
        self.displaybook.remove_tabs()
        if self.command not in ('extract', 'train', 'convert'):
            return
        if self.command in ('extract', 'convert'):
            Images().pathoutput = next(item['value'].get()
                                       for item in opts[self.command] if item['opts'][0] == '-o')
        self.displaybook.command_display(self.command)

    def clear_display_panel(self):
        """ Clear the preview window and graph """
        self.displaybook.remove_tabs()
        Images().delete_preview()

class FaceswapControl(object):
    """ Control the underlying Faceswap tasks """

    def __init__(self, wrapper):

        self.wrapper = wrapper
        self.command = None
        self.args = None
        self.process = None
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

        self.wrapper.session.add_loss(loss)

        message = ''
        for item in loss:
            message += '{}: {}  '.format(item[0], item[1])
        if not message:
            return False

        elapsed = self.wrapper.session.timestats['elapsed']
        iterations = self.wrapper.session.iterations

        message = 'Elapsed: {}  Iteration: {}  {}'.format(elapsed, iterations, message)
        StatusBar().progress_update(message, 0, False)
        return True

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
