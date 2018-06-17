#!/usr/bin python3
""" Process wrapper for underlying faceswap commands for the GUI """
import os
import re
import signal
import subprocess
from subprocess import PIPE, Popen, TimeoutExpired
import sys
import tkinter as tk
from threading import Thread
from time import time

from .utils import Images


class ProcessWrapper(object):
    """ Builds command, launches and terminates the underlying
        faceswap process. Updates GUI display depending on state """

    def __init__(self, statusbar, session=None, pathscript=None, cliopts=None):
        self.tk_vars = self.set_tk_vars()
        self.session = session
        self.pathscript = pathscript
        self.cliopts = cliopts
        self.command = None
        self.statusbar = statusbar
        self.task = FaceswapControl(self)

    def set_tk_vars(self):
        """ TK Variables to be triggered by ProcessWrapper to indicate
            what state various parts of the GUI should be in """
        display = tk.StringVar()
        display.set(None)

        runningtask = tk.BooleanVar()
        runningtask.set(False)

        actioncommand = tk.StringVar()
        actioncommand.set(None)
        actioncommand.trace("w", self.action_command)

        generatecommand = tk.StringVar()
        generatecommand.set(None)
        generatecommand.trace("w", self.generate_command)

        consoleclear = tk.BooleanVar()
        consoleclear.set(False)

        return {"display": display,
                "runningtask": runningtask,
                "action": actioncommand,
                "generate": generatecommand,
                "consoleclear": consoleclear}

    def action_command(self, *args):
        """ The action to perform when the action button is pressed """
        if not self.tk_vars["action"].get():
            return
        category, command = self.tk_vars["action"].get().split(",")

        if self.tk_vars["runningtask"].get():
            self.task.terminate()
        else:
            self.command = command
            args = self.prepare(category)
            self.task.execute_script(command, args)
        self.tk_vars["action"].set(None)

    def generate_command(self, *args):
        """ Generate the command line arguments and output """
        if not self.tk_vars["generate"].get():
            return
        category, command = self.tk_vars["generate"].get().split(",")
        args = self.build_args(category, command=command, generate=True)
        self.tk_vars["consoleclear"].set(True)
        print(" ".join(args))
        self.tk_vars["generate"].set(None)

    def prepare(self, category):
        """ Prepare the environment for execution """
        self.tk_vars["runningtask"].set(True)
        self.tk_vars["consoleclear"].set(True)
        print("Loading...")

        self.statusbar.status_message.set("Executing - "
                                          + self.command + ".py")
        mode = "indeterminate" if self.command == "train" else "determinate"
        self.statusbar.progress_start(mode)

        args = self.build_args(category)
        self.tk_vars["display"].set(self.command)

        return args

    def build_args(self, category, command=None, generate=False):
        """ Build the faceswap command and arguments list """
        command = self.command if not command else command
        script = "{}.{}".format(category, "py")
        pathexecscript = os.path.join(self.pathscript, script)

        args = ["python"] if generate else ["python", "-u"]
        args.extend([pathexecscript, command])

        for cliopt in self.cliopts.gen_cli_arguments(command):
            args.extend(cliopt)
            if command == "train" and not generate:
                self.set_session_stats(cliopt)
        if command == "train" and not generate:
            args.append("-gui")  # Embed the preview pane
        return args

    def set_session_stats(self, cliopt):
        """ Set the session stats for batchsize and modeldir """
        if cliopt[0] == "-bs":
            self.session.stats["batchsize"] = int(cliopt[1])
        if cliopt[0] == "-m":
            self.session.modeldir = cliopt[1]

    def terminate(self, message):
        """ Finalise wrapper when process has exited """
        self.tk_vars["runningtask"].set(False)
        self.statusbar.progress_stop()
        self.statusbar.status_message.set(message)
        self.tk_vars["display"].set(None)
        Images().delete_preview()
        if self.command == "train":
            self.session.save_session()
        self.session.__init__()
        self.command = None
        print("Process exited.")


class FaceswapControl(object):
    """ Control the underlying Faceswap tasks """
    __group_processes = ["effmpeg"]

    def __init__(self, wrapper):

        self.wrapper = wrapper
        self.statusbar = wrapper.statusbar
        self.command = None
        self.args = None
        self.process = None
        self.consoleregex = {"loss": re.compile(r"([a-zA-Z_]+):.*?(\d+\.\d+)"),
                             "tqdm": re.compile(r"(\d+%|\d+/\d+|\d+:\d+|\d+\.\d+[a-zA-Z/]+)")}

    def execute_script(self, command, args):
        """ Execute the requested Faceswap Script """
        self.command = command
        kwargs = {"stdout": PIPE,
                  "stderr": PIPE,
                  "bufsize": 1,
                  "universal_newlines": True}

        if self.command in self.__group_processes:
            kwargs["preexec_fn"] = os.setsid

        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = Popen(args, **kwargs)
        self.thread_stdout()
        self.thread_stderr()

    def read_stdout(self):
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        while True:
            output = self.process.stdout.readline()
            if output == "" and self.process.poll() is not None:
                break
            if output:
                if (self.command == "train" and self.capture_loss(output)) or (
                        self.command != "train" and self.capture_tqdm(output)):
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
            if output == "" and self.process.poll() is not None:
                break
            if output:
                if self.command != "train" and self.capture_tqdm(output):
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

        if not str.startswith(string, "["):
            return False

        loss = self.consoleregex["loss"].findall(string)
        if len(loss) < 2:
            return False

        self.wrapper.session.add_loss(loss)

        message = ""
        for item in loss:
            message += "{}: {}  ".format(item[0], item[1])
        if not message:
            return False

        elapsed = self.wrapper.session.timestats["elapsed"]
        iterations = self.wrapper.session.stats["iterations"]

        message = "Elapsed: {}  Iteration: {}  {}".format(elapsed,
                                                          iterations,
                                                          message)
        self.statusbar.progress_update(message, 0, False)
        return True

    def capture_tqdm(self, string):
        """ Capture tqdm output for progress bar """
        tqdm = self.consoleregex["tqdm"].findall(string)
        if len(tqdm) != 5:
            return False

        percent = tqdm[0]
        processed = tqdm[1]
        processtime = "Elapsed: {}  Remaining: {}".format(tqdm[2], tqdm[3])
        rate = tqdm[4]
        message = "{}  |  {}  |  {}  |  {}".format(processtime,
                                                   rate,
                                                   processed,
                                                   percent)

        current, total = processed.split("/")
        position = int((float(current) / float(total)) * 1000)

        self.statusbar.progress_update(message, position, True)
        return True

    def terminate(self):
        """ Terminate the subprocess """
        if self.command == "train":
            print("Sending Exit Signal", flush=True)
            try:
                now = time()
                if os.name == "nt":
                    os.kill(self.process.pid, signal.CTRL_BREAK_EVENT)
                else:
                    self.process.send_signal(signal.SIGINT)
                while True:
                    timeelapsed = time() - now
                    if self.process.poll() is not None:
                        break
                    if timeelapsed > 30:
                        raise ValueError("Timeout reached sending Exit Signal")
                return
            except ValueError as err:
                print(err)
        elif self.command in self.__group_processes:
            print("Terminating Process Group...")
            pgid = os.getpgid(self.process.pid)
            try:
                os.killpg(pgid, signal.SIGINT)
                self.process.wait(timeout=10)
                print("Terminated")
            except TimeoutExpired:
                print("Termination timed out. Killing Process Group...")
                os.killpg(pgid, signal.SIGKILL)
                print("Killed")
        else:
            print("Terminating Process...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                print("Terminated")
            except TimeoutExpired:
                print("Termination timed out. Killing Process...")
                self.process.kill()
                print("Killed")

    def set_final_status(self, returncode):
        """ Set the status bar output based on subprocess return code """
        if returncode == 0 or returncode == 3221225786:
            status = "Ready"
        elif returncode == -15:
            status = "Terminated - {}.py".format(self.command)
        elif returncode == -9:
            status = "Killed - {}.py".format(self.command)
        elif returncode == -6:
            status = "Aborted - {}.py".format(self.command)
        else:
            status = "Failed - {}.py. Return Code: {}".format(self.command,
                                                              returncode)
        return status
