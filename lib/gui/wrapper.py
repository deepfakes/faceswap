#!/usr/bin python3
""" Process wrapper for underlying faceswap commands for the GUI """
import os
import logging
import re
import signal
from subprocess import PIPE, Popen, TimeoutExpired
import sys
import tkinter as tk
from threading import Thread
from time import time

import psutil

from .utils import Images

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ProcessWrapper():
    """ Builds command, launches and terminates the underlying
        faceswap process. Updates GUI display depending on state """

    def __init__(self, statusbar, session=None, pathscript=None, cliopts=None):
        logger.debug("Initializing %s: (pathscript: '%s', cliopts: %s",
                     self.__class__.__name__, pathscript, cliopts)
        self.tk_vars = self.set_tk_vars()
        self.session = session
        self.pathscript = pathscript
        self.cliopts = cliopts
        self.command = None
        self.statusbar = statusbar
        self.task = FaceswapControl(self)
        logger.debug("Initialized %s", self.__class__.__name__)

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

        tk_vars = {"display": display,
                   "runningtask": runningtask,
                   "action": actioncommand,
                   "generate": generatecommand,
                   "consoleclear": consoleclear}
        logger.debug(tk_vars)
        return tk_vars

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
        logger.debug(" ".join(args))
        print(" ".join(args))
        self.tk_vars["generate"].set(None)

    def prepare(self, category):
        """ Prepare the environment for execution """
        logger.debug("Preparing for execution")
        self.tk_vars["runningtask"].set(True)
        self.tk_vars["consoleclear"].set(True)
        print("Loading...")

        self.statusbar.status_message.set("Executing - {}.py".format(self.command))
        mode = "indeterminate" if self.command in ("effmpeg", "train") else "determinate"
        self.statusbar.progress_start(mode)

        args = self.build_args(category)
        self.tk_vars["display"].set(self.command)
        logger.debug("Prepared for execution")
        return args

    def build_args(self, category, command=None, generate=False):
        """ Build the faceswap command and arguments list """
        logger.debug("Build cli arguments: (category: %s, command: %s, generate: %s)",
                     category, command, generate)
        command = self.command if not command else command
        script = "{}.{}".format(category, "py")
        pathexecscript = os.path.join(self.pathscript, script)

        args = [sys.executable] if generate else [sys.executable, "-u"]
        args.extend([pathexecscript, command])

        for cliopt in self.cliopts.gen_cli_arguments(command):
            args.extend(cliopt)
            if command == "train" and not generate:
                self.set_session_stats(cliopt)
        if not generate:
            args.append("-gui")  # Indicate to Faceswap that we are running the GUI
        logger.debug("Built cli arguments: (%s)", args)
        return args

    def set_session_stats(self, cliopt):
        """ Set the session stats for batch size and model folder """
        if cliopt[0] == "-bs":
            self.session.stats["batchsize"] = int(cliopt[1])
        if cliopt[0] == "-m":
            self.session.modeldir = cliopt[1]
        logger.debug("Set session stats: stats: (%s, modeldir: '%s')",
                     self.session.stats, self.session.modeldir)

    def terminate(self, message):
        """ Finalize wrapper when process has exited """
        logger.debug("Terminating Faceswap processes")
        self.tk_vars["runningtask"].set(False)
        self.statusbar.progress_stop()
        self.statusbar.status_message.set(message)
        self.tk_vars["display"].set(None)
        Images().delete_preview()
        if self.command == "train":
            self.session.save_session()
        self.session.__init__()
        self.command = None
        logger.debug("Terminated Faceswap processes")
        print("Process exited.")


class FaceswapControl():
    """ Control the underlying Faceswap tasks """
    def __init__(self, wrapper):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.wrapper = wrapper
        self.statusbar = wrapper.statusbar
        self.command = None
        self.args = None
        self.process = None
        self.consoleregex = {
            "loss": re.compile(r"([a-zA-Z_]+):.*?(\d+\.\d+)"),
            "tqdm": re.compile(r"(\d+%|\d+/\d+|\d+:\d+|[\\?]|[\d\.\d\\?]+[a-zA-Z/]+)")}
        logger.debug("Initialized %s", self.__class__.__name__)

    def execute_script(self, command, args):
        """ Execute the requested Faceswap Script """
        logger.debug("Executing Faceswap: (command: '%s', args: %s)", command, args)
        self.command = command
        kwargs = {"stdout": PIPE,
                  "stderr": PIPE,
                  "bufsize": 1,
                  "universal_newlines": True}

        self.process = Popen(args, **kwargs, stdin=PIPE)
        self.thread_stdout()
        self.thread_stderr()
        logger.debug("Executed Faceswap")

    def read_stdout(self):
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        logger.debug("Opening stdout reader")
        while True:
            try:
                output = self.process.stdout.readline()
            except ValueError as err:
                if str(err).lower().startswith("i/o operation on closed file"):
                    break
                raise
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
        logger.debug("Terminated stdout reader. returncode: %s", returncode)

    def read_stderr(self):
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        logger.debug("Opening stderr reader")
        while True:
            try:
                output = self.process.stderr.readline()
            except ValueError as err:
                if str(err).lower().startswith("i/o operation on closed file"):
                    break
                raise
            if output == "" and self.process.poll() is not None:
                break
            if output:
                if self.command != "train" and self.capture_tqdm(output):
                    continue
                print(output.strip(), file=sys.stderr)
        logger.debug("Terminated stderr reader")

    def thread_stdout(self):
        """ Put the subprocess stdout so that it can be read without
        blocking """
        logger.debug("Threading stdout")
        thread = Thread(target=self.read_stdout)
        thread.daemon = True
        thread.start()
        logger.debug("Threaded stdout")

    def thread_stderr(self):
        """ Put the subprocess stderr so that it can be read without
        blocking """
        logger.debug("Threading stderr")
        thread = Thread(target=self.read_stderr)
        thread.daemon = True
        thread.start()
        logger.debug("Threaded stderr")

    def capture_loss(self, string):
        """ Capture loss values from stdout """
        logger.trace("Capturing loss")
        if not str.startswith(string, "["):
            logger.trace("Not loss message. Returning False")
            return False

        loss = self.consoleregex["loss"].findall(string)
        if len(loss) < 2:
            logger.trace("Not loss message. Returning False")
            return False

        self.wrapper.session.add_loss(loss)

        message = ""
        for item in loss:
            message += "{}: {}  ".format(item[0], item[1])
        if not message:
            logger.trace("Error creating loss message. Returning False")
            return False

        elapsed = self.wrapper.session.timestats["elapsed"]
        iterations = self.wrapper.session.stats["iterations"]

        message = "Elapsed: {}  Iteration: {}  {}".format(elapsed, iterations, message)
        self.statusbar.progress_update(message, 0, False)
        logger.trace("Succesfully captured loss: %s", message)
        return True

    def capture_tqdm(self, string):
        """ Capture tqdm output for progress bar """
        logger.trace("Capturing tqdm")
        tqdm = self.consoleregex["tqdm"].findall(string)
        if len(tqdm) != 5:
            logger.trace("Not a tqdm message. Returning False")
            return False

        if "?" in tqdm:
            logger.trace("tqdm initializing. Skipping")
            return True

        percent = tqdm[0]
        processed = tqdm[1]
        processtime = "Elapsed: {}  Remaining: {}".format(tqdm[2], tqdm[3])
        rate = tqdm[4]
        message = "{}  |  {}  |  {}  |  {}".format(processtime, rate, processed, percent)

        current, total = processed.split("/")
        position = int((float(current) / float(total)) * 1000)

        self.statusbar.progress_update(message, position, True)
        logger.trace("Succesfully captured tqdm message: %s", message)
        return True

    def terminate(self):
        """ Terminate the subprocess """
        logger.debug("Terminating wrapper")
        if self.command == "train":
            logger.debug("Sending Exit Signal")
            print("Sending Exit Signal", flush=True)
            try:
                now = time()
                if os.name == "nt":
                    try:
                        logger.debug("Sending carriage return to process")
                        self.process.communicate(input="\n", timeout=60)
                    except TimeoutExpired:
                        raise ValueError("Timeout reached sending Exit Signal")
                else:
                    logger.debug("Sending SIGINT to process")
                    self.process.send_signal(signal.SIGINT)
                    while True:
                        timeelapsed = time() - now
                        if self.process.poll() is not None:
                            break
                        if timeelapsed > 60:
                            raise ValueError("Timeout reached sending Exit Signal")
                return
            except ValueError as err:
                logger.error("Error terminating process", exc_info=True)
                print(err)
        else:
            logger.debug("Terminating Process...")
            print("Terminating Process...")
            children = psutil.Process().children(recursive=True)
            for child in children:
                child.terminate()
            _, alive = psutil.wait_procs(children, timeout=10)
            if not alive:
                logger.debug("Terminated")
                print("Terminated")
                return

            logger.debug("Termination timed out. Killing Process...")
            print("Termination timed out. Killing Process...")
            for child in alive:
                child.kill()
            _, alive = psutil.wait_procs(alive, timeout=10)
            if not alive:
                logger.debug("Killed")
                print("Killed")
            else:
                for child in alive:
                    msg = "Process {} survived SIGKILL. Giving up".format(child)
                    logger.debug(msg)
                    print(msg)

    def set_final_status(self, returncode):
        """ Set the status bar output based on subprocess return code """
        logger.debug("Setting final status. returncode: %s", returncode)
        if returncode in (0, 3221225786):
            status = "Ready"
        elif returncode == -15:
            status = "Terminated - {}.py".format(self.command)
        elif returncode == -9:
            status = "Killed - {}.py".format(self.command)
        elif returncode == -6:
            status = "Aborted - {}.py".format(self.command)
        else:
            status = "Failed - {}.py. Return Code: {}".format(self.command, returncode)
        logger.debug("Set final status: %s", status)
        return status
