#!/usr/bin python3
""" Process wrapper for underlying faceswap commands for the GUI """
import os
import logging
import re
import signal
from subprocess import PIPE, Popen, TimeoutExpired
import sys
from threading import Thread
from time import time

import psutil

from .utils import get_config, get_images

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ProcessWrapper():
    """ Builds command, launches and terminates the underlying
        faceswap process. Updates GUI display depending on state """

    def __init__(self, pathscript=None):
        logger.debug("Initializing %s: (pathscript: %s)", self.__class__.__name__, pathscript)
        self.tk_vars = get_config().tk_vars
        self.set_callbacks()
        self.pathscript = pathscript
        self.command = None
        self.statusbar = get_config().statusbar
        self.task = FaceswapControl(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_callbacks(self):
        """ Set the tk variable callbacks """
        logger.debug("Setting tk variable traces")
        self.tk_vars["action"].trace("w", self.action_command)
        self.tk_vars["generate"].trace("w", self.generate_command)

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

        cli_opts = get_config().cli_opts
        for cliopt in cli_opts.gen_cli_arguments(command):
            args.extend(cliopt)
            if command == "train" and not generate:
                self.init_training_session(cliopt)
        if not generate:
            args.append("-gui")  # Indicate to Faceswap that we are running the GUI
        if generate:
            # Delimit args with spaces
            args = ['"{}"'.format(arg) if " " in arg and not arg.startswith(("[", "("))
                    and not arg.endswith(("]", ")")) else arg
                    for arg in args]
        logger.debug("Built cli arguments: (%s)", args)
        return args

    @staticmethod
    def init_training_session(cliopt):
        """ Set the session stats for disable logging, model folder and model name """
        session = get_config().session
        if cliopt[0] == "-t":
            session.modelname = cliopt[1].lower().replace("-", "_")
            logger.debug("modelname: '%s'", session.modelname)
        if cliopt[0] == "-m":
            session.modeldir = cliopt[1]
            logger.debug("modeldir: '%s'", session.modeldir)

    def terminate(self, message):
        """ Finalize wrapper when process has exited """
        logger.debug("Terminating Faceswap processes")
        self.tk_vars["runningtask"].set(False)
        self.statusbar.progress_stop()
        self.statusbar.status_message.set(message)
        self.tk_vars["display"].set(None)
        get_images().delete_preview()
        get_config().session.__init__()
        self.command = None
        logger.debug("Terminated Faceswap processes")
        print("Process exited.")


class FaceswapControl():
    """ Control the underlying Faceswap tasks """
    def __init__(self, wrapper):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.wrapper = wrapper
        self.statusbar = get_config().statusbar
        self.command = None
        self.args = None
        self.process = None
        self.train_stats = {"iterations": 0, "timestamp": None}
        self.consoleregex = {
            "loss": re.compile(r"([a-zA-Z_]+):.*?(\d+\.\d+)"),
            "tqdm": re.compile(r".*?(?P<pct>\d+%).*?(?P<itm>\d+/\d+)\W\["
                               r"(?P<tme>\d+:\d+<.*),\W(?P<rte>.*)[a-zA-Z/]*\]"),
            "ffmpeg": re.compile(r"([a-zA-Z]+)=\s*(-?[\d|N/A]\S+)")}
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
                if ((self.command == "train" and self.capture_loss(output)) or
                        (self.command == "effmpeg" and self.capture_ffmpeg(output)) or
                        (self.command not in ("train", "effmpeg") and self.capture_tqdm(output))):
                    continue
                if self.command == "train" and output.strip().endswith("saved models"):
                    logger.debug("Trigger update preview")
                    self.wrapper.tk_vars["updatepreview"].set(True)
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

        message = ""
        for item in loss:
            message += "{}: {}  ".format(item[0], item[1])
        if not message:
            logger.trace("Error creating loss message. Returning False")
            return False

        iterations = self.train_stats["iterations"]

        if iterations == 0:
            # Initialize session stats and set initial timestamp
            self.train_stats["timestamp"] = time()

        if not get_config().session.initialized and iterations > 0:
            # Don't initialize session until after the first iteration as state
            # file must exist first
            get_config().session.initialize_session(is_training=True)
            self.wrapper.tk_vars["refreshgraph"].set(True)

        iterations += 1
        if iterations % 100 == 0:
            self.wrapper.tk_vars["refreshgraph"].set(True)
        self.train_stats["iterations"] = iterations

        elapsed = self.calc_elapsed()
        message = "Elapsed: {}  Iteration: {}  {}".format(elapsed,
                                                          self.train_stats["iterations"], message)
        self.statusbar.progress_update(message, 0, False)
        logger.trace("Succesfully captured loss: %s", message)
        return True

    def calc_elapsed(self):
        """ Calculate and format time since training started """
        now = time()
        elapsed_time = now - self.train_stats["timestamp"]
        try:
            hrs = int(elapsed_time // 3600)
            if hrs < 10:
                hrs = "{0:02d}".format(hrs)
            mins = "{0:02d}".format((int(elapsed_time % 3600) // 60))
            secs = "{0:02d}".format((int(elapsed_time % 3600) % 60))
        except ZeroDivisionError:
            hrs = "00"
            mins = "00"
            secs = "00"
        return "{}:{}:{}".format(hrs, mins, secs)

    def capture_tqdm(self, string):
        """ Capture tqdm output for progress bar """
        logger.trace("Capturing tqdm")
        tqdm = self.consoleregex["tqdm"].match(string)
        if not tqdm:
            return False
        tqdm = tqdm.groupdict()
        if any("?" in val for val in tqdm.values()):
            logger.trace("tqdm initializing. Skipping")
            return True
        processtime = "Elapsed: {}  Remaining: {}".format(tqdm["tme"].split("<")[0],
                                                          tqdm["tme"].split("<")[1])
        message = "{}  |  {}  |  {}  |  {}".format(processtime,
                                                   tqdm["rte"],
                                                   tqdm["itm"],
                                                   tqdm["pct"])

        current, total = tqdm["itm"].split("/")
        position = int((float(current) / float(total)) * 1000)

        self.statusbar.progress_update(message, position, True)
        logger.trace("Succesfully captured tqdm message: %s", message)
        return True

    def capture_ffmpeg(self, string):
        """ Capture tqdm output for progress bar """
        logger.trace("Capturing ffmpeg")
        ffmpeg = self.consoleregex["ffmpeg"].findall(string)
        if len(ffmpeg) < 7:
            logger.trace("Not ffmpeg message. Returning False")
            return False

        message = ""
        for item in ffmpeg:
            message += "{}: {}  ".format(item[0], item[1])
        if not message:
            logger.trace("Error creating ffmpeg message. Returning False")
            return False

        self.statusbar.progress_update(message, 0, False)
        logger.trace("Succesfully captured ffmpeg message: %s", message)
        return True

    def terminate(self):
        """ Terminate the subprocess """
        logger.debug("Terminating wrapper")
        if self.command == "train":
            timeout = get_config().tk_vars["traintimeout"].get()
            logger.debug("Sending Exit Signal")
            print("Sending Exit Signal", flush=True)
            if os.name == "nt":
                try:
                    logger.debug("Sending carriage return to process")
                    self.process.communicate(input="\n", timeout=timeout)
                except TimeoutExpired:
                    logger.error("Timeout reached sending Exit Signal")
                    self.terminate_all_children()
            else:
                logger.debug("Sending SIGINT to process")
                now = time()
                self.process.send_signal(signal.SIGINT)
                while True:
                    timeelapsed = time() - now
                    if self.process.poll() is not None:
                        break
                    if timeelapsed > timeout:
                        logger.error("Timeout reached sending Exit Signal")
                        self.terminate_all_children()
                return
        else:
            self.terminate_all_children()

    @staticmethod
    def terminate_all_children():
        """ Terminates all children """
        logger.debug("Terminating Process...")
        print("Terminating Process...", flush=True)
        children = psutil.Process().children(recursive=True)
        for child in children:
            child.terminate()
        _, alive = psutil.wait_procs(children, timeout=10)
        if not alive:
            logger.debug("Terminated")
            print("Terminated")
            return

        logger.debug("Termination timed out. Killing Process...")
        print("Termination timed out. Killing Process...", flush=True)
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
        """ Set the status bar output based on subprocess return code
            and reset training stats """
        logger.debug("Setting final status. returncode: %s", returncode)
        self.train_stats = {"iterations": 0, "timestamp": None}
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
