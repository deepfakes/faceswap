#!/usr/bin python3
""" Process wrapper for underlying faceswap commands for the GUI """
import os
import logging
import re
import signal
from subprocess import PIPE, Popen
import sys
from threading import Thread
from time import time

import psutil

from .stats import Session
from .utils import get_config, get_images, LongRunningTask, preview_trigger

if os.name == "nt":
    import win32console  # pylint: disable=import-error


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ProcessWrapper():
    """ Builds command, launches and terminates the underlying
        faceswap process. Updates GUI display depending on state """

    def __init__(self):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.tk_vars = get_config().tk_vars
        self.set_callbacks()
        self.pathscript = os.path.realpath(os.path.dirname(sys.argv[0]))
        self.command = None
        self.statusbar = get_config().statusbar
        self._training_session_location = dict()
        self.task = FaceswapControl(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_callbacks(self):
        """ Set the tkinter variable callbacks """
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
        if self.command == "train":
            self.tk_vars["istraining"].set(True)
        print("Loading...")

        self.statusbar.message.set("Executing - {}.py".format(self.command))
        mode = "indeterminate" if self.command in ("effmpeg", "train") else "determinate"
        self.statusbar.start(mode)

        args = self.build_args(category)
        self.tk_vars["display"].set(self.command)
        logger.debug("Prepared for execution")
        return args

    def build_args(self, category, command=None, generate=False):
        """ Build the faceswap command and arguments list.

        If training, pass the model folder and name to the training :class:`lib.gui.stats.Session`
        for the GUI.
        """
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
                self._get_training_session_info(cliopt)

        if not generate:
            args.append("-gui")  # Indicate to Faceswap that we are running the GUI
        if generate:
            # Delimit args with spaces
            args = ['"{}"'.format(arg) if " " in arg and not arg.startswith(("[", "("))
                    and not arg.endswith(("]", ")")) else arg
                    for arg in args]
        logger.debug("Built cli arguments: (%s)", args)
        return args

    def _get_training_session_info(self, cli_option):
        """ Set the model folder and model name to :`attr:_training_session_location` so the global
        session picks them up for logging to the graph and analysis tab.

        Parameters
        ----------
        cli_option: list
            The command line option to be checked for model folder or name
        """
        if cli_option[0] == "-t":
            self._training_session_location["model_name"] = cli_option[1].lower().replace("-", "_")
            logger.debug("model_name: '%s'", self._training_session_location["model_name"])
        if cli_option[0] == "-m":
            self._training_session_location["model_folder"] = cli_option[1]
            logger.debug("model_folder: '%s'", self._training_session_location["model_folder"])

    def terminate(self, message):
        """ Finalize wrapper when process has exited """
        logger.debug("Terminating Faceswap processes")
        self.tk_vars["runningtask"].set(False)
        if self.task.command == "train":
            self.tk_vars["istraining"].set(False)
        self.statusbar.stop()
        self.statusbar.message.set(message)
        self.tk_vars["display"].set(None)
        get_images().delete_preview()
        Session.stop_training()
        preview_trigger().clear()
        self.command = None
        logger.debug("Terminated Faceswap processes")
        print("Process exited.")


class FaceswapControl():
    """ Control the underlying Faceswap tasks """
    def __init__(self, wrapper):
        logger.debug("Initializing %s", self.__class__.__name__)
        self.wrapper = wrapper
        self._session_info = wrapper._training_session_location
        self.config = get_config()
        self.statusbar = self.config.statusbar
        self.command = None
        self.args = None
        self.process = None
        self.thread = None  # Thread for LongRunningTask termination
        self.train_stats = {"iterations": 0, "timestamp": None}
        self.consoleregex = {
            "loss": re.compile(r"[\W]+(\d+)?[\W]+([a-zA-Z\s]*)[\W]+?(\d+\.\d+)"),
            "tqdm": re.compile(r"(?P<dsc>.*?)(?P<pct>\d+%).*?(?P<itm>\S+/\S+)\W\["
                               r"(?P<tme>[\d+:]+<.*),\W(?P<rte>.*)[a-zA-Z/]*\]"),
            "ffmpeg": re.compile(r"([a-zA-Z]+)=\s*(-?[\d|N/A]\S+)")}
        logger.debug("Initialized %s", self.__class__.__name__)

    def execute_script(self, command, args):
        """ Execute the requested Faceswap Script """
        logger.debug("Executing Faceswap: (command: '%s', args: %s)", command, args)
        self.thread = None
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
                if self.command == "train" and self.wrapper.tk_vars["istraining"].get():
                    if "[saved models]" in output.strip().lower():
                        logger.debug("Trigger GUI Training update")
                        logger.trace("tk_vars: %s", {itm: var.get()
                                                     for itm, var in self.wrapper.tk_vars.items()})
                        if not Session.is_training:
                            # Don't initialize session until after the first save as state
                            # file must exist first
                            logger.debug("Initializing curret training session")
                            Session.initialize_session(
                                self._session_info["model_folder"],
                                self._session_info["model_name"],
                                is_training=True)
                        self.wrapper.tk_vars["updatepreview"].set(True)
                        self.wrapper.tk_vars["refreshgraph"].set(True)
                    if "[preview updated]" in output.strip().lower():
                        self.wrapper.tk_vars["updatepreview"].set(True)
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
                if self.command == "train" and output.startswith("Reading training images"):
                    print(output.strip(), file=sys.stdout)
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
        if len(loss) != 2 or not all(len(itm) == 3 for itm in loss):
            logger.trace("Not loss message. Returning False")
            return False

        message = "Total Iterations: {} | ".format(int(loss[0][0]))
        message += "  ".join(["{}: {}".format(itm[1], itm[2]) for itm in loss])
        if not message:
            logger.trace("Error creating loss message. Returning False")
            return False

        iterations = self.train_stats["iterations"]

        if iterations == 0:
            # Set initial timestamp
            self.train_stats["timestamp"] = time()

        iterations += 1
        self.train_stats["iterations"] = iterations

        elapsed = self.calc_elapsed()
        message = "Elapsed: {} | Session Iterations: {}  {}".format(
            elapsed,
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
        description = tqdm["dsc"].strip()
        description = description if description == "" else "{}  |  ".format(description[:-1])
        processtime = "Elapsed: {}  Remaining: {}".format(tqdm["tme"].split("<")[0],
                                                          tqdm["tme"].split("<")[1])
        message = "{}{}  |  {}  |  {}  |  {}".format(description,
                                                     processtime,
                                                     tqdm["rte"],
                                                     tqdm["itm"],
                                                     tqdm["pct"])

        position = tqdm["pct"].replace("%", "")
        position = int(position) if position.isdigit() else 0

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
        """ Terminate the running process in a LongRunningTask so we can still
            output to console """
        if self.thread is None:
            logger.debug("Terminating wrapper in LongRunningTask")
            self.thread = LongRunningTask(target=self.terminate_in_thread,
                                          args=(self.command, self.process))
            if self.command == "train":
                self.wrapper.tk_vars["istraining"].set(False)
            self.thread.start()
            self.config.root.after(1000, self.terminate)
        elif not self.thread.complete.is_set():
            logger.debug("Not finished terminating")
            self.config.root.after(1000, self.terminate)
        else:
            logger.debug("Termination Complete. Cleaning up")
            _ = self.thread.get_result()  # Terminate the LongRunningTask object
            self.thread = None

    def terminate_in_thread(self, command, process):
        """ Terminate the subprocess """
        logger.debug("Terminating wrapper")
        if command == "train":
            timeout = self.config.user_config_dict.get("timeout", 120)
            logger.debug("Sending Exit Signal")
            print("Sending Exit Signal", flush=True)
            now = time()
            if os.name == "nt":
                logger.debug("Sending carriage return to process")
                con_in = win32console.GetStdHandle(  # pylint:disable=c-extension-no-member
                    win32console.STD_INPUT_HANDLE)  # pylint:disable=c-extension-no-member
                keypress = self.generate_windows_keypress("\n")
                con_in.WriteConsoleInput([keypress])
            else:
                logger.debug("Sending SIGINT to process")
                process.send_signal(signal.SIGINT)
            while True:
                timeelapsed = time() - now
                if process.poll() is not None:
                    break
                if timeelapsed > timeout:
                    logger.error("Timeout reached sending Exit Signal")
                    self.terminate_all_children()
        else:
            self.terminate_all_children()
        return True

    @staticmethod
    def generate_windows_keypress(character):
        """ Generate an 'Enter' key press to terminate Windows training """
        buf = win32console.PyINPUT_RECORDType(  # pylint:disable=c-extension-no-member
            win32console.KEY_EVENT)  # pylint:disable=c-extension-no-member
        buf.KeyDown = 1
        buf.RepeatCount = 1
        buf.Char = character
        return buf

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
