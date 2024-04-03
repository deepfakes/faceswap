#!/usr/bin python3
""" Process wrapper for underlying faceswap commands for the GUI """
from __future__ import annotations
import os
import logging
import re
import signal
import sys
import typing as T

from subprocess import PIPE, Popen
from threading import Thread
from time import time

import psutil

from .analysis import Session
from .utils import get_config, get_images, LongRunningTask, preview_trigger

if os.name == "nt":
    import win32console  # pylint:disable=import-error

logger = logging.getLogger(__name__)


class ProcessWrapper():
    """ Builds command, launches and terminates the underlying
        faceswap process. Updates GUI display depending on state """

    def __init__(self) -> None:
        logger.debug("Initializing %s", self.__class__.__name__)
        self._tk_vars = get_config().tk_vars
        self._set_callbacks()
        self._command: str | None = None
        """ str | None: The currently executing command, when process running or ``None`` """

        self._statusbar = get_config().statusbar
        self._training_session_location: dict[T.Literal["model_name", "model_folder"], str] = {}
        self._task = FaceswapControl(self)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def task(self) -> FaceswapControl:
        """ :class:`FaceswapControl`: The object that controls the underlying faceswap process """
        return self._task

    def _set_callbacks(self) -> None:
        """ Set the tkinter variable callbacks for performing an action or generating a command """
        logger.debug("Setting tk variable traces")
        self._tk_vars.action_command.trace("w", self._action_command)
        self._tk_vars.generate_command.trace("w", self._generate_command)

    def _action_command(self, *args: tuple[str, str, str]):  # pylint:disable=unused-argument
        """ Callback for when the Action button is pressed. Process command line options and
        launches the action

        Parameters
        ----------
        args:
            tuple[str, str, str]
                Tkinter variable callback args. Required but unused
        """
        if not self._tk_vars.action_command.get():
            return
        category, command = self._tk_vars.action_command.get().split(",")

        if self._tk_vars.running_task.get():
            self._task.terminate()
        else:
            self._command = command
            fs_args = self._prepare(T.cast(T.Literal["faceswap", "tools"], category))
            self._task.execute_script(command, fs_args)
        self._tk_vars.action_command.set("")

    def _generate_command(self,  # pylint:disable=unused-argument
                          *args: tuple[str, str, str]) -> None:
        """ Callback for when the Generate button is pressed. Process command line options and
        output the cli command

        Parameters
        ----------
        args:
            tuple[str, str, str]
                Tkinter variable callback args. Required but unused
        """
        if not self._tk_vars.generate_command.get():
            return
        category, command = self._tk_vars.generate_command.get().split(",")
        fs_args = self._build_args(category, command=command, generate=True)
        self._tk_vars.console_clear.set(True)
        logger.debug(" ".join(fs_args))
        print(" ".join(fs_args))
        self._tk_vars.generate_command.set("")

    def _prepare(self, category: T.Literal["faceswap", "tools"]) -> list[str]:
        """ Prepare the environment for execution, Sets the 'running task' and 'console clear'
        global tkinter variables. If training, sets the 'is training' variable

        Parameters
        ----------
        category: str, ["faceswap", "tools"]
            The script that is executing the command

        Returns
        -------
        list[str]
            The command line arguments to execute for the faceswap job
        """
        logger.debug("Preparing for execution")
        assert self._command is not None
        self._tk_vars.running_task.set(True)
        self._tk_vars.console_clear.set(True)
        if self._command == "train":
            self._tk_vars.is_training.set(True)
        print("Loading...")

        self._statusbar.message.set(f"Executing - {self._command}.py")
        mode: T.Literal["indeterminate",
                        "determinate"] = ("indeterminate" if self._command in ("effmpeg", "train")
                                          else "determinate")
        self._statusbar.start(mode)

        args = self._build_args(category)
        self._tk_vars.display.set(self._command)
        logger.debug("Prepared for execution")
        return args

    def _build_args(self,
                    category: str,
                    command: str | None = None,
                    generate: bool = False) -> list[str]:
        """ Build the faceswap command and arguments list.

        If training, pass the model folder and name to the training
        :class:`lib.gui.analysis.Session` for the GUI.

        Parameters
        ----------
        category: str, ["faceswap", "tools"]
            The script that is executing the command
        command: str, optional
            The main faceswap command to execute, if provided. The currently running task if
            ``None``. Default: ``None``
        generate: bool, optional
            ``True`` if the command is just to be generated for display. ``False`` if the command
            is to be executed

        Returns
        -------
        list[str]
            The full faceswap command to be executed or displayed
        """
        logger.debug("Build cli arguments: (category: %s, command: %s, generate: %s)",
                     category, command, generate)
        command = self._command if not command else command
        assert command is not None
        script = f"{category}.py"
        pathexecscript = os.path.join(os.path.realpath(os.path.dirname(sys.argv[0])), script)

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
            args = [f'"{arg}"' if " " in arg and not arg.startswith(("[", "("))
                    and not arg.endswith(("]", ")")) else arg
                    for arg in args]
        logger.debug("Built cli arguments: (%s)", args)
        return args

    def _get_training_session_info(self, cli_option: list[str]) -> None:
        """ Set the model folder and model name to :`attr:_training_session_location` so the global
        session picks them up for logging to the graph and analysis tab.

        Parameters
        ----------
        cli_option: list[str]
            The command line option to be checked for model folder or name
        """
        if cli_option[0] == "-t":
            self._training_session_location["model_name"] = cli_option[1].lower().replace("-", "_")
            logger.debug("model_name: '%s'", self._training_session_location["model_name"])
        if cli_option[0] == "-m":
            self._training_session_location["model_folder"] = cli_option[1]
            logger.debug("model_folder: '%s'", self._training_session_location["model_folder"])

    def terminate(self, message: str) -> None:
        """ Finalize wrapper when process has exited. Stops the progress bar, sets the status
        message. If the terminating task is 'train', then triggers the training close down actions

        Parameters
        ----------
        message: str
            The message to display in the status bar
        """
        logger.debug("Terminating Faceswap processes")
        self._tk_vars.running_task.set(False)
        if self._task.command == "train":
            self._tk_vars.is_training.set(False)
            Session.stop_training()
        self._statusbar.stop()
        self._statusbar.message.set(message)
        self._tk_vars.display.set("")
        get_images().delete_preview()
        preview_trigger().clear(trigger_type=None)
        self._command = None
        logger.debug("Terminated Faceswap processes")
        print("Process exited.")


class FaceswapControl():
    """ Control the underlying Faceswap tasks.

    wrapper: :class:`ProcessWrapper`
        The object responsible for managing this faceswap task
    """
    def __init__(self, wrapper: ProcessWrapper) -> None:
        logger.debug("Initializing %s (wrapper: %s)", self.__class__.__name__, wrapper)
        self._wrapper = wrapper
        self._session_info = wrapper._training_session_location
        self._config = get_config()
        self._statusbar = self._config.statusbar
        self._command: str | None = None
        self._process: Popen | None = None
        self._thread: LongRunningTask | None = None
        self._train_stats: dict[T.Literal["iterations", "timestamp"],
                                int | float | None] = {"iterations": 0, "timestamp": None}
        self._consoleregex: dict[T.Literal["loss", "tqdm", "ffmpeg"], re.Pattern] = {
            "loss": re.compile(r"[\W]+(\d+)?[\W]+([a-zA-Z\s]*)[\W]+?(\d+\.\d+)"),
            "tqdm": re.compile(r"(?P<dsc>.*?)(?P<pct>\d+%).*?(?P<itm>\S+/\S+)\W\["
                               r"(?P<tme>[\d+:]+<.*),\W(?P<rte>.*)[a-zA-Z/]*\]"),
            "ffmpeg": re.compile(r"([a-zA-Z]+)=\s*(-?[\d|N/A]\S+)")}
        self._first_loss_seen = False
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def command(self) -> str | None:
        """ str | None: The currently executing command, when process running or ``None`` """
        return self._command

    def execute_script(self, command: str, args: list[str]) -> None:
        """ Execute the requested Faceswap Script

        Parameters
        ----------
        command: str
            The faceswap command that is to be run
        args: list[str]
            The full command line arguments to be executed
        """
        logger.debug("Executing Faceswap: (command: '%s', args: %s)", command, args)
        self._thread = None
        self._command = command

        proc = Popen(args,  # pylint:disable=consider-using-with
                     stdout=PIPE,
                     stderr=PIPE,
                     bufsize=1,
                     text=True,
                     stdin=PIPE,
                     errors="backslashreplace")
        self._process = proc
        self._thread_stdout()
        self._thread_stderr()
        logger.debug("Executed Faceswap")

    def _process_training_determinate_function(self, output: str) -> bool:
        """ Process an stdout/stderr message to check for determinate TQDM output when training

        Parameters
        ----------
        output: str
            The stdout/stderr string to test

        Returns
        -------
        bool
            ``True`` if a determinate TQDM line was parsed when training otherwise ``False``
        """
        if self._command == "train" and not self._first_loss_seen and self._capture_tqdm(output):
            self._statusbar.set_mode("determinate")
            return True
        return False

    def _process_progress_stdout(self, output: str) -> bool:
        """ Process stdout for any faceswap processes that update the status/progress bar(s)

        Parameters
        ----------
        output: str
            The output line read from stdout

        Returns
        -------
        bool
            ``True`` if all actions have been completed on the output line otherwise ``False``
        """
        if self._process_training_determinate_function(output):
            return True

        if self._command == "train" and self._capture_loss(output):
            return True

        if self._command == "effmpeg" and self._capture_ffmpeg(output):
            return True

        if self._command not in ("train", "effmpeg") and self._capture_tqdm(output):
            return True

        return False

    def _process_training_stdout(self, output: str) -> None:
        """ Process any triggers that are required to update the GUI when Faceswap is running a
        training session.

        Parameters
        ----------
        output: str
            The output line read from stdout
        """
        tk_vars = get_config().tk_vars
        if self._command != "train" or not tk_vars.is_training.get():
            return

        t_output = output.strip().lower()
        if "[saved model]" not in t_output or t_output.endswith("[saved model]"):
            # Not a saved model line or saving the model for a reason other than standard saving
            return

        logger.debug("Trigger GUI Training update")
        logger.trace("tk_vars: %s", {itm: var.get()  # type:ignore[attr-defined]
                                     for itm, var in tk_vars.__dict__.items()})
        if not Session.is_training:
            # Don't initialize session until after the first save as state file must exist first
            logger.debug("Initializing curret training session")
            Session.initialize_session(self._session_info["model_folder"],
                                       self._session_info["model_name"],
                                       is_training=True)
        tk_vars.refresh_graph.set(True)

    def _read_stdout(self) -> None:
        """ Read stdout from the subprocess. """
        logger.debug("Opening stdout reader")
        assert self._process is not None
        while True:
            try:
                buff = self._process.stdout
                assert buff is not None
                output: str = buff.readline()
            except ValueError as err:
                if str(err).lower().startswith("i/o operation on closed file"):
                    break
                raise

            if output == "" and self._process.poll() is not None:
                break

            if output and self._process_progress_stdout(output):
                continue

            if output:
                self._process_training_stdout(output)
                print(output.rstrip())

        returncode = self._process.poll()
        assert returncode is not None
        self._first_loss_seen = False
        message = self._set_final_status(returncode)
        self._wrapper.terminate(message)
        logger.debug("Terminated stdout reader. returncode: %s", returncode)

    def _read_stderr(self) -> None:
        """ Read stdout from the subprocess. If training, pass the loss
        values to Queue """
        logger.debug("Opening stderr reader")
        assert self._process is not None
        while True:
            try:
                buff = self._process.stderr
                assert buff is not None
                output: str = buff.readline()
            except ValueError as err:
                if str(err).lower().startswith("i/o operation on closed file"):
                    break
                raise
            if output == "" and self._process.poll() is not None:
                break
            if output:
                if self._command != "train" and self._capture_tqdm(output):
                    continue
                if self._process_training_determinate_function(output):
                    continue
                if os.name == "nt" and "Call to CreateProcess failed. Error code: 2" in output:
                    # Suppress ptxas errors on Tensorflow for Windows
                    logger.debug("Suppressed call to subprocess error: '%s'", output)
                    continue
                print(output.strip(), file=sys.stderr)
        logger.debug("Terminated stderr reader")

    def _thread_stdout(self) -> None:
        """ Put the subprocess stdout so that it can be read without blocking """
        logger.debug("Threading stdout")
        thread = Thread(target=self._read_stdout)
        thread.daemon = True
        thread.start()
        logger.debug("Threaded stdout")

    def _thread_stderr(self) -> None:
        """ Put the subprocess stderr so that it can be read without blocking """
        logger.debug("Threading stderr")
        thread = Thread(target=self._read_stderr)
        thread.daemon = True
        thread.start()
        logger.debug("Threaded stderr")

    def _capture_loss(self, string: str) -> bool:
        """ Capture loss values from stdout

        Parameters
        ----------
        string: str
            An output line read from stdout

        Returns
        -------
        bool
            ``True`` if a loss line was captured from stdout, otherwise ``False``
        """
        logger.trace("Capturing loss")  # type:ignore[attr-defined]
        if not str.startswith(string, "["):
            logger.trace("Not loss message. Returning False")  # type:ignore[attr-defined]
            return False

        loss = self._consoleregex["loss"].findall(string)
        if len(loss) != 2 or not all(len(itm) == 3 for itm in loss):
            logger.trace("Not loss message. Returning False")  # type:ignore[attr-defined]
            return False

        message = f"Total Iterations: {int(loss[0][0])} | "
        message += "  ".join([f"{itm[1]}: {itm[2]}" for itm in loss])
        if not message:
            logger.trace(  # type:ignore[attr-defined]
                "Error creating loss message. Returning False")
            return False

        iterations = self._train_stats["iterations"]
        assert isinstance(iterations, int)

        if iterations == 0:
            # Set initial timestamp
            self._train_stats["timestamp"] = time()

        iterations += 1
        self._train_stats["iterations"] = iterations

        elapsed = self._calculate_elapsed()
        message = (f"Elapsed: {elapsed} | "
                   f"Session Iterations: {self._train_stats['iterations']}  {message}")

        if not self._first_loss_seen:
            self._statusbar.set_mode("indeterminate")
            self._first_loss_seen = True

        self._statusbar.progress_update(message, 0, False)
        logger.trace("Succesfully captured loss: %s", message)  # type:ignore[attr-defined]
        return True

    def _calculate_elapsed(self) -> str:
        """ Calculate and format time since training started

        Returns
        -------
        str
            The amount of time elapsed since training started in HH:mm:ss format
        """
        now = time()
        timestamp = self._train_stats["timestamp"]
        assert isinstance(timestamp, float)
        elapsed_time = now - timestamp
        try:
            i_hrs = int(elapsed_time // 3600)
            hrs = f"{i_hrs:02d}" if i_hrs < 10 else str(i_hrs)
            mins = f"{(int(elapsed_time % 3600) // 60):02d}"
            secs = f"{(int(elapsed_time % 3600) % 60):02d}"
        except ZeroDivisionError:
            hrs = mins = secs = "00"
        return f"{hrs}:{mins}:{secs}"

    def _capture_tqdm(self, string: str) -> bool:
        """ Capture tqdm output for progress bar

        Parameters
        ----------
        string: str
            An output line read from stdout

        Returns
        -------
        bool
            ``True`` if a tqdm line was captured from stdout, otherwise ``False``
        """
        logger.trace("Capturing tqdm")  # type:ignore[attr-defined]
        mtqdm = self._consoleregex["tqdm"].match(string)
        if not mtqdm:
            return False
        tqdm = mtqdm.groupdict()
        if any("?" in val for val in tqdm.values()):
            logger.trace("tqdm initializing. Skipping")  # type:ignore[attr-defined]
            return True
        description = tqdm["dsc"].strip()
        description = description if description == "" else f"{description[:-1]}  |  "
        processtime = (f"Elapsed: {tqdm['tme'].split('<')[0]}  "
                       f"Remaining: {tqdm['tme'].split('<')[1]}")
        msg = f"{description}{processtime}  |  {tqdm['rte']}  |  {tqdm['itm']}  |  {tqdm['pct']}"

        position = tqdm["pct"].replace("%", "")
        position = int(position) if position.isdigit() else 0

        self._statusbar.progress_update(msg, position, True)
        logger.trace("Succesfully captured tqdm message: %s", msg)  # type:ignore[attr-defined]
        return True

    def _capture_ffmpeg(self, string: str) -> bool:
        """ Capture ffmpeg output for progress bar

        Parameters
        ----------
        string: str
            An output line read from stdout

        Returns
        -------
        bool
            ``True`` if an ffmpeg line was captured from stdout, otherwise ``False``
        """
        logger.trace("Capturing ffmpeg")  # type:ignore[attr-defined]
        ffmpeg = self._consoleregex["ffmpeg"].findall(string)
        if len(ffmpeg) < 7:
            logger.trace("Not ffmpeg message. Returning False")  # type:ignore[attr-defined]
            return False

        message = ""
        for item in ffmpeg:
            message += f"{item[0]}: {item[1]}  "
        if not message:
            logger.trace(  # type:ignore[attr-defined]
                "Error creating ffmpeg message. Returning False")
            return False

        self._statusbar.progress_update(message, 0, False)
        logger.trace("Succesfully captured ffmpeg message: %s",  # type:ignore[attr-defined]
                     message)
        return True

    def terminate(self) -> None:
        """ Terminate the running process in a LongRunningTask so console can still be updated
        console """
        if self._thread is None:
            logger.debug("Terminating wrapper in LongRunningTask")
            self._thread = LongRunningTask(target=self._terminate_in_thread,
                                           args=(self._command, self._process))
            if self._command == "train":
                get_config().tk_vars.is_training.set(False)
            self._thread.start()
            self._config.root.after(1000, self.terminate)
        elif not self._thread.complete.is_set():
            logger.debug("Not finished terminating")
            self._config.root.after(1000, self.terminate)
        else:
            logger.debug("Termination Complete. Cleaning up")
            _ = self._thread.get_result()  # Terminate the LongRunningTask object
            self._thread = None

    def _terminate_in_thread(self, command: str, process: Popen) -> bool:
        """ Terminate the subprocess

        Parameters
        ----------
        command: str
            The command that is running

        process: :class:`subprocess.Popen`
            The running process

        Returns
        -------
        bool
            ``True`` when this function exits
        """
        logger.debug("Terminating wrapper")
        if command == "train":
            timeout = self._config.user_config_dict.get("timeout", 120)
            logger.debug("Sending Exit Signal")
            print("Sending Exit Signal", flush=True)
            now = time()
            if os.name == "nt":
                logger.debug("Sending carriage return to process")
                con_in = win32console.GetStdHandle(  # pylint:disable=c-extension-no-member
                    win32console.STD_INPUT_HANDLE)  # pylint:disable=c-extension-no-member
                keypress = self._generate_windows_keypress("\n")
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
                    self._terminate_all_children()
        else:
            self._terminate_all_children()
        return True

    @classmethod
    def _generate_windows_keypress(cls, character: str) -> bytes:
        """ Generate a Windows keypress

        Parameters
        ----------
        character: str
            The caracter to generate the keypress for

        Returns
        -------
        bytes
            The generated Windows keypress
        """
        buf = win32console.PyINPUT_RECORDType(  # pylint:disable=c-extension-no-member
            win32console.KEY_EVENT)  # pylint:disable=c-extension-no-member
        buf.KeyDown = 1
        buf.RepeatCount = 1
        buf.Char = character
        return buf

    @classmethod
    def _terminate_all_children(cls) -> None:
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
                msg = f"Process {child} survived SIGKILL. Giving up"
                logger.debug(msg)
                print(msg)

    def _set_final_status(self, returncode: int) -> str:
        """ Set the status bar output based on subprocess return code and reset training stats

        Parameters
        ----------
        returncode: int
            The returncode from the terminated process

        Returns
        -------
        str
            The final statusbar text
        """
        logger.debug("Setting final status. returncode: %s", returncode)
        self._train_stats = {"iterations": 0, "timestamp": None}
        if returncode in (0, 3221225786):
            status = "Ready"
        elif returncode == -15:
            status = f"Terminated - {self._command}.py"
        elif returncode == -9:
            status = f"Killed - {self._command}.py"
        elif returncode == -6:
            status = f"Aborted - {self._command}.py"
        else:
            status = f"Failed - {self._command}.py. Return Code: {returncode}"
        logger.debug("Set final status: %s", status)
        return status
