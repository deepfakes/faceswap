#! /usr/env/bin/python3
"""GPU profiling for throughput optimization"""
from __future__ import annotations

import logging
import math
import typing as T
from dataclasses import dataclass, InitVar, field
from operator import itemgetter
from threading import Event, Lock
from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from lib.logger import parse_class_init
from lib.multithreading import FSThread
from lib.utils import get_module_objects
from plugins.extract import extract_config as cfg

from .runner import get_pipeline
from .plugin_utils import get_torch_modules, random_input_from_plugin, warmup_plugin

if T.TYPE_CHECKING:
    import numpy.typing as npt
    from lib.multithreading import ErrorState
    from plugins.extract.base import ExtractPlugin
    from .handler import ExtractHandler
    from .runner import ExtractRunner

logger = logging.getLogger(__name__)


# TODO roll back to max and refine


class ModelProfile():
    """Benchmark a single PyTorch GPU plugin for inference

    Parameters
    ----------
    plugin
        The plugin to benchmark for inference
    max_batch_size
        The maximum batch size to benchmark to
    channels_last
        ``True`` if the input to the plugin is channels last
    run_time
        The amount of time, in seconds, to benchmark the plugin at each batch size
    """
    # TODO This is not currently used as information from single model profiling is limited and
    # adds additional time to profiling. However this is likely to be useful for deciding on device
    # allocation if/when multi-gpu support is added
    def __init__(self,
                 plugin: ExtractPlugin,
                 max_batch_size: int = 128,
                 channels_last: bool = False,
                 run_time: int = 10) -> None:
        logger.debug(parse_class_init(locals()))
        self.plugin = plugin
        self._max_batch_size = max_batch_size
        self.channels_last = channels_last
        """True if the plugin expects channels last input"""
        self._run_time = run_time

        num_tests = int(math.log2(self._max_batch_size)) + 1
        self.batch_sizes = np.fromiter((2 ** i for i in range(num_tests)), dtype=np.int64)
        self.iterations = np.zeros((num_tests, ), dtype=np.int64) - 1
        self.vram = np.zeros((2, num_tests), dtype=np.int64) - 1

        torch.cuda.empty_cache()
        plugin.batch_size = 1
        plugin.model = plugin.load_model()

    @property
    def run_time(self) -> int:
        """The amount of time, in seconds, that benchmarks were ran per batch"""
        return self._run_time

    def __repr__(self) -> str:
        """Pretty print for logging"""
        params = {k[1:]: repr(v) for k, v in self.__dict__.items()
                  if k in ("_plugin", "_max_batch_size", "_channels_last", "_run_time")}
        results = {k: v.tolist() for k, v in self.__dict__.items()
                   if k in ("batch_sizes", "iterations", "vram")}
        s_params = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({s_params}) {results}"

    def _predict(self, inputs: np.ndarray, seconds: float) -> int:
        """Run inference on a plugin for the given number of seconds

        Parameters
        ----------
        inputs
            The input to use for benchmarking the plugin
        seconds
            The number of seconds to run benchmarking

        Returns
        -------
        The number of iterations that were processed through the plugin
        """
        start = perf_counter()
        iters = 0
        while perf_counter() - start < seconds:
            self.plugin.process(inputs)
            iters += 1
        torch.cuda.synchronize()
        return iters

    def _output_stats(self) -> None:
        """Print the benchmark results to screen in a format that is easy to read and can be
        copy and pasted (fixed width)"""
        egs = [(i * b) / self._run_time for i, b in zip(self.iterations, self.batch_sizes)]
        bs_str = [str(i) for i in self.batch_sizes]
        eg_str = ["N/A" if i < 0 else f"{i:.1f}" for i in egs]
        vram_alloc_str = ["N/A" if i < 0 else str(int(round(i / (1024 * 1024))))
                          for i in self.vram[0]]
        vram_res_str = ["N/A" if i < 0 else str(int(round(i / (1024 * 1024))))
                        for i in self.vram[1]]
        labels = ["BatchSize", "EG/S", "VRAM(MB) Allocated", "VRAM(MB) Reserved"]

        lbl_width = max(len(i) for i in labels)
        col_width = max(len(i) for i in bs_str + eg_str + vram_alloc_str + vram_res_str) + 2

        for lbl, data in zip(labels, (bs_str, eg_str, vram_alloc_str, vram_res_str)):
            dat = "".join([d.rjust(col_width) for d in data])
            print(f"    {lbl.ljust(lbl_width)}{dat}")

    def __call__(self) -> None:
        """Runs benchmarking through the plugin, stores the data and outputs stats"""
        logger.info("Profiling %s", self.plugin.name)
        prog_bar = tqdm(self.batch_sizes, desc="Batch size 1", leave=False, smoothing=0)
        for idx, batch_size in enumerate(prog_bar):
            inputs = random_input_from_plugin(self.plugin, batch_size, self.channels_last)
            try:
                torch.cuda.empty_cache()
                self._predict(inputs, 2)  # warmup
                torch.cuda.reset_peak_memory_stats()

                iters = self._predict(inputs, self._run_time)

                self.iterations[idx] = iters
                self.vram[0, idx] = torch.cuda.max_memory_allocated()
                self.vram[1, idx] = torch.cuda.max_memory_reserved()
            except torch.cuda.OutOfMemoryError:
                logger.debug("Exiting benchmark early as out of VRAM")
                break
            prog_bar.set_description(f"Batch size {batch_size}")

        self._output_stats()
        del self.plugin.model


@dataclass
class Events:
    """Holds thread events for communicating between main thread and plugins during benchmarking

    Parameters
    ----------
    ready
        List of events for each plugin in the pipeline to be tested
    """
    ready: list[Event]
    start = Event()
    stop = Event()
    continue_ = Event()

    def set_ready(self, index: int):
        """Set the ready event for the given index

        Parameters
        ----------
        index
            The index of the ready event to set
        """
        self.ready[index].set()

    def wait_ready(self) -> None:
        """Wait for all ready events to set their ready flag and clear the flag"""
        for ready in self.ready:
            ready.wait()
            ready.clear()


@dataclass
class DataTracker:  # pylint:disable=too-many-instance-attributes
    """Stores data from the benchmarking process

    Parameters
    ----------
    size
        The number of plugins that data is being tracked for
    face_scaling
        The amount of scaling to apply to downstream non-detection plugins
    has_detector
        ``True`` if the first plugin in the pipeline is a detector
    max_vram
        The maximum amount of total VRAM to allow Cuda to reserve when profiling
    """
    size: InitVar[int]
    max_vram: InitVar[float]
    face_scaling: int
    has_detector: bool

    vram: list[tuple[int, int]] = field(init=False, default_factory=list)
    """list of (max allocated, max reserved) VRAM for each testing phase"""
    vram_limit: float = field(init=False)
    """The limit that Cuda reserved memory must remain within"""
    combos_exhausted: bool = field(init=False, default=False)
    """``True`` if we have run out of possible combinations to attempt"""

    _all_batch_sizes: npt.NDArray[np.int64] = field(init=False)
    """The processed batch size configurations including failed tests"""
    _iterations: npt.NDArray[np.int64] = field(init=False)
    """Iterations put through each plugin at each testing phase"""
    _batch_size_adjust: npt.NDArray[np.int64] = field(init=False)
    """Amount to adjust batch sizes by when we are approaching VRAM limits"""
    _success: npt.NDArray[np.bool_] = field(init=False)
    """Booleans that show if each test successfully completed or OOM'd"""

    _lock: Lock = field(init=False, default_factory=Lock)
    """Threading lock for updating iteration counts"""
    _name: str = field(init=False, default="Profile.DataTracker")
    """Name of dataclass for logging"""

    def __post_init__(self, size: int, max_vram: float) -> None:
        """Create the data storage arrays for the given input size

        Parameters
        ----------
        size
            The number of plugins that data is being tracked for
        max_vram
            The maximum amount of total VRAM to allow Cuda to reserve when profiling
        """
        self.vram_limit = torch.cuda.get_device_properties().total_memory * max_vram
        self._all_batch_sizes = np.ones((1, size), dtype=np.int64)
        self._success = np.array([True], dtype=bool)
        self._batch_size_adjust = np.zeros((size, ), dtype=np.int64) - 1
        self._iterations = np.zeros((1, size, ), dtype="int") - 1

    @property
    def has_oom(self) -> bool:
        """``True`` if the last iteration hit an OOM or fell outside our max VRAM threshold"""
        if not self.vram:
            return False
        return any([np.any(self._iterations[-1] < 0), self.vram[-1][1] > self.vram_limit])

    @property
    def batch_sizes(self) -> npt.NDArray[np.int64]:
        """All batch size combinations that did not OOM"""
        return self._all_batch_sizes[self._success]

    def update_iterations(self, iterations: int, matrix_id: int) -> None:
        """Update the iteration count from a plugin runner in a thread-safe way

        Parameters
        ----------
        iterations
            The iteration count for the plugin
        matrix_id
            The column id that belongs to the plugin
        """
        with self._lock:
            self._iterations[-1, matrix_id] = iterations

    def add_iterations_row(self) -> None:
        """Add a new row to the iterations list"""
        with self._lock:
            new_row = np.zeros((1, len(self._iterations[-1])), dtype="int") - 1
            self._iterations = np.concatenate([self._iterations, new_row])

    def collect_vram(self) -> None:
        """Store the currently allocated and reserved Cuda VRAM stats"""
        self.vram.append((torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()))
        logger.debug("[%s] VRAM collected: %s", self._name,  self.vram[-1])

    def get_samples(self, index: int | None = None, adjusted: bool = False
                    ) -> npt.NDArray[np.float64]:
        """Obtain the number of sample processed by each plugin for a certain valid batch size
        combination

        Parameters
        ----------
        index
            The testing index to obtain the samples for or ``None`` for all tests
        adjusted
            ``True`` to obtain results adjusted for any non-detector scaling. Default: ``False``

        Returns
        -------
        The number of samples processed by each plugin
        """
        iters = self._iterations[self._success]
        batches = self.batch_sizes
        if index is not None:
            iters = iters[index]
            batches = batches[index]

        retval = (iters * batches).astype(np.float64)
        if adjusted and self.has_detector and self.face_scaling > 1:
            if index is None:
                retval[:, 1:] /= self.face_scaling
            else:
                retval[1:] /= self.face_scaling
        logger.debug("[%s] Calculated samples/plugin: %s", self._name,  retval.tolist())
        return retval

    def get_samples_stats(self,
                          method: T.Literal["mean", "min"],
                          index: int | None = None,
                          adjusted: bool = False) -> npt.NDArray[np.float64]:
        """Obtain the average or minimum samples processed for all plugins for a certain batch size
        combination

        Parameters
        ----------
        method
            ``mean`` to obtain the mean number of samples for all plugins. ``min`` to obtain the
            minimum number of samples processed by a plugin
        index
            The testing index to obtain the average samples for or ``None`` for all tests.
            Default: ``None``
        adjusted
            ``True`` to obtain results adjusted for any non-detector scaling. Default: ``False``

        Returns
        -------
        The average number of samples processed by all plugins
        """
        samples = self.get_samples(index=index, adjusted=adjusted)
        dim = 1 if index is None else 0
        if method == "mean":
            retval = samples.mean(axis=dim)
        else:
            retval = samples.min(axis=dim)
        logger.debug("[%s] Calculated Average samples/plugin: %s", self._name,  retval.tolist())
        return retval

    def _handle_oom(self) -> None:
        """Update :attr:`_batch_size_adjust` in cases when we hit an OOM or exceeded our VRAM
        threshold. In these instances we will either shrink our search window, or exit if we have
        gone as far as we can"""
        if not self.has_oom:
            return
        self._success[-1] = False

        changed_mask = self._all_batch_sizes[-1] != self._all_batch_sizes[-2]
        diff = abs(self._all_batch_sizes[-1][changed_mask] -
                   self._all_batch_sizes[-2][changed_mask])
        if diff <= 4:
            logger.debug("[%s] Minimum batch size adjustment hit. All combos exhausted",
                         self._name)
            self.combos_exhausted = True
            return
        self._batch_size_adjust[changed_mask] = diff // 2
        logger.debug("[%s] batch_size_adjust updated to: %s",
                     self._name,  self._batch_size_adjust.tolist())

    def add_next_batch_sizes(self) -> None:
        """Add the next batch size configuration to the batch size array based on the output from
        the last test"""
        self._handle_oom()
        if self.combos_exhausted:
            return

        samples = self.get_samples(-1, adjusted=True)
        p_idx = samples.argmin()
        _batch_size_adjust = self._batch_size_adjust[p_idx]

        next_batch = self.batch_sizes[-1].copy()
        if _batch_size_adjust == -1:
            next_batch[p_idx] *= 2
        else:
            next_batch[p_idx] += _batch_size_adjust
        logger.debug("[%s] next batch sizes: %s", self._name, next_batch.tolist())
        self._all_batch_sizes = np.concatenate([self._all_batch_sizes, next_batch[None]])
        self._success = np.concatenate([self._success, [True]])


class Output:
    """Handles outputting of information at each test step

    Parameters
    ----------
    plugin_names
        The list of plugin names in the order that they are executed
    data
        The DataTracker object that collects stats
    run_time
        The amount of time, in seconds, that each test is run
    """
    def __init__(self, plugin_names: list[str], data: DataTracker, run_time: int):
        logger.debug(parse_class_init(locals()))
        self._data = data
        self._run_time = run_time
        self._header_row = [" " * 18] + plugin_names + ["Average", "Min"]
        self._spacer = "    "
        self._label_widths = [len(h) + 2 for h in self._header_row]

    def _write(self, message_list: list[str], left_justify: bool = False) -> None:
        """TQDM write a message with leading indentation

        Parameters
        ----------
        message_list
            The message to write split over columns
        left_justify
            ``True`` to left justify the data, ``False`` to right justify the data.
            Default: ``False``
        """
        label = message_list[0].ljust(self._label_widths[0])
        message_list = message_list[1:]
        if left_justify:
            msg = " ".join(m.ljust(l) for m, l in zip(message_list, self._label_widths[1:]))
        else:
            msg = " ".join(m.rjust(l) for m, l in zip(message_list, self._label_widths[1:]))
        tqdm.write(f"{self._spacer}{label}{msg}")

    def __call__(self):
        """Output the latest test stats"""
        if self._data.has_oom:
            return

        self._write(self._header_row)
        self._write(["Batch Size"] + [str(int(b)) for b in self._data.batch_sizes[-1]])
        egs = [f"{e:.1f}" for e in self._data.get_samples(-1) / self._run_time]
        avg_egs = [f"{(self._data.get_samples_stats('mean', -1) / self._run_time):.1f}"]
        min_egs = [f"{(self._data.get_samples_stats('min', -1) / self._run_time):.1f}"]
        self._write(["EG/S"] + egs + avg_egs + min_egs)

        if self._data.has_detector and self._data.face_scaling > 1:
            lbl = [f"Scaled EG/S ({self._data.face_scaling}x)"]
            egs = [f"{e:.1f}" for e in self._data.get_samples(-1, adjusted=True) / self._run_time]
            avg_egs = [
                f"{self._data.get_samples_stats('mean', -1, adjusted=True) / self._run_time:.1f}"]
            min_egs = [
                f"{self._data.get_samples_stats('min', -1, adjusted=True) / self._run_time:.1f}"]
            self._write(lbl + egs + avg_egs + min_egs)

        vram_alloc, vram_res = (str(int(round(v / 1024 / 1024))) for v in self._data.vram[-1])
        vram_res = f"{vram_res}/{str(int(round(self._data.vram_limit / 1024 / 1024)))}"
        self._write(["VRAM(MB) Allocated", vram_alloc], left_justify=True)
        self._write(["VRAM(MB) Reserved", vram_res], left_justify=True)

        line = "-" * (sum(self._label_widths) + (len(self._label_widths) - 2))
        tqdm.write(f"{self._spacer}{line}")


class PipelineProfile():
    """Benchmark multiple PyTorch GPU plugins running simultaneously for inference

    Parameters
    ----------
    plugins
        The plugins to benchmark for inference
    error_state
        The global FSThread error state object for the pipeline
    channels_last
        List indicating whether each model is channels first or last
    warmup_time
        The amount of time, in seconds, to warmup the plugin at each batch size
    run_time
        The amount of time, in seconds, to benchmark the plugin at each batch size
    has_detector
        ``True`` if the first plugin in the pipeline is a detector
    face_scaling
        The amount of scaling to apply to downstream plugins (ie estimate of average number of
        faces per frame). Default: 2
    max_vram
        The maximum percentage of total VRAM to allow Cuda to reserve when profiling, Default: 90
    """
    def __init__(self,
                 plugins: list[ExtractPlugin],
                 error_state: ErrorState,
                 channels_last: list[bool],
                 warmup_time: int,
                 run_time: int,
                 has_detector: bool,
                 face_scaling: int = 2,
                 max_vram: int = 90) -> None:
        logger.debug(parse_class_init(locals()))
        self._warmup_time = warmup_time
        self._run_time = run_time
        self._current_index = 0
        self._plugins = plugins
        self._error_state = error_state

        self._events = Events(ready=[Event() for _ in range(len(plugins))])
        self._data = DataTracker(len(plugins),
                                 max_vram / 100.,
                                 face_scaling,
                                 has_detector)
        self._output_stats = Output([p.name for p in plugins], self._data, run_time)

        self._threads = [FSThread(self._plugin_runner,
                                  name=f"{p.name}_thread",
                                  args=(p, i, c))
                         for i, (p, c) in enumerate(zip(plugins, channels_last))]

    @classmethod
    def _predict(cls, plugin: ExtractPlugin, inputs: np.ndarray, seconds: float) -> int:
        """Run inference on a plugin for the given number of seconds

        Parameters
        ----------
        plugin
            The plugin to run inference through
        inputs
            The input to use for benchmarking the plugin
        seconds
            The number of seconds to run benchmarking

        Returns
        -------
        The number of iterations that were processed through the plugin
        """
        start = perf_counter()
        iters = 0
        while perf_counter() - start < seconds:
            plugin.process(inputs)
            iters += 1
        torch.cuda.synchronize()
        return iters

    def _plugin_runner(self, plugin: ExtractPlugin, matrix_id: int, channels_last: bool) -> None:
        """Runs a plugin inside a thread, waits and reports to main thread by means of events

        Parameters
        ----------
        plugin
            The plugin that this thread will run
        matrix_id
            The column id to obtain the batch size for this plugin from :attr:`matrix`
        channels_last
            ``True`` if the input to the plugin is channels last
        """
        name = plugin.name
        logger.debug("[PipelineProfile] Loading '%s' (id: %s)", name, matrix_id)
        plugin.batch_size = 1
        plugin.model = plugin.load_model()
        while True:
            if self._error_state.has_error:
                self._error_state.re_raise()
            self._events.start.wait()
            if self._events.stop.is_set():
                break
            batch_size = self._data.batch_sizes[-1][matrix_id]
            inputs = random_input_from_plugin(plugin, batch_size, channels_last)
            logger.debug("[PipelineProfile] Running test '%s'. input: %s", name, inputs.shape)
            try:
                self._predict(plugin, inputs, self._warmup_time)  # warmup
                self._events.set_ready(matrix_id)

                self._events.continue_.wait()
                iters = self._predict(plugin, inputs, self._run_time)
                self._data.update_iterations(iters, matrix_id)
                self._events.set_ready(matrix_id)

            except torch.cuda.OutOfMemoryError:
                logger.debug("[PipelineProfile] Exiting benchmark early as out of VRAM")
                self._events.set_ready(matrix_id)
                if self._events.stop.is_set():
                    break
        del plugin.model

    def _update_batch_sizes(self) -> None:
        """Output final batch sizes and update the plugins"""
        best_idx = self._data.get_samples(adjusted=True).min(axis=1).argmax()
        best_batch_sizes = self._data.batch_sizes[best_idx]
        plugin_names = [p.name for p in self._plugins]
        logger.info("[Profiler] Setting optimal batch sizes: %s",
                    ", ".join(f"{p}: {b}" for p, b in zip(plugin_names,
                                                          self._data.batch_sizes[best_idx])))

        for plugin, batch_size in zip(self._plugins, best_batch_sizes):
            logger.debug("[PipelineProfile] Updating batch size for '%s': %s",
                         plugin.name, batch_size)
            plugin.batch_size = int(batch_size)

    def __call__(self) -> None:
        """Runs benchmarking through all plugins concurrently, store the data and output stats"""
        prog_length = 5
        for thread in self._threads:
            thread.start()

        while True:
            if self._error_state.has_error:
                self._error_state.re_raise()

            msg = f"[{self._current_index}] Batches {tuple(self._data.batch_sizes[-1].tolist())}"
            prog_bar = tqdm(desc=f"Benchmarking Pipeline {msg}", total=prog_length, leave=False)
            torch.cuda.empty_cache()

            # Warmup
            prog_bar.update()
            self._events.start.set()
            self._events.wait_ready()
            prog_bar.update()
            self._events.start.clear()

            # Benchmark
            torch.cuda.reset_peak_memory_stats()
            self._events.continue_.set()
            prog_bar.update()
            self._events.wait_ready()
            prog_bar.update()
            self._events.continue_.clear()
            self._data.collect_vram()

            self._output_stats()

            prog_bar.update()
            self._data.add_next_batch_sizes()
            if self._data.combos_exhausted:
                prog_bar.close()
                break

            self._data.add_iterations_row()
            self._current_index += 1
            prog_bar.close()

        self._events.stop.set()
        self._events.start.set()
        for thread in self._threads:
            thread.join()
        self._update_batch_sizes()


class Profiler:
    """Profiles plugins within a pipeline

    Parameters
    ----------
    runner
        The output runner from an extract pipeline that is to be profiled
    """
    def __init__(self, runner: ExtractRunner) -> None:
        logger.debug(parse_class_init(locals()))
        logger.info("Profiling models...")
        self._chain = T.cast("list[ExtractRunner[ExtractHandler]]",  # For intellisense purposes
                             get_pipeline(runner))
        self._channels_last: list[bool] = []
        self._torch_runners = self._get_torch_indices()

    def _check_for_torch(self, plugin: ExtractPlugin) -> bool:
        """Check whether the given runner uses PyTorch. We wait until the plugin is initialized
        then recurse through it's :attr:`model` property looking for Torch Modules

        Parameters
        ----------
        plugin
            The plugin to check for PyTorch usage

        Returns
        -------
        bool
            ``True`` if the runner uses PyTorch
        """
        model = plugin.load_model()
        logger.debug("[Profiler] Scanning for torch Module: %s(%s)",
                     plugin.name, model.__class__.__name__)
        modules = get_torch_modules(model)
        if not modules:
            return False

        plugin.model = model
        channels_last = warmup_plugin(plugin, 1)
        assert channels_last is not None
        self._channels_last.append(channels_last)
        del plugin.model
        return True

    def _get_torch_indices(self) -> list[int]:
        """Obtain the indices within :attr:`_chain` that contain models running on pyTorch on the
        GPU

        Returns
        -------
        The list of indices of the runners that are running PyTorch models on the GPU
        """
        retval: list[int] = []
        for idx, runner in enumerate(self._chain):
            if runner.handler.plugin.device.type == "cpu":
                logger.debug("[Profiler] Skipping CPU model: '%s'", runner.handler.plugin_name)
                continue
            if self._check_for_torch(runner.handler.plugin):
                logger.debug("[Profiler] Adding: '%s'", runner.handler.plugin.name)
                retval.append(idx)
                continue
            logger.debug("[Profiler] Skipping: '%s'", runner.handler.plugin.name)

        logger.debug("[Profiler] Torch runners indices: %s", retval)
        if len(self._channels_last) != len(retval):
            raise RuntimeError("Failed to get all channels_last information")
        return retval

    def _profile_isolated(self) -> list[ModelProfile]:
        """Benchmark the models in isolation and return the benchmark objects

        Returns
        -------
        The benchmark object for each plugin tested
        """
        retval: list[ModelProfile] = []
        for idx, chan_last in zip(self._torch_runners, self._channels_last):
            plugin = self._chain[idx].handler.plugin
            profile = ModelProfile(plugin, channels_last=chan_last)
            logger.debug("Benchmarking %s (%s/%s)", plugin.name, idx + 1, len(self._torch_runners))
            profile()
            retval.append(profile)
        return retval

    @classmethod
    def _update_config_file(cls, plugins: list[ExtractPlugin]):
        """Update the config file if requested in settings

        Parameters
        ----------
        The plugins that have had their throughput profiled
        """
        if not cfg.profile_save_config():
            return
        conf = cfg.load_config()
        f_names = [".".join(p.__class__.__module__.rsplit(".", maxsplit=2)[-2:]) for p in plugins]

        is_updated = False
        for plugin_name, plugin in zip(f_names, plugins):
            opts = conf.sections[plugin_name].options
            opt = opts.get("batch_size", opts.get("batch_size"))
            if not opt:
                logger.warning("Could not update Config file for '%s' as no 'batch_size' "
                               "entry found", plugin.name)
                continue
            old_val = opt()
            new_val = plugin.batch_size
            if old_val == new_val:
                logger.debug("[Profiler] Skipping unchanged batch size %s for '%s'",
                             old_val, plugin_name)
                continue
            logger.debug("[Profiler] Updating batch size from %s to %s for '%s'",
                         old_val, new_val, plugin_name)
            is_updated = True
            opt.set(new_val)

        if not is_updated:
            logger.info("No batch sizes were updated from their saved values. "
                        "Not saving config file")
            return
        logger.info("Saving config file with updated batch sizes")
        conf.save_config()

    def __call__(self) -> None:
        """Call the profiler"""
        # model_benchmarks = self._profile_isolated()  # Unused. Kept for if/when multi-gpu support
        plugins = [r.handler.plugin for r in itemgetter(*self._torch_runners)(self._chain)]
        has_detector = self._chain[self._torch_runners[0]].handler.plugin_type == "detect"
        pipeline_benchmarks = PipelineProfile(plugins,
                                              self._chain[0]._threads.error_state,
                                              self._channels_last,
                                              cfg.profile_warmup_time(),
                                              cfg.profile_test_time(),
                                              has_detector,
                                              cfg.profile_num_faces(),
                                              cfg.profile_max_vram())
        pipeline_benchmarks()
        self._update_config_file(plugins)
        torch.cuda.empty_cache()
        logger.debug("[Profiler] Starting plugin threads")
        for runner in self._chain:
            runner.start()


__all__ = get_module_objects(__name__)
