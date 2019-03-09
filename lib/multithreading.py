#!/usr/bin/env python3
""" Multithreading/processing utils for faceswap """

import logging
import multiprocessing as mp
from multiprocessing.sharedctypes import RawArray
from ctypes import c_float

import queue as Queue
import sys
import threading
import numpy as np
from lib.logger import LOG_QUEUE, set_root_logger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_launched_processes = set()  # pylint: disable=invalid-name


class ConsumerBuffer():
    """ Memory buffer for consuming """
    def __init__(self, dispatcher, index, data):
        logger.trace("Initializing %s: (dispatcher: '%s', index: %s, data: %s)",
                     self.__class__.__name__, dispatcher, index, data)
        self._data = data
        self._id = index
        self._dispatcher = dispatcher
        logger.trace("Initialized %s", self.__class__.__name__)

    def get(self):
        """ Return Data """
        return self._data

    def free(self):
        """ Return Free """
        self._dispatcher.free(self._id)

    def __enter__(self):
        """ On Enter """
        return self.get()

    def __exit__(self, *args):
        """ On Exit """
        self.free()


class WorkerBuffer():
    """ Memory buffer for working """
    def __init__(self, index, data, stop_event, queue):
        logger.trace("Initializing %s: (index: '%s', data: %s, stop_event: %s, queue: %s)",
                     self.__class__.__name__, index, data, stop_event, queue)
        self._id = index
        self._data = data
        self._stop_event = stop_event
        self._queue = queue
        logger.trace("Initialized %s", self.__class__.__name__)

    def get(self):
        """ Return Data """
        return self._data

    def ready(self):
        """ Worker Ready """
        if self._stop_event.is_set():
            return
        self._queue.put(self._id)

    def __enter__(self):
        """ On Enter """
        return self.get()

    def __exit__(self, *args):
        """ On Exit """
        self.ready()


class FixedProducerDispatcher():
    """
    Runs the given method in N subprocesses
    and provides fixed size shared memory to the method.
    This class is designed for endless running worker processes
    filling the provided memory with data,
    like preparing trainingsdata for neural network training.

    As soon as one worker finishes all worker are shutdown.

    Example:
        # Producer side
        def do_work(memory_gen):
            for memory_wrap in memory_gen:
                # alternative memory_wrap.get and memory_wrap.ready can be used
                with memory_wrap as memory:
                    input, exp_result = prepare_batch(...)
                    memory[0][:] = input
                    memory[1][:] = exp_result

        # Consumer side
        batch_size = 64
        dispatcher = FixedProducerDispatcher(do_work, shapes=[
            (batch_size, 256,256,3), (batch_size, 256,256,3)])
        for batch_wrapper in dispatcher:
            # alternative batch_wrapper.get and batch_wrapper.free can be used
            with batch_wrapper as batch:
                send_batch_to_trainer(batch)
    """
    CTX = mp.get_context("spawn")
    EVENT = CTX.Event

    def __init__(self, method, shapes, in_queue, out_queue,
                 args=tuple(), kwargs={}, ctype=c_float, workers=1, buffers=None):
        logger.debug("Initializing %s: (method: '%s', shapes: %s, args: %s, kwargs: %s, "
                     "ctype: %s, workers: %s, buffers: %s)", self.__class__.__name__, method,
                     shapes, args, kwargs, ctype, workers, buffers)
        if buffers is None:
            buffers = workers * 2
        else:
            assert buffers >= 2 and buffers > workers
        self.name = "%s_FixedProducerDispatcher" % str(method)
        self._target_func = method
        self._shapes = shapes
        self._stop_event = self.EVENT()
        self._buffer_tokens = in_queue
        for i in range(buffers):
            self._buffer_tokens.put(i)
        self._result_tokens = out_queue
        worker_data, self.data = self._create_data(shapes, ctype, buffers)
        proc_args = {
            'data': worker_data,
            'stop_event': self._stop_event,
            'target': self._target_func,
            'buffer_tokens': self._buffer_tokens,
            'result_tokens': self._result_tokens,
            'dtype': np.dtype(ctype),
            'shapes': shapes,
            'log_queue': LOG_QUEUE,
            'log_level': logger.getEffectiveLevel(),
            'args': args,
            'kwargs': kwargs
        }
        self._worker = tuple(self._create_worker(proc_args) for _ in range(workers))
        self._open_worker = len(self._worker)
        logger.debug("Initialized %s", self.__class__.__name__)

    @staticmethod
    def _np_from_shared(shared, shapes, dtype):
        """ Numpy array from shared memory """
        arrs = []
        offset = 0
        np_data = np.frombuffer(shared, dtype=dtype)
        for shape in shapes:
            count = np.prod(shape)
            arrs.append(np_data[offset:offset+count].reshape(shape))
            offset += count
        return arrs

    def _create_data(self, shapes, ctype, buffers):
        """ Create data """
        buffer_size = int(sum(np.prod(x) for x in shapes))
        dtype = np.dtype(ctype)
        data = tuple(RawArray(ctype, buffer_size) for _ in range(buffers))
        np_data = tuple(self._np_from_shared(arr, shapes, dtype) for arr in data)
        return data, np_data

    def _create_worker(self, kwargs):
        """ Create Worker """
        return self.CTX.Process(target=self._runner, kwargs=kwargs)

    def free(self, index):
        """ Free memory """
        if self._stop_event.is_set():
            return
        if isinstance(index, ConsumerBuffer):
            index = index.index
        self._buffer_tokens.put(index)

    def __iter__(self):
        """ Iterator """
        return self

    def __next__(self):
        """ Next item """
        return self.next()

    def next(self, block=True, timeout=None):
        """
        Yields ConsumerBuffer filled by the worker.
        Will raise StopIteration if no more elements are available OR any worker is finished.
        Will raise queue.Empty when block is False and no element is available.

        The returned data is safe until ConsumerBuffer.free() is called or the
        with context is left. If you plan to hold on to it after that make a copy.

        This method is thread safe.
        """
        if self._stop_event.is_set():
            raise StopIteration
        i = self._result_tokens.get(block=block, timeout=timeout)
        if i is None:
            self._open_worker -= 1
            raise StopIteration
        if self._stop_event.is_set():
            raise StopIteration
        return ConsumerBuffer(self, i, self.data[i])

    def start(self):
        """ Start Workers """
        for process in self._worker:
            process.start()
        _launched_processes.add(self)

    def is_alive(self):
        """ Check workers are alive """
        for worker in self._worker:
            if worker.is_alive():
                return True
        return False

    def join(self):
        """ Join Workers """
        self.stop()
        while self._open_worker:
            if self._result_tokens.get() is None:
                self._open_worker -= 1
        while True:
            try:
                self._buffer_tokens.get(block=False, timeout=0.01)
            except Queue.Empty:
                break
        for worker in self._worker:
            worker.join()

    def stop(self):
        """ Stop Workers """
        self._stop_event.set()
        for _ in range(self._open_worker):
            self._buffer_tokens.put(None)

    def is_shutdown(self):
        """ Check if stop event is set """
        return self._stop_event.is_set()

    @classmethod
    def _runner(cls, data=None, stop_event=None, target=None,
                buffer_tokens=None, result_tokens=None, dtype=None,
                shapes=None, log_queue=None, log_level=None,
                args=None, kwargs=None):
        """ Shared Memory Object runner """
        # Fork inherits the queue handler, so skip registration with "fork"
        set_root_logger(log_level, queue=log_queue)
        logger.debug("FixedProducerDispatcher worker for %s started", str(target))
        np_data = [cls._np_from_shared(d, shapes, dtype) for d in data]

        def get_free_slot():
            while not stop_event.is_set():
                i = buffer_tokens.get()
                if stop_event.is_set() or i is None or i == "EOF":
                    break
                yield WorkerBuffer(i, np_data[i], stop_event, result_tokens)

        args = tuple((get_free_slot(),)) + tuple(args)
        try:
            target(*args, **kwargs)
        except Exception as ex:
            logger.exception(ex)
            stop_event.set()
        result_tokens.put(None)
        logger.debug("FixedProducerDispatcher worker for %s shutdown", str(target))


class PoolProcess():
    """ Pool multiple processes """
    def __init__(self, method, in_queue, out_queue, *args, processes=None, **kwargs):
        self._name = method.__qualname__
        logger.debug("Initializing %s: (target: '%s', processes: %s)",
                     self.__class__.__name__, self._name, processes)

        self.procs = self.set_procs(processes)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(processes=self.procs,
                             initializer=set_root_logger,
                             initargs=(logger.getEffectiveLevel(), LOG_QUEUE))
        self._method = method
        self._kwargs = self.build_target_kwargs(in_queue, out_queue, kwargs)
        self._args = args

        logger.debug("Initialized %s: '%s'", self.__class__.__name__, self._name)

    @staticmethod
    def build_target_kwargs(in_queue, out_queue, kwargs):
        """ Add standard kwargs to passed in kwargs list """
        kwargs["in_queue"] = in_queue
        kwargs["out_queue"] = out_queue
        return kwargs

    def set_procs(self, processes):
        """ Set the number of processes to use """
        if processes is None:
            running_processes = len(mp.active_children())
            processes = max(mp.cpu_count() - running_processes, 1)
        logger.verbose("Processing '%s' in %s processes", self._name, processes)
        return processes

    def start(self):
        """ Run the processing pool """
        logging.debug("Pooling Processes: (target: '%s', args: %s, kwargs: %s)",
                      self._name, self._args, self._kwargs)
        for idx in range(self.procs):
            logger.debug("Adding process %s of %s to mp.Pool '%s'",
                         idx + 1, self.procs, self._name)
            self.pool.apply_async(self._method, args=self._args, kwds=self._kwargs)
        logging.debug("Pooled Processes: '%s'", self._name)

    def join(self):
        """ Join the process """
        logger.debug("Joining Pooled Process: '%s'", self._name)
        self.pool.close()
        self.pool.join()
        logger.debug("Joined Pooled Process: '%s'", self._name)


class SpawnProcess(mp.context.SpawnProcess):
    """ Process in spawnable context
        Must be spawnable to share CUDA across processes """
    def __init__(self, target, in_queue, out_queue, *args, **kwargs):
        name = target.__qualname__
        logger.debug("Initializing %s: (target: '%s', args: %s, kwargs: %s)",
                     self.__class__.__name__, name, args, kwargs)
        ctx = mp.get_context("spawn")
        self.event = ctx.Event()
        self.error = ctx.Event()
        kwargs = self.build_target_kwargs(in_queue, out_queue, kwargs)
        super().__init__(target=target, name=name, args=args, kwargs=kwargs)
        self.daemon = True
        logger.debug("Initialized %s: '%s'", self.__class__.__name__, name)

    def build_target_kwargs(self, in_queue, out_queue, kwargs):
        """ Add standard kwargs to passed in kwargs list """
        kwargs["event"] = self.event
        kwargs["error"] = self.error
        kwargs["log_init"] = set_root_logger
        kwargs["log_queue"] = LOG_QUEUE
        kwargs["log_level"] = logger.getEffectiveLevel()
        kwargs["in_queue"] = in_queue
        kwargs["out_queue"] = out_queue
        return kwargs

    def run(self):
        """ Add logger to spawned process """
        logger_init = self._kwargs["log_init"]
        log_queue = self._kwargs["log_queue"]
        log_level = self._kwargs["log_level"]
        logger_init(log_level, log_queue)
        super().run()

    def start(self):
        """ Add logging to start function """
        logger.debug("Spawning Process: (name: '%s', args: %s, kwargs: %s, daemon: %s)",
                     self._name, self._args, self._kwargs, self.daemon)
        super().start()
        _launched_processes.add(self)
        logger.debug("Spawned Process: (name: '%s', PID: %s)", self._name, self.pid)

    def join(self, timeout=None):
        """ Add logging to join function """
        logger.debug("Joining Process: (name: '%s', PID: %s)", self._name, self.pid)
        super().join(timeout=timeout)
        _launched_processes.remove(self)
        logger.debug("Joined Process: (name: '%s', PID: %s)", self._name, self.pid)


class FSThread(threading.Thread):
    """ Subclass of thread that passes errors back to parent """
    def __init__(self, group=None, target=None, name=None,  # pylint: disable=too-many-arguments
                 args=(), kwargs=None, *, daemon=None):
        super().__init__(group=group, target=target, name=name,
                         args=args, kwargs=kwargs, daemon=daemon)
        self.err = None

    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception:  # pylint: disable=broad-except
            self.err = sys.exc_info()
            logger.debug("Error in thread (%s): %s", self._name,
                         self.err[1].with_traceback(self.err[2]))
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


class MultiThread():
    """ Threading for IO heavy ops
        Catches errors in thread and rethrows to parent """
    def __init__(self, target, *args, thread_count=1, name=None, **kwargs):
        self._name = name if name else target.__name__
        logger.debug("Initializing %s: (target: '%s', thread_count: %s)",
                     self.__class__.__name__, self._name, thread_count)
        logger.trace("args: %s, kwargs: %s", args, kwargs)
        self.daemon = True
        self._thread_count = thread_count
        self._threads = list()
        self._target = target
        self._args = args
        self._kwargs = kwargs
        logger.debug("Initialized %s: '%s'", self.__class__.__name__, self._name)

    @property
    def has_error(self):
        """ Return true if a thread has errored, otherwise false """
        return any(thread.err for thread in self._threads)

    @property
    def errors(self):
        """ Return a list of thread errors """
        return [thread.err for thread in self._threads]

    def start(self):
        """ Start a thread with the given method and args """
        logger.debug("Starting thread(s): '%s'", self._name)
        for idx in range(self._thread_count):
            name = "{}_{}".format(self._name, idx)
            logger.debug("Starting thread %s of %s: '%s'",
                         idx + 1, self._thread_count, name)
            thread = FSThread(name=name,
                              target=self._target,
                              args=self._args,
                              kwargs=self._kwargs)
            thread.daemon = self.daemon
            thread.start()
            self._threads.append(thread)
        logger.debug("Started all threads '%s': %s", self._name, len(self._threads))

    def join(self):
        """ Join the running threads, catching and re-raising any errors """
        logger.debug("Joining Threads: '%s'", self._name)
        for thread in self._threads:
            logger.debug("Joining Thread: '%s'", thread._name)  # pylint: disable=protected-access
            thread.join()
            if thread.err:
                logger.error("Caught exception in thread: '%s'",
                             thread._name)  # pylint: disable=protected-access
                raise thread.err[1].with_traceback(thread.err[2])
        logger.debug("Joined all Threads: '%s'", self._name)


class BackgroundGenerator(threading.Thread):
    """ Run a queue in the background. From:
        https://stackoverflow.com/questions/7323664/ """
    # See below why prefetch count is flawed
    def __init__(self, generator, prefetch=1):
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(maxsize=prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        """ Put until queue size is reached.
            Note: put blocks only if put is called while queue has already
            reached max size => this makes 2 prefetched items! One in the
            queue, one waiting for insertion! """
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def iterator(self):
        """ Iterate items out of the queue """
        while True:
            next_item = self.queue.get()
            if next_item is None:
                break
            yield next_item


def terminate_processes():
    """ Join all active processes on unexpected shutdown

        If the process is doing long running work, make sure you
        have a mechanism in place to terminate this work to avoid
        long blocks
    """
    logger.debug("Processes to join: %s", [process.name
                                           for process in _launched_processes
                                           if process.is_alive()])
    for process in list(_launched_processes):
        if process.is_alive():
            process.join()
