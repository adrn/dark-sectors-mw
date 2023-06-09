"""Modified implementation of the MPIPoolExecutor from mpi4py.futures"""

import functools
import itertools
import sys
import threading
import time

from mpi4py.futures import _lib
from mpi4py.futures._base import Executor, Future, as_completed


class MPIPoolExecutor(Executor):
    """MPI-based asynchronous executor."""

    Future = Future

    def __init__(self, max_workers=None, initializer=None, initargs=(), **kwargs):
        """Initialize a new MPIPoolExecutor instance.

        Args:
            max_workers: The maximum number of MPI processes that can be used
                to execute the given calls. If ``None`` or not given then the
                number of worker processes will be determined from the MPI
                universe size attribute if defined, otherwise a single worker
                process will be spawned.
            initializer: An callable used to initialize workers processes.
            initargs: A tuple of arguments to pass to the initializer.

        Keyword Args:
            python_exe: Path to Python executable used to spawn workers.
            python_args: Command line arguments to pass to Python executable.
            mpi_info: Mapping or iterable with ``(key, value)`` pairs.
            globals: Mapping with global variables to set in workers.
            main: If ``False``, do not import ``__main__`` in workers.
            path: List of paths to append to ``sys.path`` in workers.
            wdir: Path to set current working directory in workers.
            env: Environment variables to update ``os.environ`` in workers.
            use_pkl5: If ``True``, use pickle5 out-of-band for communication.

        """
        if max_workers is not None:
            max_workers = int(max_workers)
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            kwargs["max_workers"] = max_workers
        if initializer is not None:
            if not callable(initializer):
                raise TypeError("initializer must be a callable")
            kwargs["initializer"] = initializer
            kwargs["initargs"] = tuple(initargs)

        self._options = kwargs
        self._shutdown = False
        self._broken = None
        self._lock = threading.Lock()
        self._pool = None

        _comm = _lib.get_comm_world()
        self.size = _comm.Get_size() - 1

    _make_pool = staticmethod(_lib.WorkerPool)

    def _bootstrap(self):
        if self._pool is None:
            self._pool = self._make_pool(self)

    @property
    def _max_workers(self):
        with self._lock:
            if self._broken:
                return None
            if self._shutdown:
                return None
            self._bootstrap()
            self._pool.wait()
            return self._pool.size

    def bootup(self, wait=True):
        """Allocate executor resources eagerly.

        Args:
            wait: If ``True`` then bootup will not return until the
                executor resources are ready to process submissions.

        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot bootup after shutdown")
            self._bootstrap()
            if wait:
                self._pool.wait()
            return self

    def submit(self, fn, *args, **kwargs):
        """Submit a callable to be executed with the given arguments.

        Schedule the callable to be executed as ``fn(*args, **kwargs)`` and
        return a `Future` instance representing the execution of the callable.

        Returns:
            A `Future` representing the given call.

        """
        # pylint: disable=arguments-differ
        with self._lock:
            if self._broken:
                raise _lib.BrokenExecutor(self._broken)
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            self._bootstrap()
            future = self.Future()
            task = (fn, args, kwargs)
            self._pool.push((future, task))
            return future

    if sys.version_info >= (3, 8):  # pragma: no branch
        submit.__text_signature__ = "($self, fn, /, *args, **kwargs)"

    def map(
        self, fn, *iterables, callback=None, timeout=None, chunksize=1, unordered=False
    ):
        """Return an iterator equivalent to ``map(fn, *iterables)``.

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            iterables: Iterables yielding positional arguments to be passed to
                the callable.
            callback: A callable to be called with the result of each worker.
            timeout: The maximum number of seconds to wait. If ``None``, then
                there is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a worker process.
            unordered: If ``True``, yield results out-of-order, as completed.

        Returns:
            An iterator equivalent to built-in ``map(func, *iterables)``
            but the calls may be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If ``fn(*args)`` raises for any values.

        """  # noqa: D402
        return self.starmap(
            fn,
            zip(*iterables),
            callback=callback,
            timeout=timeout,
            chunksize=chunksize,
            unordered=unordered,
        )

    def starmap(
        self, fn, iterable, callback=None, timeout=None, chunksize=1, unordered=False
    ):
        """Return an iterator equivalent to ``itertools.starmap(...)``.

        Args:
            fn: A callable that will take positional argument from iterable.
            iterable: An iterable yielding ``args`` tuples to be used as
                positional arguments to call ``fn(*args)``.
            callback: A callable to be called with the result of each worker.
            timeout: The maximum number of seconds to wait. If ``None``, then
                there is no limit on the wait time.
            chunksize: The size of the chunks the iterable will be broken into
                before being passed to a worker process.
            unordered: If ``True``, yield results out-of-order, as completed.

        Returns:
            An iterator equivalent to ``itertools.starmap(fn, iterable)``
            but the calls may be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If ``fn(*args)`` raises for any values.

        """  # noqa: D402
        # pylint: disable=too-many-arguments
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")
        if chunksize == 1:
            return _starmap_helper(
                self.submit, fn, iterable, timeout, unordered, callback
            )
        else:
            return _starmap_chunks(
                self.submit, fn, iterable, timeout, unordered, chunksize, callback
            )

    def shutdown(self, wait=True, *, cancel_futures=False):
        """Clean-up the resources associated with the executor.

        It is safe to call this method several times. Otherwise, no other
        methods can be called after this one.

        Args:
            wait: If ``True`` then shutdown will not return until all running
                futures have finished executing and the resources used by the
                executor have been reclaimed.
            cancel_futures: If ``True`` then shutdown will cancel all pending
                futures. Futures that are completed or running will not be
                cancelled.

        """
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                if self._pool is not None:
                    self._pool.done()
            if cancel_futures:
                if self._pool is not None:
                    self._pool.cancel()
            pool = None
            if wait:
                pool = self._pool
                self._pool = None
        if pool is not None:
            pool.join()


def _default_callback(x):
    return x


def _starmap_helper(submit, function, iterable, timeout, unordered, callback=None):
    if timeout is not None:
        timer = getattr(time, "monotonic", time.time)
        end_time = timeout + timer()

    if callback is None:
        callback = _default_callback

    futures = [submit(function, *args) for args in iterable]
    if unordered:
        futures = set(futures)

    def result(future, timeout=None):
        try:
            try:
                res = future.result(timeout)
                callback(res)
                return res
            finally:
                future.cancel()
        finally:
            del future

    def result_iterator():
        try:
            if unordered:
                if timeout is None:
                    iterator = as_completed(futures)
                else:
                    iterator = as_completed(futures, end_time - timer())
                for future in iterator:
                    futures.remove(future)
                    future = [future]
                    yield result(future.pop())
            else:
                futures.reverse()
                if timeout is None:
                    while futures:
                        yield result(futures.pop())
                else:
                    while futures:
                        yield result(futures.pop(), end_time - timer())
        finally:
            while futures:
                futures.pop().cancel()

    return result_iterator()


def _apply_chunks(function, chunk):
    return [function(*args) for args in chunk]


def _build_chunks(chunksize, iterable):
    iterable = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(iterable, chunksize))
        if not chunk:
            return
        yield (chunk,)


def _chain_from_iterable_of_lists(iterable):
    for item in iterable:
        item.reverse()
        while item:
            yield item.pop()


def _starmap_chunks(
    submit, function, iterable, timeout, unordered, chunksize, callback
):
    # pylint: disable=too-many-arguments
    function = functools.partial(_apply_chunks, function)
    iterable = _build_chunks(chunksize, iterable)
    result = _starmap_helper(submit, function, iterable, timeout, unordered, callback)
    return _chain_from_iterable_of_lists(result)
