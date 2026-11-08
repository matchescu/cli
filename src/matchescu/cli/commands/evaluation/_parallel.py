import logging
import multiprocessing
import os
import threading
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, TypeVar

import click
from rich.progress import Progress, TaskID

TKey = TypeVar("TKey")
TResult = TypeVar("TResult")

_SILENCED_LOGGERS = (
    "accelerate",
    "transformers",
    "transformers.modeling_utils",
    "transformers.modeling_parallel_utils",
    "transformers.distributed.tensor_parallel",
)


def default_worker_count() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def workers_option(**kwargs: Any) -> Callable:
    defaults = {
        "help": "number of worker processes (1 runs in-process)",
        "default": None,
    }
    return click.option(
        "-w", "--workers", type=click.IntRange(min=1), **{**defaults, **kwargs}
    )


_worker_queue: Any = None


class UnitReporter:
    """Per-unit progress handle passed as the last worker argument.

    ``advance()`` posts ``(unit_key, amount)`` to the parent's queue. In
    worker processes the queue comes from the pool initializer (queues may
    only cross the process boundary via inheritance); in serial mode the
    runner attaches the queue directly. No-op when no queue is attached.
    """

    def __init__(self, key: TKey, queue: Any = None) -> None:
        self._key = key
        self._queue = queue

    def advance(self, amount: int = 1) -> None:
        queue = self._queue if self._queue is not None else _worker_queue
        if queue is not None:
            queue.put_nowait((self._key, amount))


def _init_worker(progress_queue: Any) -> None:
    """Silence child-process noise: tqdm bars, transformers/accelerate logs."""
    global _worker_queue
    _worker_queue = progress_queue
    os.environ["TQDM_DISABLE"] = "1"
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    try:
        import transformers.utils.logging as tf_logging

        tf_logging.disable_progress_bar()
        tf_logging.set_verbosity_error()
    except ImportError:
        pass


class UnitRunner:
    """Run work units (optionally in worker processes) with per-unit progress.

    Each unit is a ``(key, args, total)`` triple: ``fn(*args, reporter)`` is
    executed for it, where ``reporter`` is a ``UnitReporter`` bound to the
    unit's key. When a ``Progress`` is given, one task per unit is created
    (total from the unit, indeterminate when ``None``) and advanced from a
    drainer thread as workers call ``reporter.advance()``.

    With ``workers=1`` execution is serial and in-process. Results stream in
    completion order via ``as_completed``. On ``KeyboardInterrupt`` /
    ``GeneratorExit`` / worker errors, pending work is cancelled and all worker
    processes are terminated so no orphans remain.
    """

    def __init__(self, workers: int | None, progress: Progress | None = None) -> None:
        self._worker_count = workers or default_worker_count()
        if progress is None:
            progress = Progress(disable=True)
        self._progress = progress

    def run(
        self,
        units: Iterable[tuple[TKey, tuple[Any, ...], int | None]],
        fn: Callable[..., TResult],
    ) -> Iterator[tuple[TKey, TResult]]:
        items = list(units)
        track = self._progress is not None
        queue: Any = multiprocessing.Queue() if track else None
        stop = threading.Event()
        task_ids: dict[TKey, TaskID] = {}
        if track:
            for key, _, total in items:
                task_ids[key] = self._progress.add_task(str(key), total=total)
        drainer: threading.Thread | None = None
        if track:
            drainer = threading.Thread(
                target=self._drain,
                args=(queue, task_ids, stop),
                daemon=True,
            )
            drainer.start()
        try:
            if self._worker_count == 1:
                for key, args, _ in items:
                    yield key, fn(*args, UnitReporter(key, queue))
            else:
                yield from self._run_pool(items, fn, queue)
        finally:
            if track:
                stop.set()
                assert drainer is not None
                drainer.join(timeout=2)

    def _run_pool(
        self,
        items: list[tuple[TKey, tuple[Any, ...], int | None]],
        fn: Callable[..., TResult],
        queue: Any,
    ) -> Iterator[tuple[TKey, TResult]]:
        pool = ProcessPoolExecutor(
            max_workers=self._worker_count, initializer=_init_worker, initargs=(queue,)
        )
        try:
            futures: dict[Future[TResult], TKey] = {
                pool.submit(fn, *args, UnitReporter(key)): key for key, args, _ in items
            }
            for future in as_completed(futures):
                yield futures[future], future.result()
        except BaseException:
            # KeyboardInterrupt, GeneratorExit (consumer stopped early), worker
            # errors: cancel pending work and kill workers so no orphans remain.
            # Workers already received SIGINT via the foreground process group;
            # terminate() also covers workers stuck in non-interruptible calls.
            pool.shutdown(wait=False, cancel_futures=True)
            children = multiprocessing.active_children()
            for child in children:
                child.terminate()
            for child in children:
                child.join(timeout=5)
            raise
        else:
            pool.shutdown(wait=True)

    def _drain(
        self, queue: Any, task_ids: dict[TKey, TaskID], stop: threading.Event
    ) -> None:
        import queue as queue_module

        while True:
            try:
                key, amount = queue.get(timeout=0.2)
            except queue_module.Empty:
                if stop.is_set():
                    return
                continue
            self._progress.advance(task_ids[key], amount)
