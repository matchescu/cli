import os
import time

import click
from click.testing import CliRunner

from matchescu.cli.commands.evaluation import _parallel
from matchescu.cli.commands.evaluation._cmd_group import evaluate


def _square(x, reporter):
    return x * x


def test_default_worker_count_is_cpu_count_minus_one(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert _parallel.default_worker_count() == 7


def test_default_worker_count_is_at_least_one(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _parallel.default_worker_count() == 1


def test_runner_serial_preserves_keys_and_results():
    units = [("a", (2,), 1), ("b", (3,), 1), ("c", (5,), 1)]
    results = list(_parallel.UnitRunner(1).run(units, _square))
    assert results == [("a", 4), ("b", 9), ("c", 25)]


def test_runner_multiprocess_preserves_keys_and_results():
    units = [("a", (2,), 1), ("b", (3,), 1), ("c", (5,), 1), ("d", (7,), 1)]
    results = list(_parallel.UnitRunner(2).run(units, _square))
    assert dict(results) == {"a": 4, "b": 9, "c": 25, "d": 49}


def test_runner_streams_in_completion_order():
    units = [("slow", (4,), 4), ("fast", (0,), 1)]
    started = time.monotonic()
    streamed = []
    for key, _ in _parallel.UnitRunner(2).run(units, _sleep_square):
        streamed.append((key, time.monotonic() - started))
    assert [key for key, _ in streamed] == ["fast", "slow"]
    assert streamed[0][1] < streamed[1][1] - 1.0


def test_runner_empty():
    assert list(_parallel.UnitRunner(2).run([], _square)) == []


def _sleep_square(seconds, reporter):
    import time

    time.sleep(seconds)
    return seconds * seconds


def _make_ctx_obj():
    return {"verbose": False, "root_dir": None, "config": None}


def test_every_evaluate_subcommand_accepts_workers():
    runner = CliRunner()
    for name, command in evaluate.commands.items():
        result = runner.invoke(
            command, ["--help"], obj=_make_ctx_obj(), catch_exceptions=False
        )
        assert result.exit_code == 0, result.output
        assert "--workers" in result.output, f"{name} missing --workers"


def test_workers_option_rejects_zero():
    @click.command()
    @_parallel.workers_option()
    def cmd(workers):
        pass

    runner = CliRunner()
    result = runner.invoke(cmd, ["-w", "0"])
    assert result.exit_code != 0


def test_init_worker_silences_loggers_and_tqdm(monkeypatch):
    import logging
    import queue as queue_module

    monkeypatch.delenv("TQDM_DISABLE", raising=False)
    loggers = {name: logging.getLogger(name) for name in _parallel._SILENCED_LOGGERS}
    for logger in loggers.values():
        monkeypatch.setattr(logger, "level", logging.WARNING)

    _parallel._init_worker(queue_module.Queue())

    assert os.environ["TQDM_DISABLE"] == "1"
    for logger in loggers.values():
        assert logger.level == logging.ERROR


def _reporting_worker(n, reporter):
    for _ in range(n):
        reporter.advance(1)
    return n


def test_reporter_noop_without_queue():
    _parallel.UnitReporter("k", queue=None).advance(5)


def test_runner_creates_one_task_per_unit_serial():
    from rich.progress import Progress

    with Progress(disable=True) as progress:
        runner = _parallel.UnitRunner(1, progress)
        results = []
        for key, value in runner.run(
            [("a", (3,), 3), ("b", (3,), 3)], _reporting_worker
        ):
            results.append((key, value))
        time.sleep(0.5)
        completed = {t.description: t.completed for t in progress.tasks}
        assert completed == {"a": 3, "b": 3}
    assert dict(results) == {"a": 3, "b": 3}


def test_runner_creates_one_task_per_unit_multiprocess():
    from rich.progress import Progress

    with Progress(disable=True) as progress:
        runner = _parallel.UnitRunner(2, progress)
        results = []
        for key, value in runner.run(
            [("a", (4,), 4), ("b", (6,), 6)], _reporting_worker
        ):
            results.append((key, value))
        time.sleep(0.5)
        completed = {t.description: t.completed for t in progress.tasks}
        assert completed == {"a": 4, "b": 6}
    assert dict(results) == {"a": 4, "b": 6}


def test_count_comparison_space_rows(tmp_path):
    from matchescu.cli.config._config import ComparisonSpaceConfig
    from matchescu.cli.data import count_comparison_space_rows

    (tmp_path / "cs.csv").write_text("a,b\n1,2\n3,4\n5,6\n")
    assert (
        count_comparison_space_rows(tmp_path, ComparisonSpaceConfig(file_name="cs.csv"))
        == 3
    )
    (tmp_path / "noheader.csv").write_text("1,2\n3,4\n")
    assert (
        count_comparison_space_rows(
            tmp_path, ComparisonSpaceConfig(file_name="noheader.csv", has_header=False)
        )
        == 2
    )
    assert (
        count_comparison_space_rows(
            tmp_path, ComparisonSpaceConfig(file_name="missing.csv")
        )
        is None
    )


def test_runner_close_kills_workers():
    import multiprocessing

    gen = _parallel.UnitRunner(2).run(
        [("a", (10,), 10), ("b", (10,), 10), ("c", (10,), 10)], _sleep_square
    )
    next(gen)
    gen.close()  # type: ignore[attr-defined]
    assert multiprocessing.active_children() == []


_SIGINT_SCRIPT = """
import sys, time
from matchescu.cli.commands.evaluation._parallel import UnitRunner

def slow(x, reporter):
    time.sleep(x)
    return x

if __name__ == "__main__":
    try:
        for _ in UnitRunner(3).run([(str(i), (30,), 1) for i in range(6)], slow):
            pass
    except KeyboardInterrupt:
        print("parent caught KeyboardInterrupt", flush=True)
        raise
    sys.exit(0)
"""


def test_sigint_kills_workers_no_orphans(tmp_path):
    import signal
    import subprocess
    import sys

    import psutil

    script = tmp_path / "sigint_child.py"
    script.write_text(_SIGINT_SCRIPT)
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    time.sleep(8)  # let workers spawn and start sleeping
    pgid = proc.pid
    os.killpg(pgid, signal.SIGINT)
    try:
        out, err = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()
        raise AssertionError(f"child hung after SIGINT; stderr:\n{err}")
    assert proc.returncode != 0
    assert "parent caught KeyboardInterrupt" in out

    leftovers = [
        p.info["pid"]
        for p in psutil.process_iter(["pid", "ppid"])
        if p.info["ppid"] == pgid
    ]
    assert leftovers == [], f"orphaned workers after SIGINT: {leftovers}"
