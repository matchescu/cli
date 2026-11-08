from types import SimpleNamespace

import pytest

from matchescu.cli.commands.evaluation._eval_opts import EvalOptions


@pytest.fixture
def eval_obj(tmp_path):
    cfg = SimpleNamespace(
        seed=11,
        benchmark_data=[
            SimpleNamespace(
                dataset=SimpleNamespace(directory="d1"),
                comparison_space=SimpleNamespace(),
            ),
            SimpleNamespace(
                dataset=SimpleNamespace(directory="d2"),
                comparison_space=SimpleNamespace(),
            ),
        ],
        matching=[SimpleNamespace(name="m", type="binary")],
        clustering=SimpleNamespace(algorithms=["WCC"]),
    )
    return EvalOptions(config=cfg, root_dir=tmp_path)
