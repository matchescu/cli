import csv
import json

import pytest
from click.testing import CliRunner

from matchescu.cli import matchescu

USAGE_ERROR = 2
IDS_CSV = """id,name
1,Blue Moon Cafe
2,Blue Moon Cafe Inc
3,Blue Moon Diner
4,Blue Moon Diner Co
5,Red Sun Grill
6,Red Sun Grill House
7,Red Star Grill
8,Red Star Grill Bar
"""
MAPPING_CSV = """1,2
3,4
5,6
7,8
"""


@pytest.fixture
def write_config(tmp_path):
    dataset_dir = tmp_path / "tiny"
    dataset_dir.mkdir()
    (dataset_dir / "ids.csv").write_text(IDS_CSV)
    (dataset_dir / "mapping.csv").write_text(MAPPING_CSV)

    def _write(alphas=(0.65,), **top_level):
        entry = {
            "dataset": {
                "type": "csv",
                "directory": "./tiny",
                "sources": [
                    {
                        "fileName": "ids.csv",
                        "traits": [{"type": "string", "keys": ["name"]}],
                        "idCol": "id",
                    }
                ],
                "pairwiseMapping": {"fileName": "mapping.csv", "hasHeader": False},
            },
            "ambiguityTargets": ["name"],
            "alphas": list(alphas),
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"inputData": [entry], **top_level}))
        return config_path

    return _write


@pytest.fixture
def invoke(tmp_path):
    def _invoke(*args):
        return CliRunner().invoke(matchescu, ["-d", str(tmp_path), "ambiguity", *args])

    return _invoke


REMOVED_SHARED_FLAGS = [
    pytest.param(["--alphas", "0.5"], id="alphas"),
    pytest.param(["--target-floor", "0.5"], id="target-floor"),
    pytest.param(["--target-links", "2"], id="target-links"),
    pytest.param(["--secondary-features", "year"], id="secondary-features"),
    pytest.param(["--surface-baseline"], id="surface-baseline"),
    pytest.param(["--seed", "7"], id="seed"),
    pytest.param(["--split-ratio", "8,1,1"], id="split-ratio"),
]


@pytest.mark.parametrize("flag", REMOVED_SHARED_FLAGS)
def test_generate_rejects_removed_option(write_config, invoke, tmp_path, flag):
    result = invoke("generate", "-c", str(write_config()), "-O", str(tmp_path), *flag)

    assert result.exit_code == USAGE_ERROR


@pytest.mark.parametrize(
    "flag",
    [
        *REMOVED_SHARED_FLAGS,
        pytest.param(["--alpha-criterion", "smallest"], id="alpha-criterion"),
        pytest.param(["--gmm-restarts", "5"], id="gmm-restarts"),
        pytest.param(["--threshold-analysis-step-count", "3"], id="step-count"),
    ],
)
def test_analyse_rejects_removed_option(write_config, invoke, flag):
    result = invoke("analyse", "-c", str(write_config()), *flag)

    assert result.exit_code == USAGE_ERROR


def test_analyse_reports_unmet_criterion_requirement_as_usage_error(
    write_config, invoke
):
    config = write_config(analysis={"selection": {"criterion": "order-headroom"}})

    result = invoke("analyse", "-c", str(config))

    assert result.exit_code == USAGE_ERROR


def test_generate_writes_one_variant_per_configured_alpha(
    write_config, invoke, tmp_path
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    invoke("generate", "-c", str(write_config((0.5, 0.8))), "-O", str(output_dir))

    assert sorted(p.name for p in output_dir.glob("*_dir")) == [
        "amb_tiny_a050_dir",
        "amb_tiny_a080_dir",
    ]


@pytest.mark.parametrize(
    "command, extra_args",
    [
        pytest.param("generate", ("-O", "."), id="generate"),
        pytest.param("analyse", (), id="analyse"),
    ],
)
def test_invalid_config_is_reported_as_usage_error(
    write_config, invoke, command, extra_args
):
    config = write_config(alphas=(1.5,))

    result = invoke(command, "-c", str(config), *extra_args)

    assert "Invalid value for '-c' / '--config-file'" in result.output


@pytest.fixture
def analyse_rows(write_config, invoke, tmp_path):
    def _run(report, **top_level):
        report_dir = tmp_path / "report"
        invoke(
            "analyse",
            "-c",
            str(write_config((0.5, 0.8), **top_level)),
            "--plot-dir",
            str(report_dir),
        )
        with (report_dir / report).open(newline="") as handle:
            return list(csv.DictReader(handle))

    return _run


def test_analyse_reports_configured_alpha_criterion(analyse_rows):
    (row,) = analyse_rows(
        "ambiguity_methodology.csv",
        analysis={"selection": {"criterion": "gmm-balance"}},
    )

    assert row["alpha_criterion"] == "gmm-balance"


def test_analyse_reports_every_configured_criterion(analyse_rows):
    (row,) = analyse_rows(
        "ambiguity_methodology.csv",
        analysis={"selection": {"criterion": ["smallest", "gmm-balance"]}},
    )

    assert row["alpha_criterion"] == "smallest,gmm-balance"
    assert "alpha_star_smallest" in row
    assert "alpha_star_gmm-balance_reason" in row


def test_analyse_writes_one_sweep_csv_with_selected_flag_per_criterion(
    analyse_rows,
):
    rows = analyse_rows(
        "ambiguity_alpha_sweep.csv",
        analysis={"selection": {"criterion": ["smallest", "gmm-balance"]}},
    )

    assert [float(row["alpha"]) for row in rows] == [0.5, 0.8]
    assert "selected_smallest" in rows[0]
    assert "selected_gmm-balance" in rows[0]


def test_analyse_writes_one_alpha_sweep_plot_per_criterion(
    write_config, invoke, tmp_path
):
    report_dir = tmp_path / "report"
    config = write_config(
        (0.5, 0.8),
        analysis={"selection": {"criterion": ["smallest", "gmm-balance"]}},
    )

    result = invoke("analyse", "-c", str(config), "--plot-dir", str(report_dir))

    assert result.exit_code == 0
    assert (report_dir / "alpha_sweep_tiny_smallest.png").exists()
    assert (report_dir / "alpha_sweep_tiny_gmm-balance.png").exists()


def test_analyse_sweeps_configured_threshold_count(analyse_rows):
    rows = analyse_rows(
        "ambiguity_threshold_sweep.csv", analysis={"threshold": {"stepCount": 3}}
    )

    assert len(rows) == 3


def test_analyse_sweeps_theta_plus_minus_five_by_default(
    invoke, tmp_path, write_config
):
    report_dir = tmp_path / "report"
    config = write_config((0.5, 0.8))

    result = invoke("analyse", "-c", str(config), "--plot-dir", str(report_dir))

    assert result.exit_code == 0
    with (report_dir / "ambiguity_threshold_sweep.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    thresholds = [float(row["threshold"]) for row in rows]
    with (report_dir / "ambiguity_methodology.csv").open(newline="") as handle:
        theta = float(next(csv.DictReader(handle))["theta"])

    assert len(thresholds) == 11
    assert thresholds[0] == pytest.approx(theta - 5.0, abs=0.01)
    assert thresholds[-1] == pytest.approx(theta + 5.0, abs=0.01)


def test_analyse_profiles_every_configured_alpha(analyse_rows):
    rows = analyse_rows("ambiguity_alpha_sweep.csv")

    assert [float(row["alpha"]) for row in rows] == [0.5, 0.8]


def test_analyse_accepts_auto_min_ambiguity(write_config, invoke, tmp_path):
    report_dir = tmp_path / "report"
    config = write_config((0.5, 0.8))
    payload = json.loads(config.read_text())
    payload["inputData"][0]["minAmbiguity"] = "auto"
    config.write_text(json.dumps(payload))

    result = invoke("analyse", "-c", str(config), "--plot-dir", str(report_dir))

    assert result.exit_code == 0
    assert (report_dir / "ambiguity_alpha_sweep.csv").exists()


def test_generate_rejects_auto_min_ambiguity(write_config, invoke, tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    entry_config = write_config((0.65,))
    payload = json.loads(entry_config.read_text())
    payload["inputData"][0]["minAmbiguity"] = "auto"
    entry_config.write_text(json.dumps(payload))

    result = invoke("generate", "-c", str(entry_config), "-O", str(output_dir))

    assert result.exit_code == USAGE_ERROR
    assert "auto" in result.output


def test_generate_succeeds_when_simple_pairs_selected(write_config, invoke, tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = invoke(
        "generate",
        "-c",
        str(write_config((0.65,), pairingStrategy="simple-pairs")),
        "-O",
        str(output_dir),
    )

    assert result.exit_code == 0


def test_generate_records_pairing_strategy_in_metadata(write_config, invoke, tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    invoke(
        "generate",
        "-c",
        str(write_config((0.65,), pairingStrategy="simple-pairs")),
        "-O",
        str(output_dir),
    )

    with (output_dir / "metadata.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["pairing_strategy"] for row in rows} == {"simple-pairs"}


def test_generate_uses_per_dataset_pairing_override(write_config, invoke, tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    config = write_config((0.65,), pairingStrategy="greedy-cluster-reps")
    payload = json.loads(config.read_text())
    payload["inputData"][0]["pairing"] = {
        "pairingStrategy": "simple-pairs",
        "pairingScope": "mutual",
    }
    config.write_text(json.dumps(payload))

    result = invoke("generate", "-c", str(config), "-O", str(output_dir))

    assert result.exit_code == 0
    with (output_dir / "metadata.csv").open(newline="") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["pairing_strategy"] == "simple-pairs"
    assert row["pairing_scope"] == "mutual"
    # unset fields fall back to the top-level pairing keys, here the defaults
    assert row["pairing_neighbours"] == "1"


def test_analyse_accepts_per_dataset_pairing_override(write_config, invoke, tmp_path):
    report_dir = tmp_path / "report"
    config = write_config((0.5, 0.8))
    payload = json.loads(config.read_text())
    payload["inputData"][0]["pairing"] = {
        "pairingStrategy": "simple-pairs",
        "pairingNeighbours": 2,
    }
    config.write_text(json.dumps(payload))

    result = invoke("analyse", "-c", str(config), "--plot-dir", str(report_dir))

    assert result.exit_code == 0
    assert (report_dir / "ambiguity_methodology.csv").exists()
