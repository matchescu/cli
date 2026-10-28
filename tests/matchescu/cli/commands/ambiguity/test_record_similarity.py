import pytest

from matchescu.cli.commands.ambiguity._record_similarity import RecordSimilarity
from matchescu.cli.commands.ambiguity._similarity import weighted_similarity


def test_record_similarity_averages_weighted_similarity_over_fields():
    sim = RecordSimilarity(["name", "city"])

    actual = sim({"name": "abc", "city": "x y"}, {"name": "abc", "city": "z"})

    assert actual == pytest.approx((1.0 + weighted_similarity("x y", "z")) / 2)


def test_record_similarity_ignores_fields_outside_its_selection():
    sim = RecordSimilarity(["name"])

    actual = sim({"name": "abc", "city": "x"}, {"name": "abc", "city": "y"})

    assert actual == pytest.approx(1.0)


@pytest.mark.parametrize(
    "left",
    [
        pytest.param({}, id="missing"),
        pytest.param({"name": None}, id="none"),
    ],
)
def test_record_similarity_treats_absent_values_as_empty(left):
    sim = RecordSimilarity(["name"])

    assert sim(left, {"name": ""}) == pytest.approx(weighted_similarity("", ""))


def test_record_similarity_without_fields_raises():
    with pytest.raises(ValueError, match="field"):
        RecordSimilarity([])
