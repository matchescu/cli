from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

LOWER_QUARTILE: float = 25.0
UPPER_QUARTILE: float = 75.0
POSITIVE: int = 1
NEGATIVE: int = 0

type Statistic = Callable[[Sequence[float]], float]


def _summarise(statistic: Statistic, values: Sequence[float]) -> float | None:
    return float(statistic(values)) if len(values) else None


def _iqr(values: Sequence[float]) -> float:
    lower, upper = np.percentile(values, (LOWER_QUARTILE, UPPER_QUARTILE))
    return upper - lower


def ratio(numerator: float, denominator: float) -> float | None:
    """Return ``numerator / denominator``, or ``None`` for a zero denominator."""
    return numerator / denominator if denominator else None


def avg(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean, or ``None`` for no values."""
    return _summarise(np.mean, values)


def stdev(values: Sequence[float]) -> float | None:
    """Return the population standard deviation, or ``None`` for no values.

    Profiles describe whole datasets rather than samples, hence ddof=0.
    """
    return _summarise(np.std, values)


def median(values: Sequence[float]) -> float | None:
    """Return the median, or ``None`` for no values."""
    return _summarise(np.median, values)


def iqr(values: Sequence[float]) -> float | None:
    """Return the interquartile range (linear interpolation), or ``None``."""
    return _summarise(_iqr, values)


def spread(name: str, values: Sequence[float]) -> dict[str, float | None]:
    """Return ``{name}_avg`` and ``{name}_stdev`` of `values`."""
    values = list(values)
    return {f"{name}_avg": avg(values), f"{name}_stdev": stdev(values)}


def quartiles(name: str, values: Sequence[float]) -> dict[str, float | None]:
    """Return ``{name}_median`` and ``{name}_iqr`` of `values`."""
    values = list(values)
    return {f"{name}_median": median(values), f"{name}_iqr": iqr(values)}


def auc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """Return the ROC AUC of scores ranking `positives` above `negatives`.

    :returns: the probability that a random positive outscores a random
        negative (ties count half), or ``None`` when either side is empty.
    """
    if not positives or not negatives:
        return None
    labels = [POSITIVE] * len(positives) + [NEGATIVE] * len(negatives)
    return float(roc_auc_score(labels, [*positives, *negatives]))


def separability(auc_value: float | None) -> float | None:
    """Return ``|2 * auc - 1|``: ranking strength regardless of direction.

    0 means the scores do not tell the two sides apart; 1 means they do so
    perfectly, whether positives rank above or below negatives.
    """
    return None if auc_value is None else abs(2 * auc_value - 1)


def oriented_auc(auc_value: float | None) -> float | None:
    """Return ``max(auc, 1 - auc)``: AUC read in its stronger direction."""
    return None if auc_value is None else max(auc_value, 1 - auc_value)
