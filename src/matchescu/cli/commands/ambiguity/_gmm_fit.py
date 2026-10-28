from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.mixture import GaussianMixture


class InsufficientScoresError(ValueError):
    """Signal that a score population is too small for a two-component fit.

    A two-component mixture needs at least as many scores as components.
    Datasets with very few clusters produce fewer pairing scores than that,
    so this is an expected outcome of analysing real data rather than a
    programming error. A dedicated type lets callers skip such datasets
    while every other failure still propagates.
    """


@dataclass
class GmmFit:
    """Result of fitting a two-component Gaussian mixture to ambiguity scores.

    Best-pairing ambiguity scores are modelled as a mixture of two
    populations: *genuine* overlap (the high-mean component, pairs whose
    representatives meaningfully resemble each other, so a bridge record is
    plausibly extractable) and *superficial* overlap (the low-mean component,
    pairs whose resemblance is only on the surface).

    :meth:`from_scores` is the intended constructor; it derives `threshold`
    as the Bayes-optimal decision boundary between the two components.

    :ivar threshold: exact, unrounded score at which the posterior probability
        of the genuine-overlap component equals 0.5, or the midpoint of the
        two component means when the components are not separable.
    :ivar mu_genuine: mean of the genuine-overlap component.
    :ivar sigma_genuine: standard deviation of the genuine-overlap component.
    :ivar pi_genuine: mixture weight of the genuine-overlap component.
    :ivar mu_superficial: mean of the superficial-overlap component.
    :ivar sigma_superficial: standard deviation of the superficial-overlap
        component.
    :ivar pi_superficial: mixture weight of the superficial-overlap component.
    :ivar boundary_type: :attr:`BAYES_OPTIMAL` when the decision boundary was
        solved exactly, :attr:`MEAN_MIDPOINT` for the degenerate fallback;
        distinguishes the two cases so degenerate fits are reportable rather
        than silent.

    .. note::
        The threshold is deliberately unrounded: presentation precision is a
        caller decision.
    """

    BAYES_OPTIMAL: ClassVar[str] = "bayes_optimal"
    MEAN_MIDPOINT: ClassVar[str] = "mean_midpoint"
    COMPONENT_COUNT: ClassVar[int] = 2

    threshold: float
    mu_genuine: float
    sigma_genuine: float
    pi_genuine: float
    mu_superficial: float
    sigma_superficial: float
    pi_superficial: float
    boundary_type: str

    @classmethod
    def from_scores(cls, scores: list[float], n_init: int = 50) -> "GmmFit":
        """Fit a two-component Gaussian mixture and derive the ambiguity threshold.

        The threshold is the exact score at which the posterior probability of
        the genuine-overlap component equals 0.5, obtained by root-finding
        `posterior_diff` between the two component means. When the posterior
        never crosses 0.5 between the means (inseparable components), the
        threshold falls back to the midpoint of the two means and the fit gets
        tagged :attr:`MEAN_MIDPOINT`, so the degenerate fit stays reportable
        instead of pretending precision.

        :param scores: best-pairing ambiguity scores of one dataset; each is
            treated as an independent draw from the mixture.
        :param n_init: number of random restarts for the EM algorithm, used to
            escape poor local optima; ``GaussianMixture`` keeps the run with
            the highest log-likelihood automatically. Defaults to 50.
        :returns: the resolved fit, with `threshold` and `boundary_type`
            populated and every component parameter unrounded.
        :raises InsufficientScoresError: when there are fewer scores than
            mixture components; the check runs before fitting, so it replaces
            the opaque error ``GaussianMixture`` would raise.
        """
        if len(scores) < cls.COMPONENT_COUNT:
            raise InsufficientScoresError(
                f"{len(scores)} pairing score(s) found, "
                f"at least {cls.COMPONENT_COUNT} are needed"
            )
        X = np.array(scores).reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=cls.COMPONENT_COUNT,
            covariance_type="full",
            n_init=n_init,
            random_state=0,
        )
        gmm.fit(X)

        means = gmm.means_.flatten()
        high_idx = int(np.argmax(means))
        low_idx = 1 - high_idx

        fit = cls(
            threshold=0.0,
            mu_genuine=float(means[high_idx]),
            sigma_genuine=float(np.sqrt(gmm.covariances_[high_idx, 0, 0])),
            pi_genuine=float(gmm.weights_[high_idx]),
            mu_superficial=float(means[low_idx]),
            sigma_superficial=float(np.sqrt(gmm.covariances_[low_idx, 0, 0])),
            pi_superficial=float(gmm.weights_[low_idx]),
            boundary_type=cls.BAYES_OPTIMAL,
        )
        lo = min(fit.mu_superficial, fit.mu_genuine)
        hi = max(fit.mu_superficial, fit.mu_genuine)
        try:
            fit.threshold = float(brentq(fit.posterior_diff, lo, hi))
        except ValueError:
            fit.threshold = (fit.mu_superficial + fit.mu_genuine) / 2.0
            fit.boundary_type = cls.MEAN_MIDPOINT
        return fit

    def genuine_density(self, x):
        """Return the weighted normal density of the genuine-overlap component.

        :param x: evaluation point(s); arrays are evaluated pointwise, which
            lets callers plot smooth density curves.
        :returns: mixture weight times normal density at ``x`` for the
            genuine-overlap component.
        """
        return self.pi_genuine * norm.pdf(x, self.mu_genuine, self.sigma_genuine)

    def superficial_density(self, x):
        """Return the weighted normal density of the superficial-overlap component.

        :param x: evaluation point(s); arrays are evaluated pointwise, which
            lets callers plot smooth density curves.
        :returns: mixture weight times normal density at ``x`` for the
            superficial-overlap component.
        """
        return self.pi_superficial * norm.pdf(
            x, self.mu_superficial, self.sigma_superficial
        )

    def posterior_diff(self, x: float) -> float:
        """Return the genuine-component posterior probability at ``x`` minus 0.5.

        The sign of the result is the Bayes decision at ``x``: non-negative
        means ``x`` is likelier to have come from the genuine-overlap
        component. Root-finding this between the component means yields the
        `threshold`; when the components are not separable no root exists and
        callers must fall back to :attr:`MEAN_MIDPOINT`.

        .. note::
            Degenerate fits can make both component densities evaluate to 0
            at some ``x``; in that case the result is ``-0.5`` (definitively
            superficial) instead of undefined, keeping the posterior a
            well-defined real function.

        :param x: score at which the Bayes decision is evaluated.
        :returns: posterior probability of the genuine-overlap component at
            ``x`` minus 0.5, in [-0.5, 0.5].
        """
        genuine = self.genuine_density(x)
        total = genuine + self.superficial_density(x)
        if total == 0:
            return -0.5
        return genuine / total - 0.5

    def bridge_count(self, scores: list[float]) -> int:
        """Return how many of the given scores qualify as bridge scores.

        The ``>=`` comparison mirrors the threshold sweep: a score exactly at
        the threshold still yields a bridge.

        :param scores: ambiguity scores to classify against `threshold`.
        :returns: number of qualifying scores; 0 when `threshold` exceeds all.
        """
        return sum(1 for score in scores if score >= self.threshold)
