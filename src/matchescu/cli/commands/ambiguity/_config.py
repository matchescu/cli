from typing import Annotated, ClassVar, Self

from matchescu.matching.config import AnyDatasetConfig, ConfigModel
from pydantic import Field, model_validator

from .profiling import ProfileOptions, SplitRatio
from .selection import SelectionSettings, SeparabilityBounds

type Alpha = Annotated[float, Field(gt=0, lt=1)]


class ProfilingConfig(ConfigModel):
    """Settings of every dataset profile, shared by all datasets.

    Values are validated by building the profile options once, so a bad
    configuration fails while it loads.
    """

    negative_similarity: bool = False
    surface_baseline: bool = False
    synthetic_detectability: bool = False
    seed: int = ProfileOptions.DEFAULT_SEED
    cv_folds: int = ProfileOptions.DEFAULT_CV_FOLDS
    neg_pos_ratio: float = ProfileOptions.DEFAULT_NEG_POS_RATIO
    split_ratio: tuple[float, ...] = SplitRatio.default().parts

    @model_validator(mode="after")
    def _check(self) -> Self:
        self.to_profile_options()
        return self

    def to_profile_options(self) -> ProfileOptions:
        """Return the profile options these settings describe."""
        return ProfileOptions(
            split_ratio=SplitRatio(self.split_ratio),
            negative_similarity=self.negative_similarity,
            neg_pos_ratio=self.neg_pos_ratio,
            surface_baseline=self.surface_baseline,
            synthetic_detectability=self.synthetic_detectability,
            seed=self.seed,
            cv_folds=self.cv_folds,
        )


class SelectionConfig(ConfigModel):
    """How ``analyse`` chooses an alpha, overridable per dataset.

    Values are validated by building the selection settings once, so a bad
    configuration fails while it loads.
    """

    criterion: str = SelectionSettings.DEFAULT_CRITERION
    min_separability: float = SeparabilityBounds.DEFAULT_MIN_SEPARABILITY
    max_auc: float = SeparabilityBounds.DEFAULT_MAX_AUC
    target_resolvable: float = SelectionSettings.DEFAULT_TARGET_RESOLVABLE
    max_detect_auc: float = SelectionSettings.DEFAULT_MAX_DETECT_AUC

    @model_validator(mode="after")
    def _check(self) -> Self:
        self.to_selection_settings()
        return self

    def to_selection_settings(self) -> SelectionSettings:
        """Return the selection settings these values describe."""
        return SelectionSettings(
            SeparabilityBounds(self.min_separability, self.max_auc),
            self.criterion,
            self.target_resolvable,
            self.max_detect_auc,
        )


class ThresholdAnalysisConfig(ConfigModel):
    """The minimum-ambiguity sweep and mixture fit of ``analyse``.

    :ivar start_value: first threshold of the sweep.
    :ivar step_count: number of thresholds in the sweep, at least 1.
    :ivar step_size: distance between consecutive thresholds, at least 0.1.
    :ivar gmm_restarts: random restarts of the mixture fit, at least 1.
    """

    MIN_STEP_SIZE: ClassVar[float] = 0.1
    DEFAULT_START_VALUE: ClassVar[float] = 0.0
    DEFAULT_STEP_COUNT: ClassVar[int] = 10
    DEFAULT_STEP_SIZE: ClassVar[float] = 1.0
    DEFAULT_GMM_RESTARTS: ClassVar[int] = 50

    start_value: float = DEFAULT_START_VALUE
    step_count: int = Field(default=DEFAULT_STEP_COUNT, ge=1)
    step_size: float = Field(default=DEFAULT_STEP_SIZE, ge=MIN_STEP_SIZE)
    gmm_restarts: int = Field(default=DEFAULT_GMM_RESTARTS, ge=1)


class AnalysisConfig(ConfigModel):
    """Settings only ``analyse`` reads.

    Top-level values are the default for every dataset; an entry may replace
    the whole block.

    :ivar selection: how an alpha is chosen from the sweep.
    :ivar threshold: the minimum-ambiguity sweep and mixture fit.
    """

    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    threshold: ThresholdAnalysisConfig = Field(default_factory=ThresholdAnalysisConfig)


class AmbiguityDataConfig(ConfigModel):
    """Settings of one input dataset.

    :ivar dataset: where the dataset lives and how its files are laid out.
    :ivar ambiguity_targets: properties that receive synthesised values.
    :ivar secondary_features: properties copied from the home
        representative to every bridge.
    :ivar min_ambiguity: minimum pairing score of a bridge; ``None``
        disables the filter.
    :ivar alphas: blend weights, each in (0, 1). ``analyse`` searches them;
        ``generate`` writes one variant per weight.
    :ivar target_floor: minimum target similarity, as a fraction of the
        symmetric pick's target similarity, in [0, 1].
    :ivar target_links: links each bridge gets inside its target cluster.
    :ivar max_length_ratio: highest bridge-value length, in tokens, as a
        multiple of the mean length of the two blended values; ``None``
        disables the limit.
    :ivar analysis: analysis settings of this dataset; ``None`` falls back to
        the top-level block.
    """

    dataset: AnyDatasetConfig
    ambiguity_targets: list[str] | None = None
    secondary_features: list[str] | None = None
    min_ambiguity: float | None = None
    alphas: list[Alpha] = Field(min_length=1)
    target_floor: float = Field(default=0.0, ge=0, le=1)
    target_links: int = Field(default=1, ge=1)
    max_length_ratio: float | None = Field(default=None, ge=1)
    analysis: AnalysisConfig | None = None


class AmbiguityConfig(ConfigModel):
    """Configuration file of the ``ambiguity`` commands.

    :ivar profiling: profile settings shared by every dataset.
    :ivar analysis: default analysis settings; an entry may replace them.
    :ivar input_data: one entry per input dataset.
    """

    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    input_data: list[AmbiguityDataConfig]

    def analysis_for(self, entry: AmbiguityDataConfig) -> AnalysisConfig:
        """Return the analysis settings of one dataset.

        :param entry: input dataset whose settings are resolved.
        :returns: the entry's own block when present, the top-level block
            otherwise.
        """
        return entry.analysis or self.analysis
