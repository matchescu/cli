from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationSettings:
    """Effective parameters of one ambiguity generator.

    Dataset configuration and CLI options are merged before a generator is
    built, so the generator is the only place that knows the values it
    actually used; reports read them from here.

    :ivar dataset_name: name of the source dataset.
    :ivar ambiguity_targets: properties that receive synthesised values, in
        configuration order.
    :ivar secondary_features: properties copied from the home
        representative, in configuration order.
    :ivar min_ambiguity: minimum pairing score; ``None`` disables the filter.
    :ivar alphas: blend weights, ascending and without duplicates.
    :ivar target_floor: minimum target similarity, as a fraction of the
        symmetric pick's target similarity.
    :ivar target_links: links each bridge gets inside its target cluster.
    :ivar max_length_ratio: highest bridge-value length, in tokens, as a
        multiple of the mean length of the two blended values; ``None``
        disables the limit.
    """

    dataset_name: str
    ambiguity_targets: tuple[str, ...]
    secondary_features: tuple[str, ...]
    min_ambiguity: float | None
    alphas: tuple[float, ...]
    target_floor: float
    target_links: int
    max_length_ratio: float | None = None
