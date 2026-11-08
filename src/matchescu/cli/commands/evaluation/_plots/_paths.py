from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlotDir:
    """File locations of one dataset's plots under a plot root directory.

    The ``graph-`` and ``clusters-`` prefixes let both evaluation commands
    write to the same root without overwriting each other.
    """

    root: Path
    dataset: str

    @property
    def directory(self) -> Path:
        return self.root / self.dataset

    def ensure(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory

    @property
    def graph_layout(self) -> Path:
        return self.directory / "layout-graph.json"

    @property
    def graph_ground_truth(self) -> Path:
        return self.directory / "ground-truth-graph.png"

    def graph_prediction(self, model: str, order: str) -> Path:
        return self.directory / f"graph-{model}-{order}.png"

    @property
    def cluster_layout(self) -> Path:
        return self.directory / "layout-clusters.json"

    @property
    def cluster_ground_truth(self) -> Path:
        return self.directory / "ground-truth-clusters.png"

    def cluster_prediction(self, algorithm: str, matcher: str, order: str) -> Path:
        return self.directory / f"clusters-{algorithm}-{matcher}-{order}.png"
