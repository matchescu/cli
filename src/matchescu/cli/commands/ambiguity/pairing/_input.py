from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

from matchescu.reference_store.id_table import IdTable
from matchescu.typing import EntityReference
from matchescu.typing import EntityReferenceIdentifier as RefId


@dataclass(frozen=True)
class PairingInput:
    """Everything a pairing strategy requires to function.

    The ambiguity generator owns the ambiguity score and the cluster layout. These
     are used in a pairing strategy to create bridge seeds. This de-couples the
     logic of ambiguity scoring from the logic of record selection.

    :ivar clusters: cluster id to member ids, ids assigned by the generator.
    :ivar id_table: references of the input dataset, bridges excluded.
    :ivar score: ambiguity score of two reference ids, home side first.
    :ivar score_refs: ambiguity score of two references, home side first.
    :ivar target_fields: properties permitted to receive bridged values.
    :ivar target_links: links each bridge gets inside its target cluster.
    :ivar on_progress: called once per unit of work with a description.
    """

    clusters: Mapping[int, Collection[RefId]]
    id_table: IdTable
    score: Callable[[RefId, RefId], float]
    score_refs: Callable[[EntityReference, EntityReference], float]
    target_fields: tuple[str, ...]
    target_links: int
    on_progress: Callable[[str], None]
