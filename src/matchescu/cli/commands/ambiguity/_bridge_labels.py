from abc import ABC, abstractmethod
from collections.abc import Collection, Hashable
from numbers import Integral
from typing import ClassVar
from uuid import UUID, uuid5


class BridgeLabelStrategy(ABC):
    """Produce the identifier label of each new bridge record.

    Bridge records join an existing dataset, so their labels must not clash
    with the labels already in use and must have a type the rest of the
    dataset can be written and compared with. How to achieve that depends on
    the label type, hence one strategy per type. Every strategy is built with
    the same arguments so that :class:`BridgeLabels` can create any of them.
    """

    @abstractmethod
    def __call__(self, key: Hashable) -> Hashable:
        """Return the label of one bridge record.

        Called exactly once per bridge. The label is shared across alpha
        variants.

        :param key: identity of the bridge; strategies derive it from the
            clusters or records the bridge links.
        :returns: a label not used by any existing reference or earlier
            bridge.
        """


class IntegerLabels(BridgeLabelStrategy):
    """Number bridges consecutively after the largest existing label.

    Labels are handed out in call order. Only clusters that actually contain
    bridges increment the counter.

    :param existing_labels: integer labels of the existing references; an
        empty collection starts numbering at 1.
    """

    def __init__(self, existing_labels: Collection[Hashable]):
        self.__last = max((int(label) for label in existing_labels), default=0)

    def __call__(self, key: Hashable) -> int:
        """Return the next unused integer label.

        :param key: ignored; the label depends on call order only.
        :returns: one more than the previously returned (or largest existing)
            label.
        """
        self.__last += 1
        return self.__last


class Uuid5Labels(BridgeLabelStrategy):
    """Derive a reproducible UUID label from the dataset and home cluster.

    A UUID version 5 is a hash of a namespace and a name, so the same dataset
    and home cluster always yield the same label: regenerating a dataset
    reproduces its bridge ids. Including the dataset name keeps labels
    distinct when datasets are combined.

    .. note::
        Home cluster ids are assigned in a deterministic order by the
        ambiguity generator, which is what makes the labels stable across
        runs. Without a dataset name the labels are still unique within one
        dataset, but not across datasets.

    :param dataset_name: name hashed into every label.
    """

    NAMESPACE: ClassVar[UUID] = UUID("6f1c4d6e-2b7a-4f0e-9a51-3c8d2e7b9a40")
    SEPARATOR: ClassVar[str] = "\x1f"

    def __init__(self, dataset_name: str):
        self.__dataset_name = dataset_name

    def __call__(self, key: Hashable) -> str:
        """Return the UUID label of one bridge.

        :param key: identity of the bridge; hashed into the label.
        :returns: the canonical string form of the version 5 UUID.
        """
        name = f"{self.__dataset_name}{self.SEPARATOR}{key}"
        return str(uuid5(self.NAMESPACE, name))


class BridgeLabels:
    """Choose the bridge label strategy that matches a dataset's labels.

    Datasets carry labels of arbitrary hashable types. The registry maps a
    label type to the strategy able to extend labels of that type; the first
    entry whose type matches every existing label wins. Supporting a new
    label type means adding one registry entry.
    """

    _SUPPORTED_TYPES: ClassVar[set[type]] = {Integral, str}

    @classmethod
    def labelling_strategy(
        cls, labels: Collection[Hashable], dataset_name: str
    ) -> BridgeLabelStrategy:
        """Build the label strategy for a dataset's existing labels.

        :param labels: labels of every reference in the dataset; an empty
            collection matches the first registry entry.
        :param dataset_name: name of the dataset receiving the bridges.
        :returns: a fresh strategy, ready to label the dataset's bridges.
        :raises TypeError: when the labels mix types or no registry entry
            matches them; the message lists the label types found.
        """
        supported_types = {
            supported_type
            for supported_type in cls._SUPPORTED_TYPES
            if any(isinstance(label, supported_type) for label in labels)
        }
        unsupported_types = {
            type(label).__name__
            for label in labels
            if not any(isinstance(label, t) for t in cls._SUPPORTED_TYPES)
        }
        if unsupported_types:
            raise TypeError(f"unsupported label types: {sorted(unsupported_types)}")
        if len(supported_types) > 1:
            raise TypeError(
                f"mixed label types: {sorted(x.__name__ for x in supported_types)}"
            )

        if len(supported_types) == 0:
            return IntegerLabels(existing_labels=labels)
        label_type = supported_types.pop()
        if issubclass(label_type, Integral):
            return IntegerLabels(existing_labels=labels)
        if issubclass(label_type, str):
            return Uuid5Labels(dataset_name=dataset_name)

        # unreachable code
        raise TypeError(
            f"no bridge label strategy for labels of type {label_type.__name__}"
        )
