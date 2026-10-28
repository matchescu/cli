from collections.abc import Hashable, Iterable


class DisjointSets:
    """Union-find over hashable items, with path compression.

    Items join the structure the first time they are seen.
    """

    def __init__(self) -> None:
        self.__parent: dict[Hashable, Hashable] = {}

    def find(self, item: Hashable) -> Hashable:
        """Return the representative of the set holding `item`."""
        parent = self.__parent.setdefault(item, item)
        if parent == item:
            return item
        root = self.find(parent)
        self.__parent[item] = root
        return root

    def union(self, items: Iterable[Hashable]) -> None:
        """Merge the sets holding every item of `items` into one."""
        roots = [self.find(item) for item in items]
        for root in roots[1:]:
            self.__parent[self.find(root)] = self.find(roots[0])
