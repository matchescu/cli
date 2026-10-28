from collections.abc import Mapping
from dataclasses import dataclass

from matchescu.typing import EntityReferenceIdentifier as RefId

from ._components import BridgeComponents
from ._footprint import BridgeFootprint
from ._input import ProfileInput
from ._options import ProfileOptions
from ._pair_roles import PairRoles


@dataclass(frozen=True)
class ProfileContext:
    """Shared, computed-once view of a dataset that calculators read from.

    :ivar profile_input: the dataset being profiled.
    :ivar options: settings of the profile.
    :ivar roles: positive pairs grouped by role.
    :ivar footprints: clusters each bridge touches.
    :ivar components: bridge components, largest first.
    """

    profile_input: ProfileInput
    options: ProfileOptions
    roles: PairRoles
    footprints: Mapping[RefId, BridgeFootprint]
    components: BridgeComponents

    @classmethod
    def build(
        cls, profile_input: ProfileInput, options: ProfileOptions
    ) -> "ProfileContext":
        """Derive roles, footprints and components of a dataset once.

        :param profile_input: the dataset to profile.
        :param options: settings of the profile.
        :returns: the context every calculator receives.
        """
        roles = PairRoles.from_input(profile_input)
        footprints = BridgeFootprint.of_bridges(roles, profile_input.cluster_of)
        components = BridgeComponents.from_footprints(
            footprints, profile_input.clusters
        )
        return cls(profile_input, options, roles, footprints, components)
