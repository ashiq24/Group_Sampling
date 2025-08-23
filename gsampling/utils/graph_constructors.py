import numpy as np
from escnn.group import dihedral_group, cyclic_group, directsum

from ..core.graphs.factory import GroupGraphFactory

# Import the new graph implementations from core/graphs/
from ..core.graphs.dihedral import DihedralGraph
from ..core.graphs.cyclic import CycleGraph


class GraphConstructor:
    """Factory for creating graph structures for different group types."""

    def __init__(
        self,
        group_size: int,
        group_type: str,
        group_generator: str,
        subgroup_type: str,
        subsampling_factor: int,
    ):
        """
        Initialize graph constructor.

        Args:
            group_size: Size of the original group
            group_type: Type of the original group (e.g., 'dihedral', 'cyclic')
            group_generator: Generator for the group
            subgroup_type: Type of the subgroup
            subsampling_factor: Factor by which to subsample
        """
        self.group_size = group_size
        self.group_type = group_type
        self.group_generator = group_generator
        self.subgroup_type = subgroup_type
        self.subsampling_factor = subsampling_factor

        # Create nodes for the original group
        self.nodes = list(range(group_size))

        # Calculate subgroup size
        self.subgroup_size = group_size // subsampling_factor
        subsampled_nodes = list(range(self.subgroup_size))

        # Use factory to create original group graph
        self.graph = GroupGraphFactory.create(group_type, self.nodes, group_generator)

        # Use factory to create subgroup graph (enables extension to new groups)
        self.subgroup_graph = GroupGraphFactory.create(self.subgroup_type, subsampled_nodes, group_generator)