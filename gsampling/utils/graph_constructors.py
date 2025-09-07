import numpy as np
from escnn.group import dihedral_group, cyclic_group, directsum

from ..core.graphs.factory import GroupGraphFactory
from ..core.subsampling import subsample_with_strategy

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

        # ========================================================================
        # SUBGROUP SIZE CALCULATION USING PROPER SUBSAMPLING STRATEGIES
        # ========================================================================
        # This section uses the proper subsampling strategies to identify the correct
        # subgroup elements based on the mathematical structure of the groups.
        #
        # Mathematical Foundation:
        # - Main group G has |G| = group_size elements
        # - Subgroup H has |H| = subgroup_size elements  
        # - For 3D rotations: Octahedral (24) → C4 (4) for 90° rotations around z-axis
        # - For 2D rotations: Cyclic groups with appropriate order based on application
        #
        # Channel Reduction Formula:
        # Input channels:  C * |G|  (C features × |G| group elements)
        # Output channels: C * |H|  (C features × |H| subgroup elements)
        # Reduction factor: |G| / |H| = k
        
        # Use proper subsampling strategies to get the correct subgroup elements
        try:
            # This will use the registered strategies (e.g., OctahedralToCycleStrategy)
            # which correctly identify C4 subgroups for 90° rotations around z-axis
            subsampled_nodes = subsample_with_strategy(
                group_size=group_size,
                group_type=group_type,
                group_generator=group_generator,
                subgroup_type=subgroup_type,
                subsampling_factor=subsampling_factor,
            )
            self.subgroup_size = len(subsampled_nodes)
        except Exception as e:
            # Fallback to simple division if no strategy is registered
            print(f"Warning: No subsampling strategy found for {group_type}→{subgroup_type}, using simple division: {e}")
            if subgroup_type in ["cycle", "cyclic"]:
                # For cyclic groups, use the correct order based on application
                # For 90° rotations around z-axis, we want C4 (order 4)
                self.subgroup_size = group_size // subsampling_factor
                # Only enforce minimum order for general applications, not specific 3D rotation
                if group_type not in ["octahedral", "full_octahedral"]:
                    self.subgroup_size = max(6, self.subgroup_size)
            else:
                self.subgroup_size = group_size // subsampling_factor
            subsampled_nodes = list(range(self.subgroup_size))

        # Use factory to create original group graph
        self.graph = GroupGraphFactory.create(group_type, self.nodes, group_generator)

        # Use factory to create subgroup graph (enables extension to new groups)
        self.subgroup_graph = GroupGraphFactory.create(self.subgroup_type, subsampled_nodes, group_generator)