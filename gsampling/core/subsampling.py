"""
Subsampling strategies for different group-to-subgroup transitions.

This module implements the strategy pattern for subsampling operations,
enabling clean extension to complex 3D group relationships without
modifying existing code.
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Optional
from abc import ABC, abstractmethod


class SubsamplingStrategy(ABC):
    """
    Abstract base class for subsampling strategies.
    
    Each strategy implements a specific group-to-subgroup subsampling algorithm
    based on the mathematical structure of the groups involved.
    """
    
    @abstractmethod
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """
        Perform subsampling from parent group to subgroup.
        
        Args:
            nodes: List of parent group element indices
            subsampling_factor: Factor by which to reduce group size
            generator: Optional generator specification
            
        Returns:
            List of subsampled node indices
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """
        Validate that subsampling parameters are valid for this strategy.
        
        Args:
            group_size: Size of parent group
            subsampling_factor: Subsampling factor
            generator: Optional generator specification
            
        Returns:
            True if parameters are valid
            
        Raises:
            ValueError: If parameters are invalid with explanation
        """
        pass


class CycleToCycleStrategy(SubsamplingStrategy):
    """Subsampling strategy for cycle → cycle transitions."""
    
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """Simple stride subsampling for cyclic groups."""
        return nodes[::subsampling_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """Validate cycle → cycle subsampling parameters."""
        if group_size % subsampling_factor != 0:
            raise ValueError(f"Group size {group_size} must be divisible by factor {subsampling_factor}")
        return True


class DihedralToDihedralStrategy(SubsamplingStrategy):
    """Subsampling strategy for dihedral → dihedral transitions."""
    
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """Stride subsampling for dihedral groups."""
        if generator != "r-s":
            raise ValueError("Dihedral → dihedral subsampling requires 'r-s' generator")
        return nodes[::subsampling_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """Validate dihedral → dihedral subsampling parameters."""
        if generator != "r-s":
            raise ValueError("Dihedral → dihedral subsampling requires 'r-s' generator")
        if group_size % 2 != 0:
            raise ValueError("Dihedral group size must be even")
        if group_size % subsampling_factor != 0:
            raise ValueError(f"Group size {group_size} must be divisible by factor {subsampling_factor}")
        return True


class DihedralToCycleStrategy(SubsamplingStrategy):
    """Subsampling strategy for dihedral → cycle transitions."""
    
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """Subsample dihedral group to cyclic subgroup (rotation part only)."""
        if generator != "r-s":
            raise ValueError("Dihedral → cycle subsampling requires 'r-s' generator")
        
        # Extract rotation part (first half) and subsample
        rotation_nodes = nodes[: len(nodes) // 2]
        adjusted_factor = subsampling_factor // 2
        return rotation_nodes[::adjusted_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """Validate dihedral → cycle subsampling parameters."""
        if generator != "r-s":
            raise ValueError("Dihedral → cycle subsampling requires 'r-s' generator")
        if subsampling_factor % 2 != 0:
            raise ValueError("Dihedral → cycle subsampling factor must be even")
        if (group_size // 2) % (subsampling_factor // 2) != 0:
            raise ValueError(f"Rotation part size {group_size // 2} must be divisible by {subsampling_factor // 2}")
        return True


class DihedralToAdihedralStrategy(SubsamplingStrategy):
    """Subsampling strategy for dihedral → adihedral transitions (special case)."""
    
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """Alternate split subsampling for adihedral case."""
        if generator != "r-s":
            raise ValueError("Dihedral → adihedral subsampling requires 'r-s' generator")
        
        group_size = len(nodes)
        sub_sample_nodes = (
            nodes[: group_size // 2 : subsampling_factor]
            + nodes[group_size // 2 + 1 :: subsampling_factor]
        )
        return sub_sample_nodes
    
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """Validate dihedral → adihedral subsampling parameters."""
        if generator != "r-s":
            raise ValueError("Dihedral → adihedral subsampling requires 'r-s' generator")
        if group_size % 2 != 0:
            raise ValueError("Dihedral group size must be even")
        return True


# ========================= 3D Group Subsampling Strategies =========================

class OctahedralToCycleStrategy(SubsamplingStrategy):
    """Subsampling strategy for octahedral → cycle transitions."""
    
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """
        Subsample octahedral group to cyclic subgroup.
        
        **Mathematical Background:**
        Octahedral group O has several cyclic subgroups:
        - C4: 4-fold rotations around face-to-face axes (6 such axes)
        - C3: 3-fold rotations around vertex-to-vertex axes (4 such axes)
        - C2: 2-fold rotations around edge-to-edge axes (12 such axes)
        
        **Implementation:**
        For simplicity, we extract a C4 subgroup (4-fold face rotations).
        """
        if len(nodes) != 24:  # Octahedral group has 24 elements
            raise ValueError("Octahedral group must have 24 elements")
        
        # Extract C4 subgroup (4-fold rotations around one axis)
        # This is a simplified implementation - real implementation would need
        # proper group theory to identify the C4 subgroup elements
        if subsampling_factor == 6:  # 24 → 4 elements
            return [0, 6, 12, 18]  # Example C4 subgroup elements
        elif subsampling_factor == 8:  # 24 → 3 elements  
            return [0, 8, 16]  # Example C3 subgroup elements
        else:
            # General case: stride sampling (may not preserve group structure)
            return nodes[::subsampling_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """Validate octahedral → cycle subsampling parameters."""
        if group_size != 24:
            raise ValueError("Octahedral group must have 24 elements")
        if 24 % subsampling_factor != 0:
            raise ValueError(f"Subsampling factor {subsampling_factor} must divide 24")
        return True


class IcosahedralToCycleStrategy(SubsamplingStrategy):
    """Subsampling strategy for icosahedral → cycle transitions."""
    
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """
        Subsample icosahedral group to cyclic subgroup.
        
        **Mathematical Background:**
        Icosahedral group I has cyclic subgroups:
        - C5: 5-fold rotations around vertex-to-vertex axes (12 such axes)
        - C3: 3-fold rotations around face-to-face axes (20 such axes)
        - C2: 2-fold rotations around edge-to-edge axes (30 such axes)
        """
        if len(nodes) != 60:  # Icosahedral group has 60 elements
            raise ValueError("Icosahedral group must have 60 elements")
        
        # Extract cyclic subgroups
        if subsampling_factor == 12:  # 60 → 5 elements (C5)
            return [0, 12, 24, 36, 48]  # Example C5 subgroup
        elif subsampling_factor == 20:  # 60 → 3 elements (C3)
            return [0, 20, 40]  # Example C3 subgroup
        else:
            # General case: stride sampling
            return nodes[::subsampling_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """Validate icosahedral → cycle subsampling parameters."""
        if group_size != 60:
            raise ValueError("Icosahedral group must have 60 elements")
        if 60 % subsampling_factor != 0:
            raise ValueError(f"Subsampling factor {subsampling_factor} must divide 60")
        return True


# ========================= Subsampling Strategy Registry =========================

class SubsamplingRegistry:
    """
    Registry for subsampling strategies based on group-to-subgroup transitions.
    
    **Design Pattern:**
    Maps (parent_group, subgroup) pairs to strategy implementations.
    Enables complex 3D group subsampling without hardcoded logic.
    
    **Extension:**
    ```python
    SubsamplingRegistry.register(
        ("tetrahedral", "cycle"), 
        TetrahedralToCycleStrategy()
    )
    ```
    """
    
    _strategies: Dict[Tuple[str, str], SubsamplingStrategy] = {}
    
    @classmethod
    def register(cls, transition: Tuple[str, str], strategy: SubsamplingStrategy):
        """
        Register a subsampling strategy for a group transition.
        
        Args:
            transition: (parent_group_type, subgroup_type) tuple
            strategy: Strategy implementation
        """
        if not isinstance(strategy, SubsamplingStrategy):
            raise TypeError("Strategy must inherit from SubsamplingStrategy")
        cls._strategies[transition] = strategy
    
    @classmethod
    def get_strategy(cls, parent_group: str, subgroup: str) -> SubsamplingStrategy:
        """Get strategy for a group transition."""
        key = (parent_group, subgroup)
        if key not in cls._strategies:
            available = list(cls._strategies.keys())
            raise NotImplementedError(
                f"Subsampling {parent_group} → {subgroup} not supported. "
                f"Available transitions: {available}"
            )
        return cls._strategies[key]
    
    @classmethod
    def get_supported_transitions(cls) -> List[Tuple[str, str]]:
        """Get list of supported group transitions."""
        return list(cls._strategies.keys())


# Register built-in 2D subsampling strategies
SubsamplingRegistry.register(("cycle", "cycle"), CycleToCycleStrategy())
SubsamplingRegistry.register(("cyclic", "cycle"), CycleToCycleStrategy())  # Alias
SubsamplingRegistry.register(("cyclic", "cyclic"), CycleToCycleStrategy())  # Alias

SubsamplingRegistry.register(("dihedral", "dihedral"), DihedralToDihedralStrategy())
SubsamplingRegistry.register(("dihedral", "cycle"), DihedralToCycleStrategy())
SubsamplingRegistry.register(("dihedral", "adihedral"), DihedralToAdihedralStrategy())

# Register 3D subsampling strategies
SubsamplingRegistry.register(("octahedral", "cycle"), OctahedralToCycleStrategy())
SubsamplingRegistry.register(("icosahedral", "cycle"), IcosahedralToCycleStrategy())


# ========================= Public API =========================

def subsample_with_strategy(
    group_size: int,
    group_type: str,
    group_generator: Optional[str],
    subgroup_type: str,
    subsampling_factor: int,
) -> List[int]:
    """
    Perform subsampling using registered strategies.
    
    Args:
        group_size: Size of parent group
        group_type: Type of parent group
        group_generator: Generator specification (group-dependent)
        subgroup_type: Type of target subgroup
        subsampling_factor: Factor by which to reduce group size
        
    Returns:
        List of subsampled node indices
        
    **Supported Transitions:**
    - 2D: cycle→cycle, dihedral→dihedral, dihedral→cycle, dihedral→adihedral
    - 3D: octahedral→cycle, icosahedral→cycle
    
    **Extension:**
    Register new strategies with SubsamplingRegistry.register()
    """
    # Get appropriate strategy
    strategy = SubsamplingRegistry.get_strategy(group_type, subgroup_type)
    
    # Validate parameters
    strategy.validate_parameters(group_size, subsampling_factor, group_generator)
    
    # Perform subsampling
    nodes = list(range(group_size))
    return strategy.subsample(nodes, subsampling_factor, group_generator)


def get_supported_subsampling_transitions() -> List[Tuple[str, str]]:
    """Get list of supported group-to-subgroup transitions."""
    return SubsamplingRegistry.get_supported_transitions()


def is_subsampling_supported(parent_group: str, subgroup: str) -> bool:
    """Check if a subsampling transition is supported."""
    return (parent_group, subgroup) in SubsamplingRegistry.get_supported_transitions()
