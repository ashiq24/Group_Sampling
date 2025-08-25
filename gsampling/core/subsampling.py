"""
Subsampling strategies for different group-to-subgroup transitions.

This module implements the strategy pattern for subsampling operations,
enabling clean extension to complex 3D group relationships without
modifying existing code.
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Optional
from abc import ABC, abstractmethod
from escnn.group import octa_group, full_octa_group

class SubsamplingStrategy(ABC):
    """
    Abstract base class for subsampling strategies.
    
    Each strategy implements a specific group-to-subgroup subsampling algorithm
    based on the mathematical structure of the groups involved.
    """
    
    @abstractmethod
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
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
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
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
    
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
                 generator: Optional[str] = None) -> List[int]:
        """Simple stride subsampling for cyclic groups."""
        return nodes[::subsampling_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
                          generator: Optional[str] = None) -> bool:
        """Validate cycle → cycle subsampling parameters."""
        if group_size % subsampling_factor != 0:
            raise ValueError(f"Group size {group_size} must be divisible by factor {subsampling_factor}")
        return True


class DihedralToDihedralStrategy(SubsamplingStrategy):
    """Subsampling strategy for dihedral → dihedral transitions."""
    
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
                 generator: Optional[str] = None) -> List[int]:
        """Stride subsampling for dihedral groups."""
        if generator != "r-s":
            raise ValueError("Dihedral → dihedral subsampling requires 'r-s' generator")
        return nodes[::subsampling_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
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
    
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
                 generator: Optional[str] = None) -> List[int]:
        """Subsample dihedral group to cyclic subgroup (rotation part only)."""
        if generator != "r-s":
            raise ValueError("Dihedral → cycle subsampling requires 'r-s' generator")
        
        # Extract rotation part (first half) and subsample
        rotation_nodes = nodes[: len(nodes) // 2]
        adjusted_factor = subsampling_factor // 2
        return rotation_nodes[::adjusted_factor]
    
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
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
    
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
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
    
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
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
    
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
                 generator: Optional[str] = None) -> List[int]:
        """
        Subsample octahedral group to cyclic subgroup.
        
        **Mathematical Background:**
        Octahedral group O has several cyclic subgroups:
        - C4: 4-fold rotations around face-to-face axes (6 such axes)
        """
        G = octa_group()
        elements = list(G.elements)
        subgroup_indices = []
        
        for i, g in enumerate(elements):
            # Convert to rotation matrix
            rot_mat = g.to('MAT')
            
            # Check if this rotation fixes the z-axis
            # For a rotation around z-axis, the third row and column should be [0,0,1]
            if (np.allclose(rot_mat[2, :], [0, 0, 1]) and 
                np.allclose(rot_mat[:, 2], [0, 0, 1])):
                subgroup_indices.append(i)
        
        # We should have exactly 4 elements in a C4 subgroup
        if len(subgroup_indices) != 4:
            raise ValueError("Failed to identify C4 subgroup elements around z-axis")
        
        # Sort by rotation angle for consistent ordering
        # The identity should be first, then 90°, 180°, 270° rotations
        def get_rotation_angle(idx):
            rot_mat = elements[idx].to('MAT')
            # Extract rotation angle from matrix
            trace = np.trace(rot_mat)
            angle = np.arccos((trace - 1) / 2)
            return angle
        
        subgroup_indices.sort(key=get_rotation_angle)
        
        return subgroup_indices
    
    def validate_parameters(self, group_size: int, subsampling_factor: int, 
                          generator: Optional[str] = None) -> bool:
        """Validate octahedral → cycle subsampling parameters."""
        if group_size != 24:
            raise ValueError("Octahedral group must have 24 elements")
        return True


class FullOctahedralToCycleStrategy(SubsamplingStrategy):
    """Subsampling strategy for full octahedral → cyclic subgroup transitions around z-axis."""
    
    def __init__(self):
        # Precompute the z-axis cyclic subgroup indices
        self._z_axis_subgroup_indices = self._find_z_axis_subgroup()
    
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
                 generator: Optional[str] = None) -> List[int]:
        """
        Subsample full octahedral group to cyclic subgroup around z-axis.
        
        **Mathematical Background:**
        The full octahedral group O_h contains a C4 cyclic subgroup representing
        4-fold rotations around the z-axis:
        - Identity (0° rotation)
        - 90° rotation around z-axis
        - 180° rotation around z-axis  
        - 270° rotation around z-axis
        
        **Implementation:**
        We identify the actual subgroup elements that preserve the z-axis.
        """

        # Return the corresponding node indices for the z-axis subgroup
        return [nodes[i] for i in self._z_axis_subgroup_indices]
    
    def _find_z_axis_subgroup(self):
        """Find indices of C4 subgroup elements (rotations around z-axis)."""
        G = full_octa_group()
        elements = list(G.elements)
        subgroup_indices = []
        
        for i, g in enumerate(elements):
            # Convert to rotation matrix using correct format for full octahedral group
            try:
                result = g.to('[int | MAT]')
                ref = result[0]  # 0 for proper rotations, 1 for improper rotations
                rot_mat = result[1]  # Extract the 3x3 matrix from tuple
            except Exception:
                # Fallback to other format
                try:
                    result = g.to('[MAT | MAT]')
                    rot_mat = result[1]  # Extract the 3x3 matrix from tuple
                    # Determine if it's proper or improper rotation
                    det = np.linalg.det(rot_mat)
                    ref = 0 if np.allclose(det, 1.0, atol=1e-10) else 1
                except Exception:
                    # Skip this element if conversion fails
                    continue
            
            # Check if this rotation fixes the z-axis AND is a proper rotation
            # For a rotation around z-axis, the third row and column should be [0,0,1]
            if (np.allclose(rot_mat[2, :], [0, 0, 1], atol=1e-10) and 
                np.allclose(rot_mat[:, 2], [0, 0, 1], atol=1e-10) and ref == 0):
                subgroup_indices.append(i)  # Add the index, not ref
        
        # We should have exactly 4 elements in a C4 subgroup
        if len(subgroup_indices) != 4:
            raise ValueError("Failed to identify C4 subgroup elements around z-axis")
        
        # Sort by rotation angle for consistent ordering (identity first, then by increasing angle)
        def get_rotation_angle(idx):
            g = elements[idx]
            try:
                result = g.to('[int | MAT]')
                rot_mat = result[1]
            except Exception:
                try:
                    result = g.to('[MAT | MAT]')
                    rot_mat = result[1]
                except Exception:
                    raise Exception(f"Failed to convert element {g} to matrix")
            
            # Check if it's the identity
            if np.allclose(rot_mat, np.eye(3), atol=1e-10):
                return 0.0
            
            # Calculate rotation angle from trace
            trace = np.trace(rot_mat)
            if np.allclose(trace, -1.0, atol=1e-10):
                # 180° rotation
                return np.pi
            else:
                # General case: extract angle from trace
                cos_angle = (trace - 1) / 2
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                return angle
        
        # Sort by rotation angle (identity first, then increasing angles)
        subgroup_indices.sort(key=get_rotation_angle)
        
        return subgroup_indices
    
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
                          generator: Optional[str] = None) -> bool:
        """Validate full octahedral → cycle subsampling parameters."""
        if group_size != 48:
            raise ValueError("Full octahedral group must have 48 elements")
       
        return True


class FullOctahedralToDihedralStrategy(SubsamplingStrategy):
    """Subsampling strategy for full octahedral → dihedral subgroup transitions around z-axis."""
    
    def __init__(self):
        # Precompute the z-axis dihedral subgroup indices
        self._z_axis_subgroup_indices = self._find_z_axis_dihedral_subgroup()
    
    def subsample(self, nodes: List[int], subsampling_factor: int, 
                 generator: Optional[str] = None) -> List[int]:
        """
        Subsample full octahedral group to dihedral subgroup around z-axis.
        
        **Mathematical Background:**
        The full octahedral group O_h contains a D4 dihedral subgroup representing
        symmetries of a square in the xy-plane:
        - 4 rotations around z-axis (0°, 90°, 180°, 270°)
        - 4 reflections (through xz, yz, and diagonal planes)
        
        **Implementation:**
        We identify the actual subgroup elements that preserve the z-axis.
        """
        if len(nodes) != 48:  # Full octahedral group has 48 elements
            raise ValueError("Full octahedral group must have 48 elements")
        
        # Return the corresponding node indices for the z-axis subgroup
        return [nodes[i] for i in self._z_axis_subgroup_indices]
    
    def _find_z_axis_dihedral_subgroup(self):
        """Find indices of D4 dihedral subgroup elements around z-axis."""
        G = full_octa_group()
        elements = list(G.elements)
        subgroup_indices = []
        
        
        for i, g in enumerate(elements):
            # Convert to rotation matrix
            group_rep = g.to('[int | MAT]')
            rot_mat = group_rep[1]
            
            # Check if this transformation fixes the z-axis
            # For transformations around z-axis, the third row and column should be [0,0,1]
            if (np.allclose(rot_mat[2, :], [0, 0, 1]) and np.allclose(rot_mat[:, 2], [0, 0, 1])):
                subgroup_indices.append(i)
        
        # We should have exactly 8 elements in a D4 subgroup
        if len(subgroup_indices) != 8:
            raise ValueError("Failed to identify D4 subgroup elements around z-axis")
        
        # Sort by type (rotations first, then reflections)
        # Within each type, sort by rotation angle (identity first, then by increasing angle)
        def get_sorting_key(idx):
            g = elements[idx]
            
            # Convert to matrix using correct format for full octahedral group
            try:
                result = g.to('[int | MAT]')
                ref = result[0]  # 0 for proper rotations, 1 for improper rotations
                rot_mat = result[1]  # Extract the 3x3 matrix from tuple
            except Exception:
                # Fallback to other format
                try:
                    result = g.to('[MAT | MAT]')
                    rot_mat = result[1]  # Extract the 3x3 matrix from tuple
                    # Determine if it's proper or improper rotation
                    det = np.linalg.det(rot_mat)
                    ref = 0 if np.allclose(det, 1.0, atol=1e-10) else 1
                except Exception:
                    # Skip this element if conversion fails
                    return (2, 0)  # Put problematic elements at the end
            
            # Calculate rotation angle from matrix (works for both proper and improper rotations)
            if np.allclose(rot_mat, np.eye(3), atol=1e-10):
                # Identity element gets highest priority (smallest key)
                angle = 0.0
            else:
                # Calculate rotation angle from trace
                trace = np.trace(rot_mat)
                if np.allclose(trace, -1.0, atol=1e-10):
                    # 180° rotation
                    angle = np.pi
                else:
                    # General case: extract angle from trace
                    cos_angle = (trace - 1) / 2
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.arccos(cos_angle)
            
            if ref == 0:
                # Proper rotation - sort by angle
                return (0, angle)  # Proper rotations first, sorted by angle
            else:
                # Improper rotation/reflection - put after proper rotations, but also sorted by angle
                return (1, angle)  # Reflections after rotations, sorted by angle
        
        # Sort: proper rotations first (by angle), then reflections (by angle)
        subgroup_indices.sort(key=get_sorting_key)
        
        return subgroup_indices
    
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
                          generator: Optional[str] = None) -> bool:
        """Validate full octahedral → dihedral subsampling parameters."""
        if group_size != 48:
            raise ValueError("Full octahedral group must have 48 elements")
        
        return True

    
class FullOctahedralToOctahedralStrategy(SubsamplingStrategy):
    """Subsampling strategy for full octahedral → octahedral transitions."""
    
    def __init__(self):
        # Precompute the octahedral subgroup indices
        self._octahedral_subgroup_indices = self._find_octahedral_subgroup()
    
    def subsample(self, nodes: List[int], subsampling_factor: Optional[int] = None, 
                 generator: Optional[str] = None) -> List[int]:
        """
        Subsample full octahedral group to octahedral subgroup.
        
        **Mathematical Background:**
        The full octahedral group O_h contains the octahedral group O as a subgroup.
        O_h consists of 48 elements: 24 proper rotations (O) and 24 improper rotations
        (rotations combined with inversion).
        
        **Implementation:**
        We identify the proper rotations (determinant +1) to extract the octahedral subgroup.
        """
        if len(nodes) != 48:  # Full octahedral group has 48 elements
            raise ValueError("Full octahedral group must have 48 elements")
        
        # Return the corresponding node indices for the octahedral subgroup
        return [nodes[i] for i in self._octahedral_subgroup_indices]
    
    def _find_octahedral_subgroup(self):
        """Find indices of octahedral subgroup elements (proper rotations)."""
        G = full_octa_group()
        elements = list(G.elements)
        subgroup_indices = []
        
        for i, g in enumerate(elements):
            # Convert to transformation matrix using correct format for full octahedral group
            try:
                result = g.to('[int | MAT]')
                ref = result[0]  # 0 for proper rotations, 1 for improper rotations
                transform_mat = result[1]  # Extract the 3x3 matrix from tuple
            except Exception:
                # Fallback to other format
                try:
                    result = g.to('[MAT | MAT]')
                    transform_mat = result[1]  # Extract the 3x3 matrix from tuple
                    # Determine if it's proper or improper rotation
                    det = np.linalg.det(transform_mat)
                    ref = 0 if np.allclose(det, 1.0, atol=1e-10) else 1
                except Exception:
                    # Skip this element if conversion fails
                    continue
            
            # Check if this is a proper rotation (ref = 0 and determinant = +1)
            if ref == 0 and np.allclose(np.linalg.det(transform_mat), 1.0, atol=1e-10):
                subgroup_indices.append(i)
        
        # We should have exactly 24 elements in the octahedral subgroup
        if len(subgroup_indices) != 24:
            raise ValueError("Failed to identify octahedral subgroup elements")
        
        return subgroup_indices
    
    def validate_parameters(self, group_size: int, subsampling_factor: Optional[int] = None, 
                          generator: Optional[str] = None) -> bool:
        """Validate full octahedral → octahedral subsampling parameters."""
        if group_size != 48:
            raise ValueError("Full octahedral group must have 48 elements")
        
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
SubsamplingRegistry.register(("full_octahedral", "cycle"), FullOctahedralToCycleStrategy())
SubsamplingRegistry.register(("full_octahedral", "dihedral"), FullOctahedralToDihedralStrategy())
SubsamplingRegistry.register(("full_octahedral", "octahedral"), FullOctahedralToOctahedralStrategy())

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
