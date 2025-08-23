"""
Group utilities with extensible registry for 2D and 3D groups.

This module provides a registry-based approach for handling different group types,
making it easy to extend to new groups (tetrahedral, octahedral, etc.) without
modifying existing code.
"""

from typing import Dict, Callable, Any, Optional
import torch
import torch.nn as nn
from escnn.group import cyclic_group, dihedral_group, trivial_group, octa_group, full_octa_group
from escnn import gspaces
import escnn.nn as enn


# ========================= Group Registry =========================

class GroupRegistry:
    """
    Registry for group types with their ESCNN functions and gspace constructors.
    
    **Design Pattern:**
    Uses registry pattern to map group type names to their ESCNN implementations.
    This eliminates hardcoded if/elif chains and enables clean extension to 3D groups.
    
    **Extension:**
    To add new group types:
    ```python
    GroupRegistry.register("tetrahedral", {
        "escnn_func": tetrahedral_group,
        "gspace_func": lambda order: gspaces.rot3dOnR3(),
        "dimension": 3,
        "description": "Tetrahedral symmetry group T"
    })
    ```
    """
    
    _registry: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, group_type: str, config: Dict[str, Any]):
        """
        Register a new group type.
        
        Args:
            group_type: String identifier for the group type
            config: Dictionary with 'escnn_func', 'gspace_func', 'dimension', 'description'
        """
        required_keys = ['escnn_func', 'gspace_func', 'dimension']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Group config must include '{key}'")
        
        cls._registry[group_type] = config
    
    @classmethod
    def get_group_config(cls, group_type: str) -> Dict[str, Any]:
        """Get configuration for a group type."""
        if group_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Group type '{group_type}' not supported. Available: {available}")
        return cls._registry[group_type]
    
    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported group types."""
        return list(cls._registry.keys())
    
    @classmethod
    def is_supported(cls, group_type: str) -> bool:
        """Check if a group type is supported."""
        return group_type in cls._registry


# Register built-in 2D groups
GroupRegistry.register("cycle", {
    "escnn_func": cyclic_group,
    "gspace_func": lambda order: gspaces.rot2dOnR2(order),
    "dimension": 2,
    "description": "Cyclic group C_n (2D rotations)"
})

GroupRegistry.register("cyclic", {  # Alias for cycle
    "escnn_func": cyclic_group,
    "gspace_func": lambda order: gspaces.rot2dOnR2(order),
    "dimension": 2,
    "description": "Cyclic group C_n (2D rotations) - alias for 'cycle'"
})

GroupRegistry.register("dihedral", {
    "escnn_func": dihedral_group,
    "gspace_func": lambda order: gspaces.flipRot2dOnR2(order),
    "dimension": 2,
    "description": "Dihedral group D_n (2D rotations + reflections)"
})

GroupRegistry.register("trivial", {
    "escnn_func": lambda order: trivial_group(),  # Trivial group doesn't use order
    "gspace_func": lambda order: gspaces.trivialOnR2() if order <= 4 else gspaces.trivialOnR3(),
    "dimension": 2,  # Can be 2D or 3D depending on context
    "description": "Trivial group (identity only)"
})

# Register 2D groups acting on 3D data
GroupRegistry.register("dihedralOnR3", {
    "escnn_func": dihedral_group,
    "gspace_func": lambda order: gspaces.rot2dOnR3(order),  # 2D rotations in 3D space
    "dimension": 3,
    "description": "Dihedral group D_n acting on 3D data (rotations around z-axis + reflections)"
})

GroupRegistry.register("rot2dOnR3", {
    "escnn_func": cyclic_group,
    "gspace_func": lambda order: gspaces.rot2dOnR3(order),  # 2D rotations in 3D space
    "dimension": 3,
    "description": "Cyclic group C_n acting on 3D data (rotations around z-axis)"
})

# Register 3D groups (ready for implementation)
GroupRegistry.register("octahedral", {
    "escnn_func": lambda order: octa_group(),  # Octahedral group doesn't take order parameter
    "gspace_func": lambda order: gspaces.octaOnR3(),
    "dimension": 3,
    "description": "Octahedral group O (24 elements: rotational symmetries of cube/octahedron)"
})

GroupRegistry.register("full_octahedral", {
    "escnn_func": lambda order: full_octa_group(),  # Full octahedral group doesn't take order parameter
    "gspace_func": lambda order: gspaces.fullOctaOnR3(),
    "dimension": 3,
    "description": "Full octahedral group O_h (48 elements with inversion)"
})

GroupRegistry.register("icosahedral", {
    "escnn_func": lambda order: ico_group(),  # Icosahedral group doesn't take order parameter
    "gspace_func": lambda order: gspaces.icoOnR3(),
    "dimension": 3,
    "description": "Icosahedral group I (60 elements: rotational symmetries of icosahedron)"
})

GroupRegistry.register("full_icosahedral", {
    "escnn_func": lambda order: full_ico_group(),  # Full icosahedral group doesn't take order parameter
    "gspace_func": lambda order: gspaces.fullIcoOnR3(),
    "dimension": 3,
    "description": "Full icosahedral group I_h (120 elements with inversion)"
})

GroupRegistry.register("so3", {
    "escnn_func": lambda order: so3_group(maximum_frequency=order),  # SO3 uses maximum_frequency parameter
    "gspace_func": lambda order: gspaces.rot3dOnR3(maximum_frequency=order),
    "dimension": 3,
    "description": "Special orthogonal group SO(3) (continuous 3D rotations)"
})


# ========================= Public API Functions =========================

def get_group(group_type: str, order: int):
    """
    Get ESCNN group object for the specified group type.
    
    Args:
        group_type: Name of the group ('cycle', 'dihedral', 'octahedral', etc.)
        order: Order of the group (interpretation depends on group type)
        
    Returns:
        ESCNN group object
        
    **Supported Groups:**
    - 2D: cycle, dihedral, trivial
    - 3D: octahedral, icosahedral, so3 (and their full versions)
    
    **Extension:**
    To add new groups, use GroupRegistry.register()
    """
    config = GroupRegistry.get_group_config(group_type)
    return config["escnn_func"](order)


def get_gspace(
    *, group_type: str, order: int, num_features: int, representation: str = "regular", domain: int = None
):
    """
    Get ESCNN FieldType (gspace feature type) for the specified group.
    
    Args:
        group_type: Name of the group
        order: Order of the group
        num_features: Number of features/channels
        representation: Type of representation ('regular' or 'trivial')
        domain: Spatial dimension (2 or 3). If None, uses group's natural dimension.
        
    Returns:
        ESCNN FieldType object
        
    **Dimension Handling:**
    - 2D groups: Use 2D gspaces (rot2dOnR2, flipRot2dOnR2)
    - 3D groups: Use 3D gspaces (rot3dOnR3, octaOnR3, icoOnR3)
    - 2D groups on 3D data: Use 3D gspaces (rot2dOnR3, dihedralOnR3)
    
    **Examples:**
    - get_gspace("dihedral", 8, 32, domain=2) → flipRot2dOnR2(8)
    - get_gspace("dihedral", 8, 32, domain=3) → rot2dOnR3(8) (2D rotations in 3D space)
    - get_gspace("octahedral", 24, 32) → octaOnR3() (natural 3D group)
    """
    # If domain is specified, use it to determine gspace type
    if domain is not None:
        if domain == 2:
            # Force 2D gspace
            if group_type in ["dihedral", "cycle", "cyclic"]:
                if group_type == "dihedral":
                    gspace = gspaces.flipRot2dOnR2(order)
                else:
                    gspace = gspaces.rot2dOnR2(order)
            else:
                # For 3D groups, we need to check if they support 2D operations
                config = GroupRegistry.get_group_config(group_type)
                if config["dimension"] == 3:
                    raise ValueError(f"Group type '{group_type}' is inherently 3D and cannot be used in 2D domain")
                gspace = config["gspace_func"](order)
        elif domain == 3:
            # Force 3D gspace
            if group_type in ["dihedral", "cycle", "cyclic"]:
                # Use 2D groups acting on 3D data
                if group_type == "dihedral":
                    gspace = gspaces.rot2dOnR3(order)
                else:
                    gspace = gspaces.rot2dOnR3(order)
            else:
                # Use natural 3D gspace
                config = GroupRegistry.get_group_config(group_type)
                gspace = config["gspace_func"](order)
        else:
            raise ValueError(f"Domain {domain} not supported. Use 2 or 3.")
    else:
        # Use group's natural dimension
        config = GroupRegistry.get_group_config(group_type)
        gspace = config["gspace_func"](order)
    
    if representation == "regular" or representation is None:
        g_feature = enn.FieldType(gspace, num_features * [gspace.regular_repr])
    elif representation == "trivial":
        g_feature = enn.FieldType(gspace, num_features * [gspace.trivial_repr])
    else:
        raise ValueError(f"Representation '{representation}' not supported")
    
    return g_feature


def get_group_dimension(group_type: str) -> int:
    """
    Get the spatial dimension for a group type.
    
    Args:
        group_type: Name of the group
        
    Returns:
        Spatial dimension (2 for 2D groups, 3 for 3D groups)
    """
    config = GroupRegistry.get_group_config(group_type)
    return config["dimension"]


def get_supported_group_types() -> list[str]:
    """Get list of all supported group types."""
    return GroupRegistry.get_supported_types()


def is_3d_group(group_type: str) -> bool:
    """Check if a group type is a 3D group."""
    try:
        return get_group_dimension(group_type) == 3
    except ValueError:
        return False



