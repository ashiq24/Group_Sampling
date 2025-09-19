"""
Group Canonicalization Layer for Equivariant Neural Networks

This module implements group canonicalization, a technique for normalizing group
representations to a canonical form. It computes coset representatives and applies
group transformations to align features with a standard reference frame.

Mathematical Foundation:
------------------------
Group canonicalization works by:

1. Finding coset representatives: For each group element g ∈ G, find a representative
   g₀ ∈ H (subgroup) such that g = g₀ · h for some h ∈ H.

2. Canonical transformation: Apply the inverse transformation g₀^-1 to align features
   with the canonical reference frame.

3. The canonical form ensures that equivalent group elements are mapped to the same
   representation, reducing redundancy and improving learning efficiency.

Key Features:
- Support for cyclic and dihedral groups
- Forward and backward canonicalization modes
- Automatic coset representative computation
- Integration with E(2)-CNN framework

Reference:
[1] Jin Xu, Hyunjik Kim, Tom Rainforth, Yee Whye Teh "Group Equivariant Subsampling"

Author: Group Sampling Team
"""

from escnn import nn
from escnn.group import *
from escnn import gspaces
import torch


class Cannonicalizer(nn.EquivariantModule):
    def __init__(
        self,
        group: str,  # Original group type ("dihedral" or "cycle")
        nodes_num: int,  # Number of elements in the original group G
        subgroup: str,  # Subgroup type ("dihedral" or "cycle")
        sub_nodes_num: int,  # Number of elements in the subgroup H
        in_channels: int,  # Number of input channels
        dtype: torch.dtype = torch.float32,  # Data type for computations
        device: str = "cpu",  # Compute device
    ):
        """
        Initialize the group canonicalization layer.
        
        This layer computes coset representatives and applies canonical transformations
        to normalize group representations. It supports both cyclic and dihedral groups
        with their respective subgroups.
        
        Parameters:
        -----------
        group : str
            Type of the original group G ("dihedral" or "cycle")
        nodes_num : int
            Order of the original group G (number of group elements)
        subgroup : str
            Type of the subgroup H ("dihedral" or "cycle")
        sub_nodes_num : int
            Order of the subgroup H (number of subgroup elements)
        in_channels : int
            Number of input feature channels
        dtype : torch.dtype
            Data type for tensor computations
        device : str
            Device for tensor operations
            
        Reference:
        ----------
        [1] Jin Xu, Hyunjik Kim, Tom Rainforth, Yee Whye Teh "Group Equivariant Subsampling"
        """
        super().__init__()
        
        # Store group structure parameters
        self.group = group  # Original group type
        self.nodes_num = nodes_num  # Order of original group G
        self.subgroup = subgroup  # Subgroup type
        self.sub_nodes_num = sub_nodes_num  # Order of subgroup H
        self.in_channels = in_channels  # Number of input channels
        self.dtype = dtype  # Data type for computations
        self.device = device  # Compute device

        # Initialize group space and field type based on group type
        if group == "dihedral":
            # Dihedral group D_n: rotations and reflections
            # For dihedral groups, nodes_num must be even (n rotations + n reflections)
            assert nodes_num % 2 == 0, "Dihedral group requires even number of nodes"
            
            # Create 2D rotation-flip group space with n/2 rotations
            self.gspace = gspaces.flipRot2dOnR2(nodes_num // 2)
            
            # Define field type with regular representation for each channel
            # Regular representation captures all group symmetries
            self.feature = nn.FieldType(
                self.gspace, in_channels * [self.gspace.regular_repr]
            )
        elif group == "cycle":
            # Cyclic group C_n: only rotations
            # Create 2D rotation group space with n rotations
            self.gspace = gspaces.rot2dOnR2(nodes_num)
            
            # Define field type with regular representation for each channel
            self.feature = nn.FieldType(
                self.gspace, in_channels * [self.gspace.regular_repr]
            )
        else:
            raise ValueError("Unknown group:", group)

        # Buffer to store coset representatives for backward pass
        self.buffer = None

    def coset_rep_r2(self, x):
        """Compute coset representatives for 2D spatial tensors.
        
        This method finds the dominant group element at each spatial location
        and computes the corresponding coset representative for canonicalization.
        
        Mathematical Process:
        1. Extract group fiber: x[:, :nodes_num, :, :] (first group dimension)
        2. Find argmax: dominant group element at each spatial location
        3. Compute coset representative: g₀ such that g = g₀ · h for h ∈ H
        
        Args:
            x: Input tensor of shape (batch, group * in_channels, h, w)
            
        Returns:
            List of coset representatives for each batch element
        """
        # Extract the group fiber (first nodes_num channels)
        # This represents the group dimension of the feature map
        fiber = x[:, :self.nodes_num, :, :]
        
        # Reshape to (batch, h*w, nodes_num) for easier argmax computation
        # Permute spatial dimensions to the front, then flatten h*w
        fiber = torch.permute(fiber, (0, 2, 3, 1)).reshape(x.shape[0], -1)
        
        # Find the dominant group element at each spatial location
        # argmax gives the index of the maximum activation across group dimension
        v = torch.argmax(fiber, dim=1)

        # Normalize to valid group element indices
        v = v % self.nodes_num
        
        # Compute coset representatives based on group and subgroup types
        if self.group == "dihedral" and self.subgroup == "dihedral":
            # D_n → D_m canonicalization
            # For dihedral groups, separate rotation and reflection components
            v = v % (self.nodes_num // 2)  # Rotation component
            v = v % (self.nodes_num // self.sub_nodes_num)  # Subgroup rotation
            # Return as (reflection, rotation) tuples with reflection=0
            v = [(0, i) for i in v.tolist()]
        elif self.group == "cycle" and self.subgroup == "cycle":
            # C_n → C_m canonicalization
            # For cyclic groups, just modulo by subgroup order
            v = v % (self.nodes_num // self.sub_nodes_num)
            v = v.tolist()
        elif self.group == "dihedral" and self.subgroup == "cycle":
            # D_n → C_m canonicalization
            # Extract reflection component (0 or 1)
            r = v // (self.nodes_num // 2)
            # Extract rotation component
            v = v % (self.nodes_num // 2)
            v = v % (self.nodes_num // (self.sub_nodes_num * 2))
            # Return as (reflection, rotation) tuples
            v = [(j, i) for (j, i) in zip(r.tolist(), v.tolist())]
        else:
            raise ValueError("Unknown group or subgroup")

        return v

    def coset_rep(self, x):
        """Compute coset representatives for 1D group tensors.
        
        This method finds the dominant group element in a 1D group signal
        and computes the corresponding coset representative for canonicalization.
        
        Mathematical Process:
        1. Find argmax: dominant group element in the signal
        2. Compute coset representative: g₀ such that g = g₀ · h for h ∈ H
        
        Args:
            x: Input tensor of shape (group,) representing a group signal
            
        Returns:
            Coset representative (group element or tuple)
        """
        # The input is already a group fiber (1D tensor)
        fiber = x
        
        # Find the dominant group element (maximum activation)
        v = torch.argmax(fiber)
        
        # Normalize to valid group element indices
        v = v % self.nodes_num
        
        # Compute coset representatives based on group and subgroup types
        if self.group == "dihedral" and self.subgroup == "dihedral":
            # D_n → D_m canonicalization
            # For dihedral groups, separate rotation and reflection components
            v = v % (self.nodes_num // 2)  # Rotation component
            v = v % (self.nodes_num // self.sub_nodes_num)  # Subgroup rotation
            # Return as (reflection, rotation) tuple with reflection=0
            v = (0, v.item())
        elif self.group == "cycle" and self.subgroup == "cycle":
            # C_n → C_m canonicalization
            # For cyclic groups, just modulo by subgroup order
            v = v % (self.nodes_num // self.sub_nodes_num)
            v = v.item()
        elif self.group == "dihedral" and self.subgroup == "cycle":
            # D_n → C_m canonicalization
            # Extract reflection component (0 or 1)
            r = v // (self.nodes_num // 2)
            # Extract rotation component
            v = v % (self.nodes_num // 2)
            v = v % (self.nodes_num // (self.sub_nodes_num * 2))
            # Return as (reflection, rotation) tuple
            v = (r.item(), v.item())
        else:
            raise ValueError("Unknown group or subgroup")

        return v

    def forward(self, x, coset_rep=None, mode="forward"):
        """Forward pass for group canonicalization.
        
        This method applies canonical transformations to normalize group representations.
        It supports both forward canonicalization (align to canonical form) and
        backward canonicalization (restore from canonical form).
        
        Mathematical Process:
        - Forward: Apply g₀^-1 to align features with canonical reference frame
        - Backward: Apply g₀ to restore features from canonical form
        
        Args:
            x: Input tensor (1D group signal or 4D spatial tensor)
            coset_rep: Coset representatives for backward mode (optional)
            mode: "forward" for canonicalization, "backward" for restoration
            
        Returns:
            Tuple of (canonicalized_tensor, coset_representatives)
        """
        if mode == "forward":
            # Forward canonicalization: align features to canonical form
            if x.dim() == 4:
                # 4D case: (batch, group*channels, height, width)
                # Compute coset representatives for each batch element
                v = self.coset_rep_r2(x)
                self.buffer = v  # Store for potential backward pass
                
                # Apply canonical transformation to each batch element
                # Transform: x_canonical = g₀^-1 · x where g₀ is coset representative
                result = torch.cat(
                    [
                        self.feature.transform(
                            j.unsqueeze(0),  # Add batch dimension
                            self.feature.fibergroup.element(
                                self.feature.fibergroup._inverse(u)  # g₀^-1
                            ),
                        )
                        for (j, u) in zip(x, v)  # Apply to each batch element
                    ],
                    dim=0,
                )
            elif x.dim() == 1:
                # 1D case: (group,) - single group signal
                # Compute coset representative
                v = self.coset_rep(x)
                self.buffer = v  # Store for potential backward pass
                
                # Apply canonical transformation using regular representation
                # x_canonical = ρ(g₀^-1) · x where ρ is regular representation
                result = (
                    torch.tensor(
                        self.feature.fibergroup.regular_representation(
                            self.feature.fibergroup.element(
                                self.feature.fibergroup._inverse(v)  # g₀^-1
                            )
                        )
                    ).to(device=x.device, dtype=x.dtype)
                    @ x  # Matrix-vector multiplication
                )
            assert result.shape == x.shape, "Canonicalization should preserve shape"
            return result, v
            
        elif mode == "backward":
            # Backward canonicalization: restore from canonical form
            # Use provided coset representatives or stored buffer
            if coset_rep is not None:
                v = coset_rep
            elif self.buffer is not None:
                v = self.buffer
            else:
                raise ValueError("coset_rep is None - no representatives available")

            if x.dim() == 4:
                # 4D case: restore spatial features from canonical form
                # Transform: x_restored = g₀ · x_canonical
                result = torch.cat(
                    [
                        self.feature.transform(
                            j.unsqueeze(0),  # Add batch dimension
                            self.feature.fibergroup.element(u)  # g₀
                        )
                        for (j, u) in zip(x, v)  # Apply to each batch element
                    ],
                    dim=0,
                )
            elif x.dim() == 1:
                # 1D case: restore group signal from canonical form
                # Transform: x_restored = ρ(g₀) · x_canonical
                result = (
                    torch.tensor(
                        self.feature.fibergroup.regular_representation(
                            self.feature.fibergroup.element(v)  # g₀
                        )
                    ).to(device=x.device, dtype=x.dtype)
                    @ x  # Matrix-vector multiplication
                )

            assert result.shape == x.shape, "Restoration should preserve shape"
            return result, v
        else:
            raise ValueError("Unknown mode - must be 'forward' or 'backward'")


