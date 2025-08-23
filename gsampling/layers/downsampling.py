import torch
import torch.nn as nn
from gsampling.layers.sampling import SamplingLayer
from .anti_aliasing import AntiAliasingLayer
from gsampling.utils.graph_constructors import GraphConstructor
from gsampling.utils.group_utils import *
from gsampling.layers.cannonicalizer import Cannonicalizer
import pdb

class SubgroupDownsample(nn.Module):
    def __init__(
        self,
        group_type: str,
        order: int,
        sub_group_type: str,
        subsampling_factor: int,
        num_features: int,
        generator: str = "r-s",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        sample_type: str = "sample",
        apply_antialiasing: bool = False,
        anti_aliasing_kwargs: dict = None,
        cannonicalize: bool = False,
    ):
        """
        A PyTorch module for downsampling features over groups (regular representation) to subgroups
        with optional anti-aliasing and canonicalization following [1].

        **2D Groups:**
        - For cyclic groups, the generator is 'r'. The order of the elements is: e, r, r^2, ..., r^(order-1).
        - For dihedral groups, the generator is 'r-s'. The order of the elements is: e, r, r^2, ..., r^(order-1), s, sr, sr^2, ..., sr^(order-1).

        **3D Groups:**
        - For octahedral groups (O), order=24, generator is ignored
        - For full octahedral groups (O_h), order=48, generator is ignored
        - For icosahedral groups (I), order=60, generator is ignored
        - For full icosahedral groups (I_h), order=120, generator is ignored

        Args:
            group_type (str): Type of the main group (e.g., 'cycle', 'dihedral', 'octahedral', 'full_octahedral').
            order (int): Order of the main group (number of group elements).
            sub_group_type (str): Type of the subgroup to downsample to.
            subsampling_factor (int): Factor by which to reduce group size.
            num_features (int): Number of input channels/features.
            generator (str, optional): Group generator type. 'r-s' by default assuming dihedral group. For cyclic group, use 'r'.
                Ignored for 3D groups.
            device (str, optional): Device to use ('cpu' or 'cuda:0').
            dtype (torch.dtype, optional): Tensor data type.
            sample_type (str, optional): Type of sampling strategy (e.g., 'sample', 'pool'). 'sample' by default, which
                discards elements that are not in the subgroup.
                'pool' will perform trivial max pooling over the group dimension.
            apply_antialiasing (bool, optional): Whether to apply anti-aliasing.
            anti_aliasing_kwargs (dict, optional): Arguments for anti-aliasing class.
            cannonicalize (bool, optional): Whether to apply canonicalization following [1].
        ---------
        1. Jin Xu, Hyunjik Kim, Tom Rainforth, Yee Whye Teh "Group Equivariant Subsampling"

        """
        super(SubgroupDownsample, self).__init__()

        # Initialize parameters
        self.group_type = group_type
        self.order = order
        self.sub_group_type = sub_group_type
        self.subsampling_factor = subsampling_factor
        self.num_features = num_features
        self.generator = generator
        self.device = device
        self.dtype = dtype
        self.sample_type = sample_type
        self.apply_antialiasing = apply_antialiasing
        self.anti_aliasing_kwargs = anti_aliasing_kwargs or {
            "smooth_operator": "adjacency",
            "mode": "linear_optim",
            "iterations": 100000,
            "smoothness_loss_weight": 1.0,
            "threshold": 0.0,
            "equi_constraint": True,
            "equi_correction": True,
        }

        # Initialize groups
        self.G = get_group(group_type, order)
        
        # Calculate subgroup order based on group types
        if self._is_3d_group(group_type):
            # For 3D groups, use the actual group order from ESCNN
            actual_order = self.G.order()
            if actual_order != order:
                self.order = actual_order
            
            # Calculate subgroup order based on subsampling factor
            if group_type == sub_group_type:
                # Same group type: simple division
                self.sub_order = actual_order // subsampling_factor
            else:
                # Different group types: use subsampling strategies
                # This will be handled by the subsampling layer
                self.sub_order = actual_order // subsampling_factor
        else:
            # 2D groups: use existing logic
            self.sub_order = (
                order // subsampling_factor
                if group_type == sub_group_type
                else order // max(subsampling_factor // 2, 1)
            )
        
        self.G_sub = get_group(sub_group_type, self.sub_order)

        # Initialize graph constructor
        self.graphs = GraphConstructor(
            group_size=self.G.order(),
            group_type=self.group_type,
            group_generator=self.generator,
            subgroup_type=self.sub_group_type,
            subsampling_factor=self.subsampling_factor,
        )
        
        # Initialize sampling layer
        self.sample = SamplingLayer(
            sampling_factor=self.subsampling_factor,
            nodes=self.graphs.graph.nodes,
            subsample_nodes=self.graphs.subgroup_graph.nodes,
            type=sample_type,
        )
        self.sample.to(device=self.device, dtype=self.dtype)

        # Initialize anti-aliasing layer if applicable
        if apply_antialiasing:
            self.anti_aliaser = AntiAliasingLayer(
                nodes=self.graphs.graph.nodes,
                adjaceny_matrix=self.graphs.graph.adjacency_matrix,
                basis=self.graphs.graph.fourier_basis,
                subsample_nodes=self.graphs.subgroup_graph.nodes,
                sub_basis=self.graphs.subgroup_graph.fourier_basis,
                dtype=self.dtype,
                device=self.device,
                raynold_op=self.graphs.graph.equi_raynold_op,
                **self.anti_aliasing_kwargs
            )
            self.anti_aliaser.to(device=self.device, dtype=self.dtype)
        else:
            self.anti_aliaser = None

                        # Initialize canonicalizer if applicable
        self.cannonicalize = None

    def _is_3d_group(self, group_type: str) -> bool:
        """Check if the group type is a 3D group."""
        return group_type in ['octahedral', 'full_octahedral', 'icosahedral', 'full_icosahedral', 'so3']

    def forward(self, x: torch.Tensor):
        """
        x: Input tensor of shape (batch_size, group_size * num_features, height, width) or 
           (batch_size, group_size * num_features, height, width, depth) for 3D data.
        When the input is a feature of shape (batch_size, group_size * num_features, height, width) or
        (batch_size, group_size * num_features, height, width, depth),
        The features in the channel is distributed as follows:
        - Channels are organized as [feature₁@group_elem₁, feature₂@group_elem₁, ..., featureₙ@group_elemₖ]
        where group elements are ordered before feature channels within each group block.

        Returns:
            x: Downsampled tensor of shape (batch_size, sub_group_size * num_features, height, width) or
               (batch_size, sub_group_size * num_features, height, width, depth) for 3D data.
            v: List of tuples containing the canonicalization information for each element in the batch.
        """
        # Apply canonicalization if enabled
        if self.cannonicalize is not None:
            x, v = self.cannonicalize(x)
        else:
            v = [(-1, -1)]

        # Apply anti-aliasing if enabled
        if self.apply_antialiasing:
            x = self.anti_aliaser(x)

        # Apply sampling
        x = self.sample(x)

        return x, v

    def upsample(self, x: torch.Tensor):
        """
        x: Input tensor of shape (batch_size, sub_group_size * num_features, height, width) or
           (batch_size, sub_group_size * num_features, height, width, depth) for 3D data.
        Returns:
            x: Upsampled tensor of shape (batch_size, group_size * num_features, height, width) or
               (batch_size, group_size * num_features, height, width, depth) for 3D data.
        """
        if self.anti_aliaser is not None:
            x = self.anti_aliaser.up_sample(x)
        else:
            x = self.sample.up_sample(x)

        return x
