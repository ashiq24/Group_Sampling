import torch
import torch.nn as nn
from gsampling.layers.sampling import SamplingLayer
from .anti_aliasing import AntiAliasingLayer
from gsampling.utils.graph_constructors import GraphConstructor
from gsampling.utils.group_utils import *
from gsampling.layers.cannonicalizer import Cannonicalizer


class SubgroupDownsample(nn.Module):
    def __init__(
        self,
        group_type: str,
        order: int = None,
        sub_group_type: str = None,
        subsampling_factor: int = 1,
        num_features: int = 1,
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

        For cyclic groups, the generator is 'r'. The order of the elements is: e, r, r^2, ..., r^(order-1).
        For dihedral groups, the generator is 'r-s'. The order of the elements is: e, r, r^2, ..., r^(order-1), s, sr, sr^2, ..., sr^(order-1).
        For octahedral groups, works on 3D data and order is fixed (24 for octahedral, 48 for full_octahedral).

        Args:
            group_type (str): Type of the main group ('cycle', 'dihedral', 'octahedral', 'full_octahedral').
            order (int, optional): Order of the main group (ignored for octahedral groups).
            sub_group_type (str, optional): Type of the subgroup to downsample to (defaults to same as group_type).
            subsampling_factor (int, optional): Factor by which to reduce group size (default: 1).
            num_features (int, optional): Number of input channels/features (default: 1).
            generator (str, optional): Group generator type. 'r-s' by default for dihedral, 'r' for cyclic.
            device (str, optional): Device to use ('cpu' or 'cuda:0').
            dtype (torch.dtype, optional): Tensor data type.
            sample_type (str, optional): Type of sampling strategy ('sample' or 'pool'). 
                'sample' discards elements not in subgroup, 'pool' performs max pooling.
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
        self.sub_group_type = sub_group_type or group_type  # Default to same group type
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
        
        # ========================================================================
        # SUBGROUP ORDER CALCULATION
        # ========================================================================
        # This section calculates the order of the subgroup H based on the main group G
        # and the subsampling factor. The subgroup order determines how many channels
        # we'll have after downsampling: |H| = |G| / subsampling_factor
        #
        # Mathematical Foundation:
        # - Group downsampling maps from L²(G) to L²(H) where H ⊆ G
        # - Channel reduction: C * |G| → C * |H| 
        # - For proper group structure, |H| must divide |G|
        # - The subsampling factor k satisfies: |G| = k * |H|
        #
        # Different group types require different handling:
        # 1. 3D groups (octahedral, full_octahedral): Fixed orders, special handling
        # 2. 2D groups (cycle, dihedral): Variable orders based on parameters
        
        if group_type in ["octahedral", "full_octahedral"]:
            # ====================================================================
            # 3D GROUP HANDLING (Octahedral Groups)
            # ====================================================================
            # 3D groups have fixed orders: octahedral = 24, full_octahedral = 48
            # They don't take order parameters in their constructors
            
            if self.sub_group_type == group_type:
                # Same group type subsampling: G → G (e.g., octahedral → octahedral)
                # Mathematical: |H| = |G| / k where k is the subsampling factor
                sub_order = self.G.order() // subsampling_factor
            else:
                # Cross-group subsampling: G → H (e.g., full_octahedral → octahedral)
                # This is more complex as we're changing group types
                
                if self.sub_group_type in ["octahedral", "full_octahedral"]:
                    # 3D to 3D group transition
                    # Create the target 3D group to get its order
                    sub_group_G = get_group(self.sub_group_type, 1)  # Use dummy order for 3D groups
                    # Calculate subgroup order based on subsampling factor
                    sub_order = sub_group_G.order() // max(subsampling_factor, 1)
                else:
                    # 3D to 2D group transition (e.g., octahedral → cycle)
                    # This is the most complex case requiring careful order calculation
                    
                    if self.sub_group_type in ["cycle", "cyclic"]:
                        # ================================================================
                        # OCTAHEDRAL TO CYCLIC DOWNSAMPLING
                        # ================================================================
                        # Mathematical: We want to map from octahedral (24) to cyclic (6)
                        # with subsampling factor 6. However, we need to ensure the
                        # cyclic group has a reasonable order for the application.
                        #
                        # Calculation: |H| = |G| / k = 24 / 6 = 4
                        # But cyclic groups work better with order 6, so we use max(6, 4) = 6
                        # This ensures we have enough rotational symmetry for the task
                        
                        sub_order = 24 // max(subsampling_factor, 1)  # Calculate base order
                        # Ensure minimum order of 6 for cycle groups to maintain useful symmetry
                        sub_order = max(6, sub_order)
                        # Create the cyclic group and get its actual order2
                        sub_group_G = get_group(self.sub_group_type, sub_order)
                        sub_order = sub_group_G.order()  # Use the actual order of the created group
                    else:
                        # Other 2D groups (dihedral, etc.)
                        sub_group_G = get_group(self.sub_group_type, None)
                        sub_order = sub_group_G.order() // max(subsampling_factor, 1)
        else:
            # ====================================================================
            # 2D GROUP HANDLING (Cycle, Dihedral Groups)
            # ====================================================================
            # 2D groups have variable orders and simpler subsampling logic
            # Mathematical: |H| = |G| / k for same group type, |G| / (k/2) for different types
            
            sub_order = (
                order // subsampling_factor
                if group_type == sub_group_type
                else order // max(subsampling_factor // 2, 1)
            )
        
        self.sub_order = sub_order
        # Ensure sub_order is at least 1 for all groups
        if self.sub_group_type in ["octahedral", "full_octahedral"]:
            self.G_sub = get_group(self.sub_group_type, 1)  # 3D groups don't use order
        else:
            self.G_sub = get_group(self.sub_group_type, max(1, sub_order))  # Ensure minimum order 1

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
            print("Initializing anti-aliasing layer")
            # Filter out apply_antialiasing and anti_aliasing_kwargs from kwargs
            filtered_kwargs = {k: v for k, v in self.anti_aliasing_kwargs.items() if k not in ['apply_antialiasing', 'anti_aliasing_kwargs']}
            self.anti_aliaser = AntiAliasingLayer(
                nodes=self.graphs.graph.nodes,
                adjaceny_matrix=self.graphs.graph.adjacency_matrix,
                basis=self.graphs.graph.fourier_basis,
                subsample_nodes=self.graphs.subgroup_graph.nodes,
                subsample_adjacency_matrix=self.graphs.subgroup_graph.adjacency_matrix,
                sub_basis=self.graphs.subgroup_graph.fourier_basis,
                dtype=self.dtype,
                device=self.device,
                raynold_op=self.graphs.graph.equi_raynold_op,
                **filtered_kwargs
            )
            self.anti_aliaser.to(device=self.device, dtype=self.dtype)
        else:
            print("Anti-aliasing layer not applied")
            self.anti_aliaser = None

        # Initialize canonicalizer if applicable
        if cannonicalize:
            self.cannonicalize = Cannonicalizer(
                group=group_type,
                nodes_num=self.G.order(),
                subgroup=sub_group_type,
                sub_nodes_num=self.G_sub.order(),
                in_channels=num_features,
                dtype=self.dtype,
                device=self.device,
            )
            self.cannonicalize.to(device=self.device, dtype=self.dtype)
        else:
            self.cannonicalize = None

    def forward(self, x: torch.Tensor):
        """
        x: Input tensor of shape (batch_size, group_size * num_features, height, width) or (batch_size, group_size).
        When the input is a feature of shape (batch_size, group_size * num_features, height, width),
        The features in the channel is distributated as follows:
        - Channels are organized as [feature₁@group_elem₁, feature₂@group_elem₁, ..., featureₙ@group_elemₖ]
        where group elements are ordered before feature channels within each group block.

        Returns:
            x: Downsampled tensor of shape (batch_size, sub_group_size * num_features, height, width) or (batch_size, sub_group_size)
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
        x: Input tensor of shape (batch_size, sub_group_size * num_features, height, width) or (batch_size, sub_group_size)
        Returns:
            x: Upsampled tensor of shape (batch_size, group_size * num_features, height, width) or (batch_size, group_size)
        """
        if self.anti_aliaser is not None:
            x = self.anti_aliaser.up_sample(x)
        else:
            x = self.sample.up_sample(x)

        return x
