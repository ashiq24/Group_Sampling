from .g_cnn import Gcnn
from .g_cnn_3d import Gcnn3D


def get_model(
    input_channel,
    num_channels=None,
    num_layers=2,
    kernel_sizes=None,
    num_classes=10,
    dwn_group_types=None,
    init_group_order=12,
    spatial_subsampling_factors=None,
    subsampling_factors=None,
    domain=2,
    pooling_type="max",
    apply_antialiasing=True,
    cannonicalize=False,
    dropout_rate=0.0,
    layer_kwargs=None,
    antialiasing_kwargs=None,
    fully_convolutional=False,
):
    """
    Creates and returns a Gcnn model with the specified configuration.

    Args:
        input_channel (int): Number of input channels. Default is 3.
        num_channels (list): List of integers specifying the number of channels for each layer.
        num_layers (list): List of integers specifying the number of layers.
        kernel_sizes (list): List of integers specifying the kernel sizes for each layer.
        num_classes (int): Number of output classes. Default is 10.
        dwn_group_types (list): List of lists specifying the group types for downsampling.
        init_group_order (int): Initial group order. Default is 12.
        spatial_subsampling_factors (list): List of spatial subsampling factors.
        subsampling_factors (list): List of subsampling factors.
        domain (int): Domain parameter. Default is 2.
        pooling_type (str): Type of pooling to apply. Default is 'max'.
        apply_antialiasing (bool): Whether to apply antialiasing. Default is True.
        cannonicalize (bool): Whether to canonicalize. Default is False.
        dropout_rate (float): Dropout rate. Default is 0.0.
        layer_kwargs (dict): Additional arguments for layers.
        antialiasing_kwargs (dict): Additional arguments for antialiasing.
        fully_convolutional (bool): Whether the model is fully convolutional. Default is False.

    Returns:
        Gcnn: Configured Gcnn model.
    """
    if num_channels is None:
        num_channels =  num_layers * [32]
    if kernel_sizes is None:
        kernel_sizes =  num_layers * [7]
    if dwn_group_types is None:
        dwn_group_types =  num_layers * [["dihedral", "dihedral"]]
    if spatial_subsampling_factors is None:
        spatial_subsampling_factors =  num_layers * [1]
    if subsampling_factors is None:
        subsampling_factors = num_layers * [1]
    if layer_kwargs is None:
        layer_kwargs = {"dilation": 2}
    if antialiasing_kwargs is None:
        antialiasing_kwargs = {
            "smooth_operator": "adjacency",
            "mode": "linear_optim",
            "iterations": 100,
            "smoothness_loss_weight": 8.0,
            "threshold": 0.0,
            "equi_constraint": True,
            "equi_correction": True,
        }

    num_channels = [input_channel] + num_channels
    model = Gcnn(
        num_channels=num_channels,
        num_layers=num_layers,
        kernel_sizes=kernel_sizes,
        num_classes=num_classes,
        dwn_group_types=dwn_group_types,
        init_group_order=init_group_order,
        spatial_subsampling_factors=spatial_subsampling_factors,
        subsampling_factors=subsampling_factors,
        domain=domain,
        pooling_type=pooling_type,
        apply_antialiasing=apply_antialiasing,
        canonicalize=cannonicalize,
        dropout_rate=dropout_rate,
        layer_kwargs=layer_kwargs,
        antialiasing_kwargs=antialiasing_kwargs,
        fully_convolutional=fully_convolutional,
    )
    return model


def get_3d_model(
    input_channel,
    num_channels=None,
    num_layers=2,
    kernel_sizes=None,
    num_classes=10,
    dwn_group_types=None,
    init_group_order=24,  # Default to octahedral order
    spatial_subsampling_factors=None,
    subsampling_factors=None,
    domain=3,
    pooling_type="max",
    apply_antialiasing=True,
    dropout_rate=0.0,
    layer_kwargs=None,
    antialiasing_kwargs=None,
    fully_convolutional=False,
):
    """
    Creates and returns a 3D Gcnn model with the specified configuration.

    Args:
        input_channel (int): Number of input channels.
        num_channels (list): List of integers specifying the number of channels for each layer.
        num_layers (list): List of integers specifying the number of layers.
        kernel_sizes (list): List of integers specifying the 3D kernel sizes for each layer.
        num_classes (int): Number of output classes. Default is 10.
        dwn_group_types (list): List of lists specifying the group types for downsampling.
            Examples:
            - Octahedral → Cycle: [["octahedral", "cycle"], ["cycle", "cycle"]]
            - Full octahedral → Dihedral: [["full_octahedral", "dihedral"], ["dihedral", "dihedral"]]
            - Full octahedral → Octahedral → Cycle: [["full_octahedral", "octahedral"], ["octahedral", "cycle"]]
        init_group_order (int): Initial group order. Default is 24 (octahedral).
        spatial_subsampling_factors (list): List of 3D spatial subsampling factors.
        subsampling_factors (list): List of group subsampling factors.
        domain (int): Domain parameter. Must be 3 for 3D models.
        pooling_type (str): Type of pooling to apply. Default is 'max'.
        apply_antialiasing (bool): Whether to apply antialiasing. Default is True.
        dropout_rate (float): Dropout rate. Default is 0.0.
        layer_kwargs (dict): Additional arguments for layers.
        antialiasing_kwargs (dict): Additional arguments for antialiasing.
        fully_convolutional (bool): Whether the model is fully convolutional. Default is False.

    Returns:
        Gcnn3D: Configured 3D Gcnn model.

    Example:
        >>> # Octahedral → Cycle model
        >>> model = get_3d_model(
        ...     input_channel=1,
        ...     num_channels=[32, 64],
        ...     num_layers=2,
        ...     dwn_group_types=[["octahedral", "cycle"], ["cycle", "cycle"]],
        ...     init_group_order=24,
        ...     spatial_subsampling_factors=[2, 1],
        ...     subsampling_factors=[6, 1]  # 24/4 = 6
        ... )
    """
    # Validate domain
    if domain != 3:
        raise ValueError(f"get_3d_model requires domain=3, got {domain}")

    # Set 3D-specific defaults
    if num_channels is None:
        num_channels = num_layers * [32]
    if kernel_sizes is None:
        kernel_sizes = num_layers * [3]  # 3x3x3 kernels for 3D
    if dwn_group_types is None:
        # Default: no group downsampling (same input/output group)
        dwn_group_types = num_layers * [["octahedral", "octahedral"]]
    if spatial_subsampling_factors is None:
        spatial_subsampling_factors = num_layers * [1]  # Aggressive 3D subsampling
    if subsampling_factors is None:
        # Calculate based on group types
        subsampling_factors = []
        current_order = init_group_order
        for group_pair in dwn_group_types:
            group_type, sub_group_type = group_pair
            if group_type == sub_group_type:
                # Same group type: no subsampling
                subsampling_factors.append(1)
            else:
                # Different group types: calculate based on expected subgroup size
                if group_type == "octahedral" and sub_group_type == "cycle":
                    subsampling_factors.append(6)  # 24/4 = 6
                elif group_type == "full_octahedral" and sub_group_type == "cycle":
                    subsampling_factors.append(12)  # 48/4 = 12
                elif group_type == "full_octahedral" and sub_group_type == "dihedral":
                    subsampling_factors.append(6)  # 48/8 = 6
                elif group_type == "full_octahedral" and sub_group_type == "octahedral":
                    subsampling_factors.append(2)  # 48/24 = 2
                else:
                    subsampling_factors.append(1)  # Default no subsampling
    if layer_kwargs is None:
        layer_kwargs = {"dilation": 1}  # No dilation for 3D by default
    if antialiasing_kwargs is None:
        antialiasing_kwargs = {
            "smooth_operator": "adjacency",
            "mode": "analytical",  # Use faster analytical mode for 3D
            "iterations": 50,  # Reduced iterations for 3D complexity
            "smoothness_loss_weight": 5.0,
            "threshold": 0.0,
            "equi_constraint": True,
            "equi_correction": False,
        }

    # Add input_channel to num_channels for proper indexing
    num_channels = [input_channel] + num_channels
    model = Gcnn3D(
        num_channels=num_channels,
        num_layers=num_layers,
        kernel_sizes=kernel_sizes,
        num_classes=num_classes,
        dwn_group_types=dwn_group_types,
        init_group_order=init_group_order,
        spatial_subsampling_factors=spatial_subsampling_factors,
        subsampling_factors=subsampling_factors,
        domain=domain,
        pooling_type=pooling_type,
        apply_antialiasing=apply_antialiasing,
        dropout_rate=dropout_rate,
        layer_kwargs=layer_kwargs,
        antialiasing_kwargs=antialiasing_kwargs,
        fully_convolutional=fully_convolutional,
    )
    return model
