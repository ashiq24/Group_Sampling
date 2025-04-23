from gsampling.models.g_cnn import Gcnn


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
        num_channels = [32, 64]
    if kernel_sizes is None:
        kernel_sizes = [7, 7]
    if dwn_group_types is None:
        dwn_group_types = [["dihedral", "dihedral"], ["dihedral", "dihedral"]]
    if spatial_subsampling_factors is None:
        spatial_subsampling_factors = [1, 2]
    if subsampling_factors is None:
        subsampling_factors = [2, 1]
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
