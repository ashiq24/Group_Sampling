from .g_cnn import Gcnn
from .g_cnn_3d import Gcnn3D
from .g_cnn_3d_seg import Gcnn3DSegmentation


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
    dropout_rate=0.0,
    layer_kwargs=None,
    antialiasing_kwargs=None,
    fully_convolutional=False,
):
    """Creates and returns a 2D Gcnn model."""
    if num_channels is None:
        num_channels = num_layers * [32]
    if kernel_sizes is None:
        kernel_sizes = num_layers * [7]
    if dwn_group_types is None:
        dwn_group_types = num_layers * [["dihedral", "dihedral"]]
    if spatial_subsampling_factors is None:
        spatial_subsampling_factors = num_layers * [1]
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
    init_group_order=24,
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
    """Creates and returns a 3D Gcnn model."""
    if num_channels is None:
        num_channels = [32, 64]
    if kernel_sizes is None:
        kernel_sizes = [3] * num_layers
    if dwn_group_types is None:
        dwn_group_types = [["octahedral", "octahedral"], ["octahedral", "cycle"]][:num_layers]
    if spatial_subsampling_factors is None:
        spatial_subsampling_factors = [1] * num_layers
    if subsampling_factors is None:
        subsampling_factors = [1, 6][:num_layers]
    if layer_kwargs is None:
        layer_kwargs = {}
    if antialiasing_kwargs is None:
        antialiasing_kwargs = {
            "smooth_operator": "adjacency",
            "mode": "analytical",
            "iterations": 50,
            "smoothness_loss_weight": 1.0,
            "threshold": 0.0,
            "equi_constraint": True,
            "equi_correction": False,
        }

    num_channels = [input_channel] + num_channels
    model = Gcnn3D(
        num_layers=num_layers,
        num_channels=num_channels,
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


def get_3d_segmentation_model(
    input_channel=1,
    num_channels=[1, 16, 32],
    num_layers=2,
    kernel_sizes=[3, 3],
    num_classes=2,
    dwn_group_types=[["octahedral", "octahedral"], ["octahedral", "cycle"]],
    init_group_order=24,
    spatial_subsampling_factors=[1, 1],
    subsampling_factors=[1, 6],
    domain=3,
    apply_antialiasing=False,
    antialiasing_kwargs=None,
    dropout_rate=0.1,
):
    """
    Create a 3D Group Equivariant Segmentation model.
    
    Uses existing Gcnn3D encoder + adds decoder with group upsampling.
    """
    if antialiasing_kwargs is None:
        antialiasing_kwargs = {
            "smooth_operator": "adjacency",
            "mode": "analytical",
            "iterations": 50,
            "smoothness_loss_weight": 1.0,
            "threshold": 0.0,
            "equi_constraint": True,
            "equi_correction": False,
        }
    
    return Gcnn3DSegmentation(
        num_layers=num_layers,
        num_channels=num_channels,
        kernel_sizes=kernel_sizes,
        num_classes=num_classes,
        dwn_group_types=dwn_group_types,
        init_group_order=init_group_order,
        spatial_subsampling_factors=spatial_subsampling_factors,
        subsampling_factors=subsampling_factors,
        domain=domain,
        apply_antialiasing=apply_antialiasing,
        antialiasing_kwargs=antialiasing_kwargs,
        dropout_rate=dropout_rate,
    )