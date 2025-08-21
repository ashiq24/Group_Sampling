import pytest
import torch
from tests.conftest import device_real_dtype_parametrize, tolerance_config

# Import the modules under test
try:
    from escnn.group import *
    from gsampling.layers.downsampling import SubgroupDownsample
    from gsampling.utils.group_utils import *
except ImportError as e:
    pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)


def layer_tester_rn(
    *,
    downsampling_layer: SubgroupDownsample,
    spatial_size: list = [32, 32],
    padding: int = 10,
):

    group_type = downsampling_layer.group_type
    order = downsampling_layer.order
    sub_group_type = downsampling_layer.sub_group_type
    subsampling_factor = downsampling_layer.subsampling_factor
    num_features = downsampling_layer.num_features
    sub_order = (
        order // subsampling_factor
        if group_type == sub_group_type
        else order // max(subsampling_factor // 2, 1)
    )

    G = get_group(group_type, order)
    G_sub = get_group(sub_group_type, sub_order)

    assert group_type == "dihedral" or group_type == "cycle"

    # print group types and order
    print(
        f"Testing group type: {group_type}, order: {order}, subgroup type: {sub_group_type}, subsampling factor: {subsampling_factor}, sub_group_order: {G_sub.order()}"
    )

    gspace = get_gspace(group_type=group_type, order=order, num_features=num_features)

    x = torch.randn(32, G.order() * num_features, *spatial_size).to(
        downsampling_layer.device, dtype=downsampling_layer.dtype
    )
    if len(x.shape) == 3:
        x = torch.nn.functional.pad(x, (padding, padding, padding, padding))

    x_sub, _ = downsampling_layer(x)
    print("Data tensor ", x.shape, x_sub.shape)
    x_sub_up = downsampling_layer.upsample(x_sub)

    for g in G.elements:
        x_t = gspace.transform(x.clone(), g)
        x_t_sub, _ = downsampling_layer(x_t)
        x_t_sub_up = downsampling_layer.upsample(x_t_sub)

        assert x_t.shape == x.shape
        assert x_t_sub.shape == x_sub.shape
        assert x_t_sub_up.shape == x_sub_up.shape


class TestDownsamplingLayer:
    """Test group downsampling layer functionality."""

    @pytest.mark.parametrize("group_type,order,sub_group_type,subsampling_factor", [
        ("dihedral", 12, "dihedral", 2),
        ("dihedral", 8, "cycle", 2),
        ("cycle", 8, "cycle", 2),
    ])
    @device_real_dtype_parametrize
    def test_downsampling_layer_functionality(
        self, group_type, order, sub_group_type, subsampling_factor, device, dtype
    ):
        """Test downsampling layer with different group configurations."""
        print(f"*****Testing {group_type}->{sub_group_type} Downsampling Layer******")
        
        d_layer = SubgroupDownsample(
            group_type=group_type,
            order=order,
            sub_group_type=sub_group_type,
            subsampling_factor=subsampling_factor,
            num_features=10,
            generator="r-s",
            device=device,
            dtype=dtype,
            sample_type="sample",
            apply_antialiasing=True,
            anti_aliasing_kwargs={
                "smooth_operator": "adjacency",
                "mode": "analytical",  # Use faster analytical mode for tests
                "iterations": 100,
                "smoothness_loss_weight": 1.0,
                "threshold": 0.0,
                "equi_constraint": True,
                "equi_correction": False,
            },
            cannonicalize=False,
        )
        layer_tester_rn(downsampling_layer=d_layer)

    # def test_original_configuration(self):
    #     """Test the original configuration for backward compatibility."""
    #     print("*****Testing Original Group Downsampling Layer******")
    #     d_layer = SubgroupDownsample(
    #         group_type="dihedral",
    #         order=12,
    #         sub_group_type="dihedral",
    #         subsampling_factor=2,
    #         num_features=10,
    #         generator="r-s",
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         dtype=torch.float32,
    #         sample_type="sample",
    #         apply_antialiasing=True,
    #         anti_aliasing_kwargs={
    #             "smooth_operator": "adjacency",
    #             "mode": "analytical",  # Use faster mode for testing
    #             "iterations": 100,
    #             "smoothness_loss_weight": 5.0,
    #             "threshold": 0.0,
    #             "equi_constraint": True,
    #             "equi_correction": False,
    #         },
    #         cannonicalize=False,
    #     )
    #     layer_tester_rn(downsampling_layer=d_layer)
