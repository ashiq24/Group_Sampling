import pytest
import torch
from tests.conftest import device_real_dtype_parametrize, tolerance_config
from tests.common_test_utils import (
    STANDARD_2D_GROUP_CONFIGS,
    FAST_ANTI_ALIASING_CONFIG,
    create_test_tensor,
    verify_tensor_shapes,
    test_group_equivariance_basic
)

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

    print(
        f"Testing group type: {group_type}, order: {order}, subgroup type: {sub_group_type}, subsampling factor: {subsampling_factor}, sub_group_order: {G_sub.order()}"
    )

    gspace = get_gspace(group_type=group_type, order=order, num_features=num_features)

    x = create_test_tensor(
        batch_size=32,
        channels=num_features,
        spatial_dims=spatial_size,
        device=downsampling_layer.device,
        dtype=downsampling_layer.dtype,
        group_order=G.order()
    )
    if len(x.shape) == 3:
        x = torch.nn.functional.pad(x, (padding, padding, padding, padding))

    x_sub, _ = downsampling_layer(x)
    print("Data tensor ", x.shape, x_sub.shape)
    x_sub_up = downsampling_layer.upsample(x_sub)

    def test_transform(x_input):
        """Test function for group equivariance testing."""
        x_sub, _ = downsampling_layer(x_input)
        x_sub_up = downsampling_layer.upsample(x_sub)
        return x_sub_up
    
    equivariance_tests_passed = test_group_equivariance_basic(
        input_tensor=x,
        group_type=group_type,
        order=order,
        transform_func=test_transform,
        num_test_elements=3,
        test_name="2D downsampling layer equivariance"
    )
    
    print(f"✅ Group equivariance tests passed: {equivariance_tests_passed}/3")


class TestDownsamplingLayer:
    """Test group downsampling layer functionality."""

    @pytest.mark.parametrize("group_config", STANDARD_2D_GROUP_CONFIGS)
    @device_real_dtype_parametrize
    def test_downsampling_layer_functionality(
        self, group_config, device, dtype
    ):
        """Test downsampling layer with different group configurations."""
        group_type = group_config["group_type"]
        order = group_config["order"]
        sub_group_type = group_config["sub_group_type"]
        subsampling_factor = group_config["subsampling_factor"]
        
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
            anti_aliasing_kwargs=FAST_ANTI_ALIASING_CONFIG,
            cannonicalize=False,
        )
        layer_tester_rn(downsampling_layer=d_layer)


