import torch
import torch.nn as nn
from gsampling.layers.rnconv import rnConv
from gsampling.layers.downsampling import SubgroupDownsample


class HybridConvGroupResample3D(nn.Module):
    """
    Hybrid layer: rnConv (keeps in/out group orders the same) + group resampling.

    - If out_group_order == in_group_order: applies only rnConv.
    - If out_group_order < in_group_order: applies rnConv then subgroup downsampling (forward).
    - If out_group_order > in_group_order: applies rnConv then subgroup upsampling (upsample()).
    """

    def __init__(
        self,
        *,
        in_group_type: str,
        in_group_order: int,
        in_num_features: int,
        out_group_type: str,
        out_group_order: int,
        out_num_features: int,
        representation: str = "regular",
        kernel_size: int = 3,
        domain: int = 3,
        apply_antialiasing: bool = False,
        anti_aliasing_kwargs: dict | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.in_group_type = in_group_type
        self.in_group_order = in_group_order
        self.out_group_type = out_group_type
        self.out_group_order = out_group_order
        self.apply_antialiasing = apply_antialiasing
        self.anti_aliasing_kwargs = anti_aliasing_kwargs or {}

        # We'll build two convs depending on direction to keep orders equal across each conv
        self.conv_increase = None
        self.conv_decrease = None

        # Prepare resampler if orders differ
        self.resampler: SubgroupDownsample | None = None
        if out_group_order != in_group_order:
            if out_group_order < in_group_order:
                # Downsample in group space
                subsampling_factor = in_group_order // out_group_order
                self.resampler = SubgroupDownsample(
                    group_type=in_group_type,
                    order=in_group_order,
                    sub_group_type=out_group_type,
                    subsampling_factor=subsampling_factor,
                    num_features=out_num_features,
                    generator="r-s",
                    device=device,
                    dtype=dtype,
                    sample_type="sample",
                    apply_antialiasing=apply_antialiasing,
                    anti_aliasing_kwargs=self.anti_aliasing_kwargs,
                )
                # Conv at higher (input) order
                self.conv_decrease = rnConv(
                    in_group_type=in_group_type,
                    in_order=in_group_order,
                    in_num_features=in_num_features,
                    in_representation=representation,
                    out_group_type=in_group_type,
                    out_num_features=out_num_features,
                    out_representation=representation,
                    domain=domain,
                    kernel_size=kernel_size,
                ).to(device=device, dtype=dtype)
            else:
                # Upsample in group space
                upsampling_factor = out_group_order // in_group_order
                self.resampler = SubgroupDownsample(
                    group_type=out_group_type,
                    order=out_group_order,
                    sub_group_type=in_group_type,
                    subsampling_factor=upsampling_factor,
                    num_features=out_num_features,
                    generator="r-s",
                    device=device,
                    dtype=dtype,
                    sample_type="sample",
                    apply_antialiasing=apply_antialiasing,
                    anti_aliasing_kwargs=self.anti_aliasing_kwargs,
                )
                # Conv at higher (output) order
                self.conv_increase = rnConv(
                    in_group_type=out_group_type,
                    in_order=out_group_order,
                    in_num_features=in_num_features,
                    in_representation=representation,
                    out_group_type=out_group_type,
                    out_num_features=out_num_features,
                    out_representation=representation,
                    domain=domain,
                    kernel_size=kernel_size,
                ).to(device=device, dtype=dtype)
        else:
            # No resampling, single conv at this order
            self.conv_decrease = rnConv(
                in_group_type=in_group_type,
                in_order=in_group_order,
                in_num_features=in_num_features,
                in_representation=representation,
                out_group_type=in_group_type,
                out_num_features=out_num_features,
                out_representation=representation,
                domain=domain,
                kernel_size=kernel_size,
            ).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Debug prints
        # print(f"Hybrid forward: in_order={self.in_group_order}, out_order={self.out_group_order}, x_ch={x.shape[1]}")
        if self.resampler is None:
            # print("Path: no-resample -> conv_decrease")
            return self.conv_decrease(x)
        # Decrease order: conv first at high order, then downsample
        if self.out_group_order < self.in_group_order:
            # print("Path: decrease -> conv_decrease -> downsample")
            x = self.conv_decrease(x)
            return self.resampler(x)
        # Increase order: upsample first, then conv at high order
        # print("Path: increase -> upsample -> conv_increase")
        x = self.resampler.upsample(x)
        return self.conv_increase(x)


