from escnn import gspaces
import escnn.nn as enn
import torch.nn as nn
from gsampling.utils.group_utils import *


class rnConv(nn.Module):
    def __init__(
        self,
        *,
        in_group_type: str,
        in_order: int,
        in_num_features: int,
        in_representation: str,
        out_group_type: str,
        out_num_features: int,
        out_representation: str,
        domain: int = 2,
        kernel_size: int = 3,
        layer_kwargs: dict = {},
    ):
        """
        This method simply wraps the escnn.R2Conv layer.
        Makes is more convenient to use in the context of group equivariant CNNs. We can pass reglar tensors
        and the layer will convert them to gspace tensors and  perform the convolution.
        And return the regular tensor.

        useful for the case when we want to use the layer as a part of the model that might operates mostly on regular tensors.



        Provides simplified interface for ES-CNN's R2Conv by automatically handling:
        - Geometric tensor conversion (regular tensor ↔ escnn.GeometricTensor)
        - Group space initialization
        - Representation type management

        Args:
            in_group_type (str): Input symmetry group type. Options: 'cyclic', 'dihedral', 'so2', etc.
            in_order (int): Order of finite subgroup (e.g., 4 for C₄ cyclic group)
            in_num_features (int): Number of input channels/features
            in_representation (str): Input representation type. Options: 'regular', 'irreducible', 'trivial'
            out_group_type (str): Output symmetry group type (same options as input)
            out_num_features (int): Number of output channels/features
            out_representation (str): Output representation type (same options as input)
            domain (int, optional): Spatial dimension of operation. Default: 2 (2D convolution)
            kernel_size (int, optional): Convolution kernel size. Default: 3
            layer_kwargs (dict, optional): Additional arguments for ES-CNN R2Conv layer. Default: {}

        Shapes:
            - Input: (batch, in_num_features * group_order, height, width)
            - Output: (batch, out_num_features * group_order, height, width)

        Example:
            >>> conv = rnConv(in_group_type='cycle', in_order=4, in_num_features=32,
                            out_group_type='cycle', out_order=4, out_num_features=64,
                            in_representation='regular', out_representation='regular')
            >>> output = conv(input_tensor)
        """
        super(rnConv, self).__init__()
        self.in_group_type = in_group_type
        self.in_order = in_order
        self.in_num_features = in_num_features
        self.in_representation = in_representation
        self.out_group_type = out_group_type
        self.out_num_features = out_num_features
        self.out_representation = out_representation

        self.G_in = get_group(in_group_type, in_order)
        self.G_out = get_group(out_group_type, in_order)
        self.gspace_in = get_gspace(
            group_type=in_group_type,
            order=in_order,
            num_features=in_num_features,
            representation=in_representation,
        )
        self.gspace_out = get_gspace(
            group_type=out_group_type,
            order=in_order,
            num_features=out_num_features,
            representation=out_representation,
        )

        if domain == 2:
            self.conv = enn.R2Conv(
                in_type=self.gspace_in,
                out_type=self.gspace_out,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                **layer_kwargs,
            )
        else:
            raise ValueError(f"Domain {domain} not found")

    def forward(self, x):
        f_x = self.gspace_in(x)
        f_x_out = self.conv(f_x)
        return f_x_out.tensor

    def get_group(self):
        return self.G

    def get_gspace(self):
        return self.gspace
