import torch
import torch.nn as nn
from einops import rearrange


class SamplingLayer(nn.Module):
    def __init__(
        self,
        *,
        sampling_factor: int = 2,
        nodes: list = None,
        subsample_nodes: list = None,
        type: str = "sample"
    ):
        """
        Perform subsampling operation from group to subgroup.
        Args:
            sampling_factor (int): Subsampling factor.
            nodes (list): List of nodes in the group. Nodes are represented as indices in the adjacency matrix.
            subsample_nodes (list): List of nodes in the subgroup.
            type (str): Type of sampling operation. Can be 'sample' or 'pool'.
        """

        super().__init__()
        self.sampling_factor = sampling_factor
        self.nodes = nodes
        self.subsample_nodes = subsample_nodes
        self.type = type
        self.sampled_nodes = None

        if self.type == "sample":
            sampling_matrix = torch.zeros(len(self.subsample_nodes), len(self.nodes))
            for i, node in enumerate(self.subsample_nodes):
                sampling_matrix[i, node] = 1

            up_sampling_matrix = torch.linalg.pinv(sampling_matrix)

            self.register_buffer(
                "sampling_matrix", torch.tensor(sampling_matrix).clone()
            )
            self.register_buffer(
                "up_sampling_matrix", torch.tensor(up_sampling_matrix).clone()
            )
        elif self.type == "pool":
            self.pool = nn.MaxPool1d(
                kernel_size=self.sampling_factor, stride=self.sampling_factor
            )
        else:
            raise ValueError("Unknown type")

    def forward(self, x):
        if self.type == "sample":
            if len(x.shape) == 1:
                return self.sampling_matrix @ x
            elif len(x.shape) == 4:
                x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.nodes))
                x = torch.einsum("fg,bcghw->bcfhw", self.sampling_matrix, x)
                x = rearrange(x, "b  c g h w -> b (c g) h w")
                return x
            elif len(x.shape) == 5:
                x = rearrange(x, "b (c g) h w d -> b c g h w d", g=len(self.nodes))
                x = torch.einsum("fg,bcghwd->bcfhwd", self.sampling_matrix, x)
                x = rearrange(x, "b  c g h w d -> b (c g) h w d")
                return x
        elif self.type == "pool":
            x = x.unsqueeze(0)
            x = self.pool(x)
            return x.squeeze(0)
        else:
            raise ValueError("Unknown type")

    def up_sample(self, x):
        if len(x.shape) == 4:
            x = rearrange(x, "b (c g) h w -> b c g h w", g=len(self.subsample_nodes))
            x_upsampled = torch.einsum("fg,bcghw->bcfhw", self.up_sampling_matrix, x)
        elif len(x.shape) == 5:
            x = rearrange(x, "b (c g) h w d -> b c g h w d", g=len(self.subsample_nodes))
            x_upsampled = torch.einsum("fg,bcghwd->bcfhwd", self.up_sampling_matrix, x)
        else:
            x_upsampled = self.up_sampling_matrix @ x
        if len(x.shape) == 5:
            x_upsampled = rearrange(x_upsampled, "b c g h w -> b (c g) h w")
        elif len(x.shape) == 6:
            x_upsampled = rearrange(x_upsampled, "b c g h w d -> b (c g) h w d")
        return x_upsampled
