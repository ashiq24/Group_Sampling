from escnn import nn
from escnn.group import *
from escnn import gspaces
import torch


class Cannonicalizer(nn.EquivariantModule):
    def __init__(
        self,
        group: str,
        nodes_num: int,
        subgroup: str,
        sub_nodes_num: int,
        in_channels: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """
        method from [1]
        ---------
        1. Jin Xu, Hyunjik Kim, Tom Rainforth, Yee Whye Teh "Group Equivariant Subsampling"
        """
        super().__init__()
        self.group = group
        self.nodes_num = nodes_num
        self.subgroup = subgroup
        self.sub_nodes_num = sub_nodes_num
        self.in_channels = in_channels
        self.dtype = dtype
        self.device = device

        if group == "dihedral":
            assert nodes_num % 2 == 0
            self.gspace = gspaces.flipRot2dOnR2(nodes_num // 2)
            self.feature = nn.FieldType(
                self.gspace, in_channels * [self.gspace.regular_repr]
            )
        elif group == "cycle":
            self.gspace = gspaces.rot2dOnR2(nodes_num)
            self.feature = nn.FieldType(
                self.gspace, in_channels * [self.gspace.regular_repr]
            )
        else:
            raise ValueError("Unknown group : ", group)

        self.buffer = None

    def coset_rep_r2(self, x):
        """
        x: (batch, group * in_channels, h, w)
        """
        fiber = x[:, : self.nodes_num, :, :]
        fiber = torch.permute(fiber, (0, 2, 3, 1)).reshape(x.shape[0], -1)
        v = torch.argmax(fiber, dim=1)
        k, _ = torch.max(fiber, dim=1)

        v = v % self.nodes_num
        if self.group == "dihedral" and self.subgroup == "dihedral":
            v = v % (self.nodes_num // 2)
            v = v % (self.nodes_num // self.sub_nodes_num)
            v = [(0, i) for i in v.tolist()]
        elif self.group == "cycle" and self.subgroup == "cycle":
            v = v % (self.nodes_num // self.sub_nodes_num)
            v = v.tolist()
        elif self.group == "dihedral" and self.subgroup == "cycle":
            r = v // (self.nodes_num // 2)
            v = v % (self.nodes_num // 2)
            v = v % (self.nodes_num // (self.sub_nodes_num * 2))
            v = [(j, i) for (j, i) in zip(r.tolist(), v.tolist())]
        else:
            raise ValueError("Unknown group or subgroup")

        return v

    def coset_rep(self, x):
        """
        x: ( group )
        """
        fiber = x
        v = torch.argmax(fiber)
        v = v % self.nodes_num
        if self.group == "dihedral" and self.subgroup == "dihedral":
            v = v % (self.nodes_num // 2)
            v = v % (self.nodes_num // self.sub_nodes_num)
            v = (0, v.item())
        elif self.group == "cycle" and self.subgroup == "cycle":
            v = v % (self.nodes_num // self.sub_nodes_num)
            v = v.item()
        elif self.group == "dihedral" and self.subgroup == "cycle":
            r = v // (self.nodes_num // 2)
            v = v % (self.nodes_num // 2)
            v = v % (self.nodes_num // (self.sub_nodes_num * 2))
            v = (r.item(), v.item())
        else:
            raise ValueError("Unknown group or subgroup")

        return v

    def forward(self, x, coset_rep=None, mode="forward"):
        if mode == "forward":
            if x.dim() == 4:
                v = self.coset_rep_r2(x)
                self.buffer = v
                result = torch.cat(
                    [
                        self.feature.transform(
                            j.unsqueeze(0),
                            self.feature.fibergroup.element(
                                self.feature.fibergroup._inverse(u)
                            ),
                        )
                        for (j, u) in zip(x, v)
                    ],
                    dim=0,
                )
            elif x.dim() == 1:
                v = self.coset_rep(x)
                self.buffer = v
                result = (
                    torch.tensor(
                        self.feature.fibergroup.regular_representation(
                            self.feature.fibergroup.element(
                                self.feature.fibergroup._inverse(v)
                            )
                        )
                    ).to(device=x.device, dtype=x.dtype)
                    @ x
                )
            assert result.shape == x.shape
            return result, v
        elif mode == "backward":
            if coset_rep is not None:
                v = coset_rep
            elif self.buffer is not None:
                v = self.buffer
            else:
                raise ValueError("coset_rep is None")

            if x.dim() == 4:
                result = torch.cat(
                    [
                        self.feature.transform(
                            j.unsqueeze(0), self.feature.fibergroup.element(u)
                        )
                        for (j, u) in zip(x, v)
                    ],
                    dim=0,
                )
            elif x.dim() == 1:
                result = (
                    torch.tensor(
                        self.feature.fibergroup.regular_representation(
                            self.feature.fibergroup.element(v)
                        )
                    ).to(device=x.device, dtype=x.dtype)
                    @ x
                )

            assert result.shape == x.shape
            return result, v
        else:
            raise ValueError("Unknown mode")


