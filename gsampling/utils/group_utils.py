from escnn.group import *
from escnn import gspaces
import escnn.nn as enn
import torch
import torch.nn as nn


def get_group(group_type: str, order: int):
    """
    group_type: str : name of the group
    order: int : order of the group
    """
    if group_type == "dihedral":
        return dihedral_group(order)
    elif group_type in ["cycle", "cyclic"]:
        return cyclic_group(order)
    elif group_type == "trivial":
        return trivial_group(order)
    else:
        raise ValueError(f"Group type {group_type} not found")


def get_gspace(
    *, group_type: str, order: int, num_features: int, representation: str = "regular"
):
    """
    group_type: str : name of the group
    order: int : order of the group
    """
    if group_type == "dihedral":
        gspace = gspaces.flipRot2dOnR2(order)
        if representation == "regular" or representation == None:
            g_feature = enn.FieldType(gspace, num_features * [gspace.regular_repr])
        elif representation == "trivial":
            g_feature = enn.FieldType(gspace, num_features * [gspace.trivial_repr])
        else:
            raise ValueError(f"Representation {representation} not found")
    elif group_type == "cycle":
        gspace = gspaces.rot2dOnR2(order)
        if representation == "regular" or representation == None:
            g_feature = enn.FieldType(gspace, num_features * [gspace.regular_repr])
        elif representation == "trivial":
            g_feature = enn.FieldType(gspace, num_features * [gspace.trivial_repr])
        else:
            raise ValueError(f"Representation {representation} not found")
    else:
        raise ValueError(f"Group type {group_type} not found")
    return g_feature


def get_sub_group_element(element, group_type, sub_group_type, subsampling_factor):
    """
    group_type: str : name of the group
    sub_group_type: str : name of the subgroup
    order: int : order of the group
    subsampling_factor: int : subsampling factor
    """
    if group_type == "dihedral" and sub_group_type == "dihedral":
        f, r = element._element
        return (f, r // subsampling_factor)
    elif group_type == "cycle" and sub_group_type == "cycle":
        return element._element // subsampling_factor
    elif group_type == "dihedral" and sub_group_type == "cycle":
        return element._element[1] // max(subsampling_factor // 2, 1)
    else:
        raise ValueError(f"Group type {group_type} not found")
