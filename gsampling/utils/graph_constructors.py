import numpy as np
from escnn.group import *
from ..core.graphs.factory import GroupGraphFactory


def subsample(
    group_size: int,
    group_type: str,
    group_generator: str,
    subgroup_type: str,
    subsampling_factor: int,
):
    nodes = [i for i in range(group_size)]
    if group_type == "dihedral":
        if subgroup_type == "dihedral":
            assert group_generator == "r-s"
            sub_sample_nodes = nodes[::subsampling_factor]
        elif subgroup_type == "adihedral":
            assert group_generator == "r-s"
            sub_sample_nodes = (
                nodes[: group_size // 2 : subsampling_factor]
                + nodes[group_size // 2 + 1 :: subsampling_factor]
            )
        elif subgroup_type == "cycle":
            assert subsampling_factor % 2 == 0
            assert (group_size // 2) % (subsampling_factor // 2) == 0
            subsampling_factor = subsampling_factor // 2
            nodes = nodes[: group_size // 2]
            sub_sample_nodes = nodes[::subsampling_factor]
        else:
            raise NotImplementedError
    elif group_type == "cycle":
        sub_sample_nodes = nodes[::subsampling_factor]
    else:
        raise NotImplementedError
    return sub_sample_nodes


class GraphConstructor:
    def __init__(
        self,
        group_size: int,
        group_type: str,
        group_generator: str,
        subgroup_type: str,
        subsampling_factor: int,
    ):
        """Constructs Cayley graphs for finite groups and their subgroups.

        Parameters:
            group_size (int): Order of main group |G|
            group_type (str): 'dihedral' or 'cycle'
            group_generator (str): Cayley graph generator convention
            subgroup_type (str): Type of subgroup H ≤ G
            subsampling_factor (int): subsampling factor

        Methods:
            __init__: Builds G and H graphs via Cayley constructions
        """
        self.group_size = group_size
        self.group_type = group_type
        self.group_generator = group_generator

        assert group_size % subsampling_factor == 0
        self.subgroup_size = group_size // subsampling_factor
        self.subsampling_factor = subsampling_factor

        self.subgroup_type = subgroup_type
        self._subgroup_type = ["cycle", "dihedral", "cyclic"]

        self.nodes = [i for i in range(group_size)]
        
        # Use factory to create group graph (enables extension to new groups)
        self.graph = GroupGraphFactory.create(self.group_type, self.nodes, group_generator)

        ## Sub group Sampling algorithm

        subsampled_nodes = subsample(
            self.group_size,
            self.group_type,
            self.group_generator,
            self.subgroup_type,
            self.subsampling_factor,
        )

        # Use factory to create subgroup graph (enables extension to new groups)
        self.subgroup_graph = GroupGraphFactory.create(self.subgroup_type, subsampled_nodes, group_generator)


# Legacy DihedralGraph and CycleGraph classes moved to gsampling/core/graphs/
# Import them for backward compatibility
from ..core.graphs.dihedral import DihedralGraph
from ..core.graphs.cyclic import CycleGraph

# The old implementations below are deprecated and will be removed
# TODO: Remove these after ensuring all imports are updated

class _LegacyDihedralGraph:
    """
    nodes starts from 0
    nodes are connecte to elemmets of Dihedral group
    Only the graph is changed based on the generator
    """

    def __init__(self, nodes: list, generator: str):

        assert generator in ["r-s", "s-sr"]
        group_size = len(nodes)
        self.nodes = nodes
        assert group_size % 2 == 0
        self._init_edges(group_size, generator)

        # calculated dyhedra Fourier basis
        self.fourier_basis = self.get_basis(group_size // 2)

        self.equi_raynold_op = self.get_equi_raynold(group_size // 2)

    def _init_edges(self, group_size: int, generator: str):
        """Initialize edges and adjacency matrices based on the generator."""

        self.edges = []
        self.edges_generator_1 = []
        self.edges_generator_2 = []
        if generator == "r-s":
            for i in range(group_size // 2):
                edge = (i, (i + 1) % (group_size // 2))
                self.edges.append(edge)
                self.edges_generator_1.append(edge)

            for i in range(group_size // 2):
                edge = (
                    group_size // 2 + i,
                    group_size // 2 + ((i + 1) % (group_size // 2)),
                )
                self.edges.append(
                    (
                        group_size // 2 + i,
                        group_size // 2 + ((i + 1) % (group_size // 2)),
                    )
                )
                self.edges_generator_1.append(edge)

            for i in range(group_size // 2):
                edge = (i, i + group_size // 2)
                self.edges.append(edge)
                self.edges_generator_2.append(edge)

            self.adjacency_matrix = np.zeros((group_size, group_size))
            for edge in self.edges:
                self.adjacency_matrix[edge[0], edge[1]] = 1
                self.adjacency_matrix[edge[1], edge[0]] = 1

            # construct directed adjacency matrix from edges for generator 1
            self.directed_adjacency_matrix = np.zeros((group_size, group_size))
            for edge in self.edges:
                self.directed_adjacency_matrix[edge[0], edge[1]] = 1

            # construct directed adjacency matrix from edges for generator 2
            self.directed_adjacency_matrix_generator_1 = np.zeros(
                (group_size, group_size)
            )
            for edge in self.edges_generator_1:
                self.directed_adjacency_matrix_generator_1[edge[0], edge[1]] = 1

            self.directed_adjacency_matrix_generator_2 = np.zeros(
                (group_size, group_size)
            )
            for edge in self.edges_generator_2:
                self.directed_adjacency_matrix_generator_2[edge[0], edge[1]] = 1
                self.directed_adjacency_matrix_generator_2[edge[1], edge[0]] = 1

            # concatenate the directed adjacency matrix for generator 1 and generator 2
            self.smoother = np.concatenate(
                (
                    self.directed_adjacency_matrix_generator_1,
                    self.directed_adjacency_matrix_generator_2,
                ),
                axis=0,
            )

        else:
            r = self.nodes[: group_size // 2]
            sr = self.nodes[group_size // 2 :]
            sr = sr[:1] + sr[:0:-1]
            cycle_node = []

            for i in range(group_size // 2):
                cycle_node.append(r[i])
                cycle_node.append(sr[i])
            # make a cycle graph from cycle nodes
            for i in range(group_size):
                self.edges.append((cycle_node[i], cycle_node[(i + 1) % group_size]))

            # make directed adjacency matrix
            self.smoother = np.zeros((group_size, group_size))
            for edge in self.edges:
                self.smoother[edge[0], edge[1]] = 1

        # construct the adjacency matrix
        self.adjacency_matrix = np.zeros((group_size, group_size))
        for edge in self.edges:
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1

        self.directed_adjacency_matrix = np.zeros((group_size, group_size))
        for edge in self.edges:
            self.directed_adjacency_matrix[edge[0], edge[1]] = 1

    def get_basis(self, order: int):
        """
        Constructs unitary Fourier basis from irreps.
        order: int: order of the Dihedral group Dₙ. Half of the number of nodes.
        """
        G = dihedral_group(order)
        basis = []
        for rho in G.irreps():
            d = rho.size**0.5
            rho_g = np.stack([rho(g) for g in G.elements], axis=0)
            # this following reshape is vital
            rho_g = (
                np.moveaxis(rho_g, -2, -1).reshape(rho_g.shape[0], -1)
                * d
                / (2 * order) ** 0.5
            )
            basis.append(rho_g)
        return np.concatenate(basis, axis=1)

    def get_irreps(self, order: int):
        """
        directsum of irreps for Dₙ group.
        """
        G = dihedral_group(order)
        g_dir_sum = []
        for rho in G.irreps():
            d = rho.size
            for i in range(d):
                g_dir_sum.append(rho)
        return directsum(g_dir_sum)

    def get_equi_raynold(self, order: int):
        """
        Computes Reynolds operator for Dₙ.

        Action/ representation: directsum of irreps "ρ_Irrep⊕". (action on Fourier Coefficients)

        R = 1/(2n) Σ_{g∈Dₙ} ρ_Irrep⊕(g)⊗ρ_Irrep⊕(g^{-1})^T
        """
        G = dihedral_group(order)
        k = self.get_irreps(order)
        size = G.regular_representation.change_of_basis.shape[-1]
        equi_rey = np.zeros(size**2)
        for i in G.elements:
            equi_rey = equi_rey + np.kron(k(i), k(i.__invert__()).T)
        equi_rey = equi_rey / (2 * order)
        return equi_rey


class _LegacyCycleGraph:
    def __init__(self, nodes: list, generator: str = None):
        self.nodes = nodes
        group_size = len(nodes)
        self.edges = []
        for i in range(group_size):
            self.edges.append((i, (i + 1) % group_size))

        # construct the adjacency matrix
        self.adjacency_matrix = np.zeros((group_size, group_size))
        for edge in self.edges:
            self.adjacency_matrix[edge[0], edge[1]] = 1
            self.adjacency_matrix[edge[1], edge[0]] = 1

        self.directed_adjacency_matrix = np.zeros((group_size, group_size))
        for edge in self.edges:
            self.directed_adjacency_matrix[edge[0], edge[1]] = 1

        self.smoother = self.directed_adjacency_matrix

        # calclate cyclic Fourier basis
        self.fourier_basis = self.get_basis(group_size)
        self.equi_raynold_op = self.get_equi_raynold(group_size)

    def get_basis(self, order: int):
        """
        returns the unitary Fourier basis from irreps for cyclic group
        """
        G = cyclic_group(order)
        return G.regular_representation.change_of_basis

    def get_irreps(self, order: int):
        """
        returns directsum of irreps for Cₙ group.
        """
        G = cyclic_group(order)
        g_dir_sum = []
        for rho in G.irreps():
            g_dir_sum.append(rho)
        return directsum(g_dir_sum)

    def get_equi_raynold(self, order: int):
        """
        returns the Reynolds operator for Cₙ equivariant projections.

        """
        G = cyclic_group(order)
        k = self.get_irreps(order)
        size = G.regular_representation.change_of_basis.shape[-1]
        equi_rey = np.zeros(size**2)
        for i in G.elements:
            equi_rey = equi_rey + np.kron(k(i), k(i.__invert__()).T)
        equi_rey = equi_rey / (order)
        return equi_rey