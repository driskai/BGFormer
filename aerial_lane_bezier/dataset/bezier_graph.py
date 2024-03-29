from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from torch_geometric.data import Data


class BezierGraph(nx.DiGraph):
    """
    Class for storing a Bezier graph.

    Inherits from nx.DiGraph, can be used in the same way. Adds methods for realising
    edge parameters into Bezier curves, and for plotting. Note we store log distances
    on the edges to avoid negative edge lengths during the JAX bezier optimisation -
    however this can be turned off at both PyG import and export, see the
    assume_log_distances and return_log_distances parameters.
    """

    @classmethod
    def from_pyg_graph(cls, data: Data, assume_log_distances: bool = False):
        return cls.from_graph_data(
            data.x.cpu().numpy(),
            data.edge_index.cpu().numpy() if data.edge_index is not None else None,
            data.edge_attr.cpu().numpy(),
            assume_log_distances=assume_log_distances,
        )

    @classmethod
    def from_graph_data(
        cls,
        x: np.array,
        edge_index: Optional[np.array],
        edge_attr: Optional[np.array],
        assume_log_distances: bool = False,
    ):
        graph = cls()

        # Add nodes
        for i, x_ in enumerate(x):
            graph.add_node(i, pos=x_[:2], direction=x_[2:])

        # Add edges
        if edge_index is not None and len(edge_index) > 0:
            for (u, v), attr in zip(edge_index.T, edge_attr):
                if assume_log_distances:
                    log_l1 = attr[0]
                    log_l2 = attr[1]
                else:
                    log_l1 = np.log(attr[0])
                    log_l2 = np.log(attr[1])
                graph.add_edge(
                    u,
                    v,
                    log_l1=log_l1,
                    log_l2=log_l2,
                )

        return graph

    def get_bezier_control_points_from_edge(
        self,
        u_index: int,
        v_index: int,
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Get the Bezier control points for an edge.

        Parameters
        ----------
        u_index : int
            Index of the start node of the edge.
        v_index : int
            Index of the end node of the edge.

        Returns
        -------
        Tuple[np.array, np.array, np.array, np.array]
            Tuple of the four Bezier control points.
        """
        P0 = self.nodes[u_index]["pos"]
        P3 = self.nodes[v_index]["pos"]

        log_l1 = self.edges[u_index, v_index]["log_l1"]
        d1 = np.exp(log_l1)

        unit_P1_direction = self.nodes[u_index]["direction"]
        unit_P2_direction = self.nodes[v_index]["direction"]
        P1 = P0 + d1 * unit_P1_direction

        log_l2 = self.edges[u_index, v_index]["log_l2"]
        d2 = np.exp(log_l2)
        P2 = P3 - d2 * unit_P2_direction

        return P0, P1, P2, P3

    def interpolate_points_along_bezier(
        self,
        P0: np.array,
        P1: np.array,
        P2: np.array,
        P3: np.array,
        num_points: int = 1000,
    ) -> np.array:
        """
        Interpolate points along a Bezier curve.

        Parameters
        ----------
        P0 : np.array
            First control point.
        P1 : np.array
            Second control point.
        P2 : np.array
            Third control point.
        P3 : np.array
            Fourth control point.
        num_points : int, optional
            Number of points to interpolate, by default 100

        Returns
        -------
        np.array
            Array of interpolated points.
        """
        ts = np.linspace(0, 1, num_points)[:-1]
        bezier_points = (
            np.outer((1 - ts) ** 3, P0)
            + np.outer(3 * (1 - ts) ** 2 * ts, P1)
            + np.outer(3 * (1 - ts) * ts**2, P2)
            + np.outer(ts**3, P3)
        )
        return bezier_points

    def to_simple(
        self,
        nodes_per_m: float = 0.5,
    ) -> nx.DiGraph:
        """Create a simple graph representation by interpolating the original."""
        G = nx.DiGraph()
        for idx, data in self.nodes(data=True):
            G.add_node(idx, pos=data["pos"])
        for u, v in self.edges():
            P0, P1, P2, P3 = self.get_bezier_control_points_from_edge(u, v)
            pts = self.interpolate_points_along_bezier(P0, P1, P2, P3, num_points=10000)
            length = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
            num_points = int(np.round(np.maximum(2, length * nodes_per_m)))
            bezier_points = self.interpolate_points_along_bezier(
                P0, P1, P2, P3, num_points=num_points
            )
            if len(bezier_points) <= 2:
                G.add_edge(u, v, weight=np.linalg.norm(P3 - P0))
                continue
            prev_idx = u
            dist = np.linalg.norm(np.diff(bezier_points, axis=0), axis=1)
            for idx, ((x, y), d) in enumerate(zip(bezier_points[1:], dist)):
                if idx == 0:
                    continue
                if idx == len(bezier_points) - 2:
                    new_idx = v
                else:
                    new_idx = G.number_of_nodes()
                    G.add_node(new_idx, pos=np.array([x, y]))
                G.add_edge(prev_idx, new_idx, weight=d)
                prev_idx = new_idx
        return G

    def to_poly(self, lane_width: float = 5.0) -> Polygon:
        """Convert to a Shapely polygon."""
        lanes = []
        for u, v in self.edges():
            P0, P1, P2, P3 = self.get_bezier_control_points_from_edge(u, v)
            length = (
                np.linalg.norm(
                    [[P3 - P0], [P0 - P1], [P2 - P1], [P3 - P2]], axis=1
                ).sum()
                / 2
            )
            num_points = int(np.maximum(2, length * 0.33))
            bezier_points = self.interpolate_points_along_bezier(
                P0, P1, P2, P3, num_points=num_points
            )
            if len(bezier_points) <= 2:
                bezier_points = np.array([P0, P3])
            line = LineString(bezier_points)
            lanes.append(line.buffer(lane_width / 2))
        return unary_union(lanes)

    def rescale(self, scaling_factor: float):
        """
        Rescale the Bezier graph.

        Parameters
        ----------
        scaling_factor : float
            Scaling factor to apply.

        Returns
        -------
        BezierGraph
            Rescaled Bezier graph.
        """
        rescaled_graph = self.copy()
        # Scaling the nodes is simple. Note the direction vectors are normalised and
        # should not change - we will instead modify the distances on the edges.
        for node in rescaled_graph.nodes():
            rescaled_graph.nodes[node]["pos"] *= scaling_factor
        # Scaling the edges is more complicated. Note that the distances are stored as
        # log distances, so we need to exponentiate them.
        for u, v in rescaled_graph.edges():
            for key in ["log_l1", "log_l2"]:
                new_log_l = rescaled_graph.edges[u, v][key] + np.log(scaling_factor)
                rescaled_graph.edges[u, v][key] = new_log_l
        return rescaled_graph

    def to_pyg_graph(self, return_log_distances: bool = False) -> Data:
        """
        Convert the Bezier graph to a PyG graph.

        Returns
        -------
        PyG graph
            PyG graph representation of the Bezier graph.
        """
        pos_list = []
        direction_list = []
        node_index_to_pyg_index = {}
        for i, (n, data) in enumerate(self.nodes(data=True)):
            pos_list.append(data["pos"])
            direction_list.append(data["direction"])

            node_index_to_pyg_index[n] = i

        pos_tensor = torch.tensor(np.array(pos_list), dtype=torch.float)
        direction_tensor = torch.tensor(np.array(direction_list), dtype=torch.float)

        # Extract edge indices and attributes
        edge_index = []
        l1_list = []
        l2_list = []
        for u, v, data in self.edges(data=True):
            u = node_index_to_pyg_index[u]
            v = node_index_to_pyg_index[v]
            edge_index.append((u, v))
            log_l1 = data["log_l1"]
            log_l2 = data["log_l2"]
            if return_log_distances:
                return_l1 = log_l1
                return_l2 = log_l2
            else:
                return_l1 = np.exp(log_l1)
                return_l2 = np.exp(log_l2)
            l1_list.append(return_l1)
            l2_list.append(return_l2)

        # Edge attributes to tensors
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        l1_tensor = torch.tensor(l1_list, dtype=torch.float)
        l2_tensor = torch.tensor(l2_list, dtype=torch.float)

        # Create PyTorch Geometric data
        data = Data(
            x=torch.cat([pos_tensor, direction_tensor], dim=1),
            edge_index=edge_index_tensor,
            edge_attr=torch.stack([l1_tensor, l2_tensor], dim=1),
        )

        return data

    def plot_single_bezier(
        self,
        axis: plt.Axes,
        u: int,
        v: int,
        bezier_colour: str = "r",
        p1_colour: str = "b",
        p2_colour: str = "g",
    ):
        """
        Plot a single Bezier curve on an axis.

        Parameters
        ----------
        axis : plt.Axes
            Axis on which to plot.
        u : int
            Index of the start node of the edge.
        v : int
            Index of the end node of the edge.
        bezier_colour : str, optional
            Colour to plot the Bezier curve, by default "r"
        p1_colour : str, optional
            Colour to plot the first control point, by default "b"
        p2_colour : str, optional
            Colour to plot the second control point, by default "g"
        """
        P0, P1, P2, P3 = self.get_bezier_control_points_from_edge(u, v)
        bezier_points = self.interpolate_points_along_bezier(P0, P1, P2, P3)
        axis.plot(bezier_points[:, 0], bezier_points[:, 1], c=bezier_colour)

        axis.scatter([P1[0]], [P1[1]], c=p1_colour, s=1)
        axis.scatter([P2[0]], [P2[1]], c=p2_colour, s=1)

    def plot(
        self,
        axis: plt.Axes,
    ):
        """
        Plot the Bezier graph on an axis.

        Parameters
        ----------
        axis : plt.Axes
            Axis on which to plot.
        """
        for u, v in self.edges():
            self.plot_single_bezier(axis, u, v)

        # Plot big node positions
        node_positions = nx.get_node_attributes(self, "pos")
        node_positions = np.array(list(node_positions.values()))
        if len(node_positions) > 0:
            axis.scatter(
                *node_positions.T,
                s=100,
                marker="x",
                c="g",
            )
