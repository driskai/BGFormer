import os
from copy import deepcopy
from dataclasses import dataclass
from time import time
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optax
import pandas as pd
from jax import jit
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import medfilt
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm

from aerial_lane_bezier.dataset.bezier_graph import BezierGraph

# Do not change - Urban Lane Graph dataset parameter. There are 15cm per pixel.
M_PER_PIXEL = 0.15


class NoBezierPathsError(Exception):
    pass


@dataclass
class BezierOptimisationMetrics:
    mean_hausdorff_distance: float
    max_hausdorff_distance: float
    sum_hausdorff_distance: float
    num_hausdorff_distances: int
    num_source_graph_nodes: int
    num_bezier_graph_nodes: int
    num_source_graph_edges: int
    num_bezier_graph_edges: int
    source_graph: nx.DiGraph
    bezier_graph: BezierGraph

    @classmethod
    def _compute_metrics(
        cls,
        optimised_bezier_graph: BezierGraph,
        source_graph: nx.DiGraph,
        edge_indices_to_xy: Dict[Tuple[int, int], np.ndarray],
        edge_indices_to_mask: Dict[Tuple[int, int], np.ndarray],
    ):
        # We'll use the Hausdorff distance as a metric for how well the optimised bezier
        # graph matches the ground truth bezier graph
        hausdorff_distances = []
        edge_indices = []
        for u, v in optimised_bezier_graph.edges():
            P0, P1, P2, P3 = optimised_bezier_graph.get_bezier_control_points_from_edge(
                u, v
            )
            bezier_points = optimised_bezier_graph.interpolate_points_along_bezier(
                P0, P1, P2, P3
            )
            ground_truth_points = edge_indices_to_xy[(u, v)][
                edge_indices_to_mask[(u, v)]
            ]
            hausdorff_distances.append(
                directed_hausdorff(ground_truth_points, bezier_points)[0]
            )
            edge_indices.append((u, v))
        hausdorff_distances = np.array(hausdorff_distances)

        return cls(
            mean_hausdorff_distance=hausdorff_distances.mean(),
            max_hausdorff_distance=hausdorff_distances.max(),
            sum_hausdorff_distance=hausdorff_distances.sum(),
            num_hausdorff_distances=len(hausdorff_distances),
            num_source_graph_nodes=source_graph.number_of_nodes(),
            num_bezier_graph_nodes=optimised_bezier_graph.number_of_nodes(),
            num_source_graph_edges=source_graph.number_of_edges(),
            num_bezier_graph_edges=optimised_bezier_graph.number_of_edges(),
            source_graph=source_graph,
            bezier_graph=optimised_bezier_graph,
        )

    @classmethod
    def metrics_from_batch(
        cls,
        batched_optimised_graphs: BezierGraph,
        g2b_object,
        batch_index_to_bezier_indices: Dict[int, int],
        batch_index_to_source_indices: Dict[int, int],
    ) -> list:
        metric_objects = []
        edge_indices_to_xy = {
            (int(u), int(v)): np.array(xyt[:, :2])
            for (u, v), xyt in zip(
                g2b_object.bezier_edge_indices.T, g2b_object.start_end_index_to_xyt
            )
        }
        edge_indices_to_mask = {
            (int(u), int(v)): np.array(mask)
            for (u, v), mask in zip(g2b_object.bezier_edge_indices.T, g2b_object.masks)
        }
        # First index and second False index are both incorrectly False in this
        # context
        for u, v in edge_indices_to_mask.keys():
            mask = edge_indices_to_mask[(u, v)]
            if not mask[0]:
                mask[0] = True
            first_non_true_index = np.argwhere(np.logical_not(mask))[0][0]
            mask[first_non_true_index] = True
            edge_indices_to_mask[(u, v)] = mask
        for batch_index in batch_index_to_bezier_indices.keys():
            source_indices = batch_index_to_source_indices[batch_index]
            bezier_indices = batch_index_to_bezier_indices[batch_index]

            bezier_graph = BezierGraph(
                nx.subgraph(batched_optimised_graphs, bezier_indices)
            )
            source_graph = nx.subgraph(g2b_object.lane_graph, source_indices)

            if bezier_graph.number_of_edges() > 0:
                metric_objects.append(
                    cls._compute_metrics(
                        bezier_graph,
                        source_graph,
                        edge_indices_to_xy,
                        edge_indices_to_mask,
                    )
                )

        return metric_objects

    @classmethod
    def display_aggregated_metrics(
        cls,
        city_to_metrics_list: Dict[str, list],
        path_to_write_to: Optional[str] = None,
    ):
        # Convert to pandas dataframe
        metrics_df = pd.DataFrame(
            [
                {
                    "city": city,
                    "metric_local_index": metric_local_index,
                    "Mean Hausdorff (pixels)": metric.mean_hausdorff_distance,
                    "Max Hausdorff (pixels)": metric.max_hausdorff_distance,
                    "Sum Hausdorff (pixels)": metric.sum_hausdorff_distance,
                    "Num Hausdorff": metric.num_hausdorff_distances,
                    "|V_{source}|": metric.num_source_graph_nodes,
                    "|V_{bezier}|": metric.num_bezier_graph_nodes,
                    "|E_{source}|": metric.num_source_graph_edges,
                    "|E_{bezier}|": metric.num_bezier_graph_edges,
                }
                for city, metrics_list in city_to_metrics_list.items()
                for metric_local_index, metric in enumerate(metrics_list)
            ]
        )
        metrics_df["Mean Hausdorff (m)"] = (
            metrics_df["Mean Hausdorff (pixels)"] * M_PER_PIXEL
        )
        metrics_df["Max Hausdorff (m)"] = (
            metrics_df["Max Hausdorff (pixels)"] * M_PER_PIXEL
        )
        metrics_df["Sum Hausdorff (m)"] = (
            metrics_df["Sum Hausdorff (pixels)"] * M_PER_PIXEL
        )

        # Aggregate metrics
        aggregated_df = metrics_df.groupby("city").agg(
            {
                "Sum Hausdorff (pixels)": ["sum"],
                "Sum Hausdorff (m)": ["sum"],
                "Max Hausdorff (pixels)": ["max", "min", "mean", "std"],
                "Max Hausdorff (m)": ["max", "min", "mean", "std"],
                "Num Hausdorff": ["sum"],
                "|V_{source}|": ["max", "min", "mean", "std"],
                "|V_{bezier}|": ["max", "min", "mean", "std"],
                "|E_{source}|": ["max", "min", "mean", "std"],
                "|E_{bezier}|": ["max", "min", "mean", "std"],
            }
        )

        aggregated_df["Mean Hausdorff (pixels)"] = (
            aggregated_df["Sum Hausdorff (pixels)"] / aggregated_df["Num Hausdorff"]
        )
        aggregated_df["Mean Hausdorff (m)"] = (
            aggregated_df["Sum Hausdorff (m)"] / aggregated_df["Num Hausdorff"]
        )
        del aggregated_df["Sum Hausdorff (pixels)"]
        del aggregated_df["Sum Hausdorff (m)"]
        del aggregated_df["Num Hausdorff"]

        print("Aggregated Metrics:")
        # Limit of 100 should be more than ever needed
        # Display floats to 2dp only
        # Use the full width of the terminal
        with pd.option_context(
            "display.max_rows",
            100,
            "display.max_columns",
            100,
            "display.float_format",
            "{:.2f}".format,
            "display.width",
            None,
        ):
            print(aggregated_df)

        if path_to_write_to is not None:
            # Write the aggregated df to a csv
            aggregated_df.to_csv(
                os.path.join(path_to_write_to, "aggregated_metrics.csv")
            )
            # Find the worst performing metric local index per city
            worst_performing_metric_local_indices = (
                metrics_df.groupby("city")["Max Hausdorff (m)"].idxmax().values
            )

            worst_performing_df = metrics_df.iloc[worst_performing_metric_local_indices]

            for _, row in worst_performing_df.iterrows():
                city = row["city"]
                metric_local_index = row["metric_local_index"]
                source_graph = city_to_metrics_list[city][
                    metric_local_index
                ].source_graph
                bezier_graph = city_to_metrics_list[city][
                    metric_local_index
                ].bezier_graph

                plt.figure(figsize=(20, 20))
                plt.title(
                    f"City: {city} Ind: {metric_local_index} "
                    + f"Max Hausdorff: {row['Max Hausdorff (m)']:.2f}m"
                )
                plt.axis("equal")
                ax = plt.gca()
                Graph2Bezier.plot_lane_graph(ax, source_graph)
                bezier_graph.plot(ax)
                plt.savefig(
                    os.path.join(
                        path_to_write_to,
                        f"worst_performing_{city}_{metric_local_index}.png",
                    )
                )


def create_bezier_loss(
    params: Dict[str, Dict[str, jnp.array]],
) -> Callable[[jnp.array, jnp.array, jnp.array, jnp.array], jnp.array]:
    """
    Create a bezier loss function for a given set of parameters.

    Only the node features are used in this loss function creation step. Edge features
    are passed in to the returned callable.

    Parameters
    ----------
    params : Dict[str, Dict[str, jnp.array]]
        Dictionary of parameters, with keys "node" and "edge", and values
        dictionaries of parameters for nodes and edges respectively.

    Returns
    -------
    Callable[[jnp.array, jnp.array, jnp.array, jnp.array], jnp.array]
        Bezier loss function; returns the L2 distances between the predicted Bezier
        points and the ground truth points.
    """
    node_features = params["node"]["features"]

    def bezier_loss(
        edge_indices: jnp.array,
        edge_features: jnp.array,
        ground_truth_coordinates: jnp.array,
        ground_truth_mask: jnp.array,
    ):
        """
        Bezier loss function.

        Note this is a vectorised function, to be used with jax.vmap. It should
        paralellise over the number of edges present in the _Bezier_ (not ground truth)
        graph.

        Parameters
        ----------
        edge_indices : jnp.array
            Array of two node indices, representing a single edge.
            Shape: (2,)
        edge_features : jnp.array
            Array of 2 edge features, corresponding to the lengths of the P1 and P2
            Bezier whiskers respectively. Shape: (2,)
        ground_truth_coordinates : jnp.array
            Array of ground truth coordinates corresponding to this edge, where each
            ground truth coordinates is a (x, y, t) tuple. The t values in this case
            correspond to the estimated Bezier parametric value, corresponding to the
            length of this coordinate along the graph path divided by the total length
            of the graph path. Shape: (padded_num_ground_truth_coordinates, 3)
        ground_truth_mask : jnp.array
            Array of ground truth masks, where each mask is a boolean value
            indicating whether the corresponding ground truth coordinate is a genuine
            ground truth position (alternatively, it is just there to pad the arrays
            to equal length).
            Shape: (padded_num_ground_truth_coordinates,)

        Returns
        -------
        jnp.array
            Array of L2 distances between the predicted Bezier points and the ground
            truth points. Shape: (padded_num_ground_truth_coordinates,)
        """
        P0 = ground_truth_coordinates[0][:2]

        # Get the final _valid_ coordinate, i.e. the last coordinate that is not a
        # padding coordinate
        P3_index = sum(ground_truth_mask) + 1
        P3 = ground_truth_coordinates[P3_index][:2]

        l1, l2 = edge_features
        d1, d2 = jnp.exp(l1), jnp.exp(l2)

        unit_P1_direction = node_features[edge_indices[0]]
        unit_P2_direction = node_features[edge_indices[1]]

        P1 = d1 * unit_P1_direction + P0

        P2 = -d2 * unit_P2_direction + P3

        ts = ground_truth_coordinates[:, 2]
        bezier_points = (
            jnp.outer((1 - ts) ** 3, P0)
            + jnp.outer(3 * (1 - ts) ** 2 * ts, P1)
            + jnp.outer(3 * (1 - ts) * ts**2, P2)
            + jnp.outer(ts**3, P3)
        )

        xys = ground_truth_coordinates[:, :2]

        # Compute L2 loss
        l2_diff = optax.l2_loss(bezier_points, xys).mean(axis=1)
        # Mask out the padded coordinates
        l2_diff = jnp.where(ground_truth_mask, l2_diff, 0)

        return l2_diff

    return jit(bezier_loss)


def get_global_loss(
    bezier_edge_indices: jnp.ndarray,
    ground_truth_coordinates: jnp.ndarray,
    masks: jnp.ndarray,
) -> Callable[[dict], jnp.array]:
    """
    Return loss function for entire bezier graph: sum of L2 distances from every edge.

    Parameters
    ----------
    bezier_edge_indices : jnp.ndarray
        Array of bezier edge indices, where each edge index is a tuple of node indices.
        Shape: (2, num_edges)
    ground_truth_coordinates : jnp.ndarray
        Array of ground truth coordinates, where each coordinate is a (x, y, t) tuple.
        The t values in this case correspond to the estimated Bezier parametric value,
        corresponding to the length of this coordinate along the graph path divided by
        the total length of the graph path.
        Shape: (padded_num_ground_truth_coordinates, 3)
    masks : jnp.ndarray
        Array of ground truth masks, where each mask is a boolean value indicating
        whether the corresponding ground truth coordinate is a genuine ground truth
        position (alternatively, it is just there to pad the arrays to equal length).
        Shape: (padded_num_ground_truth_coordinates,)

    Returns
    -------
    Callable[[dict], jnp.array]
        Global loss function for the entire graph.
    """

    def global_loss(params: dict):
        bezier_loss_fn = jax.vmap(create_bezier_loss(params))
        edge_features = params["edge"]["features"]
        losses = bezier_loss_fn(
            bezier_edge_indices.T, edge_features, ground_truth_coordinates, masks
        )
        return jnp.sum(losses)

    return jit(global_loss)


class Graph2Bezier:
    """
    Class for conversion of lane graphs to Bezier graphs.

    Parameters
    ----------
    lane_graph : nx.DiGraph
        Lane graph to be transformed into a bezier graph.
    crossed_wires_dot_product_threshold : float, optional
        Threshold for the dot product between the incoming and outgoing direction
        vectors of a node, above which the path is considered valid, by
        default 0.85
    path_length_threshold : int, optional
        Threshold for the length of a path. See algorithm description in
        get_bezier_paths for more details. By default 2
    curvature_threshold : float, optional
        Threshold for the curvature of a path. See algorithm description in
        get_bezier_paths for more details. By default 0.001
    medfilt_kernel_size : int, optional
        Kernel size for the median filter used to smooth the curvature. See
        algorithm description in get_bezier_paths for more details. By default 5
    """

    @staticmethod
    def plot_lane_graph(
        axis,
        lane_graph,
        arrow_width: float = 1e-3,
        **kwargs,
    ):
        arrow_xs = []
        arrow_ys = []
        arrow_us = []
        arrow_vs = []
        for u, v in lane_graph.edges():
            ux, uy = lane_graph.nodes[u]["pos"]
            vx, vy = lane_graph.nodes[v]["pos"]

            arrow_xs.append(ux)
            arrow_ys.append(uy)
            arrow_us.append(vx - ux)
            arrow_vs.append(vy - uy)

        axis.quiver(
            arrow_xs,
            arrow_ys,
            arrow_us,
            arrow_vs,
            angles="xy",
            scale_units="xy",
            scale=1,
            width=arrow_width,
            **kwargs,
        )

    @classmethod
    def batch_lane_graphs(self, lane_graphs: list[nx.DiGraph], **kwargs):
        """
        Speed up optimisation by batching a list of lane graphs.

        Graphs will be combined into one single "graph" with multiple disconnected
        components. Optimisation will then occur over that entire graph.

        Parameters
        ----------
        lane_graphs : list[nx.DiGraph]
            List of lane graphs to be batched together.
        **kwargs
            Keyword arguments to be passed to the Graph2Bezier constructor.

        Returns
        -------
        Graph2Bezier
            Graph2Bezier object with the batched graph as lane_graph.
        dict[int, int]
            Dictionary mapping the indices of the _bezier_ nodes in the batched graph
            to the batch index (index of the original lane graph from which they came).
        """

        batched_lane_graph_node_index_to_source_lane_graph_index = {}
        batched_lane_graph = nx.DiGraph()
        i = 0
        for lane_graph_index, lane_graph in enumerate(lane_graphs):
            source_lane_graph_node_index_to_batched_lane_graph_node_index = {}
            for node, data in lane_graph.nodes(data=True):
                batched_lane_graph.add_node(i, **data)
                source_lane_graph_node_index_to_batched_lane_graph_node_index[node] = i
                batched_lane_graph_node_index_to_source_lane_graph_index[i] = (
                    lane_graph_index
                )
                i += 1
            for u, v, data in lane_graph.edges(data=True):
                batched_lane_graph.add_edge(
                    source_lane_graph_node_index_to_batched_lane_graph_node_index[u],
                    source_lane_graph_node_index_to_batched_lane_graph_node_index[v],
                    **data,
                )

        g2b = Graph2Bezier(batched_lane_graph, **kwargs)

        # Now create a mapping from the batch index (index of the source lane graph) to
        # the batched graph _bezier_ node indices
        batch_index_to_bezier_indices = {
            source_lane_graph_index: set()
            for source_lane_graph_index in range(len(lane_graphs))
        }
        # And a mapping from the batch index to the source lane graph node indices
        batch_index_to_source_indices = {
            source_lane_graph_index: set()
            for source_lane_graph_index in range(len(lane_graphs))
        }
        for bezier_index, source_node_index in enumerate(g2b.initial_node_ids):
            batch_index_to_bezier_indices[
                batched_lane_graph_node_index_to_source_lane_graph_index[
                    source_node_index
                ]
            ].add(bezier_index)
        for (
            source_index,
            batch_index,
        ) in batched_lane_graph_node_index_to_source_lane_graph_index.items():
            batch_index_to_source_indices[batch_index].add(source_index)

        batches_to_remove = set()
        for batch_index, node_set in batch_index_to_bezier_indices.items():
            if (
                len(node_set) == 0
                and len(batch_index_to_source_indices[batch_index]) != 0
            ):
                print(
                    "WARNING: Bezier graph is empty, while source graph is not. "
                    + "This happens for certain types of source graph which are "
                    + "unsupported, e.g. loops. If you see lots of these errors, "
                    + "something is probably wrong."
                )
                batches_to_remove.add(batch_index)
        batch_index_to_bezier_indices = {
            k: v
            for k, v in batch_index_to_bezier_indices.items()
            if k not in batches_to_remove
        }
        batch_index_to_source_indices = {
            k: v
            for k, v in batch_index_to_source_indices.items()
            if k not in batches_to_remove
        }
        return (
            g2b,
            batch_index_to_bezier_indices,
            batch_index_to_source_indices,
            batches_to_remove,
        )

    def __init__(
        self,
        lane_graph: nx.DiGraph,
        path_length_threshold: int = 2,
        curvature_threshold: float = 0.001,
        medfilt_kernel_size: int = 5,
        verbose: bool = False,
    ):
        self.lane_graph = deepcopy(lane_graph)
        # Tag the lane graph nodes with their "source node ID" which will be used to
        # map back to the original IDs
        for node in self.lane_graph.nodes():
            self.lane_graph.nodes[node]["source_node_id"] = node
        self.verbose = verbose
        self.fix_crossed_wires()
        (
            self.initial_node_ids,
            self.initial_node_features,
            self.node_positions,
            self.bezier_edge_indices,
            self.initial_edge_features,
            self.start_end_index_to_xyt,
            self.masks,
        ) = self.initialise_bezier_graph(
            path_length_threshold=path_length_threshold,
            curvature_threshold=curvature_threshold,
            medfilt_kernel_size=medfilt_kernel_size,
        )

    def fix_crossed_wires(self) -> None:
        """
        Fix crossed wires in the lane graph.

        Notes
        -----
        This function modifies the lane graph (self.lane_graph) in place.

        Crossed wires are defined as nodes which connect multiple paths (i.e. have
        in-degree and out-degree > 1), but which do so erroneously, i.e. the paths
        do cross but there should not be an allowed connection between them.

        These can be effectively filtered by finding the nodes with the same number of
        in and out edges, finding the optimal match between the incoming and outgoing
        directions using the Hungarian algorithm, and splitting the node into
        multiple nodes, each with a single incoming and outgoing edge.

        As a side effect, this function also separates bidirectional lanes (previously
        encoded by a path of nodes with edges in both directions) into overlapping
        separate paths of nodes with edges in only one direction; this is what is
        required for the Bezier fit.
        """
        # Nodes with the same in degree as out degree
        same_in_out = [
            n
            for n in self.lane_graph.nodes()
            if (
                (self.lane_graph.in_degree(n) > 1)
                and (self.lane_graph.out_degree(n) > 1)
            )
        ]
        # Isolated bidirectional lanes will still give us a problem. We need a concept
        # of "orphaned nodes" to avoid infinite curvature cycles - nodes which only
        # connect to the same_in_out set defined above.
        same_in_out_set = set(same_in_out)  # For O(1) membership checking
        orphaned_nodes = []
        for node in self.lane_graph.nodes():
            if (
                (self.lane_graph.in_degree(node) == 1)
                and (self.lane_graph.out_degree(node) == 1)
                and (list(self.lane_graph.predecessors(node))[0] in same_in_out_set)
                and (list(self.lane_graph.successors(node))[0] in same_in_out_set)
            ):
                orphaned_nodes.append(node)
        for crossed_wire_node in same_in_out:
            in_nodes = list(self.lane_graph.predecessors(crossed_wire_node))
            out_nodes = list(self.lane_graph.successors(crossed_wire_node))

            crossed_wire_node_position = np.array(
                self.lane_graph.nodes[crossed_wire_node]["pos"]
            )

            in_node_positions = np.array(
                [self.lane_graph.nodes[in_node]["pos"] for in_node in in_nodes]
            )
            out_node_positions = np.array(
                [self.lane_graph.nodes[out_node]["pos"] for out_node in out_nodes]
            )

            in_node_directions = crossed_wire_node_position - in_node_positions
            in_node_directions /= np.linalg.norm(
                in_node_directions, axis=1, keepdims=True
            )

            out_node_directions = out_node_positions - crossed_wire_node_position
            out_node_directions /= np.linalg.norm(
                out_node_directions, axis=1, keepdims=True
            )

            # Compute the optimal pairwise matching between in and out directions
            # according to the Hungarian algorithm
            cost_matrix = -np.dot(in_node_directions, out_node_directions.T)
            in_indices, out_indices = linear_sum_assignment(cost_matrix)

            # Delete the existing connection node
            self.lane_graph.remove_node(crossed_wire_node)

            # For each connected component, add a new node to the lane graph
            # and connect the neighbours to it
            new_node_id = None
            for in_index, out_index in zip(in_indices, out_indices):
                if new_node_id is None:
                    new_node_id = max(self.lane_graph.nodes) + 1
                else:
                    new_node_id += 1
                self.lane_graph.add_node(
                    new_node_id,
                    pos=crossed_wire_node_position,
                    source_node_id=crossed_wire_node,
                )
                self.lane_graph.add_edge(
                    in_nodes[in_index],
                    new_node_id,
                )
                self.lane_graph.add_edge(
                    new_node_id,
                    out_nodes[out_index],
                )

        for o in orphaned_nodes:
            self.lane_graph.remove_node(o)

    def get_lane_node_degrees(self) -> Tuple[dict[int, int], dict[int, int]]:
        """
        Get the in and out degrees of each node in the lane graph.

        Returns
        -------
        Tuple[dict[int, int], dict[int, int]]
            Tuple of dictionaries, where the first dictionary maps node ids to their
            in degrees, and the second dictionary maps node ids to their out degrees.
        """
        return dict(self.lane_graph.in_degree()), dict(self.lane_graph.out_degree())

    def _breadth_first_search_for_paths_between_nodes(
        self, start_end_nodes: list
    ) -> list[list[int]]:
        # Set for fast membership checking
        start_end_node_set = set(start_end_nodes)
        paths = []
        for start_node in tqdm(start_end_nodes, disable=not self.verbose):
            candidate_paths = [
                [start_node, successor]
                for successor in self.lane_graph.successors(start_node)
            ]
            while len(candidate_paths) > 0:
                next_path = candidate_paths.pop()
                if next_path[-1] in start_end_node_set:
                    paths.append(next_path)
                    continue
                for successor in self.lane_graph.successors(next_path[-1]):
                    candidate_paths.append(next_path + [successor])
        return paths

    def _split_paths_by_curvature(
        self,
        input_paths: list[list[int]],
        path_length_threshold: int = 2,
        curvature_threshold: float = 0.001,
        medfilt_kernel_size: int = 5,
    ) -> list[list[int]]:
        split_paths = []
        for path in input_paths:
            path_coordinates = np.array([self.lane_graph.nodes[n]["pos"] for n in path])
            # Plot the start and end coordinates as red crosses
            if len(path) <= path_length_threshold:
                split_paths.append(path)
            else:
                x1 = np.gradient(path_coordinates, axis=0)
                x2 = np.gradient(x1, axis=0)
                cross_product_mag = np.abs(x1[:, 0] * x2[:, 1] - x1[:, 1] * x2[:, 0])
                curvature = cross_product_mag / np.power(np.linalg.norm(x1, axis=1), 3)

                high_curvature = curvature > curvature_threshold

                high_curvature = medfilt(
                    high_curvature.astype(int), kernel_size=medfilt_kernel_size
                ).astype(bool)

                no_end_mask = np.ones(len(high_curvature) - 1).astype(bool)
                no_end_mask[0] = False
                no_end_mask[-1] = False

                slice_indices = np.argwhere(
                    np.logical_and(
                        high_curvature[:-1] != high_curvature[1:], no_end_mask
                    )
                ).T.squeeze(axis=0)

                # Filter out slice indices within medfilt_kernel_size of either the
                # start or end of the path
                slice_indices = slice_indices[
                    np.logical_and(
                        slice_indices > medfilt_kernel_size,
                        slice_indices < len(path) - medfilt_kernel_size,
                    )
                ]

                if len(slice_indices) == 0:
                    split_paths.append(path)
                    continue
                else:
                    slice_indices = np.concatenate(
                        [[0], slice_indices, [len(path) - 1]]
                    )
                    for i in range(len(slice_indices) - 1):
                        shortened_path = path[
                            slice_indices[i] : slice_indices[i + 1] + 1
                        ]
                        split_paths.append(shortened_path)
        return split_paths

    def get_bezier_paths(
        self,
        path_length_threshold: int = 2,
        curvature_threshold: float = 0.001,
        medfilt_kernel_size: int = 5,
    ) -> list[list[int]]:
        """
        Get all bezier paths in the lane graph.

        Bezier paths are determined using the following algorithm:
        1. Find all nodes where the in or out degrees are not equal to 1. Call these
            the start/end nodes.
        2. For each start/end node, use a breadth first search to find all paths
            to other start/end nodes, stopping when they reach the next start/end
            node.
        3. For each path, check if it is longer than the path length threshold. If
            it is, then check if the path has any high curvature sections. If it does,
            then split the path at the high curvature sections, and add the new paths
            to the list of bezier paths. If it does not, then add the path to the list
            of bezier paths.

        Parameters
        ----------
        path_length_threshold : int, optional
            Threshold for the length of a path, above which step 3 in the above
            algorithm will be run, by default 2
        curvature_threshold : float, optional
            Threshold for the curvature of a path, used in step 3 in the above
            algorithm, by default 0.001
        medfilt_kernel_size : int, optional
            Kernel size for the median filter used to smooth the curvature, by default

        Returns
        -------
        list[list[int]]
            List of bezier paths, where each bezier path is a list of node ids.
        """
        # Breadth first search to find all possible paths between nodes where either
        # the in or out degrees are not equal to 1
        lane_node_in_degrees, lane_node_out_degrees = self.get_lane_node_degrees()
        start_end_nodes = set(
            [
                n
                for n in self.lane_graph.nodes()
                if (lane_node_in_degrees[n] != 1) or (lane_node_out_degrees[n] != 1)
            ]
        )
        bezier_paths = self._breadth_first_search_for_paths_between_nodes(
            start_end_nodes
        )

        split_paths = self._split_paths_by_curvature(
            bezier_paths,
            path_length_threshold=path_length_threshold,
            curvature_threshold=curvature_threshold,
            medfilt_kernel_size=medfilt_kernel_size,
        )

        return split_paths

    def _linearly_interpolate_coordinates(self, xy, fixed_length):
        # Calculate the distances between each consecutive point
        deltas = np.diff(xy, axis=0)
        segment_lengths = np.linalg.norm(deltas, axis=1)

        # Calculate the cumulative distance of each point from the first point
        cumulative_distances = np.insert(np.cumsum(segment_lengths), 0, 0)

        # Create an equally spaced array of cumulative distances for interpolation
        new_cumulative_distances = np.linspace(
            0, cumulative_distances[-1], fixed_length
        )

        # Interpolate the x and y coordinates
        interpolator_x = interp1d(
            cumulative_distances, xy[:, 0], kind="linear", fill_value="extrapolate"
        )
        interpolator_y = interp1d(
            cumulative_distances, xy[:, 1], kind="linear", fill_value="extrapolate"
        )

        new_x = interpolator_x(new_cumulative_distances)
        new_y = interpolator_y(new_cumulative_distances)

        # Combine x and y coordinates
        new_xy = np.vstack((new_x, new_y)).T

        return new_xy

    def get_start_end_to_bezier_path(
        self,
        path_length_threshold: int = 2,
        curvature_threshold: float = 0.001,
        medfilt_kernel_size: int = 5,
        overlapping_lane_max_dist_threshold: float = 10.0,
    ) -> dict[tuple[int, int], list[int]]:
        """
        Get a mapping from start/end node tuples to paths, to create the bezier graph.

        Where Bezier paths of length 2 are found, a new node is added to the lane graph
        at the average position of the two nodes, in order that the Bezier graph can
        be optimised correctly.

        Parameters
        ----------
        path_length_threshold : int, optional
            Threshold for the length of a path. See algorithm description in
            get_bezier_paths for more details. By default 2
        curvature_threshold : float, optional
            Threshold for the curvature of a path. See algorithm description in
            get_bezier_paths for more details. By default 0.001
        medfilt_kernel_size : int, optional
            Kernel size for the median filter used to smooth the curvature. See
            algorithm description in get_bezier_paths for more details. By default 5
        overlapping_lane_max_dist_threshold : float, optional
            Maximum distance between two overlapping lanes, above which an error will
            be raised. By default 10.0

        Returns
        -------
        dict[tuple[int, int], list[int]]
            Dictionary mapping start/end node tuples to bezier paths.

        Raises
        ------
        ValueError
            If a bezier path has fewer than 2 nodes.
        ValueError
            If a start/end node tuple has multiple bezier paths.
        """
        bezier_paths = self.get_bezier_paths(
            path_length_threshold=path_length_threshold,
            curvature_threshold=curvature_threshold,
            medfilt_kernel_size=medfilt_kernel_size,
        )
        start_end_to_bezier_path = {}
        for bezier_path in bezier_paths:
            if len(bezier_path) == 2:
                average_position = np.mean(
                    [self.lane_graph.nodes[n]["pos"] for n in bezier_path], axis=0
                )
                # Add new node to lane graph at that position
                node_id = max(self.lane_graph.nodes) + 1
                self.lane_graph.add_node(
                    node_id, pos=average_position, source_node_id=bezier_path[0]
                )
                # Remove existing edge from the lane graph
                self.lane_graph.remove_edge(*bezier_path)
                # Add new edges to the lane graph
                self.lane_graph.add_edge(bezier_path[0], node_id)
                self.lane_graph.add_edge(node_id, bezier_path[1])
                # Add new node to middle of bezier path
                bezier_path = [bezier_path[0], node_id, bezier_path[1]]
            elif len(bezier_path) < 2:
                raise ValueError(f"Bezier path {bezier_path} has fewer than 2 nodes.")
            start_end = (bezier_path[0], bezier_path[-1])
            if start_end[0] == start_end[1]:
                print("Warning: start and end nodes are the same.")
                continue
            if start_end in start_end_to_bezier_path:
                # Nodes start_end have multiple bezier paths
                # This sometimes happens where there are simply multiple overlapping
                # paths. We will catch this by checking the node positions:
                existing_bezier_path_positions = np.array(
                    [
                        self.lane_graph.nodes[n]["pos"]
                        for n in start_end_to_bezier_path[start_end]
                    ]
                )
                new_bezier_path_positions = np.array(
                    [self.lane_graph.nodes[n]["pos"] for n in bezier_path]
                )
                if len(existing_bezier_path_positions) != len(
                    new_bezier_path_positions
                ):
                    # Interpolate these paths to a fixed number of points
                    fixed_num_points = 100
                    existing_bezier_path_positions = (
                        self._linearly_interpolate_coordinates(
                            existing_bezier_path_positions, fixed_num_points
                        )
                    )
                    new_bezier_path_positions = self._linearly_interpolate_coordinates(
                        new_bezier_path_positions, fixed_num_points
                    )
                max_distance = np.linalg.norm(
                    (existing_bezier_path_positions - new_bezier_path_positions), axis=1
                ).max()
                if max_distance > overlapping_lane_max_dist_threshold:
                    print(
                        f"WARNING: Nodes {start_end} have multiple bezier paths, and "
                        + "these paths have node positions that differ more than the "
                        + "max distance threshold of "
                        + f"{overlapping_lane_max_dist_threshold}. The second path will"
                        + " be ignored."
                    )
                else:
                    print("Warning: nodes have multiple overlapping bezier paths.")
                    # Select the shortest path
                    existing_bezier_path_length = np.linalg.norm(
                        existing_bezier_path_positions[1:]
                        - existing_bezier_path_positions[:-1],
                        axis=1,
                    ).sum()
                    new_bezier_path_length = np.linalg.norm(
                        new_bezier_path_positions[1:] - new_bezier_path_positions[:-1],
                        axis=1,
                    ).sum()
                    if new_bezier_path_length < existing_bezier_path_length:
                        start_end_to_bezier_path[start_end] = bezier_path
            start_end_to_bezier_path[start_end] = bezier_path
        return start_end_to_bezier_path

    def initialise_bezier_graph(
        self,
        path_length_threshold: int = 2,
        curvature_threshold: float = 0.001,
        medfilt_kernel_size: int = 5,
    ) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
        """
        Transform the lane graph into the parameters to be optimised for bezier graph.

        Parameters to be optimised are as follows:
        - Node features: unit direction vectors for each node, representing the Bezier
            whisker in the "direction" of that node. Shape: (num_nodes, 2)
        - Edge features: log lengths of the P1 and P2 Bezier whiskers for each edge.
            Shape: (num_edges, 2)

        Parameters
        ----------
        path_length_threshold : int, optional
            Threshold for the length of a path. See algorithm description in
            get_bezier_paths for more details. By default 2
        curvature_threshold : float, optional
            Threshold for the curvature of a path. See algorithm description in
            get_bezier_paths for more details. By default 0.001
        medfilt_kernel_size : int, optional
            Kernel size for the median filter used to smooth the curvature. See
            algorithm description in get_bezier_paths for more details. By default 5

        Returns
        -------
        Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]
            Tuple of node features, node positions, bezier edge indices, edge features,
            ground truth coordinates, and masks.
        """
        start_end_to_beizer_path = self.get_start_end_to_bezier_path(
            path_length_threshold=path_length_threshold,
            curvature_threshold=curvature_threshold,
            medfilt_kernel_size=medfilt_kernel_size,
        )
        if len(start_end_to_beizer_path) == 0:
            raise NoBezierPathsError("No bezier paths found.")
        edge_indices = np.array([(u, v) for u, v in start_end_to_beizer_path.keys()]).T

        # Get all unique node ids from the node ID tuples in the keys of
        # start_end_to_bezier_path
        node_ids = set([])
        for start_end in start_end_to_beizer_path.keys():
            node_ids.add(start_end[0])
            node_ids.add(start_end[1])
        node_ids = list(node_ids)

        node_direction_vectors = []
        for n in node_ids:
            direction_vectors = []
            for adjacent_edges in list(self.lane_graph.in_edges(n)) + list(
                self.lane_graph.out_edges(n)
            ):
                direction_vector = (
                    self.lane_graph.nodes[adjacent_edges[1]]["pos"]
                    - self.lane_graph.nodes[adjacent_edges[0]]["pos"]
                )
                direction_vectors.append(direction_vector)
            direction_vectors = np.array(direction_vectors)
            normalised_direction_vectors = direction_vectors / jnp.linalg.norm(
                direction_vectors, axis=1, keepdims=True
            )
            average_direction_vector = np.mean(normalised_direction_vectors, axis=0)
            node_direction_vectors.append(average_direction_vector)
        node_direction_vectors = np.array(node_direction_vectors)
        node_direction_vectors = (
            node_direction_vectors.T / np.linalg.norm(node_direction_vectors, axis=1)
        ).T
        if jnp.isnan(node_direction_vectors).any():
            raise ValueError("NaNs in node direction vectors.")
        node_features = node_direction_vectors

        gt_index_to_bezier_index = {n: i for i, n in enumerate(node_ids)}

        padding_length = max([len(v) for v in start_end_to_beizer_path.values()])

        ground_truth_coordinates = []
        masks = []
        bezier_edge_indices = np.array(
            [
                (gt_index_to_bezier_index[u], gt_index_to_bezier_index[v])
                for u, v in edge_indices.T
            ]
        ).T
        edge_features = np.zeros((len(bezier_edge_indices.T), 2))
        for start, end in tqdm(edge_indices.T, disable=not self.verbose):
            bezier_path = start_end_to_beizer_path[(start, end)]
            X = np.array([self.lane_graph.nodes[n]["pos"] for n in bezier_path])
            X_dists = np.linalg.norm(X[1:] - X[:-1], axis=1)
            X_dists = np.insert(X_dists, 0, 0)
            X_cumulative_dists = np.cumsum(X_dists)
            X_total_dist = X_cumulative_dists[-1]
            X_ts = X_cumulative_dists / X_total_dist

            xyt = np.concatenate([X, X_ts[:, None]], axis=1)
            xyt = np.array(xyt)

            m, n = xyt.shape

            padded_array = np.zeros((padding_length, n))

            padded_array[:m] = xyt

            mask = np.zeros(padding_length, dtype=bool)
            mask[1 : m - 1] = True

            ground_truth_coordinates.append(padded_array)
            masks.append(mask)

        ground_truth_coordinates = np.array(ground_truth_coordinates)
        masks = np.array(masks)

        initial_node_ids = [
            self.lane_graph.nodes[n]["source_node_id"] for n in node_ids
        ]

        return (
            initial_node_ids,
            jnp.array(node_features),
            jnp.array([self.lane_graph.nodes[n]["pos"] for n in node_ids]),
            jnp.array(bezier_edge_indices),
            jnp.array(edge_features),
            jnp.array(ground_truth_coordinates),
            jnp.array(masks),
        )

    def bezier_parameters_to_bezier_graph(
        self,
        node_features: jnp.array,
        edge_features: jnp.array,
    ) -> BezierGraph:
        """
        Convert the bezier parameters to a bezier graph.

        Note at this point we normalise the node features to be unit vectors, by
        combining all length information into the edge features.

        Parameters
        ----------
        node_features : jnp.array
            Array of node features, where each node feature is a unit direction vector
            for the Bezier whisker in the "direction" of that node.
            Shape: (num_nodes, 2)
        edge_features : jnp.array
            Array of edge features, where each edge feature is the log length of the P1
            and P2 Bezier whiskers for that edge.
            Shape: (num_edges, 2)

        Returns
        -------
        BezierGraph
            Bezier graph.
        """
        bezier_graph = BezierGraph()

        node_to_former_direction_length = {}

        for i, node_position in enumerate(self.node_positions):
            node_direction = np.array(node_features[i])
            direction_length = np.linalg.norm(node_direction)
            # Add a small value to ensure never dividing by zero
            direction_length += 1e-6
            node_to_former_direction_length[int(i)] = direction_length
            # normalise the node direction vector
            unit_node_direction = node_direction / direction_length

            bezier_graph.add_node(
                i, pos=np.array(node_position), direction=unit_node_direction
            )

        for i, (u, v) in enumerate(self.bezier_edge_indices.T):
            u = int(u)
            v = int(v)
            u_direction_length = node_to_former_direction_length[u]
            v_direction_length = node_to_former_direction_length[v]

            # The stored edge features are the natural logarithm of the bezier whisker
            # length. Therefore we need to exponentiate them to get the actual length
            # before multiplying by the old length of the node direction vector.
            u_new_direction_length = np.exp(edge_features[i, 0]) * u_direction_length
            log_l1 = np.log(u_new_direction_length)
            v_new_direction_length = np.exp(edge_features[i, 1]) * v_direction_length
            log_l2 = np.log(v_new_direction_length)

            bezier_graph.add_edge(
                u,
                v,
                log_l1=float(log_l1),
                log_l2=float(log_l2),
            )

        return bezier_graph

    def optimise_bezier_graph(
        self,
        max_num_iterations: int = 10000,
        adam_b1: float = 0.9,
        adam_b2: float = 0.999,
        adam_eps: float = 1e-8,
        adam_lr: float = 0.01,
        early_stopping: bool = True,
        early_stopping_window: int = 10,
        early_stopping_fraction_threshold: float = 1e-4,
    ) -> nx.DiGraph:
        """
        Optimise the bezier graph parameters.

        Parameters
        ----------
        max_num_iterations : int, optional
            Maximum number of iterations to run the optimisation for, by default 10000
        adam_b1 : float, optional
            Adam beta 1 parameter, by default 0.9
        adam_b2 : float, optional
            Adam beta 2 parameter, by default 0.999
        adam_eps : float, optional
            Adam epsilon parameter, by default 1e-8
        adam_lr : float, optional
            Adam learning rate parameter, by default 0.01
        early_stopping : bool, optional
            Whether to stop the optimisation early if the loss has converged, by
            default True
        early_stopping_window : int, optional
            Number of previous losses to use to calculate the standard deviation and
            mean, by default 10
        early_stopping_fraction_threshold : float, optional
            Threshold for the standard deviation / mean ratio, below which the
            optimisation will be stopped, by default 1e-4

        Returns
        -------
        nx.DiGraph
            Optimised bezier graph.
        """
        loss_fn = get_global_loss(
            self.bezier_edge_indices, self.start_end_index_to_xyt, self.masks
        )
        node_features = self.initial_node_features
        edge_features = self.initial_edge_features

        params = {
            "node": {"features": node_features},
            "edge": {"features": edge_features},
        }

        adam = optax.chain(
            optax.scale_by_adam(b1=adam_b1, b2=adam_b2, eps=adam_eps),
            optax.scale(-1 * adam_lr),
        )
        opt_state = adam.init(params)
        t1 = time()
        previous_losses = []
        for i in range(max_num_iterations):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            if jnp.isnan(loss).any():
                raise ValueError(f"Hit NaN loss on iteration {i}")
            if (
                jnp.isnan(grads["edge"]["features"]).any()
                or jnp.isnan(grads["node"]["features"]).any()
            ):
                raise ValueError(f"Hit NaN grad on iteration {i}")
            updates, opt_state = adam.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if early_stopping:
                # Stop early if loss has converged
                previous_losses.append(loss)
                if len(previous_losses) > early_stopping_window:
                    previous_losses = previous_losses[1:]
                    if (
                        np.std(previous_losses) / np.mean(previous_losses)
                        < early_stopping_fraction_threshold
                    ):
                        if self.verbose:
                            print("Stopped optimisation early.")
                        break
        t2 = time()
        if self.verbose:
            print(f"Time taken to optimise: {t2 - t1}")

        optimised_bezier_graph = self.bezier_parameters_to_bezier_graph(
            params["node"]["features"], params["edge"]["features"]
        )

        return optimised_bezier_graph

    def plot_ground_truth(
        self,
        axis: plt.Axes,
        arrow_width: float = 1e-3,
        plot_bezier_start_ends: bool = True,
        **kwargs,
    ):
        """
        Plot the underlying ground truth graph on an axis.

        Parameters
        ----------
        axis : plt.Axes
            Axis on which to plot.
        arrow_width : float, optional
            Arrow width used to visualise edges, by default 1e-3
        plot_bezier_start_ends : bool, optional
            Whether to plot the start and end nodes of the Bezier paths, by default True
        **kwargs
            Keyword arguments passed to plt.quiver
        """
        Graph2Bezier.plot_lane_graph(
            axis, self.lane_graph, arrow_width=arrow_width, **kwargs
        )

        if plot_bezier_start_ends:
            start_end_to_bezier_paths = self.get_start_end_to_bezier_path()
            start_ends = set([])
            immediate_successors = set([])
            immediate_predecessors = set([])
            for start, end in start_end_to_bezier_paths.keys():
                start_ends.add(start)
                start_ends.add(end)

                immediate_successors |= set(self.lane_graph.successors(start))
                immediate_successors |= set(self.lane_graph.successors(end))
                immediate_predecessors |= set(self.lane_graph.predecessors(end))
                immediate_predecessors |= set(self.lane_graph.predecessors(end))
            start_end_positions = np.array(
                [self.lane_graph.nodes[n]["pos"] for n in start_ends]
            )
            axis.scatter(
                start_end_positions[:, 0], start_end_positions[:, 1], c="r", s=1
            )
            immediate_successor_positions = np.array(
                [self.lane_graph.nodes[n]["pos"] for n in immediate_successors]
            )
            axis.scatter(
                immediate_successor_positions[:, 0],
                immediate_successor_positions[:, 1],
                c="b",
                s=1,
            )
            immediate_predecessor_positions = np.array(
                [self.lane_graph.nodes[n]["pos"] for n in immediate_predecessors]
            )
            axis.scatter(
                immediate_predecessor_positions[:, 0],
                immediate_predecessor_positions[:, 1],
                c="g",
                s=1,
            )

    def compute_optimisation_metrics(
        self, optimised_bezier_graph: nx.DiGraph
    ) -> dict[str, float]:
        """
        Compute metrics for the optimised bezier graph.

        Parameters
        ----------
        optimised_bezier_graph : nx.DiGraph
            Optimised bezier graph.

        Returns
        -------
        dict[str, float]
            Dictionary of metrics.
        """
        # We'll use the Hausdorff distance as a metric for how well the optimised bezier
        # graph matches the ground truth bezier graph
        hausdorff_distances = []
        edge_indices = []
        edge_indices_to_xy = {
            (int(u), int(v)): np.array(xyt[:, :2])
            for (u, v), xyt in zip(
                self.bezier_edge_indices.T, self.start_end_index_to_xyt
            )
        }
        edge_indices_to_mask = {
            (int(u), int(v)): np.array(mask)
            for (u, v), mask in zip(self.bezier_edge_indices.T, self.masks)
        }
        # First index and second False index are both incorrectly False in this
        # context
        for u, v in edge_indices_to_mask.keys():
            mask = edge_indices_to_mask[(u, v)]
            if not mask[0]:
                mask[0] = True
            first_non_true_index = np.argwhere(np.logical_not(mask))[0][0]
            mask[first_non_true_index] = True
            edge_indices_to_mask[(u, v)] = mask
        for u, v in optimised_bezier_graph.edges():
            P0, P1, P2, P3 = optimised_bezier_graph.get_bezier_control_points_from_edge(
                u, v
            )
            bezier_points = optimised_bezier_graph.interpolate_points_along_bezier(
                P0, P1, P2, P3
            )
            ground_truth_points = edge_indices_to_xy[(u, v)][
                edge_indices_to_mask[(u, v)]
            ]
            hausdorff_distances.append(
                directed_hausdorff(ground_truth_points, bezier_points)[0]
            )
            edge_indices.append((u, v))
        hausdorff_distances = np.array(hausdorff_distances)

        metrics = {
            "mean_hausdorff_distance": np.mean(hausdorff_distances),
            "max_hausdorff_distance": np.max(hausdorff_distances),
            "max_hausdorff_distance_edge_index": edge_indices[
                np.argmax(hausdorff_distances)
            ],
        }

        return metrics
