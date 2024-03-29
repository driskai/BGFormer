from typing import Any, Dict, List, Tuple

import numpy as np
import rustworkx as rx
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csgraph
from scipy.spatial import KDTree
from tqdm import tqdm

from .utils import interpolate_graph

Node = Any


def geo_metric(
    gt: rx.PyDiGraph,
    pred: rx.PyDiGraph,
    radius: float = 8.0,
    interpolate: bool = True,
    pos_key: str = "pos",
    nodes_per_m: float = 0.25,
    num_node_threshold: int = 1_000_000,
    dense_threshold: float = 0.01,
    force_dense: bool = False,
) -> Tuple[float, float, List[Tuple[Node, Node]]]:
    """
    Compute the GEO metric for a ground truth and predicated graph pair.

    Parameters:
    ----------
    gt: rx.PyDiGraph
        The ground truth graph.
    pred: rx.PyDiGraph
        The predicted graph.
    radius: float
        The radius to use for matching.
    interpolate: bool
        If True, interpolate the graphs to a higher resolution before matching.
    pos_key: str
        The key in the node attributes to use for position.
    nodes_per_m: float
        The number of nodes per meter to interpolate to.
    dense_threshold: float
        The threshold for using the dense matching algorithm. If the ratio of
        non-zero elements to total elements in the distance matrix is greater
        than this value, the dense algorithm will be used.
    force_dense: bool
        If True, force the use of the dense linear sum assignment algorithm.

    Returns:
    -------
    precision: float
        The precision of the matching.
    recall: float
        The recall of the matching.
    matches: list
        Matched node pairs used in the computation.
    """
    if pred.num_nodes() == 0 or gt.num_nodes() == 0:
        if pred.num_nodes() == gt.num_nodes():
            # if both graphs are empty, return perfect score otherwise 0
            return 1.0, 1.0
        return 0.0, 0.0

    if interpolate:
        gt = interpolate_graph(gt, nodes_per_m=nodes_per_m, pos_key=pos_key)
        pred = interpolate_graph(pred, nodes_per_m=nodes_per_m, pos_key=pos_key)

    gt_pos = np.array([n["pos"] for n in gt.nodes()])
    pred_pos = np.array([n["pos"] for n in pred.nodes()])

    gt_tree = KDTree(gt_pos)
    pred_tree = KDTree(pred_pos)
    dists = gt_tree.sparse_distance_matrix(pred_tree, radius, output_type="coo_matrix")

    if (
        force_dense
        or (gt_pos.shape[0] <= num_node_threshold)
        or ((dists.size / (dists.shape[0] * dists.shape[1])) > dense_threshold)
    ):
        dists_dense = dists.toarray()
        mask = np.ones_like(dists_dense, dtype=bool)
        mask[dists.row, dists.col] = False
        dists_dense[mask] = radius + 1.0
        row_idx, col_idx = linear_sum_assignment(dists_dense)

        matches = np.array([row_idx, col_idx]).T
        valid = dists_dense[row_idx, col_idx] < radius
        valid_matches = matches[valid]
        valid_count = valid.sum()
    else:
        matching = csgraph.maximum_bipartite_matching(dists)
        row_idx = matching[matching != -1]
        col_idx = np.argwhere(matching != -1).squeeze(1)
        if len(row_idx) > 0:
            dists = dists.tocsr()
            row_idx_2, col_idx_2 = csgraph.min_weight_full_bipartite_matching(
                dists[:, col_idx][row_idx]
            )
            row_idx = row_idx[row_idx_2]
            col_idx = col_idx[col_idx_2]
        valid_count = len(row_idx)
        valid_matches = np.array([row_idx, col_idx]).T

    precision = valid_count / pred_pos.shape[0]
    recall = valid_count / gt_pos.shape[0]
    return precision, recall, valid_matches


def geo_topo_metric(
    gt: rx.PyDiGraph,
    pred: rx.PyDiGraph,
    walk_dist: float = 50.0,
    radius: float = 8.0,
    interpolate: bool = True,
    pos_key: str = "pos",
    nodes_per_m: float = 0.25,
    verbose: bool = False,
    force_dense: bool = False,
) -> Dict[str, float]:
    """
    Compute the GEO and TOPO metrics for a ground truth and predicated graph pair.

    Parameters:
    ----------
    gt: rx.PyDiGraph
        The ground truth graph.
    pred: rx.PyDiGraph
        The predicted graph.
    walk_dist: float
        The distance to walk to gather subgraphs for the TOPO metric.
    radius: float
        The radius to use for matching.
    interpolate: bool
        If True, interpolate the graphs to a higher resolution before matching.
    pos_key: str
        The key in the node attributes to use for position.
    nodes_per_m: float
        The number of nodes per meter to interpolate to.
    force_dense: bool
        If True, force the use of the dense linear sum assignment algorithm.

    Returns:
    -------
    geo_precision: float
        The GEO precision of the matching.
    geo_recall: float
        The GEO recall of the matching.
    topo_precision: float
        The TOPO precision of the matching.
    topo_recall: float
        The TOPO recall of the matching.
    """
    if pred.num_nodes() == 0 or gt.num_nodes() == 0:
        if gt.num_nodes() == pred.num_nodes():
            return {
                "geo_precision": 1.0,
                "geo_recall": 1.0,
                "topo_precision": 1.0,
                "topo_recall": 1.0,
            }
        return {
            "geo_precision": 0.0,
            "geo_recall": 0.0,
            "topo_precision": 0.0,
            "topo_recall": 0.0,
        }

    if interpolate:
        gt = interpolate_graph(gt, nodes_per_m=nodes_per_m, pos_key=pos_key)
        pred = interpolate_graph(pred, nodes_per_m=nodes_per_m, pos_key=pos_key)

    geo_precision, geo_recall, matches = geo_metric(
        gt, pred, radius=radius, interpolate=False, force_dense=force_dense
    )

    tot_topo_precision, tot_topo_recall = 0.0, 0.0
    for gt_idx, pred_idx in tqdm(matches) if verbose else matches:
        gt_visitor = MaxDistVisitor(walk_dist)
        rx.bfs_search(gt, [gt_idx], gt_visitor)
        gt_sub = [
            x for x, v in gt_visitor.distances.items() if x != gt_idx and v < walk_dist
        ]
        gt_sub = gt.subgraph(gt_sub)

        pred_visitor = MaxDistVisitor(walk_dist)
        rx.bfs_search(pred, [pred_idx], pred_visitor)
        pred_sub = [
            x
            for x, v in pred_visitor.distances.items()
            if x != pred_idx and v < walk_dist
        ]
        pred_sub = pred.subgraph(pred_sub)

        if not gt_sub or not pred_sub:
            if len(gt_sub) == len(pred_sub):
                tot_topo_precision += 1.0
                tot_topo_recall += 1.0
            continue

        topo_pre, topo_rec, _ = geo_metric(
            gt_sub, pred_sub, radius=radius, interpolate=False, force_dense=force_dense
        )
        tot_topo_precision += topo_pre
        tot_topo_recall += topo_rec

    return {
        "geo_precision": geo_precision,
        "geo_recall": geo_recall,
        "topo_precision": tot_topo_precision / pred.num_nodes(),
        "topo_recall": tot_topo_recall / gt.num_nodes(),
    }


def get_topo_subgraph(graph, tree, radius, node, node_pos):
    """Get the nodes that are path connected to the given node within the radius."""
    _, subgraph = tree.query(node_pos, radius)
    node_component = rx.algorithms.components.node_connected_component(graph, node)
    subgraph = list(set(subgraph).intersection(node_component))
    if not subgraph:
        return None
    return graph.subgraph(subgraph)


class MaxDistVisitor(rx.visit.BFSVisitor):
    def __init__(self, max_dist: float):
        self.max_dist = max_dist
        self.distances = {}

    def tree_edge(self, e) -> None:
        out_v, in_v, weight = e
        if out_v not in self.distances:
            self.distances[out_v] = 0.0
        self.distances[in_v] = self.distances[out_v] + weight["weight"]
        if self.distances[in_v] > self.max_dist:
            raise rx.visit.StopSearch()
