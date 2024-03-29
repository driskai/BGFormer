from typing import Union

import numpy as np
import rustworkx as rx
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

Graph = Union[rx.PyDiGraph, Polygon]


def graph_iou(gt: Graph, pred: Graph, lane_width: float = 5.0) -> float:
    """Calculate the intersection over union (IoU) between two graphs."""
    if isinstance(gt, rx.PyDiGraph) or isinstance(pred, rx.PyDiGraph):
        if gt.num_edges() == 0 or pred.num_edges() == 0:
            return float(pred.num_edges() == gt.num_edges())
    if isinstance(gt, rx.PyDiGraph):
        gt = graph_to_poly(gt, lane_width)
    if isinstance(pred, rx.PyDiGraph):
        pred = graph_to_poly(pred, lane_width)
    if gt.is_empty or pred.is_empty:
        return float(pred.is_empty == gt.is_empty)
    return gt.intersection(pred).area / gt.union(pred).area


def graph_to_poly(g: Graph, width: float = 5.0) -> Polygon:
    """Convert a graph to a shapely polygon."""
    node_pos = np.array([x["pos"] for x in g.nodes()])
    edges = [(s, t) for s, t, _ in g.edge_index_map().values()]
    lines = [LineString(e).buffer(width / 2) for e in node_pos[np.array(edges)]]
    return unary_union(lines)
