from typing import Any

import numpy as np
import rustworkx as rx
import torch
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from torch_geometric.data import Data

from aerial_lane_bezier.dataset.bezier_graph import BezierGraph


def tg_to_rx(g: Data, scale=1.0) -> rx.PyDiGraph:
    """Convert a torch_geometric Data to a rustworkx PyDiGraph."""
    G = rx.PyDiGraph()
    X = g.x.numpy()
    for x in X:
        G.add_node(dict(pos=x[:2] * scale))
    for u, v in g.edge_index.t().numpy():
        G.add_edge(u, v, {"weight": np.linalg.norm(X[u, :2] - X[v, :2])})
    return G


def bg_to_rx(g: BezierGraph) -> rx.PyDiGraph:
    """Convert a BezierGraph to a rustworkx PyDiGraph."""
    G = rx.PyDiGraph()
    old_to_new = {}
    for n, data in g.nodes().items():
        old_to_new[n] = G.add_node(dict(pos=data["pos"]))
    for (idx, jdx), data in g.edges().items():
        G.add_edge(old_to_new[idx], old_to_new[jdx], data["l2"])
    return G


def rx_to_poly(g: rx.PyDiGraph, lane_width: float = 10.0) -> Polygon:
    """Convert a rustworkx PyDiGraph to a shapely Polygon."""
    X = np.array([x["pos"] for x in g.nodes()])
    E = np.array([(u, v) for u, v, _ in g.edge_index_map().values()])
    lines = [LineString(e) for e in X[E]]
    return unary_union(lines).buffer(lane_width / 2.0)


def interpolate_graph(
    g: rx.PyDiGraph,
    pos_key: str = "pos",
    nodes_per_m: float = 1.0,
) -> rx.PyDiGraph:
    """
    Interpolate a lane graph to a given precision.

    Interpolation is done by decomposing the graph into paths and interpolating
    each path to the desired precision. Edges between paths are then interpolated
    and added in. This is done so the interpolated graph is closer to the desired
    precision. Any cycles are handled by deleting one edge from the cycle and
    adding it back at the end.

    Parameters:
    ----------
    g: rx.PyDiGraph
        The lane graph to interpolate.
    pos_key: str
        The key in the node attributes to use for position.
    nodes_per_m: float
        The number of nodes per meter to interpolate to.
    """
    g = g.copy()
    for n_idx, data in zip(g.node_indices(), g.nodes()):
        data["n_idx"] = n_idx
    new = rx.PyDiGraph()

    edges_to_add = []
    deleted = []
    for cycle in rx.simple_cycles(g):
        cycle = list(cycle)
        c_idx, c_jdx = cycle[:2]
        if (c_idx, c_jdx) in deleted:
            continue
        g.remove_edge(c_idx, c_jdx)
        deleted.append((c_idx, c_jdx))
        edges_to_add.append((c_idx, c_jdx))

    runs = rx.collect_runs(g, lambda _: True)
    ends = []
    for run in runs:
        start_idx = new.add_node(run[0])
        if len(run) == 1:
            ends.append((run[-1]["n_idx"], start_idx))
            continue
        coords = np.array([p[pos_key] for p in run])
        s = np.hstack([[0.0], np.linalg.norm(np.diff(coords, axis=0), axis=1).cumsum()])
        dists = np.linspace(0, s.max(), max(int(np.round(s.max() * nodes_per_m)), 2))
        new_pos = interp1d(s, coords, axis=0, bounds_error=False)(dists)

        prev_idx, prev_s = start_idx, 0.0
        for p, s in zip(new_pos[1:-1:], dists[1:-1]):
            new_idx = new.add_node({pos_key: p})
            new.add_edge(prev_idx, new_idx, {"weight": s - prev_s})
            prev_idx, prev_s = new_idx, s

        end_idx = new.add_node(run[-1])
        new.add_edge(prev_idx, end_idx, {"weight": dists[-1] - prev_s})
        ends.append((run[-1]["n_idx"], end_idx))

    to_add = []
    poses = np.array([x[pos_key] for x in new.nodes()])
    for old_idx, new_idx in ends:
        for suc in g.successors(old_idx):
            d = np.linalg.norm(poses - suc[pos_key][None], axis=-1)
            d[new_idx] = 999999
            source_idx = new_idx
            target_idx = d.argmin()
            source_pose = poses[source_idx]
            target_pose = poses[target_idx]
            dist = np.linalg.norm(source_pose - target_pose)
            to_add.append((source_idx, target_idx, source_pose, target_pose, dist))

    for c_idx, c_jdx in edges_to_add:
        pose = g.get_node_data(c_idx)[pos_key]
        source_idx = np.linalg.norm(poses - pose[None], axis=-1).argmin()
        pose = g.get_node_data(c_jdx)[pos_key]
        target_idx = np.linalg.norm(poses - pose[None], axis=-1).argmin()
        source_pose = poses[source_idx]
        target_pose = poses[target_idx]
        dist = np.linalg.norm(source_pose - target_pose)
        to_add.append((source_idx, target_idx, source_pose, target_pose, dist))

    for source_idx, target_idx, source_pose, target_pose, dist in to_add:
        if source_idx != target_idx:
            num = int(np.maximum(np.round(dist * nodes_per_m), 2))
            prev_node = source_idx
            prev_dist = 0.0
            for s in np.linspace(0, dist, num)[1:-1]:
                w = s / dist
                p = (1 - w) * source_pose + w * target_pose
                next_idx = new.add_node({pos_key: p})
                new.add_edge(prev_node, next_idx, {"weight": s - prev_dist})
                prev_node, prev_dist = next_idx, s
            new.add_edge(prev_node, target_idx, {"weight": dist - prev_dist})
    return new


def obj_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, (tuple, list)):
        return type(x)(obj_to_device(y, device) for y in x)
    elif isinstance(x, dict):
        return type(x)(**{k: obj_to_device(v, device) for k, v in x.items()})
    return x
