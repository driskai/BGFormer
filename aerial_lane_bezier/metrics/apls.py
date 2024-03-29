import numpy as np
import rustworkx as rx
from scipy.spatial import KDTree
from tqdm import tqdm

from .utils import interpolate_graph


def simple_apls(
    gt: rx.PyDiGraph,
    pred: rx.PyDiGraph,
    pos_key: str = "pos",
    weight: str = "length",
    interpolate: bool = False,
    verbose: bool = False,
    **kwargs,
) -> float:
    """
    Compuate a simplified version of the APLS metric.

    This is the APLS metric as defined here https://arxiv.org/abs/1807.01232
    in section 4.2 without any augmentation or node snapping in 4.2.1-3.
    """
    if interpolate:
        gt = interpolate_graph(gt, **kwargs)
        pred = interpolate_graph(pred, **kwargs)

    if pred.num_nodes() == 0 or gt.num_nodes() == 0:
        # if both graphs are empty return the perfect score else 0
        return float(gt.num_nodes() == pred.num_nodes())

    X_gt = np.array([n[pos_key] for n in gt.nodes()])
    X_pred = np.array([n[pos_key] for n in pred.nodes()])

    tree = KDTree(X_pred)
    _, idxs = tree.query(X_gt)
    gt_idxs, pred_idxs = list(gt.node_indexes()), list(pred.node_indexes())
    gt_to_pred = {gt_idxs[i]: pred_idxs[j] for i, j in enumerate(idxs)}

    tot, count = 0.0, 0

    def edge_cost_fn(x):
        if isinstance(x, dict):
            return x["weight"]
        return x

    all_paths_gt = rx.all_pairs_dijkstra_path_lengths(gt, edge_cost_fn=edge_cost_fn)
    all_paths_pred = rx.all_pairs_dijkstra_path_lengths(pred, edge_cost_fn=edge_cost_fn)
    all_paths_pred = {k: v for k, v in all_paths_pred.items()}

    for gt_source, paths in (
        tqdm(all_paths_gt.items()) if verbose else all_paths_gt.items()
    ):
        pred_source = gt_to_pred[gt_source]
        for gt_target, length in paths.items():
            if length == 0.0:
                # if the ground truth length is 0. then do not count it
                continue

            pred_target = gt_to_pred[gt_target]
            if pred_target == pred_source:
                # if the predicted source and target are mapped to the same node
                # then do not count it
                continue

            count += 1
            if pred_target not in all_paths_pred[pred_source]:
                tot += 1.0
                continue
            pred_length = all_paths_pred[pred_source][pred_target]
            tot += min(1.0, abs(length - pred_length) / length)
    return (1 - tot / count) if count > 0 else 0.0
