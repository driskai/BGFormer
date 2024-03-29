from typing import Tuple

import numpy as np
import rustworkx as rx
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def sda(
    gt: rx.PyDiGraph, pred: rx.PyDiGraph, threshold: float = 10, pos_key: str = "pos"
) -> Tuple[float, float, float]:
    """
    Calculate the split detection accuracy metric (SDA).

    Returns precision, recall and accruacy.
    """
    gt_split_points = np.array(
        [
            data[pos_key]
            for (n, data) in zip(gt.node_indices(), gt.nodes())
            if gt.out_degree(n) > 1
        ]
    )
    pred_split_points = np.array(
        [
            data[pos_key]
            for (n, data) in zip(pred.node_indices(), pred.nodes())
            if pred.out_degree(n) > 1
        ]
    )
    if len(gt_split_points) == 0 or len(pred_split_points) == 0:
        if len(pred_split_points) == len(gt_split_points) == 0:
            return (1.0, 1.0, 1.0)
        return (0.0, 0.0, 0.0)
    dists = cdist(gt_split_points, pred_split_points)
    assign_gt, assign_pred = linear_sum_assignment(dists)
    valid = np.sum(dists[assign_gt, assign_pred] < threshold)
    precision = valid / len(pred_split_points)
    recall = valid / len(gt_split_points)
    accuracy = valid / (len(pred_split_points) + len(gt_split_points) - valid)
    return (precision, recall, accuracy)
