import argparse
import os
import pickle as pkl
from glob import glob

import numpy as np
import pandas as pd
import rustworkx as rx

# import cdist
from scipy.spatial.distance import cdist
from tqdm import tqdm

from aerial_lane_bezier.metrics.apls import simple_apls
from aerial_lane_bezier.metrics.geo_topo import geo_topo_metric
from aerial_lane_bezier.metrics.iou import graph_iou
from aerial_lane_bezier.metrics.sda import sda
from aerial_lane_bezier.metrics.utils import interpolate_graph

nodes_per_pixel = 0.6


def filter_and_relabel_rx_graph(target, source, threshold=100):
    if source.num_nodes() == 0:
        return source
    # Extract positions and compute the distance matrix
    pos_target = np.array([target[node]["pos"] for node in target.node_indexes()])
    pos_source = np.array([source[node]["pos"] for node in source.node_indexes()])
    distance_matrix = cdist(pos_target, pos_source)
    # Determine nodes to keep based on the threshold
    keep_node = np.min(distance_matrix, axis=0) < threshold
    # Create a new graph
    new_graph = rx.PyDiGraph()
    node_mapping = {}
    # Add nodes to the new graph
    for i, node in enumerate(source.node_indexes()):
        if keep_node[i]:
            new_index = new_graph.add_node(source[node])
            node_mapping[node] = new_index
    # Add edges to the new graph
    for edge in source.edge_list():
        src, tgt = edge
        if src in node_mapping and tgt in node_mapping:
            new_src = node_mapping[src]
            new_tgt = node_mapping[tgt]
            edge_data = source.get_edge_data(src, tgt)
            new_graph.add_edge(new_src, new_tgt, edge_data)
    return new_graph


parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_dataset_root",
    type=str,
    help="Path to the root of the raw dataset. "
    + "Should be a directory ending with 'urbanlanegraph-dataset-pub-v1.1'",
)

if __name__ == "__main__":
    args = parser.parse_args()

    predicted_gpickles = glob(os.path.join("aggregated_output", "*.pickle"))

    tile_id_to_gt_pred_rx_interpolated = {}
    for predicted_gpickle in predicted_gpickles:
        with open(predicted_gpickle, "rb") as f:  #
            pred_nx = pkl.load(f)
        interpolated_pred_rx = rx.networkx_converter(pred_nx, keep_attributes=True)
        # interpolated_pred_rx = interpolate_graph(pred_rx, nodes_per_m=nodes_per_pixel)

        basename = os.path.basename(predicted_gpickle)
        city, _ = basename.split("_", 1)
        tile_id = basename.split("_metric")[0]
        origin = np.array(list(map(int, tile_id.split("_")[-2:])))
        corresponding_gt = os.path.join(
            args.raw_dataset_root, city, "tiles", "eval", tile_id + ".gpickle"
        )

        with open(corresponding_gt, "rb") as f:
            gt_nx = pkl.load(f)
        # Subtract the origin from every pos attribute of every node
        for node in gt_nx.nodes:
            gt_nx.nodes[node]["pos"] = np.array(gt_nx.nodes[node]["pos"]) - origin
        gt_rx = rx.networkx_converter(gt_nx, keep_attributes=True)
        interpolated_gt_rx = interpolate_graph(gt_rx, nodes_per_m=nodes_per_pixel)

        tile_id_to_gt_pred_rx_interpolated[tile_id] = (
            interpolated_gt_rx,
            interpolated_pred_rx,
        )

    tile_id_to_metrics = {}
    for tile_id, (interpolated_gt_rx, interpolated_pred_rx) in tqdm(
        tile_id_to_gt_pred_rx_interpolated.items()
    ):
        metric_dict = geo_topo_metric(
            interpolated_gt_rx, interpolated_pred_rx, interpolate=False
        )
        (
            metric_dict["sda20_precision"],
            metric_dict["sda20_recall"],
            metric_dict["sda20_accuracy"],
        ) = sda(interpolated_gt_rx, interpolated_pred_rx, threshold=20)
        (
            metric_dict["sda50_precision"],
            metric_dict["sda50_recall"],
            metric_dict["sda50_accuracy"],
        ) = sda(interpolated_gt_rx, interpolated_pred_rx, threshold=50)
        metric_dict["apls"] = simple_apls(interpolated_gt_rx, interpolated_pred_rx)
        metric_dict["iou"] = graph_iou(
            interpolated_gt_rx, interpolated_pred_rx, lane_width=10.0
        )

        tile_id_to_metrics[tile_id] = metric_dict

    # convert tile_id_to_metrics dict to dataframe
    df = pd.DataFrame.from_dict(tile_id_to_metrics, orient="index")
    # Save to file
    df.to_csv(f"aggregated_results.csv")

    print(df.mean())
