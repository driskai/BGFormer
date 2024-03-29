import argparse
import glob
import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rustworkx as rx
import torch
import torch_geometric
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from aerial_lane_bezier.dataset.bezier_graph import BezierGraph
from aerial_lane_bezier.metrics.utils import interpolate_graph, tg_to_rx
from aerial_lane_bezier.model.bezier_detr import BezierDETR
from aerial_lane_bezier.model.image_processing import ImageProcessingBezierDETR

TILE_SIZE = 512
BOUNDAY = 16
NODE_PER_M = 0.6
OVERLAP = 14
H_THRESHOLD = 64


def compute_hungarian_matching(
    graph, previous_node_indices, node_indices, threshold=50
):
    """
    Computes the Hungarian matching between nodes in the combined graph and the translated graph.

    Parameters:
    - graph: The graph containing all previously processed nodes.
    - previous_node_indices: Indices of the nodes from the last processed tile in the combined graph.
    - translated_graph: The graph of the current tile, translated to fit in the overall image.

    Returns:
    - A list of pairs [(i, j), ...] indicating matched nodes between the two sets.
    """

    node_set1 = [graph.nodes[node]["pos"] for node in previous_node_indices]
    node_set2 = [graph.nodes[node]["pos"] for node in node_indices]

    direction_set1 = [graph.nodes[node]["direction"] for node in previous_node_indices]
    direction_set2 = [graph.nodes[node]["direction"] for node in node_indices]

    cost_matrix = np.zeros((len(node_set1), len(node_set2)), dtype=np.float32)

    for i, ((x1, y1), dir1) in enumerate(zip(node_set1, direction_set1)):
        for j, ((x2, y2), dir2) in enumerate(zip(node_set2, direction_set2)):
            spatial_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            direction_cost = 0 if np.dot(dir1, dir2) > 0 else 1e4
            cost_matrix[i, j] = spatial_distance + direction_cost

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_pairs = [
        (previous_node_indices[i], node_indices[j])
        for i, j in zip(row_indices, col_indices)
        if cost_matrix[i, j] < threshold
    ]

    return matched_pairs


def if_nodes_on_edges(result, current_index):
    edge_info = {"top": [], "bottom": [], "left": [], "right": []}
    nodes = list(result.nodes(data=True))
    for i, (node, data) in enumerate(nodes):
        x, y = data["pos"]
        if x < BOUNDAY:
            edge_info["left"].append(i + current_index)
        if x > TILE_SIZE - BOUNDAY:
            edge_info["right"].append(i + current_index)
        if y < BOUNDAY:
            edge_info["top"].append(i + current_index)
        if y > TILE_SIZE - BOUNDAY:
            edge_info["bottom"].append(i + current_index)

    return edge_info


def find_directed_triangles(G):
    triangles = []
    for n in G.nodes():
        for s in G.successors(n):
            for t in G.successors(s):
                if G.has_edge(n, t):
                    triangles.append((n, s, t))
    return triangles


def filter_graph(target, source, threshold=100):
    if not source.nodes():
        return source

    pos_target = np.array([target.nodes[n]["pos"] for n in target.nodes()])
    pos_source = np.array([source.nodes[n]["pos"] for n in source.nodes()])
    distance_matrix = cdist(pos_target, pos_source)

    is_close_to_target = np.min(distance_matrix, axis=0) < threshold
    for i, n in enumerate(list(source.nodes())):
        if not is_close_to_target[i]:
            source.remove_node(n)

    return source


def graph_to_poly(graph: nx.DiGraph, lane_width=5.0) -> Polygon:
    lanes = []
    for u, v in graph.edges():
        P0 = np.array(graph.nodes[u]["pos"])
        P1 = np.array(graph.nodes[v]["pos"])
        line = LineString([P0, P1])
        lanes.append(line.buffer(lane_width / 2))
    return unary_union(lanes)


def rotate_graph(graph, angle):
    center = TILE_SIZE // 2
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))

    for node, data in graph.nodes(data=True):
        # Rotate position
        x, y = data["pos"]
        x -= center
        y -= center
        new_x = cos_angle * x - sin_angle * y + center
        new_y = sin_angle * x + cos_angle * y + center
        data["pos"] = np.array([new_x, new_y], dtype=np.float32)

        # Rotate direction vector
        u, v = data["direction"]
        new_u = cos_angle * u - sin_angle * v
        new_v = sin_angle * u + cos_angle * v
        data["direction"] = np.array([new_u, new_v], dtype=np.float32)

    return graph


parser = argparse.ArgumentParser()

checkpoint_group = parser.add_mutually_exclusive_group()
checkpoint_group.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help="Directory to load checkpoints from. "
    + "If None, defaults to the checkpoints subdirectory. "
    + "Most recent checkpoint in the directory will be used.",
)
checkpoint_group.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="Name that this model was run under. If None, defaults to 'default'.",
)
parser.add_argument(
    "--node_threshold",
    type=float,
    default=0.6,
    help="Probability threshold below which to discard nodes. "
    + "Note this is also used to pre-filter potential edges.",
)
parser.add_argument(
    "--edge_threshold",
    type=float,
    default=0.6,
    help="Probability threshold below which to discard edges.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=24,
    help="Batch size.",
)
if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        if args.wandb_run_name is None:
            wandb_run_name = "default"
        else:
            wandb_run_name = args.wandb_run_name
        checkpoint_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "aerial_lane_bezier",
            "model",
            "checkpoints",
            wandb_run_name,
        )

    # Load the latest checkpoint
    possible_checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    latest_checkpoint_path = max(
        possible_checkpoint_paths, key=lambda x: int(x.split("checkpoint-")[-1])
    )

    image_processor = ImageProcessingBezierDETR.from_pretrained(latest_checkpoint_path)
    model = BezierDETR.from_pretrained(latest_checkpoint_path)

    image_directory = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "aerial_lane_bezier",
        "dataset",
        "processed_files",
        "raw",
        "eval_full_lgp",
    )
    image_paths = glob.glob(os.path.join(image_directory, "washington", "*.png"))

    for image_path in image_paths:
        image = Image.open(image_path)

        with open(image_path.replace(".png", ".gpickle"), "rb") as f:
            gt_graph = pickle.load(f)
        combined_bezier_graph = BezierGraph()
        width, height = image.size
        num_tiles_x = math.ceil(width / (TILE_SIZE - BOUNDAY))
        num_tiles_y = math.ceil(height / (TILE_SIZE - BOUNDAY))
        tile_nodes_dict = {}

        tiles_batch = []
        positions_batch = []
        edges_batch = {}
        # save image and predicted graphs
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image, alpha=0.8)

        def process_batch(tiles_batch, positions_batch):
            original_tiles_batch = list(tiles_batch)

            # Process the batch with the model
            with torch.no_grad():
                all_edge_counts = [[] for _ in original_tiles_batch]
                all_results = [[] for _ in original_tiles_batch]

                # Process and store results for all rotations
                for rotate_angle in [0, 90, 180, 270]:
                    rotated_tiles_batch = [
                        tile.rotate(
                            rotate_angle,
                            resample=Image.BICUBIC,
                            center=(TILE_SIZE // 2, TILE_SIZE // 2),
                        )
                        for tile in original_tiles_batch
                    ]
                    inputs = image_processor(
                        images=rotated_tiles_batch, return_tensors="pt"
                    )
                    outputs = model(
                        **inputs,
                        inference_edge_query_node_threshold=args.node_threshold,
                    )
                    tile_results = image_processor.post_process_object_detection(
                        outputs,
                        node_threshold=args.node_threshold,
                        edge_threshold=args.edge_threshold,
                        target_sizes=[[TILE_SIZE, TILE_SIZE]],
                    )

                    for idx, tile_result in enumerate(tile_results):
                        if tile_result["pyg_graph"] is not None:
                            edge_count = len(tile_result["pyg_graph"].edge_index[0])
                            all_edge_counts[idx].append(edge_count)
                            all_results[idx].append((rotate_angle, tile_result))

                # Select the best tile for each position
                best_results = [None] * len(original_tiles_batch)
                best_rotations = [0] * len(original_tiles_batch)

                for idx in range(len(original_tiles_batch)):
                    mean_edge_count = np.mean(all_edge_counts[idx])
                    closest_idx = np.argmin(
                        [abs(count - mean_edge_count) for count in all_edge_counts[idx]]
                    )
                    best_rotations[idx], best_results[idx] = all_results[idx][
                        closest_idx
                    ]

                for idx, (tile_result, (i, j, x, y), best_rotation) in enumerate(
                    zip(best_results, positions_batch, best_rotations)
                ):
                    if tile_result["pyg_graph"] is not None:
                        tile_bezier_graph = BezierGraph.from_pyg_graph(
                            tile_result["pyg_graph"]
                        )
                        tile_bezier_graph = rotate_graph(
                            tile_bezier_graph, best_rotation
                        )

                        triangles = find_directed_triangles(tile_bezier_graph)
                        for triangle in triangles:
                            # You can decide which edge to remove. Here, I'm removing the last edge.
                            if tile_bezier_graph.has_edge(triangle[0], triangle[2]):
                                tile_bezier_graph.remove_edge(triangle[0], triangle[2])

                        # First, get the isolated nodes
                        isolated_nodes = list(nx.isolates(tile_bezier_graph))
                        # Now, remove them from the graph
                        tile_bezier_graph.remove_nodes_from(isolated_nodes)
                        # Create a mapping from the old node names to the new ones
                        mapping = {
                            node: i for i, node in enumerate(tile_bezier_graph.nodes())
                        }

                        # Relabel the nodes
                        tile_bezier_graph = nx.relabel_nodes(tile_bezier_graph, mapping)
                        current_index = len(combined_bezier_graph.nodes())
                        edges_batch[(i, j)] = if_nodes_on_edges(
                            tile_bezier_graph, current_index
                        )

                        for node in tile_bezier_graph.nodes():
                            x_, y_ = tile_bezier_graph.nodes[node]["pos"]
                            tile_bezier_graph.nodes[node]["pos"] = np.array(
                                [x_ + x, y_ + y], dtype=np.float32
                            )

                        current_node_index = []
                        for node, data in tile_bezier_graph.nodes(data=True):
                            combined_bezier_graph.add_node(node + current_index, **data)
                            current_node_index.append(node + current_index)

                        for u, v, data in tile_bezier_graph.edges(data=True):
                            combined_bezier_graph.add_edge(
                                u + current_index, v + current_index, **data
                            )

                        tile_nodes_dict[(i, j)] = current_node_index

        start_time = time.time()
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                x = j * (TILE_SIZE - OVERLAP)
                y = i * (TILE_SIZE - OVERLAP)

                tile = image.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
                tiles_batch.append(tile)
                positions_batch.append((i, j, x, y))

                # If the batch has reached the desired size, process it
                if len(tiles_batch) == args.batch_size:
                    process_batch(tiles_batch, positions_batch)
                    tiles_batch = []
                    positions_batch = []
        # Process any remaining tiles in the batch
        if tiles_batch:
            process_batch(tiles_batch, positions_batch)

        global_matched_pairs = []
        combined_bezier_graph_merged = combined_bezier_graph.copy()
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                if j > 0:
                    left_nodes = edges_batch[(i, j - 1)]["right"]
                    current_nodes = edges_batch[(i, j)]["left"]
                    if left_nodes and current_nodes:
                        left_matched_pairs = compute_hungarian_matching(
                            combined_bezier_graph_merged,
                            left_nodes,
                            current_nodes,
                            H_THRESHOLD,
                        )
                        global_matched_pairs.extend(left_matched_pairs)
                if i > 0:
                    top_nodes = edges_batch[(i - 1, j)]["bottom"]
                    current_nodes = edges_batch[(i, j)]["top"]
                    if top_nodes and current_nodes:
                        top_matched_pairs = compute_hungarian_matching(
                            combined_bezier_graph_merged,
                            top_nodes,
                            current_nodes,
                            H_THRESHOLD,
                        )
                        global_matched_pairs.extend(top_matched_pairs)

        nodes_to_remove = set()

        # Adjust node positions based on tile bottom left values
        tile_bottom_left_x, tile_bottom_left_y = map(
            int, os.path.splitext(os.path.basename(image_path))[0].split("_")[-2:]
        )
        adjusted_node_positions = {
            node: (pos[0] - tile_bottom_left_x, pos[1] - tile_bottom_left_y)
            for node, pos in nx.get_node_attributes(gt_graph, "pos").items()
        }
        nx.set_node_attributes(gt_graph, adjusted_node_positions, "pos")

        # # Filter and plot the combined bezier graph
        mapping = {
            node: i for i, node in enumerate(combined_bezier_graph_merged.nodes())
        }

        # # Relabel the nodes
        combined_bezier_graph_merged = nx.relabel_nodes(
            combined_bezier_graph_merged, mapping
        )
        combined_bezier_graph_merged = filter_graph(
            gt_graph, combined_bezier_graph_merged.to_simple(nodes_per_m=NODE_PER_M), 50
        )
        nodes_to_remove = set()
        for node, data in combined_bezier_graph_merged.nodes(data=True):
            x, y = data["pos"]
            if x < 0 or x > width or y < 0 or y > height:
                nodes_to_remove.add(node)
        combined_bezier_graph_merged.remove_nodes_from(nodes_to_remove)

        mapping = {
            node: i for i, node in enumerate(combined_bezier_graph_merged.nodes())
        }

        # # Relabel the nodes
        combined_bezier_graph_merged = nx.relabel_nodes(
            combined_bezier_graph_merged, mapping
        )
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Now, create the filename for the pickled graph
        pickle_filename = f"aggregated_outputs/{base_name}_metric.pickle"

        # Use pickle to serialize the graph object
        with open(pickle_filename, "wb") as pickle_file:
            pickle.dump(combined_bezier_graph_merged, pickle_file)

        end_time = time.time()  # End timing
        duration = end_time - start_time  # Calculate the duration
        print(
            f"Time taken for generating combined Bezier graph for {os.path.basename(image_path)}: {duration} seconds"
        )
        pred_rx = rx.networkx_converter(
            combined_bezier_graph_merged, keep_attributes=True
        )

        # Metric calculations
        x_data = [
            np.append(data["pos"], [0.0, 0.0]) for _, data in gt_graph.nodes(data=True)
        ]
        x = torch.Tensor(np.array(x_data))
        edge_index = (
            torch.tensor(list(gt_graph.edges()), dtype=torch.long).t().contiguous()
        )
        gt_d = torch_geometric.data.Data(x=x, edge_index=edge_index)
        gt_g = tg_to_rx(gt_d)
        gt_g_interpolated = interpolate_graph(gt_g, nodes_per_m=NODE_PER_M)
        lines_x = []
        lines_y = []
        pos = [gt_g_interpolated[node] for node in gt_g_interpolated.node_indexes()]
        # Loop through all edges to build lists of line segments
        for edge in gt_g_interpolated.edge_list():
            start_pos = pos[edge[0]]["pos"]
            end_pos = pos[edge[1]]["pos"]
            lines_x.extend(
                [start_pos[0], end_pos[0], None]
            )  # 'None' to prevent lines from connecting
            lines_y.extend([start_pos[1], end_pos[1], None])

        ax.plot(lines_x, lines_y, color="lime", linewidth=1.2, alpha=0.8)

        lines_x = []
        lines_y = []
        pos = [pred_rx[node] for node in pred_rx.node_indexes()]
        # Loop through all edges to build lists of line segments
        for edge in pred_rx.edge_list():
            start_pos = pos[edge[0]]["pos"]
            end_pos = pos[edge[1]]["pos"]
            lines_x.extend(
                [start_pos[0], end_pos[0], None]
            )  # 'None' to prevent lines from connecting
            lines_y.extend([start_pos[1], end_pos[1], None])

        ax.plot(lines_x, lines_y, color="red", linewidth=1, alpha=1)

        plt.savefig(
            f"aggregated_outputs/{os.path.splitext(os.path.basename(image_path))[0]}_metric.png",
            dpi=300,
        )
        plt.close()
