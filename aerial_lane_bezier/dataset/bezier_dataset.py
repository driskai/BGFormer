import glob
import json
import os
import pickle
import shutil
from collections import deque
from functools import cached_property, lru_cache
from typing import List, Literal, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from torch_geometric.data import Dataset
from tqdm import tqdm

from aerial_lane_bezier.dataset.bezier_graph import BezierGraph
from aerial_lane_bezier.dataset.graph_to_bezier import (
    BezierOptimisationMetrics,
    Graph2Bezier,
)

Image.MAX_IMAGE_PIXELS = None

# Global variables for the Urban Lane Graph dataset. These should not be changed.
CITIES = [
    "austin",
    "detroit",
    "miami",
    "paloalto",
    "pittsburgh",
    "washington",
]
TILE_DIMENSION = 5000


class AerialBezierGraphDataset(Dataset):
    """
    Dataset class for bezier lane graphs overlaid over aerial images.

    Parameters
    ----------
    raw_dataset_root : str
        Path to the root of the raw dataset (i.e. directory ending with
        urbanlanegraph-dataset-pub-v1.1
    split : Literal["train", "eval_succ_lgp", "eval_full_lgp"]
        Whether to process and return the training or one of the evaluation splits of
        the dataset.
    processed_dataset_root : str, optional
        Path to the root where the processed dataset should be saved. If None, defaults
        to processed_files in the same directory as this file.
    output_image_crop_size : int, optional
        Size of the crops to generate, in pixels. Defaults to 512.
    agglomerative_clustering_multiple : int, optional
        Multiple of the output_image_crop_size to use. This is used to determine the
        distance_threshold parameter passed to the sklearn AgglomerativeClustering
        algorithm, as distance_threshold = agglomerative_clustering_multiple *
        output_image_crop_size. Defaults to 2.
    random_sampling_tile_fraction : float, optional
        Fraction of the eventual dataset that will be made up of uniformly randomly
        sampled tiles. These have the disadvantage that they will have a much greater
        fraction of empty lane graphs (and even sometimes will contain real roads that
        haven't been annotated). However, including some of these is important to ensure
        good performance on the downstream task, of predicting lane graphs over an
        entire area (lots of which will be empty space). Defaults to 0.1.
    avoid_degree_1_nodes : bool, optional
        Whether to avoid returning crops that contain degree 1 nodes. Defaults to True.
    degree_1_radius_fraction : float, optional
        If avoid_degree_1_nodes is True, this is the fractional radius to search for
        degree 1 nodes around the center of the cluster. Defaults to 0.75.
    g2bkwargs : dict, optional
        Keyword arguments to pass to the Graph2Bezier class. Defaults to
        {"medfilt_kernel_size": 3}.
    max_num_optimisation_iterations : int, optional
        Maximum number of iterations to run the Bezier optimisation for. Defaults to
        1000.
    bezier_optimisation_batch_size : int, optional
        Number of crops to optimise Bezier curves for at once. Defaults to 500.
    **kwargs
        Keyword arguments to pass to the torch_geometric.data.Dataset class.
    """

    def __init__(
        self,
        raw_dataset_root: str,
        split: Literal["train", "eval", "eval_succ_lgp", "eval_full_lgp"],
        processed_dataset_root: str = None,
        output_image_crop_size: int = 512,
        agglomerative_clustering_multiple: int = 2,
        random_sampling_tile_fraction: float = 0.1,
        avoid_degree_1_nodes: bool = True,
        degree_1_radius_fraction: float = 0.75,
        g2bkwargs: dict = {"medfilt_kernel_size": 3},
        max_num_optimisation_iterations: int = 1000,
        bezier_optimisation_batch_size: int = 500,
        **kwargs,
    ):
        self.split = split

        self.output_image_crop_size = output_image_crop_size

        self.agglomerative_clustering_multiple = agglomerative_clustering_multiple
        self.random_sampling_tile_fraction = random_sampling_tile_fraction
        self.avoid_degree_1_nodes = avoid_degree_1_nodes
        self.degree_1_radius_fraction = degree_1_radius_fraction

        self.raw_dataset_root = raw_dataset_root

        self.g2bkwargs = g2bkwargs
        self.max_num_optimisation_iterations = max_num_optimisation_iterations
        self.bezier_optimisation_batch_size = bezier_optimisation_batch_size

        if processed_dataset_root is None:
            processed_dataset_root = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "processed_files"
            )

        # Have to set this before super init to allow preprocessing
        self.root = processed_dataset_root

        self.preprocess_data()

        super().__init__(self.root, **kwargs)

    def _split_into_sub_tiles(
        self,
        node_positions: np.ndarray,
        node_indices: np.ndarray,
        tile_bottom_left_x: int,
        tile_bottom_left_y: int,
        global_image_width: int,
        global_image_height: int,
        node_degrees: Optional[np.ndarray] = None,
    ) -> Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Split the given tile (represented by the node positions) into sub-tiles.

        Both clusters the existing nodes to generate sub-tiles guaranteed to contain
        lanes, and also samples random sub-tiles.

        Parameters
        ----------
        node_positions : np.ndarray
            Array of shape (N, 2) containing the node positions.
        node_indices : np.ndarray
            Array of shape (N,) containing the node indices.
        tile_bottom_left_x : int
            The x coordinate of the bottom left of the tile.
        tile_bottom_left_y : int
            The y coordinate of the bottom left of the tile.
        node_degrees : Optional[np.ndarray], optional
            Array of shape (N,) containing the node degrees. If None, this is ignored.
            Defaults to None. Used to avoid returning crops that contain degree 1.

        Returns
        -------
        Tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]
            Tuple of (window_bottom_lefts, clustered_node_indices,
            random_window_bottom_lefts, random_node_indices). window_bottom_lefts
            is a list of bottom left coordinates of the crops. clustered_node_indices
            is a list of lists of node indices, where the ith list of node indices
            corresponds to the ith crop. Their random counterparts are the same, but
            randomly sampled rather than clustered.
        """
        # Cluster the nodes
        clusters = AgglomerativeClustering(
            distance_threshold=self.output_image_crop_size
            * float(self.agglomerative_clustering_multiple),
            n_clusters=None,
        ).fit(node_positions)

        # Construct KD tree of node positions for later querying
        node_position_kd_tree = scipy.spatial.cKDTree(node_positions)
        # Radius to pre-search for nodes sitting within the cluster.
        radius = self.output_image_crop_size / np.sqrt(2)

        clustered_window_bottom_lefts = []
        clustered_node_indices = []

        # Min and max x and y to avoid generating clusters (tiles) that go outside
        # the overall tile.
        buffer = self.output_image_crop_size / 2.0
        min_center_x = int(tile_bottom_left_x + buffer)
        min_center_y = int(tile_bottom_left_y + buffer)
        max_center_x = int(tile_bottom_left_x + TILE_DIMENSION - buffer)
        max_center_y = int(tile_bottom_left_y + TILE_DIMENSION - buffer)
        # And ensure the centers lie within the global image
        max_center_x = min(max_center_x, global_image_width - buffer)
        max_center_y = min(max_center_y, global_image_height - buffer)
        min_center_x = max(min_center_x, buffer)
        min_center_y = max(min_center_y, buffer)
        if min_center_x > max_center_x or min_center_y > max_center_y:
            print("Warning: Tile is too small to generate any crops.")
            return [], [], [], []

        existing_bottom_lefts = set()

        for cluster in np.unique(clusters.labels_):
            cluster_mask = clusters.labels_ == cluster
            cluster_X = node_positions[cluster_mask]
            cluster_center = np.mean(cluster_X, axis=0).astype(int)
            # We need this cluster center to create a cluster that sits within the
            # overall tile.
            cluster_center[0] = np.clip(cluster_center[0], min_center_x, max_center_x)
            cluster_center[1] = np.clip(cluster_center[1], min_center_y, max_center_y)

            candidate_indices = node_position_kd_tree.query_ball_point(
                cluster_center, radius
            )

            # Don't save this cluster if it contains nodes of degree 1 (and that option
            # is enabled)
            if self.avoid_degree_1_nodes and node_degrees is not None:
                # Do not include this cluster if it contains a degree 1 node close to
                # the center (defined as within a fraction of the radius)
                close_indices = node_position_kd_tree.query_ball_point(
                    cluster_center, radius * self.degree_1_radius_fraction
                )
                if np.any(node_degrees[close_indices] == 1):
                    continue

            cluster_indices = node_indices[candidate_indices]
            window_bottom_left = cluster_center - self.output_image_crop_size / 2.0

            window_bottom_left = window_bottom_left.astype(int)
            if tuple(window_bottom_left) in existing_bottom_lefts:
                continue
            else:
                existing_bottom_lefts.add(tuple(window_bottom_left))
            clustered_window_bottom_lefts.append(window_bottom_left.astype(int))
            clustered_node_indices.append(cluster_indices)

        # Now randomly sample some tiles too.
        # First, we work out the multiple of the number of clusters to sample, in
        # order to end up with self.random_sampling_tile_fraction of the total number of
        # samples.
        assert (
            self.random_sampling_tile_fraction < 0.5
        ), "Random samples shouldn't make up more than half of the eventual dataset."
        num_random_samples = (
            (self.random_sampling_tile_fraction)
            / (1 - self.random_sampling_tile_fraction)
            * len(clustered_node_indices)
        )
        # We'll take the random samples only from the area spanned by the actual node
        # positions, to try and avoid sampling areas where there is no image (resulting
        # in black tiles).
        min_random_center_x = int(np.min(node_positions[:, 0]) + buffer)
        min_random_center_y = int(np.min(node_positions[:, 1]) + buffer)
        max_random_center_x = int(np.max(node_positions[:, 0]) - buffer)
        max_random_center_y = int(np.max(node_positions[:, 1]) - buffer)
        # And check these values do not lie outside the overall tile
        min_random_center_x = max(min_random_center_x, min_center_x)
        min_random_center_y = max(min_random_center_y, min_center_y)
        max_random_center_x = min(max_random_center_x, max_center_x)
        max_random_center_y = min(max_random_center_y, max_center_y)

        if (
            min_random_center_x > max_random_center_x
            or min_random_center_y > max_random_center_y
        ):
            num_random_samples = 0
        else:
            num_random_samples = int(np.round(num_random_samples))
        random_window_bottom_lefts = []
        random_node_indices = []
        for _ in range(num_random_samples):
            # Sample a random window bottom left
            window_center_x = np.random.randint(
                min_random_center_x, max_random_center_x
            )
            window_center_y = np.random.randint(
                min_random_center_y, max_random_center_y
            )
            window_bottom_left = (
                np.array([window_center_x, window_center_y])
                - self.output_image_crop_size / 2.0
            )
            window_bottom_left = window_bottom_left.astype(int)
            if tuple(window_bottom_left) in existing_bottom_lefts:
                continue
            else:
                existing_bottom_lefts.add(tuple(window_bottom_left))
            random_window_bottom_lefts.append(window_bottom_left)

            # Find all nodes within this window
            candidate_indices = node_position_kd_tree.query_ball_point(
                window_bottom_left + self.output_image_crop_size / 2.0, radius
            )
            random_indices = node_indices[candidate_indices]
            random_node_indices.append(random_indices)

        return (
            clustered_window_bottom_lefts,
            clustered_node_indices,
            random_window_bottom_lefts,
            random_node_indices,
        )

    def _crop_train_data(self):
        """Preprocess the larger crops into tiles of size output_image_crop_size."""
        all_filepaths = set()

        for city in CITIES:
            print(f"Preprocessing {city}.")

            combined_processed_directory = os.path.join(self.crop_path, city)
            combined_plotting_directory = os.path.join(self.plot_path, city)
            os.makedirs(combined_processed_directory, exist_ok=True)
            os.makedirs(combined_plotting_directory, exist_ok=True)

            tile_filename_to_expected_plot_path = {}

            # Get all the tile filenames for this city
            tile_filenames = self.get_tile_filenames_from_city(city)

            filtered_tile_filenames = []
            filtered_tile_graphs = []
            for tile_filename in tqdm(tile_filenames, desc="Loading tile graphs."):
                tile_basename = os.path.basename(tile_filename).split(".gpickle")[0]
                expected_plot_path = os.path.join(
                    combined_plotting_directory, tile_basename + ".png"
                )
                tile_filename_to_expected_plot_path[tile_filename] = expected_plot_path

                # If this plot already exists, the tile has already been processed, so
                # we should skip it
                if os.path.exists(expected_plot_path):
                    continue

                filtered_tile_filenames.append(tile_filename)
                filtered_tile_graphs.append(pickle.load(open(tile_filename, "rb")))

            # Get global image dimensions to ensure our crops do not go outside of this
            global_image = self.get_image_from_city(city)
            # This is not an expensive operation as it doesn't require loading the
            # image into memory.
            global_image_width, global_image_height = global_image.size

            for tile_graph, tile_filename in tqdm(
                zip(filtered_tile_graphs, filtered_tile_filenames),
                desc="Computing clusters for each tile.",
                total=len(filtered_tile_graphs),
            ):
                basename = os.path.basename(tile_filename).split(".gpickle")[0]
                tile_bottom_left_x, tile_bottom_left_y = map(
                    int, basename.split("_")[-2:]
                )
                node_indices = sorted(list(tile_graph.nodes))
                # Check that the node indices are contiguous
                if not (node_indices == list(range(len(node_indices)))):
                    raise ValueError("Tile node indices are not contiguous.")
                node_positions = np.array(
                    [tile_graph.nodes[n]["pos"] for n in node_indices]
                )
                node_degrees = np.array(
                    [tile_graph.degree[n] for n in node_indices], dtype=int
                )
                (
                    clustered_bottom_lefts,
                    clustered_indices,
                    random_bottom_lefts,
                    random_node_indices,
                ) = self._split_into_sub_tiles(
                    node_positions,
                    np.array(node_indices),
                    tile_bottom_left_x,
                    tile_bottom_left_y,
                    global_image_width,
                    global_image_height,
                    node_degrees=node_degrees,
                )

                # Create subgraphs for each cluster
                clustered_subgraphs = []
                for clustered_bottom_left, cluster_indices in zip(
                    clustered_bottom_lefts, clustered_indices
                ):
                    x_origin, y_origin = clustered_bottom_left

                    # Further filter down to just the cluster indices that actually
                    # exist within the tile bounding box defined by
                    # [
                    #     y_origin : y_origin + self.output_image_crop_size,
                    #     x_origin : x_origin + self.output_image_crop_size,
                    # ]
                    filtered_cluster_indices = []
                    for cluster_index in cluster_indices:
                        pos = tile_graph.nodes[cluster_index]["pos"]
                        if (
                            pos[0] >= x_origin
                            and pos[0] < x_origin + self.output_image_crop_size
                            and pos[1] >= y_origin
                            and pos[1] < y_origin + self.output_image_crop_size
                        ):
                            filtered_cluster_indices.append(cluster_index)

                    clustered_subgraphs.append(
                        tile_graph.subgraph(filtered_cluster_indices)
                    )

                random_subgraphs = []
                for random_bottom_left, random_indices in zip(
                    random_bottom_lefts, random_node_indices
                ):
                    x_origin, y_origin = random_bottom_left

                    filtered_random_indices = []
                    for random_index in random_indices:
                        pos = tile_graph.nodes[random_index]["pos"]
                        if (
                            pos[0] >= x_origin
                            and pos[0] < x_origin + self.output_image_crop_size
                            and pos[1] >= y_origin
                            and pos[1] < y_origin + self.output_image_crop_size
                        ):
                            filtered_random_indices.append(random_index)

                    random_subgraphs.append(
                        tile_graph.subgraph(filtered_random_indices)
                    )

                for bottom_left, subgraph in zip(
                    clustered_bottom_lefts + random_bottom_lefts,
                    clustered_subgraphs + random_subgraphs,
                ):
                    file = f"{bottom_left[0]}_{bottom_left[1]}" ".gpickle"
                    processed_filename = os.path.join(
                        combined_processed_directory,
                        file,
                    )
                    if processed_filename in all_filepaths:
                        print(
                            f"WARNING: Processed filename {processed_filename} "
                            "exists. This can happen occasionally due to clipping "
                            "the tiles to not go outside the image. However, if "
                            "this happens frequently, it may be a bug."
                        )
                        continue
                    else:
                        all_filepaths.add(processed_filename)
                    with open(processed_filename, "wb") as f:
                        pickle.dump(nx.DiGraph(subgraph), f, pickle.HIGHEST_PROTOCOL)

                # Plot the tiles over the source tile graph, save this as a png.
                # This has two purposes; as a debugging tool, as well as as a "checksum"
                # to ensure that the tiles have been generated, to check if
                # preprocessing needs to be run.
                plt.figure(figsize=(20, 20))
                ax = plt.gca()
                ax.set_aspect("equal")
                # Plot the tile graph
                ax.scatter(
                    node_positions[:, 0],
                    node_positions[:, 1],
                    c="red",
                    s=1,
                    alpha=0.5,
                )
                # Plot each clustered bounding box in black
                for bottom_left in clustered_bottom_lefts:
                    bounding_box = plt.Rectangle(
                        bottom_left,
                        self.output_image_crop_size,
                        self.output_image_crop_size,
                        fill=False,
                        color="black",
                    )
                    ax.add_patch(bounding_box)

                # Plot each random bounding box in blue
                for bottom_left in random_bottom_lefts:
                    bounding_box = plt.Rectangle(
                        bottom_left,
                        self.output_image_crop_size,
                        self.output_image_crop_size,
                        fill=False,
                        color="blue",
                    )
                    ax.add_patch(bounding_box)
                basename = os.path.basename(tile_filename).split(".gpickle")[0]
                plt.savefig(
                    os.path.join(combined_plotting_directory, basename + ".png")
                )
                plt.close()

    def _copy_eval_data(self):
        for city in CITIES:
            print(f"Preprocessing {city}.")

            combined_processed_directory = os.path.join(self.crop_path, city)
            os.makedirs(combined_processed_directory, exist_ok=True)

            # Get the eval split files for this city
            if self.split == "eval_succ_lgp":
                eval_subdirectory = os.path.join("successor-lgp", "eval")
            elif self.split == "eval_full_lgp":
                eval_subdirectory = os.path.join("tiles", "eval")
            else:
                raise ValueError(f"Invalid split {self.split}.")

            eval_filenames = glob.glob(
                os.path.join(self.raw_dataset_root, city, eval_subdirectory, "*")
            )

            for filename in eval_filenames:
                basename = os.path.basename(filename)
                processed_filename = os.path.join(
                    combined_processed_directory, basename
                )
                # Copy the file
                shutil.copy(filename, processed_filename)

    def preprocess_data(self):
        # We only crop the train splits (or when "eval" is specified, for cropping the
        # full LGP files). For the other splits we use the provided files.
        if self.split in ("train", "eval"):
            self._crop_train_data()
        else:
            self._copy_eval_data()

    def _swap_single_str_instance_in_path(
        self, path: str, old: str, new: str, sep=os.path.sep
    ) -> str:
        # Simply replace "/raw/" in the crop path, double checking that it exists only
        # once
        # Use the system separator from os module for cross-platform compatibility
        string_to_replace = sep + old + sep  # e.g. "/raw/"
        if path.count(string_to_replace) != 1:
            raise ValueError(
                f"Path {path} does not contain exactly 1 instance of "
                f"{string_to_replace}."
            )
        new_path = path.replace(string_to_replace, sep + new + sep)
        return new_path

    @property
    def split_dir(self) -> str:
        """Get the directory in which to save the current split."""
        split_dir = os.path.join(self.raw_dir, self.split)
        os.makedirs(split_dir, exist_ok=True)
        return split_dir

    @property
    def crop_path(self) -> str:
        """Get the directory in which to save the preprocessed crop files."""
        # If split is train, we want to separate this by possible crops
        if self.split in ("train", "eval"):
            crop_path = os.path.join(
                self.split_dir,
                f"{self.output_image_crop_size}_{self.agglomerative_clustering_multiple}",
            )
        else:  # Otherwise, we will be doing no cropping
            crop_path = self.split_dir

        os.makedirs(crop_path, exist_ok=True)
        return crop_path

    @property
    def plot_path(self) -> str:
        """Get the directory in which to save the "checksum" plots."""
        return self._swap_single_str_instance_in_path(self.crop_path, "raw", "plot")

    def get_image_from_city(self, city_name: str) -> Image:
        """Get Image object for the given city name. Note does not load into memory."""
        image = Image.open(
            os.path.join(self.raw_dataset_root, city_name, city_name + ".png")
        )
        return image

    @lru_cache(maxsize=None)
    def get_tile_filenames_from_city(self, city_name: str) -> List[str]:
        """Get all raw tile filenames for the given city name."""
        tile_filenames = glob.glob(
            os.path.join(
                self.raw_dataset_root, city_name, "tiles", self.split, "*.gpickle"
            )
        )
        return tile_filenames

    @property
    def raw_paths(self):
        """Get paths of all .gpickle files in any subdirectory of self.crop_path."""
        all_raw_paths = []
        for city in CITIES:
            combined_processed_directory = os.path.join(self.crop_path, city)
            raw_paths = glob.glob(
                os.path.join(combined_processed_directory, "*.gpickle")
            )
            all_raw_paths.extend(raw_paths)
        return all_raw_paths

    def raw_to_processed_path(self, raw_path: str):
        """Convert a path to a raw file to a path to the associated processed file."""
        return self._swap_single_str_instance_in_path(
            self._swap_single_str_instance_in_path(raw_path, "raw", "processed"),
            ".gpickle",
            ".pt",
            sep="",
        )

    @cached_property
    def processed_paths(self):
        """Return all processed files that the dataset will look for at init."""
        return [self.raw_to_processed_path(r) for r in self.raw_paths]

    def _get_data_and_metrics_for_city(
        self, city: str, return_image_crops: bool = True
    ):
        combined_raw_directory = os.path.join(self.crop_path, city)
        # Get all of the raw tile paths for this city
        raw_paths = glob.glob(os.path.join(combined_raw_directory, "*.gpickle"))
        raw_to_processed_path = {
            raw_path: self.raw_to_processed_path(raw_path) for raw_path in raw_paths
        }

        global_image = self.get_image_from_city(city)
        if return_image_crops:
            print("Loading global image. This will take some time (and RAM).")
            rgb_global = np.array(global_image)
            rgb_global = np.ascontiguousarray(
                cv2.cvtColor(rgb_global, cv2.COLOR_BGR2RGB)
            )
            print("Finished loading global image.")

        num_processed_paths = len(raw_to_processed_path.values())

        # Running lists for batching
        rgb_crops = []
        tile_graphs = []
        x_origins = []
        y_origins = []
        batch_raw_paths = []

        # Overall list that will be returned at the end
        data_list = []

        # List of metric objects for each graph
        metric_list = []

        for raw_path_index, raw_path in tqdm(
            enumerate(raw_paths),
            desc=f"Processing tiles for {city}",
            total=len(raw_paths),
        ):
            # Extract origin coordinates for this tile
            basename = os.path.basename(raw_path).split(".gpickle")[0]
            bottom_left, bottom_right = basename.split("_")
            x_origin = int(bottom_left)
            y_origin = int(bottom_right)

            # Make sure our x and y origin keeps us within the bounds of the global
            # image
            # Note we can't fix this at this point, as then the nodes contained within
            # this crop would be incorrect.
            # However, this should never happen, as we have already checked for it
            if x_origin < 0:
                raise ValueError(f"x_origin {x_origin} is less than 0 for {raw_path}.")
            if y_origin < 0:
                raise ValueError(f"y_origin {y_origin} is less than 0 for {raw_path}.")
            if x_origin > global_image.width - self.output_image_crop_size:
                raise ValueError(
                    f"x_origin {x_origin} is greater than the max allowed value of "
                    + f"{global_image.width - self.output_image_crop_size} for "
                    + str(raw_path)
                )
            if y_origin > global_image.height - self.output_image_crop_size:
                raise ValueError(
                    f"y_origin {y_origin} is greater than the max allowed value of "
                    + f"{global_image.height - self.output_image_crop_size} for "
                    + str(raw_path)
                )

            # Load the tile graph
            with open(raw_path, "rb") as f:
                tile_graph = pickle.load(f)

            if return_image_crops:
                # Get the correct crop of the global image
                rgb_crop = rgb_global[
                    y_origin : y_origin + self.output_image_crop_size,
                    x_origin : x_origin + self.output_image_crop_size,
                ].copy()
                assert rgb_crop.shape == (
                    self.output_image_crop_size,
                    self.output_image_crop_size,
                    3,
                ), f"RGB crop had unexpected shape: {rgb_crop.shape}"
            else:
                rgb_crop = None

            # Rescale the node positions to overlay the image
            for n, d in tile_graph.nodes(data=True):
                pos = d["pos"]
                # Adjust the coordinates
                pos = np.array(pos)
                pos[0] -= x_origin
                pos[1] -= y_origin

                # Update the node position
                tile_graph.nodes[n]["pos"] = pos

            # Append to running lists. These will be computed in batches (see next
            # if statement)
            rgb_crops.append(rgb_crop)
            tile_graphs.append(tile_graph)
            x_origins.append(x_origin)
            y_origins.append(y_origin)
            batch_raw_paths.append(raw_path)

            if (len(tile_graphs) > 0) and (
                len(tile_graphs) > self.bezier_optimisation_batch_size
                or (raw_path_index == (num_processed_paths - 1))
            ):
                assert len(rgb_crops) == len(tile_graphs)

                (
                    g2b,
                    batch_index_to_bezier_indices,
                    batch_index_to_source_indices,
                    removed_batch_indices,
                ) = Graph2Bezier.batch_lane_graphs(tile_graphs, **self.g2bkwargs)

                batched_bezier_graphs = g2b.optimise_bezier_graph(
                    max_num_iterations=self.max_num_optimisation_iterations
                )

                # In order to avoid re-calculating everything, we need to remove the
                # raw files for the batches which were removed.
                for batch_index in removed_batch_indices:
                    print("Removing malformed raw file.")
                    raw_file_to_remove = batch_raw_paths[batch_index]
                    os.remove(raw_file_to_remove)

                # Compute metrics
                metric_list += BezierOptimisationMetrics.metrics_from_batch(
                    batched_bezier_graphs,
                    g2b,
                    batch_index_to_bezier_indices,
                    batch_index_to_source_indices,
                )

                for (
                    batch_index,
                    bezier_indices,
                ) in batch_index_to_bezier_indices.items():
                    bezier_graph = BezierGraph(
                        nx.subgraph(batched_bezier_graphs, bezier_indices)
                    )
                    rgb_crop = rgb_crops[batch_index]
                    x_origin = x_origins[batch_index]
                    y_origin = y_origins[batch_index]
                    raw_path = batch_raw_paths[batch_index]

                    pyg_bezier_graph = bezier_graph.to_pyg_graph()

                    data = {
                        "image": (
                            torch.from_numpy(rgb_crop) if rgb_crop is not None else None
                        ),
                        "graph": pyg_bezier_graph,
                        "metadata": {
                            "x_origin": x_origin,
                            "y_origin": y_origin,
                            "raw_path": raw_path,
                            "city": city,
                        },
                        "source_graph": tile_graphs[batch_index],
                    }
                    data_list.append(data)

                rgb_crops = []
                tile_graphs = []
                x_origins = []
                y_origins = []
                batch_raw_paths = []

        return data_list, metric_list

    def _process_and_save_city(self, city: str):
        combined_raw_directory = os.path.join(self.crop_path, city)
        # Get all of the raw tile paths for this city
        raw_paths = glob.glob(os.path.join(combined_raw_directory, "*.gpickle"))
        raw_to_processed_path = {
            raw_path: self.raw_to_processed_path(raw_path) for raw_path in raw_paths
        }

        # Ensure the directories exist. They should all follow the same stucture
        # (for each city), so just running this with the first one should work.
        os.makedirs(
            os.path.dirname(list(raw_to_processed_path.values())[0]), exist_ok=True
        )

        data_list, metrics_list = self._get_data_and_metrics_for_city(city)

        for data_dict in data_list:
            raw_path = data_dict["metadata"]["raw_path"]
            processed_path = raw_to_processed_path[raw_path]

            # Save the processed file
            torch.save(data_dict, processed_path)

        return metrics_list

    def _convert_eval_data_to_processed(self):
        for city in CITIES:
            combined_raw_directory = os.path.join(self.crop_path, city)
            # Get all of the raw tile paths for this city
            raw_paths = glob.glob(os.path.join(combined_raw_directory, "*.gpickle"))
            id_to_nx = {}
            id_to_image = {}
            if self.split == "eval_succ_lgp":
                # Filenames have a -graph.gpickle and -rgb.png suffix
                for raw_path in raw_paths:
                    basename = os.path.basename(raw_path)
                    id = basename.split("-graph.gpickle")[0]
                    with open(raw_path, "rb") as f:
                        id_to_nx[id] = pickle.load(f)
                # Need to avoid viz files
                raw_images = glob.glob(
                    os.path.join(combined_raw_directory, "*-rgb.png")
                )
                for raw_image in raw_images:
                    basename = os.path.basename(raw_image)
                    id = basename.split("-rgb.png")[0]
                    id_to_image[id] = Image.open(raw_image)
            elif self.split == "eval_full_lgp":
                for raw_path in raw_paths:
                    basename = os.path.basename(raw_path)
                    id = basename.split(".gpickle")[0]
                    with open(raw_path, "rb") as f:
                        id_to_nx[id] = pickle.load(f)
                raw_images = glob.glob(os.path.join(combined_raw_directory, "*.png"))
                for raw_image in raw_images:
                    basename = os.path.basename(raw_image)
                    id = basename.split(".png")[0]
                    id_to_image[id] = Image.open(raw_image)
            else:
                raise ValueError(f"Invalid split {self.split}.")
            raw_to_processed_path = {
                raw_path: self.raw_to_processed_path(raw_path) for raw_path in raw_paths
            }

            # Ensure the directories exist. They should all follow the same stucture
            # (for each city), so just running this with the first one should work.
            os.makedirs(
                os.path.dirname(list(raw_to_processed_path.values())[0]), exist_ok=True
            )

            for raw_path, processed_path in raw_to_processed_path.items():
                if self.split == "eval_succ_lgp":
                    basename = os.path.basename(raw_path)
                    id = basename.split("-graph.gpickle")[0]
                else:
                    basename = os.path.basename(raw_path)
                    id = basename.split(".gpickle")[0]
                nx_graph = id_to_nx[id]
                # Fix what appears to be a typo in the filename
                if id == "detroit_136_10700_35709":
                    id = "detroit_136_10700_30709"
                image = id_to_image[id]

                if self.split == "eval_full_lgp":
                    *_, bottom_left, bottom_right = id.split("_")
                    x_origin = int(bottom_left)
                    y_origin = int(bottom_right)

                    # Rescale the node positions to overlay the image
                    for n, d in nx_graph.nodes(data=True):
                        pos = d["pos"]
                        # Adjust the coordinates
                        pos = np.array(pos)
                        pos[0] -= x_origin
                        pos[1] -= y_origin

                        # Update the node position
                        nx_graph.nodes[n]["pos"] = pos

                rgb_array = np.array(image)
                rgb_array = np.ascontiguousarray(
                    cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
                )

                data = {
                    "image": torch.from_numpy(rgb_array),
                    "graph": None,
                    "metadata": {
                        "x_origin": None,
                        "y_origin": None,
                        "raw_path": None,
                        "city": city,
                    },
                    "source_graph": nx_graph,
                }

                # Save the processed file
                torch.save(data, processed_path)

    def process(self):
        """Process all the cropped tiles into fitted Bezier graphs, display metrics."""
        if self.split in ("train", "eval"):
            city_to_metrics_list = {}
            for city in CITIES:
                print(f"Processing {city}.")
                city_to_metrics_list[city] = self._process_and_save_city(city)
            path_to_write_to = os.path.join(self.plot_path, "metrics")
            # Create the path_to_write_to
            os.makedirs(path_to_write_to, exist_ok=True)
            BezierOptimisationMetrics.display_aggregated_metrics(
                city_to_metrics_list,
                path_to_write_to=path_to_write_to,
            )

            # Delete the cached property as it is no longer guaranteed to be correct,
            # and needs to be re-computed. This is because some raw paths may have been
            # deleted (where their Bezier graphs were malformed).
            del self.processed_paths
        else:
            self._convert_eval_data_to_processed()

    def len(self):
        """Length of the dataset."""
        return len(self.processed_paths)

    def get(self, idx):
        """Get the data at the given index."""
        processed_path = self.processed_paths[idx]
        data = torch.load(processed_path)
        return data

    def get_max_num_nodes_and_index(self):
        """Get the maximum number of nodes in any graph, and the index of that graph."""
        max_num_nodes = 0
        max_index = None
        for i, data in enumerate(self):
            graph = data["graph"]
            num_nodes = graph.num_nodes
            if num_nodes > max_num_nodes:
                max_num_nodes = num_nodes
                max_index = i
        return max_num_nodes, max_index

    def plot(self, i, path: str):
        """Plot the data at the given index, saving to the given path."""
        dataset_data = self[i]
        image_array = np.array(dataset_data["image"])
        height, _, _ = image_array.shape

        graph = dataset_data["graph"]
        bezier_graph = BezierGraph.from_pyg_graph(graph)
        bezier_graph = bezier_graph.rescale(float(height))

        plt.imshow(image_array)
        ax = plt.gca()
        bezier_graph.plot(ax)
        plt.savefig(path)
        plt.close()

    def to_huggingface(
        self,
        huggingface_folder: Optional[str] = None,
        rendered_images_filepath: Optional[str] = None,
        save_rendered_image_probability: float = 0.1,
    ):
        """
        Convert the dataset to the HuggingFace format.

        To be loaded using the huggingface imagefolder load_dataset functionality, see
        https://huggingface.co/docs/datasets/image_load#imagefolder

        Parameters
        ----------
        huggingface_folder : Optional[str], optional
            The folder to save the HuggingFace dataset to. Defaults to None, which
            saves to self.root/huggingface.
        rendered_images_filepath : Optional[str], optional
            If not None, the folder to save the rendered images to. Defaults to None.
        save_rendered_image_probability : float, optional
            The probability of saving a rendered image for a given example, used to
            render just a subsample of the overall dataset. Defaults to 0.1. Note
            that this is only used if rendered_images_filepath is not None.
        """
        if huggingface_folder is None:
            huggingface_folder = os.path.join(self.root, "huggingface")
        if rendered_images_filepath is not None:
            # Append the data split
            rendered_images_filepath = os.path.join(
                rendered_images_filepath, self.split
            )

        # Check the folders exists
        os.makedirs(huggingface_folder, exist_ok=True)
        os.makedirs(rendered_images_filepath, exist_ok=True)
        # Make split folder
        split_folder = os.path.join(huggingface_folder, self.split)
        os.makedirs(split_folder, exist_ok=True)

        max_num_images = len(self)

        metadata_list = []

        saved_file_index = 0
        for i, dataset_data in tqdm(enumerate(self), total=len(self)):
            if (
                dataset_data["graph"] is not None
                and dataset_data["graph"].num_nodes > 99
            ):
                print(f"Graph {i} has more than 99 nodes. Skipping.")
                continue
            image_array = np.array(dataset_data["image"])
            image = Image.fromarray(image_array)

            # Zero pad filename
            filename = str(saved_file_index).zfill(len(str(max_num_images))) + ".png"
            image.save(os.path.join(split_folder, filename))

            height, width, _ = image_array.shape

            metadata = {
                "image_id": saved_file_index,
                "file_name": filename,
                "objects": [],
                "width": width,
                "height": height,
            }

            objects = {
                "node_attributes": [],
                "categories": [],
                "edge_indices": [],
                "edge_attributes": [],
            }
            # This objects field will be populated with the Bezier graph for train
            # data...
            if self.split in ("train", "eval"):
                if dataset_data["graph"] is not None:
                    if len(dataset_data["graph"].x) > 0:
                        assert (
                            dataset_data["graph"].x[:, :2] >= 0.0
                        ).all(), "Negative positions found."
                        assert (
                            dataset_data["graph"].x[:, :2] <= height
                        ).all(), "Positions greater than image height found."
                    for x_attribute in dataset_data["graph"].x:
                        x_center = x_attribute[:2]
                        x_direction = x_attribute[2:]
                        x, y = x_center
                        u, v = x_direction
                        objects["node_attributes"].append(
                            [float(x), float(y), float(u), float(v)]
                        )
                        objects["categories"].append(0)
                    for (u, v), edge_lengths in zip(
                        dataset_data["graph"].edge_index.T,
                        dataset_data["graph"].edge_attr,
                    ):
                        objects["edge_indices"].append([int(u), int(v)])
                        edge_lengths = np.array(edge_lengths)
                        # Check these lie in [0, 1]
                        assert np.all(edge_lengths >= 0)
                        if not np.all(edge_lengths <= height):
                            print(
                                "Warning, found edge length greater than image dimension "
                                + f"{height}: {edge_lengths}, clipping."
                            )
                        edge_lengths = np.clip(edge_lengths, 0.0, height)
                        l1, l2 = edge_lengths
                        objects["edge_attributes"].append([float(l1), float(l2)])
            else:  # And with the ground truth (raw) graph for eval data
                if dataset_data["source_graph"] is not None:
                    g = dataset_data["source_graph"]
                    n_to_index = {}
                    for i, (n, data) in enumerate(g.nodes(data=True)):
                        n_to_index[n] = i
                        x, y = data["pos"]
                        objects["node_attributes"].append(
                            [float(x), float(y), 0.0, 0.0]
                        )
                        objects["categories"].append(0)
                    for u, v in g.edges():
                        objects["edge_indices"].append([n_to_index[u], n_to_index[v]])
                        objects["edge_attributes"].append([0.0, 0.0])

            metadata["objects"].append(objects)

            metadata_list.append(metadata)
            saved_file_index += 1

            if rendered_images_filepath is not None:
                if np.random.rand() < save_rendered_image_probability:
                    plt.figure(figsize=(20, 20))
                    plt.imshow(image_array)
                    ax = plt.gca()
                    if dataset_data["graph"] is not None:
                        if len(dataset_data["graph"].x) > 0:
                            bezier_graph = BezierGraph.from_pyg_graph(
                                dataset_data["graph"]
                            )
                            bezier_graph.plot(ax)
                    if len(dataset_data["source_graph"].nodes()) > 0:
                        g2b = Graph2Bezier(dataset_data["source_graph"])
                        g2b.plot_ground_truth(ax)
                    plt.savefig(
                        os.path.join(rendered_images_filepath, filename),
                        bbox_inches="tight",
                    )
                    plt.close()

        # Save the metadata as metadata.jsonl
        with open(os.path.join(split_folder, "metadata.jsonl"), "w") as f:
            for metadata in metadata_list:
                json.dump(metadata, f)
                f.write("\n")


class SuccessorAerialBezierGraphDataset(AerialBezierGraphDataset):
    def __init__(self, *args, processed_dataset_root: Optional[str] = None, **kwargs):
        if processed_dataset_root is None:
            processed_dataset_root = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "successor_processed_files"
            )
        super().__init__(*args, processed_dataset_root=processed_dataset_root, **kwargs)

    def preprocess_data(self):
        self._copy_data()

    def _copy_data(self):
        for city in CITIES:
            print(f"Preprocessing {city}.")

            combined_processed_directory = os.path.join(self.crop_path, city)
            os.makedirs(combined_processed_directory, exist_ok=True)

            filenames = glob.glob(
                os.path.join(self.raw_dataset_root, city, self.split, "*")
            )

            for filename in filenames:
                basename = os.path.basename(filename)
                # Copy only the graph.gpickle and rgb.png files
                if not basename.endswith("-graph.gpickle") and not basename.endswith(
                    "-rgb.png"
                ):
                    continue
                processed_filename = os.path.join(
                    combined_processed_directory, basename
                )
                # Copy the file
                shutil.copy(filename, processed_filename)

    @property
    def crop_path(self) -> str:
        """The successor dataset has no cropping."""
        crop_path = self.split_dir
        os.makedirs(crop_path, exist_ok=True)
        return crop_path

    def process(self):
        """Process all the cropped tiles into fitted Bezier graphs, display metrics."""
        city_to_metrics_list = {}
        for city in CITIES:
            print(f"Processing {city}.")
            city_to_metrics_list[city] = self._process_and_save_city(city)
        path_to_write_to = os.path.join(self.plot_path, "metrics")
        # Create the path_to_write_to
        os.makedirs(path_to_write_to, exist_ok=True)
        BezierOptimisationMetrics.display_aggregated_metrics(
            city_to_metrics_list,
            path_to_write_to=path_to_write_to,
        )

        # Delete the cached property as it is no longer guaranteed to be correct,
        # and needs to be re-computed. This is because some raw paths may have been
        # deleted (where their Bezier graphs were malformed).
        del self.processed_paths

    def _filter_crossed_wires(self, tile_graph: nx.DiGraph) -> Tuple[nx.DiGraph, bool]:
        # Lots of the data examples seem to have lanes mistakenly included in the
        # successor graphs, and these lanes come off at strange unphysical angles.
        # We will try to filter out these lanes here.

        # First, find the "starting" node. This is the node that is closest to the
        # bottom middle of the image
        bottom_middle = np.array([0.5, 1.0]) * 256
        tile_graph_indices = np.array(list(tile_graph.nodes()))
        tile_graph_positions = np.array(
            [tile_graph.nodes[n]["pos"] for n in tile_graph_indices]
        )

        distances = np.linalg.norm(tile_graph_positions - bottom_middle, axis=1)
        start_node_index = np.argmin(distances)
        start_node_index = tile_graph_indices[start_node_index]

        # Now, do a breadth first search to find all nodes that are reachable from
        # the start node without changing direction such that their normalised dot
        # product is less than 0.05 s
        # We will also keep track of the previous node, so that we can compute the
        # angle between the vectors
        node_queue = deque([(start_node_index, None)])
        visited_nodes = set()
        incorrect_turn = False
        while len(node_queue) > 0:
            node, previous_node = node_queue.popleft()
            if node in visited_nodes:
                continue
            visited_nodes.add(node)
            if previous_node is None:
                previous_vector = np.array([0.0, -1.0])
            else:
                previous_vector = np.array(tile_graph.nodes[node]["pos"]) - np.array(
                    tile_graph.nodes[previous_node]["pos"]
                )
                previous_vector /= np.linalg.norm(previous_vector)
            for neighbour in tile_graph.successors(node):
                neighbour_vector = np.array(
                    tile_graph.nodes[neighbour]["pos"]
                ) - np.array(tile_graph.nodes[node]["pos"])
                neighbour_vector /= np.linalg.norm(neighbour_vector)
                if neighbour_vector @ previous_vector > 0.05:
                    node_queue.append((neighbour, node))
                else:
                    incorrect_turn = True

        return nx.DiGraph(tile_graph.subgraph(visited_nodes)), incorrect_turn

    def _get_data_and_metrics_for_city(
        self,
        city: str,
    ):
        combined_raw_directory = os.path.join(self.crop_path, city)
        # Get all of the raw tile paths for this city
        raw_paths = glob.glob(os.path.join(combined_raw_directory, "*.gpickle"))
        raw_to_processed_path = {
            raw_path: self.raw_to_processed_path(raw_path) for raw_path in raw_paths
        }

        num_processed_paths = len(raw_to_processed_path.values())

        # Running lists for batching
        rgb_crops = []
        tile_graphs = []
        batch_raw_paths = []

        # Overall list that will be returned at the end
        data_list = []

        # List of metric objects for each graph
        metric_list = []

        broken_number = 0
        total_number = 0

        for raw_path_index, raw_path in tqdm(
            enumerate(raw_paths),
            desc=f"Processing tiles for {city}",
            total=len(raw_paths),
        ):
            # Load the tile graph. Note, unlike before, this is already translated
            with open(raw_path, "rb") as f:
                original_tile_graph = pickle.load(f)

            tile_graph, was_truncated = self._filter_crossed_wires(original_tile_graph)
            if was_truncated:
                broken_number += 1
            total_number += 1

            image = Image.open(raw_path.replace("graph.gpickle", "rgb.png"))
            rgb_crop = np.array(image)
            rgb_crop = np.ascontiguousarray(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))

            # Append to running lists. These will be computed in batches (see next
            # if statement)
            rgb_crops.append(rgb_crop)
            tile_graphs.append(tile_graph)
            batch_raw_paths.append(raw_path)

            if (len(tile_graphs) > 0) and (
                len(tile_graphs) > self.bezier_optimisation_batch_size
                or (raw_path_index == (num_processed_paths - 1))
            ):
                assert len(rgb_crops) == len(tile_graphs)

                (
                    g2b,
                    batch_index_to_bezier_indices,
                    batch_index_to_source_indices,
                    removed_batch_indices,
                ) = Graph2Bezier.batch_lane_graphs(tile_graphs, **self.g2bkwargs)

                batched_bezier_graphs = g2b.optimise_bezier_graph(
                    max_num_iterations=self.max_num_optimisation_iterations
                )

                # In order to avoid re-calculating everything, we need to remove the
                # raw files for the batches which were removed.
                for batch_index in removed_batch_indices:
                    print("Removing malformed raw file.")
                    raw_file_to_remove = batch_raw_paths[batch_index]
                    os.remove(raw_file_to_remove)

                # Compute metrics
                metric_list += BezierOptimisationMetrics.metrics_from_batch(
                    batched_bezier_graphs,
                    g2b,
                    batch_index_to_bezier_indices,
                    batch_index_to_source_indices,
                )

                for (
                    batch_index,
                    bezier_indices,
                ) in batch_index_to_bezier_indices.items():
                    bezier_graph = BezierGraph(
                        nx.subgraph(batched_bezier_graphs, bezier_indices)
                    )
                    rgb_crop = rgb_crops[batch_index]
                    raw_path = batch_raw_paths[batch_index]

                    pyg_bezier_graph = bezier_graph.to_pyg_graph()

                    data = {
                        "image": (
                            torch.from_numpy(rgb_crop) if rgb_crop is not None else None
                        ),
                        "graph": pyg_bezier_graph,
                        "metadata": {
                            "x_origin": 0,
                            "y_origin": 0,
                            "raw_path": raw_path,
                            "city": city,
                        },
                        "source_graph": tile_graphs[batch_index],
                    }
                    data_list.append(data)

                rgb_crops = []
                tile_graphs = []
                batch_raw_paths = []

        print(f"Broken number: {broken_number}, total number: {total_number}")
        print(f"Broken fraction = {broken_number / total_number}")

        return data_list, metric_list
