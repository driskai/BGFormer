import pathlib
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
)
from transformers.models.detr.image_processing_detr import (
    AnnotionFormat,
    DetrImageProcessor,
)

from aerial_lane_bezier.dataset.bezier_graph import BezierGraph


class ImageProcessingBezierDETR(DetrImageProcessor):
    def prepare_annotation(
        self,
        image: np.ndarray,
        target: Dict,
        format: Optional[AnnotionFormat] = None,
        return_segmentation_masks: bool = None,
        masks_path: Optional[Union[str, pathlib.Path]] = None,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ) -> Dict:
        """
        Prepare an annotation for feeding into DETR model.
        """
        format = format if format is not None else self.format

        return_segmentation_masks = (
            False if return_segmentation_masks is None else return_segmentation_masks
        )
        target = prepare_detection_annotation(
            image,
            target,
            return_segmentation_masks,
            input_data_format=input_data_format,
        )
        return target

    def resize_annotation(
        self,
        annotation,
        orig_size,
        size,
        resample: PILImageResampling = PILImageResampling.NEAREST,
    ) -> Dict:
        """
        Resize the annotation to match the resized image.

        If size is an int, smaller edge of the mask will be matched to this number.
        """
        target_size = size
        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(target_size, orig_size)
        )
        ratio_height, ratio_width = ratios

        new_annotation = {}
        new_annotation["size"] = target_size

        for key, value in annotation.items():
            if key == "node_attributes":
                node_attributes = value
                node_positions = node_attributes[:, :2]
                node_directions = node_attributes[:, 2:]
                node_positions *= np.asarray(
                    [ratio_width, ratio_height], dtype=np.float32
                )
                new_annotation["node_attributes"] = np.concatenate(
                    [node_positions, node_directions], axis=1
                )
            elif key == "edge_attributes":
                assert np.isclose(ratio_height, ratio_width), (
                    f"Only square images are supported, got ratios {ratios} from "
                    + "sizes {target_size} and {orig_size}."
                )
                edge_attributes = value
                edge_attributes *= ratio_height
                new_annotation["edge_attributes"] = edge_attributes
            else:
                new_annotation[key] = value

        return new_annotation

    def normalize_annotation(
        self, annotation: Dict, image_size: Tuple[int, int]
    ) -> Dict:
        """Normalize the node and edge attributes."""
        image_height, image_width = image_size
        norm_annotation = {}
        for key, value in annotation.items():
            if key == "node_attributes":
                node_attributes = value
                node_positions = node_attributes[:, :2]
                node_directions = node_attributes[:, 2:]
                node_positions /= np.asarray(
                    [image_width, image_height], dtype=np.float32
                )
                norm_annotation[key] = np.concatenate(
                    [node_positions, node_directions], axis=1
                )
            elif key == "edge_attributes":
                assert np.isclose(
                    image_width, image_height
                ), "Only square images are supported"
                edge_attributes = value
                edge_attributes /= image_width
                norm_annotation[key] = edge_attributes
            else:
                norm_annotation[key] = value
        return norm_annotation

    def post_process_object_detection(
        self,
        outputs,
        node_threshold: float = 0.5,
        edge_threshold: float = 0.5,
        target_sizes: Union[torch.Tensor, List[Tuple]] = None,
    ):
        (
            out_node_logits,
            out_node_attributes,
            out_edge_index_list,
            out_edge_attributes_list,
        ) = (
            outputs.logits,
            outputs.pred_node_attributes,
            outputs.pred_edge_indices,
            outputs.pred_edge_attributes,
        )

        node_prob = nn.functional.softmax(out_node_logits, -1)
        node_scores, node_labels = node_prob[..., :-1].max(-1)

        # Convert from relative [0, 1] to absolute [0, height] coordinates
        if target_sizes is not None:
            if isinstance(target_sizes, List):
                img_h = torch.Tensor([i[0] for i in target_sizes])
                img_w = torch.Tensor([i[1] for i in target_sizes])
            else:
                img_h, img_w = target_sizes.unbind(1)

            assert img_w.item() == img_h.item(), "Only square images are supported"

            scale_fct = torch.stack(
                [img_w, img_h, torch.tensor([1.0]), torch.tensor([1.0])], dim=1
            ).to(out_node_attributes.device)
            out_node_attributes = out_node_attributes * scale_fct[:, None, :]

            scaled_out_edge_attributes = []

            for out_edge_attributes in out_edge_attributes_list:
                # Rescale the l values
                out_edge_attributes[:, 1:] *= scale_fct[0][0]
                scaled_out_edge_attributes.append(out_edge_attributes)

        results = []
        for s, l, a, e, e_i in zip(
            node_scores,
            node_labels,
            out_node_attributes,
            scaled_out_edge_attributes,
            out_edge_index_list,
        ):
            score = s[s > node_threshold]
            label = l[s > node_threshold]
            out_node_attribute = a[s > node_threshold]

            edge_probabilities = e[:, 0][e[:, 0] > edge_threshold]
            edge_l_values = e[:, 1:][e[:, 0] > edge_threshold]
            edge_indices = e_i[e[:, 0] > edge_threshold]
            # But these edge indices are relative to the total node indices, so we
            # need to modify them to work with the filtered node indices
            node_index_map = torch.argwhere(s > node_threshold).flatten()
            if node_index_map.numel() == 0:
                edge_index = torch.empty((2, 0), device=node_index_map.device)
            else:
                # Create an empty tensor filled with large numbers (larger than the
                # length of node_index_map)
                inverse_map = torch.full(
                    (node_index_map.max().item() + 1,),
                    len(node_index_map),
                    dtype=torch.long,
                )
                inverse_map[node_index_map] = torch.arange(len(node_index_map))

                converted_edge_indices = inverse_map[edge_indices]
                edge_index = converted_edge_indices.T

            pyg_graph = Data(
                x=out_node_attribute,
                edge_index=edge_index,
                edge_attr=edge_l_values,
                edge_probabilities=edge_probabilities,
            )

            filtered_graph = BezierGraph.from_pyg_graph(pyg_graph)

            if True:
                # Additionally filter out "triangles" or short circuited paths
                short_circuit_edges = []
                for n in filtered_graph.nodes:
                    for successor in filtered_graph.successors(n):
                        for second_order_successor in filtered_graph.successors(
                            successor
                        ):
                            if filtered_graph.has_edge(n, second_order_successor):
                                short_circuit_edges.append((n, second_order_successor))

                filtered_graph.remove_edges_from(short_circuit_edges)

            filtered_pyg_graph = filtered_graph.to_pyg_graph()

            results.append(
                {
                    "scores": score,
                    "labels": label,
                    "out_node_attributes": out_node_attribute,
                    "pyg_graph": filtered_pyg_graph,
                }
            )

        return results

    def post_process_successor_lgp(
        self,
        outputs,
        start_node_position: torch.Tensor = None,
        start_node_prior_direction: torch.Tensor = None,
        node_threshold: float = 0.5,
        edge_threshold: float = 0.5,
        target_sizes: Union[torch.Tensor, List[Tuple]] = None,
    ):
        results = self.post_process_object_detection(
            outputs,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            target_sizes=target_sizes,
        )

        if start_node_prior_direction is None:
            start_node_prior_direction = torch.tensor([0.0, -1.0]).to(
                results[0]["scores"].device
            )
        # Make sure normalised
        start_node_prior_direction /= start_node_prior_direction.norm()

        if start_node_position is None:
            start_node_position = torch.tensor([0.5, 1.0]).to(
                results[0]["scores"].device
            )
            start_node_position = (start_node_position * target_sizes).squeeze(0)

        filtered_results = []
        for result in results:
            if len(result["scores"]) == 0:
                filtered_results.append(result)
                continue
            # Otherwise, there exist nodes.
            # We'll work with just the pyg graph for ease
            bezier_graph = BezierGraph.from_pyg_graph(result["pyg_graph"])
            node_positions = result["pyg_graph"].x[:, :2]
            node_directions = result["pyg_graph"].x[:, 2:]
            # Normalise the node directions
            node_directions /= node_directions.norm(dim=1, keepdim=True)
            # Filter to just the nodes which have directions pointing away from the
            # bottom of the image (with dot product > 0 with start_node_prior_direction)
            correct_direction_node_indices = torch.argwhere(
                node_directions @ start_node_prior_direction > 0.0
            )
            if len(correct_direction_node_indices) == 0:
                print("No nodes found with directions along the start direction.")
                filtered_results.append(result)
                continue
            # Find now the closest node index to the start node position
            node_distances = torch.norm(
                node_positions[correct_direction_node_indices] - start_node_position,
                dim=-1,
            )
            closest_node_index = correct_direction_node_indices[node_distances.argmin()]
            # Now just filter to all the paths reachable from this node
            reachable_nodes = nx.descendants(bezier_graph, closest_node_index.item())
            reachable_nodes.add(closest_node_index.item())
            reachable_nodes = sorted(list(reachable_nodes))
            # Now filter the graph to just these nodes
            filtered_graph = BezierGraph(bezier_graph.subgraph(reachable_nodes))
            if True:
                # Additionally filter out "triangles" or short circuited paths
                short_circuit_edges = []
                for n in filtered_graph.nodes:
                    for successor in filtered_graph.successors(n):
                        for second_order_successor in filtered_graph.successors(
                            successor
                        ):
                            if filtered_graph.has_edge(n, second_order_successor):
                                short_circuit_edges.append((n, second_order_successor))

                filtered_graph.remove_edges_from(short_circuit_edges)

            filtered_pyg_graph = filtered_graph.to_pyg_graph()

            filtered_result = {
                "scores": result["scores"][reachable_nodes],
                "labels": result["labels"][reachable_nodes],
                "out_node_attributes": filtered_pyg_graph.x,
                "pyg_graph": filtered_pyg_graph,
            }
            filtered_results.append(filtered_result)

        return filtered_results


def prepare_detection_annotation(
    image,
    target,
    return_segmentation_masks: bool = False,
    input_data_format: Optional[Union[ChannelDimension, str]] = None,
):
    image_height, image_width = get_image_size(image, channel_dim=input_data_format)

    image_id = target["image_id"]
    image_id = np.asarray([image_id], dtype=np.int64)

    annotations = target["annotations"]

    node_annotations = [obj for obj in annotations if "node_attributes" in obj]
    edge_annotations = [obj for obj in annotations if "edge_indices" in obj]

    classes = [obj["category_id"] for obj in node_annotations]
    classes = np.asarray(classes, dtype=np.int64)

    node_attributes = [obj["node_attributes"] for obj in node_annotations]

    node_attributes = np.asarray(node_attributes, dtype=np.float32).reshape(-1, 4)

    new_target = {}
    new_target["image_id"] = image_id
    new_target["class_labels"] = classes
    new_target["node_attributes"] = node_attributes
    new_target["orig_size"] = np.asarray(
        [int(image_height), int(image_width)], dtype=np.int64
    )

    edge_indices = [obj["edge_indices"] for obj in edge_annotations]
    edge_indices = np.asarray(edge_indices, dtype=np.int64)

    edge_attributes = [obj["edge_attributes"] for obj in edge_annotations]
    edge_attributes = np.asarray(edge_attributes, dtype=np.float32)

    new_target["edge_indices"] = edge_indices
    new_target["edge_attributes"] = edge_attributes

    if annotations and "keypoints" in annotations[0]:
        raise NotImplementedError

    if return_segmentation_masks:
        raise NotImplementedError

    return new_target
