from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import batched_negative_sampling
from transformers import DeformableDetrConfig, DeformableDetrModel
from transformers.models.deformable_detr import DeformableDetrPreTrainedModel
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrLoss,
    DeformableDetrMLPPredictionHead,
    DeformableDetrObjectDetectionOutput,
)
from transformers.utils import ModelOutput


# Based on https://github.com/facebookresearch/detr/blob/master/models/matcher.py
# And adapted from HuggingFace's DETR Hungarian Matcher implementation
class BezierHungarianMatcher(nn.Module):
    """
    This class computes an assignment between targets and predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this,
    in general, there are more predictions than targets. In this case, we do a 1-to-1
    matching of the best predictions, while the others are un-matched (and thus treated
    as non-objects).

    Parameters
    ----------
    class_cost:
        The relative weight of the classification error in the matching cost.
    position_cost:
        The relative weight of the L1 error of the node positions in the matching cost.
        Defaults to 1.
    direction_cost:
        The relative weight of the L1 error of the node directions in the matching cost.
        Defaults to 1.
    """

    def __init__(
        self, class_cost: float = 1, position_cost: float = 1, direction_cost: float = 1
    ):
        super().__init__()

        self.class_cost = class_cost
        self.position_cost = position_cost
        self.direction_cost = direction_cost
        if class_cost == 0 and position_cost == 0 and direction_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching.

        Parameters
        ----------
        outputs:
            This is a dict that contains at least these entries:
            * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                classification logits
            * "pred_node_attributes": Tensor of dim [batch_size, num_queries, 4] with
                the node attributes. In each vector, the first two elements are the
                predicted position, the last two elements are the predicted direction.
        targets:
            A list of targets (len(targets) = batch_size), where each target is a dict
            containing:
            * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes
                is the number of ground-truth objects in the target) containing the
                class labels.
            * "node_attributes": Tensor of dim [num_target_boxes, 4] containing the
                target node attributes. In each vector, the first two elements are the
                target position, the last two elements are the target direction.

        Returns
        -------
        List[Tuple]
            List of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_node_properties = outputs["pred_node_attributes"].flatten(
            0, 1
        )  # [batch_size * num_queries, 4]
        out_node_positions = out_node_properties[:, :2]
        out_node_directions = out_node_properties[:, 2:]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_attributes = torch.cat([v["node_attributes"] for v in targets])
        target_positions = target_attributes[:, :2]
        target_directions = target_attributes[:, 2:]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between positions
        position_cost = torch.cdist(out_node_positions, target_positions, p=1)
        direction_cost = torch.cdist(out_node_directions, target_directions, p=1)

        # Final cost matrix
        cost_matrix = (
            self.position_cost * position_cost
            + self.class_cost * class_cost
            + self.direction_cost * direction_cost
        )

        # Compute the direction cost to have the same dimensions as the position cost
        # which uses cmatrix, but computes the cosine similarity

        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["node_attributes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(cost_matrix.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


class BezierDetrConfig(DeformableDetrConfig):
    def __init__(
        self,
        position_cost: int = 5,
        direction_cost: int = 2,
        position_loss_coefficient: int = 5,
        direction_loss_coefficient: int = 2,
        edge_existence_loss_coefficient: int = 1,
        edge_attribute_loss_coefficient: int = 1,
        mlp_head_hidden_dimension: int = 256,
        num_node_queries: Optional[int] = 99,
        num_negative_edges_per_positive_edge: int = 3,
        **kwargs,
    ):
        if "num_queries" in kwargs and num_node_queries is None:
            num_node_queries = kwargs["num_queries"] - 1  # -1 for the edge query
        kwargs["num_queries"] = num_node_queries + 1  # +1 for the edge query
        super().__init__(**kwargs)

        self.position_cost = position_cost
        self.direction_cost = direction_cost
        self.position_loss_coefficient = position_loss_coefficient
        self.direction_loss_coefficient = direction_loss_coefficient
        self.edge_existence_loss_coefficient = edge_existence_loss_coefficient
        self.edge_attribute_loss_coefficient = edge_attribute_loss_coefficient
        self.mlp_head_hidden_dimension = mlp_head_hidden_dimension
        self.num_negative_edges_per_positive_edge = num_negative_edges_per_positive_edge


class BezierDetrLoss(DeformableDetrLoss):
    def __init__(
        self,
        num_classes: int,
        eos_coef: float,
        losses: List[str],
    ):
        # Not the best code, but basically we want to remove matcher functionality
        # from the loss in order that we can compute it first for edge prediction.
        # This should throw a slightly more understandable exception than a NoneType
        # exception if the matcher is called.
        matcher = lambda *_: (_ for _ in ()).throw(  # noqa E731
            NotImplementedError(
                "Matcher functionality has been moved out of the loss function."
            )
        )
        super().__init__(matcher, num_classes, eos_coef, losses)

    def loss_positions(self, outputs, targets, indices, num_nodes):
        if "pred_node_attributes" not in outputs:
            raise KeyError("No predicted nodes found in outputs")
        idx = self._get_source_permutation_idx(indices)

        source_node_attributes = outputs["pred_node_attributes"][idx]
        source_node_positions = source_node_attributes[:, :2]

        target_node_attributes = torch.cat(
            [t["node_attributes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        target_node_positions = target_node_attributes[:, :2]

        loss_position = nn.functional.l1_loss(
            source_node_positions, target_node_positions, reduction="none"
        )

        losses = {}
        losses["loss_position"] = loss_position.sum() / num_nodes

        return losses

    def loss_directions(self, outputs, targets, indices, num_nodes):
        if "pred_node_attributes" not in outputs:
            raise KeyError("No predicted nodes found in outputs")
        idx = self._get_source_permutation_idx(indices)

        source_node_attributes = outputs["pred_node_attributes"][idx]
        source_node_directions = source_node_attributes[:, 2:]

        target_node_attributes = torch.cat(
            [t["node_attributes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )
        target_node_directions = target_node_attributes[:, 2:]

        loss_direction = nn.functional.l1_loss(
            source_node_directions, target_node_directions, reduction="none"
        )

        losses = {}
        losses["loss_direction"] = loss_direction.sum() / num_nodes

        return losses

    def loss_boxes(self, *args, **kwargs):
        raise NotImplementedError

    def loss_masks(self, *args, **kwargs):
        raise NotImplementedError

    def get_loss(self, loss, outputs, targets, indices, num_nodes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "positions": self.loss_positions,
            "directions": self.loss_directions,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_nodes)

    def get_edge_loss(
        self,
        edge_outputs_dict,
        edge_existence_labels,
        ground_truth_edge_attributes,
    ):
        edge_existence_loss = nn.functional.binary_cross_entropy_with_logits(
            edge_outputs_dict["logits"],
            edge_existence_labels,
            reduction="mean",
        )
        # BCE with logits again as we use sigmoid activation
        edge_attribute_loss = nn.functional.mse_loss(
            edge_outputs_dict["l_logits"].sigmoid(),
            ground_truth_edge_attributes,
            reduction="mean",
        )
        loss_dict = {
            "edge_existence": edge_existence_loss,
            "edge_attributes": edge_attribute_loss,
        }
        return loss_dict

    def forward(self, outputs, targets, indices):
        """
        Perform the loss computation.

        Parameters
        ----------
        outputs: Dict
            Dict of tensors, see the output specification of the model for the format.
        targets: List[Dict]
            List of dicts, such that len(targets) == batch_size. The expected keys in
            each dict depends on the losses applied, see each loss' doc.
        """
        # Compute the average number of target boxes across all nodes, for normalization
        # purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # Comments below are from original Huggingface implementation. Left in in case
        # distributed training is added later.
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each
        # intermediate layer.
        if "auxiliary_outputs" in outputs:
            raise NotImplementedError("Auxiliary outputs not yet implemented")

        return losses


class BezierDetrObjectDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    pred_node_attributes: Optional[torch.FloatTensor] = None
    pred_edge_attributes: Optional[List[torch.FloatTensor]] = None
    pred_edge_indices: Optional = None


class BezierDETR(DeformableDetrPreTrainedModel):
    config_class = BezierDetrConfig

    def __init__(self, config: BezierDetrConfig):
        super().__init__(config)

        # DETR encoder-decoder model
        self.model = DeformableDetrModel(config)

        # Object detection heads
        # Note we expect num_labels to be 1 here initially
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels + 1
        )  # We add one for the "no object" class

        # Although the docstring states this is to be used to generate bounding boxes,
        # there is nothing preventing us from using it to predict other "node"
        # attributes - we will use it to generate position and direction, for which we
        # don't even need to change the output dim
        self.node_attribute_head = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model,
            hidden_dim=config.mlp_head_hidden_dimension,
            output_dim=4,
            num_layers=3,
        )

        self.edge_attribute_head = DeformableDetrMLPPredictionHead(
            input_dim=config.d_model * 3
            + 8,  # In node embedding, Out node embedding,  edge embedding
            hidden_dim=config.mlp_head_hidden_dimension,
            output_dim=3,
            num_layers=2,
        )
        # First: create the matcher
        self.matcher = BezierHungarianMatcher(
            class_cost=self.config.class_cost,
            position_cost=self.config.position_cost,
            direction_cost=self.config.direction_cost,
        )

        # Second: create the criterion
        losses = [
            "labels",
            "positions",
            "directions",
            "cardinality",
        ]
        self.criterion = BezierDetrLoss(
            num_classes=self.config.num_labels,
            eos_coef=self.config.eos_coefficient,
            losses=losses,
        )
        self.criterion.to(self.device)
        # Initialize weights and apply final processing
        self.post_init()

    def update_configs(
        self,
        mlp_head_hidden_dimension: int,
        position_loss_coefficient: int,
        direction_loss_coefficient: int,
        edge_existence_loss_coefficient: int,
        edge_attribute_loss_coefficient: int,
        num_negative_edges_per_positive_edge: int,
    ):
        self.config.mlp_head_hidden_dimension = mlp_head_hidden_dimension
        self.config.position_loss_coefficient = position_loss_coefficient
        self.config.direction_loss_coefficient = direction_loss_coefficient
        self.config.edge_existence_loss_coefficient = edge_existence_loss_coefficient
        self.config.edge_attribute_loss_coefficient = edge_attribute_loss_coefficient
        self.config.num_negative_edges_per_positive_edge = (
            num_negative_edges_per_positive_edge
        )

        self.node_attribute_head = DeformableDetrMLPPredictionHead(
            input_dim=self.config.d_model,
            hidden_dim=self.config.mlp_head_hidden_dimension,
            output_dim=4,
            num_layers=3,
        )
        self.edge_attribute_head = DeformableDetrMLPPredictionHead(
            input_dim=self.config.d_model * 3
            + 8,  # In node embedding, Out node embedding,  edge embedding
            hidden_dim=self.config.mlp_head_hidden_dimension,
            output_dim=3,
            num_layers=2,
        )

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"logits": a, "pred_node_attributes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inference_edge_query_node_threshold: float = 0.5,
    ) -> Union[Tuple[torch.FloatTensor], DeformableDetrObjectDetectionOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # First, sent images through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_mask=pixel_mask,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        node_embeddings = outputs[1][:, :-1, :]
        edge_embedding = outputs[1][:, -1:, :]

        batch_size, num_node_queries, _ = node_embeddings.shape
        # class logits + predicted bounding boxes
        logits = self.class_labels_classifier(node_embeddings)
        pred_node_logit_attributes = self.node_attribute_head(node_embeddings)
        pred_node_positions = pred_node_logit_attributes[:, :, :2].sigmoid()
        pred_node_direction_vectors = pred_node_logit_attributes[:, :, 2:]
        pred_node_attributes = torch.cat(
            (pred_node_positions, pred_node_direction_vectors), dim=-1
        )

        loss, loss_dict, auxiliary_outputs = None, None, None
        pred_edge_attributes, pred_edge_indices = None, None

        # The node index pairs we will query depends on if we have access to ground
        # truth matching or not, i.e. if we are training or running inference
        if labels is None:
            # If the label is None, we are running inference. Therefore, we want to
            # make a "best guess" at which nodes have been predicted, and test every
            # pairing of these. We use nodes which pass the threshold set by
            # inference_edge_query_node_threshold.
            predicted_node_probabilities = logits.softmax(dim=-1)
            # Note last element in the logit tensor is the no-object class
            # We therefore filter to just the nodes which are not the no-object class
            node_masks = (
                predicted_node_probabilities[..., -1]
                < 1 - inference_edge_query_node_threshold
            )
            # Per batch item
            predicted_edge_attribute_list = []
            predicted_edge_index_list = []
            for batch_edge_embedding, node_embedding, node_attribute, node_mask in zip(
                edge_embedding, node_embeddings, pred_node_attributes, node_masks
            ):
                # Construct every pairing of indices which pass the threshold
                # Note we do this by taking the cartesian product of the indices of the
                # nodes which pass the threshold
                passed_node_indices = torch.argwhere(node_mask).squeeze(1)
                node_index_combinations = torch.combinations(passed_node_indices, r=2)
                # To get the possible permutations, we also need to consider the reverse
                # of these pairs
                node_index_permutations = torch.cat(
                    (node_index_combinations, node_index_combinations.flip(dims=(1,)))
                ).T

                out_node_embeddings = node_embedding[node_index_permutations[0]]
                in_node_embeddings = node_embedding[node_index_permutations[1]]
                out_node_attributes = node_attribute[node_index_permutations[0]]
                in_node_attributes = node_attribute[node_index_permutations[1]]
                repeated_edge_embedding = batch_edge_embedding.repeat(
                    (out_node_embeddings.shape[0], 1)
                )
                concatenated_edge_embeddings = torch.cat(
                    [
                        out_node_embeddings,
                        repeated_edge_embedding,
                        in_node_embeddings,
                        out_node_attributes,
                        in_node_attributes,
                    ],
                    dim=1,
                )

                pred_edge_attributes = self.edge_attribute_head(
                    concatenated_edge_embeddings
                )
                pred_edge_probabilities = pred_edge_attributes[:, :1].sigmoid()
                pred_edge_l = pred_edge_attributes[:, 1:].sigmoid()
                pred_edge_attributes = torch.cat(
                    (pred_edge_probabilities, pred_edge_l), dim=-1
                )
                predicted_edge_attribute_list.append(pred_edge_attributes)
                predicted_edge_index_list.append(node_index_permutations.T)

            pred_edge_attributes = predicted_edge_attribute_list
            pred_edge_indices = predicted_edge_index_list
        else:
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_node_attributes"] = pred_node_attributes
            if self.config.auxiliary_loss:
                raise NotImplementedError

            outputs_without_aux = {
                k: v for k, v in outputs_loss.items() if k != "auxiliary_outputs"
            }

            # Retrieve the matching between the outputs of the last layer and the
            # targets
            matched_indices = self.matcher(outputs_without_aux, labels)

            # Assemble labels into torch_geometric Data objects and from there into
            # a Batch object
            # We want to get the node embeddings for each matched index
            source_idx = self.criterion._get_source_permutation_idx(matched_indices)
            target_idx = self.criterion._get_target_permutation_idx(matched_indices)
            max_nodes_in_batch = target_idx[0].bincount().max().item()
            num_batches, _, embedding_dimension = node_embeddings.shape
            padded_batched_node_attributes = torch.zeros(
                num_batches, max_nodes_in_batch, 4
            ).to(self.device)
            padded_batched_node_attributes[target_idx] = pred_node_attributes[
                source_idx
            ]

            padded_batched_node_embeddings = torch.zeros(
                num_batches, max_nodes_in_batch, embedding_dimension
            ).to(self.device)
            padded_batched_node_embeddings[target_idx] = node_embeddings[source_idx]
            graph_data_objects = []
            for (
                batch_edge_embedding,
                corresponding_node_embeddings,
                corresponding_node_attributes,
                l,
            ) in zip(
                edge_embedding,
                padded_batched_node_embeddings,
                padded_batched_node_attributes,
                labels,
            ):
                if len(l.edge_indices) == 0:
                    graph_data_objects.append(
                        Data(
                            x=torch.empty(
                                [0, corresponding_node_embeddings.shape[1] + 4],
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            edge_index=torch.empty(
                                [2, 0], dtype=torch.long, device=self.device
                            ),
                            edge_attr=torch.empty(
                                [0, 2], dtype=torch.long, device=self.device
                            ),
                        )
                    )
                    continue
                corresponding_len = len(
                    corresponding_node_embeddings[: len(l.node_attributes)]
                )
                node_attr_len = len(l.node_attributes)
                assert corresponding_len == node_attr_len, (
                    "Node attributes and corresponding node attributes should have the "
                    + f"same length: {corresponding_len}, {node_attr_len}"
                )
                node_x = torch.cat(
                    [
                        corresponding_node_embeddings[: len(l.node_attributes)],
                        corresponding_node_attributes[: len(l.node_attributes)],
                    ],
                    dim=1,
                )

                data = Data(
                    x=node_x,
                    edge_index=l.edge_indices.T,
                    edge_attr=l.edge_attributes,
                )
                graph_data_objects.append(data)

            graph_data_batch = Batch.from_data_list(graph_data_objects)
            # Assemble in, out pairs for the edge prediction task
            positive_sample_edge_indices = graph_data_batch.edge_index
            avg_positive_examples_per_batch = (
                positive_sample_edge_indices.shape[1] / batch_size
            )
            # To get the number of negative samples
            negative_sample_edge_indices = batched_negative_sampling(
                graph_data_batch.edge_index,
                batch=graph_data_batch.batch,
                num_neg_samples=np.ceil(
                    avg_positive_examples_per_batch
                    * self.config.num_negative_edges_per_positive_edge
                ).astype(int),
            )
            stacked_edges = torch.cat(
                (positive_sample_edge_indices, negative_sample_edge_indices), dim=-1
            )
            # Edge labels are 1 for positive samples, 0 for negative samples
            edge_existence_labels = torch.cat(
                (
                    torch.ones(positive_sample_edge_indices.shape[1]),
                    torch.zeros(negative_sample_edge_indices.shape[1]),
                )
            ).to(self.device)
            # We can just use the out node, since the edges should always be within
            # a batch.
            edge_batches = graph_data_batch.batch[stacked_edges[0]]
            out_node_embeddings = graph_data_batch.x[stacked_edges[0]][:, :-4]
            out_node_attributes = graph_data_batch.x[stacked_edges[0]][:, -4:]
            in_node_embeddings = graph_data_batch.x[stacked_edges[1]][:, :-4]
            in_node_attributes = graph_data_batch.x[stacked_edges[1]][:, -4:]
            # Repeat the edge embedding for each edge
            repeated_edge_embedding = edge_embedding[edge_batches].squeeze(1)
            concatenated_edge_embeddings = torch.cat(
                [
                    out_node_embeddings,
                    repeated_edge_embedding,
                    in_node_embeddings,
                    out_node_attributes,
                    in_node_attributes,
                ],
                dim=1,
            )
            # Then predict edge existence, send to loss
            pred_edge_attributes = self.edge_attribute_head(
                concatenated_edge_embeddings
            )
            pred_edge_probability_logits = pred_edge_attributes[:, 0]
            # Truncate at positive_sample_edge_indices.shape[1] since we don't have
            # ground truth to compute loss for the negative samples
            pred_edge_l_logits = pred_edge_attributes[:, 1:][
                : positive_sample_edge_indices.shape[1]
            ]
            # Third: compute the losses, based on outputs and labels
            node_loss_dict = self.criterion(outputs_loss, labels, matched_indices)
            # We compute edge losses differently as they are not batched in the same
            # way.
            ground_truth_edge_attributes = graph_data_batch.edge_attr
            edge_outputs_dict = {}
            edge_outputs_dict["logits"] = pred_edge_probability_logits
            edge_outputs_dict["l_logits"] = pred_edge_l_logits
            edge_loss_dict = self.criterion.get_edge_loss(
                edge_outputs_dict, edge_existence_labels, ground_truth_edge_attributes
            )

            # Take union of both loss dicts
            loss_dict = {**node_loss_dict, **edge_loss_dict}

            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {
                "loss_ce": 1,
                "loss_position": self.config.position_loss_coefficient,
                "loss_direction": self.config.direction_loss_coefficient,
                "edge_existence": self.config.edge_existence_loss_coefficient,
                "edge_attributes": self.config.edge_attribute_loss_coefficient,
            }
            loss = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            )

        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_node_attributes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_node_attributes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return BezierDetrObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_node_attributes=pred_node_attributes,
            pred_edge_indices=pred_edge_indices,
            pred_edge_attributes=pred_edge_attributes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
