from functools import partial
from multiprocessing import get_context

import rustworkx as rx
import torch
import torch_geometric
from tqdm import tqdm

from ..dataset.bezier_graph import BezierGraph
from .apls import simple_apls
from .geo_topo import geo_topo_metric
from .iou import graph_iou
from .sda import sda
from .utils import interpolate_graph, obj_to_device, rx_to_poly, tg_to_rx


@torch.inference_mode()
def run_batch(
    image_processor,
    model,
    inputs,
    labels,
    device,
    experiment_mode="full",
    node_threshold=0.5,
    edge_threshold=0.5,
):
    inputs = obj_to_device(inputs, device)
    outputs = model(**inputs, inference_edge_query_node_threshold=node_threshold)
    outputs = obj_to_device(outputs, torch.device("cpu"))
    target_sizes = torch.stack([labels[0]["orig_size"]])
    post_process_method = (
        image_processor.post_process_object_detection
        if experiment_mode == "full"
        else image_processor.post_process_successor_lgp
    )
    decoded = post_process_method(
        outputs,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        target_sizes=target_sizes,
    )
    return decoded


def predict_graphs(
    image_processor,
    model,
    batch,
    device,
    experiment_mode="full",
    node_threshold=0.5,
    edge_threshold=0.5,
):
    model.eval()
    labels = batch.pop("labels")
    decoded = run_batch(
        image_processor,
        model,
        batch,
        labels,
        device,
        experiment_mode=experiment_mode,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    predicted_graphs = []
    for lab, dec in zip(labels, decoded):
        pred_tg = dec["pyg_graph"]
        x = lab["node_attributes"]
        if x.numel() > 0:
            x[:, :2] *= lab["orig_size"][None]
        edge = lab["edge_attributes"]
        if edge.numel() > 0:
            edge = edge * lab["orig_size"][None]

        predicted_graphs.append(
            (
                (x.numpy(), lab["edge_indices"].t().numpy(), edge.numpy()),
                (
                    pred_tg.x.numpy(),
                    pred_tg.edge_index.numpy(),
                    pred_tg.edge_attr.numpy(),
                ),
            )
        )
    return predicted_graphs


def evaluate_graphs(
    predicted_graphs,
    nodes_per_m: float = 1.0,
    lane_width: float = 10.0,
    force_dense: bool = False,
):
    results = []
    _, predicted_graphs = predicted_graphs  # can get the process index here
    pbar = tqdm(enumerate(predicted_graphs))
    for j, (gt, pred) in pbar:
        pred_bg = BezierGraph.from_graph_data(*pred, assume_log_distances=False)
        pred_rx = rx.networkx_converter(
            pred_bg.to_simple(nodes_per_m=nodes_per_m), keep_attributes=True
        )

        gt_d = torch_geometric.data.Data(
            x=torch.Tensor(gt[0]), edge_index=torch.from_numpy(gt[1])
        )
        gt_g = tg_to_rx(gt_d)
        gt_g_interpolated = interpolate_graph(gt_g, nodes_per_m=nodes_per_m)

        res = {}
        res.update(
            geo_topo_metric(
                gt_g_interpolated, pred_rx, interpolate=False, force_dense=force_dense
            )
        )
        res["apls"] = simple_apls(gt_g_interpolated, pred_rx)
        res["iou"] = graph_iou(
            rx_to_poly(gt_g, lane_width=lane_width),
            pred_bg.to_poly(lane_width=lane_width),
        )
        for t in [20, 50]:
            p, r, a = sda(gt_g_interpolated, pred_rx, threshold=t)
            res.update(
                {f"sda_{t}_precision": p, f"sda_{t}_recall": r, f"sda_{t}_accuracy": a}
            )

        results.append(res)
    return results


def evaluate_graphs_multi(predicted_graphs, num_processes, **kwargs):
    kwarg_evaluate_graphs = partial(evaluate_graphs, **kwargs)
    chunks = [(i, predicted_graphs[i::num_processes]) for i in range(num_processes)]
    with get_context("spawn").Pool(num_processes) as p:
        out = p.map(kwarg_evaluate_graphs, chunks)
    return [r for res in out for r in res]
