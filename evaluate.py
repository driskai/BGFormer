import glob
import json
import os
import pickle as pkl

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from aerial_lane_bezier.metrics.evaluate import (
    evaluate_graphs,
    evaluate_graphs_multi,
    predict_graphs,
)
from aerial_lane_bezier.model.bezier_detr import BezierDETR
from aerial_lane_bezier.model.image_processing import ImageProcessingBezierDETR
from inference import parser
from train import transform_aug_ann

parser.add_argument("--split", type=str, default="eval")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--num-samples", type=int, default=None)
parser.add_argument("--output", type=str, default="results.json")
parser.add_argument("--num_processes", type=int, default=1)
parser.add_argument("--process_batch_size", type=int, default=200)
parser.add_argument(
    "--force_dense",
    action="store_true",
    help=(
        "Force use of the dense linear sum assignment algorithm. Useful to try if your "
        "evaluation is running for a very long time (and if you have enough RAM)."
    ),
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

    if args.checkpoint is None:
        # Load the latest checkpoint
        possible_checkpoint_paths = glob.glob(
            os.path.join(checkpoint_dir, "checkpoint-*")
        )
        checkpoint_path = max(
            possible_checkpoint_paths, key=lambda x: int(x.split("checkpoint-")[-1])
        )
    else:
        checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)

    if args.experiment_type not in ("full", "successor"):
        raise ValueError(
            f"Invalid experiment type {args.experiment_type}, must be one of "
            + "'full' or 'successor'."
        )

    processed_files_dir = (
        "processed_files"
        if args.experiment_type == "full"
        else "successor_processed_files"
    )
    eval_dir = "eval" if args.experiment_type == "full" else "eval_succ_lgp"

    dataset_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "aerial_lane_bezier",
        "dataset",
        processed_files_dir,
        "huggingface",
    )

    dataset_features = pkl.load(
        open(
            os.path.join(
                dataset_dir,
                "..",
                "huggingface_features.pkl",
            ),
            "rb",
        )
    )

    # Manually create the dataset using from_dict, since I can't figure out how to
    # modify load_dataset do work with our different eval sub-directories.
    image_paths = glob.glob(os.path.join(dataset_dir, args.split, "*.png"))
    image_id_to_path = {
        int(os.path.basename(image_path).split(".")[0]): image_path
        for image_path in image_paths
    }
    metadata = pd.read_json(
        os.path.join(dataset_dir, args.split, "metadata.jsonl"), lines=True
    )

    ordered_image_ids = sorted(image_id_to_path.keys())
    ordered_image_paths = [image_id_to_path[image_id] for image_id in ordered_image_ids]
    ordered_metadata = metadata.loc[metadata["image_id"].isin(ordered_image_ids)]
    ordered_objects = ordered_metadata["objects"].values
    ordered_widths = ordered_metadata["width"].values
    ordered_heights = ordered_metadata["height"].values

    eval_dataset = Dataset.from_dict(
        {
            "image": ordered_image_paths,
            "image_id": ordered_image_ids,
            "objects": ordered_objects,
            "width": ordered_widths,
            "height": ordered_heights,
        }
    ).cast(dataset_features)

    dataset = eval_dataset.with_transform(
        lambda examples: transform_aug_ann(examples, args.split, image_processor)
    )

    categories = ["Node"]
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}
    image_processor = ImageProcessingBezierDETR.from_pretrained(checkpoint_path)
    model = BezierDETR.from_pretrained(
        checkpoint_path,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = image_processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    kwargs = dict(
        nodes_per_m=0.6,
        lane_width=10.0,
        force_dense=args.force_dense,
    )

    num_processes = args.num_processes
    predicted_graphs = []
    results = []
    for idx, batch in tqdm(
        enumerate(dataloader), total=len(dataset) // args.batch_size
    ):
        if args.num_samples is not None and idx > args.num_samples // args.batch_size:
            break
        graphs = predict_graphs(
            image_processor,
            model,
            batch,
            device,
            experiment_mode=args.experiment_type,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )

        if num_processes > 1:
            predicted_graphs.extend(graphs)
            if len(predicted_graphs) > args.process_batch_size * num_processes:
                res = evaluate_graphs_multi(predicted_graphs, num_processes, **kwargs)
                results.extend(res)
                predicted_graphs = []
        else:
            results.extend(evaluate_graphs((0, graphs), **kwargs))

    if len(predicted_graphs) > 0:
        if num_processes <= 1:
            raise ValueError("some graphs not evaluated")
        res = evaluate_graphs_multi(predicted_graphs, num_processes, **kwargs)
        results.extend(res)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
