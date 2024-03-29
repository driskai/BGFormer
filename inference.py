import argparse
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from aerial_lane_bezier.dataset.bezier_graph import BezierGraph
from aerial_lane_bezier.model.bezier_detr import BezierDETR
from aerial_lane_bezier.model.image_processing import ImageProcessingBezierDETR

parser = argparse.ArgumentParser()

checkpoint_group = parser.add_mutually_exclusive_group()
parser.add_argument(
    "--experiment_type",
    type=str,
    default="full",
    help="Experiment type: full or successor",
)
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
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint to run. " + "If None, defaults to the latest checkpoint.",
)
parser.add_argument(
    "--num_visualisations",
    type=int,
    default=8,
    help="Number of images to visualise. Note: will be run through in one batch, so "
    + "this cannot be too large.",
)
parser.add_argument(
    "--node_threshold",
    type=float,
    default=0.5,
    help="Probability threshold below which to discard nodes. "
    + "Note this is also used to pre-filter potential edges.",
)
parser.add_argument(
    "--edge_threshold",
    type=float,
    default=0.5,
    help="Probability threshold below which to discard edges.",
)
parser.add_argument(
    "--output_filename",
    type=str,
    default="output.png",
    help="Name of the output image.",
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

    image_processor = ImageProcessingBezierDETR.from_pretrained(checkpoint_path)
    model = BezierDETR.from_pretrained(checkpoint_path)

    image_directory = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "aerial_lane_bezier",
        "dataset",
        processed_files_dir,
        "huggingface",
        eval_dir,
    )
    possible_image_paths = glob.glob(os.path.join(image_directory, "*.png"))
    # Pick random images
    selected_image_paths = random.sample(possible_image_paths, args.num_visualisations)
    print("Randomly picked images: " + str(selected_image_paths))

    images = [
        Image.open(
            image_path,
        )
        for image_path in selected_image_paths
    ]

    image_ids = [
        int(os.path.basename(image_path).split(".")[0])
        for image_path in selected_image_paths
    ]

    metadata = pd.read_json(
        path_or_buf=os.path.join(
            os.path.dirname(selected_image_paths[0]), "metadata.jsonl"
        ),
        lines=True,
    )

    # NOTE: HAS TO BE SYNCED! Hence global variable
    NODE_THRESHOLD = args.node_threshold

    with torch.no_grad():
        print("Running forward pass.")
        inputs = image_processor(images=images, return_tensors="pt")
        outputs = model(**inputs, inference_edge_query_node_threshold=NODE_THRESHOLD)
        target_sizes = torch.tensor([images[0].size[::-1]])
        results = image_processor.post_process_object_detection(
            outputs,
            node_threshold=NODE_THRESHOLD,
            edge_threshold=args.edge_threshold,
            target_sizes=target_sizes,
        )
        print("Finished forward pass.")

    # Create figure with subplots, adjust its size
    fig, axs = plt.subplots(
        nrows=len(selected_image_paths) // 2, ncols=2, figsize=(20, 40)
    )

    for image, image_id, result, ax in tqdm(
        zip(images, image_ids, results, axs.flatten()), total=len(images)
    ):
        draw = ImageDraw.Draw(image)
        for score, label, node_attribute in zip(
            result["scores"], result["labels"], result["out_node_attributes"]
        ):
            x, y, u, v = tuple(node_attribute)
            # Draw circle of 10 pixels centered at x,y
            x, y = int(x), int(y)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="red", width=1)

            # Normalise u, v to a unit vector
            direction_vector = np.array([u, v])
            direction_vector = direction_vector / np.linalg.norm(direction_vector)
            u, v = direction_vector.tolist()

            # Draw line of 20 pixels from x,y to x+u*20, y+v*20
            x2, y2 = int(x + u * 20), int(y + v * 20)
            draw.line((x, y, x2, y2), fill="blue", width=1)

        for node_attribute in metadata[metadata["image_id"] == image_id][
            "objects"
        ].values[0][0]["node_attributes"]:
            x, y, _, _ = tuple(node_attribute)
            # Draw circle of 10 pixels centered at x,y
            x, y = int(x), int(y)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline="green", width=1)

        ax.imshow(image)

        pyg_graph = result["pyg_graph"]
        if pyg_graph is not None:
            bezier_graph = BezierGraph.from_pyg_graph(pyg_graph)
            bezier_graph.plot(ax)
    fig.tight_layout()
    plt.savefig(args.output_filename)
