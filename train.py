import argparse
import os
from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import Trainer, TrainingArguments

import wandb
from aerial_lane_bezier.model.bezier_detr import BezierDETR, BezierDetrConfig
from aerial_lane_bezier.model.image_processing import ImageProcessingBezierDETR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="SenseTime/deformable-detr",
    help="Huggingface checkpoint to load DETR backbone from.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for reproducibility."
)
parser.add_argument(
    "--experiment_type",
    type=str,
    default="full",
    help="Experiment type: full or successor",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default=None,
    help=(
        "Directory to save checkpoints to. "
        "If None, defaults to the checkpoints subdirectory."
    ),
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default=None,
    help=(
        "Path to the dataset directory. "
        "If None, defaults to the processed_files/huggingface subdirectory."
    ),
)
parser.add_argument("--epochs", type=int, default=250, help="Number of epochs to run.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument(
    "--val_split_size", type=float, default=0.2, help="Fractional validation split."
)
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
parser.add_argument(
    "--lr_scheduler_type",
    type=str,
    default="cosine",
    help="The scheduler type to use, passed to Huggingface.",
)
parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=16,
    help="Number of dataloader workers.",
)
parser.add_argument(
    "--save_steps", type=int, default=1000, help="Checkpoint save frequency in steps."
)
parser.add_argument(
    "--logging_steps", type=int, default=50, help="Logging frequency in steps."
)
parser.add_argument(
    "--evaluation_steps", type=int, default=1000, help="Evaluation frequency in steps."
)
parser.add_argument(
    "--save_total_limit",
    type=int,
    default=2,
    help="Maximum number of checkpoints to save.",
)
parser.add_argument(
    "--log_to_wandb",
    action="store_true",
    help="If true, will login and log to Weights and Biases.",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help=(
        "If provided, this is the name of the run as logged in wandb. This will also "
        "be the name of the checkpoint directory."
    ),
)
parser.add_argument(
    "--wandb_project_name",
    type=str,
    default="aerial_lane_bezier",
    help="Name to assign to the Weights and Biases project.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help=(
        "If provided, this is the entity (e.g. team) that this run will be "
        "logged to in wandb."
    ),
)


def formatted_anns(image_id, category, node_attribute, edge_indices, edge_attributes):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "node_attributes": list(node_attribute[i]),
        }
        annotations.append(new_ann)

    for i in range(0, len(edge_indices)):
        new_ann = {
            "image_id": image_id,
            "edge_indices": list(edge_indices[i]),
            "edge_attributes": list(edge_attributes[i]),
        }
        annotations.append(new_ann)

    return annotations


def rotate_graph(graph_dict, angle, center):
    # Create a copy of nodes to avoid modifying the original list while iterating
    new_nodes = []

    for node_attrs in graph_dict["nodes"]:
        x, y, u, v = node_attrs

        # Rotate (x, y)
        x -= center
        y -= center
        new_x = x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle))
        new_y = x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle))
        new_x += center
        new_y += center

        # Rotate bezier direction vector
        new_u = u * np.cos(np.radians(angle)) - v * np.sin(np.radians(angle))
        new_v = u * np.sin(np.radians(angle)) + v * np.cos(np.radians(angle))

        new_nodes.append([new_x, new_y, new_u, new_v])

    graph_dict["nodes"] = new_nodes

    return graph_dict


def random_rotate(image, graph):
    rotate_direction = np.random.choice([0, 90, 180, 270])
    center = image.size[1] // 2
    image = image.rotate(
        -rotate_direction, resample=Image.BICUBIC, center=(center, center)
    )
    graph = rotate_graph(graph, rotate_direction, center)

    return image, graph


def transform_aug_ann(examples, split, image_processor, rotate=True):
    image_ids = examples["image_id"]
    images, node_attributes, categories, edge_indices, edge_attributes = (
        [],
        [],
        [],
        [],
        [],
    )

    train_transform = transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
    )
    for image, objects in zip(examples["image"], examples["objects"]):
        graph = {
            "nodes": objects[0]["node_attributes"],
            "edges": objects[0]["edge_indices"],
        }

        if split == "train":
            if rotate:
                image, graph = random_rotate(image, graph)
            image = train_transform(image)

        image = np.array(image.convert("RGB"))[:, :, ::-1]

        images.append(image)
        node_attributes.append(graph["nodes"])
        categories.append(objects[0]["categories"])
        edge_indices.append(graph["edges"])
        edge_attributes.append(objects[0]["edge_attributes"])

    targets = [
        {
            "image_id": id_,
            "annotations": formatted_anns(id_, cat_, attr_, edge_ind_, edge_attr_),
        }
        for id_, cat_, attr_, edge_ind_, edge_attr_ in zip(
            image_ids, categories, node_attributes, edge_indices, edge_attributes
        )
    ]

    tensor_values = image_processor(
        images=images, annotations=targets, return_tensors="pt"
    )

    return tensor_values


class CustomTrainer(Trainer):
    """Trainer for the model."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.rolling_loss_dict = {}
        self.rolling_loss_count = 0

        self.eval_rolling_loss_dict = {}
        self.eval_rolling_loss_count = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        are_eval = not model.training and len(outputs["loss_dict"]) > 0
        with torch.no_grad():
            if are_eval:
                self.eval_rolling_loss_count += 1
                count = self.eval_rolling_loss_count
            else:
                self.rolling_loss_count += 1
                count = self.rolling_loss_count
            rolling_dict = (
                self.eval_rolling_loss_dict if are_eval else self.rolling_loss_dict
            )
            loss_dict = outputs["loss_dict"]
            for key, value in loss_dict.items():
                if key not in rolling_dict:
                    rolling_dict[key] = torch.tensor(0.0).to(loss.device)
                rolling_dict[key] += (value - rolling_dict[key]) / count

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float]):
        for key, value in self.rolling_loss_dict.items():
            logs[key] = float(value.item())
        self.rolling_loss_dict = {}
        self.rolling_loss_count = 0

        for key, value in self.eval_rolling_loss_dict.items():
            logs[f"eval_{key}"] = float(value.item())
        self.eval_rolling_loss_dict = {}
        self.eval_rolling_loss_count = 0
        super().log(logs)


if __name__ == "__main__":
    args = parser.parse_args()

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

    # Login to wandb
    if args.log_to_wandb:
        wandb.login()

    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        # Default to the processed data huggingface directory
        dataset_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "aerial_lane_bezier",
            "dataset",
            processed_files_dir,
            "huggingface",
        )

    original_dataset = load_dataset(
        "imagefolder",
        data_dir=dataset_dir,
        split="train",
        keep_in_memory=True,
    ).train_test_split(test_size=args.val_split_size, seed=args.seed)

    categories = ["Node"]

    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    config_dir = checkpoint_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "aerial_lane_bezier",
        "model",
        "configs",
    )

    model_config_path = os.path.join(config_dir, "model_config_changes.json")

    model_config_changes = BezierDetrConfig.from_json_file(model_config_path)

    checkpoint = args.checkpoint
    image_processor = ImageProcessingBezierDETR.from_pretrained(checkpoint)
    model = BezierDETR.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.update_configs(
        model_config_changes.mlp_head_hidden_dimension,
        model_config_changes.position_loss_coefficient,
        model_config_changes.direction_loss_coefficient,
        model_config_changes.edge_existence_loss_coefficient,
        model_config_changes.edge_attribute_loss_coefficient,
        model_config_changes.num_negative_edges_per_positive_edge,
    )

    # transforming a batch
    # Apply image processor
    original_dataset["train"] = original_dataset["train"].with_transform(
        lambda examples: transform_aug_ann(
            examples, "train", image_processor, rotate=args.experiment_type == "full"
        )
    )
    original_dataset["test"] = original_dataset["test"].with_transform(
        lambda examples: transform_aug_ann(
            examples, "test", image_processor, rotate=args.experiment_type == "full"
        )
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

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        if args.wandb_run_name is None:
            # Default to the checkpoints/default directory
            checkpoint_dir = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "aerial_lane_bezier",
                "model",
                "checkpoints",
                "default",
            )
        else:
            # Save to a directory called the model name
            checkpoint_dir = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "aerial_lane_bezier",
                "model",
                "checkpoints",
                args.wandb_run_name,
            )
        # Ensure that directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        logging_first_step=True,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb" if args.log_to_wandb else "none",
        dataloader_num_workers=args.dataloader_num_workers,
        evaluation_strategy="steps",
        eval_steps=args.evaluation_steps,
        load_best_model_at_end=True,
    )

    train_dataset = original_dataset["train"]
    validation_dataset = original_dataset["test"]

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=image_processor,
    )

    if args.log_to_wandb:
        with wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=args,
            config_exclude_keys=[
                "wandb_run_name",
                "wandb_entity",
                "dataset_dir",
                "save_steps",
                "logging_steps",
                "evaluation_steps",
                "save_total_limit",
                "log_to_wandb",
                "wandb_project_name",
            ],
        ) as run:
            trainer.train()
    else:
        trainer.train()

    trainer.save_model()
