import argparse
import os

from aerial_lane_bezier.dataset.bezier_dataset import (
    AerialBezierGraphDataset,
    SuccessorAerialBezierGraphDataset,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_dataset_root",
    type=str,
    help="Path to the root of the raw dataset. "
    + "Should be a directory ending with 'urbanlanegraph-dataset-pub-v1.1'",
)
parser.add_argument(
    "--ulg_dataset_root",
    type=str,
    help="Path to the root of the `raw` dataset processed by the ULG code.",
)
parser.add_argument(
    "--huggingface_output_folder",
    type=str,
    default=None,
    help="Path to the folder to output the HuggingFace dataset to. "
    + "If not specified, outputs to the processed_files subdirectory.",
)
parser.add_argument(
    "--rendered_images_filepath",
    type=str,
    default=None,
    help="Path to the folder to output rendered dataset samples too, for visualisation"
    + " and debugging.",
)
parser.add_argument(
    "--save_rendered_image_probability",
    type=float,
    default=0.01,
    help="The probability of saving a rendered image for a given example, used to "
    + "render just a subsample of the overall dataset.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    processed_dataset_root = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "aerial_lane_bezier",
        "dataset",
        "successor_processed_files",
    )

    rendered_images_filepath = args.rendered_images_filepath
    if rendered_images_filepath is None:
        rendered_images_filepath = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "successor_rendered_dataset_samples",
        )

    # First process train data
    print("Processing train split...")
    dataset = SuccessorAerialBezierGraphDataset(
        raw_dataset_root=args.ulg_dataset_root,
        processed_dataset_root=processed_dataset_root,
        split="train",
    )
    dataset.to_huggingface(
        huggingface_folder=args.huggingface_output_folder,
        rendered_images_filepath=rendered_images_filepath,
        save_rendered_image_probability=args.save_rendered_image_probability,
    )

    # Now process eval data
    print("Processing eval split...")
    dataset = AerialBezierGraphDataset(
        raw_dataset_root=args.raw_dataset_root,
        processed_dataset_root=processed_dataset_root,
        split="eval_succ_lgp",
    )
    dataset.to_huggingface(
        huggingface_folder=args.huggingface_output_folder,
        rendered_images_filepath=rendered_images_filepath,
        save_rendered_image_probability=1.0,
    )
