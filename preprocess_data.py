import argparse
import os

from aerial_lane_bezier.dataset.bezier_dataset import AerialBezierGraphDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--raw_dataset_root",
    type=str,
    help="Path to the root of the raw dataset. "
    + "Should be a directory ending with 'urbanlanegraph-dataset-pub-v1.1'",
)
parser.add_argument(
    "--huggingface_output_folder",
    type=str,
    default=None,
    help="Path to the folder to output the HuggingFace dataset to. "
    + "If not specified, outputs to the processed_files subdirectory.",
)
parser.add_argument(
    "--random_sample_fraction",
    type=float,
    default=0.0,
    help="Fraction of the eventual dataset which should be made up of randomly sampled "
    + "tiles (rather than clusters).",
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
parser.add_argument("--output_image_crop_size", type=int, default=512)
parser.add_argument(
    "--medfilt_kernel_size", type=int, default=5, help="Median filter kernel size"
)
parser.add_argument(
    "--curvature_threshold",
    type=float,
    default=0.005,
    help="Threshold for curvature-based bezier graph simplification",
)

if __name__ == "__main__":
    args = parser.parse_args()

    rendered_images_filepath = args.rendered_images_filepath
    if rendered_images_filepath is None:
        rendered_images_filepath = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "full_rendered_dataset_samples",
        )

    for split in ["train", "eval", "eval_full_lgp"]:
        print(f"Processing split {split}...")
        dataset = AerialBezierGraphDataset(
            raw_dataset_root=args.raw_dataset_root,
            split=split,
            g2bkwargs={
                "medfilt_kernel_size": args.medfilt_kernel_size,
                "curvature_threshold": args.curvature_threshold,
            },
            output_image_crop_size=args.output_image_crop_size,
            random_sampling_tile_fraction=args.random_sample_fraction,
        )
        if split == "eval_full_lgp":
            continue
        dataset.to_huggingface(
            huggingface_folder=args.huggingface_output_folder,
            rendered_images_filepath=rendered_images_filepath,
            save_rendered_image_probability=args.save_rendered_image_probability
            if split == "train"
            else 1.0,
        )
