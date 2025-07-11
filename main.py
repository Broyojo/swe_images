#!/usr/bin/env python3

import argparse
import json
import os

import docker
from swebench import build_instance_images
from swebench.harness.constants import LATEST
from swebench.harness.utils import load_swebench_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build SWE-bench images and save instance ID to image name mapping"
    )
    parser.add_argument(
        "dataset_name",
        help="Name of the SWE-bench dataset to download (e.g., 'princeton-nlp/SWE-bench_Verified')",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split to use (default: test)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="swebench_images.json",
        help="Output JSON file path (default: swebench_images.json)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads (default: number of CPU cores)",
    )

    args = parser.parse_args()

    # Set max_workers to CPU count if not specified
    max_workers = args.max_workers if args.max_workers is not None else os.cpu_count()

    print(f"Loading dataset: {args.dataset_name}, split: {args.split}")

    # Load the dataset
    try:
        dataset = load_swebench_dataset(args.dataset_name, args.split)
        print(f"Loaded {len(dataset)} instances from dataset")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1

    # Initialize Docker client
    client = docker.from_env(max_pool_size=1024)

    print(f"Building images with {max_workers} workers...")

    # Build instance images
    try:
        successful, failed = build_instance_images(
            client=client, dataset=dataset, max_workers=max_workers, tag=LATEST
        )

        print(f"Successfully built {len(successful)} images")
        print(f"Failed to build {len(failed)} images")

        if failed:
            print("\nFailed instances:")
            for failure in failed[:5]:  # Show first 5 failures
                print(f"  - {failure}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

    except Exception as e:
        print(f"Error building images: {e}")
        return 1

    # Create instance_id -> image_name mapping
    image_mapping = {}
    for success_tuple in successful:
        instance = success_tuple[0]  # The instance object
        instance_id = instance.instance_id
        image_name = instance.instance_image_key
        image_mapping[instance_id] = image_name

    # Save to JSON file
    try:
        with open(args.output, "w") as f:
            json.dump(image_mapping, f, indent=2)
        print(f"\nSaved image mapping to {args.output}")
        print(f"Total mappings saved: {len(image_mapping)}")

        # Show a few examples
        if image_mapping:
            print("\nExample mappings:")
            for i, (instance_id, image_name) in enumerate(
                list(image_mapping.items())[:3]
            ):
                print(f"  {instance_id} -> {image_name}")
            if len(image_mapping) > 3:
                print(f"  ... and {len(image_mapping) - 3} more")

    except Exception as e:
        print(f"Error saving to file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
