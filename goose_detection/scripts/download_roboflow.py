"""
download_roboflow.py
Downloads the Canadian Geese Roboflow dataset in COCO format.

Usage:
    python scripts/download_roboflow.py --api_key YOUR_ROBOFLOW_API_KEY

Get your free API key at: https://app.roboflow.com/ → Settings → Roboflow API
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Your Roboflow API key")
    parser.add_argument("--output", default="data/raw/roboflow", help="Output directory")
    args = parser.parse_args()

    from roboflow import Roboflow

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("ml-dlq4x").project("goose-4um8t")
    dataset = project.version(1).download("coco", location=args.output)

    print(f"Downloaded to: {args.output}")
    print("Expected structure:")
    print("  data/raw/roboflow/")
    print("    train/  (images + _annotations.coco.json)")
    print("    valid/  (images + _annotations.coco.json)")
    print("    test/   (images + _annotations.coco.json)")


if __name__ == "__main__":
    main()
