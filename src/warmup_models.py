#!/usr/bin/env python3
"""Pre-load YOLO and classification models at build time."""
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    print("Loading YOLO detection model...")
    model_path = "/app/src/best.pt"
    if Path(model_path).exists():
        model = YOLO(model_path)
        print(f"YOLO model loaded: {model_path}")
    else:
        print(f"Warning: {model_path} not found at build time")

    print("Loading classification model...")
    ckpt_path = "/app/src/model.pt"
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        print(f"arcface_W shape: {ckpt['arcface_W'].shape}")
    else:
        print(f"Warning: {ckpt_path} not found at build time")

    print("All models are ready.")


if __name__ == "__main__":
    main()
