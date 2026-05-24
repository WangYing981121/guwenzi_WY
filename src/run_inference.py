#!/usr/bin/env python3
"""
Ancient-chars OCR inference pipeline:
  1. YOLOv8 detects character bounding boxes on rubbing (拓片) images
  2. EfficientNet-B0 + ArcFace classifies each cropped character
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
from ultralytics import YOLO


# ─── Config (via env vars) ──────────────────────────────────────────────────
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
YOLO_WEIGHTS = Path(os.getenv("YOLO_WEIGHTS", "/app/src/best.pt"))
CLS_WEIGHTS = Path(os.getenv("CLS_WEIGHTS", "/app/src/model.pt"))
ID2CHAR_PATH = Path(os.getenv("ID2CHAR_PATH", "/app/src/ID_to_chinese.json"))

DET_CONF = float(os.getenv("DET_CONFIDENCE", "0.25"))
CLS_CONF = float(os.getenv("CLS_CONFIDENCE", "0.0"))   # 0 = keep all predictions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── ArcFace (inference-only) ───────────────────────────────────────────────
class ArcFaceHead(nn.Module):
    """Normalised linear layer matching ArcFace training head."""
    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_classes, feat_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features)
        W = F.normalize(self.W)
        return features @ W.T       # (N, num_classes)


# ─── Classification model ──────────────────────────────────────────────────
def build_classifier(num_classes=1307, feat_dim=512):
    """EfficientNet-B0 → Linear(1280→512) → BN → ArcFace."""
    backbone = efficientnet_b0(weights=None)
    in_features = backbone.classifier[1].in_features          # 1280
    backbone.classifier = nn.Identity()

    embedding = nn.Sequential(
        nn.Linear(in_features, feat_dim),
        nn.BatchNorm1d(feat_dim),
    )
    arcface = ArcFaceHead(feat_dim, num_classes)
    return backbone, embedding, arcface


def load_classifier_weights(backbone, embedding, arcface, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(ckpt["backbone"])
    embedding.load_state_dict(ckpt["embedding"])
    arcface.W.data.copy_(ckpt["arcface_W"])
    return backbone, embedding, arcface


# ─── Image transform for classification ────────────────────────────────────
cls_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─── Class mapping ──────────────────────────────────────────────────────────
def load_id2char(path: Path):
    """Load ID_to_chinese.json and return a list indexed by sorted class ID."""
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    keys = sorted(mapping.keys(), key=lambda x: int(x))
    return [mapping[k] for k in keys]


# ─── Find images ────────────────────────────────────────────────────────────
def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def _collect(path):
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in suffixes)

    if INPUT_DIR.exists():
        images = _collect(INPUT_DIR)
        if images:
            return images
        print(f"Warning: {INPUT_DIR} exists but no images found, "
              f"contents: {list(INPUT_DIR.iterdir())}")

    fallback_root = Path("/saisdata")
    if fallback_root.exists() and fallback_root != INPUT_DIR:
        images = _collect(fallback_root)
        if images:
            print(f"Fallback: found {len(images)} images in {fallback_root}")
            return images

    print(f"Error: no images found in {INPUT_DIR} or /saisdata")
    return []


# ─── Inference on one image ────────────────────────────────────────────────
@torch.no_grad()
def infer_one(yolo, backbone, embedding, arcface, idx2char, image_path):
    """Detect and classify characters in a single rubbing image."""
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # Step 1: YOLO detection
    results = yolo(img, conf=DET_CONF, verbose=False)
    boxes = results[0].boxes
    detections = []

    if boxes is None or len(boxes) == 0:
        return detections

    # Sort boxes top-to-bottom, left-to-right (for reading order)
    xyxy = boxes.xyxy.cpu().tolist()
    xyxy.sort(key=lambda b: (b[1], b[0]))

    for x1, y1, x2, y2 in xyxy:
        x1 = max(0, min(img_w - 1, int(round(x1))))
        y1 = max(0, min(img_h - 1, int(round(y1))))
        x2 = max(0, min(img_w, int(round(x2))))
        y2 = max(0, min(img_h, int(round(y2))))

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue

        # Step 2: Crop and classify
        crop = img.crop((x1, y1, x2, y2))
        input_tensor = cls_transform(crop).unsqueeze(0).to(DEVICE)

        feat = backbone(input_tensor)
        feat = embedding(feat)
        logits = arcface(feat)                     # (1, num_classes)

        scores = F.softmax(logits, dim=1)          # (1, num_classes)
        top_score, pred_idx = scores.max(dim=1)
        score = top_score.item()

        # Skip low-confidence predictions if threshold is set
        if CLS_CONF > 0 and score < CLS_CONF:
            continue

        predicted_char = idx2char[pred_idx.item()] if pred_idx.item() < len(idx2char) else ""

        detections.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "text": predicted_char,
        })

    return detections


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Locate input images
    image_paths = find_images()
    print(f"Input directory: {INPUT_DIR}")
    print(f"Images found: {len(image_paths)}")
    print(f"Device: {DEVICE}")
    if not image_paths:
        print("No images to process, writing empty result.")
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return

    # Load YOLO detector
    print(f"Loading detection model from {YOLO_WEIGHTS}...")
    if not YOLO_WEIGHTS.exists():
        print(f"FATAL: YOLO weights not found at {YOLO_WEIGHTS}")
        sys.exit(1)
    yolo = YOLO(str(YOLO_WEIGHTS))

    # Load classifier
    print(f"Loading classification model from {CLS_WEIGHTS}...")
    if not CLS_WEIGHTS.exists():
        print(f"FATAL: classifier weights not found at {CLS_WEIGHTS}")
        sys.exit(1)
    backbone, embedding, arcface = build_classifier()
    load_classifier_weights(backbone, embedding, arcface, str(CLS_WEIGHTS))
    backbone.to(DEVICE).eval()
    embedding.to(DEVICE).eval()
    arcface.to(DEVICE).eval()

    # Load character mapping
    idx2char = load_id2char(ID2CHAR_PATH)
    print(f"Character classes: {len(idx2char)}")

    # Run inference
    results = {}
    n_imgs = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        if index == 1 or index % 50 == 0 or index == n_imgs:
            print(f"[{index}/{n_imgs}] {image_path.name}")

        image_id = image_path.stem
        try:
            results[image_id] = infer_one(
                yolo, backbone, embedding, arcface, idx2char, image_path
            )
        except Exception as exc:
            print(f"Warning: failed to process {image_path}: {exc}")
            traceback.print_exc()
            results[image_id] = []

    # Save output
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"Saved: {OUTPUT_FILE}  ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
