#!/usr/bin/env python3
import json
import os
import traceback
from pathlib import Path

from paddleocr import PaddleOCR
from PIL import Image


INPUT_DIR = Path(os.getenv("INPUT_DIR", "/saisdata/13/eval/images"))
OUTPUT_FILE = Path(os.getenv("OUTPUT_FILE", "/saisresult/prediction.json"))
REQUEST_USE_GPU = os.getenv("USE_GPU", "1") not in {"0", "false", "False", "no", "NO"}
USE_ANGLE_CLS = os.getenv("USE_ANGLE_CLS", "1") not in {"0", "false", "False", "no", "NO"}
LANG = os.getenv("PADDLEOCR_LANG", "ch")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.0"))


def find_images():
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    if INPUT_DIR.exists():
        return sorted(path for path in INPUT_DIR.iterdir() if path.suffix.lower() in suffixes)

    fallback_root = Path("/saisdata")
    if fallback_root.exists():
        return sorted(path for path in fallback_root.rglob("*") if path.suffix.lower() in suffixes)

    return []


def normalize_ocr_lines(result):
    if not result:
        return []

    if isinstance(result, list) and len(result) == 1:
        return result[0] or []

    return result if isinstance(result, list) else []


def detect_use_gpu():
    if not REQUEST_USE_GPU:
        print("GPU disabled by USE_GPU=0")
        return False

    try:
        import paddle

        is_cuda_build = False
        for checker in (
            lambda: paddle.device.is_compiled_with_cuda(),
            lambda: paddle.is_compiled_with_cuda(),
        ):
            try:
                is_cuda_build = bool(checker())
                break
            except Exception:
                continue

        try:
            gpu_count = int(paddle.device.cuda.device_count())
        except Exception:
            gpu_count = 0

        print(f"Paddle CUDA build: {is_cuda_build}")
        print(f"Visible CUDA devices: {gpu_count}")

        if is_cuda_build and gpu_count > 0:
            return True
    except Exception as exc:
        print(f"Warning: failed to check CUDA devices: {exc}")

    print("GPU requested but no usable CUDA device was found; falling back to CPU.")
    return False


def polygon_to_bbox(points, image_width, image_height):
    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]

    x1 = max(0, min(image_width - 1, int(round(min(x_values)))))
    y1 = max(0, min(image_height - 1, int(round(min(y_values)))))
    x2 = max(0, min(image_width, int(round(max(x_values)))))
    y2 = max(0, min(image_height, int(round(max(y_values)))))

    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def infer_one(ocr, image_path):
    with Image.open(image_path) as img:
        image_width, image_height = img.size

    raw_result = ocr.ocr(str(image_path), cls=USE_ANGLE_CLS)
    lines = normalize_ocr_lines(raw_result)

    detections = []
    for line in lines:
        if not line or len(line) < 2:
            continue

        polygon = line[0]
        text_score = line[1]
        text = text_score[0] if text_score else ""
        score = float(text_score[1]) if text_score and len(text_score) > 1 else 0.0

        if not text or score < MIN_SCORE:
            continue

        bbox = polygon_to_bbox(polygon, image_width, image_height)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue

        detections.append({
            "bbox": [int(v) for v in bbox],
            "text": str(text),
        })

    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return detections


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    image_paths = find_images()
    print(f"Input directory: {INPUT_DIR}")
    print(f"Images found: {len(image_paths)}")
    use_gpu = detect_use_gpu()
    print(f"Use GPU requested: {REQUEST_USE_GPU}")
    print(f"Use GPU actual: {use_gpu}")
    print(f"Use angle classifier: {USE_ANGLE_CLS}")
    print(f"Language: {LANG}")
    print(f"Min score: {MIN_SCORE}")

    results = {}
    if not image_paths:
        print("No images found; writing an empty prediction file.")
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved: {OUTPUT_FILE}")
        return

    try:
        ocr = PaddleOCR(
            use_angle_cls=USE_ANGLE_CLS,
            lang=LANG,
            use_gpu=use_gpu,
            show_log=False,
        )
    except Exception:
        if not use_gpu:
            raise
        print("Warning: failed to initialize PaddleOCR with GPU; retrying on CPU.")
        traceback.print_exc()
        use_gpu = False
        ocr = PaddleOCR(
            use_angle_cls=USE_ANGLE_CLS,
            lang=LANG,
            use_gpu=False,
            show_log=False,
        )

    for index, image_path in enumerate(image_paths, start=1):
        if index == 1 or index % 50 == 0:
            print(f"[{index}/{len(image_paths)}] {image_path.name}")

        image_id = image_path.stem
        try:
            results[image_id] = infer_one(ocr, image_path)
        except Exception as exc:
            print(f"Warning: failed to process {image_path}: {exc}")
            traceback.print_exc()
            results[image_id] = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
