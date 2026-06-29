import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np
from skimage.feature import hog


# Canonical window size used during training.
BASE_H = 40
BASE_W = 32


def resize_image(img: np.ndarray, target_width: int = 150) -> np.ndarray:
    img_h, img_w = img.shape[:2]
    scale = target_width / img_w
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def resolve_model_path(model_path: str) -> str:
    explicit = Path(model_path)
    if explicit.exists():
        return str(explicit)

    fallback = Path("output") / "models" / "svm_face_detector.joblib"
    if fallback.exists():
        return str(fallback)

    raise FileNotFoundError(
        f"Model was not found at '{model_path}' or '{fallback}'. Train/export model first."
    )


def extract_hog(patch: np.ndarray) -> np.ndarray:
    return hog(
        patch,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
    )


def box_area(box: Tuple[int, int, int, int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def is_inside(
    inner: Tuple[int, int, int, int],
    outer: Tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    return (
        inner[0] >= outer[0] + margin
        and inner[1] >= outer[1] + margin
        and inner[2] <= outer[2] - margin
        and inner[3] <= outer[3] - margin
    )


def suppress_nested_boxes(
    boxes: List[Tuple[int, int, int, int]],
    scores: List[float],
    margin: int = 0,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
    if not boxes:
        return boxes, scores

    keep_flags = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep_flags[i]:
            continue
        for j in range(len(boxes)):
            if i == j or not keep_flags[j]:
                continue
            # Drop box i if it is fully inside j (allow overlap, forbid containment).
            if box_area(boxes[i]) <= box_area(boxes[j]) and is_inside(
                boxes[i], boxes[j], margin=margin
            ):
                keep_flags[i] = False
                break

    kept_boxes = [b for b, keep in zip(boxes, keep_flags) if keep]
    kept_scores = [s for s, keep in zip(scores, keep_flags) if keep]
    return kept_boxes, kept_scores


def detect_faces(
    model,
    gray_resized: np.ndarray,
    score_threshold: float,
    step: int,
    min_scale: float,
    max_scale: float,
    scale_step: float,
    nms_threshold: float,
    suppress_nested: bool = True,
    containment_margin: int = 0,
) -> Tuple[List[Tuple[int, int, int, int]], List[float], int]:
    raw_boxes: List[List[int]] = []
    raw_scores: List[float] = []
    image_h, image_w = gray_resized.shape[:2]

    scales = np.arange(min_scale, max_scale + 1e-9, scale_step)
    for scale in scales:
        win_h = int(round(BASE_H * scale))
        win_w = int(round(BASE_W * scale))
        if win_h < 8 or win_w < 8:
            continue
        if win_h >= image_h or win_w >= image_w:
            continue

        local_step = max(2, int(round(step * scale)))
        scale_boxes: List[List[int]] = []
        scale_features: List[np.ndarray] = []

        for y in range(0, image_h - win_h + 1, local_step):
            for x in range(0, image_w - win_w + 1, local_step):
                patch = gray_resized[y : y + win_h, x : x + win_w]
                patch_for_hog = cv2.resize(
                    patch, (BASE_W, BASE_H), interpolation=cv2.INTER_AREA
                )

                feature_vector = extract_hog(patch_for_hog)
                scale_boxes.append([x, y, win_w, win_h])
                scale_features.append(feature_vector)

        if not scale_features:
            continue

        # Batch scoring per scale is faster than calling decision_function
        # for each window individually.
        feature_matrix = np.asarray(scale_features, dtype=np.float32)
        scale_scores = model.decision_function(feature_matrix)
        for box, score in zip(scale_boxes, scale_scores):
            score_f = float(score)
            if score_f > score_threshold:
                raw_boxes.append(box)
                raw_scores.append(score_f)

    if not raw_boxes:
        return [], [], 0

    idxs = cv2.dnn.NMSBoxes(raw_boxes, raw_scores, score_threshold, nms_threshold)
    if len(idxs) == 0:
        return [], [], len(raw_boxes)

    final_boxes: List[Tuple[int, int, int, int]] = []
    final_scores: List[float] = []

    for idx in np.array(idxs).reshape(-1):
        x, y, w, h = raw_boxes[idx]
        final_boxes.append((x, y, x + w, y + h))
        final_scores.append(raw_scores[idx])

    if suppress_nested:
        final_boxes, final_scores = suppress_nested_boxes(
            final_boxes, final_scores, margin=containment_margin
        )

    return final_boxes, final_scores, len(raw_boxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="HOG + LinearSVM face detection")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        default="output/models/svm_face_detector.joblib",
        help="Path to trained joblib model",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=150,
        help="Resize width before scanning (must match training regime closely)",
    )
    parser.add_argument("--step", type=int, default=4, help="Base sliding-window step")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=1.5,
        help="Decision score threshold",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.25,
        help="IoU threshold for NMS",
    )
    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.8,
        help="Minimum box scale relative to base window",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=2.6,
        help="Maximum box scale relative to base window",
    )
    parser.add_argument(
        "--scale-step",
        type=float,
        default=0.2,
        help="Scale increment",
    )
    parser.add_argument(
        "--output",
        default="output/results/detection.jpg",
        help="Path to save visualization",
    )
    parser.add_argument(
        "--allow-nested",
        action="store_true",
        help="Keep boxes fully inside other boxes",
    )
    parser.add_argument(
        "--containment-margin",
        type=int,
        default=0,
        help="Pixel margin for inside-box suppression",
    )
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    model = joblib.load(model_path)

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = resize_image(gray, target_width=args.target_width)

    boxes, scores, raw_count = detect_faces(
        model=model,
        gray_resized=gray_resized,
        score_threshold=args.score_threshold,
        step=args.step,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_step=args.scale_step,
        nms_threshold=args.nms_threshold,
        suppress_nested=not args.allow_nested,
        containment_margin=args.containment_margin,
    )

    # Map detections back to original image coordinates.
    resized_h, resized_w = gray_resized.shape[:2]
    orig_h, orig_w = gray.shape[:2]
    scale_x = orig_w / float(resized_w)
    scale_y = orig_h / float(resized_h)

    vis = image.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        ox1 = int(round(x1 * scale_x))
        oy1 = int(round(y1 * scale_y))
        ox2 = int(round(x2 * scale_x))
        oy2 = int(round(y2 * scale_y))
        cv2.rectangle(vis, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{score:.2f}",
            (ox1, max(0, oy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

    print(f"Model: {model_path}")
    print(f"Image: {args.image}")
    print(f"Raw positives: {raw_count}")
    print(f"Final detections (after NMS): {len(boxes)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
