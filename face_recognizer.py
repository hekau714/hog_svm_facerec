import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from tqdm.auto import tqdm

from detect import detect_faces, resolve_model_path, resize_image


Detection = Union[
    Tuple[int, int, int, int],  # (x1, y1, x2, y2)
    Sequence[int],  # can be (x, y, w, h) or (x1, y1, x2, y2)
    dict,  # {"x1","y1","x2","y2"} or {"x","y","w","h"}
]


def _safe_crop(
    image_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int, margin_ratio: float = 0.10
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx)
    y2 = min(h, y2 + my)
    return image_bgr[y1:y2, x1:x2]


def prepare_cropped_faces_dataset(
    source_dataset: str = "known_faces",
    target_dataset: str = "known_faces_cropped",
    detector_model_path: str = "output/models/svm_face_detector.joblib",
    target_width: int = 200,
    step: int = 4,
    score_threshold: float = 1.2,
    nms_threshold: float = 0.25,
    min_scale: float = 0.8,
    max_scale: float = 2.8,
    scale_step: float = 0.2,
) -> None:
    detector = joblib.load(resolve_model_path(detector_model_path))
    src_root = Path(source_dataset)
    dst_root = Path(target_dataset)
    dst_root.mkdir(parents=True, exist_ok=True)

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    total, saved, skipped = 0, 0, 0
    for person_dir in sorted(src_root.iterdir()):
        if not person_dir.is_dir():
            continue
        out_person = dst_root / person_dir.name
        out_person.mkdir(parents=True, exist_ok=True)

        files = [p for p in sorted(person_dir.iterdir()) if p.is_file()]
        for img_file in tqdm(files, desc=f"Cropping {person_dir.name}", leave=False):
            total += 1
            image = cv2.imread(str(img_file))
            if image is None:
                skipped += 1
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_resized = resize_image(gray, target_width=target_width)
            boxes, scores, _ = detect_faces(
                model=detector,
                gray_resized=gray_resized,
                score_threshold=score_threshold,
                step=step,
                min_scale=min_scale,
                max_scale=max_scale,
                scale_step=scale_step,
                nms_threshold=nms_threshold,
            )
            if not boxes:
                skipped += 1
                continue

            # Keep the strongest detection for identity training.
            best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
            x1, y1, x2, y2 = boxes[best_idx]

            # Map detection box from resized-space to original image.
            rh, rw = gray_resized.shape[:2]
            oh, ow = gray.shape[:2]
            sx = ow / float(rw)
            sy = oh / float(rh)
            ox1 = int(round(x1 * sx))
            oy1 = int(round(y1 * sy))
            ox2 = int(round(x2 * sx))
            oy2 = int(round(y2 * sy))

            crop = _safe_crop(image, ox1, oy1, ox2, oy2, margin_ratio=0.10)
            if crop.size == 0:
                skipped += 1
                continue

            out_path = out_person / img_file.name
            cv2.imwrite(str(out_path), crop)
            saved += 1

    print(f"Cropped dataset saved to: {dst_root}")
    print(f"Total images: {total} | Saved crops: {saved} | Skipped: {skipped}")


def extract_hog_features(face_img_bgr: np.ndarray) -> np.ndarray:
    face_img = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
    face_img = resize(face_img, (64, 64), anti_aliasing=True)
    return hog(
        face_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
    )


def preprocess_lbph(face_img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
    # Normalize local contrast to improve robustness across different lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return gray


def _augment_lbph(gray_face: np.ndarray) -> List[np.ndarray]:
    variants: List[np.ndarray] = [gray_face]
    variants.append(cv2.flip(gray_face, 1))
    variants.append(cv2.convertScaleAbs(gray_face, alpha=1.15, beta=6))
    variants.append(cv2.convertScaleAbs(gray_face, alpha=0.88, beta=-6))
    variants.append(cv2.GaussianBlur(gray_face, (3, 3), 0))
    return variants


def load_known_faces(
    dataset_path: str = "known_faces", backend: str = "lbph"
) -> Tuple[List[np.ndarray], np.ndarray]:
    X: List[np.ndarray] = []
    y: List[str] = []
    root = Path(dataset_path)

    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        for img_file in sorted(person_dir.iterdir()):
            if not img_file.is_file():
                continue
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            if backend == "lbph":
                feat = preprocess_lbph(img)
            else:
                feat = extract_hog_features(img)
            X.append(feat)
            y.append(person_name)

    if not X:
        raise ValueError(
            f"No training images loaded from '{dataset_path}'. "
            "Expected folders like known_faces/Alice/*.jpg"
        )

    return X, np.asarray(y)


def _require_lbph_support() -> None:
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "LBPH backend requires opencv-contrib package. Install one of:\n"
            "- pip install opencv-contrib-python\n"
            "- pip install opencv-contrib-python-headless"
        )


def train_and_save_recognizer(
    dataset_path: str = "known_faces",
    model_out: str = "output/models/face_recognizer.joblib",
    backend: str = "lbph",
) -> None:
    backend = backend.strip().lower()
    if backend not in {"lbph", "hog_svm"}:
        raise ValueError("backend must be one of: lbph, hog_svm")

    X_train, y_train = load_known_faces(dataset_path, backend=backend)
    unique_people = sorted(set(y_train.tolist()))
    if len(unique_people) < 2:
        raise ValueError(
            "Need at least 2 different people to train recognizer. "
            f"Found {len(unique_people)} class: {unique_people}. "
            "Add another person folder in known_faces/<PersonName>/..."
        )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "lbph":
        _require_lbph_support()
        X_lbph: List[np.ndarray] = []
        y_lbph: List[int] = []
        for img, label in zip(X_train, y_encoded):
            for aug in _augment_lbph(img):
                X_lbph.append(aug)
                y_lbph.append(int(label))

        lbph = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=9, grid_y=9
        )
        lbph.train(X_lbph, np.asarray(y_lbph, dtype=np.int32))
        xml_path = out_path.with_suffix(".lbph.xml")
        lbph.write(str(xml_path))

        packed = {
            "backend": "lbph",
            "label_encoder": label_encoder,
            "model_xml": str(xml_path),
            "meta": {
                "dataset_path": dataset_path,
                "num_samples": int(len(y_train)),
                "augmented_samples": int(len(y_lbph)),
                "default_threshold": 0.42,  # confidence threshold [0..1]
                "distance_scale": 120.0,
            },
        }
    else:
        X_arr = np.asarray(X_train, dtype=np.float32)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
        svm = SVC(
            kernel="rbf",
            probability=True,
            C=10.0,
            gamma="scale",
            class_weight="balanced",
        )
        svm.fit(X_scaled, y_encoded)
        packed = {
            "backend": "hog_svm",
            "recognizer": svm,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "meta": {
                "dataset_path": dataset_path,
                "num_samples": int(len(y_train)),
                "default_threshold": 0.75,
                "default_margin": 0.12,
            },
        }

    joblib.dump(packed, out_path)
    print(f"Saved recognizer to: {out_path}")
    print(f"Backend: {backend}")
    print(f"People: {list(label_encoder.classes_)}")
    print(f"Samples: {len(y_train)}")


class FaceRecognizer:
    def __init__(self, model_path: str = "output/models/face_recognizer.joblib") -> None:
        packed = joblib.load(model_path)
        self.backend = packed.get("backend", "hog_svm")
        self.meta = packed.get("meta", {})
        self.label_encoder: LabelEncoder = packed["label_encoder"]

        if self.backend == "lbph":
            _require_lbph_support()
            self.lbph = cv2.face.LBPHFaceRecognizer_create()
            self.lbph.read(packed["model_xml"])
            self.recognizer = None
            self.scaler = None
        else:
            self.recognizer: SVC = packed["recognizer"]
            self.scaler: StandardScaler = packed["scaler"]
            self.lbph = None

    @staticmethod
    def _lbph_distance_to_conf(distance: float, scale: float = 120.0) -> float:
        # LBPH distance: lower is better; this maps it to [0..1] confidence.
        conf = np.exp(-max(0.0, float(distance)) / max(1.0, float(scale)))
        return float(np.clip(conf, 0.0, 1.0))

    def recognize_face(
        self,
        face_crop_bgr: np.ndarray,
        threshold: float | None = None,
        min_margin: float | None = None,
    ) -> Tuple[str, float]:
        if threshold is None:
            threshold = float(self.meta.get("default_threshold", 0.70))

        if self.backend == "lbph":
            gray = preprocess_lbph(face_crop_bgr)
            pred_label, distance = self.lbph.predict(gray)
            distance_scale = float(self.meta.get("distance_scale", 120.0))
            confidence = self._lbph_distance_to_conf(distance, scale=distance_scale)
            if confidence < threshold:
                return "Unknown", confidence
            name = self.label_encoder.inverse_transform([int(pred_label)])[0]
            return str(name), confidence

        # hog_svm backend
        if min_margin is None:
            min_margin = float(self.meta.get("default_margin", 0.12))
        features = extract_hog_features(face_crop_bgr)
        features = self.scaler.transform([features])
        probs = self.recognizer.predict_proba(features)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])

        sorted_probs = np.sort(probs)
        second = float(sorted_probs[-2]) if len(sorted_probs) > 1 else 0.0
        margin = confidence - second

        if confidence < threshold or margin < min_margin:
            return "Unknown", confidence
        name = self.label_encoder.inverse_transform([best_idx])[0]
        return str(name), confidence

    @staticmethod
    def _to_xyxy(det: Detection) -> Tuple[int, int, int, int]:
        if isinstance(det, dict):
            if {"x1", "y1", "x2", "y2"}.issubset(det):
                return int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            if {"x", "y", "w", "h"}.issubset(det):
                x, y, w, h = int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])
                return x, y, x + w, y + h
            raise ValueError(f"Unsupported dict detection format: {det}")

        if len(det) < 4:
            raise ValueError(f"Detection must have at least 4 values: {det}")
        x1, y1, a, b = [int(v) for v in det[:4]]
        if a > 0 and b > 0 and (a < x1 or b < y1):
            return x1, y1, x1 + a, y1 + b
        return x1, y1, a, b

    def annotate(
        self,
        image_bgr: np.ndarray,
        detections: Iterable[Detection],
        threshold: float | None = None,
        min_margin: float | None = None,
    ) -> np.ndarray:
        output = image_bgr.copy()
        h, w = image_bgr.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = self._to_xyxy(det)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = image_bgr[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            name, conf = self.recognize_face(
                face_crop, threshold=threshold, min_margin=min_margin
            )
            label = f"{name} ({conf:.2f})"
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/load face recognizer")
    parser.add_argument(
        "--prepare-crops",
        action="store_true",
        help="Create cropped-face dataset using detector before training",
    )
    parser.add_argument(
        "--source-dataset",
        default="known_faces",
        help="Source dataset with raw images (person-per-folder)",
    )
    parser.add_argument(
        "--cropped-dataset",
        default="known_faces_cropped",
        help="Target dataset for detector-based face crops",
    )
    parser.add_argument(
        "--detector-model",
        default="output/models/svm_face_detector.joblib",
        help="Detector model path used for face cropping",
    )
    parser.add_argument("--train", action="store_true", help="Train and save recognizer")
    parser.add_argument("--dataset", default="known_faces", help="Path to known faces folder")
    parser.add_argument(
        "--model-out",
        default="output/models/face_recognizer.joblib",
        help="Path to save recognizer",
    )
    parser.add_argument(
        "--backend",
        default="lbph",
        choices=["lbph", "hog_svm"],
        help="Recognition backend. lbph recommended for easier setup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.prepare_crops:
        prepare_cropped_faces_dataset(
            source_dataset=args.source_dataset,
            target_dataset=args.cropped_dataset,
            detector_model_path=args.detector_model,
        )

    if args.train:
        train_dataset = args.dataset
        if args.prepare_crops:
            train_dataset = args.cropped_dataset
        train_and_save_recognizer(
            dataset_path=train_dataset, model_out=args.model_out, backend=args.backend
        )
    else:
        print("Use --train to train model, e.g.:")
        print("python face_recognizer.py --prepare-crops --train --backend lbph")
