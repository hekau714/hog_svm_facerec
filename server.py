import asyncio
import base64
import inspect
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from detect import detect_faces, resolve_model_path, resize_image
from face_recognizer import FaceRecognizer


BASE_DIR = Path(__file__).parent
WEB_DIR = BASE_DIR / "web"
DEFAULT_MODEL = "output/models/svm_face_detector.joblib"
DEFAULT_RECOGNIZER_MODEL = "output/models/face_recognizer.joblib"


class FaceDetectorService:
    def __init__(self, model_path: str) -> None:
        self.model_path = resolve_model_path(model_path)
        self.model = joblib.load(self.model_path)
        self.detect_params = set(inspect.signature(detect_faces).parameters.keys())
        self.recognizer = None
        self.recognizer_path = Path(DEFAULT_RECOGNIZER_MODEL)
        if self.recognizer_path.exists():
            try:
                self.recognizer = FaceRecognizer(str(self.recognizer_path))
                print(f"[recog] Loaded face recognizer: {self.recognizer_path}")
            except Exception as exc:
                print(f"[recog] Failed to load recognizer: {exc}")
        else:
            print(
                f"[recog] Recognizer model not found at {self.recognizer_path}, detection-only mode."
            )

    def infer(
        self,
        frame_bgr: np.ndarray,
        target_width: int = 150,
        step: int = 4,
        score_threshold: float = 1.5,
        nms_threshold: float = 0.25,
        min_scale: float = 0.8,
        max_scale: float = 2.6,
        scale_step: float = 0.2,
        min_area_ratio: float = 0.05,
        max_detections: int = 1,
        enable_recognition: bool = True,
        recognition_threshold: float = 0.60,
        recognition_min_margin: float = 0.12,
    ) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = resize_image(gray, target_width=target_width)

        detect_kwargs: Dict[str, Any] = {
            "model": self.model,
            "gray_resized": gray_resized,
            "score_threshold": score_threshold,
            "step": step,
            "min_scale": min_scale,
            "max_scale": max_scale,
            "scale_step": scale_step,
            "nms_threshold": nms_threshold,
            "min_area_ratio": min_area_ratio,
            "max_detections": max_detections,
        }
        detect_kwargs = {
            key: value
            for key, value in detect_kwargs.items()
            if key in self.detect_params
        }

        boxes, scores, raw_count = detect_faces(**detect_kwargs)

        resized_h, resized_w = gray_resized.shape[:2]
        orig_h, orig_w = gray.shape[:2]
        scale_x = orig_w / float(resized_w)
        scale_y = orig_h / float(resized_h)

        payload_boxes: List[Dict[str, Any]] = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            ox1 = int(round(x1 * scale_x))
            oy1 = int(round(y1 * scale_y))
            ox2 = int(round(x2 * scale_x))
            oy2 = int(round(y2 * scale_y))
            payload_boxes.append(
                {
                    "x1": ox1,
                    "y1": oy1,
                    "x2": ox2,
                    "y2": oy2,
                    "score": round(float(score), 4),
                }
            )

        if enable_recognition and self.recognizer is not None:
            frame_h, frame_w = frame_bgr.shape[:2]
            for box in payload_boxes:
                x1 = max(0, min(frame_w, int(box["x1"])))
                y1 = max(0, min(frame_h, int(box["y1"])))
                x2 = max(0, min(frame_w, int(box["x2"])))
                y2 = max(0, min(frame_h, int(box["y2"])))
                if x2 <= x1 or y2 <= y1:
                    box["name"] = "Unknown"
                    box["name_confidence"] = 0.0
                    continue
                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    box["name"] = "Unknown"
                    box["name_confidence"] = 0.0
                    continue
                name, conf = self.recognizer.recognize_face(
                    crop,
                    threshold=recognition_threshold,
                    min_margin=recognition_min_margin,
                )
                box["name"] = name
                box["name_confidence"] = round(float(conf), 4)

        return {
            "boxes": payload_boxes,
            "raw_positives": raw_count,
            "final_detections": len(payload_boxes),
        }


def decode_data_url_to_bgr(data_url: str) -> np.ndarray:
    try:
        _, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid image payload: expected data URL") from exc

    image_bytes = base64.b64decode(encoded)
    frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise ValueError("Could not decode frame bytes into image")
    return frame_bgr


app = FastAPI(title="HOG SVM Face Detector")
app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")
service = FaceDetectorService(DEFAULT_MODEL)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def get_local_ipv4() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # No traffic is sent; this just lets OS pick the active interface.
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


@app.on_event("startup")
async def log_mobile_url() -> None:
    # Set APP_PORT / APP_SCHEME when launching to keep logs accurate.
    port = int(os.getenv("APP_PORT", "8000"))
    scheme = os.getenv("APP_SCHEME", "http").strip().lower() or "http"
    if scheme not in {"http", "https"}:
        scheme = "http"
    ip = get_local_ipv4()
    print(f"[mobile] Open on phone: {scheme}://{ip}:{port}")
    print(f"[local ] Open on pc:    {scheme}://127.0.0.1:{port}")


@app.websocket("/ws")
async def websocket_detect(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            try:
                raw = await websocket.receive_text()
                packet = json.loads(raw)

                frame_id = packet.get("frame_id")
                image_data = packet.get("image")
                if not image_data:
                    await websocket.send_text(
                        json.dumps({"frame_id": frame_id, "error": "Missing image"})
                    )
                    continue

                t0 = time.perf_counter()
                frame_bgr = decode_data_url_to_bgr(image_data)
                result = service.infer(
                    frame_bgr=frame_bgr,
                    target_width=int(packet.get("target_width", 150)),
                    step=int(packet.get("step", 4)),
                    score_threshold=float(packet.get("score_threshold", 1.5)),
                    nms_threshold=float(packet.get("nms_threshold", 0.25)),
                    min_scale=float(packet.get("min_scale", 0.8)),
                    max_scale=float(packet.get("max_scale", 2.6)),
                    scale_step=float(packet.get("scale_step", 0.2)),
                    min_area_ratio=float(packet.get("min_area_ratio", 0.05)),
                    max_detections=int(packet.get("max_detections", 1)),
                    enable_recognition=bool(packet.get("enable_recognition", True)),
                    recognition_threshold=float(packet.get("recognition_threshold", 0.60)),
                    recognition_min_margin=float(packet.get("recognition_min_margin", 0.12)),
                )

                # Client sends compressed frame (e.g. width=150). Rescale boxes back
                # to original camera resolution for correct overlay rendering.
                orig_width = int(packet.get("orig_width", 0))
                orig_height = int(packet.get("orig_height", 0))
                sent_height, sent_width = frame_bgr.shape[:2]
                if (
                    orig_width > 0
                    and orig_height > 0
                    and sent_width > 0
                    and sent_height > 0
                ):
                    sx = orig_width / float(sent_width)
                    sy = orig_height / float(sent_height)
                    for box in result.get("boxes", []):
                        box["x1"] = int(round(box["x1"] * sx))
                        box["y1"] = int(round(box["y1"] * sy))
                        box["x2"] = int(round(box["x2"] * sx))
                        box["y2"] = int(round(box["y2"] * sy))

                result["frame_id"] = frame_id
                result["infer_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

                await websocket.send_text(json.dumps(result))
            except WebSocketDisconnect:
                break
            except Exception as exc:
                # Client may already be closed; in that case do not send.
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_text(json.dumps({"error": str(exc)}))
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        return
