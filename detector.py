"""
Real-time web object detection backend.
Runs inference on frames received from browser camera and returns bounding boxes.
"""
import base64
import io
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    class_id: int

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x1": round(self.x1, 2), "y1": round(self.y1, 2),
            "x2": round(self.x2, 2), "y2": round(self.y2, 2),
            "label": self.label, "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "width": round(self.width, 2), "height": round(self.height, 2),
        }


@dataclass
class DetectionResult:
    frame_id: str
    detections: List[BoundingBox]
    inference_ms: float
    frame_width: int
    frame_height: int
    timestamp: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.detections)

    def by_label(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.label] = counts.get(d.label, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "count": self.count,
            "inference_ms": round(self.inference_ms, 1),
            "frame_size": f"{self.frame_width}x{self.frame_height}",
            "detections": [d.to_dict() for d in self.detections],
            "by_label": self.by_label(),
            "timestamp": self.timestamp,
        }


class FramePreprocessor:
    """Decodes and preprocesses incoming frames from browser."""

    def from_base64(self, b64_data: str) -> Optional[np.ndarray]:
        if not CV2_AVAILABLE:
            return None
        try:
            header, encoded = b64_data.split(",", 1) if "," in b64_data else ("", b64_data)
            img_bytes = base64.b64decode(encoded)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return frame
        except Exception as exc:
            logger.error("Frame decode error: %s", exc)
            return None

    def resize(self, frame: np.ndarray, target: int = 640) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = target / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h))

    def stub_frame(self, width: int = 640, height: int = 480) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class YOLODetector:
    """
    Wraps YOLOv8 for object detection with stub fallback when model is unavailable.
    """

    DEFAULT_CLASSES = ["person", "forklift", "pallet", "vehicle", "hardhat",
                        "box", "conveyor", "crane", "safety_vest", "barrier"]

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self._model = None
        if YOLO_AVAILABLE:
            try:
                self._model = YOLO(model_path)
                logger.info("YOLO model loaded from %s", model_path)
            except Exception as exc:
                logger.warning("YOLO load failed: %s. Using stub.", exc)

    def detect(self, frame: np.ndarray) -> Tuple[List[BoundingBox], float]:
        t0 = time.perf_counter()
        if self._model is not None:
            try:
                results = self._model(frame, conf=self.confidence_threshold, verbose=False)
                boxes = []
                for result in results:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0])
                        label = result.names.get(cls_id, str(cls_id))
                        boxes.append(BoundingBox(
                            x1=float(xyxy[0]), y1=float(xyxy[1]),
                            x2=float(xyxy[2]), y2=float(xyxy[3]),
                            label=label,
                            confidence=float(box.conf[0]),
                            class_id=cls_id,
                        ))
                inf_ms = (time.perf_counter() - t0) * 1000
                return boxes, inf_ms
            except Exception as exc:
                logger.error("YOLO inference error: %s", exc)
        return self._stub_detect(frame, t0)

    def _stub_detect(self, frame: np.ndarray, t0: float) -> Tuple[List[BoundingBox], float]:
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        h, w = frame.shape[:2] if len(frame.shape) >= 2 else (480, 640)
        num_detections = int(rng.integers(0, 4))
        boxes = []
        for _ in range(num_detections):
            x1 = float(rng.integers(0, w - 100))
            y1 = float(rng.integers(0, h - 100))
            x2 = x1 + float(rng.integers(50, 200))
            y2 = y1 + float(rng.integers(50, 200))
            label = self.DEFAULT_CLASSES[rng.integers(0, len(self.DEFAULT_CLASSES))]
            boxes.append(BoundingBox(
                x1=min(x1, w), y1=min(y1, h),
                x2=min(x2, w), y2=min(y2, h),
                label=label,
                confidence=float(rng.uniform(0.45, 0.99)),
                class_id=self.DEFAULT_CLASSES.index(label),
            ))
        inf_ms = (time.perf_counter() - t0) * 1000
        return boxes, inf_ms


class DetectionStore:
    """Persists detection results in SQLite for audit and analytics."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS detections (
        frame_id TEXT,
        label TEXT,
        confidence REAL,
        x1 REAL, y1 REAL, x2 REAL, y2 REAL,
        frame_width INTEGER,
        frame_height INTEGER,
        inference_ms REAL,
        detected_at REAL
    );
    CREATE TABLE IF NOT EXISTS cropped_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        frame_id TEXT,
        label TEXT,
        image_b64 TEXT,
        created_at REAL
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def save_result(self, result: DetectionResult) -> None:
        rows = [
            (result.frame_id, d.label, d.confidence, d.x1, d.y1, d.x2, d.y2,
             result.frame_width, result.frame_height, result.inference_ms, result.timestamp)
            for d in result.detections
        ]
        if rows:
            self.conn.executemany(
                "INSERT INTO detections VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
            )
            self.conn.commit()

    def recent_counts(self, hours: float = 1.0) -> List[Dict]:
        since = time.time() - hours * 3600
        import pandas as pd
        df = pd.read_sql_query(
            f"SELECT label, COUNT(*) AS count, AVG(confidence) AS avg_conf "
            f"FROM detections WHERE detected_at >= {since} GROUP BY label ORDER BY count DESC",
            self.conn,
        )
        return df.to_dict(orient="records")


class ObjectDetectionService:
    """
    High-level service combining preprocessing, detection, and storage.
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.45,
                 db_path: str = ":memory:"):
        self.preprocessor = FramePreprocessor()
        self.detector = YOLODetector(model_path=model_path,
                                      confidence_threshold=confidence_threshold)
        self.store = DetectionStore(db_path=db_path)
        self._frame_counter = 0

    def process_b64_frame(self, b64_data: str) -> DetectionResult:
        self._frame_counter += 1
        frame_id = f"frame_{self._frame_counter:06d}"
        frame = self.preprocessor.from_base64(b64_data)
        if frame is None:
            frame = self.preprocessor.stub_frame()
        boxes, inf_ms = self.detector.detect(frame)
        h, w = frame.shape[:2] if len(frame.shape) >= 2 else (480, 640)
        result = DetectionResult(
            frame_id=frame_id,
            detections=boxes,
            inference_ms=inf_ms,
            frame_width=w,
            frame_height=h,
        )
        self.store.save_result(result)
        return result

    def process_numpy_frame(self, frame: np.ndarray) -> DetectionResult:
        self._frame_counter += 1
        frame_id = f"frame_{self._frame_counter:06d}"
        boxes, inf_ms = self.detector.detect(frame)
        h, w = frame.shape[:2]
        result = DetectionResult(
            frame_id=frame_id,
            detections=boxes,
            inference_ms=inf_ms,
            frame_width=w,
            frame_height=h,
        )
        self.store.save_result(result)
        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    service = ObjectDetectionService(model_path="yolov8n.pt", confidence_threshold=0.45)
    preprocessor = FramePreprocessor()

    print("Object Detection Service Demo\n")
    for i in range(5):
        stub_frame = preprocessor.stub_frame(640, 480)
        result = service.process_numpy_frame(stub_frame)
        print(f"Frame {result.frame_id}: {result.count} detections in {result.inference_ms:.1f}ms")
        for det in result.detections:
            print(f"  [{det.label}] conf={det.confidence:.2f} "
                  f"box=({det.x1:.0f},{det.y1:.0f},{det.x2:.0f},{det.y2:.0f})")

    print("\nDetection counts (last hour):")
    for row in service.store.recent_counts(1.0):
        print(f"  {row['label']}: {row['count']} detections, avg_conf={row['avg_conf']:.2f}")
