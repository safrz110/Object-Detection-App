"""
detection.py — Core YOLOv8 inference utilities
Handles model loading, inference, bounding-box drawing, and color management.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any


# ─────────────────────────────────────────────
# 80 COCO class names (standard order)
# ─────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def get_class_colors(num_classes: int = 80) -> List[tuple]:
    """
    Generate visually distinct BGR colors for each class using HSV spacing.
    Returns a list of (B, G, R) tuples.
    """
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        hsv = np.uint8([[[hue, 220, 230]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


CLASS_COLORS = get_class_colors(len(COCO_CLASSES))


# ─────────────────────────────────────────────
# ObjectDetector
# ─────────────────────────────────────────────
class ObjectDetector:
    """
    Wraps the Ultralytics YOLOv8 model with convenience methods for
    inference, threshold updates, and metadata retrieval.
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        max_det: int = 50,
    ) -> None:
        """
        Parameters
        ----------
        model_name : str
            One of yolov8n / yolov8s / yolov8m / yolov8l / yolov8x.
            Ultralytics auto-downloads weights on first use.
        conf_threshold : float
            Minimum confidence to keep a detection.
        iou_threshold : float
            Non-Maximum Suppression (NMS) overlap threshold.
        max_det : int
            Maximum detections returned per image.
        """
        from ultralytics import YOLO  # lazy import — keeps startup fast

        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        # Load model (downloads ~6 MB for nano on first run)
        self.model = YOLO(f"{model_name}.pt")
        self.class_names: List[str] = self.model.names  # type: ignore

    # ── public API ──────────────────────────────

    def update_params(
        self,
        conf: float | None = None,
        iou: float | None = None,
        max_det: int | None = None,
    ) -> None:
        """Hot-update inference parameters without reloading the model."""
        if conf is not None:
            self.conf_threshold = conf
        if iou is not None:
            self.iou_threshold = iou
        if max_det is not None:
            self.max_det = max_det

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a BGR numpy array.

        Returns
        -------
        List of dicts, each containing:
            label       : str   — class name
            confidence  : float — confidence score (0–1)
            x1, y1, x2, y2 : int — bounding box pixel coordinates
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = self.class_names.get(cls_id, f"class_{cls_id}")
                detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )
        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """Return a dictionary of model metadata for display."""
        try:
            params = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        except Exception:
            params = "N/A"

        return {
            "Model": self.model_name,
            "Classes": len(self.class_names),
            "Parameters": f"{params:.1f}M" if isinstance(params, float) else params,
            "Confidence Threshold": f"{self.conf_threshold:.0%}",
            "IoU Threshold": f"{self.iou_threshold:.0%}",
            "Max Detections": self.max_det,
            "Framework": "Ultralytics YOLOv8",
        }


# ─────────────────────────────────────────────
# Drawing Utilities
# ─────────────────────────────────────────────
def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_labels: bool = True,
    show_conf: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and optional labels on a BGR image in-place.

    Parameters
    ----------
    image      : BGR numpy array (will be modified in-place)
    detections : output of ObjectDetector.detect()
    show_labels : draw class name text
    show_conf  : append confidence % to label
    thickness  : box line thickness in pixels

    Returns
    -------
    The annotated BGR image.
    """
    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]

        # Pick a stable color per class
        try:
            cls_idx = COCO_CLASSES.index(label)
        except ValueError:
            cls_idx = hash(label) % len(CLASS_COLORS)
        color = CLASS_COLORS[cls_idx]

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Label background + text
        if show_labels:
            text = label
            if show_conf:
                text += f" {conf:.0%}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            label_y1 = max(y1 - th - baseline - 4, 0)
            label_y2 = y1

            # Semi-transparent background using addWeighted
            overlay = image.copy()
            cv2.rectangle(overlay, (x1, label_y1), (x1 + tw + 6, label_y2 + baseline), color, -1)
            cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

            # Text
            cv2.putText(
                image,
                text,
                (x1 + 3, y1 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )

    return image


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    """Proportionally resize a frame so width ≤ max_width."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
