from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class VisionResult:
    annotated_image: np.ndarray
    objects: list[dict[str, Any]]
    classification: dict[str, Any]


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def classify_image(image_bgr: np.ndarray) -> dict[str, Any]:
    try:
        import torch
        from torchvision import models, transforms

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.eval()

        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = preprocess(rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probs, dim=0)

        return {
            "method": "resnet18",
            "class_id": int(class_idx.item()),
            "confidence": float(confidence.item()),
            "label": f"imagenet_class_{int(class_idx.item())}",
        }
    except Exception:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        brightness_label = "bright" if brightness >= 127 else "dark"
        return {
            "method": "fallback_brightness",
            "label": f"{brightness_label}_scene",
            "confidence": 0.5,
        }


def detect_objects(image_bgr: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    annotated = image_bgr.copy()
    detections: list[dict[str, Any]] = []

    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        result = model(annotated, verbose=False)[0]
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            label = model.names.get(cls_id, f"class_{cls_id}")
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (36, 106, 255), 2)
            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (int(x1), max(int(y1) - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (36, 106, 255),
                2,
            )
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                }
            )
        return annotated, detections
    except Exception:
        gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (36, 106, 255), 2)
            cv2.putText(
                annotated,
                "face 0.50",
                (x, max(y - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (36, 106, 255),
                2,
            )
            detections.append(
                {
                    "label": "face",
                    "confidence": 0.5,
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                }
            )

        if not detections:
            detections.append(
                {
                    "label": "no_object_detected",
                    "confidence": 0.0,
                    "bbox": None,
                }
            )

        return annotated, detections


def analyze_image(image: Image.Image) -> VisionResult:
    image_bgr = pil_to_bgr(image)
    annotated, objects = detect_objects(image_bgr)
    classification = classify_image(image_bgr)
    return VisionResult(
        annotated_image=annotated,
        objects=objects,
        classification=classification,
    )
