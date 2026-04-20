"""
Phase A + B + D inference pipeline for panoramic X-rays.

Pipeline:
    Input image (any size)
    → CLAHE preprocessing
    → Phase B: YOLOv8x-seg → per-tooth instance masks + FDI class IDs
    → Phase D: crop each detected tooth → EfficientNetV2-M + GradCAM → disease label
    → Render: overlay masks on original image, color by disease
    → Output: annotated image + per-tooth results table
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.visualization import (
    DISEASE_COLORS,
    apply_clahe,
    blend_gradcam,
    draw_tooth_overlay,
)
from inference.models import DISEASE_NAMES, FDI_NUMBERS, N_CLS

if TYPE_CHECKING:
    from inference.models import ModelRegistry

# ─── Constants ───────────────────────────────────────────────────────────────

DISEASE_THRESHOLD = 0.50   # max disease prob below this → "Healthy"

_CLS_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((380, 380), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_tensor_cls(patch_rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB → (1, 3, 380, 380) float tensor, ImageNet-normalized."""
    return _CLS_TRANSFORM(patch_rgb).unsqueeze(0)


def _safe_mps(fn, *args, **kwargs):
    """Run fn(*args) on MPS; fall back to CPU if op not supported."""
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, NotImplementedError) as e:
        warnings.warn(f"MPS fallback to CPU: {e}", stacklevel=2)
        cpu_args = [a.cpu() if isinstance(a, torch.Tensor) else a for a in args]
        return fn(*cpu_args, **kwargs)


# ─── Main inference function ──────────────────────────────────────────────────

def run_panoramic(
    image_path: str,
    registry: "ModelRegistry",
    progress_cb=None,
) -> tuple[np.ndarray, list[dict], list[np.ndarray]]:
    """
    Run Phase B → D pipeline on a panoramic X-ray.

    Args:
        image_path: path to input X-ray image
        registry: loaded ModelRegistry
        progress_cb: optional callable(step, total, desc) for progress updates

    Returns:
        annotated_image: HxWx3 RGB numpy array with tooth overlays
        results: list of dicts with keys: fdi, disease, confidence, bbox
        gradcam_crops: list of HxWx3 RGB arrays (GradCAM-blended tooth crops)
    """
    def _progress(step, desc):
        if progress_cb:
            progress_cb(step, 3, desc)

    # ── Load + preprocess ─────────────────────────────────────────────────────
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_clahe = apply_clahe(img_rgb)

    _progress(1, "Phase B — FDI instance labeling (YOLO)")

    # ── Phase B: YOLO instance segmentation ──────────────────────────────────
    if registry.yolo_model is None:
        raise RuntimeError("Phase B (YOLO) model not loaded.")

    yolo_results = registry.yolo_model(
        img_clahe, verbose=False, imgsz=640, conf=0.25, iou=0.45
    )

    annotated = img_rgb.copy()
    results: list[dict] = []
    gradcam_crops: list[np.ndarray] = []

    H, W = img_rgb.shape[:2]

    if len(yolo_results) == 0 or yolo_results[0].masks is None:
        return annotated, results, gradcam_crops

    r = yolo_results[0]
    masks_raw = r.masks.data.cpu().numpy()          # (N, Hm, Wm) float32 0-1
    boxes     = r.boxes.xyxy.cpu().numpy()           # (N, 4) xyxy in resized space
    cls_ids   = r.boxes.cls.cpu().numpy().astype(int) # (N,) YOLO class 0-31

    # boxes from YOLO are in the resized (640) image space — scale to original
    orig_boxes = r.boxes.xyxy.cpu().numpy().copy()
    if r.orig_shape != img_clahe.shape[:2]:
        sy = H / r.orig_shape[0]
        sx = W / r.orig_shape[1]
        orig_boxes[:, [0, 2]] *= sx
        orig_boxes[:, [1, 3]] *= sy

    _progress(2, "Phase D — Disease classification + GradCAM")

    for i in range(len(cls_ids)):
        fdi_class = cls_ids[i]
        if fdi_class >= len(FDI_NUMBERS):
            continue
        fdi_num = FDI_NUMBERS[fdi_class]

        # Resize mask to original image size
        mask_raw = masks_raw[i]
        mask = cv2.resize(mask_raw, (W, H), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.uint8)

        x1, y1, x2, y2 = orig_boxes[i]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))

        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        # ── Phase D: disease classification ──────────────────────────────────
        disease = "Healthy"
        confidence = 1.0
        cam_crop = None

        if registry.cls_model is not None:
            patch = img_clahe[y1:y2, x1:x2]
            patch_clahe = apply_clahe(patch)
            tensor = _to_tensor_cls(patch_clahe).to(registry.device)

            with torch.no_grad():
                logits = registry.cls_model(tensor)
                probs  = torch.softmax(logits, dim=1)[0].cpu()

            max_prob, pred_cls = probs.max(0)
            max_prob = float(max_prob)
            pred_cls = int(pred_cls)

            if max_prob >= DISEASE_THRESHOLD:
                disease    = DISEASE_NAMES[pred_cls]
                confidence = max_prob
            else:
                disease    = "Healthy"
                confidence = 1.0 - max_prob

            # ── GradCAM ───────────────────────────────────────────────────────
            if registry.gradcam is not None and disease != "Healthy":
                try:
                    tensor_grad = _to_tensor_cls(patch_clahe).to(registry.device)
                    cam_np, _ = registry.gradcam(tensor_grad, cls=pred_cls)
                    # cam_np is (380, 380); resize to patch size
                    cam_resized = cv2.resize(cam_np, (x2 - x1, y2 - y1))
                    cam_crop = blend_gradcam(patch, cam_resized, alpha=0.45)
                except Exception as e:
                    warnings.warn(f"GradCAM failed for tooth {fdi_num}: {e}")

        # Draw overlay on annotated image
        annotated = draw_tooth_overlay(
            annotated, mask, fdi_num, disease, (x1, y1, x2, y2)
        )

        results.append({
            "fdi":        fdi_num,
            "disease":    disease,
            "confidence": confidence,
            "bbox":       (x1, y1, x2, y2),
        })

        if cam_crop is not None:
            gradcam_crops.append(cam_crop)
        elif disease != "Healthy":
            # Store plain tooth crop so the committee can still see it
            gradcam_crops.append(patch.copy())

    # Sort: findings first, then healthy teeth
    results.sort(key=lambda d: (d["disease"] == "Healthy", d["fdi"]))

    _progress(3, "Done")
    return annotated, results, gradcam_crops
