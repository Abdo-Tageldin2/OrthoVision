"""
Phase G1 + G2 inference pipeline for cephalometric X-rays.

Pipeline:
    Input image (any size)
    → Letterbox pad to 512×512 (preserve aspect ratio)
    → Phase G1: CephLandmarkNet → 29 (x, y) landmark coordinates
    → Phase G2: ConvNeXt-V2-T → CVM stage (0=Early, 1=Peak, 2=Late)
    → Render: draw landmark dots; build CVM stage card
    → Output: landmark image + CVM stage card
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.visualization import build_cvm_card, draw_landmarks

if TYPE_CHECKING:
    from inference.models import ModelRegistry

# ─── Constants ───────────────────────────────────────────────────────────────

CEPH_SIZE = 512    # letterbox target size

# Aariz 29-landmark ordering (matches training annotations)
LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N'", "Pog'", "Sn",
]

LANDMARK_FULL_NAMES = {
    "A":   "Subspinale",          "ANS": "Anterior nasal spine",
    "B":   "Supramentale",        "Me":  "Menton",
    "N":   "Nasion",              "Or":  "Orbitale",
    "Pog": "Pogonion",            "PNS": "Posterior nasal spine",
    "Pn":  "Pronasale",           "R":   "Registration point",
    "S":   "Sella",               "Ar":  "Articulare",
    "Co":  "Condylion",           "Gn":  "Gnathion",
    "Go":  "Gonion",              "Po":  "Porion",
    "LPM": "Lower premolar",      "LIT": "Lower incisor tip",
    "LMT": "Lower molar",         "UPM": "Upper premolar",
    "UIA": "Upper incisor apex",  "UIT": "Upper incisor tip",
    "UMT": "Upper molar",         "LIA": "Lower incisor apex",
    "Li":  "Labrale inferius",    "Ls":  "Labrale superius",
    "N'":  "Soft tissue nasion",  "Pog'": "Soft tissue pogonion",
    "Sn":  "Subnasale",
}

CVM_LABELS = ["Early", "Peak", "Late"]
CVM_INTERPRETATIONS = [
    "Pre-peak growth phase. Orthodontic intervention may be premature.",
    "Active skeletal growth. Optimal window for functional appliance therapy.",
    "Post-peak phase. Growth nearing completion; intervention window closing.",
]

_CEPH_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_CVM_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _letterbox(img_np: np.ndarray, target: int = CEPH_SIZE):
    """Letterbox-pad image to (target, target) preserving aspect ratio."""
    H, W = img_np.shape[:2]
    sc = min(target / H, target / W)
    nH, nW = int(H * sc), int(W * sc)
    res = cv2.resize(img_np, (nW, nH))
    px, py = (target - nW) // 2, (target - nH) // 2
    if img_np.ndim == 3:
        out = np.zeros((target, target, 3), dtype=img_np.dtype)
    else:
        out = np.zeros((target, target), dtype=img_np.dtype)
    out[py:py + nH, px:px + nW] = res
    return out, sc, px, py


def _soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """Differentiable soft-argmax. Input: (B, N, H, W). Output: (B, N, 2) in [0,1]."""
    B, N, H, W = heatmap.shape
    hm = heatmap.reshape(B, N, -1).softmax(-1).reshape(B, N, H, W)
    grid_y = torch.linspace(0, 1, H, device=heatmap.device)
    grid_x = torch.linspace(0, 1, W, device=heatmap.device)
    coord_y = (hm * grid_y.view(1, 1, H, 1)).sum(dim=(2, 3))
    coord_x = (hm * grid_x.view(1, 1, 1, W)).sum(dim=(2, 3))
    return torch.stack([coord_x, coord_y], dim=-1)   # (B, N, 2)


def _apply_clahe(img_np: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ─── Main inference function ──────────────────────────────────────────────────

def run_cephalometric_structured(
    image_path: str,
    registry: "ModelRegistry",
):
    """Run Phase G1 + G2 and return structured results for JSON transport.

    Returns dict with:
        landmark_image : HxWx3 RGB uint8 with dots drawn
        landmarks      : list of {symbol, name, x, y} dicts (pixel coords, 512-space)
        cvm_stage      : int or None
        cvm_probs      : list[3] or None
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    padded, _, _, _ = _letterbox(img_rgb, CEPH_SIZE)
    padded_clahe = _apply_clahe(padded)

    landmark_image = padded_clahe.copy()
    landmarks: list[dict] = []
    cvm_stage: int | None = None
    cvm_probs: list[float] | None = None

    if registry.ceph_model is not None:
        try:
            tensor = _CEPH_TRANSFORM(padded_clahe).unsqueeze(0).to(registry.device)
            with torch.no_grad():
                hm = registry.ceph_model(tensor)
            coords_norm = _soft_argmax_2d(hm.float())
            coords_px = (coords_norm[0].cpu().numpy() * CEPH_SIZE)
            landmark_image = draw_landmarks(padded_clahe, coords_px)
            for i, (x, y) in enumerate(coords_px):
                sym = LANDMARK_SYMBOLS[i] if i < len(LANDMARK_SYMBOLS) else f"L{i+1:02d}"
                landmarks.append({
                    "symbol": sym,
                    "name":   LANDMARK_FULL_NAMES.get(sym, sym),
                    "x":      round(float(x), 1),
                    "y":      round(float(y), 1),
                })
        except Exception as e:
            warnings.warn(f"Phase G1 inference failed: {e}")

    if registry.cvm_model is not None:
        try:
            cvm_tensor = _CVM_TRANSFORM(padded_clahe).unsqueeze(0).to(registry.device)
            with torch.no_grad():
                logits = registry.cvm_model(cvm_tensor)
            probs_t = torch.softmax(logits, dim=1)[0].cpu().tolist()
            cvm_probs = [round(float(p), 4) for p in probs_t]
            cvm_stage = int(np.argmax(cvm_probs))
        except Exception as e:
            warnings.warn(f"Phase G2 inference failed: {e}")

    return {
        "landmark_image": landmark_image,
        "landmarks":      landmarks,
        "cvm_stage":      cvm_stage,
        "cvm_probs":      cvm_probs,
        "cvm_label":      CVM_LABELS[cvm_stage] if cvm_stage is not None else None,
        "cvm_interpretation":
            CVM_INTERPRETATIONS[cvm_stage] if cvm_stage is not None else None,
    }


def run_cephalometric(
    image_path: str,
    registry: "ModelRegistry",
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Run Phase G1 + G2 on a cephalometric X-ray.

    Args:
        image_path: path to input X-ray image
        registry: loaded ModelRegistry
        progress_cb: optional callable(step, total, desc)

    Returns:
        landmark_image: 512×512 RGB with 29 landmark dots
        cvm_card: 480×200 RGB image with CVM stage result (or None if G2 unavailable)
    """
    def _progress(step, desc):
        if progress_cb:
            progress_cb(step, 2, desc)

    # ── Load image ────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Letterbox pad (never stretch — cephalogram geometry must be preserved)
    padded, sc, px, py = _letterbox(img_rgb, CEPH_SIZE)
    padded_clahe = _apply_clahe(padded)

    _progress(1, "Phase G1 — Landmark detection")

    # ── Phase G1: landmark detection ──────────────────────────────────────────
    landmark_image = padded_clahe.copy()
    cvm_card = None

    if registry.ceph_model is None:
        # Return plain image with no landmarks
        return landmark_image, cvm_card

    tensor = _CEPH_TRANSFORM(padded_clahe).unsqueeze(0).to(registry.device)

    try:
        with torch.no_grad():
            hm = registry.ceph_model(tensor)   # (1, N_LM, 128, 128)
        coords_norm = _soft_argmax_2d(hm.float())          # (1, N_LM, 2) in [0,1]
        coords_px   = (coords_norm[0].cpu().numpy() * CEPH_SIZE)  # (N_LM, 2) pixels
        landmark_image = draw_landmarks(padded_clahe, coords_px)
    except Exception as e:
        warnings.warn(f"Phase G1 inference failed: {e}")
        coords_px = None

    _progress(2, "Phase G2 — CVM bone maturity staging")

    # ── Phase G2: CVM staging ─────────────────────────────────────────────────
    if registry.cvm_model is not None:
        try:
            cvm_tensor = _CVM_TRANSFORM(padded_clahe).unsqueeze(0).to(registry.device)
            with torch.no_grad():
                logits = registry.cvm_model(cvm_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
            stage_idx = int(torch.tensor(probs).argmax())
            cvm_card = build_cvm_card(stage_idx, probs)
        except Exception as e:
            warnings.warn(f"Phase G2 inference failed: {e}")

    return landmark_image, cvm_card
