"""Rendering helpers for OrthoAI demo overlays."""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# Disease color palette (RGB)
DISEASE_COLORS = {
    "Healthy":           (76,  175,  80),   # green
    "Caries+DeepCaries": (255, 152,   0),   # orange
    "Periapical":        (244,  67,  54),   # red
    "Impacted":          (156,  39, 176),   # purple
}

DISEASE_ALPHA = 120   # overlay transparency (0-255)

# CVM stage labels and interpretations
CVM_STAGES = [
    ("S1/S2 — Pre-peak",   "Early growth phase — not yet optimal for intervention."),
    ("S3/S4 — Peak",       "Active growth phase — optimal timing for orthodontic intervention."),
    ("S5/S6 — Post-peak",  "Growth nearing completion — intervention window closing."),
]


def apply_clahe(img_np: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement. Input: HxWx3 uint8 RGB. Output: same shape RGB."""
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def draw_tooth_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    fdi: int,
    disease: str,
    bbox: tuple,
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Draw a semi-transparent colored overlay for one tooth on image.

    Args:
        image: HxWx3 uint8 RGB (modified in-place copy returned)
        mask: HxW bool or uint8 — tooth instance mask
        fdi: FDI tooth number (e.g., 11, 21, 36)
        disease: disease label key in DISEASE_COLORS
        bbox: (x1, y1, x2, y2) bounding box in image coords
        font_scale: OpenCV font scale for FDI label

    Returns:
        Modified image (HxWx3 RGB)
    """
    color = DISEASE_COLORS.get(disease, DISEASE_COLORS["Healthy"])
    overlay = image.copy()

    # Fill mask region with disease color
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color

    # Blend overlay with original
    image = cv2.addWeighted(overlay, DISEASE_ALPHA / 255.0, image, 1 - DISEASE_ALPHA / 255.0, 0)

    # Draw bbox outline
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Label FDI number inside bbox
    label = str(fdi)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    tx = max(x1, min(x1 + 2, x2 - tw - 2))
    ty = max(y1 + th + 2, y1 + 14)
    cv2.rectangle(image, (tx - 1, ty - th - 2), (tx + tw + 1, ty + 2), color, -1)
    cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)

    return image


def draw_landmarks(image: np.ndarray, coords: np.ndarray, radius: int = 4) -> np.ndarray:
    """
    Draw 29 cephalometric landmark dots on image.

    Args:
        image: HxWx3 uint8 RGB
        coords: (N, 2) array of (x, y) pixel coordinates
        radius: dot radius in pixels

    Returns:
        Modified image copy
    """
    out = image.copy()
    for i, (x, y) in enumerate(coords):
        cx, cy = int(round(x)), int(round(y))
        if cx < 0 or cy < 0:
            continue
        # Dot with white outline for visibility
        cv2.circle(out, (cx, cy), radius + 1, (255, 255, 255), -1)
        cv2.circle(out, (cx, cy), radius, (220, 50, 50), -1)
        # Landmark index (1-indexed)
        label = str(i + 1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.putText(out, label, (cx + radius + 1, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 220, 50), 1, cv2.LINE_AA)
    return out


def blend_gradcam(tooth_crop: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay GradCAM heatmap on a tooth crop.

    Args:
        tooth_crop: HxWx3 uint8 RGB
        cam: HxW float32 in [0, 1] (already resized to match crop)
        alpha: heatmap opacity

    Returns:
        Blended HxWx3 RGB image
    """
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(heatmap, alpha, tooth_crop, 1 - alpha, 0)


def build_cvm_card(stage_idx: int, stage_probs: list) -> np.ndarray:
    """
    Build a CVM stage result card as an RGB image (400x180).

    Args:
        stage_idx: 0=Early, 1=Peak, 2=Late
        stage_probs: list of 3 floats (softmax probabilities)

    Returns:
        400x180 RGB numpy array
    """
    W, H = 480, 200
    img = np.full((H, W, 3), 18, dtype=np.uint8)  # dark background

    label, interpretation = CVM_STAGES[stage_idx]
    conf = stage_probs[stage_idx]

    stage_colors = [(100, 180, 255), (80, 220, 100), (255, 160, 60)]
    color = stage_colors[stage_idx]

    # Stage label (large)
    cv2.putText(img, "CVM Stage:", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, label, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                color, 2, cv2.LINE_AA)

    # Confidence
    conf_str = f"Confidence: {conf*100:.1f}%"
    cv2.putText(img, conf_str, (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (160, 160, 160), 1, cv2.LINE_AA)

    # Interpretation (word-wrap at ~55 chars)
    words = interpretation.split()
    lines, line = [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > 55:
            lines.append(" ".join(line)); line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    for j, ln in enumerate(lines):
        cv2.putText(img, ln, (16, 128 + j * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (210, 210, 210), 1, cv2.LINE_AA)

    # 3-stage reference bar
    bar_x, bar_y, bar_w, bar_h = 16, H - 36, (W - 32) // 3, 20
    stage_names = ["Early", "Peak", "Late"]
    for si in range(3):
        bx = bar_x + si * bar_w
        bc = stage_colors[si] if si == stage_idx else (60, 60, 60)
        cv2.rectangle(img, (bx, bar_y), (bx + bar_w - 4, bar_y + bar_h), bc, -1)
        cv2.putText(img, stage_names[si], (bx + 6, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        if si == stage_idx:
            cv2.rectangle(img, (bx, bar_y), (bx + bar_w - 4, bar_y + bar_h), (255, 255, 255), 2)

    return img
