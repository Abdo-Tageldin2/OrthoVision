"""
ModelRegistry — loads and caches all OrthoAI checkpoints at startup.

Architecture classes are ported from orthodontic_ai_framework_v23.ipynb.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tvm

# ─── Constants ───────────────────────────────────────────────────────────────

N_CLS = 3          # disease classes: Impacted, Caries+DeepCaries, Periapical
N_LM  = 29         # cephalometric landmarks
N_CVM = 3          # CVM stages: Early, Peak, Late
N_FDI = 32         # YOLO tooth classes (FDI 11-48)

DISEASE_NAMES  = ["Impacted", "Caries+DeepCaries", "Periapical"]
FDI_NUMBERS    = [
    11, 12, 13, 14, 15, 16, 17, 18,
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
]

# ─── CBAM Attention ──────────────────────────────────────────────────────────

class CBAMBlock(nn.Module):
    """Channel + Spatial Attention (Woo et al. 2018)."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ch_avg = nn.AdaptiveAvgPool2d(1)
        self.ch_max = nn.AdaptiveMaxPool2d(1)
        self.ch_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(1, channels // reduction), bias=False),
            nn.ReLU(),
            nn.Linear(max(1, channels // reduction), channels, bias=False),
        )
        self.sp_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = torch.sigmoid(
            self.ch_mlp(self.ch_avg(x)) + self.ch_mlp(self.ch_max(x))
        ).view(x.size(0), x.size(1), 1, 1)
        x = x * ca
        sp = torch.sigmoid(self.sp_conv(
            torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True).values], dim=1)
        ))
        return x * sp


# ─── Disease Classifier (Phase D) ────────────────────────────────────────────

class DiseaseClassifierV6(nn.Module):
    """EfficientNetV2-M + CBAM attention + optional SSL features."""

    def __init__(self, n_cls: int = N_CLS, ssl_path: str | None = None):
        super().__init__()
        from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
        _bb = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        self.eff = _bb
        in_f = self.eff.classifier[1].in_features  # 1280
        self.cbam = CBAMBlock(channels=1280)

        # Optional SSL ResNet-50 branch
        self.use_ssl = ssl_path is not None and Path(ssl_path).exists()
        ssl_dim = 0
        if self.use_ssl:
            bb = tvm.resnet50(weights=None)
            sd = torch.load(ssl_path, map_location="cpu", weights_only=True)
            if next(iter(sd), "").split(".")[0].isdigit():
                _m = {"0": "conv1", "1": "bn1", "4": "layer1",
                      "5": "layer2", "6": "layer3", "7": "layer4"}
                sd = {".".join([_m.get(k.split(".")[0], k.split(".")[0])]
                               + k.split(".")[1:]): v
                      for k, v in sd.items() if k.split(".")[0] in _m}
            bb.load_state_dict(sd, strict=False)
            self.ssl_feat = nn.Sequential(
                bb.conv1, bb.bn1, bb.relu, bb.maxpool,
                bb.layer1, bb.layer2, bb.layer3,
                nn.AdaptiveAvgPool2d(1), nn.Flatten()
            )
            for p in self.ssl_feat.parameters():
                p.requires_grad = False
            ssl_dim = 1024

        self.eff.classifier = nn.Identity()
        self.clf = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f + ssl_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.eff.features(x)
        feat = self.cbam(feat)
        f = self.eff.avgpool(feat).flatten(1)
        if self.use_ssl:
            xs = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
            f = torch.cat([f, self.ssl_feat(xs)], 1)
        return self.clf(f)


# ─── Cephalometric Landmark Net (Phase G1) ────────────────────────────────────

class CephLandmarkNet(nn.Module):
    """ResNet-50 + U-Net skip decoder → 128×128 heatmaps.
    Input: (B, 3, 512, 512) → Output: (B, N_LM, 128, 128).
    """

    def __init__(self, n_lm: int = N_LM):
        super().__init__()
        resnet = tvm.resnet50(weights="IMAGENET1K_V2")
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 128×128, 256 ch
        self.layer2 = resnet.layer2   # 64×64,   512 ch
        self.layer3 = resnet.layer3   # 32×32,   1024 ch
        self.layer4 = resnet.layer4   # 16×16,   2048 ch
        self.bridge = nn.Sequential(
            nn.Conv2d(2048, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True))
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512 + 1024, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256 + 512, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128 + 256, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.head = nn.Conv2d(64, n_lm, 1)
        for p in self.stem.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s  = self.stem(x)
        e1 = self.layer1(s)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        d  = self.bridge(e4)
        d  = self.up3(torch.cat([F.interpolate(d, e3.shape[2:], mode="bilinear", align_corners=False), e3], 1))
        d  = self.up2(torch.cat([F.interpolate(d, e2.shape[2:], mode="bilinear", align_corners=False), e2], 1))
        d  = self.up1(torch.cat([F.interpolate(d, e1.shape[2:], mode="bilinear", align_corners=False), e1], 1))
        return self.head(d)   # (B, N_LM, 128, 128)


# ─── GradCAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """GradCAM for EfficientNetV2-M disease classifier.

    Usage:
        cam_fn = GradCAM(cls_model, cls_model.eff.features[-1])
        cam, pred_cls = cam_fn(x_tensor)  # x: (1, 3, H, W)
    """

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self._f: torch.Tensor | None = None
        self._g: torch.Tensor | None = None
        self._hooks = [
            layer.register_forward_hook(lambda m, i, o: setattr(self, "_f", o)),
            layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "_g", go[0])),
        ]

    def __call__(self, x: torch.Tensor, cls: int | None = None):
        self.model.eval()
        x = x.detach().requires_grad_(True)
        out = self.model(x)
        if cls is None:
            cls = int(out.argmax(1).item())
        self.model.zero_grad()
        out[0, cls].backward()
        w = self._g.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._f).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam[0, 0].detach().cpu().numpy(), cls

    def remove(self):
        for h in self._hooks:
            h.remove()


# ─── ModelRegistry ────────────────────────────────────────────────────────────

class ModelRegistry:
    """Loads and caches all checkpoints once at server startup."""

    def __init__(self, mdl_dir: str | Path, ssl_path: str | Path | None = None,
                 device: torch.device | None = None):
        self.mdl_dir = Path(mdl_dir)
        self.device  = device or torch.device("cpu")
        self.ssl_path = str(ssl_path) if ssl_path and Path(ssl_path).exists() else None

        # Phase A — binary seg (optional, may be None on MPS)
        self.seg_model = None
        # Phase B — YOLOv8x-seg instance segmentation
        self.yolo_model = None
        # Phase D — disease classifier
        self.cls_model: DiseaseClassifierV6 | None = None
        # GradCAM attached to cls_model
        self.gradcam: GradCAM | None = None
        # Phase G1 — cephalometric landmarks
        self.ceph_model: CephLandmarkNet | None = None
        # Phase G2 — CVM staging
        self.cvm_model: nn.Module | None = None

    def load_all(self, skip_phase_a: bool = False):
        """Load all available checkpoints. Missing checkpoints produce a warning."""
        if not skip_phase_a:
            self._load_phase_a()
        self._load_phase_b()
        self._load_phase_d()
        self._load_phase_g1()
        self._load_phase_g2()
        print(f"[ModelRegistry] Models loaded on {self.device}")

    # ── Phase A ──────────────────────────────────────────────────────────────

    def _load_phase_a(self):
        """Phase A: DINOv2 ViT-B/14 + FPN binary segmentation.
        Skipped in the demo — Phase B (YOLO) provides tooth masks and is faster.
        DINOv2 is 8-20s on MPS and only adds an underlay; the committee sees Phase B output.
        """
        print("[ModelRegistry] Phase A: skipped — Phase B masks used for demo")
        self.seg_model = None

    # ── Phase B ──────────────────────────────────────────────────────────────

    def _load_phase_b(self):
        ckpt = self.mdl_dir / "seg_instance_v23_best.pt"
        if not ckpt.exists():
            print(f"[ModelRegistry] Phase B checkpoint not found: {ckpt}")
            return
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(str(ckpt))
            print(f"[ModelRegistry] Phase B (YOLO) loaded: {ckpt.name}")
        except Exception as e:
            print(f"[ModelRegistry] Phase B load failed: {e}")

    # ── Phase D ──────────────────────────────────────────────────────────────

    def _load_phase_d(self):
        ckpt = self.mdl_dir / "cls_v23_best.pth"
        if not ckpt.exists():
            print(f"[ModelRegistry] Phase D checkpoint not found: {ckpt}")
            return
        try:
            model = DiseaseClassifierV6(n_cls=N_CLS, ssl_path=self.ssl_path)
            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True))
            model = model.to(self.device).eval()
            self.cls_model = model
            self.gradcam = GradCAM(model, model.eff.features[-1])
            print(f"[ModelRegistry] Phase D loaded: {ckpt.name} | GradCAM ready")
        except Exception as e:
            print(f"[ModelRegistry] Phase D load failed: {e}")

    # ── Phase G1 ─────────────────────────────────────────────────────────────

    def _load_phase_g1(self):
        ckpt = self.mdl_dir / "ceph_landmark_v23_best.pth"
        if not ckpt.exists():
            print(f"[ModelRegistry] Phase G1 checkpoint not found: {ckpt} — Tab 2 unavailable")
            return
        try:
            model = CephLandmarkNet(n_lm=N_LM)
            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True))
            self.ceph_model = model.to(self.device).eval()
            print(f"[ModelRegistry] Phase G1 loaded: {ckpt.name}")
        except Exception as e:
            print(f"[ModelRegistry] Phase G1 load failed: {e}")

    # ── Phase G2 ─────────────────────────────────────────────────────────────

    def _load_phase_g2(self):
        ckpt = self.mdl_dir / "cvm_v23_best.pth"
        if not ckpt.exists():
            print(f"[ModelRegistry] Phase G2 checkpoint not found: {ckpt} — CVM unavailable")
            return
        try:
            import timm
            model = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=N_CVM)
            model.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True))
            self.cvm_model = model.to(self.device).eval()
            print(f"[ModelRegistry] Phase G2 loaded: {ckpt.name}")
        except Exception as e:
            print(f"[ModelRegistry] Phase G2 load failed: {e}")
