"""
step5c_gradcam.py
=================
Grad-CAM interpretability for the FCD detection ensemble, with QUANTITATIVE
validation against the ds004199 ground-truth lesion masks (ROIs).

For each FCD subject we generate Grad-CAM heatmaps on the model and measure how
well the heatmap localises to the true lesion using three metrics:

  pointing game  - does the Grad-CAM peak fall inside the lesion mask?
  IoU            - overlap between the thresholded CAM and the mask
  energy ratio   - fraction of total CAM activation that lands inside the mask

Leakage-consistent: each subject is explained using the fold checkpoint in which
that subject was in the VALIDATION set (its OOF model), mirroring the project's
out-of-fold philosophy - the explaining model never trained on the subject.

CRITICAL pipeline-matching notes (do not change without checking step2/step4):
  * Model = torchvision convnext_tiny, classifier[2] -> Linear(.,2). 2-logit
    softmax head; FCD is class index 1. Grad-CAM target = model.features[-1].
  * Input transform replicates step4 VAL_TFM exactly (ImageNet normalise).
  * Grad-CAM REQUIRES gradients - it is NOT wrapped in torch.no_grad(), and it
    is run in fp32 (no AMP autocast), unlike step6 inference.
  * Masks are aligned by the z-index encoded in each PNG filename
    (_slice{idx:04d}.png), pushed through the SAME RAS+/transpose geometry as
    step2 and resized to 224 with NEAREST-neighbour (never bilinear on a mask).

TRANSFORMER SUPPORT
  Swin-T emits token tensors, not convolutional feature maps, so pytorch_grad_cam
  needs a reshape_transform to recover a (B, C, H, W) layout. Grad-CAM on windowed
  attention is not the identical operation it is on a conv feature map; the
  reshape approach is used here so the comparison with the CNNs stays like-for-
  like, and this limitation should be stated when the results are reported.

Usage:
    pip install grad-cam nibabel opencv-python-headless timm
    python step5c_gradcam.py --model convnext_tiny
    python step5c_gradcam.py --model swin_tiny --max-subjects 0
    python step5c_gradcam.py --model swin_tiny --method occlusion --max-subjects 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
from PIL import Image
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMG_SIZE = 224
FCD_CLASS = 1                      # label_int: FCD=1, Control=0
TOPK_VIS = 1                       # lesion slices to visualise per subject
CAM_THRESH_FRAC = 0.5             # IoU threshold = frac * cam.max()
TOPK_FRAC = 0.05                  # top 5% hottest brain pixels for top-k hit
OCC_PATCH = 24                    # occlusion patch size (px)
OCC_STRIDE = 12                   # occlusion stride (px)
OCC_BATCH = 16                    # occluded variants per forward batch (2GB-safe)
NCC_MIN = 0.30                    # min FLAIR match to trust an orientation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NORMALIZE = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)


# --------------------------------------------------------------------------- #
# Model registry  (builder + Grad-CAM target layer + checkpoint folder)
# Torchvision models use features[-1] as the last spatial block. Swin-T needs a
# transformer block's norm plus a reshape (see swin_reshape below).
# --------------------------------------------------------------------------- #
def _build_convnext_tiny() -> nn.Module:
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    return m


def _build_efficientnet_b0() -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def _build_efficientnet_v2s() -> nn.Module:
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def _build_xception() -> nn.Module:
    import timm
    return timm.create_model("xception", pretrained=False, num_classes=2)


def _build_resnet50d() -> nn.Module:
    import timm
    return timm.create_model("resnet50d", pretrained=False, num_classes=2)


def _build_swin_tiny() -> nn.Module:
    import timm
    return timm.create_model("swin_tiny_patch4_window7_224",
                             pretrained=False, num_classes=2)


def _xception_target(m):
    return dict(m.named_modules())["act4"]      # final spatial activation


def _swin_target(m):
    """Last transformer block's first norm - the standard Swin Grad-CAM target."""
    layers = getattr(m, "layers", None)
    if layers is None:
        raise ValueError("Unexpected Swin structure: no .layers attribute. "
                         "Check the installed timm version.")
    return layers[-1].blocks[-1].norm1


def swin_reshape(tensor):
    """
    Swin activations are (B, H, W, C) in timm >= 0.9 and (B, L, C) in older
    versions; pytorch_grad_cam expects (B, C, H, W). Both are handled.
    """
    if tensor.dim() == 4:                       # (B, H, W, C)
        return tensor.permute(0, 3, 1, 2)
    if tensor.dim() == 3:                       # (B, L, C)
        B, L, C = tensor.shape
        h = int(round(L ** 0.5))
        return tensor.reshape(B, h, L // h, C).permute(0, 3, 1, 2)
    return tensor


# Models whose Grad-CAM target needs a token -> feature-map reshape.
RESHAPE_MODELS = {"swin_tiny": swin_reshape}


MODEL_REGISTRY = {
    # model_name: (builder, target_layer_fn, default_checkpoint_folder)
    "convnext_tiny":    (_build_convnext_tiny,    lambda m: m.features[-1], "convnext_tiny"),
    "efficientnet":     (_build_efficientnet_b0,  lambda m: m.features[-1], "efficientnet"),
    "efficientnet_b0":  (_build_efficientnet_b0,  lambda m: m.features[-1], "efficientnet"),
    "efficientnet_v2s": (_build_efficientnet_v2s, lambda m: m.features[-1], "efficientnet_v2s"),
    "xception":         (_build_xception,         _xception_target,         "xception"),
    "resnet50v2":       (_build_resnet50d,        lambda m: m.layer4[-1],   "resnet50v2"),
    "swin_tiny":        (_build_swin_tiny,        _swin_target,             "swin_tiny"),
}


# --------------------------------------------------------------------------- #
# Project root
# --------------------------------------------------------------------------- #
def find_project_root(explicit):
    if explicit:
        p = Path(explicit).resolve()
        if (p / "models").is_dir():
            return p
        raise FileNotFoundError(f"--root {p} has no models/ dir")
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        for d in [start, *start.parents]:
            if (d / "models").is_dir() and (d / "data").is_dir():
                return d
    raise FileNotFoundError("Pass --root path/to/fcd_project")


# --------------------------------------------------------------------------- #
# Step-2 geometry, replicated so masks align with the saved 224x224 PNGs
# --------------------------------------------------------------------------- #
def axial_transpose(arr: np.ndarray) -> np.ndarray:
    """Identical axis-fix logic to step2.process_subject."""
    if arr.shape[1] < arr.shape[2] and arr.shape[1] < arr.shape[0]:
        return np.transpose(arr, (0, 2, 1))
    if arr.shape[0] < arr.shape[2] and arr.shape[0] < arr.shape[1]:
        return np.transpose(arr, (1, 2, 0))
    return arr


def slice_index_from_png(png_name: str) -> int:
    m = re.search(r"slice(\d+)\.png$", png_name)
    return int(m.group(1)) if m else -1


# --------------------------------------------------------------------------- #
# FLAIR-anchored orientation resolver
# Step 2 saves slices in RAW FLAIR orientation when HD-BET runs (HD-BET reads the
# raw file) and canonical+transposed otherwise, so the correct mask transform is
# per-subject. We find the frame in which the FLAIR reproduces that subject's
# saved PNGs (image cross-correlation - unambiguous, since the FLAIR is
# information-rich), then apply that same transform to the mask.
# --------------------------------------------------------------------------- #
def _first3(a):
    if a.ndim == 4:
        a = a[..., 0]
    return np.squeeze(a)


def _dihedral(s, k):
    return [s, np.rot90(s, 1), np.rot90(s, 2), np.rot90(s, 3),
            np.fliplr(s), np.flipud(s), s.T, np.fliplr(np.rot90(s, 1))][k]


def _to224(s, nearest=False):
    im = Image.fromarray(s.astype(np.float32))
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST if nearest else Image.BILINEAR)
    return np.array(im)


def _pct01(s):
    nz = s[s > 0]
    if nz.size == 0:
        return s * 0
    lo, hi = np.percentile(nz, 1), np.percentile(nz, 99)
    if hi <= lo:
        return s * 0
    return np.clip((s - lo) / (hi - lo), 0, 1)


def _masked_ncc(a, b, m):
    a = a[m > 0]; b = b[m > 0]
    if a.size < 10:
        return -1.0
    a = a - a.mean(); b = b - b.mean()
    da = np.sqrt((a * a).sum()); db = np.sqrt((b * b).sum())
    if da < 1e-6 or db < 1e-6:
        return 0.0
    return float((a * b).sum() / (da * db))


def candidate_frames(flair_full: str, mask_full: str) -> dict:
    """4 candidate frames; the SAME transform is applied to FLAIR and mask."""
    f_raw = _first3(nib.load(str(flair_full)).get_fdata())
    m_raw = _first3(nib.load(str(mask_full)).get_fdata())
    f_can = _first3(nib.as_closest_canonical(nib.load(str(flair_full))).get_fdata())
    m_can = _first3(nib.as_closest_canonical(nib.load(str(mask_full))).get_fdata())
    return {
        "raw":         (f_raw, m_raw),
        "raw_axial":   (axial_transpose(f_raw), axial_transpose(m_raw)),
        "canon":       (f_can, m_can),
        "canon_axial": (axial_transpose(f_can), axial_transpose(m_can)),
    }


def resolve_orientation(cands: dict, saved: list):
    """Pick (frame, dihedral, ncc) where FLAIR best reproduces the saved PNGs."""
    best = (-2.0, None)
    for name, (fv, mv) in cands.items():
        for d in range(8):
            sc, n = 0.0, 0
            for z, png in saved:
                if not (0 <= z < fv.shape[2]):
                    continue
                p = np.array(Image.open(png)).astype(np.float32) / 65535.0
                fsl = _to224(_pct01(_dihedral(fv[:, :, z].T, d)))
                v = _masked_ncc(fsl, p, (p > 0))
                if v > -1:
                    sc += v; n += 1
            if n >= max(2, len(saved) // 2):
                avg = sc / n
                if avg > best[0]:
                    best = (avg, (name, d))
    score, choice = best
    return score, choice


def mask_slice_resolved(mask_vol: np.ndarray, z_idx: int, d: int) -> np.ndarray:
    """One mask slice in the resolved frame: [:,:,z].T -> dihedral -> 224 NEAREST."""
    if z_idx < 0 or z_idx >= mask_vol.shape[2]:
        return np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    sl = _dihedral(mask_vol[:, :, z_idx].T, d)
    return (_to224(sl.astype(np.float32), nearest=True) > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Image loading (replicates step4 VAL_TFM and the rgb image for overlay)
# --------------------------------------------------------------------------- #
def load_png_for_model(png_path: str):
    """Returns (input_tensor[1,3,224,224], rgb_float[224,224,3] in [0,1])."""
    arr = np.array(Image.open(png_path)).astype(np.float32) / 65535.0
    arr = np.clip(arr, 0, 1)
    uint8 = (arr * 255).astype(np.uint8)
    pil = Image.fromarray(uint8).convert("RGB")
    # step4 VAL_TFM: Resize(224) -> CenterCrop(224); PNGs are already 224x224
    pil = transforms.CenterCrop(IMG_SIZE)(transforms.Resize(IMG_SIZE)(pil))
    rgb = np.asarray(pil).astype(np.float32) / 255.0
    tensor = NORMALIZE(transforms.ToTensor()(pil)).unsqueeze(0)
    return tensor, rgb


# --------------------------------------------------------------------------- #
# Quantitative metrics on a single (cam, mask) pair, both 224x224
# --------------------------------------------------------------------------- #
def pointing_hit(cam: np.ndarray, mask: np.ndarray) -> bool:
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return bool(mask[peak] > 0)


def cam_iou(cam: np.ndarray, mask: np.ndarray, frac: float = CAM_THRESH_FRAC) -> float:
    if cam.max() <= 0:
        return 0.0
    cam_bin = cam >= (frac * cam.max())
    inter = np.logical_and(cam_bin, mask > 0).sum()
    union = np.logical_or(cam_bin, mask > 0).sum()
    return float(inter / union) if union else 0.0


def energy_ratio(cam: np.ndarray, mask: np.ndarray) -> float:
    tot = cam.sum()
    return float((cam * (mask > 0)).sum() / tot) if tot > 0 else 0.0


def concentration_lift(cam: np.ndarray, mask: np.ndarray, brain: np.ndarray):
    """Mean CAM density inside the lesion / mean density over the brain.
    >1 = attends to the lesion ABOVE what its size alone would give (chance)."""
    b = brain > 0
    m = (mask > 0) & b
    if m.sum() == 0 or b.sum() == 0:
        return None
    denom = cam[b].mean()
    return float(cam[m].mean() / denom) if denom > 0 else None


def topk_hit(cam: np.ndarray, mask: np.ndarray, brain: np.ndarray,
             frac: float = TOPK_FRAC) -> bool:
    """Does any of the hottest `frac` of brain pixels fall in the lesion?"""
    b = brain > 0
    vals = cam[b]
    if vals.size == 0:
        return False
    thr = np.quantile(vals, 1.0 - frac)
    hot = (cam >= thr) & b
    return bool((hot & (mask > 0)).any())


def occlusion_map(model, tensor, device, patch=OCC_PATCH, stride=OCC_STRIDE,
                  fcd_class=FCD_CLASS, batch=OCC_BATCH) -> np.ndarray:
    """Perturbation saliency: slide a patch, measure the drop in FCD probability.
    Model-agnostic (no target layer). Returns a [0,1] map, forward-only."""
    model.eval()
    tensor = tensor.to(device)
    with torch.no_grad():
        base = torch.softmax(model(tensor), dim=1)[0, fcd_class].item()
    _, _, H, W = tensor.shape
    ys = list(range(0, max(1, H - patch + 1), stride))
    xs = list(range(0, max(1, W - patch + 1), stride))
    if ys[-1] != H - patch: ys.append(H - patch)
    if xs[-1] != W - patch: xs.append(W - patch)
    positions = [(y, x) for y in ys for x in xs]

    heat = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    for i in range(0, len(positions), batch):
        chunk = positions[i:i + batch]
        bt = tensor.repeat(len(chunk), 1, 1, 1).clone()
        for j, (y, x) in enumerate(chunk):
            bt[j, :, y:y + patch, x:x + patch] = 0.0   # 0 = ImageNet mean (grey)
        with torch.no_grad():
            probs = torch.softmax(model(bt), dim=1)[:, fcd_class].cpu().numpy()
        for j, (y, x) in enumerate(chunk):
            drop = base - float(probs[j])              # hiding region lowers FCD?
            heat[y:y + patch, x:x + patch] += drop
            cnt[y:y + patch, x:x + patch] += 1
    cnt[cnt == 0] = 1
    heat /= cnt
    heat = np.maximum(heat, 0.0)                        # keep FCD-supporting regions
    if heat.max() > 0:
        heat /= heat.max()
    return heat


# --------------------------------------------------------------------------- #
# Checkpoint -> fold lookup
# --------------------------------------------------------------------------- #
def subject_val_fold(subj_id: str, folds: list) -> int:
    for i, fd in enumerate(folds):
        if subj_id in fd.get("val", []):
            return i + 1                    # checkpoints are 1-indexed: fold{k}_best.pt
    return -1


def load_model(builder, ckpt_path: Path, device) -> nn.Module:
    model = builder()
    # weights_only=False: checkpoints written by this project may carry plain
    # metadata (fold index, best epoch, AUC) beside the weights. torch 2.6
    # flipped this default to True, which rejects numpy scalars.
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except TypeError:                       # torch < 2.4 has no such argument
        ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(device)
    return model


# --------------------------------------------------------------------------- #
# Mask discovery
# --------------------------------------------------------------------------- #
_debug_listed = False   # print one anat listing on the first miss, to aid setup

# real image modalities to exclude (only exact suffixes, not substrings)
_MODALITY_SUFFIXES = ("_flair.nii.gz", "_flair.nii", "_t1w.nii.gz", "_t1w.nii",
                      "_t2w.nii.gz", "_t2w.nii", "_t2.nii.gz", "_t2.nii")
_MASK_KEYS = ("roi", "lesion", "label", "mask", "seg")


def find_mask(flair_path: str, suffix_hint: str | None) -> Path | None:
    """
    Find the lesion ROI next to the FLAIR. A candidate is any .nii/.nii.gz in the
    anat folder that is NOT itself a modality image (FLAIR/T1w/T2w) and whose name
    contains an ROI keyword (or the user-supplied --mask-suffix). Matching is
    case-insensitive and only excludes the actual modality images by exact suffix,
    so an ROI named '..._acq-tse3dvfl_roi.nii.gz' is correctly kept.
    """
    global _debug_listed
    d = Path(flair_path).parent
    key = suffix_hint.lower() if suffix_hint else None

    cands = []
    for f in sorted(d.glob("*.nii*")):
        n = f.name.lower()
        if n.endswith(_MODALITY_SUFFIXES):
            continue
        if (key and key in n) or any(k in n for k in _MASK_KEYS):
            cands.append(f)

    if not cands and not _debug_listed:
        listing = [p.name for p in sorted(d.glob("*"))]
        print(f"      [debug] no ROI in {d}")
        print(f"      [debug] anat contents: {listing}")
        _debug_listed = True
    return cands[0] if cands else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--model", default="convnext_tiny", choices=list(MODEL_REGISTRY))
    ap.add_argument("--model-folder", default=None,
                    help="checkpoint folder under models/ (defaults per model)")
    ap.add_argument("--mask-suffix", default=None,
                    help="hint for ROI filename, e.g. 'roi' (auto-detected otherwise)")
    ap.add_argument("--max-subjects", type=int, default=0,
                    help="limit FCD subjects processed (0 = all)")
    ap.add_argument("--method", default="gradcam",
                    choices=["gradcam", "occlusion"],
                    help="saliency method: gradcam (fast) or occlusion (slower, "
                         "model-agnostic, causal)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_project_root(args.root)
    builder, target_fn, default_folder = MODEL_REGISTRY[args.model]
    reshape_fn = RESHAPE_MODELS.get(args.model)      # None for the CNNs
    ckpt_folder = root / "models" / (args.model_folder or default_folder)
    out_dir = root / "results" / args.method / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(root / "data/processed/subject_map.json") as f:
        subject_map = json.load(f)
    with open(root / "data/processed/master_split.json") as f:
        master_split = json.load(f)
    folds = master_split["folds"]

    # FCD trainval subjects only (test set stays locked for step 8)
    trainval = set()
    for fd in folds:
        trainval.update(fd.get("train", []))
        trainval.update(fd.get("val", []))
    fcd_subjects = [s for s in trainval
                    if str(subject_map[s].get("label")).upper() == "FCD"]
    fcd_subjects.sort()
    if args.max_subjects:
        fcd_subjects = fcd_subjects[:args.max_subjects]

    print(f"Project root : {root}")
    print(f"Model        : {args.model}  (checkpoints: {ckpt_folder})")
    print(f"Method       : {args.method}")
    if reshape_fn is not None and args.method == "gradcam":
        print(f"               token reshape applied (transformer backbone)")
    print(f"Device       : {device}")
    print(f"FCD subjects : {len(fcd_subjects)}\n")

    # cache one model per fold (built lazily)
    fold_models: dict[int, nn.Module] = {}

    per_subject = {}
    slice_hits, slice_ious, slice_energies = [], [], []
    slice_lifts, slice_topk = [], []
    subj_level_hits = 0
    subjects_scored = 0
    vis_done = 0
    skipped_lowalign = []

    for subj in fcd_subjects:
        info = subject_map[subj]
        fold = subject_val_fold(subj, folds)
        if fold < 0:
            print(f"  [skip] {subj}: not found in any val fold")
            continue
        ckpt_path = ckpt_folder / f"fold{fold}_best.pt"
        if not ckpt_path.exists():
            print(f"  [skip] {subj}: missing {ckpt_path.name}")
            continue

        # resolve mask<->image orientation by anchoring on the FLAIR
        flair_full = (Path(info["flair_path"]) if Path(info["flair_path"]).is_absolute()
                      else root / info["flair_path"])
        mask_path = find_mask(str(flair_full), args.mask_suffix)
        if mask_path is None:
            print(f"  [skip] {subj}: no lesion mask found near {flair_full}")
            continue

        slice_paths = info.get("slice_paths", [])
        saved = [(slice_index_from_png(Path(p).name),
                  (Path(p) if Path(p).is_absolute() else root / p))
                 for p in slice_paths]
        saved = [(z, p) for z, p in saved if z >= 0 and Path(p).exists()]
        if not saved:
            print(f"  [skip] {subj}: no saved PNGs")
            continue

        cands = candidate_frames(str(flair_full), str(mask_path))
        ncc, choice = resolve_orientation(cands, saved)
        if choice is None or ncc < NCC_MIN:
            print(f"  [skip] {subj}: orientation unresolved (NCC={ncc:.2f} "
                  f"< {NCC_MIN}); excluded to avoid a wrong measurement")
            skipped_lowalign.append({"subject": subj, "ncc": round(float(ncc), 4),
                                     "mask_file": mask_path.name})
            continue
        frame_name, dih = choice
        mask_vol = cands[frame_name][1]

        if fold not in fold_models:
            fold_models[fold] = load_model(builder, ckpt_path, device)
        model = fold_models[fold]
        cam_engine = None
        if args.method == "gradcam":
            # reshape_transform is required for transformer backbones (Swin) and
            # must stay None for the convolutional models.
            cam_engine = GradCAM(model=model, target_layers=[target_fn(model)],
                                 reshape_transform=reshape_fn)

        lesion_records = []      # (z_idx, png, prob, cam, mask2d, hit, iou, energy)

        for z, png in saved:
            m2d = mask_slice_resolved(mask_vol, z, dih)
            if m2d.sum() == 0:
                continue                       # only score lesion-bearing slices

            tensor, rgb = load_png_for_model(str(png))
            tensor = tensor.to(device)
            brain = (rgb.mean(axis=2) > 0).astype(np.uint8)

            with torch.no_grad():
                prob = torch.softmax(model(tensor), dim=1)[0, FCD_CLASS].item()

            if args.method == "gradcam":
                # Grad-CAM: gradients ON, fp32, no autocast
                cam = cam_engine(input_tensor=tensor,
                                 targets=[ClassifierOutputTarget(FCD_CLASS)])[0]
            else:  # occlusion: forward-only, model-agnostic
                cam = occlusion_map(model, tensor, device)

            hit = pointing_hit(cam, m2d)
            iou = cam_iou(cam, m2d)
            en = energy_ratio(cam, m2d)
            lift = concentration_lift(cam, m2d, brain)
            tkh = topk_hit(cam, m2d, brain)
            slice_hits.append(hit); slice_ious.append(iou); slice_energies.append(en)
            slice_topk.append(tkh)
            if lift is not None:
                slice_lifts.append(lift)
            lesion_records.append((z, png, prob, cam, m2d, hit, iou, en, lift, tkh))

        if not lesion_records:
            print(f"  [skip] {subj}: no lesion slices among saved PNGs")
            continue

        subjects_scored += 1
        subj_hit = any(r[5] for r in lesion_records)
        subj_level_hits += int(subj_hit)
        per_subject[subj] = {
            "fold_used": fold,
            "mask_file": mask_path.name,
            "orientation": f"{frame_name}|d{dih}",
            "flair_ncc": round(float(ncc), 4),
            "n_lesion_slices": len(lesion_records),
            "pointing_hit_rate": round(np.mean([r[5] for r in lesion_records]), 4),
            "mean_iou": round(float(np.mean([r[6] for r in lesion_records])), 4),
            "mean_energy_ratio": round(float(np.mean([r[7] for r in lesion_records])), 4),
            "mean_concentration_lift": (
                round(float(np.mean([r[8] for r in lesion_records if r[8] is not None])), 4)
                if any(r[8] is not None for r in lesion_records) else None),
            "topk_hit_rate": round(float(np.mean([r[9] for r in lesion_records])), 4),
            "subject_hit": subj_hit,
        }
        print(f"  {subj}: fold{fold}  {frame_name}|d{dih} NCC={ncc:.2f}  "
              f"lesion_slices={len(lesion_records)}  "
              f"hit_rate={per_subject[subj]['pointing_hit_rate']:.2f}  "
              f"IoU={per_subject[subj]['mean_iou']:.2f}")

        # visualise the most confident lesion slice
        if vis_done < 12:
            rec = max(lesion_records, key=lambda r: r[2])   # highest FCD prob
            _, png, prob, cam, m2d, *_ = rec
            _, rgb = load_png_for_model(png)
            overlay = show_cam_on_image(rgb, cam, use_rgb=True)
            contour, _ = cv2.findContours((m2d * 255).astype(np.uint8),
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay_c = overlay.copy()
            cv2.drawContours(overlay_c, contour, -1, (0, 255, 0), 2)
            method_label = "Grad-CAM" if args.method == "gradcam" else "Occlusion"
            fig, ax = plt.subplots(1, 3, figsize=(11, 4))
            ax[0].imshow(rgb)
            ax[0].set_title(f"{subj} (p={prob:.2f})"); ax[0].axis("off")
            ax[1].imshow(overlay)
            ax[1].set_title(f"{method_label} - {args.model}"); ax[1].axis("off")
            ax[2].imshow(overlay_c)
            ax[2].set_title("Saliency + lesion (green)"); ax[2].axis("off")
            plt.tight_layout()
            plt.savefig(out_dir / f"{subj}_{args.method}.png", dpi=120); plt.close()
            vis_done += 1

    # ---- aggregate ---- #
    summary = {
        "model": args.model,
        "method": args.method,
        "reshape_transform_applied": reshape_fn is not None and args.method == "gradcam",
        "n_fcd_subjects_scored": subjects_scored,
        "n_lesion_slices_scored": len(slice_hits),
        "slice_level_pointing_game": round(float(np.mean(slice_hits)), 4) if slice_hits else None,
        "subject_level_pointing_game": round(subj_level_hits / subjects_scored, 4) if subjects_scored else None,
        "topk_hit_rate": round(float(np.mean(slice_topk)), 4) if slice_topk else None,
        "mean_iou": round(float(np.mean(slice_ious)), 4) if slice_ious else None,
        "mean_energy_ratio": round(float(np.mean(slice_energies)), 4) if slice_energies else None,
        "mean_concentration_lift": round(float(np.mean(slice_lifts)), 4) if slice_lifts else None,
        "cam_threshold_frac": CAM_THRESH_FRAC,
        "topk_frac": TOPK_FRAC,
        "n_skipped_low_alignment": len(skipped_lowalign),
        "skipped_low_alignment": skipped_lowalign,
        "per_subject": per_subject,
    }
    with open(out_dir / f"{args.method}_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{args.method} interpretability - {args.model}")
    print("-" * 60)
    print(f"  FCD subjects scored        : {subjects_scored}")
    print(f"  Lesion slices scored       : {len(slice_hits)}")
    print(f"  Pointing game (slice-level): {summary['slice_level_pointing_game']}")
    print(f"  Pointing game (subj-level) : {summary['subject_level_pointing_game']}")
    print(f"  Top-{int(TOPK_FRAC*100)}% hit rate         : {summary['topk_hit_rate']}")
    print(f"  Mean IoU                   : {summary['mean_iou']}")
    print(f"  Mean energy ratio          : {summary['mean_energy_ratio']}")
    print(f"  Mean concentration lift    : {summary['mean_concentration_lift']}"
          f"   (>1 = above-chance lesion attention)")
    if skipped_lowalign:
        print(f"  Skipped (low FLAIR match) : {len(skipped_lowalign)} "
              f"-> {[s['subject'] for s in skipped_lowalign]}")
    print("=" * 60)
    print(f"Outputs -> {out_dir}")


if __name__ == "__main__":
    main()