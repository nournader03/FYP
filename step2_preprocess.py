"""
Step 2: Preprocessing Pipeline for FCD Detection DL Project
============================================================
Processes FLAIR NIfTI volumes from ds004199 into 16-bit PNG slices.

Pipeline (in order):
  1. Load + reorient to RAS+ canonical
  2. Skull stripping (HD-BET, fallback to intensity mask)
  3. Percentile normalization [0, 1]
  4. Evenly spaced real slice selection (N=64, no resampling)
  5. Slice filtering (skip near-empty slices)
  6. Resize to 224x224, save as 16-bit PNG (grayscale stacked to RGB)
  7. Cache + metadata
  8. Visual verification plots

Usage:
    pip install nibabel numpy pillow tqdm matplotlib scipy
    pip install hd-bet   # optional but recommended
    python step2_preprocess.py
"""

import os
import json
import warnings
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- CONFIG --------------------------------------------------------------------
SUBJECT_MAP_PATH = Path("data/processed/subject_map.json")
PNG_DIR          = Path("data/processed/png")
CACHE_DIR        = Path("data/processed/cache")
META_PATH        = Path("data/processed/png_meta.json")
RESULTS_DIR      = Path("results")

N_SLICES         = 64       # evenly spaced real slices per subject
MIN_BRAIN_RATIO  = 0.05     # slice filter: skip if <5% pixels exceed threshold
FILTER_THRESH    = 0.05     # intensity threshold for slice filter (post-norm)
RESIZE           = 224      # output slice size

for d in [PNG_DIR / "FCD", PNG_DIR / "Control", CACHE_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- 1. SKULL STRIPPING --------------------------------------------------------
def skull_strip_hdbet(nifti_path: Path, cache_dir: Path):
    """
    Run HD-BET on a single NIfTI file.
    Returns masked NIfTI image (brain only, skull zeroed).
    HD-BET saves output as <stem>_bet.nii.gz in cache_dir.
    """
    import subprocess
    # HD-BET requires output path to end with .nii.gz
    bare_stem = nifti_path.name.replace(".nii.gz", "").replace(".nii", "")
    out_path  = cache_dir / (bare_stem + "_bet.nii.gz")

    if not out_path.exists():
        cmd = [
            "hd-bet", "-i", str(nifti_path),
            "-o", str(out_path),
            "-device", "cuda:0",    # GPU 0 - 2GB VRAM sufficient for HD-BET
            "--disable_tta",        # no test-time augmentation - faster
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"HD-BET failed: {result.stderr}")

    return nib.load(str(out_path))


def skull_strip_fallback(img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Fast intensity-only skull strip fallback when HD-BET unavailable.
    Avoids expensive 3D connected component labeling - uses per-slice
    hole-filling only, which is fast and sufficient for FLAIR.
    """
    from scipy.ndimage import binary_fill_holes, binary_erosion

    data    = img.get_fdata(dtype=np.float32)
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        return img
    # Threshold at 15th percentile of non-zero voxels
    thresh  = np.percentile(nonzero, 15)
    binary  = data > thresh

    # Per-slice hole fill - fast, avoids 3D connected components entirely
    filled = np.zeros_like(binary)
    for z in range(binary.shape[2]):
        filled[:, :, z] = binary_fill_holes(binary[:, :, z])

    # Single erosion pass to trim skull rim
    brain_mask = binary_erosion(filled, iterations=1)

    masked  = data * brain_mask
    new_img = nib.Nifti1Image(masked, img.affine, img.header)
    return new_img


def apply_skull_strip(img: nib.Nifti1Image, nifti_path: Path,
                      cache_dir: Path, use_hdbet: bool) -> np.ndarray:
    """Returns skull-stripped 3D numpy array."""
    if use_hdbet:
        try:
            stripped = skull_strip_hdbet(nifti_path, cache_dir)
            return stripped.get_fdata(dtype=np.float32)
        except Exception as e:
            print(f"\n    HD-BET failed ({e}), using fallback...")

    stripped = skull_strip_fallback(img)
    return stripped.get_fdata(dtype=np.float32)


# --- 2. CHECK HD-BET AVAILABILITY ----------------------------------------------
def check_hdbet() -> bool:
    import shutil
    available = shutil.which("hd-bet") is not None
    if available:
        print("  [OK] HD-BET found - will use for skull stripping")
    else:
        print("  WARNING: HD-BET not found - using intensity+morphology fallback")
        print("    To install: pip install hd-bet")
    return available


# --- 3. PERCENTILE NORMALIZATION -----------------------------------------------
def percentile_normalize(data: np.ndarray) -> np.ndarray:
    """
    Clip to [1st, 99th] percentile of non-zero voxels only.
    Rescale clipped range to [0, 1].
    More robust than z-score for multi-scanner FLAIR data.
    """
    nonzero = data[data > 0]
    if len(nonzero) == 0:
        return data.astype(np.float32)
    p1  = np.percentile(nonzero, 1)
    p99 = np.percentile(nonzero, 99)
    if p99 <= p1:
        return np.zeros_like(data, dtype=np.float32)
    clipped = np.clip(data, p1, p99)
    normed  = (clipped - p1) / (p99 - p1)
    # Zero out voxels that were background before normalization
    normed[data == 0] = 0.0
    return normed.astype(np.float32)


# --- 4. SLICE SELECTION --------------------------------------------------------
def get_brain_bbox_z(data: np.ndarray) -> tuple:
    """Find Z-axis brain bounding box (first and last slice with brain signal)."""
    brain_mask = data > 0.01   # after normalization
    z_has_brain = brain_mask.any(axis=(0, 1))
    z_indices   = np.where(z_has_brain)[0]
    if len(z_indices) == 0:
        return 0, data.shape[2] - 1
    return int(z_indices[0]), int(z_indices[-1])


def select_slices(data: np.ndarray, n: int = N_SLICES) -> list:
    """
    Extract N evenly spaced REAL slices from within brain bounding box.
    No resampling or interpolation - only real acquired slices.
    If fewer than N slices available, use all of them (no upsampling).
    """
    z_min, z_max = get_brain_bbox_z(data)
    z_range      = z_max - z_min + 1

    if z_range <= n:
        # Thin volume - use all available slices rather than upsampling
        indices = list(range(z_min, z_max + 1))
    else:
        indices = list(np.linspace(z_min, z_max, n, dtype=int))
        indices = sorted(set(indices))   # remove duplicates from linspace rounding

    return indices


# --- 5. SLICE FILTERING --------------------------------------------------------
def is_informative_slice(slice_2d: np.ndarray,
                         min_ratio: float = MIN_BRAIN_RATIO,
                         thresh: float    = FILTER_THRESH) -> bool:
    """
    Returns True if >= min_ratio of pixels exceed thresh intensity.
    Filters out near-empty edge slices after skull stripping.
    """
    above = np.sum(slice_2d > thresh)
    total = slice_2d.size
    return (above / total) >= min_ratio


# --- 6. SAVE 16-BIT PNG --------------------------------------------------------
def save_slice_png(slice_2d: np.ndarray, out_path: Path):
    """
    Resize to 224x224, stack grayscale 3x -> RGB, save as 16-bit PNG.
    16-bit preserves intensity precision lost in 8-bit conversion.
    """
    # Scale [0,1] float -> [0, 65535] uint16
    scaled = (slice_2d * 65535).astype(np.uint16)
    pil_img = Image.fromarray(scaled, mode="I;16")
    # Resize to 224x224 using BILINEAR
    pil_img = pil_img.resize((RESIZE, RESIZE), Image.BILINEAR)
    pil_img.save(str(out_path))


# --- 7. PROCESS ONE SUBJECT ----------------------------------------------------
def process_subject(subj_id: str, info: dict, use_hdbet: bool) -> dict:
    """
    Full pipeline for one subject. Returns metadata dict.
    """
    flair_path = Path(info["flair_path"])
    label      = info["label"]          # "FCD" or "Control"
    out_dir    = PNG_DIR / label / f"subj_{subj_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Load ----------------------------------------------------------------
    img = nib.load(str(flair_path))

    # -- Reorient to RAS+ canonical ------------------------------------------
    img = nib.as_closest_canonical(img)

    # -- Handle 4D volumes ---------------------------------------------------
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    # Squeeze any remaining singleton dims
    data = np.squeeze(data)
    assert data.ndim == 3, f"{subj_id}: expected 3D after squeeze, got {data.shape}"

    # -- Ensure axial axis is axis 2 ------------------------------------------
    # After canonical reorientation, axial (Z) should be the smallest dimension
    # on axis 2. Some acquisitions (e.g. coronal FLAIR CUBE) end up with the
    # smallest dimension on axis 1 -> transpose to (0, 2, 1) to fix.
    # Shape (X, Z, Y) -> (X, Y, Z) so slice selection along axis 2 is axial.
    if data.shape[1] < data.shape[2] and data.shape[1] < data.shape[0]:
        data = np.transpose(data, (0, 2, 1))
        print(f"    {subj_id}: transposed axes (0,2,1) -> axial now on axis 2, shape={data.shape}")
    elif data.shape[0] < data.shape[2] and data.shape[0] < data.shape[1]:
        data = np.transpose(data, (1, 2, 0))
        print(f"    {subj_id}: transposed axes (1,2,0) -> axial now on axis 2, shape={data.shape}")

    # -- Skull strip ---------------------------------------------------------
    # Rebuild temp NIfTI with canonical affine for HD-BET input
    canonical_img = nib.Nifti1Image(data, img.affine, img.header)
    stripped = apply_skull_strip(canonical_img, flair_path, CACHE_DIR, use_hdbet)

    # -- Percentile normalize ------------------------------------------------
    normed = percentile_normalize(stripped)

    # -- Evenly spaced slice selection ---------------------------------------
    slice_indices = select_slices(normed)

    # -- Filter + save slices ------------------------------------------------
    saved_slices  = []
    skipped       = 0
    for idx in slice_indices:
        sl = normed[:, :, idx]
        # Transpose: NIfTI stores (X, Y, Z) - we want (Y, X) for display
        sl = sl.T

        if not is_informative_slice(sl):
            skipped += 1
            continue

        fname   = out_dir / f"{subj_id}_slice{idx:04d}.png"
        save_slice_png(sl, fname)
        saved_slices.append(str(fname))

    meta = {
        "subj_id":       subj_id,
        "label":         label,
        "label_int":     info["label_int"],
        "flair_path":    str(flair_path),
        "original_shape": list(data.shape),
        "n_slices_selected": len(slice_indices),
        "n_slices_saved":    len(saved_slices),
        "n_slices_skipped":  skipped,
        "slice_paths":   saved_slices,
        "hemisphere":    info.get("hemisphere"),
        "lobe":          info.get("lobe"),
    }
    return meta


# --- 8. VISUAL VERIFICATION ----------------------------------------------------
def save_preview_grid(all_meta: dict, n_subjects: int = 6, n_slices_show: int = 5):
    """Save a grid of sample slices - FCD and Control side by side."""
    fig, axes = plt.subplots(
        n_subjects, n_slices_show,
        figsize=(n_slices_show * 3, n_subjects * 3)
    )
    fig.suptitle("Preprocessing Preview - Sample Slices", fontsize=14, y=1.01)

    fcd_subs     = [m for m in all_meta.values() if m["label"] == "FCD"     and m["slice_paths"]]
    control_subs = [m for m in all_meta.values() if m["label"] == "Control" and m["slice_paths"]]

    # Interleave FCD and Control
    subjects = []
    for i in range(n_subjects // 2):
        if i < len(fcd_subs):     subjects.append(fcd_subs[i])
        if i < len(control_subs): subjects.append(control_subs[i])
    subjects = subjects[:n_subjects]

    for row, meta in enumerate(subjects):
        paths = meta["slice_paths"]
        # Pick evenly spaced slices for display
        display_indices = np.linspace(0, len(paths) - 1, n_slices_show, dtype=int)
        for col, pidx in enumerate(display_indices):
            ax = axes[row][col]
            img_arr = np.array(Image.open(paths[pidx]))
            ax.imshow(img_arr, cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"{meta['subj_id']}\n{meta['label']}",
                              fontsize=8, rotation=0, labelpad=60, va="center")

    plt.tight_layout()
    out = RESULTS_DIR / "preview_grid.png"
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved preview grid -> {out}")


def save_intensity_histogram(all_meta: dict, n_sample: int = 10):
    """Compare intensity distributions of FCD vs Control slices."""
    fcd_vals, ctrl_vals = [], []

    fcd_subs  = [m for m in all_meta.values() if m["label"] == "FCD"     and m["slice_paths"]][:n_sample]
    ctrl_subs = [m for m in all_meta.values() if m["label"] == "Control" and m["slice_paths"]][:n_sample]

    for meta in fcd_subs:
        mid = meta["slice_paths"][len(meta["slice_paths"]) // 2]
        arr = np.array(Image.open(mid)).astype(np.float32) / 65535.0
        fcd_vals.extend(arr[arr > 0.01].ravel().tolist())

    for meta in ctrl_subs:
        mid = meta["slice_paths"][len(meta["slice_paths"]) // 2]
        arr = np.array(Image.open(mid)).astype(np.float32) / 65535.0
        ctrl_vals.extend(arr[arr > 0.01].ravel().tolist())

    plt.figure(figsize=(8, 4))
    plt.hist(fcd_vals,  bins=100, alpha=0.6, color="tomato",      label="FCD",     density=True)
    plt.hist(ctrl_vals, bins=100, alpha=0.6, color="steelblue",   label="Control", density=True)
    plt.xlabel("Normalised Intensity")
    plt.ylabel("Density")
    plt.title("FLAIR Intensity Distribution - FCD vs Control (mid slices)")
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "intensity_histogram.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved intensity histogram -> {out}")


def save_slice_count_chart(all_meta: dict):
    """Bar chart of saved slice count per subject."""
    subj_ids = list(all_meta.keys())
    counts   = [all_meta[s]["n_slices_saved"] for s in subj_ids]
    labels   = [all_meta[s]["label"] for s in subj_ids]
    colors   = ["tomato" if l == "FCD" else "steelblue" for l in labels]

    plt.figure(figsize=(max(12, len(subj_ids) * 0.15), 4))
    plt.bar(range(len(subj_ids)), counts, color=colors, width=1.0)
    plt.axhline(N_SLICES, color="black", linestyle="--", linewidth=1,
                label=f"Target N={N_SLICES}")
    plt.xlabel("Subject index")
    plt.ylabel("Saved slice count")
    plt.title("Slice Count per Subject (red=FCD, blue=Control)")
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "slice_count_chart.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved slice count chart -> {out}")


# --- MAIN ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FCD PROJECT - STEP 2: PREPROCESSING")
    print("=" * 60 + "\n")

    # -- Load subject map ----------------------------------------------------
    if not SUBJECT_MAP_PATH.exists():
        raise FileNotFoundError(
            f"subject_map.json not found at {SUBJECT_MAP_PATH}\n"
            "Run step1_download_verify.py first."
        )
    with open(SUBJECT_MAP_PATH) as f:
        subject_map = json.load(f)
    print(f"  Loaded {len(subject_map)} subjects from subject_map.json\n")

    # -- Load existing cache -------------------------------------------------
    all_meta = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            all_meta = json.load(f)
        print(f"  Cache: {len(all_meta)} subjects already processed - skipping those\n")

    # -- Check HD-BET --------------------------------------------------------
    use_hdbet = check_hdbet()
    print()

    # -- Process subjects ----------------------------------------------------
    subjects_to_process = {
        sid: info for sid, info in subject_map.items()
        if sid not in all_meta
    }
    print(f"  Processing {len(subjects_to_process)} subjects...\n")

    failed = []
    for subj_id, info in tqdm(subjects_to_process.items(), desc="Preprocessing"):
        try:
            meta = process_subject(subj_id, info, use_hdbet)
            all_meta[subj_id] = meta

            # Save metadata after each subject - crash-safe
            with open(META_PATH, "w") as f:
                json.dump(all_meta, f, indent=2)

        except Exception as e:
            print(f"\n  [X] {subj_id} failed: {e}")
            traceback.print_exc()
            failed.append((subj_id, str(e)))

    # -- Summary -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    n_fcd  = sum(1 for m in all_meta.values() if m["label"] == "FCD")
    n_ctrl = sum(1 for m in all_meta.values() if m["label"] == "Control")
    total_slices = sum(m["n_slices_saved"] for m in all_meta.values())
    fcd_slices   = sum(m["n_slices_saved"] for m in all_meta.values() if m["label"] == "FCD")
    ctrl_slices  = sum(m["n_slices_saved"] for m in all_meta.values() if m["label"] == "Control")

    print(f"  Subjects processed : {len(all_meta)}")
    print(f"    FCD              : {n_fcd}")
    print(f"    Control          : {n_ctrl}")
    print(f"  Total slices saved : {total_slices}")
    print(f"    FCD slices       : {fcd_slices}")
    print(f"    Control slices   : {ctrl_slices}")
    print(f"  Failed subjects    : {len(failed)}")
    if failed:
        for sid, err in failed:
            print(f"    {sid}: {err}")

    slice_counts = [m["n_slices_saved"] for m in all_meta.values()]
    if slice_counts:
        print(f"\n  Slice count stats  :")
        print(f"    mean  : {np.mean(slice_counts):.1f}")
        print(f"    min   : {np.min(slice_counts)}")
        print(f"    max   : {np.max(slice_counts)}")
        thin = [sid for sid, m in all_meta.items() if m["n_slices_saved"] < N_SLICES]
        if thin:
            print(f"\n  WARNING: {len(thin)} subjects had fewer than {N_SLICES} brain slices:")
            for sid in thin:
                print(f"    {sid}: {all_meta[sid]['n_slices_saved']} slices")

    # -- Save subject_map enriched with slice paths ---------------------------
    # Merge slice paths back into subject_map for downstream steps
    for sid, meta in all_meta.items():
        if sid in subject_map:
            subject_map[sid]["slice_paths"]      = meta["slice_paths"]
            subject_map[sid]["n_slices_saved"]   = meta["n_slices_saved"]
            subject_map[sid]["original_shape"]   = meta["original_shape"]

    enriched_path = Path("data/processed/subject_map.json")
    with open(enriched_path, "w") as f:
        json.dump(subject_map, f, indent=2)
    print(f"\n  [OK] subject_map.json enriched with slice paths -> {enriched_path}")

    # -- Visual verification -------------------------------------------------
    print("\n  Generating visual verification plots...")
    try:
        save_preview_grid(all_meta)
        save_intensity_histogram(all_meta)
        save_slice_count_chart(all_meta)
        print("  [OK] All verification plots saved to results/")
    except Exception as e:
        print(f"  WARNING: Plot generation failed: {e}")
        traceback.print_exc()

    print("\n  Next step: run step3_split.py")
    print("=" * 60 + "\n")