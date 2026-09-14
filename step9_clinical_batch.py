"""
step9_clinical_batch.py
=======================
Unattended batch pipeline for the CLINICAL cohort.

For every subject in the clinical subject map it:
  1. preprocesses the FLAIR exactly as training did (reusing step2_preprocess
     helpers: canonical reorient -> axial-axis fix -> skull strip -> percentile
     normalise -> slice selection -> 16-bit PNG),
  2. writes the PNGs under data/clinical/processed/png/<label>/subj_<id>/,
  3. enriches subject_map_clinical.json with slice_paths (so step8 can run), and
  4. predicts FCD / Control per subject with ConvNeXt-Tiny and the soft-voting
     ensemble, using thresholds FIXED on the out-of-fold development data.

Nothing under data/processed/ (the ds004199 files) is read for writing or
modified in any way.

Resumable: subjects that already have PNGs are skipped unless --force.

Run:
    python build_clinical_map.py          # first, to create the map
    python step9_clinical_batch.py        # then this
    python step9_clinical_batch.py --no-predict     # preprocessing only
    python step9_clinical_batch.py --only-predict   # skip preprocessing

Afterwards, for the full metrics (AUC, CI, confusion matrices):
    python step8_test_eval.py \
        --subject-map data/clinical/processed/subject_map_clinical.json \
        --split       data/clinical/processed/master_split_clinical.json \
        --out-name    clinical_eval
"""

from __future__ import annotations
import argparse, json, sys, traceback
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import roc_curve

IMG_SIZE = 224
FCD_CLASS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_TFM = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def _convnext():
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2); return m
def _effb0():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2); return m
def _effv2s():
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2); return m
def _xception():
    import timm; return timm.create_model("xception", pretrained=False, num_classes=2)

ENSEMBLE = {                      # ResNet50-D excluded, as in the paper
    "ConvNeXt-Tiny": (_convnext, "convnext_tiny"),
    "EfficientNet-B0": (_effb0, "efficientnet"),
    "EfficientNet-V2-S": (_effv2s, "efficientnet_v2s"),
    "Xception": (_xception, "xception"),
}


def find_root(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    for s in [Path.cwd(), Path(__file__).resolve().parent]:
        for d in [s, *s.parents]:
            if (d / "models").is_dir() and (d / "data").is_dir():
                return d
    raise FileNotFoundError("pass --root")


# ---------------------------------------------------------------- preprocessing
def preprocess_subject(s2, subj, info, root, png_dir, cache_dir, use_hdbet):
    """Replicates step2.process_subject for one clinical subject."""
    flair = root / info["flair_path"]
    label = info["label"]
    out_dir = png_dir / label / f"subj_{subj}"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = nib.as_closest_canonical(nib.load(str(flair)))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"expected 3D, got {data.shape}")

    # same axial-axis heuristic as step2
    note = ""
    if data.shape[1] < data.shape[2] and data.shape[1] < data.shape[0]:
        data = np.transpose(data, (0, 2, 1)); note = "transposed(0,2,1)"
    elif data.shape[0] < data.shape[2] and data.shape[0] < data.shape[1]:
        data = np.transpose(data, (1, 2, 0)); note = "transposed(1,2,0)"

    canonical = nib.Nifti1Image(data, img.affine, img.header)
    stripped = s2.apply_skull_strip(canonical, flair, cache_dir, use_hdbet)
    normed = s2.percentile_normalize(stripped)

    saved, skipped = [], 0
    for idx in s2.select_slices(normed):
        sl = normed[:, :, idx].T
        if not s2.is_informative_slice(sl):
            skipped += 1
            continue
        fp = out_dir / f"{subj}_slice{idx:04d}.png"
        s2.save_slice_png(sl, fp)
        saved.append(str(fp.relative_to(root)).replace("\\", "/"))
    return saved, skipped, list(data.shape), note


# ------------------------------------------------------------------- inference
def load_folds(root, folder, builder):
    mods = []
    for k in range(1, 6):
        ck = root / "models" / folder / f"fold{k}_best.pt"
        if not ck.exists():
            continue
        m = builder()
        st = torch.load(str(ck), map_location=DEVICE)
        m.load_state_dict(st["state_dict"] if "state_dict" in st else st)
        m.eval().to(DEVICE)
        mods.append(m)
    return mods


def load_tensor(p):
    arr = np.array(Image.open(p)).astype(np.float32) / 65535.0
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    return VAL_TFM(pil)


@torch.no_grad()
def subject_prob(fold_models, slice_paths, batch=16):
    fold_means = []
    tensors = torch.stack([load_tensor(p) for p in slice_paths])
    for m in fold_models:
        probs = []
        for i in range(0, len(tensors), batch):
            chunk = tensors[i:i + batch].to(DEVICE)
            probs.append(torch.softmax(m(chunk), dim=1)[:, FCD_CLASS].cpu())
        fold_means.append(float(torch.cat(probs).mean()))
    return float(np.mean(fold_means))


def gui_thresholds(root):
    """Identical to step7_gui.compute_thresholds() (ConvNeXt-Tiny)."""
    p = np.load(root / "models" / "convnext_tiny" / "oof_probs.npy").ravel()
    y = np.load(root / "models" / "convnext_tiny" / "oof_labels.npy").astype(int).ravel()
    fpr, tpr, thr = roc_curve(y, p)
    bal = float(thr[int(np.argmax(tpr - fpr))])
    mask = tpr >= 0.95
    scr = float(thr[int(np.argmax(mask))]) if mask.any() else float(thr[-1])
    return {"balanced": bal, "screening": scr}


def ensemble_threshold(root):
    """Youden threshold on the soft-voting OOF probabilities."""
    tot, cnt, lab = {}, {}, {}
    for _, folder in ENSEMBLE.values():
        d = root / "models" / folder
        if not all((d / f).exists() for f in
                   ("oof_probs.npy", "oof_labels.npy", "oof_subj_ids.npy")):
            continue
        pr = np.load(d / "oof_probs.npy").ravel()
        ids = [str(x) for x in np.load(d / "oof_subj_ids.npy", allow_pickle=True)]
        la = np.load(d / "oof_labels.npy").astype(int).ravel()
        for s, p_, l_ in zip(ids, pr, la):
            tot[s] = tot.get(s, 0.0) + p_; cnt[s] = cnt.get(s, 0) + 1; lab[s] = l_
    subs = sorted(tot)
    p = np.array([tot[s] / cnt[s] for s in subs]); y = np.array([lab[s] for s in subs])
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--map", default="data/clinical/processed/subject_map_clinical.json")
    ap.add_argument("--png-dir", default="data/clinical/processed/png")
    ap.add_argument("--cache-dir", default="data/clinical/processed/cache")
    ap.add_argument("--force", action="store_true", help="re-preprocess even if PNGs exist")
    ap.add_argument("--no-predict", action="store_true", help="preprocess only")
    ap.add_argument("--only-predict", action="store_true", help="skip preprocessing")
    ap.add_argument("--gui-mode", choices=["balanced", "screening"], default="balanced")
    args = ap.parse_args()

    root = find_root(args.root)
    map_path = root / args.map
    if not map_path.exists():
        raise SystemExit(f"clinical map not found: {map_path}\n"
                         f"run build_clinical_map.py first")
    png_dir = root / args.png_dir
    cache_dir = root / args.cache_dir
    png_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = root / "results" / "clinical_eval"
    out.mkdir(parents=True, exist_ok=True)

    smap = json.load(open(map_path))
    print(f"Clinical subjects in map : {len(smap)}")
    print(f"Device                   : {DEVICE}")

    # ---------------- phase 1: preprocessing ----------------
    if not args.only_predict:
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            import step2_preprocess as s2
        except Exception as e:
            raise SystemExit(f"could not import step2_preprocess.py: {e}\n"
                             "place this script in the project root")
        use_hdbet = s2.check_hdbet()

        failures = []
        for i, (subj, info) in enumerate(sorted(smap.items()), 1):
            existing = info.get("slice_paths") or []
            if existing and not args.force:
                print(f"  [{i}/{len(smap)}] {subj}: skip ({len(existing)} slices exist)")
                continue
            try:
                saved, skipped, shape, note = preprocess_subject(
                    s2, subj, info, root, png_dir, cache_dir, use_hdbet)
                info["slice_paths"] = saved
                info["n_slices_saved"] = len(saved)
                info["n_slices_skipped"] = skipped
                info["original_shape"] = shape
                if note:
                    info["axis_fix"] = note
                flag = ""
                if len(saved) == 0:
                    flag = "  <-- NO SLICES SAVED"
                elif len(saved) < 20:
                    flag = "  <-- few slices"
                print(f"  [{i}/{len(smap)}] {subj} ({info['label']}): "
                      f"{len(saved)} slices {note}{flag}")
            except Exception as e:
                failures.append((subj, str(e)))
                print(f"  [{i}/{len(smap)}] {subj}: FAILED - {e}")
                traceback.print_exc(limit=1)
            # save after every subject so a crash never loses work
            with open(map_path, "w") as fh:
                json.dump(smap, fh, indent=2)

        print(f"\nPreprocessing done. Map updated -> {map_path}")
        if failures:
            print(f"  {len(failures)} failed:")
            for s, e in failures:
                print(f"    {s}: {e}")

    if args.no_predict:
        return

    # ---------------- phase 2: inference ----------------
    subjects = [s for s in sorted(smap) if smap[s].get("slice_paths")]
    if not subjects:
        raise SystemExit("no preprocessed subjects to predict on")
    print(f"\nPredicting on {len(subjects)} subjects ...")

    model_probs = {}
    for name, (builder, folder) in ENSEMBLE.items():
        folds = load_folds(root, folder, builder)
        if not folds:
            print(f"  [skip] {name}: no checkpoints"); continue
        model_probs[name] = np.array([
            subject_prob(folds, [str(root / p) for p in smap[s]["slice_paths"]])
            for s in subjects])
        print(f"  scored {name} ({len(folds)} folds)")
        del folds
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    cnx_thr = gui_thresholds(root)[args.gui_mode]
    ens_thr = ensemble_threshold(root)
    cnx_p = model_probs.get("ConvNeXt-Tiny")
    soft_p = np.vstack([model_probs[n] for n in ENSEMBLE if n in model_probs]).mean(axis=0)

    # ---------------- phase 3: report ----------------
    rows = []
    print(f"\n{'subject':12s} {'truth':8s} | {'ConvNeXt':>8s} {'call':8s} | "
          f"{'Soft':>8s} {'call':8s} | note")
    print("-" * 78)
    for i, s in enumerate(subjects):
        truth = smap[s]["label"]
        cp = float(cnx_p[i]) if cnx_p is not None else float("nan")
        sp = float(soft_p[i])
        c_call = "FCD" if cp >= cnx_thr else "Control"
        s_call = "FCD" if sp >= ens_thr else "Control"
        note = smap[s].get("acquisition_note", "")
        mark = "" if s_call == truth else "  *MISS"
        print(f"{s:12s} {truth:8s} | {cp:8.3f} {c_call:8s} | {sp:8.3f} {s_call:8s} | "
              f"{note}{mark}")
        rows.append(dict(subject=s, truth=truth, label_int=smap[s]["label_int"],
                         convnext_prob=round(cp, 4), convnext_call=c_call,
                         soft_prob=round(sp, 4), soft_call=s_call,
                         n_slices=len(smap[s]["slice_paths"]),
                         acquisition_note=note))

    with open(out / "clinical_predictions.csv", "w") as f:
        f.write("subject,truth,label_int,convnext_prob,convnext_call,"
                "soft_prob,soft_call,n_slices,acquisition_note\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in
                             ["subject", "truth", "label_int", "convnext_prob",
                              "convnext_call", "soft_prob", "soft_call",
                              "n_slices", "acquisition_note"]) + "\n")
    json.dump(dict(convnext_threshold=cnx_thr, ensemble_threshold=ens_thr,
                   gui_mode=args.gui_mode, n_subjects=len(subjects),
                   predictions=rows),
              open(out / "clinical_predictions.json", "w"), indent=2)

    y = np.array([smap[s]["label_int"] for s in subjects])
    c_acc = float(((cnx_p >= cnx_thr).astype(int) == y).mean()) if cnx_p is not None else float("nan")
    s_acc = float(((soft_p >= ens_thr).astype(int) == y).mean())
    print("-" * 78)
    print(f"Simple accuracy  ConvNeXt {c_acc:.3f}   Soft voting {s_acc:.3f}")
    print(f"Thresholds (from OOF): ConvNeXt={cnx_thr:.3f} ({args.gui_mode}), "
          f"ensemble={ens_thr:.3f}")
    print(f"\nSaved -> {out}/clinical_predictions.(csv|json)")
    print("For AUC / CI / confusion matrices, now run:\n"
          "  python step8_test_eval.py "
          "--subject-map data/clinical/processed/subject_map_clinical.json "
          "--split data/clinical/processed/master_split_clinical.json "
          "--out-name clinical_eval")


if __name__ == "__main__":
    main()