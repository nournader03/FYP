"""
step8_test_eval.py
==================
Final evaluation on the LOCKED 34-subject test set.

Scores every trained architecture individually, plus two soft-voting ensembles:

  * ENSEMBLE_4  ConvNeXt-Tiny + EfficientNet-B0 + EfficientNet-V2-S + Xception
                - the ensemble reported in the submitted paper. UNCHANGED.
  * ENSEMBLE_5  the four above + Swin-T
                - the extended ensemble. Reported ALONGSIDE, never replacing.

Both are reported so the thesis and the paper cannot disagree.

Evaluation discipline (do not violate):
  * Test subjects (master_split.json -> test_subjects) were never seen by any
    fold, so all five fold checkpoints are averaged for every test subject.
  * Every operating threshold is FIXED on out-of-fold (development) data and
    applied unchanged to the test set. No threshold is computed on test data.
  * Run once and report. Do not tune anything after seeing test results.

Threshold-independent metrics (AUC, AP, Brier) use no threshold.
Operating-point metrics (Sens/Spec/F1/Acc, confusion matrix) use the OOF threshold.

Run:
    python step8_test_eval.py
    python step8_test_eval.py --batch 8          # if VRAM is tight
    python step8_test_eval.py --gui-mode screening
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             accuracy_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE = 224
FCD_CLASS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
VAL_TFM = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


# --- MODEL BUILDERS ------------------------------------------------------------
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
    import timm
    for cand in ("xception", "legacy_xception"):
        try:
            return timm.create_model(cand, pretrained=False, num_classes=2)
        except Exception:
            continue
    raise ValueError("no Xception variant available in this timm version")
def _resnet50d():
    import timm; return timm.create_model("resnet50d", pretrained=False, num_classes=2)
def _swin():
    import timm; return timm.create_model("swin_tiny_patch4_window7_224",
                                          pretrained=False, num_classes=2)


# The paper's ensemble. DO NOT MODIFY - ResNet50-D was excluded on the basis of
# its out-of-fold AUC, in consultation with the supervisor.
ENSEMBLE_4 = {
    "ConvNeXt-Tiny":     (_convnext, "convnext_tiny"),
    "EfficientNet-B0":   (_effb0,    "efficientnet"),
    "EfficientNet-V2-S": (_effv2s,   "efficientnet_v2s"),
    "Xception":          (_xception, "xception"),
}

# Extended ensemble: the four above plus Swin-T.
ENSEMBLE_5 = dict(ENSEMBLE_4)
ENSEMBLE_5["Swin-T"] = (_swin, "swin_tiny")

# Everything scored individually on the test set.
ALL_MODELS = dict(ENSEMBLE_5)
ALL_MODELS["ResNet50-D"] = (_resnet50d, "resnet50v2")


def find_root(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    for s in [Path.cwd(), Path(__file__).resolve().parent]:
        for d in [s, *s.parents]:
            if (d / "models").is_dir() and (d / "data").is_dir():
                return d
    raise FileNotFoundError("pass --root")


def load_folds(root, folder, builder):
    mods = []
    for k in range(1, 6):
        ck = root / "models" / folder / f"fold{k}_best.pt"
        if not ck.exists():
            continue
        m = builder()
        # weights_only=False: checkpoints written by this project may store numpy
        # scalars alongside the weights. torch 2.6 flipped this default to True.
        try:
            st = torch.load(str(ck), map_location=DEVICE, weights_only=False)
        except TypeError:                       # torch < 2.4 has no such argument
            st = torch.load(str(ck), map_location=DEVICE)
        m.load_state_dict(st["state_dict"] if "state_dict" in st else st)
        m.eval().to(DEVICE)
        mods.append(m)
    return mods


def load_slice_tensor(png_path):
    arr = np.array(Image.open(png_path)).astype(np.float32) / 65535.0
    pil = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGB")
    return VAL_TFM(pil)


@torch.no_grad()
def subject_probability(fold_models, slice_paths, batch=16):
    """Mean FCD probability over all folds and all slices for one subject.
    Slices are processed in mini-batches so a 2 GB GPU can handle 60+ slices."""
    tensors = torch.stack([load_slice_tensor(p) for p in slice_paths])
    fold_means = []
    for m in fold_models:
        probs = []
        for i in range(0, len(tensors), batch):
            chunk = tensors[i:i + batch].to(DEVICE)
            probs.append(torch.softmax(m(chunk), dim=1)[:, FCD_CLASS].cpu())
        fold_means.append(float(torch.cat(probs).mean()))     # avg slices
    return float(np.mean(fold_means))                          # avg folds


# --- OOF-DERIVED THRESHOLDS (never computed on test data) ----------------------
def youden_from_oof(root, folder):
    """Balanced threshold from a single model's OOF predictions."""
    p = np.load(root / "models" / folder / "oof_probs.npy").ravel()
    y = np.load(root / "models" / folder / "oof_labels.npy").astype(int).ravel()
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def gui_thresholds(root):
    """Replicates step7_gui.compute_thresholds() EXACTLY so ConvNeXt-Tiny test
    labels match what the GUI would output (balanced and screening modes)."""
    p = np.load(root / "models" / "convnext_tiny" / "oof_probs.npy").ravel()
    y = np.load(root / "models" / "convnext_tiny" / "oof_labels.npy").astype(int).ravel()
    fpr, tpr, thr = roc_curve(y, p)
    bal = float(thr[int(np.argmax(tpr - fpr))])
    mask = tpr >= 0.95
    scr = float(thr[int(np.argmax(mask))]) if mask.any() else float(thr[-1])
    return {"balanced": bal, "screening": scr}


def ensemble_oof_threshold(root, members):
    """
    Soft-voting OOF probability per subject for the given ensemble members,
    aligned BY SUBJECT ID (model folders store OOF rows in different orders),
    then the Youden threshold on those pooled probabilities.
    """
    per_model = {}
    labels = {}
    for _, folder in members.values():
        d = root / "models" / folder
        need = ("oof_probs.npy", "oof_labels.npy", "oof_subj_ids.npy")
        if not all((d / f).exists() for f in need):
            print(f"    [warn] {folder}: OOF files incomplete, excluded from "
                  f"the ensemble threshold")
            continue
        probs = np.load(d / "oof_probs.npy").ravel()
        ids = [str(x) for x in np.load(d / "oof_subj_ids.npy", allow_pickle=True).ravel()]
        labs = np.load(d / "oof_labels.npy").astype(int).ravel()
        per_model[folder] = dict(zip(ids, probs))
        labels.update(dict(zip(ids, labs)))

    if not per_model:
        raise SystemExit("No OOF files available to derive an ensemble threshold")

    common = set.intersection(*[set(d) for d in per_model.values()])
    subs = sorted(common)
    p = np.array([np.mean([per_model[f][s] for f in per_model]) for s in subs])
    y = np.array([labels[s] for s in subs])
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))]), len(subs)


# --- METRICS -------------------------------------------------------------------
def metrics(y, p, thr, n_boot=2000):
    preds = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        boot.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2)
    return dict(
        n=int(len(y)), n_fcd=int(y.sum()),
        auc=round(float(roc_auc_score(y, p)), 4),
        auc_ci=[round(float(lo), 4), round(float(hi), 4)],
        ap=round(float(average_precision_score(y, p)), 4),
        brier=round(float(brier_score_loss(y, p)), 4),
        threshold=round(float(thr), 4),
        sensitivity=round(sens, 4), specificity=round(spec, 4),
        ppv=round(ppv, 4), npv=round(npv, 4),
        f1=round(float(f1_score(y, preds, zero_division=0)), 4),
        accuracy=round(float(accuracy_score(y, preds)), 4),
        TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))


# --- FIGURES -------------------------------------------------------------------
def plot_confusions(rows, out):
    names = list(rows)
    cols = min(4, len(names))
    nrows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.6 * cols, 3.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        m = rows[name]
        mat = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                    color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred N", "Pred FCD"], fontsize=8)
        ax.set_yticklabels(["True N", "True FCD"], fontsize=8)
        ax.set_title(f"{name}\nAUC={m['auc']}  Se={m['sensitivity']}  "
                     f"Sp={m['specificity']}", fontsize=8.5)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Locked test set - confusion matrices at OOF-derived thresholds",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, .95])
    plt.savefig(out / "test_confusion_matrices.png", dpi=200); plt.close()


def plot_test_roc(y, probs_by_name, out):
    plt.figure(figsize=(7, 6.5))
    for name, p in probs_by_name.items():
        fpr, tpr, _ = roc_curve(y, p)
        lw = 3.0 if name.startswith("Soft voting") else 1.8
        plt.plot(fpr, tpr, lw=lw,
                 label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
    plt.title(f"ROC - locked test set (n={len(y)})")
    plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / "test_roc_curves.png", dpi=200); plt.close()


# --- MAIN ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--convnext-threshold", type=float, default=None)
    ap.add_argument("--ensemble-threshold", type=float, default=None)
    ap.add_argument("--gui-mode", choices=["balanced", "screening"], default="balanced")
    ap.add_argument("--batch", type=int, default=16,
                    help="slices per forward pass (lower if VRAM is tight)")
    ap.add_argument("--subject-map", default="data/processed/subject_map.json")
    ap.add_argument("--split", default="data/processed/master_split.json")
    ap.add_argument("--out-name", default="test_eval")
    args = ap.parse_args()

    root = find_root(args.root)
    out = root / "results" / args.out_name
    out.mkdir(parents=True, exist_ok=True)

    split = json.load(open(root / args.split))
    test_ids = split["test_subjects"]
    smap = json.load(open(root / args.subject_map))
    test_ids = [s for s in test_ids if s in smap and smap[s].get("slice_paths")]
    print(f"Locked test subjects: {len(test_ids)} on {DEVICE}")

    def slices_for(sid):
        return [str((root / p).resolve()) for p in smap[sid]["slice_paths"]]

    y = np.array([int(smap[s]["label_int"]) for s in test_ids])
    print(f"  FCD={int(y.sum())}  Control={int((y == 0).sum())}\n")

    # ---- score each model (one in memory at a time) ------------------------
    model_probs = {}
    for name, (builder, folder) in ALL_MODELS.items():
        try:
            folds = load_folds(root, folder, builder)
        except Exception as e:
            print(f"  [skip] {name}: {e}"); continue
        if not folds:
            print(f"  [skip] {name}: no checkpoints in models/{folder}"); continue
        model_probs[name] = np.array([
            subject_probability(folds, slices_for(s), args.batch) for s in test_ids])
        print(f"  scored {name:20s} ({len(folds)} folds)")
        del folds
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not model_probs:
        raise SystemExit("No models could be scored - check models/<folder>/fold*_best.pt")

    rows, probs_by_name = {}, {}

    # ---- individual models -------------------------------------------------
    for name, (_, folder) in ALL_MODELS.items():
        if name not in model_probs:
            continue
        if name == "ConvNeXt-Tiny":
            # ConvNeXt uses the GUI's operating point so its test calls match
            # exactly what the deployed prototype would output.
            thr = (args.convnext_threshold if args.convnext_threshold is not None
                   else gui_thresholds(root)[args.gui_mode])
        else:
            thr = youden_from_oof(root, folder)
        rows[name] = metrics(y, model_probs[name], thr)
        probs_by_name[name] = model_probs[name]

    # ---- ensembles ---------------------------------------------------------
    for label, members in [("Soft voting (4-model)", ENSEMBLE_4),
                           ("Soft voting (5-model, +Swin)", ENSEMBLE_5)]:
        avail = [n for n in members if n in model_probs]
        if len(avail) < 2:
            print(f"  [skip] {label}: only {len(avail)} member(s) available")
            continue
        if len(avail) < len(members):
            print(f"  [warn] {label}: {len(avail)}/{len(members)} members available "
                  f"({', '.join(avail)})")
        p = np.vstack([model_probs[n] for n in avail]).mean(axis=0)
        present = {n: members[n] for n in avail}
        thr, n_oof = ensemble_oof_threshold(root, present)
        if args.ensemble_threshold is not None and members is ENSEMBLE_4:
            thr = args.ensemble_threshold
        rows[label] = metrics(y, p, thr)
        rows[label]["members"] = avail
        rows[label]["oof_subjects_for_threshold"] = n_oof
        probs_by_name[label] = p

    # ---- outputs -----------------------------------------------------------
    json.dump({"test_subjects": len(test_ids), "n_fcd": int(y.sum()),
               "gui_mode": args.gui_mode, "results": rows},
              open(out / "test_metrics.json", "w"), indent=2)

    cols = ["auc", "ap", "brier", "sensitivity", "specificity", "ppv", "npv",
            "f1", "accuracy", "threshold", "TP", "FP", "FN", "TN"]
    with open(out / "test_metrics.csv", "w") as f:
        f.write("method,auc_ci_low,auc_ci_high," + ",".join(cols) + "\n")
        for n, m in rows.items():
            f.write(f"{n},{m['auc_ci'][0]},{m['auc_ci'][1]}," +
                    ",".join(str(m[c]) for c in cols) + "\n")

    with open(out / "test_metrics.md", "w") as f:
        f.write("| Method | AUC (95% CI) | AP | Brier | Sens | Spec | PPV | NPV | "
                "F1 | Acc | Thr |\n")
        f.write("|" + "---|" * 11 + "\n")
        for n, m in rows.items():
            f.write(f"| {n} | {m['auc']} ({m['auc_ci'][0]}-{m['auc_ci'][1]}) | "
                    f"{m['ap']} | {m['brier']} | {m['sensitivity']} | "
                    f"{m['specificity']} | {m['ppv']} | {m['npv']} | {m['f1']} | "
                    f"{m['accuracy']} | {m['threshold']} |\n")

    plot_confusions(rows, out)
    plot_test_roc(y, probs_by_name, out)

    # ---- console -----------------------------------------------------------
    print("\n" + "=" * 84)
    print(f"  LOCKED TEST SET (n={len(y)}, FCD={int(y.sum())})")
    print("=" * 84)
    print(f"  {'method':30s} {'AUC':>6s} {'95% CI':>15s} {'Se':>6s} {'Sp':>6s} "
          f"{'F1':>6s} {'thr':>6s}")
    print("  " + "-" * 80)
    for n, m in rows.items():
        ci = f"{m['auc_ci'][0]:.3f}-{m['auc_ci'][1]:.3f}"
        print(f"  {n:30s} {m['auc']:6.3f} {ci:>15s} {m['sensitivity']:6.3f} "
              f"{m['specificity']:6.3f} {m['f1']:6.3f} {m['threshold']:6.3f}")
    print("=" * 84)
    print("\n  All thresholds were fixed on out-of-fold data before this run.")
    print("  ConvNeXt-Tiny uses the GUI's operating point "
          f"({args.gui_mode}), so its test calls match the deployed prototype.")
    print("  The 4-model ensemble is the one reported in the paper; the 5-model")
    print("  ensemble is an extension and is reported alongside, not instead of it.")
    print(f"\n  Saved -> {out}")
    print("    test_metrics.json / .csv / .md")
    print("    test_confusion_matrices.png, test_roc_curves.png\n")


if __name__ == "__main__":
    main()