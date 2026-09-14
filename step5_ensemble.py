"""
step5_ensemble.py
=================
FCD Detection DL Project - Step 5: Ensemble of the fine-tuned models.

Operates ENTIRELY on the subject-level out-of-fold (OOF) probabilities saved in
each model folder. The locked 34-subject test set is NEVER touched here (step 8).

Three ensemble methods are built and compared, all evaluated on OOF only:
  1. Soft voting            - unweighted mean of per-subject probabilities
  2. Weighted soft voting   - mean weighted by each model's OOF AUC
  3. Stacking meta-learner  - logistic regression on the OOF probs as features,
                              evaluated with NESTED cross-validation so the meta
                              -learner is never scored on subjects it trained on.

SUBJECT ALIGNMENT
-----------------
Model folders do NOT store OOF rows in the same subject order: step4_train.py
uses dict-insertion order, while the Colab script that trained Swin-T uses sorted
order. Averaging row-wise without realigning would silently combine different
subjects. Every model is therefore reindexed onto a common sorted subject order
using oof_subj_ids.npy before anything is computed, and the aligned subject list
is saved alongside the ensemble probabilities so downstream scripts can align too.

Outputs -> results/<out-name>/:
  <method>_oof_probs.npy       (3 files)
  ensemble_subj_ids.npy        subject order for every array written here
  ensemble_oof_labels.npy
  ensemble_metrics.json
  meta_learner.json            (weights, thresholds, LR coefficients)
  stacking_model.joblib        (fitted final meta-learner)
  roc_all.png, confusion_matrices.png, calibration_curves.png

Label convention: 1 = FCD (positive), 0 = Normal/Control.

To run:
    # 4-model ensemble as reported in the paper (default)
    python step5_ensemble.py

    # 5-model extension including Swin-T, written to a separate folder
    python step5_ensemble.py --models efficientnet,efficientnet_v2s,convnext_tiny,xception,swin_tiny --out-name ensemble_with_swin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Default = the ensemble reported in the submitted paper. ResNet50-D excluded:
# weakest model (OOF AUC 0.757), depressed ensemble AUC.
DEFAULT_MODELS = [
    "efficientnet",          # folder is 'efficientnet' (EfficientNet-B0)
    "efficientnet_v2s",
    "convnext_tiny",
    "xception",
]

RANDOM_STATE = 42
SCREENING_SENS = 0.95
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
INNER_C_GRID = [0.01, 0.1, 1.0, 10.0]
N_BINS_CAL = 10
WEIGHT_MODE = "auc"

EXPECTED_N = 135             # trainval subjects


# --------------------------------------------------------------------------- #
def find_project_root(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if (p / "models").is_dir() and (p / "results").is_dir():
            return p
        raise FileNotFoundError(f"--root {p} has no models/ + results/ dirs")
    for start in [Path.cwd(), Path(__file__).resolve().parent]:
        for d in [start, *start.parents]:
            if (d / "models").is_dir() and (d / "results").is_dir():
                return d
    raise FileNotFoundError(
        "Could not auto-detect project root. Pass --root path\\to\\fcd_project")


# --------------------------------------------------------------------------- #
# Load and align OOF arrays BY SUBJECT ID
# --------------------------------------------------------------------------- #
def load_oof(root: Path, model_list: list[str]):
    """
    Returns:
      P       : (n_models, n_subjects) OOF probabilities, aligned by subject ID
      y       : (n_subjects,) canonical ground-truth labels
      names   : model folder names in row order of P
      subj_ids: (n_subjects,) subject IDs in column order of P
    """
    prob_rows, label_arrays, id_arrays, names = [], [], [], []

    for m in model_list:
        folder = root / "models" / m
        pf, lf, idf = (folder / "oof_probs.npy", folder / "oof_labels.npy",
                       folder / "oof_subj_ids.npy")
        if not (pf.exists() and lf.exists()):
            raise FileNotFoundError(f"{m}: missing oof_probs.npy / oof_labels.npy")
        if not idf.exists():
            raise FileNotFoundError(
                f"{m}: oof_subj_ids.npy is required so models can be aligned by "
                f"subject ID. Model folders store OOF rows in different orders.")

        probs = np.load(pf).astype(np.float64).ravel()
        labels = np.load(lf).astype(int).ravel()
        ids = np.array([str(s) for s in np.load(idf, allow_pickle=True).ravel()])

        if not (probs.shape == labels.shape == ids.shape):
            raise ValueError(f"{m}: probs {probs.shape}, labels {labels.shape}, "
                             f"ids {ids.shape} disagree")
        if probs.min() < 0 or probs.max() > 1:
            raise ValueError(f"{m}: probabilities outside [0,1] "
                             f"(min={probs.min():.3f}, max={probs.max():.3f})")
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError(f"{m}: labels are not all in {{0,1}}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{m}: duplicate subject IDs in oof_subj_ids.npy")

        prob_rows.append(probs)
        label_arrays.append(labels)
        id_arrays.append(ids)
        names.append(m)
        print(f"  loaded {m:18s}  n={probs.size}  "
              f"FCD={int(labels.sum())}  Normal={int((labels == 0).sum())}")

    # --- align by subject ID ------------------------------------------------ #
    common = set(id_arrays[0])
    for ids in id_arrays[1:]:
        common &= set(ids)
    subj_ids = np.array(sorted(common))

    if len(subj_ids) == 0:
        raise ValueError("No subjects are common to all model folders - the OOF "
                         "files come from different cohorts.")
    for m, ids in zip(names, id_arrays):
        dropped = len(ids) - len(subj_ids)
        if dropped:
            print(f"  [warn] {m}: {dropped} subject(s) not shared with every "
                  f"other model, excluded from the ensemble")
    if len(subj_ids) != EXPECTED_N:
        print(f"  [warn] {len(subj_ids)} common subjects (expected {EXPECTED_N})")

    aligned_probs, aligned_labels = [], []
    for m, ids, pr, la in zip(names, id_arrays, prob_rows, label_arrays):
        pos = {s: i for i, s in enumerate(ids)}
        sel = np.array([pos[s] for s in subj_ids])
        aligned_probs.append(pr[sel])
        aligned_labels.append(la[sel])

    ref = aligned_labels[0]
    for m, lab in zip(names[1:], aligned_labels[1:]):
        if not np.array_equal(ref, lab):
            raise ValueError(
                f"Labels for '{m}' disagree with '{names[0]}' AFTER aligning by "
                f"subject ID. The OOF files come from different cohorts and must "
                f"not be combined.")
    print(f"  alignment OK: {len(subj_ids)} subjects matched by ID across "
          f"{len(names)} models.\n")

    return np.vstack(aligned_probs), ref.astype(int), names, subj_ids


# --------------------------------------------------------------------------- #
# Ensemble methods
# --------------------------------------------------------------------------- #
def soft_voting(P: np.ndarray) -> np.ndarray:
    return P.mean(axis=0)


def weighted_voting(P, aucs, mode):
    w = np.clip(aucs - 0.5, 1e-6, None) if mode == "auc_minus_chance" else aucs.copy()
    w = w / w.sum()
    return (w[:, None] * P).sum(axis=0), w


def _select_C(X, y):
    inner = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    best_C, best_auc = INNER_C_GRID[0], -1.0
    for C in INNER_C_GRID:
        scores = []
        for tr, va in inner.split(X, y):
            lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            lr.fit(X[tr], y[tr])
            scores.append(roc_auc_score(y[va], lr.predict_proba(X[va])[:, 1]))
        mean_auc = float(np.mean(scores))
        if mean_auc > best_auc:
            best_auc, best_C = mean_auc, C
    return best_C


def stacking_nested_cv(P, y):
    X = P.T
    outer = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)
    oof = np.zeros(X.shape[0], dtype=np.float64)
    chosen_Cs = []
    for tr, te in outer.split(X, y):
        C = _select_C(X[tr], y[tr])
        chosen_Cs.append(C)
        lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        lr.fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    final_C = _select_C(X, y)
    final_lr = LogisticRegression(C=final_C, max_iter=2000, solver="lbfgs")
    final_lr.fit(X, y)
    return oof, final_lr, final_C, chosen_Cs


# --------------------------------------------------------------------------- #
# Thresholds & metrics
# --------------------------------------------------------------------------- #
def balanced_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def screening_threshold(y, p, target):
    fpr, tpr, thr = roc_curve(y, p)
    mask = tpr >= target
    return float(thr[int(np.argmax(mask))]) if mask.any() else float(thr[-1])


def metrics_from_pred(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {
        "sensitivity": round(sens, 4), "specificity": round(spec, 4),
        "precision": round(prec, 4), "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "confusion_matrix": {"TP": int(tp), "FP": int(fp),
                             "FN": int(fn), "TN": int(tn)},
    }


def metrics_at_threshold(y, p, thr):
    return {"threshold": round(thr, 6), **metrics_from_pred(y, (p >= thr).astype(int))}


def hard_voting_eval(P, y, names, soft_probs):
    n_models = P.shape[0]
    per_model_thr = np.array([balanced_threshold(y, P[i]) for i in range(n_models)])
    votes = (P >= per_model_thr[:, None]).astype(int)
    vote_count = votes.sum(axis=0)

    soft_decision = (soft_probs >= balanced_threshold(y, soft_probs)).astype(int)
    majority = n_models / 2.0
    pred = np.where(vote_count > majority, 1,
                    np.where(vote_count < majority, 0, soft_decision)).astype(int)
    n_ties = int((vote_count == majority).sum()) if n_models % 2 == 0 else 0

    result = metrics_from_pred(y, pred)
    result.update({
        "per_model_vote_thresholds": {n: round(float(t), 6)
                                      for n, t in zip(names, per_model_thr)},
        "n_tie_subjects": n_ties,
        "tie_break": ("soft-voting fallback" if n_models % 2 == 0
                      else "not required (odd number of models)"),
        "ordinal_vote_auc": round(float(roc_auc_score(y, vote_count)), 4),
        "ordinal_vote_auc_note": (
            f"AUC from ordinal vote count (0..{n_models}); coarse, "
            f"{n_models + 1} levels only - not directly comparable to "
            "probability-based AUC"),
    })
    return result, pred, vote_count


def compute_ece(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p > lo) & (p <= hi) if i > 0 else (p >= lo) & (p <= hi)
        if sel.sum():
            ece += (sel.sum() / n) * abs(y[sel].mean() - p[sel].mean())
    return float(ece)


def evaluate(y, p):
    return {
        "oof_auc": round(float(roc_auc_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(compute_ece(y, p, N_BINS_CAL), 4),
        "balanced": metrics_at_threshold(y, p, balanced_threshold(y, p)),
        "screening": metrics_at_threshold(
            y, p, screening_threshold(y, p, SCREENING_SENS)),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_roc(y, indiv, ensembles, out, hard_point=None):
    plt.figure(figsize=(7.5, 7))
    for name, p in indiv.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=1, ls="--", alpha=0.7,
                 label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    for name, p in ensembles.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=2.5,
                 label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    if hard_point is not None:
        sens, spec = hard_point
        plt.scatter([1 - spec], [sens], marker="*", s=220, color="red",
                    zorder=5, edgecolor="black",
                    label=f"hard_voting (operating pt, Se={sens:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, ls=":")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("OOF ROC - individual models vs ensembles")
    plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_confusion_matrices(metrics, methods, out):
    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8))
    axes = np.atleast_2d(axes)
    for col, method in enumerate(methods):
        for row, regime in enumerate(["balanced", "screening"]):
            cm = metrics[method][regime]["confusion_matrix"]
            mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]])
            ax = axes[row, col]
            ax.imshow(mat, cmap="Blues")
            for (i, j), v in np.ndenumerate(mat):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=14,
                        color="white" if v > mat.max() / 2 else "black")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred N", "Pred FCD"])
            ax.set_yticklabels(["True N", "True FCD"])
            ax.set_title(f"{method}\n{regime} (thr="
                         f"{metrics[method][regime]['threshold']:.3f})", fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_calibration(y, ensembles, out):
    plt.figure(figsize=(7, 7))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, p in ensembles.items():
        pt, pp = calibration_curve(y, p, n_bins=N_BINS_CAL, strategy="quantile")
        plt.plot(pp, pt, marker="o", lw=2, label=name)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed FCD fraction")
    plt.title("OOF calibration - ensemble methods")
    plt.legend(loc="upper left"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Step 5 - ensemble on OOF probs")
    ap.add_argument("--root", default=None,
                    help="path to fcd_project (auto-detected if omitted)")
    ap.add_argument("--models", default=None,
                    help="comma-separated model folder names; default is the "
                         "4-model ensemble reported in the paper")
    ap.add_argument("--out-name", default="ensemble",
                    help="subfolder under results/ (use a new name when adding "
                         "a model so the paper's ensemble is not overwritten)")
    args = ap.parse_args()

    model_list = ([m.strip() for m in args.models.split(",")] if args.models
                  else list(DEFAULT_MODELS))

    root = find_project_root(args.root)
    out_dir = root / "results" / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Project root : {root}")
    print(f"Output dir   : {out_dir}")
    print(f"Models       : {', '.join(model_list)}\n")

    print("Loading OOF arrays:")
    P, y, names, subj_ids = load_oof(root, model_list)

    indiv_auc = np.array([roc_auc_score(y, P[i]) for i in range(P.shape[0])])
    indiv_probs = {n: P[i] for i, n in enumerate(names)}
    print("Per-model OOF AUC (recomputed on the aligned subjects):")
    for n, a in zip(names, indiv_auc):
        print(f"  {n:18s} {a:.4f}")
    print()

    soft = soft_voting(P)
    weighted, weights = weighted_voting(P, indiv_auc, WEIGHT_MODE)
    stacked, final_lr, final_C, chosen_Cs = stacking_nested_cv(P, y)

    ensemble_probs = {"soft_voting": soft, "weighted_voting": weighted,
                      "stacking": stacked}

    hard_result, hard_pred, hard_vote_count = hard_voting_eval(P, y, names, soft)

    metrics = {
        "config": {
            "models": names, "n_subjects": int(len(y)),
            "screening_target_sensitivity": SCREENING_SENS,
            "weight_mode": WEIGHT_MODE, "random_state": RANDOM_STATE,
            "outer_folds": N_OUTER_FOLDS, "inner_folds": N_INNER_FOLDS,
            "alignment": "by subject ID (oof_subj_ids.npy)",
        },
        "individual_models": {n: evaluate(y, P[i]) for i, n in enumerate(names)},
        "ensembles": {name: evaluate(y, p) for name, p in ensemble_probs.items()},
        "hard_voting": hard_result,
    }

    for name, p in ensemble_probs.items():
        np.save(out_dir / f"{name}_oof_probs.npy", p)
    np.save(out_dir / "hard_voting_decision.npy", hard_pred)
    np.save(out_dir / "hard_voting_vote_count.npy", hard_vote_count)
    # Subject order for every array in this folder, so downstream scripts can
    # align instead of assuming a row order.
    np.save(out_dir / "ensemble_subj_ids.npy", subj_ids)
    np.save(out_dir / "ensemble_oof_labels.npy", y)

    with open(out_dir / "ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    meta = {
        "models": names,
        "weighted_voting_weights": {n: round(float(w), 4)
                                    for n, w in zip(names, weights)},
        "stacking_feature_order": names,
        "stacking_final_C": final_C,
        "stacking_outer_fold_C": chosen_Cs,
        "stacking_coefficients": {n: round(float(c), 4)
                                  for n, c in zip(names, final_lr.coef_[0])},
        "stacking_intercept": round(float(final_lr.intercept_[0]), 4),
        "thresholds": {
            name: {"balanced": metrics["ensembles"][name]["balanced"]["threshold"],
                   "screening": metrics["ensembles"][name]["screening"]["threshold"]}
            for name in ensemble_probs},
    }
    with open(out_dir / "meta_learner.json", "w") as f:
        json.dump(meta, f, indent=2)
    joblib.dump(final_lr, out_dir / "stacking_model.joblib")

    plot_roc(y, indiv_probs, ensemble_probs, out_dir / "roc_all.png",
             hard_point=(hard_result["sensitivity"], hard_result["specificity"]))
    plot_confusion_matrices(metrics["ensembles"], list(ensemble_probs),
                            out_dir / "confusion_matrices.png")
    plot_calibration(y, ensemble_probs, out_dir / "calibration_curves.png")

    print("=" * 64)
    print(f"{'method':20s} {'OOF AUC':>8s} {'bal.Se':>7s} {'bal.Sp':>7s} "
          f"{'scr.Se':>7s} {'scr.Sp':>7s}")
    print("-" * 64)
    for n in names:
        e = metrics["individual_models"][n]
        print(f"{n:20s} {e['oof_auc']:8.3f} {e['balanced']['sensitivity']:7.3f} "
              f"{e['balanced']['specificity']:7.3f} "
              f"{e['screening']['sensitivity']:7.3f} "
              f"{e['screening']['specificity']:7.3f}")
    print("-" * 64)
    for n in ensemble_probs:
        e = metrics["ensembles"][n]
        print(f"{n:20s} {e['oof_auc']:8.3f} {e['balanced']['sensitivity']:7.3f} "
              f"{e['balanced']['specificity']:7.3f} "
              f"{e['screening']['sensitivity']:7.3f} "
              f"{e['screening']['specificity']:7.3f}")
    print("=" * 64)
    best = max(ensemble_probs, key=lambda k: metrics["ensembles"][k]["oof_auc"])
    print(f"Best ensemble by OOF AUC: {best} "
          f"({metrics['ensembles'][best]['oof_auc']:.3f})")
    best_single = names[int(np.argmax(indiv_auc))]
    print(f"Best single model       : {best_single} ({indiv_auc.max():.3f})")
    if metrics["ensembles"][best]["oof_auc"] <= indiv_auc.max():
        print("  NOTE: no ensemble beat the best single model on OOF AUC. "
              "Test a DeLong comparison before claiming an ensemble benefit.")

    h = hard_result
    print("-" * 64)
    print("Hard (majority) voting - single operating point:")
    print(f"  sensitivity={h['sensitivity']:.3f}  specificity={h['specificity']:.3f}"
          f"  F1={h['f1']:.3f}  accuracy={h['accuracy']:.3f}")
    print(f"  ordinal vote-count AUC={h['ordinal_vote_auc']:.3f} (coarse, "
          f"{len(names) + 1} levels - not comparable to prob AUC)")
    print(f"  tie subjects resolved by soft-voting fallback: {h['n_tie_subjects']}")
    print(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()