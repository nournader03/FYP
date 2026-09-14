"""
Step 3: Master Train/Val/Test Split for FCD Detection DL Project
=================================================================
Creates and saves master_split.json - NEVER deleted or regenerated after creation.

Split strategy:
  - 80% (136 subjects) -> Train+Val pool -> 5-fold stratified CV
  - 20% (34 subjects)  -> Locked test set (used exactly once at the end)
  - Stratified by label (FCD/Control)
  - Stratified by site if multi-site data present
  - All splits at SUBJECT level - never slice level
  - Zero subject overlap guaranteed between train and val in every fold

Usage:
    pip install scikit-learn numpy pandas
    python step3_split.py
"""

import json
import random
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# --- CONFIG --------------------------------------------------------------------
SUBJECT_MAP_PATH  = Path("data/processed/subject_map.json")
SPLIT_OUTPUT_PATH = Path("data/processed/master_split.json")
RESULTS_DIR       = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS     = 5
TEST_RATIO  = 0.20   # 20% -> 34 subjects locked test set
RANDOM_SEED = 42


# --- LOAD SUBJECT MAP ----------------------------------------------------------
def load_subjects(subject_map: dict) -> pd.DataFrame:
    """
    Convert subject_map.json into a DataFrame.
    Adds site column - 'openneuro_ds004199' for all current subjects.
    When clinical data arrives, those entries will have site='clinical'.
    """
    rows = []
    for subj_id, info in subject_map.items():
        rows.append({
            "subj_id":   subj_id,
            "label":     info["label"],
            "label_int": info["label_int"],
            # site field - future-proofed for clinical data integration
            "site":      info.get("site", "openneuro_ds004199"),
            "n_slices":  info.get("n_slices_saved", 0),
        })
    return pd.DataFrame(rows)


# --- STRATIFICATION KEY --------------------------------------------------------
def make_strat_key(df: pd.DataFrame) -> np.ndarray:
    """
    Create stratification key combining label + site.
    Single-site: key = label only (2 classes)
    Multi-site:  key = label_site (4 classes for 2 labels x 2 sites)
    This ensures every fold has balanced label AND site distribution.
    """
    sites = df["site"].unique()
    if len(sites) > 1:
        print(f"  Multi-site detected: {list(sites)}")
        print("  Stratifying by label + site")
        return (df["label"] + "_" + df["site"]).values
    else:
        print(f"  Single site: {sites[0]}")
        print("  Stratifying by label only")
        return df["label"].values


# --- VERIFY ZERO OVERLAP -------------------------------------------------------
def assert_no_overlap(folds: list, test_subjects: list):
    """Hard assert: no subject appears in both train and val within any fold,
    and no subject from test set appears in any fold."""
    test_set = set(test_subjects)

    for i, fold in enumerate(folds):
        train_set = set(fold["train"])
        val_set   = set(fold["val"])

        # Train/val overlap within fold
        overlap = train_set & val_set
        assert len(overlap) == 0, \
            f"Fold {i+1}: train/val overlap detected: {overlap}"

        # Test leakage into train
        test_in_train = train_set & test_set
        assert len(test_in_train) == 0, \
            f"Fold {i+1}: test subjects in train: {test_in_train}"

        # Test leakage into val
        test_in_val = val_set & test_set
        assert len(test_in_val) == 0, \
            f"Fold {i+1}: test subjects in val: {test_in_val}"

    print("  [OK] Zero overlap verified across all folds and test set")


# --- COMPUTE SPLIT HASH --------------------------------------------------------
def compute_split_hash(folds: list, test_subjects: list) -> str:
    """
    Deterministic hash of the split for reproducibility verification.
    If this hash changes between runs, the split changed - which must never happen.
    """
    content = json.dumps({
        "folds": folds,
        "test":  sorted(test_subjects)
    }, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


# --- PRINT FOLD SUMMARY --------------------------------------------------------
def print_fold_summary(folds: list, df: pd.DataFrame, test_subjects: list):
    label_map = df.set_index("subj_id")["label"].to_dict()
    site_map  = df.set_index("subj_id")["site"].to_dict()

    print(f"\n  {'Fold':<6} {'Train':>6} {'Val':>6} "
          f"{'Tr-FCD':>8} {'Tr-Ctrl':>8} {'Val-FCD':>7} {'Val-Ctrl':>9}")
    print("  " + "-" * 60)

    for i, fold in enumerate(folds):
        tr_labels  = [label_map[s] for s in fold["train"]]
        val_labels = [label_map[s] for s in fold["val"]]
        tr_fcd     = tr_labels.count("FCD")
        tr_ctrl    = tr_labels.count("Control")
        val_fcd    = val_labels.count("FCD")
        val_ctrl   = val_labels.count("Control")
        print(f"  {i+1:<6} {len(fold['train']):>6} {len(fold['val']):>6} "
              f"{tr_fcd:>8} {tr_ctrl:>8} {val_fcd:>7} {val_ctrl:>9}")

    # Test set summary
    test_labels = [label_map[s] for s in test_subjects]
    test_sites  = [site_map[s]  for s in test_subjects]
    print(f"\n  Test set: {len(test_subjects)} subjects")
    print(f"    FCD     : {test_labels.count('FCD')}")
    print(f"    Control : {test_labels.count('Control')}")
    if len(set(test_sites)) > 1:
        print(f"    Sites   : {dict(Counter(test_sites))}")


# --- MAIN ----------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FCD PROJECT - STEP 3: MASTER SPLIT")
    print("=" * 60 + "\n")

    # -- Safety check - never regenerate if exists ---------------------------
    if SPLIT_OUTPUT_PATH.exists():
        print(f"  WARNING: master_split.json already exists at {SPLIT_OUTPUT_PATH}")
        print("  Loading and verifying existing split...\n")

        with open(SPLIT_OUTPUT_PATH) as f:
            existing = json.load(f)

        folds         = existing["folds"]
        test_subjects = existing["test_subjects"]
        saved_hash    = existing.get("split_hash", "")

        recomputed_hash = compute_split_hash(folds, test_subjects)
        if saved_hash == recomputed_hash:
            print(f"  [OK] Split hash verified: {saved_hash}")
        else:
            print(f"  [X] Hash mismatch - split file may be corrupted!")
            print(f"    Saved   : {saved_hash}")
            print(f"    Current : {recomputed_hash}")

        with open(SUBJECT_MAP_PATH) as f:
            subject_map = json.load(f)
        df = load_subjects(subject_map)
        print_fold_summary(folds, df, test_subjects)
        print("\n  Split already exists - delete master_split.json to regenerate.")
        print("  (Only do this if you are changing the dataset entirely.)")
        print("=" * 60 + "\n")
        exit(0)

    # -- Load subjects --------------------------------------------------------
    with open(SUBJECT_MAP_PATH) as f:
        subject_map = json.load(f)

    df = load_subjects(subject_map)
    print(f"  Loaded {len(df)} subjects")
    print(f"    FCD     : {(df['label'] == 'FCD').sum()}")
    print(f"    Control : {(df['label'] == 'Control').sum()}")
    print(f"    Sites   : {df['site'].value_counts().to_dict()}\n")

    # -- Filter out subjects with zero slices ---------------------------------
    zero_slice = df[df["n_slices"] == 0]
    if len(zero_slice):
        print(f"  WARNING: Excluding {len(zero_slice)} subjects with 0 slices:")
        for _, row in zero_slice.iterrows():
            print(f"    {row['subj_id']}")
        df = df[df["n_slices"] > 0].reset_index(drop=True)
        print()

    # -- Stratification key ---------------------------------------------------
    strat_key = make_strat_key(df)

    # -- 80/20 train+val / test split -----------------------------------------
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    all_ids = df["subj_id"].values

    trainval_ids, test_ids = train_test_split(
        all_ids,
        test_size=TEST_RATIO,
        stratify=strat_key,
        random_state=RANDOM_SEED,
    )

    test_subjects = sorted(test_ids.tolist())
    print(f"\n  Train+Val pool : {len(trainval_ids)} subjects")
    print(f"  Test set       : {len(test_subjects)} subjects (locked)")

    # -- 5-fold CV on train+val pool ------------------------------------------
    trainval_df   = df[df["subj_id"].isin(trainval_ids)].reset_index(drop=True)
    trainval_strat = make_strat_key(trainval_df)

    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(trainval_df["subj_id"], trainval_strat)):

        train_subjects = sorted(trainval_df.iloc[train_idx]["subj_id"].tolist())
        val_subjects   = sorted(trainval_df.iloc[val_idx]["subj_id"].tolist())

        folds.append({
            "fold":  fold_idx + 1,
            "train": train_subjects,
            "val":   val_subjects,
        })

    # -- Verify zero overlap --------------------------------------------------
    print(f"\n  Verifying split integrity...")
    assert_no_overlap(folds, test_subjects)

    # -- Print summary --------------------------------------------------------
    print(f"\n  5-Fold CV Summary:")
    print_fold_summary(folds, df, test_subjects)

    # -- Compute hash ---------------------------------------------------------
    split_hash = compute_split_hash(folds, test_subjects)
    print(f"\n  Split hash (MD5): {split_hash}")
    print("  Save this hash - if it ever changes, the split has been modified.")

    # -- Save master_split.json -----------------------------------------------
    master_split = {
        "description":   "Master train/val/test split for FCD DL project. NEVER delete or regenerate.",
        "dataset":       "OpenNeuro ds004199 v1.0.6",
        "n_subjects":    len(df),
        "n_folds":       N_FOLDS,
        "test_ratio":    TEST_RATIO,
        "random_seed":   RANDOM_SEED,
        "split_hash":    split_hash,
        "stratified_by": "label+site",
        "test_subjects": test_subjects,
        "folds":         folds,
    }

    with open(SPLIT_OUTPUT_PATH, "w") as f:
        json.dump(master_split, f, indent=2)

    print(f"\n  [OK] Saved master_split.json -> {SPLIT_OUTPUT_PATH}")
    print("  WARNING: DO NOT delete or regenerate this file.")
    print("    All training runs must use this exact split for reproducibility.")

    # -- Enrich subject_map with split assignment -----------------------------
    # Tag every subject with their split role for easy lookup
    test_set = set(test_subjects)
    for subj_id in subject_map:
        if subj_id in test_set:
            subject_map[subj_id]["split"] = "test"
        else:
            subject_map[subj_id]["split"] = "trainval"
        # Also add site field if missing (future-proofs for clinical integration)
        if "site" not in subject_map[subj_id]:
            subject_map[subj_id]["site"] = "openneuro_ds004199"

    with open(SUBJECT_MAP_PATH, "w") as f:
        json.dump(subject_map, f, indent=2)
    print(f"  [OK] subject_map.json enriched with split assignments and site field")

    print("\n  Next step: run step4_train.py")
    print("=" * 60 + "\n")