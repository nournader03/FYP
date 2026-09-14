"""
Step 1: Download ds004199 from OpenNeuro and verify dataset integrity.

Usage:
    pip install openneuro-py pandas nibabel tqdm
    python step1_download_verify.py

Run from your project root. Dataset will be saved to:
    data/raw/ds004199/
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import pandas as pd
import nibabel as nib
from tqdm import tqdm

# CONFIG
DATASET_ID      = "ds004199"
DATASET_VERSION = "1.0.6"
RAW_DIR         = Path("data/raw/ds004199")
RESULTS_DIR     = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. DOWNLOAD
def download_dataset():
    """Download ds004199 via openneuro-py. Skips already-downloaded files."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Downloading {DATASET_ID} v{DATASET_VERSION}")
    print(f"Target: {RAW_DIR.resolve()}")
    print("=" * 60)

    # Check if openneuro is installed
    try:
        import openneuro
    except ImportError:
        print("openneuro-py not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
        import openneuro

    # openneuro-py API - newer versions removed the `version` kwarg.
    # Try new signature first, fall back to legacy signature.
    import inspect
    sig = inspect.signature(openneuro.download)
    kwargs = dict(
        dataset=DATASET_ID,
        target_dir=str(RAW_DIR),
        include=["**/anat/*FLAIR*", "participants.tsv", "participants.json",
                 "dataset_description.json", "README"],
    )
    if "version" in sig.parameters:
        kwargs["version"] = DATASET_VERSION   # legacy openneuro-py

    openneuro.download(**kwargs)
    print("\nDownload complete.\n")


# FALLBACK: AWS S3 (no auth, works on Windows without datalad)
def download_via_datalad():
    """
    Fallback using AWS S3 - no datalad or git-annex needed.
    Requires: pip install awscli  OR  use the curl loop below.
    OpenNeuro datasets are publicly hosted on S3 with no authentication.
    """
    print("openneuro-py failed. Trying AWS S3 fallback (no auth required)...")

    # Try awscli first
    ret = os.system("aws --version")
    if ret == 0:
        print("Using AWS CLI...")
        cmd = (
            f"aws s3 sync --no-sign-request "
            f"s3://openneuro.org/{DATASET_ID} {RAW_DIR} "
            f"--exclude \"*\" "
            f"--include \"*/anat/*FLAIR*\" "
            f"--include \"participants.tsv\" "
            f"--include \"dataset_description.json\""
        )
        print(f"Running: {cmd}")
        ret = os.system(cmd)
        if ret == 0:
            print("AWS S3 download complete.\n")
            return

    # If awscli not available, print manual instructions and exit cleanly
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("Both openneuro-py and AWS CLI failed.")
    print("Please download manually using ONE of these options:\n")
    print("Option 1 - Install AWS CLI (recommended for Windows):")
    print("  https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-windows.html")
    print("  Then run:")
    print(f"  aws s3 sync --no-sign-request s3://openneuro.org/{DATASET_ID} {RAW_DIR} "
          f"--exclude \"*\" --include \"*/anat/*FLAIR*\" "
          f"--include \"participants.tsv\" --include \"dataset_description.json\"\n")
    print("Option 2 - Download directly from browser:")
    print(f"  https://openneuro.org/datasets/{DATASET_ID}/versions/{DATASET_VERSION}")
    print("  Click 'Download' -> select FLAIR files only\n")
    print("Option 3 - pip install datalad (Windows build available):")
    print("  pip install datalad")
    print("  Then install git-annex from: https://git-annex.branchable.com/install/Windows/\n")
    print(f"Once downloaded, place files at: {RAW_DIR.resolve()}")
    print("Then re-run this script - it will skip download and go straight to verification.")
    sys.exit(1)


# 2. VERIFY BIDS STRUCTURE
def verify_bids_structure():
    """Check expected BIDS top-level files and subject folders."""
    print("=" * 60)
    print("Verifying BIDS structure")
    print("=" * 60)

    required_top = ["participants.tsv", "dataset_description.json"]
    for fname in required_top:
        fpath = RAW_DIR / fname
        status = "[OK]" if fpath.exists() else "[X] MISSING"
        print(f"  {status}  {fname}")

    # Count subject folders
    subject_dirs = sorted([d for d in RAW_DIR.iterdir()
                           if d.is_dir() and d.name.startswith("sub-")])
    print(f"\n  Found {len(subject_dirs)} subject directories")

    # IDs are numeric (sub-00001 etc) - read group from participants.tsv
    tsv_path = RAW_DIR / "participants.tsv"
    control_dirs, fcd_dirs = [], []
    if tsv_path.exists():
        df_tsv = pd.read_csv(tsv_path, sep="\t")
        df_tsv["participant_id"] = df_tsv["participant_id"].apply(
            lambda x: x if str(x).startswith("sub-") else f"sub-{x}"
        )
        subj_id_set = {d.name for d in subject_dirs}
        for _, row in df_tsv.iterrows():
            if row["participant_id"] not in subj_id_set:
                continue
            grp = str(row.get("group", "")).strip().lower()
            if grp in ("fcd", "patient", "p", "1"):
                fcd_dirs.append(RAW_DIR / row["participant_id"])
            elif grp in ("control", "ctrl", "healthy", "hc", "c", "0"):
                control_dirs.append(RAW_DIR / row["participant_id"])
    print(f"  Controls : {len(control_dirs)}")
    print(f"  FCD      : {len(fcd_dirs)}")

    if len(subject_dirs) != 170:
        print(f"\n  WARNING: expected 170 subjects, found {len(subject_dirs)}")
    else:
        print(f"\n  [OK] Correct subject count: 170")

    return subject_dirs, control_dirs, fcd_dirs


# 3. VERIFY FLAIR FILES
def verify_flair_files(subject_dirs):
    """Confirm FLAIR .nii.gz exists for every subject. Report missing."""
    print("\n" + "=" * 60)
    print("Checking FLAIR files for all 170 subjects")
    print("=" * 60)

    present  = []
    missing  = []
    warnings = []

    for subj_dir in tqdm(subject_dirs, desc="Scanning subjects"):
        subj_id  = subj_dir.name          # e.g. sub-C001
        anat_dir = subj_dir / "anat"

        if not anat_dir.exists():
            missing.append((subj_id, "anat/ directory missing"))
            continue

        # Flexible match - some datasets use _FLAIR, some use _acq-..._FLAIR
        flair_files = list(anat_dir.glob(f"{subj_id}*FLAIR*.nii.gz"))

        if not flair_files:
            # Try uncompressed
            flair_files = list(anat_dir.glob(f"{subj_id}*FLAIR*.nii"))

        if not flair_files:
            missing.append((subj_id, "No FLAIR file found"))
        elif len(flair_files) > 1:
            warnings.append((subj_id, f"{len(flair_files)} FLAIR files found: {[f.name for f in flair_files]}"))
            present.append((subj_id, flair_files[0]))
        else:
            present.append((subj_id, flair_files[0]))

    print(f"\n   FLAIR found   : {len(present)} subjects")
    print(f"  FLAIR missing : {len(missing)} subjects")
    if warnings:
        print(f"  Warnings      : {len(warnings)} subjects (multiple FLAIR files)")

    if missing:
        print("\n  Missing FLAIR files:")
        for subj_id, reason in missing:
            print(f"    {subj_id}: {reason}")

    if warnings:
        print("\n  Multi-FLAIR warnings (will use first match):")
        for subj_id, msg in warnings:
            print(f"    {subj_id}: {msg}")

    return present, missing


# 4. SPOT-CHECK NIFTI HEADERS
def spot_check_nifti(present, n_check=10):
    """
    Load n_check NIfTI headers to check:
    - Shape (should be 3D or 4D)
    - Voxel sizes (should be roughly 1x1x1 mm, common variation expected)
    - Data type
    """
    print("\n" + "=" * 60)
    print(f"Spot-checking {n_check} NIfTI headers")
    print("=" * 60)

    samples = present[:n_check]
    header_info = []

    for subj_id, fpath in tqdm(samples, desc="Loading headers"):
        img  = nib.load(str(fpath))
        hdr  = img.header
        shape = img.shape
        zooms = hdr.get_zooms()[:3]  # voxel sizes in mm
        dtype = hdr.get_data_dtype()

        info = {
            "subject": subj_id,
            "file": fpath.name,
            "shape": list(shape),
            "voxel_mm": [round(float(z), 3) for z in zooms],
            "dtype": str(dtype),
            "is_4D": len(shape) == 4,
        }
        header_info.append(info)
        print(f"  {subj_id}: shape={shape}  voxels={[round(float(z),2) for z in zooms]}mm  dtype={dtype}")

    shapes = set(tuple(i["shape"]) for i in header_info)
    if len(shapes) > 1:
        print(f"\n  Shape variation detected across subjects: {shapes}")
        print("    This is normal - preprocessing handles it via reorientation + slice selection.")
    else:
        print(f"\n  [OK] Uniform shape: {shapes}")

    return header_info


# --- 5. INSPECT PARTICIPANTS.TSV -----------------------------------------------
def inspect_participants(present):
    """
    Load participants.tsv, check for:
    - Site / scanner / manufacturer / field_strength columns
    - Age / sex distribution
    - Label distribution matching expected 85/85
    - Any NaN values
    """
    print("\n" + "=" * 60)
    print("Inspecting participants.tsv")
    print("=" * 60)

    tsv_path = RAW_DIR / "participants.tsv"
    if not tsv_path.exists():
        print("  [X] participants.tsv NOT FOUND - cannot check metadata")
        return None

    df = pd.read_csv(tsv_path, sep="\t")
    print(f"\n  Columns : {list(df.columns)}")
    print(f"  Rows    : {len(df)}")
    print(f"\n  First 5 rows:\n{df.head().to_string()}")

    # -- Check NaNs
    nan_cols = df.isnull().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols):
        print(f"\n  NaN values found:\n{nan_cols}")
    else:
        print(f"\n  [OK] No NaN values")

    # Site / scanner detection
    SITE_KEYWORDS = [
        "site", "scanner", "manufacturer", "field_strength",
        "magnetic_field_strength", "institution", "MagneticFieldStrength",
        "ManufacturersModelName", "DeviceSerialNumber"
    ]
    site_cols = [c for c in df.columns if any(k.lower() in c.lower() for k in SITE_KEYWORDS)]

    if site_cols:
        print(f"\n  [OK] Site/scanner columns found: {site_cols}")
        for col in site_cols:
            counts = df[col].value_counts()
            print(f"\n  {col}:\n{counts.to_string()}")
        multisite = any(df[c].nunique() > 1 for c in site_cols)
        if multisite:
            print("\n  MULTI-SITE dataset detected!")
            print("    -> Use site-stratified splits in master_split.json")
            print("    -> Pass site labels to StratifiedGroupKFold or custom splitter")
        else:
            print("\n  [OK] Single-site dataset")
    else:
        print(f"\n  INFO: No site/scanner columns found in participants.tsv")
        print("    -> Check dataset_description.json and individual JSON sidecars")
        print("    -> Will proceed with label-only stratified split")

    # -- Label distribution ---------------------------------------------------
    # Infer label from participant_id prefix: C=Control, P=FCD
    if "participant_id" in df.columns:
        df["inferred_label"] = df["participant_id"].apply(
            lambda x: "FCD" if str(x).replace("sub-", "").startswith("P") else "Control"
        )
        print(f"\n  Label distribution (inferred from ID prefix):")
        print(df["inferred_label"].value_counts().to_string())

    # -- Age / sex -----------------------------------------------------------
    for col in ["age", "sex", "gender"]:
        matches = [c for c in df.columns if col in c.lower()]
        for m in matches:
            print(f"\n  {m} distribution:")
            if df[m].dtype in ["float64", "int64"]:
                print(f"    mean={df[m].mean():.1f}  std={df[m].std():.1f}  "
                      f"min={df[m].min()}  max={df[m].max()}")
            else:
                print(df[m].value_counts().to_string())

    # -- Save metadata summary ------------------------------------------------
    summary = {
        "n_subjects": len(df),
        "columns": list(df.columns),
        "site_columns": site_cols,
        "multisite": bool(site_cols and any(df[c].nunique() > 1 for c in site_cols)),
        "nan_columns": nan_cols.to_dict() if len(nan_cols) else {},
        "label_counts": df["inferred_label"].value_counts().to_dict()
                        if "inferred_label" in df.columns else {},
    }
    out_path = RESULTS_DIR / "participants_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary -> {out_path}")

    return df, summary


# --- 6. SAVE SUBJECT MAP -------------------------------------------------------
def save_subject_map(present, missing, df_participants=None):
    """
    Save subject_id -> FLAIR path mapping for use in preprocessing.

    Labels sourced from participants.tsv `group` column (ground truth).
    Falls back to ID-prefix inference only if tsv unavailable - with a warning.
    Also stores hemisphere + lobe for FCD subjects (used for Grad-CAM validation).
    """
    print("\n" + "=" * 60)
    print("Building subject_map.json")
    print("=" * 60)

    # -- Build lookup from participants.tsv -----------------------------------
    tsv_lookup = {}
    if df_participants is not None:
        # Normalise participant_id to always have sub- prefix
        df = df_participants.copy()
        df["participant_id"] = df["participant_id"].apply(
            lambda x: x if str(x).startswith("sub-") else f"sub-{x}"
        )

        # Normalise group column -> "FCD" or "Control"
        if "group" in df.columns:
            def normalise_group(val):
                v = str(val).strip().lower()
                if v in ("fcd", "patient", "p", "1"):
                    return "FCD"
                elif v in ("control", "ctrl", "healthy", "c", "0"):
                    return "Control"
                else:
                    return None   # unexpected value - will warn below

            df["label_clean"] = df["group"].apply(normalise_group)

            bad = df[df["label_clean"].isna()]
            if len(bad):
                print(f"  WARNING: Unrecognised group values for: {bad['participant_id'].tolist()}")
                print(f"    Raw values: {bad['group'].tolist()}")
                print("    These subjects will fall back to ID-prefix inference.")

            for _, row in df.iterrows():
                subj_id = row["participant_id"]
                entry = {"label_source": "participants.tsv:group"}

                # Label
                if row["label_clean"] is not None:
                    entry["label"] = row["label_clean"]
                    entry["label_int"] = 1 if row["label_clean"] == "FCD" else 0
                else:
                    # fallback for this subject
                    fb = "FCD" if subj_id.replace("sub-", "").startswith("P") else "Control"
                    entry["label"] = fb
                    entry["label_int"] = 1 if fb == "FCD" else 0
                    entry["label_source"] = "id_prefix_fallback"

                # hemisphere + lobe (FCD subjects only - NaN for controls)
                for col in ["hemisphere", "lobe"]:
                    if col in df.columns:
                        val = row.get(col, None)
                        entry[col] = None if pd.isna(val) else str(val)

                # sex + age_scan for reporting
                for col in ["sex", "age_scan"]:
                    if col in df.columns:
                        val = row.get(col, None)
                        entry[col] = None if (isinstance(val, float) and pd.isna(val)) else val

                tsv_lookup[subj_id] = entry
        else:
            print("   `group` column not found in participants.tsv - falling back to ID prefix")
    else:
        print("  participants.tsv not loaded - falling back to ID prefix for all subjects")

    # -- Build final map ------------------------------------------------------
    subject_map = {}
    label_source_counts = {"participants.tsv:group": 0, "id_prefix_fallback": 0}

    for subj_id, fpath in present:
        if subj_id in tsv_lookup:
            entry = tsv_lookup[subj_id].copy()
            entry["flair_path"] = str(fpath)
        else:
            # Subject in BIDS but not in participants.tsv - use prefix fallback
            fb = "FCD" if subj_id.replace("sub-", "").startswith("P") else "Control"
            entry = {
                "flair_path": str(fpath),
                "label": fb,
                "label_int": 1 if fb == "FCD" else 0,
                "label_source": "id_prefix_fallback",
                "hemisphere": None,
                "lobe": None,
            }
            print(f"  WARNING: {subj_id} not in participants.tsv - label inferred from ID prefix")

        subject_map[subj_id] = entry
        src = entry.get("label_source", "id_prefix_fallback")
        label_source_counts[src] = label_source_counts.get(src, 0) + 1

    # -- Sanity check label counts --------------------------------------------
    fcd_count     = sum(1 for v in subject_map.values() if v["label"] == "FCD")
    control_count = sum(1 for v in subject_map.values() if v["label"] == "Control")
    print(f"\n  Label counts from participants.tsv:group:")
    print(f"    FCD     : {fcd_count}")
    print(f"    Control : {control_count}")
    print(f"  Label sources: {label_source_counts}")

    if fcd_count != 85 or control_count != 85:
        print(f"  Expected 85 FCD + 85 Control - got {fcd_count} FCD + {control_count} Control")
        print("    Check participants.tsv group column values manually.")
    else:
        print("  [OK] Label counts correct: 85 FCD, 85 Control")

    # -- Save ----------------------------------------------------------------
    out_path = Path("data/processed/subject_map.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(subject_map, f, indent=2)

    print(f"\n  [OK] Saved subject_map.json -> {out_path}")
    print(f"  {len(subject_map)} subjects mapped")
    if missing:
        print(f"  {len(missing)} subjects excluded (no FLAIR found)")

    return subject_map


# MAIN
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FCD PROJECT - STEP 1: DOWNLOAD & VERIFY")
    print("=" * 60 + "\n")

    # Download
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"Dataset directory already exists at {RAW_DIR}")
        print("Skipping download. Delete the directory to re-download.\n")
    else:
        try:
            download_dataset()
        except Exception as e:
            print(f"openneuro-py download failed: {e}")
            print("Trying datalad fallback...")
            download_via_datalad()

    # Verify
    subject_dirs, control_dirs, fcd_dirs = verify_bids_structure()
    present, missing = verify_flair_files(subject_dirs)
    header_info = spot_check_nifti(present, n_check=min(10, len(present)))
    df_result = inspect_participants(present)

    # Pass participants dataframe so labels come from `group` column, not ID prefix
    df_participants = df_result[0] if df_result is not None else None
    subject_map = save_subject_map(present, missing, df_participants=df_participants)

    # Final summary
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"  Total subjects found   : {len(subject_dirs)}")
    print(f"  Controls               : {len(control_dirs)}")
    print(f"  FCD patients           : {len(fcd_dirs)}")
    print(f"  FLAIR files present    : {len(present)}")
    print(f"  FLAIR files missing    : {len(missing)}")

    if missing:
        print("\n   Some subjects are missing FLAIR files.")
        print("    These will be excluded from training. Investigate manually.")
    else:
        print("\n  [OK] All subjects have FLAIR files. Ready for preprocessing.")

    if df_result is not None:
        _, summary = df_result
        if summary.get("multisite"):
            print("\n  MULTI-SITE detected -> use site-stratified splitting in Step 3")
        else:
            print("\n  [OK] Single site -> label-stratified splitting in Step 3")

    print("\n  Next step: run step2_preprocess.py")
    print("=" * 60 + "\n")