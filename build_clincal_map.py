"""
build_clinical_map.py
=====================
Builds the subject_map + split for the CLINICAL cohort, so step9 can preprocess
it and step8 can evaluate it - without touching any ds004199 file.

Handles messy real-world layouts:
  * the subject ID may appear ANYWHERE in the path, at any depth
        2162/2162/FLAIR.nii
        08408742/Mri_Brain_Contrast - 8441968/t2_spc_dafl_cor_7/IM-0002-0001.dcm
  * FLAIR series are recognised by vendor shorthand as well as the word "flair"
        t2_spc_dafl_cor   (Siemens SPACE dark-fluid)
        t2_tirm_tra_dark-fluid
  * axial series are preferred over coronal/sagittal when both exist, and the
    orientation is recorded so coronal subjects can be reported separately
  * subjects still in DICOM (or still zipped) are reported separately from
    genuinely missing ones, so you know what to convert

Outputs:
    data/clinical/processed/subject_map_clinical.json
    data/clinical/processed/master_split_clinical.json   (all subjects = test)
    data/clinical/processed/unresolved.json              (what needs attention)

Manual override - for any subject the auto-pick gets wrong, create
data/clinical/processed/clinical_overrides.json:
    { "08408742": "data/clinical/raw/converted/08408742/whatever_flair.nii.gz" }
and re-run. Overrides always win.

Run:
    python build_clinical_map.py
    python build_clinical_map.py --raw data/clinical/raw
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path

# -- cohort definition --------------------------------------------------------
FCD_NIFTI = ["2909", "3004", "3191", "3266", "3271", "3346", "3384", "2703",
             "2761", "2869", "2954", "3034", "3036", "3082", "3106", "3166",
             "3283", "3303", "3369", "3391"]
FCD_DICOM = ["01323709", "05982375", "06009565", "08408742", "12469715",
             "12787369", "13488098", "16213521", "18239307", "21149712",
             "23208805", "24693847", "27894878", "31650060", "31707516",
             "32190042", "32357276"]
NORMAL_NIFTI = ["2162", "2364", "2495", "2592", "2633", "2701", "2738", "2896",
                "2953", "2995", "3085", "3141", "3224", "3230", "3260", "3278",
                "3281", "3373", "3456", "3495"]

# known acquisition caveats (from the cohort sheet)
CORONAL_ONLY = {"3346", "2162", "2364", "2592", "2738"}
FEW_SLICES = {"2703": 17, "3303": 19, "2896": 19, "3224": 22, "3278": 22}

# -- series-name heuristics ---------------------------------------------------
FLAIR_HINTS = ("flair", "dafl", "darkfluid", "dark-fluid", "dark_fluid",
               "tirm", "spcir", "spc_ir")
NEG_HINTS = ("mprage", "mp-rage", "localizer", "localiser", "scout", "survey",
             "dwi", "adc", "swi", "bold", "perf", "asl", "dti", "angio",
             "mask", "roi", "seg", "label")
AXIAL_TOKENS = {"tra", "ax", "axi", "axial", "transverse"}
CORONAL_TOKENS = {"cor", "coronal"}
SAGITTAL_TOKENS = {"sag", "sagittal"}
NEG_TOKENS = {"t1", "pd", "ph", "e2", "i0"}


def tokenize(text: str) -> set:
    return set(t for t in re.split(r"[^a-z0-9]+", text.lower()) if t)


def is_nifti(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")


def orientation_of(path_text: str):
    tk = tokenize(path_text)
    if tk & AXIAL_TOKENS:
        return "axial"
    if tk & CORONAL_TOKENS:
        return "coronal"
    if tk & SAGITTAL_TOKENS:
        return "sagittal"
    return None


def score_candidate(rel_text: str) -> float:
    """Higher = more likely to be the axial FLAIR we want."""
    low = rel_text.lower()
    tk = tokenize(low)
    s = 0.0
    if any(h in low for h in FLAIR_HINTS):
        s += 100
    if any(h in low for h in NEG_HINTS):
        s -= 90
    if tk & NEG_TOKENS:
        s -= 40
    o = orientation_of(low)
    if o == "axial":
        s += 25
    elif o in ("coronal", "sagittal"):
        s -= 15          # still usable, just less preferred
    s -= len(low) / 1000.0     # tie-break: shorter/simpler path
    return s


def subject_pattern(subj: str) -> re.Pattern:
    """Match the ID as a standalone number anywhere in the path."""
    return re.compile(rf"(?<![0-9]){re.escape(subj)}(?![0-9])")


def scan(raw: Path):
    """Index the raw tree once: nifti files, dicom-bearing dirs, zip files."""
    niftis, dicom_dirs, zips = [], set(), []
    if not raw.is_dir():
        return niftis, dicom_dirs, zips
    for p in raw.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if is_nifti(p):
            niftis.append(p)
        elif n.endswith(".zip"):
            zips.append(p)
        elif n.endswith((".dcm", ".ima")):
            dicom_dirs.add(p.parent)
    return niftis, dicom_dirs, zips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="project root (contains data/)")
    ap.add_argument("--raw", default="data/clinical/raw")
    ap.add_argument("--out", default="data/clinical/processed")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    raw = (root / args.raw).resolve()
    out = (root / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    overrides = {}
    ov_path = out / "clinical_overrides.json"
    if ov_path.exists():
        overrides = json.load(open(ov_path))
        print(f"Loaded {len(overrides)} manual override(s) from {ov_path.name}\n")

    niftis, dicom_dirs, zips = scan(raw)
    print(f"Scanned {raw}\n  {len(niftis)} NIfTI files, "
          f"{len(dicom_dirs)} DICOM folders, {len(zips)} zip archives\n")

    cohort = ([(s, "FCD", 1) for s in FCD_NIFTI] +
              [(s, "FCD", 1) for s in FCD_DICOM] +
              [(s, "Control", 0) for s in NORMAL_NIFTI])

    smap = {}
    needs_conversion, still_zipped, missing, multi = [], [], [], []

    for subj, label, label_int in cohort:
        pat = subject_pattern(subj)

        # 1) manual override wins
        if subj in overrides:
            f = (root / overrides[subj]).resolve()
            if not f.exists():
                missing.append((subj, label,
                                f"override path not found: {overrides[subj]}"))
                continue
            chosen, why = f, "override"
        else:
            # 2) match NIfTI on the FULL relative path (any depth)
            cands = [p for p in niftis if pat.search(str(p.relative_to(raw)))]
            if cands:
                scored = sorted(((score_candidate(str(p.relative_to(raw))), p)
                                 for p in cands), key=lambda x: -x[0])
                chosen, why = scored[0][1], "auto"
                if len(scored) > 1:
                    multi.append((subj, [str(p.relative_to(raw))
                                         for _, p in scored[:4]]))
            else:
                # 3) not converted yet?
                dd = [d for d in dicom_dirs if pat.search(str(d))]
                zz = [z for z in zips if pat.search(str(z))]
                if dd:
                    needs_conversion.append((subj, label, str(dd[0])))
                elif zz:
                    still_zipped.append((subj, label, str(zz[0].name)))
                else:
                    missing.append((subj, label, "no NIfTI, DICOM or zip matched"))
                continue

        rel = str(chosen.relative_to(root)).replace("\\", "/")
        entry = {"label": label, "label_int": label_int, "flair_path": rel,
                 "cohort": "clinical", "selected_by": why}
        o = orientation_of(str(chosen.relative_to(raw)))
        if o:
            entry["orientation_hint"] = o
        if subj in CORONAL_ONLY:
            entry["acquisition_note"] = "coronal_only"
        elif o == "coronal":
            entry["acquisition_note"] = "coronal_series_name"
        if subj in FEW_SLICES:
            entry["acquisition_note"] = f"few_slices_{FEW_SLICES[subj]}"
        smap[subj] = entry

    # -- write ----------------------------------------------------------------
    map_path = out / "subject_map_clinical.json"
    split_path = out / "master_split_clinical.json"
    json.dump(smap, open(map_path, "w"), indent=2)
    json.dump({"folds": [], "test_subjects": sorted(smap)},
              open(split_path, "w"), indent=2)
    json.dump({"needs_dcm2niix": needs_conversion, "still_zipped": still_zipped,
               "missing": missing, "multiple_candidates": multi},
              open(out / "unresolved.json", "w"), indent=2)

    # -- report ---------------------------------------------------------------
    n_fcd = sum(1 for v in smap.values() if v["label"] == "FCD")
    n_ctl = sum(1 for v in smap.values() if v["label"] == "Control")
    print(f"RESOLVED {len(smap)}/{len(cohort)}  (FCD={n_fcd}, Control={n_ctl})\n")
    print(f"{'subject':10s} {'label':8s} {'orient':9s} file")
    print("-" * 92)
    for s in sorted(smap):
        v = smap[s]
        tag = "   [override]" if v["selected_by"] == "override" else ""
        print(f"{s:10s} {v['label']:8s} {v.get('orientation_hint','?'):9s} "
              f"{v['flair_path']}{tag}")

    if still_zipped:
        print(f"\nSTILL ZIPPED ({len(still_zipped)}) - extract these first:")
        for s, lab, z in still_zipped:
            print(f"  {s:10s} {lab:8s} {z}")
    if needs_conversion:
        print(f"\nNEEDS dcm2niix ({len(needs_conversion)}):")
        for s, lab, d in needs_conversion:
            print(f"  {s:10s} {lab:8s} {d}")
        print("\n  dcm2niix -z y -f %i_%p -o data/clinical/raw/converted "
              "data/clinical/raw/dicom")
    if missing:
        print(f"\nMISSING ({len(missing)}) - nothing matched at all:")
        for s, lab, why in missing:
            print(f"  {s:10s} {lab:8s} {why}")
    if multi:
        print(f"\nMULTIPLE CANDIDATES ({len(multi)}) - verify the top pick is the "
              f"axial FLAIR; if not, add it to clinical_overrides.json:")
        for s, opts in multi:
            print(f"  {s}:")
            for i, o in enumerate(opts):
                print(f"      {'-> ' if i == 0 else '   '}{o}")

    coronal = [s for s, v in smap.items()
               if v.get("orientation_hint") == "coronal"
               or v.get("acquisition_note", "").startswith("coronal")]
    if coronal:
        print(f"\nCORONAL ({len(coronal)}) - check their PNGs after preprocessing "
              f"and report separately: {', '.join(sorted(coronal))}")

    print(f"\nWrote:\n  {map_path}\n  {split_path}\n  {out/'unresolved.json'}")
    print("\nNothing under data/processed/ was touched.")


if __name__ == "__main__":
    main()