"""Build per-dataset manifests from raw dataset directories.

Datasets handled:
  - CEDAR     dataset/cedar/
  - BHSig260  dataset/BHSig260/
  - SigResize dataset/SigResize/


Output: dataset/manifests/<name>.csv and dataset/manifest.csv (merged).

Usage:
    python -m preprocessing.build_manifests
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

DATASET = Path("dataset")
MANIFESTS = DATASET / "manifests"
FIELDS = ["image_path", "signer_id", "is_forgery", "split", "dataset"]


def split_signers(signers, seed=42):
    shuffled = sorted(signers)
    random.Random(seed).shuffle(shuffled)
    n_val = n_test = max(1, round(len(shuffled) * 0.15))
    labels = ["test"] * n_test + ["val"] * n_val
    labels += ["train"] * (len(shuffled) - len(labels))
    return dict(zip(shuffled, labels))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def build_cedar():
    root = DATASET / "cedar"
    rows = []
    pat = re.compile(r"(?:original|forgeries)_(\d+)_\d+\.png$", re.IGNORECASE)
    for split_name, is_forgery in [("full_org", False), ("full_forg", True)]:
        d = root / split_name
        if not d.is_dir(): continue
        for p in sorted(d.glob("*.png")):
            m = pat.match(p.name)
            if not m: continue
            sid = f"CEDAR_{m.group(1)}"
            rows.append({"image_path": p.relative_to(DATASET).as_posix(),
                         "signer_id": sid, "is_forgery": str(is_forgery).lower(),
                         "dataset": "CEDAR"})
    splits = split_signers(sorted({r["signer_id"] for r in rows}))
    rows = [{**r, "split": splits[r["signer_id"]]} for r in rows]
    write(MANIFESTS / "cedar.csv", rows)


def build_bhsig260():
    root = DATASET / "BHSig260"
    pat = re.compile(r"^([BH])-S-(\d+)-([GF])-\d+\.tif$", re.IGNORECASE)
    lang = {"B": "bn", "H": "hi"}
    rows = []
    for dd in sorted(root.iterdir()):
        if not dd.is_dir(): continue
        for p in sorted(dd.rglob("*.tif")):
            m = pat.match(p.name)
            if not m: raise ValueError(f"unexpected file: {p}")
            lang_code, num, kind = m.groups()
            is_f = kind.upper() == "F"
            sid = f"bhsig_{lang[lang_code.upper()]}_{int(num):03d}"
            rows.append({"image_path": p.relative_to(DATASET).as_posix(),
                         "signer_id": sid, "is_forgery": str(is_f).lower(),
                         "dataset": "BHSig260"})
    splits = split_signers(sorted({r["signer_id"] for r in rows}))
    rows = [{**r, "split": splits[r["signer_id"]]} for r in rows]
    write(MANIFESTS / "bhsig260.csv", rows)


def build_sigresize():
    root = DATASET / "SigResize"
    rows = []
    for split_name, is_f in [("Genuine", False), ("Forgery", True)]:
        d = root / split_name
        if not d.is_dir(): continue
        for sd in sorted(d.iterdir()):
            if not sd.is_dir(): continue
            sid = f"SigResize_{sd.name}"
            for p in sorted(sd.glob("*")):
                if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"): continue
                rows.append({"image_path": p.relative_to(DATASET).as_posix(),
                             "signer_id": sid, "is_forgery": str(is_f).lower(),
                             "dataset": "SigResize"})
    splits = split_signers(sorted({r["signer_id"] for r in rows}))
    rows = [{**r, "split": splits[r["signer_id"]]} for r in rows]
    write(MANIFESTS / "sigresize.csv", rows)


def build_thai():
    root = DATASET / "Thai "
    if not root.is_dir():
        root = DATASET / "Thai"
        if not root.is_dir():
            print("Warning: Thai dataset not found")
            return
    gd = root / "Genuine"
    if not gd.is_dir(): return
    rows = []
    for sd in sorted(gd.iterdir()):
        if not sd.is_dir(): continue
        sid = f"Thai_{sd.name}"
        for p in sorted(sd.glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"): continue
            rows.append({"image_path": p.relative_to(DATASET).as_posix(),
                         "signer_id": sid, "is_forgery": "false",
                         "dataset": "Thai"})
    if not rows: return
    splits = split_signers(sorted({r["signer_id"] for r in rows}))
    rows = [{**r, "split": splits[r["signer_id"]]} for r in rows]
    write(MANIFESTS / "thai.csv", rows)


def main():
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    build_cedar()
    build_bhsig260()
    build_sigresize()
    build_thai()
    merge()


if __name__ == "__main__":
    main()

def merge():
    merged = []
    for p in sorted(MANIFESTS.glob("*.csv")):
        with p.open(newline="", encoding="utf-8") as f:
            merged += list(csv.DictReader(f))
    write(DATASET / "manifest.csv", merged)
