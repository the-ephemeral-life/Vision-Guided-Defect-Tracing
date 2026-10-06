#!/usr/bin/env python3
"""
Merge crack datasets into the layout train_segformer.py expects:
    out/{train,val}/images/*.jpg      out/{train,val}/masks/*.png  (0 / 255)

Two ways to point at data (mix freely, repeat flags):

  --auto name=ROOT        walk ROOT, auto-detect (images_dir, masks_dir) pairs by folder name + matching filenames
  --src  name=IMGS:MASKS  explicit dirs

  --dry_run               only print what was detected (do this first!)

Example:
  python prepare_data.py --dry_run --auto crackseg9k=/kaggle/working/crackseg9k_raw --auto deepcrack=/kaggle/input/concrete-crack-segmentation
"""
import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_WORDS = ("mask", "label", "lab", "gt", "groundtruth", "ground_truth", "annot", "seg", "target")
SUFFIX_RE = re.compile(r"[_\-\.](mask|gt|label|lab|seg|annot|annotation|groundtruth)$", re.I)


def norm_stem(stem):
    return SUFFIX_RE.sub("", stem).lower()


def is_maskish(dirname):
    d = dirname.lower()
    return any(w in d for w in MASK_WORDS)


def discover_pairs(root):
    """Return list of (img_dir, mask_dir) where filenames (after stripping mask suffixes) line up."""
    root = Path(root)
    dirs = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            dirs[p.parent].append(p)
    stems = {d: {norm_stem(p.stem) for p in ps} for d, ps in dirs.items()}
    pairs = []
    for m in dirs:
        if not is_maskish(m.name):
            continue
        best, best_ov = None, 0.0
        for i in dirs:
            if i == m or is_maskish(i.name):
                continue
            ov = len(stems[i] & stems[m]) / max(1, min(len(stems[i]), len(stems[m])))
            if ov > best_ov:
                best, best_ov = i, ov
        if best is not None and best_ov >= 0.5:
            pairs.append((best, m))
    return pairs


def pair_files(idir, mdir):
    mmap = {norm_stem(p.stem): p for p in Path(mdir).iterdir() if p.suffix.lower() in IMG_EXTS}
    out = []
    for ip in sorted(Path(idir).iterdir()):
        if ip.suffix.lower() in IMG_EXTS and norm_stem(ip.stem) in mmap:
            out.append((ip, mmap[norm_stem(ip.stem)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="append", default=[], help="name=ROOT")
    ap.add_argument("--src", action="append", default=[], help="name=images_dir:masks_dir")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_side", type=int, default=1024, help="downscale so longest side <= this (0 = off)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    assert args.auto or args.src, "give at least one --auto or --src"

    # name -> list of (img, mask)
    sources = {}
    for spec in args.auto:
        name, root = spec.split("=", 1)
        prs = discover_pairs(root)
        if not prs:
            print(f"[{name}] !! no (images, masks) pairs found under {root} -- use --src instead")
        files = []
        for idir, mdir in prs:
            fp = pair_files(idir, mdir)
            print(f"[{name}] {idir}  <->  {mdir}   ({len(fp)} pairs)")
            files += fp
        sources[name] = files
    for spec in args.src:
        name, paths = spec.split("=", 1)
        idir, mdir = paths.split(":", 1)
        fp = pair_files(idir, mdir)
        print(f"[{name}] {idir}  <->  {mdir}   ({len(fp)} pairs)")
        sources.setdefault(name, []).extend(fp)

    print("total pairs:", {k: len(v) for k, v in sources.items()})
    if args.dry_run:
        return

    random.seed(args.seed)
    out = Path(args.out)
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "masks").mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    for name, files in sources.items():
        files = list(files)
        random.shuffle(files)
        n_val = int(len(files) * args.val_frac)
        bad = 0
        for k, (ip, mp) in enumerate(tqdm(files, desc=name)):
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if img is None or m is None:
                bad += 1
                continue
            if m.shape != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = ((m > 0) if m.max() <= 1 else (m > 127)).astype(np.uint8) * 255
            if args.max_side:
                h, w = img.shape[:2]
                sc = args.max_side / max(h, w)
                if sc < 1:
                    img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                    m = cv2.resize(m, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_NEAREST)
            split = "val" if k < n_val else "train"
            stem = f"{name}_{k:06d}"
            cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / split / "masks" / f"{stem}.png"), m)
            counts[split] += 1
        print(f"[{name}] unreadable: {bad}")
    print(f"done -> {out}  train={counts['train']}  val={counts['val']}")


if __name__ == "__main__":
    main()
