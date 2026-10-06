#!/usr/bin/env python3
"""
Merge multiple (images_dir, masks_dir) sources into the layout train_segformer.py expects.

    out/{train,val}/images/*.jpg
    out/{train,val}/masks/*.png     (0 / 255)

Usage (repeat --src per dataset, format  name=images_dir:masks_dir):
    python prepare_data.py --out ./data --max_side 1024 --val_frac 0.1 \
        --src crackseg9k=/path/CrackSeg9k/images:/path/CrackSeg9k/masks \
        --src deepcrack=/path/DeepCrack/train_img:/path/DeepCrack/train_lab
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_mask(mdir, stem, suffix):
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif"):
        p = mdir / f"{stem}{suffix}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True, help="name=images_dir:masks_dir")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--max_side", type=int, default=1024, help="downscale so longest side <= this (0 = off)")
    ap.add_argument("--mask_suffix", default="", help="e.g. _mask if masks are named img001_mask.png")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "masks").mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for spec in args.src:
        name, paths = spec.split("=", 1)
        idir, mdir = map(Path, paths.split(":", 1))
        imgs = sorted(p for p in idir.rglob("*") if p.suffix.lower() in EXTS)
        random.shuffle(imgs)
        n_val = int(len(imgs) * args.val_frac)
        skipped = 0
        for k, ip in enumerate(tqdm(imgs, desc=name)):
            mp = find_mask(mdir, ip.stem, args.mask_suffix)
            if mp is None:
                skipped += 1
                continue
            img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if img is None or m is None:
                skipped += 1
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
            stem = f"{name}_{ip.stem}"
            cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(out / split / "masks" / f"{stem}.png"), m)
            counts[split] += 1
        print(f"[{name}] skipped (no mask / unreadable): {skipped}")

    print(f"done -> {out}  train={counts['train']}  val={counts['val']}")


if __name__ == "__main__":
    main()
