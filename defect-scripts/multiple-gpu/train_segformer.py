#!/usr/bin/env python3
"""
SegFormer-B0 fine-tune for binary defect (crack/rust/weld) segmentation.
Single GPU or multi-GPU (DDP via torchrun) from the same file.

Data layout (see prepare_data.py):
    data_root/{train,val}/images/*.jpg   data_root/{train,val}/masks/*.png  (0 / >127)

Single GPU:   python train_segformer.py --data_root ./data --out_dir ./runs/b0
Multi GPU:    torchrun --nproc_per_node=2 train_segformer.py --data_root ./data --out_dir ./runs/b0
Note: --batch_size is PER GPU. Effective batch = batch_size * num_gpus.
"""
import argparse
import json
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------- dist helpers
def setup_dist():
    """Returns (ddp, rank, world, device). Falls back to single process if not launched by torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        return True, dist.get_rank(), dist.get_world_size(), torch.device("cuda", local)
    return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- data
class DefectDataset(Dataset):
    def __init__(self, root, split, img_size, train):
        root = Path(root) / split
        self.imgs = sorted(p for p in (root / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
        self.mask_dir = root / "masks"
        self.tf = self._build_tf(img_size, train)

    @staticmethod
    def _build_tf(s, train):
        norm = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
        if not train:
            return A.Compose([A.Resize(s, s)] + norm)
        return A.Compose([
            A.Resize(s, s),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.8, 1.2), rotate=(-30, 30), translate_percent=(-0.1, 0.1), p=0.5),
            A.ColorJitter(0.3, 0.3, 0.3, 0.05, p=0.6),
            A.OneOf([A.GaussianBlur(blur_limit=(3, 5)), A.GaussNoise(), A.MotionBlur(blur_limit=5)], p=0.3),
        ] + norm)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        ip = self.imgs[i]
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mp = next(self.mask_dir.glob(ip.stem + ".*"))
        mask = (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].float()


# ----------------------------------------------------------------------------- loss / metrics
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2))
    union = p.sum((1, 2)) + target.sum((1, 2))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def focal_bce(logits, target, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = torch.exp(-bce)
    a = alpha * target + (1 - alpha) * (1 - target)
    return (a * (1 - pt) ** gamma * bce).mean()


def criterion(logits, target):
    return focal_bce(logits, target) + dice_loss(logits, target)


def metrics_from_counts(tp, fp, fn, tn):
    e = 1e-9
    iou_d = tp / (tp + fp + fn + e)
    iou_b = tn / (tn + fp + fn + e)
    prec = tp / (tp + fp + e)
    rec = tp / (tp + fn + e)
    return {"mIoU": (iou_d + iou_b) / 2, "IoU_defect": iou_d, "IoU_bg": iou_b,
            "precision": prec, "recall": rec, "F1": 2 * prec * rec / (prec + rec + e)}


# ----------------------------------------------------------------------------- model
def forward_logits(model, x):
    """SegFormer outputs 1/4-res logits -> upsample to input res. Returns (B,H,W)."""
    lg = model(pixel_values=x).logits
    lg = F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return lg.squeeze(1)


@torch.no_grad()
def evaluate(model, loader, device, ddp, rank, thr=0.5):
    model.eval()
    # [tp, fp, fn, tn, loss_sum, n]  -> all-reduced across ranks
    acc = torch.zeros(6, device=device, dtype=torch.float64)
    for x, y in tqdm(loader, desc="val", leave=False, disable=rank != 0):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            lg = forward_logits(model, x)
        pred, gt = torch.sigmoid(lg.float()) > thr, y.bool()
        acc[0] += (pred & gt).sum(); acc[1] += (pred & ~gt).sum()
        acc[2] += (~pred & gt).sum(); acc[3] += (~pred & ~gt).sum()
        acc[4] += criterion(lg.float(), y).item() * x.size(0); acc[5] += x.size(0)
    if ddp:
        dist.all_reduce(acc, op=dist.ReduceOp.SUM)
    tp, fp, fn, tn, ls, n = acc.tolist()
    m = metrics_from_counts(tp, fp, fn, tn)
    m["val_loss"] = ls / max(n, 1)
    return m


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", default="./runs/segformer_b0")
    ap.add_argument("--pretrained", default="nvidia/mit-b0",
                    help="nvidia/mit-b0 (ImageNet encoder) or a prior checkpoint dir for stage-2 fine-tune")
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16, help="per GPU")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=8, help="per process")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None, help="path to last.pt")
    args = ap.parse_args()

    ddp, rank, world, device = setup_dist()
    main_proc = rank == 0
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)

    out = Path(args.out_dir)
    if main_proc:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    tr = DefectDataset(args.data_root, "train", args.img_size, train=True)
    va = DefectDataset(args.data_root, "val", args.img_size, train=False)
    tsamp = DistributedSampler(tr, world, rank, shuffle=True, drop_last=True) if ddp else None
    vsamp = DistributedSampler(va, world, rank, shuffle=False) if ddp else None
    tl = DataLoader(tr, args.batch_size, shuffle=tsamp is None, sampler=tsamp, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    vl = DataLoader(va, args.batch_size, shuffle=False, sampler=vsamp, num_workers=args.workers, pin_memory=True)
    if main_proc:
        print(f"train={len(tr)}  val={len(va)}  gpus={world}  per-gpu bs={args.batch_size}  "
              f"effective bs={args.batch_size * world}  device={device}")

    # rank 0 downloads/loads first so ranks don't race on the HF cache
    if ddp and not main_proc:
        dist.barrier()
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.pretrained, num_labels=1, ignore_mismatched_sizes=True).to(device)
    if ddp and main_proc:
        dist.barrier()

    core = model
    if ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # decode head has BN
        model = DDP(model, device_ids=[device.index], output_device=device.index)
        core = model.module

    enc = [p for n, p in core.named_parameters() if n.startswith("segformer.")]
    head = [p for n, p in core.named_parameters() if not n.startswith("segformer.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr},
                             {"params": head, "lr": args.lr * 10}], weight_decay=args.wd)
    total = args.epochs * len(tl)
    warm = max(1, min(500, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start, best = 0, -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); start, best = ck["epoch"] + 1, ck["best"]
        if main_proc:
            print(f"resumed from epoch {start}")

    log = []
    for ep in range(start, args.epochs):
        if tsamp is not None:
            tsamp.set_epoch(ep)
        model.train()
        run = 0.0
        pbar = tqdm(tl, desc=f"ep {ep + 1}/{args.epochs}", disable=not main_proc)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                lg = forward_logits(model, x)
            loss = criterion(lg.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        m = evaluate(model, vl, device, ddp, rank)  # identical on all ranks after all_reduce
        m.update(epoch=ep + 1, train_loss=run / len(tl))
        improved = m["mIoU"] > best
        if improved:
            best = m["mIoU"]
        if main_proc:
            log.append(m)
            print(f"  mIoU={m['mIoU']:.4f}  IoU_defect={m['IoU_defect']:.4f}  F1={m['F1']:.4f}  "
                  f"P={m['precision']:.3f} R={m['recall']:.3f}  val_loss={m['val_loss']:.4f}")
            if improved:
                core.save_pretrained(out / "best")  # HF format -> from_pretrained() in your ROS node
                print(f"  new best mIoU {best:.4f}")
            torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "epoch": ep, "best": best}, out / "last.pt")
            old = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else []
            (out / "metrics.json").write_text(json.dumps(old + [m], indent=2))
        if ddp:
            dist.barrier()

    if main_proc:
        print(f"done. best mIoU={best:.4f}  -> {out / 'best'}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
