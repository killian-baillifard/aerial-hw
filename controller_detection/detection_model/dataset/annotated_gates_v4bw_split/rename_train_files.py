#!/usr/bin/env python3
"""Tiny utility: prefix filenames listed in a train.txt and their labels.

Assumptions: train.txt lines are paths relative to the train.txt parent and
image paths contain an "images" segment which is replaced by "labels" for
corresponding .txt label files.
"""
from pathlib import Path
import argparse
import shutil

DEFAULT_PREFIX = "2026-05-13-14-55-19_"

# Hardcoded settings (no CLI arguments)
TRAIN_FILE = Path(__file__).parent / "train.txt"
PREFIX = DEFAULT_PREFIX
DRY_RUN = True


def main():
    train = TRAIN_FILE
    if not train.exists():
        print(f"train file not found: {train}")
        return

    images_dir = train.parent / 'images' / 'train'
    labels_dir = train.parent / 'labels' / 'train'
    if not images_dir.exists():
        print(f"images directory not found: {images_dir}")
        return

    imgs = sorted(images_dir.glob('*.png'))
    ops = []
    out = []
    for img in imgs:
        dst = img.parent / (PREFIX + img.name)
        ops.append((img, dst))
        # corresponding label (may not exist)
        lbl = labels_dir / (img.stem + '.txt')
        lbl_dst = labels_dir / (PREFIX + lbl.name)
        ops.append((lbl, lbl_dst))
        out.append(str(dst.relative_to(train.parent)))

    for s, d in ops:
        print(f"{s} -> {d} (exists={s.exists()})")
    if DRY_RUN:
        print("DRY RUN - no changes applied")
        return

    # backup train
    shutil.copy2(train, train.with_suffix(train.suffix + '.bak'))

    for s, d in ops:
        if not s.exists():
            # skip missing files silently
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        s.rename(d)

    train.write_text('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
