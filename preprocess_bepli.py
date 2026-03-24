#!/usr/bin/env python3
"""
BePli Dataset Preprocessor — Robust Cleaning Pipeline for Object Detection
============================================================================
Designed after audit analysis of the BePli marine litter dataset (Seanoe).

Works with the existing BePli folder structure:
    dataset/BePli_dataset_v2/plastic_coco/
    ├── annotation/
    │   ├── all_plastic_coco.json   (full dataset)
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    └── images/
        ├── original_images/
        ├── train/
        ├── val/
        └── test/

PROBLEMS IDENTIFIED FROM AUDIT:
1. EXTREME class imbalance: "fragment" 38,938 annots vs "other_string" 617 (63:1)
2. MASSIVE tiny/dust annotations: 22,424 boxes < 8px area
3. Hyper-dense images: top image has 363 annotations
4. 13 classes with very different object sizes (median 72px—1092px area)
5. Very thin annotations: 151 h<2px, 15 w<2px
6. 3 images with 0 annotations
7. Mixed resolutions: 417x313 to 2592x1944

Usage:
    # Process each split separately (RECOMMENDED — preserves original splits)
    python preprocess_bepli.py \
        --dataset_root /path/to/BePli_dataset_v2/plastic_coco \
        --output_dir /path/to/clean_dataset/ \
        --format both \
        --strategy aggressive

    # Or process a single JSON
    python preprocess_bepli.py \
        --coco_json /path/to/train.json \
        --images_dir /path/to/images/train \
        --output_dir /path/to/clean_dataset/ \
        --split_name train \
        --strategy aggressive
"""

import json
import copy
import argparse
import os
import shutil
import random
import math
from pathlib import Path
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════════════════════
# 0. CONFIG — Every threshold is tuned from the audit CSVs
# ═══════════════════════════════════════════════════════════════════════════

# ── Class Merging ──────────────────────────────────────────────────────────
# The single biggest lever for mAP on long-tail datasets.
# Aggressive: 13 → 8 classes by grouping morphologically similar rare classes
CLASS_MERGE_MAP_AGGRESSIVE = {
    "fragment":            "fragment",
    "others":              "others",
    "styrene_foam":        "styrene_foam",
    "plastic_bag":         "plastic_bag",
    # Bottles merged (other_bottle median=354 ≈ pet_bottle median=449)
    "pet_bottle":          "bottle",
    "other_bottle":        "bottle",
    # Rope-like merged (other_string median=798 ≈ rope median=1020)
    "rope":                "rope",
    "other_string":        "rope",
    # Fishing gear merged
    "fishing_net":         "fishing_gear",
    "other_fishing_gear":  "fishing_gear",
    # Containers merged (box_shaped_case median=1092 ≈ other_container median=933)
    "other_container":     "container",
    "box_shaped_case":     "container",
    # Buoy standalone (1503 annots — enough to learn)
    "buoy":                "buoy",
}

# Conservative: keep all 13 original classes
CLASS_MERGE_MAP_CONSERVATIVE = {
    "fragment": "fragment", "others": "others", "styrene_foam": "styrene_foam",
    "pet_bottle": "pet_bottle", "plastic_bag": "plastic_bag", "rope": "rope",
    "buoy": "buoy", "other_bottle": "other_bottle", "fishing_net": "fishing_net",
    "other_fishing_gear": "other_fishing_gear", "other_container": "other_container",
    "box_shaped_case": "box_shaped_case", "other_string": "other_string",
}

# ── Per-class minimum bbox area (absolute pixels) ─────────────────────────
# Derived from per_class_size_stats.csv — tuned around each class's p5/p25
CLASS_MIN_AREA = {
    # Original class names (applied before merging)
    "fragment": 16,            # median=72,  but p5_norm is very low
    "others": 20,              # median=168, 2524 tiny annotations
    "styrene_foam": 16,        # median=198
    "pet_bottle": 24,          # median=449
    "plastic_bag": 20,         # median=377
    "rope": 30,                # median=1020
    "buoy": 20,                # median=234
    "other_bottle": 24,        # median=354
    "fishing_net": 24,         # median=493
    "other_container": 30,     # median=933
    "other_fishing_gear": 24,  # median=360
    "box_shaped_case": 30,     # median=1092
    "other_string": 24,        # median=798
    # Merged class fallbacks
    "bottle": 24,
    "fishing_gear": 24,
    "container": 30,
}

# ── Global thresholds ──────────────────────────────────────────────────────
GLOBAL_MIN_BBOX_AREA  = 12     # Absolute floor: ~3.5x3.5px (audit: 2387 < 4px area)
GLOBAL_MIN_BBOX_SIDE  = 3      # Min dimension on either side (151 boxes have h<2)
MIN_NORM_AREA         = 2e-5   # Normalized area floor (19,554 below 1e-4; we keep some)
MAX_NORM_AREA         = 0.85   # Giant boxes = annotation errors
MAX_ASPECT_RATIO      = 12.0   # Kill extreme slivers

# ── Image-level thresholds ─────────────────────────────────────────────────
MAX_ANNOTATIONS_PER_IMAGE          = 120   # Densest has 363 — cap hard
MIN_ANNOTATIONS_PER_IMAGE          = 1     # Remove empties (3 images with 0)
MAX_FRAGMENT_ANNOTATIONS_PER_IMAGE = 40    # Fragment floods single images

# ── Fragment global cap ────────────────────────────────────────────────────
MAX_FRAGMENT_FRACTION_OF_DATASET   = 0.35  # Fragment is 54% raw -> force to <=35%

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD & INDEX COCO JSON
# ═══════════════════════════════════════════════════════════════════════════

def load_coco(json_path):
    """Load a COCO JSON and build lookup indices."""
    with open(json_path, 'r') as f:
        coco = json.load(f)

    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    img_id_to_info = {img['id']: img for img in coco['images']}

    anns_by_image = defaultdict(list)
    for ann in coco['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    # Include images with 0 annotations in the dict
    for img in coco['images']:
        if img['id'] not in anns_by_image:
            anns_by_image[img['id']] = []

    print(f"  Loaded: {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations, "
          f"{len(coco['categories'])} categories")
    return coco, cat_id_to_name, img_id_to_info, anns_by_image


# ═══════════════════════════════════════════════════════════════════════════
# 2. ANNOTATION-LEVEL FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def clamp_bbox(ann, img_info):
    """Clamp bbox to image boundaries. Modifies ann in-place."""
    x, y, w, h = ann['bbox']
    img_w, img_h = img_info['width'], img_info['height']
    x = max(0.0, x)
    y = max(0.0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    ann['bbox'] = [x, y, w, h]
    ann['area'] = w * h
    return ann


def filter_annotation(ann, img_info, cat_name, strategy='aggressive'):
    """
    Returns (keep: bool, reason: str) for a single annotation.
    """
    x, y, w, h = ann['bbox']
    area = w * h
    img_w, img_h = img_info['width'], img_info['height']
    img_area = img_w * img_h
    norm_area = area / img_area if img_area > 0 else 0

    # ── Hard filters (always) ──
    if w <= 0 or h <= 0:
        return False, "zero_or_negative_dim"

    if x < -5 or y < -5 or (x + w) > img_w + 5 or (y + h) > img_h + 5:
        return False, "bbox_far_out_of_bounds"

    if w < GLOBAL_MIN_BBOX_SIDE or h < GLOBAL_MIN_BBOX_SIDE:
        return False, f"side_too_small_w{w:.0f}_h{h:.0f}"

    if area < GLOBAL_MIN_BBOX_AREA:
        return False, f"area_below_{GLOBAL_MIN_BBOX_AREA}"

    if norm_area < MIN_NORM_AREA:
        return False, "norm_area_too_small"

    if norm_area > MAX_NORM_AREA:
        return False, "norm_area_too_large"

    aspect = max(w, h) / (min(w, h) + 1e-6)
    if aspect > MAX_ASPECT_RATIO:
        return False, f"extreme_aspect_{aspect:.1f}"

    # ── Class-aware area filter ──
    min_area = CLASS_MIN_AREA.get(cat_name, GLOBAL_MIN_BBOX_AREA)
    if area < min_area:
        return False, f"below_class_min_{cat_name}"

    # ── Aggressive extras ──
    if strategy == 'aggressive':
        if cat_name in ('fragment', 'others') and norm_area < 4e-5:
            return False, f"aggressive_norm_{cat_name}"

    return True, "keep"


# ═══════════════════════════════════════════════════════════════════════════
# 3. IMAGE-LEVEL FILTERING
# ═══════════════════════════════════════════════════════════════════════════

def filter_image_annotations(anns, img_info, cat_id_to_name, merge_map, strategy):
    """Filter all annotations for one image. Returns (kept_list, removal_counter)."""
    kept = []
    removed = Counter()

    for ann in anns:
        cat_name = cat_id_to_name.get(ann['category_id'], 'unknown')
        ann = clamp_bbox(ann, img_info)
        ok, reason = filter_annotation(ann, img_info, cat_name, strategy)
        if ok:
            kept.append(ann)
        else:
            removed[reason] += 1

    # ── Per-image fragment cap ──
    frag_ids = {cid for cid, n in cat_id_to_name.items()
                if merge_map.get(n, n) == 'fragment'}
    frags  = [a for a in kept if a['category_id'] in frag_ids]
    others = [a for a in kept if a['category_id'] not in frag_ids]

    if len(frags) > MAX_FRAGMENT_ANNOTATIONS_PER_IMAGE:
        frags.sort(key=lambda a: a['bbox'][2] * a['bbox'][3], reverse=True)
        removed['fragment_per_image_cap'] += len(frags) - MAX_FRAGMENT_ANNOTATIONS_PER_IMAGE
        frags = frags[:MAX_FRAGMENT_ANNOTATIONS_PER_IMAGE]

    kept = others + frags

    # ── Total annotation cap (prioritise rare classes) ──
    if len(kept) > MAX_ANNOTATIONS_PER_IMAGE:
        def priority(a):
            cn = cat_id_to_name.get(a['category_id'], '')
            boost = 0 if merge_map.get(cn, cn) == 'fragment' else 1000
            return boost + a['bbox'][2] * a['bbox'][3]
        kept.sort(key=priority, reverse=True)
        removed['total_per_image_cap'] += len(kept) - MAX_ANNOTATIONS_PER_IMAGE
        kept = kept[:MAX_ANNOTATIONS_PER_IMAGE]

    return kept, removed


# ═══════════════════════════════════════════════════════════════════════════
# 4. GLOBAL FRAGMENT BALANCING
# ═══════════════════════════════════════════════════════════════════════════

def global_fragment_balance(all_anns, cat_id_to_name, merge_map):
    """Subsample smallest fragments globally until fraction <= target."""
    frag_ids = {cid for cid, n in cat_id_to_name.items()
                if merge_map.get(n, n) == 'fragment'}

    total = sum(len(a) for a in all_anns.values())
    frag_total = sum(1 for anns in all_anns.values() for x in anns
                     if x['category_id'] in frag_ids)
    frac = frag_total / max(total, 1)
    print(f"  Fragment fraction: {frac:.3f} ({frag_total}/{total})")

    if frac <= MAX_FRAGMENT_FRACTION_OF_DATASET:
        print(f"  Already within {MAX_FRAGMENT_FRACTION_OF_DATASET:.0%} target.")
        return all_anns, 0

    non_frag = total - frag_total
    target = int(non_frag * MAX_FRAGMENT_FRACTION_OF_DATASET
                 / (1 - MAX_FRAGMENT_FRACTION_OF_DATASET))
    to_remove = frag_total - target
    print(f"  Removing {to_remove} smallest fragment annotations globally...")

    pool = []
    for img_id, anns in all_anns.items():
        for i, a in enumerate(anns):
            if a['category_id'] in frag_ids:
                pool.append((img_id, i, a['bbox'][2] * a['bbox'][3]))
    pool.sort(key=lambda x: x[2])

    kill = set((r[0], r[1]) for r in pool[:to_remove])

    removed = 0
    for img_id in list(all_anns.keys()):
        new = [a for i, a in enumerate(all_anns[img_id])
               if (img_id, i) not in kill]
        removed += len(all_anns[img_id]) - len(new)
        all_anns[img_id] = new

    return all_anns, removed


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLASS REMAPPING
# ═══════════════════════════════════════════════════════════════════════════

def build_merged_categories(original_cats, merge_map):
    """Build new category list + old_id -> new_id mapping."""
    merged_names = sorted(set(merge_map.get(c['name'], c['name'])
                              for c in original_cats))
    new_cats = [{'id': i + 1, 'name': n, 'supercategory': 'marine_litter'}
                for i, n in enumerate(merged_names)]
    name2id = {c['name']: c['id'] for c in new_cats}

    old2new = {}
    for c in original_cats:
        old2new[c['id']] = name2id[merge_map.get(c['name'], c['name'])]
    return new_cats, old2new


# ═══════════════════════════════════════════════════════════════════════════
# 6. OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════

def write_coco_json(image_ids, anns_by_image, categories, img_id_to_info,
                    old2new, output_path):
    """Write cleaned COCO JSON."""
    out = {'images': [], 'annotations': [], 'categories': categories}
    aid = 1
    for iid in sorted(image_ids):
        out['images'].append(img_id_to_info[iid])
        for ann in anns_by_image.get(iid, []):
            a = copy.deepcopy(ann)
            a['id'] = aid
            a['category_id'] = old2new[ann['category_id']]
            a['area'] = a['bbox'][2] * a['bbox'][3]
            out['annotations'].append(a)
            aid += 1
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f)
    print(f"    COCO -> {output_path}: {len(out['images'])} imgs, "
          f"{len(out['annotations'])} anns")


def write_yolo_labels(image_ids, anns_by_image, img_id_to_info, old2new,
                      labels_dir, list_path, images_rel_prefix):
    """Write YOLO .txt labels + image list file."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(list_path, 'w') as flist:
        for iid in sorted(image_ids):
            info = img_id_to_info[iid]
            iw, ih = info['width'], info['height']
            stem = Path(info['file_name']).stem

            lbl = labels_dir / f'{stem}.txt'
            with open(lbl, 'w') as lf:
                for ann in anns_by_image.get(iid, []):
                    bx, by, bw, bh = ann['bbox']
                    cx = max(0, min(1, (bx + bw / 2) / iw))
                    cy = max(0, min(1, (by + bh / 2) / ih))
                    nw = max(0, min(1, bw / iw))
                    nh = max(0, min(1, bh / ih))
                    cls = old2new[ann['category_id']] - 1  # 0-indexed
                    lf.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            flist.write(f"./{images_rel_prefix}/{info['file_name']}\n")

    print(f"    YOLO labels -> {labels_dir} ({len(image_ids)} files)")


def write_yolo_yaml(categories, output_dir):
    """Write data.yaml for YOLO training."""
    p = Path(output_dir) / 'data.yaml'
    names = [c['name'] for c in sorted(categories, key=lambda c: c['id'])]
    with open(p, 'w') as f:
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("test: test.txt\n\n")
        f.write(f"nc: {len(names)}\n")
        f.write(f"names: {names}\n")
    print(f"    data.yaml -> {p}")


def symlink_or_copy_images(image_ids, img_id_to_info, src_dir, dst_dir):
    """Symlink images (fast, saves disk) or copy if symlink fails."""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    ok, miss = 0, 0
    for iid in sorted(image_ids):
        fname = img_id_to_info[iid]['file_name']
        s = Path(src_dir) / fname
        d = dst / fname
        if d.exists():
            ok += 1
            continue
        if s.exists():
            try:
                d.symlink_to(s.resolve())
            except OSError:
                shutil.copy2(s, d)
            ok += 1
        else:
            miss += 1
    if miss:
        print(f"    WARNING: {miss} images missing from {src_dir}")
    print(f"    Images: {ok} linked/copied -> {dst_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def class_report(anns_by_image, old2new, new_cats, label=""):
    """Print per-class counts."""
    id2name = {c['id']: c['name'] for c in new_cats}
    counts = Counter()
    for anns in anns_by_image.values():
        for a in anns:
            counts[id2name[old2new[a['category_id']]]] += 1
    total = sum(counts.values())
    print(f"\n  Class distribution {label}:")
    print(f"  {'Class':<20} {'Count':>8} {'%':>7}")
    print(f"  {'-'*37}")
    for n, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {n:<20} {c:>8} {c/total:>7.1%}")
    print(f"  {'TOTAL':<20} {total:>8}")
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# 8. PROCESS ONE SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def process_split(json_path, split_name, merge_map, strategy,
                  output_dir, fmt, images_src_dir=None):
    """Full cleaning pipeline for a single split (train / val / test)."""
    print(f"\n{'━'*70}")
    print(f"  Processing split: {split_name.upper()}")
    print(f"{'━'*70}")

    coco, cat_id_to_name, img_id_to_info, anns_by_image = load_coco(json_path)
    new_cats, old2new = build_merged_categories(coco['categories'], merge_map)

    # ── Per-annotation + per-image filtering ──
    print(f"\n  Filtering annotations (strategy={strategy})...")
    clean_anns = {}
    total_removed = Counter()
    n_orig = 0

    for img_id, anns in anns_by_image.items():
        n_orig += len(anns)
        kept, reasons = filter_image_annotations(
            anns, img_id_to_info[img_id], cat_id_to_name, merge_map, strategy)
        clean_anns[img_id] = kept
        total_removed += reasons

    n_kept = sum(len(a) for a in clean_anns.values())
    print(f"  {n_orig} -> {n_kept} annotations ({n_orig - n_kept} removed, "
          f"{n_kept/max(n_orig,1):.1%} kept)")

    top_reasons = total_removed.most_common(8)
    if top_reasons:
        print(f"  Top removal reasons:")
        for r, c in top_reasons:
            print(f"    {r}: {c}")

    # ── Global fragment balance (train only — don't distort val/test) ──
    n_balanced = 0
    if split_name == 'train':
        print(f"\n  Global fragment balancing "
              f"(target <= {MAX_FRAGMENT_FRACTION_OF_DATASET:.0%})...")
        clean_anns, n_balanced = global_fragment_balance(
            clean_anns, cat_id_to_name, merge_map)

    # ── Remove empty images ──
    valid = {iid for iid, a in clean_anns.items()
             if len(a) >= MIN_ANNOTATIONS_PER_IMAGE}
    n_dropped = len(clean_anns) - len(valid)
    if n_dropped:
        print(f"  Dropped {n_dropped} images with "
              f"< {MIN_ANNOTATIONS_PER_IMAGE} annotations")
    clean_anns = {k: v for k, v in clean_anns.items() if k in valid}

    class_report(clean_anns, old2new, new_cats, label=f"({split_name})")

    # ── Write outputs ──
    out = Path(output_dir)

    if fmt in ('coco', 'both'):
        coco_dir = out / 'coco'
        write_coco_json(valid, clean_anns, new_cats, img_id_to_info,
                        old2new, coco_dir / f'{split_name}.json')

    if fmt in ('yolo', 'both'):
        yolo_dir = out / 'yolo'
        yolo_dir.mkdir(parents=True, exist_ok=True)
        write_yolo_labels(
            valid, clean_anns, img_id_to_info, old2new,
            labels_dir=yolo_dir / 'labels' / split_name,
            list_path=yolo_dir / f'{split_name}.txt',
            images_rel_prefix=f'images/{split_name}'
        )

    # ── Copy/link images ──
    if images_src_dir and os.path.isdir(images_src_dir):
        if fmt in ('coco', 'both'):
            symlink_or_copy_images(valid, img_id_to_info, images_src_dir,
                                   out / 'coco' / 'images' / split_name)
        if fmt in ('yolo', 'both'):
            symlink_or_copy_images(valid, img_id_to_info, images_src_dir,
                                   out / 'yolo' / 'images' / split_name)

    return new_cats, len(valid), sum(len(a) for a in clean_anns.values()), n_balanced


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 70)
    print("  BePli Dataset Preprocessor")
    print("=" * 70)

    strategy = args.strategy
    merge_map = (CLASS_MERGE_MAP_AGGRESSIVE if strategy == 'aggressive'
                 else CLASS_MERGE_MAP_CONSERVATIVE)
    fmt = args.format.lower()

    print(f"\n  Strategy:  {strategy}")
    print(f"  Merging:   13 -> {len(set(merge_map.values()))} classes")
    print(f"  Format:    {fmt}")

    # ── Determine what to process ──
    if args.dataset_root:
        root = Path(args.dataset_root)
        ann_dir = root / 'annotation'
        img_dir = root / 'images'

        splits = {}
        for split in ['train', 'val', 'test']:
            json_path = ann_dir / f'{split}.json'
            if json_path.exists():
                splits[split] = {
                    'json': str(json_path),
                    'images': str(img_dir / split),
                }
            else:
                print(f"  WARNING: {json_path} not found, skipping {split}")

        if not splits:
            all_json = ann_dir / 'all_plastic_coco.json'
            if all_json.exists():
                print(f"\n  No split JSONs found. Using all_plastic_coco.json "
                      f"with original_images/")
                splits['all'] = {
                    'json': str(all_json),
                    'images': str(img_dir / 'original_images'),
                }
            else:
                print(f"  ERROR: No annotation files found in {ann_dir}")
                return

    elif args.coco_json:
        sname = args.split_name or 'train'
        splits = {sname: {
            'json': args.coco_json,
            'images': args.images_dir,
        }}
    else:
        print("ERROR: Provide --dataset_root or --coco_json")
        return

    # ── Process each split ──
    summary = {}
    cats = None
    for split, info in splits.items():
        c, n_img, n_ann, n_bal = process_split(
            json_path=info['json'],
            split_name=split,
            merge_map=merge_map,
            strategy=strategy,
            output_dir=args.output_dir,
            fmt=fmt,
            images_src_dir=info.get('images'),
        )
        cats = c
        summary[split] = (n_img, n_ann, n_bal)

    # ── Write YOLO data.yaml ──
    if fmt in ('yolo', 'both') and cats:
        write_yolo_yaml(cats, Path(args.output_dir) / 'yolo')

    # ── Write preprocessing report ──
    report = Path(args.output_dir) / 'preprocessing_report.txt'
    with open(report, 'w') as f:
        f.write("BePli Preprocessing Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Classes: {len(cats)}\n")
        for c in sorted(cats, key=lambda x: x['id']):
            f.write(f"  {c['id']}: {c['name']}\n")
        f.write("\n")
        for split, (ni, na, nb) in summary.items():
            f.write(f"{split}: {ni} images, {na} annotations")
            if nb:
                f.write(f" (fragment balanced: -{nb})")
            f.write("\n")

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")
    print(f"\n  Output: {args.output_dir}")
    for split, (ni, na, nb) in summary.items():
        bal = f" (fragment balanced: -{nb})" if nb else ""
        print(f"  {split:>5}: {ni:>5} images  {na:>6} annotations{bal}")
    print(f"\n  Classes ({len(cats)}):")
    for c in sorted(cats, key=lambda x: x['id']):
        print(f"    {c['id']}: {c['name']}")

    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     TRAINING RECOMMENDATIONS                           ║
║                                                                        ║
║  After 26 experiments without mAP > 0.3, focus on these in order:      ║
║                                                                        ║
║  1  IMAGE SIZE = 1280 (not 640!)                                       ║
║     Your median box is 14x9 px. At 640 input, that becomes ~6x4 px.   ║
║     The model literally cannot see your objects at 640.                 ║
║     -> --imgsz 1280                                                    ║
║                                                                        ║
║  2  SAHI TILED INFERENCE (eval time, free mAP boost)                   ║
║     pip install sahi                                                    ║
║     Slice images into overlapping tiles for prediction.                 ║
║     Typically +5-15% mAP on small-object datasets.                     ║
║                                                                        ║
║  3  MODEL: YOLOv8m or YOLOv9c (anchor-free)                           ║
║     Anchor-based models choke on BePli's 72-to-1092px size range.      ║
║     Medium models > small models for this dataset.                     ║
║                                                                        ║
║  4  AUGMENTATION (critical for rare classes):                          ║
║     copy_paste=0.3  mosaic=1.0  scale=0.3  mixup=0.15                 ║
║                                                                        ║
║  5  TRAINING PARAMS:                                                   ║
║     --epochs 200 --patience 30 --lr0 0.005 --warmup_epochs 5          ║
║     --batch 4-8 (limited by imgsz=1280 + VRAM)                        ║
║                                                                        ║
║  6  MULTI-SCALE TRAINING:                                              ║
║     Resolutions vary 417x313 to 2592x1944. Use --multi_scale          ║
║                                                                        ║
║  QUICK START (YOLOv8):                                                 ║
║  yolo detect train data=clean/yolo/data.yaml model=yolov8m.pt \\       ║
║    imgsz=1280 epochs=200 patience=30 batch=4 \\                        ║
║    lr0=0.005 warmup_epochs=5 copy_paste=0.3 mosaic=1.0 \\              ║
║    scale=0.5 mixup=0.15 close_mosaic=20 name=bepli_clean_v1           ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Output directory structure:
  {args.output_dir}/
  ├── coco/
  │   ├── train.json          <- cleaned COCO annotations
  │   ├── val.json
  │   ├── test.json
  │   └── images/
  │       ├── train/           <- symlinked images
  │       ├── val/
  │       └── test/
  ├── yolo/
  │   ├── data.yaml            <- point YOLOv8 here
  │   ├── train.txt
  │   ├── val.txt
  │   ├── test.txt
  │   ├── labels/
  │   │   ├── train/           <- YOLO .txt labels
  │   │   ├── val/
  │   │   └── test/
  │   └── images/
  │       ├── train/           <- symlinked images
  │       ├── val/
  │       └── test/
  └── preprocessing_report.txt
""")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='BePli Dataset Preprocessor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full dataset (RECOMMENDED — processes train + val + test):
  python preprocess_bepli.py \\
      --dataset_root ./BePli_dataset_v2/plastic_coco \\
      --output_dir ./clean_bepli/ \\
      --strategy aggressive

  # Single split only:
  python preprocess_bepli.py \\
      --coco_json ./plastic_coco/annotation/train.json \\
      --images_dir ./plastic_coco/images/train \\
      --output_dir ./clean_bepli/ \\
      --split_name train

  # Conservative (keep all 13 classes):
  python preprocess_bepli.py \\
      --dataset_root ./BePli_dataset_v2/plastic_coco \\
      --output_dir ./clean_bepli_v2/ \\
      --strategy conservative
        """)

    g = p.add_mutually_exclusive_group()
    g.add_argument('--dataset_root',
                   help='Root: .../plastic_coco (contains annotation/ + images/)')
    g.add_argument('--coco_json',
                   help='Path to a single COCO annotation JSON')

    p.add_argument('--images_dir', default=None,
                   help='Images dir (only with --coco_json)')
    p.add_argument('--split_name', default=None,
                   help='Split name with --coco_json (default: train)')
    p.add_argument('--output_dir', required=True,
                   help='Output directory')
    p.add_argument('--format', choices=['coco', 'yolo', 'both'], default='both',
                   help='Output format (default: both)')
    p.add_argument('--strategy', choices=['aggressive', 'conservative'],
                   default='aggressive',
                   help='Cleaning strategy (default: aggressive)')

    args = p.parse_args()
    main(args)