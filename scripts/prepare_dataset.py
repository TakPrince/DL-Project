#!/usr/bin/env python3
"""
Dataset Acquisition & Preparation Script
Project: DL-Project Grocery Classification (Juice, Milk, Yoghurt)
Source: Marcus Klasson et al., Grocery Store Dataset (WACV 2019)
"""

import os
import sys
import hashlib
import random
import subprocess
import shutil
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

SEED = 42
TARGET_PER_CLASS = 120
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASSES = ["Juice", "Milk", "Yoghurt"]

PROJECT_ROOT = Path("/home/asus/Desktop/DL-Project")
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

BASE_URL = "https://raw.githubusercontent.com/marcusklasson/GroceryStoreDataset/master/dataset/"
TRAIN_FILE = Path("/tmp/train.txt")

def ensure_train_file():
    if not TRAIN_FILE.exists():
        url = BASE_URL + "train.txt"
        subprocess.run(["curl", "-s", "--max-time", "15", url, "-o", str(TRAIN_FILE)], check=True)

def fetch_candidates():
    print("[1/5] Reading official Grocery Store Dataset train catalog...")
    ensure_train_file()
    class_candidates = {c: [] for c in CLASSES}
    
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            rel_path = parts[0]
            for cls in CLASSES:
                if f"Packages/{cls}/" in rel_path:
                    class_candidates[cls].append(rel_path)
                    break

    for cls in CLASSES:
        unique = list(dict.fromkeys(class_candidates[cls]))
        class_candidates[cls] = unique
        print(f"  Available unique candidates for '{cls}': {len(unique)}")
    return class_candidates

def download_image(args):
    rel_path, cls, target_file = args
    img_url = BASE_URL + rel_path
    temp_file = target_file.with_suffix(".tmp")
    
    # Download with curl
    res = subprocess.run(
        ["curl", "-s", "--max-time", "12", img_url, "-o", str(temp_file)],
        capture_output=True
    )
    if res.returncode != 0 or not temp_file.exists() or temp_file.stat().st_size == 0:
        if temp_file.exists():
            temp_file.unlink()
        return (cls, target_file, False, "Download failed or empty file")

    # Validate image with PIL
    try:
        with Image.open(temp_file) as img:
            img.verify()
            
        with Image.open(temp_file) as img:
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(target_file, "JPEG", quality=95)
            
        if temp_file.exists():
            temp_file.unlink()
        return (cls, target_file, True, None)
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        if target_file.exists():
            target_file.unlink()
        return (cls, target_file, False, str(e))

def download_and_validate():
    print(f"\n[2/5] Sampling and acquiring {TARGET_PER_CLASS} images per class (seed={SEED})...")
    class_candidates = fetch_candidates()
    
    rnd = random.Random(SEED)
    downloaded_counts = {c: 0 for c in CLASSES}
    invalid_counts = {c: 0 for c in CLASSES}
    valid_images = {c: [] for c in CLASSES}
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    tasks = []
    for cls in CLASSES:
        cls_raw_dir = RAW_DIR / cls
        cls_raw_dir.mkdir(parents=True, exist_ok=True)
        
        candidates = list(class_candidates[cls])
        rnd.shuffle(candidates)
        # Select target count + 5 buffer
        selected = candidates[:TARGET_PER_CLASS + 5]
        
        for rel_path in selected:
            filename = Path(rel_path).name
            target_file = cls_raw_dir / filename
            tasks.append((rel_path, cls, target_file))
            
    print(f"  Downloading and inspecting {len(tasks)} images in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_image, t) for t in tasks]
        for f in as_completed(futures):
            cls, target_file, success, err = f.result()
            if success:
                downloaded_counts[cls] += 1
                if len(valid_images[cls]) < TARGET_PER_CLASS:
                    valid_images[cls].append(target_file)
            else:
                invalid_counts[cls] += 1
                print(f"    [WARN] Invalid/Corrupt image ({target_file.name}): {err}")
                
    for cls in CLASSES:
        print(f"    '{cls}': {len(valid_images[cls])} valid images retained.")
        
    return valid_images, downloaded_counts, invalid_counts

def create_splits(valid_images):
    print("\n[3/5] Creating train (70%), val (15%), test (15%) splits with fixed seed 42...")
    rnd = random.Random(SEED)
    
    split_counts = {s: {c: 0 for c in CLASSES} for s in ["train", "val", "test"]}
    
    for s in ["train", "val", "test"]:
        s_dir = DATA_DIR / s
        if s_dir.exists():
            shutil.rmtree(s_dir)
        for c in CLASSES:
            (s_dir / c).mkdir(parents=True, exist_ok=True)
            
    for cls in CLASSES:
        images = sorted(list(valid_images[cls]), key=lambda p: p.name)
        rnd.shuffle(images)
        
        n_total = len(images)
        n_train = int(round(n_total * SPLIT_RATIOS["train"]))
        n_val = int(round(n_total * SPLIT_RATIOS["val"]))
        n_test = n_total - n_train - n_val
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        for img in train_imgs:
            shutil.copy2(img, DATA_DIR / "train" / cls / img.name)
            split_counts["train"][cls] += 1
            
        for img in val_imgs:
            shutil.copy2(img, DATA_DIR / "val" / cls / img.name)
            split_counts["val"][cls] += 1
            
        for img in test_imgs:
            shutil.copy2(img, DATA_DIR / "test" / cls / img.name)
            split_counts["test"][cls] += 1
            
    return split_counts

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_dataset():
    print("\n[4/5] Verifying zero leakage, file hashes, and integrity...")
    split_hashes = {}
    split_filenames = {}
    
    for s in ["train", "val", "test"]:
        split_hashes[s] = set()
        split_filenames[s] = set()
        for c in CLASSES:
            folder = DATA_DIR / s / c
            for img_path in sorted(folder.glob("*.jpg")):
                h = file_hash(img_path)
                split_hashes[s].add(h)
                split_filenames[s].add(img_path.name)
                
    leakage_detected = False
    
    # Filename overlap
    tv_names = split_filenames["train"] & split_filenames["val"]
    tt_names = split_filenames["train"] & split_filenames["test"]
    vt_names = split_filenames["val"] & split_filenames["test"]
    
    # Hash overlap between splits
    tv_hashes = split_hashes["train"] & split_hashes["val"]
    tt_hashes = split_hashes["train"] & split_hashes["test"]
    vt_hashes = split_hashes["val"] & split_hashes["test"]
    
    if tv_names or tt_names or vt_names:
        print(f"  [ERROR] Filename overlap detected! Train-Val: {len(tv_names)}, Train-Test: {len(tt_names)}, Val-Test: {len(vt_names)}")
        leakage_detected = True
    else:
        print("  [PASS] Filename overlap: EXACTLY 0 overlapping filenames across splits.")
        
    if tv_hashes or tt_hashes or vt_hashes:
        print(f"  [ERROR] Hash overlap detected! Train-Val: {len(tv_hashes)}, Train-Test: {len(tt_hashes)}, Val-Test: {len(vt_hashes)}")
        leakage_detected = True
    else:
        print("  [PASS] Hash overlap: EXACTLY 0 overlapping file hashes across splits.")
        
    return not leakage_detected

def compute_disk_usage(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

def print_tree(dir_path, prefix=""):
    entries = sorted(list(dir_path.iterdir()), key=lambda e: (e.is_file(), e.name))
    for i, entry in enumerate(entries):
        is_last = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            n_files = len(list(entry.glob("*.jpg")))
            print(f"{prefix}{connector}{entry.name}/ ({n_files} images)")
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension)

def main():
    valid_images, downloaded_counts, invalid_counts = download_and_validate()
    split_counts = create_splits(valid_images)
    passed_verification = verify_dataset()
    
    if not passed_verification:
        print("[FAIL] Verification failed due to data leakage!")
        sys.exit(1)
        
    total_bytes = compute_disk_usage(DATA_DIR)
    raw_bytes = compute_disk_usage(RAW_DIR)
    splits_bytes = total_bytes - raw_bytes
    
    print("\n[5/5] Preparation Complete!")
    print(f"Total dataset disk usage (including raw cache): {total_bytes / (1024*1024):.2f} MB")
    print(f"Splits disk usage (train+val+test): {splits_bytes / (1024*1024):.2f} MB")
    
    # Summary Table
    print("\n=== DATASET SUMMARY ===")
    print(f"{'Class':<12} | {'Downloaded':<10} | {'Invalid':<8} | {'Valid Total':<12} | {'Train':<6} | {'Val':<6} | {'Test':<6}")
    print("-" * 75)
    for c in CLASSES:
        n_dl = downloaded_counts[c]
        n_inv = invalid_counts[c]
        n_val_tot = len(valid_images[c])
        n_tr = split_counts["train"][c]
        n_va = split_counts["val"][c]
        n_te = split_counts["test"][c]
        print(f"{c:<12} | {n_dl:<10} | {n_inv:<8} | {n_val_tot:<12} | {n_tr:<6} | {n_va:<6} | {n_te:<6}")
    print("-" * 75)
    total_dl = sum(downloaded_counts.values())
    total_inv = sum(invalid_counts.values())
    total_val = sum(len(valid_images[c]) for c in CLASSES)
    total_tr = sum(split_counts['train'].values())
    total_va = sum(split_counts['val'].values())
    total_te = sum(split_counts['test'].values())
    print(f"{'TOTAL':<12} | {total_dl:<10} | {total_inv:<8} | {total_val:<12} | {total_tr:<6} | {total_va:<6} | {total_te:<6}")
    
    print("\n=== DIRECTORY TREE ===")
    print_tree(DATA_DIR)

if __name__ == "__main__":
    main()
