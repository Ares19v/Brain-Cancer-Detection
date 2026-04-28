"""
NeuroScan — External Validation Dataset Downloader (4-Class)
===========================================================
Downloads the SartajBhuvaji dataset and organizes it into 
the 4 specific categories: glioma, meningioma, pituitary, notumor.

Run from project root:
    python scripts/download_external_val.py
"""

import os
import sys
import urllib.request
import zipfile
import shutil

DEST_DIR = "datasets/external_validation"
ZIP_URL  = "https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet/archive/refs/heads/master.zip"

def download_with_progress(url: str, dest: str):
    print(f"[Download] {url}")
    print(f"        -> {dest}")
    def _hook(count, block_size, total_size):
        mb = count * block_size / (1024 * 1024)
        print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()

def main():
    # Clean old directory to avoid mixing data
    if os.path.exists(DEST_DIR):
        print(f"[Clean] Removing old {DEST_DIR}...")
        shutil.rmtree(DEST_DIR)
        
    os.makedirs(DEST_DIR, exist_ok=True)
    zip_path = os.path.join(DEST_DIR, "external_dataset.zip")

    download_with_progress(ZIP_URL, zip_path)

    # ── Extract ──────────────────────────────────────────────────────────────
    extract_dir = os.path.join(DEST_DIR, "raw")
    print(f"[Extract] Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print("[Extract] Done.")

    # ── Map Folders ──────────────────────────────────────────────────────────
    # The zip contains: Brain-Tumor-Classification-DataSet-master/Testing/...
    # We map their folder names to our CLASSES: ['glioma', 'meningioma', 'notumor', 'pituitary']
    mapping = {
        "glioma_tumor": "glioma",
        "meningioma_tumor": "meningioma",
        "pituitary_tumor": "pituitary",
        "no_tumor": "notumor"
    }

    print("\n[Organize] Sorting images into 4-class structure...")
    counts = {cls: 0 for cls in mapping.values()}

    for root, dirs, files in os.walk(extract_dir):
        if "Testing" not in root:
            continue
            
        folder_name = os.path.basename(root)
        if folder_name in mapping:
            target_class = mapping[folder_name]
            out_dir = os.path.join(DEST_DIR, "organised", target_class)
            os.makedirs(out_dir, exist_ok=True)
            
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    src = os.path.join(root, f)
                    dst = os.path.join(out_dir, f"{target_class}_{counts[target_class]:04d}.jpg")
                    shutil.copy2(src, dst)
                    counts[target_class] += 1

    print("\n[Ready] External validation set prepared (4-Class):")
    for cls, count in counts.items():
        print(f"  {cls:<12}: {count} images")
    print(f"  TOTAL         : {sum(counts.values())} images")
    
    print(f"\n  Location: {DEST_DIR}/organised/")
    print("\n[Done] Now run scripts/quick_demo.py")

if __name__ == "__main__":
    main()
