r"""
NeuroScan — External Validation Test
======================================
Tests the model on the EXTERNAL validation dataset (never seen during training).
Source: Figshare "Brain MRI Images for Brain Tumor Detection" by Navoneel Chakrabarty
        (different from the Kaggle dataset used for training)

This dataset uses binary labels (tumor / notumor). Our validation logic:
  - tumor   images -> model should NOT predict HEALTHY (notumor class index 2)
  - notumor images -> model should predict HEALTHY (notumor class index 2)

Run from project root:
    python scripts\validate_external.py
    -- or with ml env for GPU --
    C:\Users\Devansh Tyagi\miniconda3\envs\ml\python.exe scripts\validate_external.py
"""

import os
import sys
import torch
import timm
from torchvision import transforms
from PIL import Image
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "efficientnet_b0.pth")
VAL_DIR    = os.path.join(os.path.dirname(__file__), "..", "datasets", "external_validation", "organised")
# Class order matches ImageFolder alphabetical order of TRAINING dataset:
# glioma=0, meningioma=1, notumor=2, pituitary=3
CLASSES       = ["glioma", "meningioma", "notumor", "pituitary"]
HEALTHY_IDX   = 2   # index of "notumor" (maps to HEALTHY in the app)
CONF_THRESHOLD = 0.75

def load_model():
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m.eval()
    return m

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict_image(model, img_path: str):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None, 0.0
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        conf, pred = torch.max(probs, 0)
    return CLASSES[pred.item()], conf.item()

def collect_images(folder: str):
    imgs = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            imgs.append(os.path.join(folder, f))
    return imgs

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    if not os.path.exists(VAL_DIR):
        print(f"[ERROR] External val dataset not found at {VAL_DIR}")
        print("        Run scripts\\download_external_val.py first.")
        sys.exit(1)

    print(f"[INFO] Device  : {DEVICE}")
    print(f"[INFO] Model   : {MODEL_PATH}")
    print(f"[INFO] Val dir : {VAL_DIR}")

    model = load_model()
    print("[INFO] Model loaded.\n")

    tumor_dir   = os.path.join(VAL_DIR, "tumor")
    notumor_dir = os.path.join(VAL_DIR, "notumor")

    tumor_imgs   = collect_images(tumor_dir)   if os.path.exists(tumor_dir)   else []
    notumor_imgs = collect_images(notumor_dir) if os.path.exists(notumor_dir) else []

    print(f"[Data] Tumor images  : {len(tumor_imgs)}")
    print(f"[Data] Notumor images: {len(notumor_imgs)}")
    print(f"[Data] Total         : {len(tumor_imgs) + len(notumor_imgs)}\n")

    if len(tumor_imgs) + len(notumor_imgs) == 0:
        print("[ERROR] No images found. Check the dataset directory.")
        sys.exit(1)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    # Tumor group: model should NOT say "notumor" (HEALTHY)
    #   PASS  = any tumor class (glioma/meningioma/pituitary)
    #   FAIL  = "notumor" (called healthy when it has cancer — CRITICAL)
    # Notumor group: model should say "notumor" (HEALTHY)
    #   PASS  = "notumor"
    #   FAIL  = any tumor class (false alarm)

    tumor_pass      = 0
    tumor_fail      = 0        # CRITICAL: cancer called healthy
    tumor_inconc    = 0
    tumor_pred_dist = defaultdict(int)

    notumor_pass    = 0
    notumor_fail    = 0
    notumor_inconc  = 0

    print("Processing tumor images...")
    for img_path in tumor_imgs:
        pred, conf = predict_image(model, img_path)
        if pred is None:
            continue
        if conf < CONF_THRESHOLD:
            tumor_inconc += 1
            continue
        tumor_pred_dist[pred] += 1
        if pred == "notumor":
            tumor_fail += 1   # CRITICAL: cancer missed
        else:
            tumor_pass += 1

    print("Processing notumor images...")
    for img_path in notumor_imgs:
        pred, conf = predict_image(model, img_path)
        if pred is None:
            continue
        if conf < CONF_THRESHOLD:
            notumor_inconc += 1
            continue
        if pred == "notumor":
            notumor_pass += 1
        else:
            notumor_fail += 1

    # ── Report ────────────────────────────────────────────────────────────────
    total_tested       = len(tumor_imgs) + len(notumor_imgs)
    total_conclusive   = tumor_pass + tumor_fail + notumor_pass + notumor_fail
    total_correct      = tumor_pass + notumor_pass
    overall_acc        = total_correct / total_conclusive * 100 if total_conclusive > 0 else 0

    tumor_sensitivity  = tumor_pass  / (tumor_pass + tumor_fail) * 100 if (tumor_pass + tumor_fail) > 0 else 0
    notumor_specificity = notumor_pass / (notumor_pass + notumor_fail) * 100 if (notumor_pass + notumor_fail) > 0 else 0

    print()
    print("=" * 65)
    print("  EXTERNAL VALIDATION REPORT (Never-Seen Dataset)")
    print("  Source: Figshare — Brain MRI Dataset (Navoneel Chakrabarty)")
    print("=" * 65)
    print(f"\n  Total images tested   : {total_tested}")
    print(f"  Conclusive predictions: {total_conclusive}")
    print(f"  Inconclusive (<75%)   : {tumor_inconc + notumor_inconc}")
    print()
    print(f"  TUMOR DETECTION (n={len(tumor_imgs)})")
    print(f"    Correctly identified as tumor : {tumor_pass}")
    print(f"    MISSED (called HEALTHY)        : {tumor_fail}  {'<< CRITICAL' if tumor_fail > 0 else '<< NONE - GOOD'}")
    print(f"    Inconclusive (low conf)        : {tumor_inconc}")
    print(f"    Sensitivity (recall)           : {tumor_sensitivity:.1f}%")
    print()
    print(f"    Tumor prediction breakdown:")
    for cls, cnt in sorted(tumor_pred_dist.items(), key=lambda x: -x[1]):
        print(f"      {cls:<12}: {cnt}")
    print()
    print(f"  HEALTHY DETECTION (n={len(notumor_imgs)})")
    print(f"    Correctly identified as healthy: {notumor_pass}")
    print(f"    False alarms (called tumor)    : {notumor_fail}")
    print(f"    Inconclusive (low conf)        : {notumor_inconc}")
    print(f"    Specificity                    : {notumor_specificity:.1f}%")
    print()
    print("-" * 65)
    print(f"  OVERALL ACCURACY   : {overall_acc:.1f}%  ({total_correct}/{total_conclusive})")
    print(f"  SENSITIVITY        : {tumor_sensitivity:.1f}%  (tumors correctly found)")
    print(f"  SPECIFICITY        : {notumor_specificity:.1f}%  (healthy correctly cleared)")
    print("=" * 65)

    # ── Pass/Fail judgment ────────────────────────────────────────────────────
    critical_fail = tumor_fail > int(len(tumor_imgs) * 0.05)   # >5% cancers called healthy
    if critical_fail:
        print(f"\n[FAIL] {tumor_fail} tumors were classified as HEALTHY. Retraining needed.")
        sys.exit(2)
    elif overall_acc < 80:
        print(f"\n[WARN] Overall accuracy {overall_acc:.1f}% on external set is low.")
        sys.exit(2)
    else:
        print(f"\n[PASS] Model is safe. At most {tumor_fail} tumors missed. Sensitivity: {tumor_sensitivity:.1f}%")
        sys.exit(0)

if __name__ == "__main__":
    main()
