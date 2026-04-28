"""
Model validation script - runs a full test against all test images.
Reports per-class accuracy and overall accuracy.
"""
import torch
import timm
import os
import sys
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "weights", "efficientnet_b0.pth")
TEST_DIR   = os.path.join(os.path.dirname(__file__), "..", "datasets", "brain_mri", "Testing")
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']
# Note: main.py uses CLASSES = ['GLIOMA','MENINGIOMA','HEALTHY','PITUITARY']
# The dataset folder is named 'notumor' which maps to index 2 in ImageFolder (alphabetical)
# ImageFolder order: glioma=0, meningioma=1, notumor=2, pituitary=3

def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Model : {MODEL_PATH}")
    print(f"[INFO] Data  : {TEST_DIR}")

    if not os.path.exists(MODEL_PATH):
        print("[ERROR] Model weight file not found!")
        sys.exit(1)

    if not os.path.exists(TEST_DIR):
        print("[ERROR] Test dataset directory not found!")
        sys.exit(1)

    # Load model
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print("[INFO] Model loaded successfully.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
    loader    = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

    print(f"[INFO] Test samples: {len(test_data)}")
    print(f"[INFO] Class order (ImageFolder): {test_data.classes}")

    # Evaluate
    per_class_correct = defaultdict(int)
    per_class_total   = defaultdict(int)
    total_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            for true, pred in zip(labels.numpy(), preds.cpu().numpy()):
                cls = test_data.classes[true]
                per_class_total[cls] += 1
                if true == pred:
                    per_class_correct[cls] += 1
                    total_correct += 1
                total += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  BRAIN TUMOR MODEL VALIDATION REPORT")
    print("=" * 55)
    print(f"  {'CLASS':<14} {'CORRECT':>8} {'TOTAL':>8} {'ACC':>8}")
    print("-" * 55)

    any_bad = False
    for cls in sorted(per_class_total.keys()):
        correct = per_class_correct[cls]
        total_c = per_class_total[cls]
        acc     = correct / total_c * 100 if total_c > 0 else 0
        flag    = "  << LOW" if acc < 70 else ""
        if acc < 70:
            any_bad = True
        print(f"  {cls:<14} {correct:>8} {total_c:>8} {acc:>7.1f}%{flag}")

    print("-" * 55)
    overall = total_correct / total * 100 if total > 0 else 0
    print(f"  {'OVERALL':<14} {total_correct:>8} {total:>8} {overall:>7.1f}%")
    print("=" * 55)

    if any_bad:
        print("\n[WARNING] One or more classes have accuracy below 70%.")
        print("[ACTION ] Retraining recommended.")
        sys.exit(2)
    elif overall < 85:
        print(f"\n[WARNING] Overall accuracy {overall:.1f}% is below 85% target.")
        sys.exit(2)
    else:
        print(f"\n[PASS] Model is healthy. Overall accuracy: {overall:.1f}%")
        sys.exit(0)

if __name__ == "__main__":
    main()
