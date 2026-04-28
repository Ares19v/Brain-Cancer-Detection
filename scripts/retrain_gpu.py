r"""
NeuroScan - GPU-Optimised Targeted Retraining (RTX 5060 / Blackwell)
=====================================================================
Run from project root:
    C:\Users\Devansh Tyagi\miniconda3\envs\ml\python.exe scripts\retrain_gpu.py
"""

import os
import sys
import copy
import torch
import timm
from torch.amp import GradScaler, autocast
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_DIR     = "datasets/brain_mri/Training"
VAL_DIR       = "datasets/brain_mri/Testing"
MODEL_IN      = "weights/efficientnet_b0.pth"
MODEL_OUT     = "weights/efficientnet_b0.pth"
BATCH_SIZE    = 64
EPOCHS        = 20
LR            = 3e-5
# glioma=0, meningioma=1, notumor=2, pituitary=3
# 27 glioma were predicted as HEALTHY — boost glioma to 5x
CLASS_WEIGHTS_CPU = [5.0, 1.2, 1.5, 1.2]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomAffine(degrees=0, shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def make_loaders(device):
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=TRAIN_TRANSFORM)
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=VAL_TRANSFORM)

    classes = train_dataset.classes
    print(f"[Data] Classes: {classes}")
    print(f"[Data] Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")

    label_counts   = Counter(train_dataset.targets)
    sample_weights = [1.0 / label_counts[t] for t in train_dataset.targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, classes


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss  = 0.0
    per_correct = [0] * 4
    per_total   = [0] * 4
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast(device_type="cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            total_loss += loss.item()
            _, preds = torch.max(out, 1)
            for t, p in zip(labels, preds):
                per_total[t.item()] += 1
                if t == p:
                    per_correct[t.item()] += 1
    total_c = sum(per_correct)
    total_t = sum(per_total)
    per_acc = [per_correct[i] / per_total[i] * 100 if per_total[i] > 0 else 0 for i in range(4)]
    return total_loss / len(val_loader), total_c / total_t * 100, per_acc


def main():
    # ── Enforce GPU ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available.")
        print("        Run: pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --upgrade")
        sys.exit(1)

    device = torch.device("cuda")
    cap    = torch.cuda.get_device_capability()
    print(f"[GPU] {torch.cuda.get_device_name(0)}  |  CUDA {torch.version.cuda}  |  sm_{cap[0]}{cap[1]}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    class_weights = torch.tensor(CLASS_WEIGHTS_CPU, dtype=torch.float32).to(device)

    train_loader, val_loader, classes = make_loaders(device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_IN, map_location=device, weights_only=True))
    print(f"[Model] Loaded weights from {MODEL_IN}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler    = GradScaler()

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n[Baseline] Evaluating before retraining...")
    base_loss, base_acc, base_per = evaluate(model, val_loader, criterion, device)
    print(f"  Overall: {base_acc:.1f}%  loss: {base_loss:.4f}")
    for c, a in zip(classes, base_per):
        print(f"  {c:<12}: {a:.1f}%")

    best_loss  = base_loss
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n[Train] {EPOCHS} epochs  |  batch={BATCH_SIZE}  |  lr={LR}  |  AMP=ON")
    print(f"{'EP':>3}  {'T_LOSS':>8}  {'V_LOSS':>8}  {'OVERALL':>8}  {'GLIOMA':>8}  {'NOTUMOR':>8}  NOTE")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for imgs, labels in train_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()

        val_loss, val_acc, per_acc = evaluate(model, val_loader, criterion, device)
        note = ""
        if val_loss < best_loss:
            best_loss  = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            note = "<-- BEST"

        print(f"{epoch:>3}  {epoch_loss/len(train_loader):>8.4f}  {val_loss:>8.4f}  "
              f"{val_acc:>7.1f}%  {per_acc[0]:>7.1f}%  {per_acc[2]:>7.1f}%  {note}")

    # ── Save best ─────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_OUT)
    print(f"\n[Save] Best model (epoch {best_epoch}) saved to {MODEL_OUT}")

    # ── Final report ──────────────────────────────────────────────────────────
    _, final_acc, final_per = evaluate(model, val_loader, criterion, device)
    print()
    print("=" * 55)
    print("  FINAL RESULTS AFTER GPU RETRAINING")
    print("=" * 55)
    for c, a in zip(classes, final_per):
        flag = "  << STILL LOW" if a < 85 else ""
        print(f"  {c:<12}: {a:.1f}%{flag}")
    print(f"  {'OVERALL':<12}: {final_acc:.1f}%")
    print("=" * 55)

    if final_per[0] < 85:
        print(f"\n[WARNING] Glioma at {final_per[0]:.1f}% — try more epochs or increase glioma weight.")
        sys.exit(2)
    else:
        print(f"\n[PASS] Glioma: {final_per[0]:.1f}%  Overall: {final_acc:.1f}%")
        sys.exit(0)


if __name__ == "__main__":
    main()
