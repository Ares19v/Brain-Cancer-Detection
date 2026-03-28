import torch, timm, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Paths
TRAIN_DIR = "datasets/brain_mri/Training"
MODEL_PATH = "weights/efficientnet_b0.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HEAVY Augmentation to stop the "NOTUMOR" bias
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(25),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

# Load existing model
model = timm.create_model("efficientnet_b0", num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))

# Lower Learning Rate (Fine-tuning)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

print(f"🚀 Refining Model (Fine-Tuning)... Target: {DEVICE}")

for epoch in range(10):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda'):
            loss = criterion(model(imgs), labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"\n✅ REFINEMENT COMPLETE: Updated model saved.")
