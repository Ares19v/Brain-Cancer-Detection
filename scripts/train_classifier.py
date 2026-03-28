import torch, timm, os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

TRAIN_DIR = "datasets/brain_mri/Training"
SAVE_PATH = "weights/efficientnet_b0.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HEAVY TRANSFORMS TO BREAK "NOTUMOR" BIAS
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
# WEIGHTED SAMPLING: Makes the model see tumors more often than "no tumors"
weights = [5.0, 5.0, 1.0, 5.0] # [glioma, meningioma, notumor, pituitary]
class_weights = torch.FloatTensor(weights).to(DEVICE)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights) # KEY FIX
scaler = torch.amp.GradScaler('cuda')

print(f"🚀 PURPLETIER FINAL ATTEMPT: Weighted Training on {DEVICE}...")

for epoch in range(20):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/20")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.amp.autocast('cuda'):
            loss = criterion(model(imgs), labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f"✅ FINAL MODEL SAVED.")
