import torch, timm, os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Lowered weights to reduce Meningioma bias
WEIGHTS = torch.tensor([2.0, 1.5, 1.0, 2.0]).to(DEVICE) 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("datasets/brain_mri/Training", transform=transform)
loader = DataLoader(train_data, batch_size=32, shuffle=True)

model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=WEIGHTS)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print("🚀 Starting Balanced Retraining (30 Epochs)...")
for epoch in range(30):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/30")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/len(loader))

torch.save(model.state_dict(), "weights/efficientnet_b0.pth")
print("✅ BALANCED MODEL SAVED.")
