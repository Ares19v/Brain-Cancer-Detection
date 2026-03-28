import torch, timm, os, wandb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Robust Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
FINAL_MODEL_PATH = "weights/efficientnet_b0.pth"

# 2. Heavier Data Augmentation (To stop the model from "memorizing")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Random lighting
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("datasets/brain_mri/Training", transform=transform)
loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# 3. Model with DROPOUT
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4, drop_rate=0.3).to(DEVICE)

# 4. LABEL SMOOTHING (Prevents 100% overconfidence)
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

wandb.init(project="tumor-detector-v1", name="Robust-Training-V2")

print(f"🛡️ Starting Robust Training on {torch.cuda.get_device_name(0)}...")
for epoch in range(25): # 25 is enough with better augmentation
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/25")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/len(loader))
    
    wandb.log({"epoch_loss": total_loss/len(loader), "epoch": epoch+1})

torch.save(model.state_dict(), FINAL_MODEL_PATH)
wandb.finish()
print("✅ ROBUST MODEL SAVED.")
