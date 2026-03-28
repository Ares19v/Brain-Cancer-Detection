import torch, timm, os, wandb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Configuration & W&B Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128  
CHECKPOINT_PATH = "weights/checkpoint.pth"
FINAL_MODEL_PATH = "weights/efficientnet_b0.pth"
# Balanced Weights: Glioma(2.0), Meningioma(1.5), NoTumor(1.0), Pituitary(2.0)
WEIGHT_TENSOR = torch.tensor([2.0, 1.5, 1.0, 2.0]).to(DEVICE)

wandb.init(
    project="tumor-detector-v1",
    name="Omen16-RTX5060-HighUtil",
    config={
        "learning_rate": 1e-4,
        "architecture": "EfficientNet-B0",
        "batch_size": BATCH_SIZE,
        "device": torch.cuda.get_device_name(0),
        "weighted_loss": True
    }
)

# 2. Data Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("datasets/brain_mri/Training", transform=transform)
loader = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=8,        # <--- Increased from 4 to 8 to use more i7 cores
    pin_memory=True, 
    prefetch_factor=4,    # <--- Tells the CPU to always have 4 batches ready in advance
    persistent_workers=True # <--- Keeps the worker threads alive between epochs
)

# 3. Model & Optimizer
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=WEIGHT_TENSOR)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 4. Resume Logic
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f"🔄 Checkpoint found! Resuming...")
    ckpt = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1

# 5. Training Loop
print(f"🔥 Training on {torch.cuda.get_device_name(0)}...")
for epoch in range(start_epoch, 30):
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
        current_batch_loss = loss.item()
        
        # Log to W&B in real-time
        wandb.log({"batch_loss": current_batch_loss})
        pbar.set_postfix(loss=total_loss/len(loader))

    avg_epoch_loss = total_loss/len(loader)
    wandb.log({"epoch": epoch+1, "epoch_loss": avg_epoch_loss})

    # Save Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_PATH)
    print(f"💾 Epoch {epoch+1} complete. Checkpoint saved.")

torch.save(model.state_dict(), FINAL_MODEL_PATH)
wandb.finish()
print("✅ TRAINING COMPLETE. Model ready for Stress Test.")
