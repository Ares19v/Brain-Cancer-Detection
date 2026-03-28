import torch, timm, os, wandb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# 1. Config for Maximum Speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256  # Maximizing the 8GB VRAM on RTX 5060
CHECKPOINT_PATH = "weights/checkpoint.pth"
FINAL_MODEL_PATH = "weights/efficientnet_b0.pth"

# Optimized Weights: Focus on Glioma & Pituitary (historically harder)
WEIGHT_TENSOR = torch.tensor([2.5, 1.8, 1.0, 2.2]).to(DEVICE)

wandb.init(project="tumor-detector-v1", name="RTX5060-AMP-Turbo")

# 2. Optimized Data Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder("datasets/brain_mri/Training", transform=transform)
loader = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=8,        # Using more i7 cores
    pin_memory=True, 
    prefetch_factor=4,    # Keep GPU fed
    persistent_workers=True
)

# 3. Model & Optimizer
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=WEIGHT_TENSOR)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4) # Slightly higher LR for larger batch
scaler = GradScaler() # For Mixed Precision

# 4. Resume Logic
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1

# 5. Fast Training Loop
print(f"🚀 Turbo Training on {torch.cuda.get_device_name(0)}...")
for epoch in range(start_epoch, 30):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/30")
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        
        # MIXED PRECISION FORWARD PASS
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        # SCALED BACKWARD PASS
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        wandb.log({"batch_loss": loss.item()})
        pbar.set_postfix(loss=total_loss/len(loader))

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_PATH)
    wandb.log({"epoch_loss": total_loss/len(loader)})

torch.save(model.state_dict(), FINAL_MODEL_PATH)
wandb.finish()
print("✅ TRAINING FINISHED AT HIGH SPEED.")
