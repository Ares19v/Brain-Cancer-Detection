import torch, timm, os, wandb
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# 1. Stable Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64  # Stable for 8GB VRAM
CHECKPOINT_PATH = "weights/checkpoint.pth"
FINAL_MODEL_PATH = "weights/efficientnet_b0.pth"

# Clinically Balanced Weights
# Glioma(2.5), Meningioma(1.8), NoTumor(1.0), Pituitary(2.2)
WEIGHT_TENSOR = torch.tensor([2.5, 1.8, 1.0, 2.2]).to(DEVICE)

wandb.init(project="tumor-detector-v1", name="RTX5060-Stable-Run")

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
    num_workers=4, 
    pin_memory=True
)

# 3. Model & Optimizer
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=4).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss(weight=WEIGHT_TENSOR)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 4. Resume Logic
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    try:
        ckpt = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"🔄 Resuming from Epoch {start_epoch+1}")
    except:
        print("⚠️ Checkpoint incompatible, starting fresh.")

# 5. Stable Training Loop
print(f"🚀 Training on {torch.cuda.get_device_name(0)} (Standard Mode)...")
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
        wandb.log({"batch_loss": loss.item()})
        pbar.set_postfix(loss=total_loss/len(loader))

    # Save Checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_PATH)
    wandb.log({"epoch_loss": total_loss/len(loader), "epoch": epoch+1})

torch.save(model.state_dict(), FINAL_MODEL_PATH)
wandb.finish()
print("✅ TRAINING COMPLETED STABLY.")
