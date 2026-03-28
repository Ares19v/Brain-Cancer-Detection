import torch, os, timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

E_PATH = "weights/efficientnet_b0.pth"
TEST_DIR = "datasets/brain_mri/Testing"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

model = timm.create_model("efficientnet_b0", num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(E_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(test_data, batch_size=32, shuffle=False)

all_preds, all_labels = [], []
print("📊 Evaluating full test set...")
with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

os.makedirs('outputs', exist_ok=True)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Final Model: Confusion Matrix')
plt.savefig('outputs/confusion_matrix.png')
print("\n📈 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASSES))
print("\n✅ Saved to: outputs/confusion_matrix.png")
