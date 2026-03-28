import torch, timm, os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/efficientnet_b0.pth"
TEST_DIR = "datasets/brain_mri/Testing"
CLASSES = ['Glioma', 'Meningioma', 'Healthy', 'Pituitary']

# Load Model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_data = datasets.ImageFolder(TEST_DIR, transform=transform)
loader = DataLoader(test_data, batch_size=32, shuffle=False)

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in loader:
        outputs = model(imgs.to(DEVICE))
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

# Create Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Purples')
plt.title('Brain Tumor Detection: Confusion Matrix')
plt.ylabel('Actual Category')
plt.xlabel('AI Prediction')
plt.savefig('reports/final_confusion_matrix.png')
print("✅ Heatmap saved to reports/final_confusion_matrix.png")
