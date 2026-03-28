import torch, timm, os, random
from torchvision import transforms
from PIL import Image

# 1. Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/efficientnet_b0.pth"
TEST_DATA = "datasets/brain_mri/Testing"
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 2. Setup Model
print(f"🧬 Loading Final Model on {DEVICE}...")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Matching the training transformations exactly
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Execution
print(f"🔍 Starting Final 50-Image Stress Test...")
all_imgs = []
for root, dirs, files in os.walk(TEST_DATA):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_imgs.append(os.path.join(root, f))

samples = random.sample(all_imgs, min(50, len(all_imgs)))
correct = 0

print(f"\n{'IMAGE':<25} | {'TRUE LABEL':<12} | {'PREDICTION':<12} | {'STATUS'}")
print("-" * 75)

with torch.no_grad():
    for s in samples:
        true_label = os.path.basename(os.path.dirname(s)).upper()
        
        # Preprocess
        img = Image.open(s).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # Predict
        outputs = model(input_tensor)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(prob, 0)
        pred_label = CLASSES[pred.item()].upper()
        
        match = true_label == pred_label
        if match: correct += 1
        
        status = "✅ PASS" if match else f"❌ FAIL ({conf:.2f})"
        print(f"{os.path.basename(s):<25} | {true_label:<12} | {pred_label:<12} | {status}")

accuracy = (correct / len(samples)) * 100
print("-" * 75)
print(f"📊 FINAL STRESS TEST ACCURACY: {accuracy:.2f}% ({correct}/{len(samples)})")
