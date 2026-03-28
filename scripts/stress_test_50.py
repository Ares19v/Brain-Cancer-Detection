import torch, timm, os, random
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

# 1. Configuration
BASE_DIR = "/mnt/c/Users/Devansh Tyagi/Desktop/Projects/brain-cancer-detection"
Y_PATH = os.path.join(BASE_DIR, "backend/runs/detect/train4/weights/best.pt")
E_PATH = os.path.join(BASE_DIR, "weights/efficientnet_b0.pth")
T_DATA = os.path.join(BASE_DIR, "datasets/brain_mri/Testing")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 2. Load Models
print(f"🧬 Loading Models on {DEV}...")
y_mod = YOLO(Y_PATH)
e_mod = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEV)
e_mod.load_state_dict(torch.load(E_PATH, map_location=DEV))
e_mod.eval()

trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(img_p):
    res = y_mod(img_p, conf=0.15, verbose=False)
    img = Image.open(img_p).convert('RGB')
    
    # If YOLO fails to find a box, we default to the full image to be safe
    if not res[0].boxes:
        crop = img
    else:
        b = res[0].boxes[0].xyxy[0].cpu().numpy()
        crop = img.crop((b[0], b[1], b[2], b[3]))
        
    tens = trans(crop).unsqueeze(0).to(DEV)
    
    with torch.no_grad():
        out = e_mod(tens)
        prob = torch.nn.functional.softmax(out, dim=1)[0]
        conf, pred = torch.max(prob, 0)
        label = CLASSES[pred.item()]
        
        # Hyper-Vigilant Logic
        if label == 'notumor' and conf < 0.98:
            prob[2] = 0
            new_conf, new_pred = torch.max(prob, 0)
            label = CLASSES[new_pred.item()]
            
    return label.upper()

# 3. Execution
print(f"🔍 Starting 50-Image Stress Test...")
all_imgs = []
# Fixed pathing to find images properly
for root, dirs, files in os.walk(T_DATA):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_imgs.append(os.path.join(root, f))

if len(all_imgs) < 50:
    print(f"⚠️ Only found {len(all_imgs)} images. Testing all of them.")
    samples = all_imgs
else:
    samples = random.sample(all_imgs, 50)

correct = 0
print(f"\n{'IMAGE':<25} | {'TRUE LABEL':<12} | {'PREDICTION':<12} | {'STATUS'}")
print("-" * 75)

for s in samples:
    # Ground truth is the folder name
    true_label = os.path.basename(os.path.dirname(s)).upper()
    pred_label = predict(s)
    
    match = true_label == pred_label
    if match: correct += 1
    
    status = "✅ PASS" if match else "❌ FAIL"
    print(f"{os.path.basename(s):<25} | {true_label:<12} | {pred_label:<12} | {status}")

accuracy = (correct / len(samples)) * 100
print("-" * 75)
print(f"📊 FINAL STRESS TEST ACCURACY: {accuracy:.2f}% ({correct}/{len(samples)})")
