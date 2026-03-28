import torch, timm, os, random
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image

Y_PATH = "backend/runs/detect/train4/weights/best.pt"
E_PATH = "weights/efficientnet_b0.pth"
T_DATA = "datasets/brain_mri/Testing"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

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
    if not res[0].boxes:
        return "NO_DETECTION", 0.0
    
    b = res[0].boxes[0].xyxy[0].cpu().numpy()
    crop = img.crop((b[0], b[1], b[2], b[3]))
    tens = trans(crop).unsqueeze(0).to(DEV)
    
    with torch.no_grad():
        out = e_mod(tens)
        prob = torch.nn.functional.softmax(out, dim=1)[0]
        conf, pred = torch.max(prob, 0)
        
        # HYPER-VIGILANT LOGIC:
        # If the model picks NOTUMOR but isn't 98% sure, we look at tumor classes
        if CLASSES[pred] == 'notumor' and conf < 0.98:
            # Create a copy of probabilities without the 'notumor' index
            tumor_probs = prob.clone()
            tumor_probs[2] = 0 # Mask the 'notumor' score
            new_conf, new_pred = torch.max(tumor_probs, 0)
            return f"{CLASSES[new_pred].upper()} (Suspected)", new_conf.item()
            
    return CLASSES[pred].upper(), conf.item()

print(f"\n--- PurpleTier Hyper-Vigilant Stress Test (20 Samples) ---")
imgs = [os.path.join(r, f) for r, d, fs in os.walk(T_DATA) for f in fs if f.endswith('.jpg')]
for s in random.sample(imgs, 20):
    label, conf = predict(s)
    marker = "🧬" if "NOTUMOR" not in label else "✅"
    print(f"{marker} [{os.path.basename(s)}] -> {label} ({conf*100:.1f}%)")
