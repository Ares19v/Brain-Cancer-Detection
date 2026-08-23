r"""
NeuroScan — Quick Demo Script (4-Class Exact Match)
==================================================
Location-aware script that works from any directory.
Randomly selects N images from the external validation set.
"""

import os
import sys
import random
import torch
import timm
from PIL import Image
from torchvision import transforms

# --- PATH LOGIC ---
# Get the absolute path to the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The project root is one level up from 'scripts/'
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(PROJECT_ROOT, "weights", "efficientnet_b0.pth")
VAL_DIR = os.path.join(PROJECT_ROOT, "datasets", "external_validation", "organised")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def run_demo(num_samples=30):
    if not os.path.exists(VAL_DIR):
        print(f"{Colors.RED}Error: External dataset not found at:{Colors.RESET}")
        print(f"  {VAL_DIR}")
        return

    model = load_model()
    
    all_images = []
    for cls in CLASSES:
        folder = os.path.join(VAL_DIR, cls)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append((os.path.join(folder, f), cls))

    if not all_images:
        print(f"{Colors.RED}Error: No images found in {VAL_DIR}{Colors.RESET}")
        return

    samples = random.sample(all_images, min(num_samples, len(all_images)))
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- NeuroScan AI: External Validation ({len(samples)} Samples) ---{Colors.RESET}\n")
    print(f"{'#':<3} | {'GROUND TRUTH':<12} | {'PREDICTION':<12} | {'RESULT':<8}")
    print("-" * 52)

    correct_count = 0
    for i, (img_path, ground_truth) in enumerate(samples, 1):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(tensor)
                probs = torch.nn.functional.softmax(output, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
                pred_label = CLASSES[pred_idx.item()]
            
            is_pass = (ground_truth == pred_label)
            status_text = f"{Colors.GREEN}PASS{Colors.RESET}" if is_pass else f"{Colors.RED}FAIL{Colors.RESET}"
            if is_pass: correct_count += 1
            print(f"{i:<3} | {ground_truth.upper():<12} | {pred_label.upper():<12} | {status_text}")
        except Exception as e:
            print(f"{i:<3} | Error: {str(e)}")

    print("-" * 52)
    acc = (correct_count / len(samples)) * 100
    score_color = Colors.GREEN if acc >= 80 else (Colors.YELLOW if acc >= 60 else Colors.RED)
    print(f"{Colors.BOLD}Final Score: {score_color}{correct_count}/{len(samples)} Correct{Colors.RESET}")
    print(f"{Colors.BOLD}Accuracy: {acc:.1f}%{Colors.RESET}\n")

if __name__ == "__main__":
    os.system('') # Enable ANSI colors
    n = 30
    if len(sys.argv) > 1:
        try: n = int(sys.argv[1])
        except: pass
    run_demo(n)
