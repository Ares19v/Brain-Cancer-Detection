r"""
NeuroScan — Quick Demo Script (4-Class Exact Match)
==================================================
Randomly selects 30 images from the external validation set,
runs inference, and checks for EXACT class match.

Example: MENINGIOMA vs MENINGIOMA = PASS
         MENINGIOMA vs PITUITARY   = FAIL

Run from project root:
    C:\Users\Devansh Tyagi\miniconda3\envs\ml\python.exe scripts\quick_demo.py
"""

import os
import random
import torch
import timm
from PIL import Image
from torchvision import transforms

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "weights/efficientnet_b0.pth"
VAL_DIR = "datasets/external_validation/organised"
# Class order must match training
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

def run_demo():
    if not os.path.exists(VAL_DIR):
        print(f"{Colors.RED}Error: External dataset not found at {VAL_DIR}{Colors.RESET}")
        print("Please run 'python scripts/download_external_val.py' first.")
        return

    model = load_model()
    
    # Collect images from all 4 subfolders
    all_images = []
    for cls in CLASSES:
        folder = os.path.join(VAL_DIR, cls)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append((os.path.join(folder, f), cls))

    if not all_images:
        print(f"{Colors.RED}Error: No images found in {VAL_DIR}. Run downloader again.{Colors.RESET}")
        return

    # Pick 30 random samples
    samples = random.sample(all_images, min(30, len(all_images)))
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- NeuroScan AI: External Validation (Exact Class Match) ---{Colors.RESET}\n")
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
            
            # EXACT MATCH CHECK
            is_pass = (ground_truth == pred_label)
            
            status_text = f"{Colors.GREEN}PASS{Colors.RESET}" if is_pass else f"{Colors.RED}FAIL{Colors.RESET}"
            if is_pass: correct_count += 1
            
            print(f"{i:<3} | {ground_truth.upper():<12} | {pred_label.upper():<12} | {status_text}")
            
        except Exception as e:
            print(f"{i:<3} | Error: {str(e)}")

    print("-" * 52)
    score_color = Colors.GREEN if correct_count >= 24 else (Colors.YELLOW if correct_count >= 18 else Colors.RED)
    print(f"{Colors.BOLD}Final Score: {score_color}{correct_count}/30 Correct{Colors.RESET}")
    print(f"{Colors.BOLD}Accuracy: {(correct_count/30)*100:.1f}%{Colors.RESET}\n")

if __name__ == "__main__":
    os.system('') # Enable ANSI colors
    run_demo()
