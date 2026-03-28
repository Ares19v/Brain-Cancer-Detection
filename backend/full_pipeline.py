import torch
import cv2
import os
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Load tumor classifier
# -------------------------------

classifier = models.efficientnet_b0(weights=None)

classifier.classifier[1] = torch.nn.Linear(
    classifier.classifier[1].in_features,
    4
)

classifier.load_state_dict(
    torch.load("../weights/brain_model.pth", map_location=device)
)

classifier = classifier.to(device)
classifier.eval()

classes = [
    "glioma",
    "meningioma",
    "pituitary",
    "notumor"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------------
# Load YOLO detector
# -------------------------------

detector = YOLO("yolov8n.pt")

# -------------------------------
# Input MRI
# -------------------------------

image_path = "../datasets/brain_mri/Testing/glioma/Te-glTr_0001.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()

# -------------------------------
# Tumor classification
# -------------------------------

pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

input_tensor = transform(pil_img).unsqueeze(0).to(device)

with torch.no_grad():

    outputs = classifier(input_tensor)

    _, pred = torch.max(outputs,1)

tumor_type = classes[pred.item()]

print("Tumor Type:", tumor_type)

# -------------------------------
# Tumor localization (YOLO)
# -------------------------------

results = detector(img)

for r in results:

    boxes = r.boxes.xyxy

    for box in boxes:

        x1,y1,x2,y2 = map(int,box)

        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(
            img,
            tumor_type,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

# -------------------------------
# Save result
# -------------------------------

os.makedirs("../results",exist_ok=True)

output_path = "../results/detected_mri.jpg"

cv2.imwrite(output_path,img)

print("Result saved to:",output_path)