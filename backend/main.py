from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch, timm, io, os
from PIL import Image
from torchvision import transforms

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "../weights/efficientnet_b0.pth"
CLASSES = ['GLIOMA', 'MENINGIOMA', 'HEALTHY', 'PITUITARY']

# Load Robust Model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    conf_score = confidence.item()
    diagnosis = CLASSES[predicted_idx.item()]
    
    # --- SAFETY FILTER ---
    if conf_score < 0.75:
        return {
            "diagnosis": "INCONCLUSIVE",
            "confidence": round(conf_score * 100, 2),
            "status": "Inconclusive / Not a clear Brain MRI",
            "warning": "Low confidence score. Please upload a clear Brain MRI scan."
        }
    
    status = "Healthy" if diagnosis == "HEALTHY" else "Tumor Detected"
    
    return {
        "diagnosis": diagnosis,
        "confidence": round(conf_score * 100, 2),
        "status": status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
