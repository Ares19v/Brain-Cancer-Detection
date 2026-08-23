# PREP — NeuroScan AI (From-Scratch Study Guide)

Welcome to the beginner-friendly developer study guide for **NeuroScan AI**! In this guide, you will learn how to build, deploy, and package a production-grade deep learning classification model using PyTorch, FastAPI, and React.

---

## 1. Deep Learning Foundations: CNNs & EfficientNet-B0

To classify MRI images, this project uses a specialized Convolutional Neural Network (CNN) called **EfficientNet-B0**.

### What is a CNN?
* Standard neural networks treat images as a flat list of pixels, losing all spatial relationships (e.g. which pixel is next to which).
* **Convolutional Neural Networks (CNNs)** use sliding filters (kernels) to capture localized features like edges, shapes, and textures directly from 2D pixel grids.

### Why EfficientNet-B0?
* **Model Scaling**: Traditionally, if you wanted higher accuracy, you had to make models wider (more channels), deeper (more layers), or feed them higher-resolution images.
* **Compound Scaling**: EfficientNet balances these three dimensions (depth, width, and resolution) mathematically.
* **B0 Variant**: The "B0" variant is highly parameter-efficient (~5.3M parameters). It runs extremely fast on basic CPUs while yielding comparable accuracy to much larger networks.

### What is Transfer Learning?
Instead of training our model from scratch (which requires millions of images and massive GPU budgets), we utilize **Transfer Learning**. We load a model pre-trained on **ImageNet** (14 million generic images) so it already understands basic structures (circles, shading, edges), and then **fine-tune** the final classification layers strictly on our brain MRI dataset.

---

## 2. The PyTorch Inference Pipeline

Once fine-tuned weights are saved to a file (`.pth`), we run local predictions using a structured PyTorch pipeline:

```
┌─────────────┐   PIL RGB   ┌────────────┐   [1, 3, 224, 224]   ┌───────────┐   Logits   ┌─────────┐   Diagnosis
│ Image Upload├────────────►│ Transforms ├─────────────────────►│  PyTorch  ├───────────►│ Softmax ├────────────►
│  (PNG/JPEG) │             │ (Rescale)  │      Tensor          │ Inference │            │ (Prob%) │  (GLIOMA)
└─────────────┘             └────────────┘                      └───────────┘            └─────────┘
```

### The Standard PyTorch Inference Pattern:
```python
import torch
import timm
from torchvision import transforms
from PIL import Image

# 1. Define image transformations (Must match training preprocessing!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean & std dev
])

# 2. Instantiate and load model
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=4)
model.load_state_dict(torch.load("weights/efficientnet_b0.pth", map_location="cpu"))
model.eval() # Disable dropout and batch-normalization update layers

# 3. Predict!
img = Image.open("sample_mri.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0) # Add a batch dimension: [C, H, W] -> [1, C, H, W]

with torch.no_grad(): # Disable gradient tracking to accelerate inference and reduce memory
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidence, predicted_class_idx = torch.max(probabilities, dim=0)

print(f"Predicted Class: {predicted_class_idx.item()} with {confidence.item() * 100:.2f}% confidence")
```

---

## 3. Clinical Safety & Confidence-Gating

AI models are statistically prone to **hallucination**—when fed an image of a cat or an empty file, they will still make a guess with a random confidence output.

* **Confidence Gating**: NeuroScan enforces a strict `CONF_THRESH = 0.75` (75%) safety barrier. If the highest class probability is less than 75%, it overrides the diagnosis and outputs **INCONCLUSIVE**.
* **Ethics / Disclaimer**: Medical AI should strictly serve as an assistant (triage tool) to reduce radiologist fatigue. Final decisions must always be validated by certified radiologists.

---

## 4. Multi-Stage Docker Architectures

To build lightweight, secure production containers, we use **Multi-Stage Dockerfiles**.

### How Multi-stage Works:
1. **Stage 1 (Build)**: Use a heavy image containing compilation utilities (like compilers, pip wheels) to build dependencies.
2. **Stage 2 (Run)**: Copy *only* the compiled assets into a clean, minimal runtime image, leaving compilers and caching waste behind.

This reduces backend container footprints by hundreds of megabytes.

---

## 5. Exercises & Self-Guided Challenges

1. **Add Grad-CAM Activation Mapping**: Write a script in `scripts/visualize.py` that utilizes Grad-CAM to highlight which pixels/regions of the MRI image the CNN focused on to trigger its classification decision.
2. **Add a Double-Click Blur Filter**: In the frontend React application, prevent users from seeing low-quality previews by applying a warning blur CSS class (`blur-md`) if they upload a file with an invalid filename or image dimensions.
3. **Save Predictions locally**: Configure the FastAPI backend to log all predictions and confidence scores into a simple local SQLite database or CSV audit trail for historical accuracy tracking.
