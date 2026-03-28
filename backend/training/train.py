from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("brain-tumor").project("brain-tumor-detection")
dataset = project.version(1).download("yolov8")

model = YOLO("yolov8m.pt")

model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    device=0
)

model.export(format="onnx")