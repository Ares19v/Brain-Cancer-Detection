from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="brain-tumor-mri-2/data.yaml",
    epochs=50,
    imgsz=640,

    batch=24,        # safe for 8GB GPU
    workers=6,       # avoids RAM overload
    cache="disk",    # prevents RAM crash

    device=0
)