from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect(image):

    results = model(image)

    detections = []

    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls
        names = model.names

        for box, cls in zip(boxes, classes):
            x1,y1,x2,y2 = map(int, box)

            detections.append({
                "tumor_type": names[int(cls)],
                "region": f"{x1},{y1},{x2},{y2}"
            })

    return detections