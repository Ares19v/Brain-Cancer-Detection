from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

input_folder = "../../datasets/brain_mri/Testing"
output_folder = "../../results"

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):

    for file in files:

        if file.endswith(".jpg") or file.endswith(".png"):

            image_path = os.path.join(root, file)

            img = cv2.imread(image_path)

            if img is None:
                continue

            results = model(img)

            for r in results:

                boxes = r.boxes.xyxy

                for box in boxes:

                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

            save_path = os.path.join(output_folder,file)

            cv2.imwrite(save_path,img)

print("Detection complete. Results saved to /results folder.")