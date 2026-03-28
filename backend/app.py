from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from inference.predict import detect

app = FastAPI()

@app.post("/detect")
async def detect_tumor(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = detect(img)

    return {"detections": results}