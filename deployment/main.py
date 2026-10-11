"""FastAPI service exposing the trained YOLOv8 detector.

Weights are located through the MODEL_PATH environment variable, defaulting
to models/best.pt. The model is loaded lazily on the first prediction request.
"""
import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best.pt")
DEFAULT_CONF = float(os.environ.get("CONF_THRESHOLD", "0.25"))

app = FastAPI(
    title="Military Object Detection API",
    description="REST API for serving the trained YOLOv8 detection model.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model():
    """Load and cache the YOLO model. Raises FileNotFoundError if weights are missing."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at '{MODEL_PATH}'")

    from ultralytics import YOLO

    return YOLO(MODEL_PATH)


@app.post("/predict/")
async def predict_image(file: UploadFile = File(...), conf: float = DEFAULT_CONF):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        model = get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    result = model.predict(source=img, conf=conf, verbose=False)[0]
    predictions = [
        {
            "class_id": int(box.cls[0]),
            "class_name": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],  # xmin, ymin, xmax, ymax
        }
        for box in result.boxes
    ]
    return {"status": "success", "predictions": predictions}


@app.get("/health")
def health_check():
    try:
        import torch

        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "weights_found": os.path.exists(MODEL_PATH),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
