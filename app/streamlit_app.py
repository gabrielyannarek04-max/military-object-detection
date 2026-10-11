"""Streamlit demo for the military vehicle detector.

Run from the repository root:

    streamlit run app/streamlit_app.py
"""
import os

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

WEIGHTS_CANDIDATES = [
    os.environ.get("MODEL_PATH", ""),
    "models/best.pt",
    "models/last.pt",
]
FALLBACK_WEIGHTS = "yolov8n.pt"

st.set_page_config(page_title="Military Object Detection", layout="wide")

st.title("Military Object Detection")
st.write(
    "Upload an image to run the detector. "
    "Confidence and NMS IoU thresholds can be adjusted in the sidebar."
)

st.sidebar.header("Inference settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.01, 1.00, 0.25, 0.01)
iou_threshold = st.sidebar.slider("IoU threshold (NMS)", 0.01, 1.00, 0.50, 0.01)
uploaded_file = st.sidebar.file_uploader("Image", type=["jpg", "jpeg", "png"])


@st.cache_resource
def load_model() -> YOLO:
    """Load the first available weights file, falling back to the base model."""
    for path in WEIGHTS_CANDIDATES:
        if path and os.path.exists(path):
            return YOLO(path)
    st.sidebar.warning(
        f"No trained weights found in models/. Using the generic {FALLBACK_WEIGHTS} "
        "model, which does not know the project classes."
    )
    return YOLO(FALLBACK_WEIGHTS)


model = load_model()

if uploaded_file is None:
    st.info("Upload an image from the sidebar to begin.")
else:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("The uploaded file could not be decoded as an image.")
        st.stop()

    col_input, col_output = st.columns(2)

    with col_input:
        st.subheader("Input")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_output:
        st.subheader("Detections")
        with st.spinner("Running inference"):
            result = model.predict(
                source=img_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False
            )[0]
        st.image(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
        st.write(f"Objects detected: {len(result.boxes)}")

        if len(result.boxes) > 0:
            rows = [
                {
                    "class": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                }
                for box in result.boxes
            ]
            st.dataframe(rows, use_container_width=True)
