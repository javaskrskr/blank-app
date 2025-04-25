# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile, os

@st.cache_resource(hash_funcs={YOLO: id})
def load_model():
    # Pull weights if not present
    weights = "best.pt"
    return YOLO(weights, task="segment", device="cpu")   # CPU-only

model = load_model()

st.title("GC Boundary Seg – demo (CPU)")
img_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

def extrema(box):
    # box = [x1,y1,x2,y2]
    x1,y1,x2,y2 = map(int,box)
    return {
        "max x, min y": (x2, y1),
        "max x, max y": (x2, y2),
        "min x, min y": (x1, y1),
        "min x, max y": (x1, y2),
    }

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Original", use_column_width=True)

    with st.spinner("Running inference…"):
        res = model.predict(img, imgsz=640, conf=0.25, iou=0.5, batch=1, device="cpu")[0]

    drawn = res.plot()          # numpy BGR
    st.image(drawn[:,:,::-1], caption="Detections", use_column_width=True)

    st.subheader("Detections")
    for i, (box, score, cls) in enumerate(zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls)):
        st.write(f"**#{i} – {model.names[int(cls)]}**")
        st.write(f"Confidence: **{score.item():.2%}**")
        for k,v in extrema(box).items():
            st.write(f"{k}: {v}")
        st.markdown("---")
