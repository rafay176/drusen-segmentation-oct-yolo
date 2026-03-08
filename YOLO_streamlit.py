import streamlit as st
from PIL import Image
import numpy as np
import tempfile
import cv2
from ultralytics import YOLO
import torch
from io import BytesIO


@st.cache_resource
def load_model():
    model = YOLO("best_final.pt")  
    return model

model = load_model()

st.title("Drusen Segmentation")
st.write("Upload an OCT image to visualize")


conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

def post_process_mask(binary_mask):
    mask = (binary_mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def contour_simplify(mask, epsilon_factor=0.01):
    mask = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified_mask = np.zeros_like(mask)

    for cnt in contours:
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(simplified_mask, [approx], -1, 1, thickness=-1)

    return simplified_mask


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file).convert("RGB")

    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    
    results = model.predict(img_path, imgsz=1024, conf=conf_threshold)[0]
    original = np.array(image)
    mask_img = original.copy()

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()  
        num_masks = masks.shape[0]

        for i in range(num_masks):
            mask = masks[i]

            
            resized_mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

            
            refined_mask = post_process_mask(resized_mask)
            simplified_mask = contour_simplify(refined_mask)

            
            colored_mask = np.zeros_like(original)
            colored_mask[simplified_mask > 0.5] = [0, 255, 0]  

            
            mask_img = cv2.addWeighted(mask_img, 1.0, colored_mask, 0.4, 0)

        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.image(mask_img, caption="Drusen Segmentation (YOLOv11)", use_container_width=True)

        
        result_pil = Image.fromarray(mask_img)
        buf = BytesIO()
        result_pil.save(buf, format="PNG")
        st.download_button(
            label="Download Segmentation Result",
            data=buf.getvalue(),
            file_name="segmentation_result.png",
            mime="image/png"
        )

    else:
        st.warning("No segmentation masks were found.")
