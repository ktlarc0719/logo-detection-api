import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from PIL import ImageDraw
import json

st.set_page_config(page_title="Logo Detection", layout="wide")

st.title("Logo Detection App")
st.markdown("Upload an image to detect logos")

# API endpoint
API_URL = "http://localhost:8000"

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidence = st.checkbox("Show Confidence", value=True)

# Main content
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # Send to API
    if st.button("Detect Logos"):
        with st.spinner("Detecting logos..."):
            try:
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Send request to API
                files = {"file": ("image.png", img_byte_arr, "image/png")}
                response = requests.post(f"{API_URL}/detect", files=files)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Draw bounding boxes
                    img_with_boxes = image.copy()
                    draw = ImageDraw.Draw(img_with_boxes)
                    
                    detections = []
                    for detection in results.get("detections", []):
                        conf = detection["confidence"]
                        if conf >= confidence_threshold:
                            bbox = detection["bbox"]
                            x1, y1, x2, y2 = bbox
                            
                            # Draw rectangle
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            
                            # Draw label and confidence
                            if show_labels or show_confidence:
                                label_parts = []
                                if show_labels:
                                    label_parts.append(detection["class_name"])
                                if show_confidence:
                                    label_parts.append(f"{conf:.2f}")
                                label = " ".join(label_parts)
                                
                                # Draw text background
                                text_bbox = draw.textbbox((x1, y1), label)
                                draw.rectangle(text_bbox, fill="red")
                                draw.text((x1, y1), label, fill="white")
                            
                            detections.append(detection)
                    
                    with col2:
                        st.subheader("Detection Results")
                        st.image(img_with_boxes, use_container_width=True)
                    
                    # Show detection details
                    if detections:
                        st.subheader("Detection Details")
                        for i, det in enumerate(detections):
                            with st.expander(f"Detection {i+1}: {det['class_name']}"):
                                st.write(f"**Confidence:** {det['confidence']:.3f}")
                                st.write(f"**Bounding Box:** {det['bbox']}")
                                st.write(f"**Class ID:** {det['class_id']}")
                    else:
                        st.info(f"No logos detected with confidence >= {confidence_threshold}")
                        
                    # Show raw JSON response
                    with st.expander("Raw API Response"):
                        st.json(results)
                        
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Make sure the backend is running on http://localhost:8000")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Instructions
with st.expander("How to use"):
    st.markdown("""
    1. Make sure the FastAPI backend is running (`python app.py`)
    2. Upload an image using the file uploader
    3. Adjust the confidence threshold if needed
    4. Click "Detect Logos" to run detection
    5. View the results with bounding boxes
    """)