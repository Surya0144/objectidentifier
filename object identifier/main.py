import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Function to select an image file using Streamlit file uploader
def select_image_file():
    st.sidebar.title('Select an Image')
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"]
    )
    if uploaded_file is not None:
        # Convert the file to an OpenCV image.
        image = Image.open(uploaded_file)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image
    return None

# Load YOLO model
model = YOLO("yolov8l.pt")

# Main Streamlit app code
def main():
    st.title('YOLO Object Detection with Streamlit')

    # Select an image
    img = select_image_file()

    if img is not None:
        # Resize image to 1280x720
        img_resized = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)

        # Perform object detection with YOLO
        results = model(img_resized)

        # Get the annotated image from results
        annotated_img = results[0].plot()

        # Convert BGR image to RGB for displaying in Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # Display original and annotated images
        st.image([img_rgb, annotated_img_rgb], caption=['Original Image', 'Annotated Image'], width=640)

# Run the Streamlit app
if __name__ == '__main__':
    main()
