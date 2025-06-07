import cv2
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Image Cartoonizer", layout="wide")

st.title("🎨 Cartoonize and Stylize Your Image")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("📸 Original Image")
    st.image(img, use_column_width=True)

    # Parameters
    line_size = 7
    blur_value = 7
    k = 7

    # Convert to grayscale and apply blur
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray_img, blur_value)

    # Gray edge detection
    edges_gray = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, line_size, blur_value
    )

    # RGB channel edge detection
    edges_colored = []
    colors = ['Red', 'Green', 'Blue']
    channels = cv2.split(img)

    for i in range(3):
        blur_channel = cv2.medianBlur(channels[i], blur_value)
        edge = cv2.adaptiveThreshold(
            blur_channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, line_size, blur_value
        )
        edges_colored.append(edge)

    combined_edges = cv2.merge(edges_colored)

    # Color reduction using K-Means
    data = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
    img_reduced = kmeans.cluster_centers_[kmeans.labels_]
    img_reduced = img_reduced.reshape(img.shape).astype(np.uint8)

    # Bilateral filter for smoothness
    blurred = cv2.bilateralFilter(img_reduced, d=7, sigmaColor=200, sigmaSpace=200)

    # Cartoon effect using colored edges
    cartoon_colored = cv2.bitwise_and(blurred, blurred, mask=cv2.cvtColor(combined_edges, cv2.COLOR_RGB2GRAY))

    # Stylized effect using OpenCV
    stylized = cv2.stylization(img, sigma_s=60, sigma_r=0.6)

    # Display results
    st.subheader("🧠 Gray Edge Detection")
    st.image(edges_gray, clamp=True, channels="GRAY", use_column_width=True)

    st.subheader("🎨 RGB Channel Edges")
    col1, col2, col3 = st.columns(3)
    for i, col in zip(range(3), [col1, col2, col3]):
        col.image(edges_colored[i], clamp=True, channels="GRAY", caption=colors[i])

    st.subheader("🌈 Combined RGB Edges")
    st.image(combined_edges, use_column_width=True)

    st.subheader("🖼️ Cartoon Effect with Colored Edges")
    st.image(cartoon_colored, use_column_width=True)

    st.subheader("✨ Stylized Image (OpenCV Artistic Effect)")
    st.image(stylized, use_column_width=True)

    # Optionally allow download
    cartoon_pil = Image.fromarray(cartoon_colored)
    stylized_pil = Image.fromarray(stylized)
    
    st.download_button("📥 Download Cartoon Image", data=cartoon_pil.tobytes(), file_name="cartoon.png", mime="image/png")
    st.download_button("📥 Download Stylized Image", data=stylized_pil.tobytes(), file_name="stylized.png", mime="image/png")

else:
    st.info("Please upload an image to begin.")
