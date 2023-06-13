import streamlit as st
import requests


# API endpoint URL
API_URL = "http://localhost:8000"

st.title("FastAPI Streamlit App")

# File uploader
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", '.mpg'])

# Make API request
if st.button("Predict") and video_file is not None:
    files = {"file": video_file.getvalue()}
    response = requests.post(f"{API_URL}/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error("Error occurred during prediction")
