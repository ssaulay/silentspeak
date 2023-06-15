import streamlit as st
import requests
import os
import base64
from params import data_path
from moviepy.editor import VideoFileClip

# API endpoint URL
API_URL = "http://0.0.0.0:8080"
# API_URL = "http://localhost:8000"


# Paths
folder_path = os.path.join('silentspeak', 'tempDir')
chat_path = os.path.join('silentspeak', 'interface', 'img', 'wired-gradient-203-chat-message.gif')
arrow_path = os.path.join('silentspeak', 'interface', 'img', 'wired-gradient-33-arrow-down.gif')
logo_path = os.path.join('silentspeak', 'interface', 'img', 'logo.png')
face_path = os.path.join('silentspeak', 'interface', 'img', 'wired-gradient-1376-face-id.gif')

# Set page title and favicon
st.set_page_config(page_title="Silent Speak App", page_icon=":lips:")

# Custom CSS styles
st.markdown(
    """
    <style>
    .stApp {
        background: rgb(255,248,224);
        background: linear-gradient(122deg, rgba(255,248,224,1) 0%, rgba(255,255,255,1) 97%);
    }
    .title {
        color: #2d2d2d;
        text-align: center
    }
    .custom-label {
        color: #2d2d2d;
        text-align: center
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        height: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    body {
        color: #2d2d2d;
        text-align: center
    }
    .custom-gif {
        width: 100px;
        height: auto;
    }
    .custom-gif2 {
        width: 300px;
        height: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App logo
logo_image = logo_path
st.image(logo_image)

# Set separator
st.markdown("<h3 class='title'>•</h3>", unsafe_allow_html=True)

# Set app title
st.markdown("<h1 class='title'>Hear with your eyes through Silent Speak</h1>", unsafe_allow_html=True)

# Set separator
st.markdown("<h3 class='title'>•</h3>", unsafe_allow_html=True)

# Add a gif
file_ = open(chat_path, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="conv gif" class="custom-gif">',
    unsafe_allow_html=True)

# File uploader
st.markdown("<p class='custom-label'>Upload a video file to start prediction</p>", unsafe_allow_html=True)
video_file = st.file_uploader('', type=["mp4", "avi", "mpg"], key="video_file")


def save_uploadedfile(uploadedfile):
    with open(os.path.join(folder_path, uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())

    return os.path.join(folder_path, uploadedfile.name)


# if video_file is not None:
#     path_mpg = save_uploadedfile(video_file)

#     clip = VideoFileClip(path_mpg)
#     clip = clip.write_videofile(path_mpg[:-4]+'.mp4')

#     path_mp4 = os.path.join(path_mpg[:-4]+'.mp4')

#     # Display the video
# video_clip =
st.video(video_file)

file_ = open(arrow_path, "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="conv gif" class="custom-gif">',
    unsafe_allow_html=True)


# Make API request
if st.button("Predict") and video_file is not None:
    file_ = open(face_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    scan = st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="conv gif" class="custom-gif2">',
        unsafe_allow_html=True)

    url = f'{API_URL}/predict'
    files = {'file':video_file.getbuffer()}

    response = requests.post(url=url, files=files)

    if response.status_code == 200:
        result = response.json()
        scan.success(f"Prediction: {result['prediction']}")
        # os.remove(path_mpg)
        # os.remove(path_mp4)

    else:
        st.error("Error occurred during prediction")
