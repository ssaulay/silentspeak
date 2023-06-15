from silentspeak.model import predict_video, load_model, load_and_compile_model
from silentspeak.loading import num_to_char
import tensorflow as tf
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model = load_model('model_def_EN_1-6.h5')

# model = load_and_compile_model()

@app.get("/")
def root():
    return {'greeting': 'Hello'}

@app.post("/predict")
async def make_prediction(file:UploadFile):
    content = await file.read()
    timestamp = str(time.time()).replace(".", "")

    with open(f"{timestamp}.mpg", "wb") as binary_file:
        binary_file.write(content)

    prediction = predict_video(model=model, video_path=f'{timestamp}.mpg')

    return {'prediction': prediction}
