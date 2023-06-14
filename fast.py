from silentspeak.model import predict, load_model, load_and_compile_model
from silentspeak.loading import num_to_char
import tensorflow as tf
import time

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

model = load_model('model_140623EN22.h5')

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

    prediction = predict(model=model, path=f'{timestamp}.mpg')

    result = prediction

    return {'prediction': result}
