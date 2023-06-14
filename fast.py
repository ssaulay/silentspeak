from silentspeak.model import predict, load_model, load_and_compile_model
from silentspeak.loading import num_to_char
import tensorflow as tf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('model_130623EN.h5')

# model = load_and_compile_model()

@app.get("/")
def root():
    return {'greeting': 'Hello'}

@app.get("/predict")
def make_prediction(path
                    ):

    prediction = predict(model=model, path=path)

    decoded_string = tf.strings.reduce_join(
                [num_to_char(tf.argmax(x)) for x in prediction[0]]
                )

    result = bytes.decode(decoded_string.numpy())
    # result = 'Une quebécoise pleurnicheuse brandie euclide lors des réunions'

    return {'prediction': result}
