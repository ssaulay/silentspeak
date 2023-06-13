from silentspeak.model import predict, load_model

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

model = '/home/clement/code/ssaulay/silentspeak/models/model_20230612-203850.h5'
app.state.model = load_model(model_filename=model)

@app.get("/")
def root():
    return {'greeting': 'Hello'}

@app.get("/predict")
def make_prediction(path: str):
    result = predict(app.state.model, path)
    return {'prediction': result}
