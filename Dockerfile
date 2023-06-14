FROM --platform=linux/amd64 python:3.10.6-bullseye


WORKDIR /prod

COPY requirements_prod.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY silentspeak silentspeak

RUN mkdir model

COPY fast.py .

COPY .keys .keys

CMD uvicorn fast:app --host 0.0.0.0 --port $PORT
