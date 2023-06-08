import cv2
import tensorflow as tf
from typing import List
import os
import numpy as np
from silentspeak.params import vocab_type, vocab_phonemes, vocab_letters


if vocab_type == "p":
    vocab = vocab_phonemes
else:
    vocab = vocab_letters


char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame[350:500,230:530,:])
        frame = cv2.resize(frame.numpy().squeeze(),(150,75),interpolation=cv2.INTER_LANCZOS4)
        frame = np.expand_dims(frame, -1)
        frames.append(tf.convert_to_tensor(frame))
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std

    return frames


def get_transcript(path: str) -> List[str]:
    """Get the file path of a transcript and return a clean list with the truth transcript."""
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]

    transcript = [line[1] for line in lines if len(line) > 1]
    transcript = [char.replace("</s>", "") for char in transcript]
    transcript = [char for char in transcript if len(char) > 0]

    return char_to_num(transcript)


def load_data(video_path: str):
    """Load the frames and the phonems for a single video
    (i.e. a single sentence pronounced by a single locutor)

    inputs:
    >> video_path: the path of a video
    """

    try:
        frames = load_video(video_path)
    except:
        video_path = bytes.decode(video_path.numpy())
        frames = load_video(video_path)
    id_code = video_path[-11:][:7]

    transcript_path = os.path.join(video_path[:-11], "..", "transcripts", f"{vocab_type}_{id_code}.txt")
    transcript = get_transcript(transcript_path)

    return frames, transcript
