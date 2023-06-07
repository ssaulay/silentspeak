import cv2
import tensorflow as tf
from typing import List
import os


def load_video(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[350:500,230:530,:])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def get_transcript(path: str) -> List[str]:
    """Get the file path of a transcript and return a clean list with the truth transcript."""
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]

    transcript = [line[1] for line in lines if len(line) > 1]
    transcript = [char.replace("</s>", "") for char in transcript]
    transcript = [char for char in transcript if len(char) > 0]

    return transcript


def load_data(path: str, locutor: int, file: int, format: str):
    """Load the frames and the phonems for a single video
    (i.e. a single sentence pronounced by a single locutor)

    inputs:
    >> path: the root path of the full database
    >> locutor: an integer between 1 and 8
    >> file: an integer between 1 and 238
    >> format: 'p' for phonemes, 'l' for letters
    """

    if file < 10:
        file_str = f"00{file}"
    elif file < 100:
        file_str = f"0{file}"
    else:
        file_str = str(file)

    if locutor < 10:
        locutor_str = f"0{locutor}"
    else:
        locutor_str = str(locutor)

    video_path = os.path.join(path, "videos", f"{file_str}_L{locutor_str}.avi")
    frames = load_video(video_path)

    transcript_path = os.path.join(path, "transcripts", f"{format}_{file_str}.txt")
    transcript = get_transcript(transcript_path)

    return frames, transcript
