import cv2
import tensorflow as tf
from typing import List
import os
import numpy as np
from silentspeak.params import data_size, vocab_type, vocab, frame_h, frame_w, accents_dict, n_frames
from silentspeak.bounding_box import bounding_box

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video_npy(path: str):
    """
    Load a .npy file representing of a video cropped to mouth-size and convert it into a tensor.
    """

    video_np = np.load(path)
    # video_tensor = tf.constant(video_np)
    video_tensor = tf.cast(video_np, tf.float32)
    return video_tensor


def process_video(path: str):
    """Process the video into a tensor representing the video cropped to mouth-size"""

    y_px_min, y_px_max, x_px_min, x_px_max = bounding_box(path=path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame[y_px_min:y_px_max,x_px_min:x_px_max,:])
        frame = cv2.resize(frame.numpy().squeeze(),(frame_w,frame_h),interpolation=cv2.INTER_LANCZOS4)
        frame = np.expand_dims(frame, -1)
        frames.append(tf.convert_to_tensor(frame))
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    frames = tf.cast((frames - mean), tf.float32) / std

    return frames




def load_video(path:str) -> List[float]:
    """
    Load the tensor representation of a video cropped to mouth-size.
    >> If the file exists as a .npy file --> load this file into a tensor
    >> If the file does not exists as a .npy file --> process the video to a tensor
    """

    # CASE IF VIDEOS ARE IN FRENCH
    if data_size in ["data", "sample_data"]:
        id_code = path[-11:][:7]
        npy_path = os.path.join(path[:-11], "..", "videos-npy")

    # CASE IF VIDEOS ARE IN ENGLISH
    else:
        id_code = path[-10:][:6]
        npy_path = os.path.join(path[:-10], "..", "videos-npy")

    # if npy file exists --> load this file
    npy_file = os.path.join(npy_path, f"{id_code}.npy")
    if os.path.isfile(npy_file):
        frames = load_video_npy(npy_file)

    # if npy file does not exist --> process the video
    else:
        frames = process_video(path)

    return frames


# load_video = process_video


def get_transcript(path: str) -> List[str]:
    """Get the file path of a transcript and return a clean list with the truth transcript."""
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]

    transcript = [line[1] for line in lines if len(line) > 1]
    transcript = [char.replace("</s>", "") for char in transcript]
    transcript = [char for char in transcript if len(char) > 0]

    # Remove accents if vocab_type is letters

    if vocab_type == "l":
        new_transcript = []
        for char in transcript:
            if char in accents_dict.keys():
                new_transcript.append(accents_dict[char])
            else:
                new_transcript.append(char)
        transcript = new_transcript

    return char_to_num(transcript)


def load_alignments(path:str) -> List[str]:
    """Load the English transcripts"""
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


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

    # CASE IF VIDEOS ARE IN FRENCH
    if data_size in ["data", "sample_data"]:

        id_code = video_path[-11:][:7]
        transcript_path = os.path.join(video_path[:-11], "..", "transcripts", f"{vocab_type}_{id_code}.txt")
        transcript = get_transcript(transcript_path)

    # CASE IF VIDEOS ARE IN ENGLISH
    else:

        id_code = video_path[-10:][:6]
        transcript_path = os.path.join(video_path[:-10], "..", "transcripts", f"{id_code}.align")
        transcript = load_alignments(transcript_path)

    return frames, transcript


if __name__ == '__main__':
    pass
