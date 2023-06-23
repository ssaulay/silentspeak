"""Functions to process the videos and the transcripts and feed the models with clean data"""

import cv2
import tensorflow as tf
from typing import List, Tuple
import os
import numpy as np
from silentspeak.params import data_size, vocab_type, vocab, frame_h, frame_w, accents_dict, n_frames
from silentspeak.bounding_box import bounding_box

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def save_to_npy(video_path: str, npy_dir: str):
    """Process a video and save it as a .npy file

    Args:
        video_path (str): The path to the video that will be saved as a .npy file
        npy_dir (str) : The path to the directory where the .npy file will be saved
    """

    video_tensor = process_video(video_path)
    video_array = video_tensor.numpy()

    npy_filename = f"{os.path.basename(video_path)[:-4]}.npy"

    np.save(
        file = os.path.join(npy_dir, npy_filename),
        arr = video_array
    )


def load_video_npy(path: str):
    """
    Load a .npy file representing of a video cropped to mouth-size and convert it into a tensor.

    Args:
        path (str): The path to the video saved as a .npy file

    Returns:
        A tensor representing the video cropped to mouth-size
    """

    video_np = np.load(path)
    video_tensor = tf.cast(video_np, tf.float32)
    return video_tensor


def process_video(path: str):
    """Process the video into a tensor representing the video cropped to mouth-size

    Args:
        path (str): The path to the video

    Returns:
        A tensor representing the video frames cropped to mouth-size
    """

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

    If the file exists as a .npy file, the function will load this file into a tensor.
    If the file does not exists as a .npy file, the function will process the video to a tensor.

    Args:
        path (str): The path to the video

    Returns:
        A tensor representing the video frames cropped to mouth-size
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


def get_transcript(path: str) -> List[int]:
    """Get the file path of a transcript and return a vectorized truth transcript.

    This function is used when transcripts are in French.

    Args:
        path (str): the path to the transcript

    Returns:
        A tensor with the vectorized transcript. For example:
        tf.Tensor([18  6  5  8 13], shape=(5,), dtype=int64)

    """
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


def load_alignments(path:str) -> List[int]:
    """Get the file path of a transcript and return a vectorized truth transcript.

    This function is used when transcripts are in English.

    Args:
        path (str): the path to the transcript

    Returns:
        A tensor with the vectorized transcript. For example:
        tf.Tensor([18  6  5  8 13], shape=(5,), dtype=int64)

    """
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(video_path: str) -> Tuple[List[float], List[int]]:
    """Load the frames and the transcript of a video

    Args:
        video_path (str): the path of a video

    Returns:
        A tuple of two elements:
        1- A tensor representing the video frames cropped to mouth-size
        2- A tensor with the vectorized transcript
    """

    try:
        frames = load_video(video_path)
    except:
        video_path = bytes.decode(video_path.numpy())
        frames = load_video(video_path)

    # If videos are in French
    if data_size in ["data", "sample_data"]:
        id_code = video_path[-11:][:7]
        transcript_path = os.path.join(video_path[:-11], "..", "transcripts", f"{vocab_type}_{id_code}.txt")
        transcript = get_transcript(transcript_path)

    # If videos are in English
    else:
        id_code = video_path[-10:][:6]
        transcript_path = os.path.join(video_path[:-10], "..", "transcripts", f"{id_code}.align")
        transcript = load_alignments(transcript_path)

    return frames, transcript


if __name__ == '__main__':
    pass
