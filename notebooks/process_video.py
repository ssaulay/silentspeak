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
        frames.append(frame)
    cap.release()
    return frames


def get_phonems(path: str) -> List[str]:
    """Get the file path of a phonem transcript and return a clean list with the truth phonems."""
    with open(path, "r") as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]

    phonems = [line[1] for line in lines if len(line) > 1]
    phonems = [phonem.replace("</s>", "") for phonem in phonems]
    phonems = [phonem for phonem in phonems if len(phonem) > 0]

    return phonems


def load_data(path: str, session: int, locutor: int, file: int):
    """Load the frames and the phonems for a single video
    (i.e. a single sentence pronounced by a single locutor)

    inputs:
    >> path: the root path of the full database
    >> session: 'I' or 'II'
    >> locutor: an integer between 1 and 8
    >> file: an integer between 1 and 238
    """

    video_path = os.path.join(
        path, f"Session {session}", f"Locuteur_{session}_{locutor}",
        "videos", f"{file}_front.avi")
    frames = load_video(video_path)

    locutor_path = os.path.join(
        path, f"Session {session}", f"Locuteur_{session}_{locutor}")
    session_dirs = os.listdir(locutor_path)
    lab_dir = [dir for dir in session_dirs if "phone" in dir.lower()][0]

    phonems_path = os.path.join(locutor_path, lab_dir, f"{file}.txt")
    phonems = get_phonems(phonems_path)

    return frames, phonems
