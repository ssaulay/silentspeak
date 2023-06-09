import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from os import walk
import math

def bounding_box(path:str):
    cap = cv2.VideoCapture(path)
    NUM_FACE = 2
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    faceMesh = mp_face_mesh.FaceMesh(max_num_faces=NUM_FACE)
    drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Get 4 points to define mouth coordinates
    x_291 = []
    y_291 = []
    x_61 = []
    y_61 = []

    with mp_face_mesh.FaceMesh() as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR frame to RGB and process it
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2XYZ)

            # Process the frame using MediaPipe
            results = face_mesh.process(rgb_frame)

            # Draw the face landmarks on the frame
            annotated_frame = frame.copy()

            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        results.multi_face_landmarks[0],
                        mp_face_mesh.FACEMESH_CONTOURS,
                        drawSpec,
                        drawSpec
                        )



            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]

                for id, landmark in enumerate(landmarks.landmark):

                    # Extract only the mouth landmarks (IDs in mouth_ids)
                    if id == 291:
                        x_291.append(landmark.x)
                        y_291.append(landmark.y)
                    if id == 61:
                        x_61.append(landmark.x)
                        y_61.append(landmark.y)


            # Get video height, width and channels
            height, width, channels = frame.shape


        cap.release()
        cv2.destroyAllWindows()

    x_px_min = min(math.floor(min(x_61) * width), width - 1) - round(0.02 * min(math.floor(max(x_61) * width), width - 1))
    y_px_min = min(math.floor(max(y_61) * height), height - 1)
    x_px_max = min(math.floor(max(x_291) * width), width - 1) + round(0.02 * min(math.floor(max(x_291)* width), width - 1))
    y_px_max = min(math.floor(max(y_291) * height), height - 1)

    Lb = x_px_max - x_px_min
    Hb = round(54/80 * Lb)

    y_px_min = round(y_px_min + Hb/2)
    y_px_max = round(y_px_max - Hb/2)
    return x_px_min, y_px_min, x_px_max, y_px_max



print(bounding_box('/home/clement/code/ssaulay/silentspeak/drafts/data/sample_data/videos/001_L14.avi'))
