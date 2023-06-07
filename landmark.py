import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

path = "/home/clement/code/ssaulay/silentspeak/raw_data/BL-Database/Session I/Locuteur_I_1/videos/1_front.avi"
cap = cv2.VideoCapture(path)
NUM_FACE = 2
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
faceMesh = mp_face_mesh.FaceMesh(max_num_faces=NUM_FACE)
drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

coordinates_by_frames = []
with mp_face_mesh.FaceMesh() as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        # Convert the BGR frame to RGB and process it
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        # Extract the mouth region
        mouth_coordinates = {}
        mouth_landmarks = []
        mouth_ids = [
        0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
        17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 0,
        13, 312, 311, 310, 415, 308, 324, 318, 402, 317,
        14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13
        ]
        reference_id = 6

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            for id, landmark in enumerate(landmarks.landmark):
                # Extract only the mouth landmarks (IDs in mouth_ids)
                if id in mouth_ids:
                    # Get the frame, id and coordinates of points of mouth
                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    coordinates_by_frames.append(pos_frame)
                    coordinates_by_frames.append(id)
                    coordinates_by_frames.append(landmark.x)
                    coordinates_by_frames.append(landmark.y)
                    coordinates_by_frames.append(landmark.z)
                    ih, iw, ip = frame.shape
                    x, y, z= int(landmark.x * iw), int(landmark.y * ih), int(landmark.y * ip)
                    mouth_landmarks.append((x, y))
                    mouth_coordinates[id] = landmarks
                    cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
                if id == reference_id:
                    # Get the frame, id and coordinates of reference point
                    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    coordinates_by_frames.append(pos_frame)
                    coordinates_by_frames.append(id)
                    coordinates_by_frames.append(landmark.x)
                    coordinates_by_frames.append(landmark.y)
                    coordinates_by_frames.append(landmark.z)

        # Draw the mouth region
        if mouth_landmarks:
            hull = cv2.convexHull(np.array(mouth_landmarks))
            cv2.drawContours(annotated_frame, [hull], -1, (0, 255, 0), 2)

        # Identify mouth using haar-based classifiers
        # mouth_cascade = cv2.CascadeClassifier('/home/clement/code/SimpleCV/SimpleCV/Features/HaarCascades/face.xml')
        # mouth = mouth_cascade.detectMultiScale(rgb_frame, 1.5, 11)
        # for(mx, my, mw, mh) in mouth:
        #     cv2.rectangle(annotated_frame, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)

        # Show the annotated frame with the mouth region
        cv2.imshow('Mouth Capture', annotated_frame)


        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
