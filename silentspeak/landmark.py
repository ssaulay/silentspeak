import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from silentspeak.params import local_data_path
from os import walk


CSV = os.path.join(local_data_path, 'csv')
VIDEO = os.path.join(local_data_path, '..', 'drafts', 'data', 'sample_data', 'videos')
NPY = os.path.join(local_data_path, 'npy')

filenames = next(walk(VIDEO), (None, None, []))[2]

path_csv_normalize = os.path.join(CSV, 'normalize')
path_csv_non_normalize = os.path.join(CSV, 'non_normalize')

path_npy_non_normalize = os.path.join(NPY, 'non_normalize')
path_npy_normalize = os.path.join(NPY, 'normalize')

def create_landmarks_csv(filenames, path_csv_normalize, path_csv_non_normalize):
    name_csv = []
    for filename in filenames:

        path = os.path.join(VIDEO, f'{filename}')
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
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2XYZ)

                # Process the frame using MediaPiSpe
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

                # Show the video
                # cv2.imshow('Mouth Capture', annotated_frame)


                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


        # Create dataframe from array
        coordinates_by_frames_array = [np.array(coordinates_by_frames[i:i+5]) for i in range(0, len(coordinates_by_frames), 5)]
        df = pd.DataFrame(coordinates_by_frames_array)
        df.rename(columns={0:'frame', 1:'point', 2:'x', 3:'y', 4:'z'}, inplace=True)

        # Add sentence and speaker
        df['sentence'] = filename[:-8]
        df['speaker'] = filename[4:-4]

        # Create csv non-scale
        name_csv = os.path.join(path_csv_non_normalize, filename[:-4])+".csv"
        df.to_csv(name_csv, index=False)

        # Dataframe scaled
        df_scale = df
        for i in range(0, df_scale['frame'].nunique()):
            # Get reference coordinates for ID 6 (at each frame)
            ref_x = df_scale.loc[df_scale['point'] == 6, 'x'].values[i]
            ref_y = df_scale.loc[df_scale['point'] == 6, 'y'].values[i]
            ref_z = df_scale.loc[df_scale['point'] == 6, 'z'].values[i]

            # Compute difference between coordinates (at each frame)
            df_scale['x'][df_scale['frame']==i+1] = df_scale['x'][df_scale['frame']==i+1] - ref_x
            df_scale['y'][df_scale['frame']==i+1] = df_scale['y'][df_scale['frame']==i+1] - ref_y
            df_scale['z'][df_scale['frame']==i+1] = df_scale['z'][df_scale['frame']==i+1] - ref_z

            # Create csv normalize
            name_csv_normalize = os.path.join(path_csv_normalize, filename[:-4])+"_normalize"+".csv"
            df_scale.to_csv(name_csv_normalize, index=False)


        cap.release()
        cv2.destroyAllWindows()



def create_landmarks_npy(path_csv_non_normalize, path_npy_non_normalize, path_npy_normalize, path_csv_normalize):

    filenames = next(walk(path_csv_non_normalize), (None, None, []))[2]
    filenames_normalize =  next(walk(path_csv_normalize), (None, None, []))[2]

    csv_files = [filename for filename in filenames]
    csv_files_normalize = [filename_normalize for filename_normalize in filenames_normalize]

    # Create super csv (non_normalize)
    li = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(path_csv_non_normalize, file))
        li.append(df)
    df_all = pd.concat(li, axis=0, ignore_index=True)

    # Drop points 6 (non_normalize)
    index_point_six = list(df_all.loc[df_all['point'] == 6, 'x'].index)
    df_all = df_all.drop(index=index_point_six)

    # Create tensor and npy files (non_normalize)
    for speaker in df_all.speaker.unique():
        for sentence in df_all.sentence.unique():
            d = df_all.loc[(df_all.sentence==sentence)&(df_all.speaker==speaker), :].drop(columns=['sentence', 'speaker'])
            if len(d)==0:
                continue
            video = []

            for f in d.frame.unique():
                video.append(d.loc[d.frame==f, ['x', 'y', 'z']].to_numpy())

            file_npy = os.path.join(path_npy_non_normalize, '0'*(3-len(str(sentence)))+str(sentence)+'_'+speaker)
            video = np.array(video)
            np.save(file_npy, video)

    # Create super csv (normalize)
    li = []
    for file in csv_files_normalize:
        df = pd.read_csv(os.path.join(path_csv_normalize, file))
        li.append(df)
    df_all_normalize = pd.concat(li, axis=0, ignore_index=True)

    # Create tensor and npy files (normalize)
    for speaker in df_all_normalize.speaker.unique():
        for sentence in df_all_normalize.sentence.unique():
            d = df_all_normalize.loc[(df_all_normalize.sentence==sentence)&(df_all_normalize.speaker==speaker), :].drop(columns=['sentence', 'speaker'])
            if len(d)==0:
                continue
            video = []

            for f in d.frame.unique():
                video.append(d.loc[d.frame==f, ['x', 'y', 'z']].to_numpy())

            file_npy = os.path.join(path_npy_normalize, '0'*(3-len(str(sentence)))+str(sentence)+'_'+speaker+'_normalize')
            video = np.array(video)
            np.save(file_npy, video)


def create_landmarks_mp4():

    cap = cv2.VideoCapture(0)
    NUM_FACE = 2
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    faceMesh = mp_face_mesh.FaceMesh(max_num_faces=NUM_FACE)
    drawSpec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


    with mp_face_mesh.FaceMesh() as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            # Convert the BGR frame to RGB and process it
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Process the frame using MediaPiSpe
            results = face_mesh.process(rgb_frame)

            # Draw the face landmarks on the frame
            annotated_frame = frame.copy()

            # if results.multi_face_landmarks:
            #     for faceLms in results.multi_face_landmarks:
            #         mp_drawing.draw_landmarks(
            #             annotated_frame,
            #             results.multi_face_landmarks[0],
            #             mp_face_mesh.FACEMESH_CONTOURS,
            #             drawSpec,
            #             drawSpec
            #             )

            # # Extract the mouth region
            # mouth_coordinates = {}
            # mouth_landmarks = []
            # mouth_ids = [
            # 0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
            # 17, 84, 181, 91, 146, 61, 185, 40, 39, 37, 0,
            # 13, 312, 311, 310, 415, 308, 324, 318, 402, 317,
            # 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13
            # ]
            # reference_id = 6

            # if results.multi_face_landmarks:
            #         landmarks = results.multi_face_landmarks[0]


            #         for id, landmark in enumerate(landmarks.landmark):

            #             # Extract only the mouth landmarks (IDs in mouth_ids)
            #             if id in mouth_ids:
            #                 # Get the frame, id and coordinates of points of mouth
            #                 pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            #                 ih, iw, ip = frame.shape
            #                 x, y, z= int(landmark.x * iw), int(landmark.y * ih), int(landmark.y * ip)
            #                 mouth_landmarks.append((x, y))
            #                 mouth_coordinates[id] = landmarks
            #                 cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)


            # # Draw the mouth region
            # if mouth_landmarks:
            #     hull = cv2.convexHull(np.array(mouth_landmarks))
            #     cv2.drawContours(annotated_frame, [hull], -1, (0, 255, 0), 2)

            # Show the video
            cv2.imshow('Mouth Capture', annotated_frame)


            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv2.destroyAllWindows()

create_landmarks_mp4()
