import os
import tensorflow as tf
import numpy as np
import pandas as pd # NEW LINE

from silentspeak.loading import *
from silentspeak.params import data_path, data_size, n_frames, n_frames_min, transcript_padding, filtered
from silentspeak.model import load_and_compile_model, checkpoint_callback, schedule_callback, instantiate_model, load_model_weigths, predict_video, models_path, predict_test, save_model, load_model, predict, ProduceExample



def mappable_function(path:str) -> List[str]:
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


# def load_data(
#     batch_size = 2,
#     padded_frames_shape = [n_frames,None,None,None],
#     padded_transcripts_shape = [transcript_padding]
# ):
#     """Create a """

#     if data_size in ["data", "sample_data"]:
#         video_format = "avi" # Videos in French
#     else:
#         video_format = "mpg" # Videos in English

#     # Only keep videos whose number of frames is between n_frames_min and n_frames
#     all_videos = os.listdir(os.path.join(data_path, data_size, "videos"))
#     all_videos = [vid for vid in all_videos if vid[-4:] == f".{video_format}"]
#     filtered_videos = [vid for vid in all_videos if (df[df["video"] == vid].iloc[0]["n_frames"] >= n_frames_min) & (df[df["video"] == vid].iloc[0]["n_frames"] <= n_frames)]

#     data = tf.data.Dataset.list_files(
#         [os.path.join(data_path, data_size, "videos", video) for video in filtered_videos]
#         )

#     data = data.map(mappable_function)
#     data = data.padded_batch(
#         batch_size,
#         padded_shapes = (padded_frames_shape, padded_transcripts_shape))
#     data = data.prefetch(tf.data.AUTOTUNE)

#     return data

#def data_train_test(
def load_data(
    batch_size = 2,
    padded_frames_shape = [n_frames,None,None,None],
    padded_transcripts_shape = [transcript_padding],
    train_split = 0.8,
):

    if data_size in ["data", "sample_data"]:
        video_format = "avi" # Videos in French
    else:
        video_format = "mpg" # Videos in English

    if filtered:

        csv_file = os.path.join(data_path, "n_frames.csv")
        df = pd.read_csv(csv_file)

        all_videos = os.listdir(os.path.join(data_path, data_size, "videos"))
        all_videos = [vid for vid in all_videos if vid[-4:] == f".{video_format}"]
        filtered_videos = [vid for vid in all_videos if (df[df["video"] == vid].iloc[0]["n_frames"] >= n_frames_min) & (df[df["video"] == vid].iloc[0]["n_frames"] <= n_frames)]

        data = tf.data.Dataset.list_files(
            [os.path.join(data_path, data_size, "videos", video) for video in filtered_videos]
            )

    else:
        data = tf.data.Dataset.list_files(
            os.path.join(data_path, data_size, "videos", f"*.{video_format}")
        )

    n_vids = len(list(data))

    data = data.map(mappable_function)
    data = data.padded_batch(
        batch_size,
        padded_shapes = (padded_frames_shape, padded_transcripts_shape))
    data = data.prefetch(tf.data.AUTOTUNE)

    if train_split < 1:
        train_size = int(train_split * (n_vids / batch_size))
        train = data.take(train_size)
        test = data.skip(train_size)

        return data, train, test

    else:
        return data


# def train_model_all(
#     model_num = 1,
#     epochs = 10,
#     batch_size = 2,
#     padded_frames_shape = [n_frames,None,None,None],
#     padded_transcripts_shape = [transcript_padding],
#     callbacks = [checkpoint_callback, schedule_callback]
# ):
#     """
#     Preprocess and train the model on all data, without train-test split.
#     To train a model with validation data, rather use the functions data_train_test and train_model
#     """

#     print("###### LOAD DATA ######")
#     data = load_data(batch_size, padded_frames_shape, padded_transcripts_shape)

#     print("###### LOAD AND COMPILE MODEL ######")
#     model = load_and_compile_model(model_num)

#     print("###### TRAIN MODEL ######")
#     model.fit(
#         data,
#         epochs = epochs,
#         callbacks = callbacks
#         )

#     return model



def train_model(
    train,
    test = None,
    model_num = 1,
    epochs = 10,
    callbacks = [checkpoint_callback, schedule_callback]
    ):

    """Instantiate, compile and train a model with train data and validation data"""

    print("###### Load and compile model ######")
    model = load_and_compile_model(model_num)

    print("###### Train model ######")
    model.fit(
        x = train,
        validation_data = test,
        epochs = epochs,
        callbacks = callbacks
        )

    return model


if __name__ == '__main__':


    # --- TEST TRAINING ---

    batch_size = 2
    data, train, test = load_data(batch_size = batch_size)
    example_callback = ProduceExample(test, batch_size = batch_size)
    callbacks = [checkpoint_callback, schedule_callback, example_callback]
    # callbacks = [checkpoint_callback, schedule_callback]

    model = train_model(
        train,
        test,
        model_num = 1,
        epochs = 2,
        callbacks = callbacks
    )


    # --- TEST PREDICTION ---

    # model_name = "model_def_EN_1-6.h5"
    # model = load_model(model_name)
    #model.summary()

    # video_name = "Welcome to SilentSpeak.MOV"
    # video_name = "lrae5a - lay_red_at_e_five_again.mpg"
    # video_name = "other_lipnet - place_red_in_a_zero_now.mpg"
    # video_name = "KING CHARLES.mp4" --> OK
    # video_name = "lrae5a - lay_red_at_e_five_again.mp4"
    # video_name = "other_lipnet - place_red_in_a_zero_now.mp4"

    # video = os.path.join(data_path, "videos_demo", video_name)
    # prediction = predict_video(model, video)

    pass
