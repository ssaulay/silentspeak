import os
import tensorflow as tf
import numpy as np
import pandas as pd # NEW LINE

from silentspeak.loading import *
from silentspeak.params import data_path, data_size, n_frames, n_frames_min, transcript_padding
from silentspeak.model import load_and_compile_model, checkpoint_callback, schedule_callback, instantiate_model, load_model_weigths, predict_video, models_path, predict_test, save_model, load_model, predict, ProduceExample


# Load csv file with number of frames per video
# (videos in English and in French in the same df)
csv_file = os.path.join(data_path, "n_frames.csv")
df = pd.read_csv(csv_file)


def mappable_function(path:str) ->List[str]:
    """
    inputs:
    >> path:
    """
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


def train_model_all(
    model_num = 1,
    epochs = 10,
    batch_size = 2,
    padded_frames_shape = [n_frames,None,None,None],
    padded_transcripts_shape = [transcript_padding],
    train_size = 10,
    callbacks = [checkpoint_callback, schedule_callback]
):
    """
    Preprocess and train the model on all data.
    --> Use data_train_test and train_model to train a model with validation data
    """

    # Load data

    if data_size in ["data", "sample_data"]:
        video_format = "avi" # Videos in French
    else:
        video_format = "mpg" # Videos in English

    # OLD WAY --> LOAD ALL VIDEOS IN A FOLDER
    # data = tf.data.Dataset.list_files(
    #     os.path.join(data_path, data_size, "videos", f"*.{video_format}")
    #     )

    # NEW WAY --> ONLY LOAD VIDEOS WITH LESS THAN N_FRAMES FRAMES
    all_videos = os.listdir(os.path.join(data_path, data_size, "videos"))
    all_videos = [vid for vid in all_videos if vid[-4:] == f".{video_format}"]
    filtered_videos = [vid for vid in all_videos if (df[df["video"] == vid].iloc[0]["n_frames"] >= n_frames_min) & (df[df["video"] == vid].iloc[0]["n_frames"] <= n_frames)]

    data = tf.data.Dataset.list_files(
        [os.path.join(data_path, data_size, "videos", video) for video in filtered_videos]
        )

    data = data.map(mappable_function)
    data = data.padded_batch(
        batch_size,
        padded_shapes = (padded_frames_shape, padded_transcripts_shape))
    data = data.prefetch(tf.data.AUTOTUNE)

    print("###### Load and compile model ######")

    model = load_and_compile_model(model_num)

    print("###### Train model ######")

    model.fit(
        data,
        epochs = epochs,
        callbacks = callbacks
        )

    return model


def data_train_test(
    batch_size = 2,
    padded_frames_shape = [n_frames,None,None,None],
    padded_transcripts_shape = [transcript_padding],
    train_split = 0.8,
):


    if data_size in ["data", "sample_data"]:
        video_format = "avi" # Videos in French
    else:
        video_format = "mpg" # Videos in English

    all_videos = os.listdir(os.path.join(data_path, data_size, "videos"))
    all_videos = [vid for vid in all_videos if vid[-4:] == f".{video_format}"]
    filtered_videos = [vid for vid in all_videos if (df[df["video"] == vid].iloc[0]["n_frames"] >= n_frames_min) & (df[df["video"] == vid].iloc[0]["n_frames"] <= n_frames)]

    data = tf.data.Dataset.list_files(
        [os.path.join(data_path, data_size, "videos", video) for video in filtered_videos]
        )

    n_vids = len(list(data))

    data = data.map(mappable_function)
    data = data.padded_batch(
        batch_size,
        padded_shapes = (padded_frames_shape, padded_transcripts_shape))
    data = data.prefetch(tf.data.AUTOTUNE)

    train_size = int(train_split * (n_vids / batch_size))
    train = data.take(train_size)
    test = data.skip(train_size)

    return data, train, test


def train_model(
    train,
    test,
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

    # TEST PREDICTION
    model_name = "model_140623EN_AM1_6.h5"
    model = load_model(model_name)

    #model = instantiate_model(model_num = 1)
    #checkpoint_demo_EN = "/Users/ArthurPech/code/ssaulay/silentspeak/models/model_demo_EN_2/checkpoint"
    #model = load_model_weigths(model, checkpoint_demo_EN)

    #video_name = "Welcome to SilentSpeak.MOV" # --> ERROR
    #video_name = "lrae5a - lay_red_at_e_five_again.mpg" # --> OK
    video_name = "other_lipnet - place_red_in_a_zero_now.mpg" # --> OK

    video = os.path.join(data_path, "videos_demo", video_name)
    prediction = predict_video(model, video)


    #print(data_path)
    # download_data
    # preprocess
    # model = load_and_compile_model(model_num = 1)
    # model = train_model_all(epochs = 2)

    # batch_size = 2
    # data, train, test = data_train_test(batch_size = batch_size)
    # example_callback = ProduceExample(test, batch_size = batch_size)
    # callbacks = [checkpoint_callback, schedule_callback, example_callback]
    # # callbacks = [checkpoint_callback, schedule_callback]

    # model = train_model(
    #     train,
    #     test,
    #     model_num = 1,
    #     epochs = 2,
    #     callbacks = callbacks
    # )

    # save_model(model)

    pass
