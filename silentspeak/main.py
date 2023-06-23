"""Main program for loading data, training a model and evaluating it."""

import os
import tensorflow as tf
import pandas as pd

from silentspeak.loading import *
from silentspeak.params import data_path, data_size, n_frames, n_frames_min, transcript_padding, filtered
from silentspeak.model import load_and_compile_model, checkpoint_callback, schedule_callback, instantiate_model, load_model_weigths, predict_video, models_path, predict_test, save_model, load_model, predict, ProduceExample
from silentspeak.metrics import CER, WER


def mappable_function(path:str) -> List[List]:
    """This function loads the videos and the transcripts in the dataset.

    This function is used in the data_train_test function.

    Args:
        path (str): The path to a video file

    Returns:
        A list of two elements:
        1- A tensor representing the video frames cropped to mouth-size
        2- A tensor with the vectorized transcript
    """
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


def data_train_test(
    batch_size: int = 2,
    padded_frames_shape: List[int] = [n_frames,None,None,None],
    padded_transcripts_shape: List[int] = [transcript_padding],
    train_split: float = 0.8,
):
    """
    This functions generates the dataset used to train the model.

    It splits the data between train and test if train_split is set between 0 and 1.

    When filtered is set to True in params.py, the function will refer
    to the csv file data_path/n_frames.csv, in which the number of frames
    is indicated video by video. In this case, it will only retain videos
    whose number of frames is between n_frames_min and n_frames.

    Args:
        batch_size (str): The number of video/transcript couple in each batch
        padded_frames_shape (list): The padding dimension of the videos
        padded_transcripts_shape (list): The padding dimension of the transcripts
        train_split (float):

    Returns:
        A prefetched tensorflow Dataset or a tuple of prefeteched tensorflow Datasets.

        If train_split is set to a value between 0 and 1, it will returns
        three objects: the full dataset, the train dataset and the test dataset.

        If train_split is not set between 0 and 1, the function will not perform
        a train-test split, and it will only return one object: the full dataset.
    """

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

    if train_split > 0 and train_split < 1:
        train_size = int(train_split * (n_vids / batch_size))
        train = data.take(train_size)
        test = data.skip(train_size)

        return data, train, test

    else:
        return data


def train_model(
    train,
    test = None,
    model_num: int = 1,
    epochs: int = 10,
    callbacks = [checkpoint_callback, schedule_callback]
    ) -> tf.keras.Model:

    """This functions instantiates, compiles and trains a model.

    Args:
        train: The train dataset (A prefetched tensorflow Dataset)
        test: The validation dataset (A prefetched tensorflow Dataset)
        model_num (int): The model number as defined in model.py (1 or 2)
        epochs (int): The number of training epochs
        callbacks: The callbacks used during training

    Returns:
        A trained model

    """

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


def evaluate_model(
    model,
    test
    ) -> dict[str, float]:
    """This function returns a dictionary containing two model evaluation metrics:
    the Character Error Rate (CER) and the Word Error Rate (WER).

    Args:
        model: The model to evaluate
        test: The dataset used for evaluation (A prefetched tensorflow Dataset)

    Returns:
        A dictionnary with the two evaluation metrics.
        The dictionnary keys are "cer" and "wer" and their corresponding values
        the model performance on these metrics.

    """

    test_size = len(test)
    for x,y in test.rebatch(test_size).take(1):
        y_pred = model.predict(x)

    cer_result = CER(y, y_pred, test_size)
    wer_result = WER(y, y_pred, test_size)

    results = {
        "cer" : cer_result,
        "wer" : wer_result
    }

    return results



if __name__ == '__main__':

    # test = mappable_function("raw_data/sample_data_EN/videos/bbaf2n.mpg")
    # print(test)
    # print(type(test))
    # print(len(test))
    # print(test[1])


    # --- TRAINING ---

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


    # --- PREDICTION ---

    # model_name = "model_def_EN_1-6.h5"
    # model = load_model(model_name)
    # model.summary()

    # video_name = "Welcome to SilentSpeak.MOV"
    # video_name = "lrae5a - lay_red_at_e_five_again.mpg"
    # video_name = "other_lipnet - place_red_in_a_zero_now.mpg"
    # video_name = "KING CHARLES.mp4"
    # video_name = "lrae5a - lay_red_at_e_five_again.mp4"
    # video_name = "other_lipnet - place_red_in_a_zero_now.mp4"

    # video = os.path.join(data_path, "videos_demo", video_name)
    # prediction = predict_video(model, video)


    # --- EVALUATION ---

    # evaluation_result = evaluate_model(model, test)
    # print(evaluation_result)

    pass
