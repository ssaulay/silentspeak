import os
import tensorflow as tf
import numpy as np

from silentspeak.loading import *
from silentspeak.params import local_data_path, instance_data_path, data_size, data_source, n_frames
from silentspeak.model import load_and_compile_model, checkpoint_callback, schedule_callback, predict_test


if data_source == "local":
    data_path = local_data_path
else:
    data_path = instance_data_path
    pass


def mappable_function(path:str) ->List[str]:
    """
    inputs:
    >> path:
    """
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


def train(
    epochs = 10,
    batch_size = 2,
    padded_frames_shape = [n_frames,None,None,None],
    padded_transcripts_shape = [50],
    train_size = 10,
    callbacks = [checkpoint_callback, schedule_callback]
):
    """
    Train the model
    """

    # Load data
    data = tf.data.Dataset.list_files(
        os.path.join(data_path, data_size, "videos", "*.avi")
        )
    data = data.map(mappable_function)
    data = data.padded_batch(
        batch_size,
        padded_shapes = (padded_frames_shape, padded_transcripts_shape))
    data = data.prefetch(tf.data.AUTOTUNE)

    frames, alignments = data.as_numpy_iterator().next()
    #sample = data.as_numpy_iterator()

    # Added for split
    # train = data.take(train_size)
    # test = data.skip(train_size)

    print("###### Load and compile model ######")

    model = load_and_compile_model()

    print("###### Train model ######")

    model.fit(
        data,
        epochs = epochs,
        callbacks = callbacks
        )

    return model


if __name__ == '__main__':
    # download_data
    # preprocess
    model = train(epochs = 2)
    predict_test(model = model)
    # evaluate
    # pred
    pass
