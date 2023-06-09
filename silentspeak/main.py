import os
import tensorflow as tf
import numpy as np

from silentspeak.loading import *
from silentspeak.params import local_data_path, data_size, data_source
from silentspeak.model import load_model

if data_source == "local":
    data_path = local_data_path
else:
    #data_path = # THE PATH OF RAW DATA ON THE VM INSTANCE
    pass

def mappable_function(path:str) ->List[str]:
    """
    inputs:
    >> path:
    """
    result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
    return result


def train(
    batch_size = 2,
    padded_frames_shape = [75,None,None,None],
    padded_transcripts_shape = [50],
    train_size = 10
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
    train = data.take(train_size)
    test = data.skip(train_size)


    model = load_model()
    # >>>> Reprendre ici



if __name__ == '__main__':
    # preprocess
    print(train())
    # evaluate
    # pred
    pass
