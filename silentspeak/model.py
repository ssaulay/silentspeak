import os
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from silentspeak.params import vocab_type, vocab, n_frames, frame_h, frame_w, data_source, local_data_path, instance_data_path, google_data_path, test_local_video
from silentspeak.loading import char_to_num, num_to_char, load_data, load_video


if data_source == "local":
    data_path = local_data_path
elif data_source == "gcp":
    data_path = instance_data_path
else:
    data_path = google_data_path

models_path = os.path.join(data_path, "..", "models")


def scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


checkpoint_callback = ModelCheckpoint(
    os.path.join(models_path,'checkpoint'),
    monitor = 'loss',
    save_weights_only = True
    )

schedule_callback = LearningRateScheduler(scheduler)


def load_and_compile_model():

    print("###### Defining model ######")

    model = Sequential()

    # >>>> Check input shape in line below
    model.add(Conv3D(64, 3, input_shape=(n_frames, frame_h, frame_w, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(48, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
    #model.add(Dense(vocab_size, kernel_initializer='he_normal', activation='softmax'))

    print(model.summary())

    print("###### Compiling model ######")

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CTCLoss
        )

    return model


def save_model(model):
    """Save model as a h5 file"""

    print("###### SAVING MODEL ######")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"model_{timestr}.h5"
    model.save(os.path.join(models_path, model_filename))


def load_model(model_filename: str):
    """Load model"""

    print("###### LOADING MODEL ######")
    print(f"load: {model_filename}")

    model = tf.keras.models.load_model(
        os.path.join(models_path, model_filename),
        custom_objects = {"CTCLoss" : CTCLoss}
        )

    return model


def predict_test(
    model = None,
    path: str = test_local_video):

    if model is None:
        model = load_and_compile_model()
        print(" ####### LOAD WEIGHTS #######")
        model.load_weights(os.path.join(models_path, "checkpoint"))

    sample = load_data(tf.convert_to_tensor(path))

    print(" ####### PAD VIDEOS #######")

    paddings = tf.constant([[n_frames-sample[0].shape[0], 0], [0, 0], [0, 0], [0, 0]])
    sample = tf.pad(sample[0], paddings)

    print(f"sample shape : {sample.shape}")

    yhat = model.predict(tf.expand_dims(sample[0], axis=0))
    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[n_frames], greedy=True)[0][0].numpy()

    print('~'*100, 'PREDICTIONS')
    [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded]


def predict(
    model = None,
    path: str = test_local_video):

    if model is None:
        saved_models = [file for file in os.listdir(models_path) if file[-3:] == ".h5"]
        default_saved_model = saved_models[0]
        print("###### LOAD DEFAULT SAVED MODEL ######")
        print(f"load: {default_saved_model}")
        model = load_model(default_saved_model)

    loaded_video = load_video(test_local_video)
    paddings = tf.constant([[n_frames - loaded_video.shape[0], 0], [0, 0], [0, 0], [0, 0]])
    loaded_video_padded = tf.pad(loaded_video, paddings)
    loaded_video_padded = tf.expand_dims(loaded_video_padded, axis=0)
    yhat = model.predict(loaded_video_padded)

    decoded = tf.keras.backend.ctc_decode(
        tf.expand_dims(yhat[0], axis = 0),
        input_length=[n_frames],
        greedy=True)

    if vocab_type == "p" :
        decoded_string = tf.strings.reduce_join(
            [num_to_char(tf.argmax(x)) for x in yhat[0]],
            separator = "."
            )
    else:
         decoded_string = tf.strings.reduce_join(
            [num_to_char(tf.argmax(x)) for x in yhat[0]]
            )

    print("###### DECODED STRING ######")
    print(decoded_string)
    return yhat
