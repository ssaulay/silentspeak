import os
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from silentspeak.params import vocab_type, vocab, n_frames, n_frames_min, frame_h, frame_w, transcript_padding, data_path, data_size, test_local_video
from silentspeak.loading import char_to_num, num_to_char, load_data, load_video, process_video


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


class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset, batch_size) -> None:
        self.original_dataset = dataset
        self.dataset = dataset.as_numpy_iterator()
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None) -> None:
        try:
          data = self.dataset.next()
        except StopIteration:
          self.dataset = self.original_dataset.as_numpy_iterator()
          data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [n_frames] * self.batch_size, greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

        # tf.keras.backend.ctc_decode(
        # tf.expand_dims(yhat[0], axis = 0),
        # input_length=[n_frames],
        # greedy=True)


def model_1():
    """Instantiate a new model"""

    print("###### Defining model - Type 1 ######")

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

    return model


def model_2():
    """Instantiate a new model"""

    print("###### Defining model - Type 2 ######")

    model = Sequential()

    model.add(Conv3D(64, kernel_size=5, strides=(1,2,2), input_shape=(n_frames, frame_h, frame_w, 1), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv3D(128, kernel_size=(1,3,3), strides = (1,2,2), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv3D(256, kernel_size=(1,3,3), strides = (1,2,2), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv3D(256, kernel_size=(1,3,3), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv3D(256, kernel_size=(1,3,3), padding="same"))
    model.add(Activation('relu'))

    model.add(Conv3D(512, kernel_size=(1,3,3), strides=(1,2,2), padding="same"))
    model.add(Activation('relu'))


    model.add(Conv3D(512, kernel_size=(1,3,3), padding="same"))
    model.add(Activation('relu'))


    model.add(Conv3D(512, kernel_size=(1,3,3), strides = (1,2,2), padding="same"))
    model.add(Activation('relu'))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    # model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    # model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

    return model


def instantiate_model(model_num = 1):
    if model_num == 1:
        model = model_1()
    elif model_num == 2:
        model = model_2()
    return model


def compile_model(model):
    """Compile an already instantiated model"""

    print("###### Compiling model ######")

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=CTCLoss
        )

    return model


def load_and_compile_model(model_num = 1):
    """Instantiate a new model and compile it."""

    model = instantiate_model(model_num)
    model = compile_model(model)
    return model


def save_model(model):
    """Save model as a h5 file"""

    print("###### SAVING MODEL ######")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"model_{data_size}_{n_frames_min}_{n_frames}_{transcript_padding}_{timestr}.h5"
    model.save(os.path.join(models_path, model_filename))


def load_model(model_filename: str):
    """Load model from h5 file"""

    print("###### LOADING MODEL ######")
    print(f"load: {model_filename}")

    model = tf.keras.models.load_model(
        os.path.join(models_path, model_filename),
        custom_objects = {"CTCLoss" : CTCLoss}
        )

    return model


def load_model_weigths(model, checkpoint: str = os.path.join(models_path, "checkpoint")):
    """Load weights into an already instantiated model"""

    print(" ####### LOAD WEIGHTS #######")
    model.load_weights(checkpoint)

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
    path: str = test_local_video,
    vocab_type = vocab_type
    ):

    if model is None:
        saved_models = [file for file in os.listdir(models_path) if file[-3:] == ".h5"]
        default_saved_model = saved_models[0]
        print("###### LOAD DEFAULT SAVED MODEL ######")
        print(f"load: {default_saved_model}")
        model = load_model(default_saved_model)

    loaded_video = load_video(path)
    paddings = tf.constant([[n_frames - loaded_video.shape[0], 0], [0, 0], [0, 0], [0, 0]])
    loaded_video_padded = tf.pad(loaded_video, paddings)
    loaded_video_padded = tf.expand_dims(loaded_video_padded, axis=0)

    print("###### PREDICT ######")
    yhat = model.predict(loaded_video_padded)

    decoded = tf.keras.backend.ctc_decode(
        tf.expand_dims(yhat[0], axis = 0),
        input_length=[n_frames],
        greedy=True)

    #print(decoded.numpy())

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



def predict_video(model, video_path: str) -> str:
    """
    Takes a video as an input and returns a prediction in string

    inputs:
    >> model: the model that will returns the prediction from the video
    >> video_path: the path to the video to predict

    output:
    >> the prediction expressed as a string
    """

    processed_video = process_video(video_path)

    pad_after = n_frames - processed_video.shape[0]

    paddings = tf.constant([[0, pad_after], [0, 0], [0, 0], [0, 0]])
    video_padded = tf.pad(processed_video, paddings)
    video_pred = tf.expand_dims(video_padded, axis=0)

    yhat = model.predict(video_pred)

    decoded = tf.keras.backend.ctc_decode(yhat, input_length=[n_frames], greedy=True)[0][0].numpy()

    prediction = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in decoded][0].numpy().decode("UTF-8")

    print("##### PREDICT #####")
    print('~'*100)
    print(f"Prediction: {prediction}")

    return prediction
