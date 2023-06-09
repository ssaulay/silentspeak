import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from silentspeak.params import vocab_type, vocab_phonemes, vocab_letters, n_frames, frame_h, frame_w, data_source, local_data_path, instance_data_path


if data_source == "local":
    data_path = local_data_path
else:
    data_path = instance_data_path
    pass

models_path = os.path.join(data_path, "..", "models")


if vocab_type == "p":
    vocab = vocab_phonemes
else:
    vocab = vocab_letters
vocab_size = len(vocab)


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

    #model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))
    model.add(Dense({vocab_size}, kernel_initializer='he_normal', activation='softmax'))

    print(model.summary())

    print("###### Compiling model ######")

    model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)

    return model
