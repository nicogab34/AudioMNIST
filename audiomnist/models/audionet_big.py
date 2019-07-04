import tensorflow as tf
from tensorflow.keras import layers, initializers

def build_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(32, 11, input_shape= (8000, 1), padding='same', activation='relu', name='conv1'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool1'))

    model.add(layers.Conv1D(32, 11, padding='same', activation='relu', name='conv2'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool2'))

    model.add(layers.Conv1D(64, 7, padding='same', activation='relu', name='conv3'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool3'))

    model.add(layers.Conv1D(64, 7, padding='same', activation='relu', name='conv4'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool4'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv5'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool5'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv6'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool6'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv7'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool7'))

    model.add(layers.Conv1D(128, 5, padding='same', activation='relu', name='conv8'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool8'))

    model.add(layers.Conv1D(256, 3, padding='same', activation='relu', name='conv9'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool9'))

    model.add(layers.Conv1D(256, 3, padding='same', activation='relu', name='conv10'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool10'))

    model.add(layers.Conv1D(256, 3, padding='same', activation='relu', name='conv11'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool11'))

    model.add(layers.Conv1D(256, 3, padding='same', activation='relu', name='conv12'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, name='pool12'))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, name='dense', activation='softmax'))

    return model