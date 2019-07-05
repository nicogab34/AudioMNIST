import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers

def build_model():
    #Encoder
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32,kernel_size = (3,3), input_shape = (227,227,1), activation='relu', name='conv_autoencoder1', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=2, strides = 2, padding='same',name='pool_autoencoder1'))
    model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', name='conv_autoencoder2', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=2,strides = 2, padding='same',name='pool_autoencoder2'))
    model.add(layers.Conv2D(8, kernel_size=(3,3), activation='relu', name='conv_autoencoder3', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=2,strides = 2, padding='same',name='pool_autoencoder3'))

    #Decoder
    model.add(layers.Conv2D(8, kernel_size=(3,3), activation='relu', name='conv_autoencoder4', padding='same'))
    model.add(layers.UpSampling2D((2,2), name='up_autoencoder1'))
    model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', name='conv_autoencoder5', padding='same'))
    model.add(layers.UpSampling2D((2,2), name='up_autoencoder2'))
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', name='conv_autoencoder6', padding='same'))
    model.add(layers.UpSampling2D((2,2),name='up_autoencoder3'))
    model.add(layers.Conv2D(1,(3,3),activation='linear',padding='same',name='conv_autoencoder7'))
    model.add(layers.Cropping2D(((0,5),(0,5)),name='crop'))
    return model