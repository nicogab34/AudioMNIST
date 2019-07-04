import tensorflow as tf
from tensorflow.keras import layers, initializers, optimizers

def build_model():
    #Encoder
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32,kernel_size = (3,3), input_shape = (227,227,1), activation='relu', name='conv1', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=2, strides = 2, padding='same',name='pool1'))
    model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', name='conv2', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=2,strides = 2, padding='same',name='pool2'))
    model.add(layers.Conv2D(8, kernel_size=(3,3), activation='relu', name='conv3', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=2,strides = 2, padding='same',name='pool3'))

    #Decoder
    model.add(layers.Conv2D(8, kernel_size=(3,3), activation='relu', name='conv4', padding='same'))
    model.add(layers.UpSampling2D((2,2), name='up1'))
    model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', name='conv5', padding='same'))
    model.add(layers.UpSampling2D((2,2), name='up2'))
    model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', name='conv6', padding='same'))
    model.add(layers.UpSampling2D((2,2),name='up3'))
    model.add(layers.Conv2D(1,(3,3),activation='linear',padding='same',name='conv7'))
    model.add(layers.Cropping2D(((0,5),(0,5)),name='crop'))
    return model