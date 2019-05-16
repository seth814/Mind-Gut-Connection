from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def preprocess_input(x):
    x = log10(x+1e-6)+6
    x = x / 8.0
    return x


def ConvBlock(depth, filters, name, kernel_size=(3,3), activation='elu'):

    tmp = Sequential(name=name)
    for i in range(depth):
        tmp.add(layers.Conv2D(filters=filters,
                              kernel_size=kernel_size,
                              activation=activation,
                              padding='same'))
    tmp.add(layers.MaxPooling2D(pool_size=(2, 2)))
    return tmp


def ConvNet(input_shape, base=3):

    # input_shape is (n, time, feats)
    b = base

    i = layers.Input(shape=(input_shape[0], input_shape[1], 1))
    s = layers.Lambda(preprocess_input)(i)
    x = ConvBlock(depth=2, filters=2**b,     kernel_size=(5,5), name='conv_block_1')(s)
    x = ConvBlock(depth=2, filters=2**(b+1), kernel_size=(3,3), name='conv_block_2')(x)
    x = ConvBlock(depth=2, filters=2**(b+2), kernel_size=(3,3), name='conv_block_3')(x)
    x = ConvBlock(depth=2, filters=2**(b+3), kernel_size=(3,3), name='conv_block_4')(x)
    x = ConvBlock(depth=2, filters=2**(b+4), kernel_size=(3,3), name='conv_block_5')(x)
    x = ConvBlock(depth=2, filters=2**(b+5), kernel_size=(3,3), name='conv_block_6')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.2, name='dropout_0.2')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(rate=0.1, name='dropout_0.1')(x)
    o = layers.Dense(6, activation='softmax')(x)

    model = Model(inputs=i, outputs=o)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-4),
                  metrics=['accuracy'])
    model.summary()

    return model


def TimeDistributed2DConvBlock(depth, filters, name, kernel_size=(3,3), activation='elu'):

    tmp = Sequential(name=name)
    for i in range(depth):
        tmp.add(layers.TimeDistributed(layers.Conv2D(filters=filters,
                                                     kernel_size=kernel_size,
                                                     activation=activation,
                                                     padding='same')))
    tmp.add(layers.TimeDistributed(layers.MaxPooling2D(pool_size=(2, 2))))
    return tmp


def Recurrent2DConvNet(input_shape, base=3):

    # input_shape is (n, time, feat_time, feats)
    b = base

    i = layers.Input(shape=input_shape)
    s = layers.TimeDistributed(layers.Lambda(preprocess_input), name='scale_input')(i)
    x = TimeDistributed2DConvBlock(depth=3, filters=2**b,     kernel_size=(3, 3), name='conv_block_1')(s)
    x = TimeDistributed2DConvBlock(depth=3, filters=2**(b+1), kernel_size=(3, 3), name='conv_block_2')(x)
    x = TimeDistributed2DConvBlock(depth=3, filters=2**(b+2), kernel_size=(3, 3), name='conv_block_3')(x)
    x = TimeDistributed2DConvBlock(depth=3, filters=2**(b+3), kernel_size=(3, 3), name='conv_block_4')(x)
    x = TimeDistributed2DConvBlock(depth=3, filters=2**(b+4), kernel_size=(3, 3), name='conv_block_5')(x)
    x = layers.TimeDistributed(layers.Flatten(), name='flatten')(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(rate=0.2, name='dropout_0.2')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(rate=0.1, name='dropout_0.1')(x)
    o = layers.Dense(6, activation='softmax')(x)

    model = Model(inputs=i, outputs=o)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-4),
                  metrics=['accuracy'])
    model.summary()

    return model
