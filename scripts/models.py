from tensorflow.keras.layers import (Dense, GlobalMaxPool1D, GlobalMaxPool2D, Input,
                                     Convolution2D, BatchNormalization,
                                     MaxPool2D, Flatten, Activation)
from tensorflow.keras import losses, models, optimizers
from tensorflow.keras.activations import softmax
# from keras_contrib.layers.capsule import Capsule


class ModelConfig(object):
    def __init__(self,
                 sampling_rate=44100, audio_duration=2,
                 n_folds=3, learning_rate=0.0001,
                 max_epochs=5, n_classes=41):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = n_classes

        self.audio_length = self.sampling_rate * self.audio_duration


def get_dummy_model(model_config, data_config):

    nclass = model_config.n_classes
    input_length = model_config.audio_length

    inp = Input(shape=(input_length, 1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(model_config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    return model


def get_aug_baseline_model(model_config, data_config):



    nclass = model_config.n_classes
    inp = Input(shape=(data_config.dim[0], data_config.dim[1], data_config.dim[2]))
    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = GlobalMaxPool2D()(x)

    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(model_config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    print(model.summary())
    return model

def get_baseline_model(model_config, data_config):

    nclass = model_config.n_classes

    inp = Input(shape=(data_config.dim[0], data_config.dim[1], 1))
    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(model_config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    print(model.summary())
    return model


def get_capsule_model(model_config, data_config):

    nclass = model_config.n_classes

    inp = Input(shape=(data_config.dim[0], data_config.dim[1], 1))
    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Capsule(num_capsule=10, dim_capsule=16, routings=5,
                activation='sigmoid', share_weights=True)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Capsule(num_capsule=10, dim_capsule=16, routings=5,
                activation='sigmoid', share_weights=True)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Capsule(num_capsule=10, dim_capsule=16, routings=5,
                activation='sigmoid', share_weights=True)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Capsule(num_capsule=10, dim_capsule=16, routings=5,
                activation='sigmoid', share_weights=True)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(model_config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy,
                  metrics=['acc'])

    return model
