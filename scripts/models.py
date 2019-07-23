from tensorflow.keras.layers import Dense, GlobalMaxPool1D, Input
from tensorflow.keras import losses, models, optimizers


class ModelConfig(object):
    def __init__(self,
                 sampling_rate=44100, audio_duration=2,
                 n_folds=10, learning_rate=0.0001,
                 max_epochs=50, n_classes=41):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_classes = n_classes

        self.audio_length = self.sampling_rate * self.audio_duration


def get_dummy_model(config):

    nclass = config.n_classes
    input_length = config.audio_length

    inp = Input(shape=(input_length, 1))
    x = GlobalMaxPool1D()(inp)
    out = Dense(nclass, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(optimizer=opt, loss=losses.categorical_crossentropy,
                  metrics=['acc'])
    return model
