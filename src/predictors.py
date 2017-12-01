# Neural Network Setup

from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Activation, Masking
from keras.layers import LSTM, TimeDistributed, RepeatVector
import numpy as np

from global_vars import *

# TODO: write a F1 score function on a batch
# def f1_score(y_true, y_predict):
# make sure to account for variable length

class EncoderDecoder(object):
    def __init__(self, max_seq_length):
        self.model = Sequential()
        self.max_seq_length = max_seq_length
        self.add_encoder()
        self.add_decoder()
        self.model.compile(optimizer='rmsprop',
            loss='categorical_crossentropy',
            sample_weight_mode='temporal',
            # metrics=[f1_score])
            metrics=['accuracy'])
        self.model.summary()

    def add_encoder(self, max_seq_length):
        pass

    def add_decoder(self, max_seq_length, activation='tanh'):
        pass

    def train(self, x_train, y_train, train_lengths, num_epochs=20, batch_size=32):
        weight_mask = np.zeros((x_train.shape[0], self.max_seq_length))
        for i in range(len(train_lengths)):
            weight_mask[i, : train_lengths[i]] = 1.0
        print x_train.shape
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_split=0.25,
          sample_weight=weight_mask)

    # def predict(self):

class BidirectionalLSTMPredictor(EncoderDecoder):
    # TODO: potentially use dropout
    def add_encoder(self, activation='tanh'):
        self.model.add(Masking(mask_value=0, input_shape=(self.max_seq_length, INPUT_DIM)))
        self.model.add(Bidirectional(LSTM(HIDDEN_DIM, activation=activation), merge_mode='concat'))

    def add_decoder(self, activation='tanh'):
        self.model.add(RepeatVector(self.max_seq_length))
        self.model.add(Bidirectional(LSTM(HIDDEN_DIM, activation=activation, return_sequences=True), merge_mode='concat'))
        self.model.add(TimeDistributed(Dense(len(LABEL_SET))))
        self.model.add(Activation('softmax'))

