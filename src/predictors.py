# Neural Network Setup

from keras.models import Sequential, Model
from keras.layers import Bidirectional, Dense, Activation, Masking
from keras.layers import LSTM, TimeDistributed, RepeatVector, Input
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD
import keras.backend as backend
import numpy as np

from global_vars import *

from keras.callbacks import Callback

#class F1Score(Callback):



def truncated_accuracy(y_true, y_predict):
    mask = backend.sum(y_true, axis=2)
    y_pred_labels = backend.cast(backend.argmax(y_predict, axis=2), 'float32')
    y_true_labels = backend.cast(backend.argmax(y_true, axis=2), 'float32')
    is_same = backend.cast(backend.equal(y_true_labels, y_pred_labels), 'float32')
    num_same = backend.sum(is_same * mask, axis=1)
    lengths = backend.sum(mask, axis=1)
    return backend.mean(num_same / lengths, axis=0)


class EncoderDecoder(object):
    def __init__(self, max_seq_length):
        self.model = Sequential()
        self.max_seq_length = max_seq_length
        self.add_encoder()
        self.add_decoder()
        self.model.compile(optimizer='adagrad',
            loss='categorical_crossentropy',
            sample_weight_mode='temporal',
            metrics=[truncated_accuracy])
        self.model.summary()

    def add_encoder(self, max_seq_length):
        pass

    def add_decoder(self, max_seq_length, activation='tanh'):
        pass

    def train(self, x_train, y_train, train_lengths, num_epochs=5, batch_size=50):
        weight_mask = np.zeros((x_train.shape[0], self.max_seq_length))
        for i in range(len(train_lengths)):
            weight_mask[i, : train_lengths[i]] = 1.0
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_split=0.25,
          shuffle=True,
          sample_weight=weight_mask)

    def evaluate(self, x_test, y_test, test_lengths, batch_size=50):
        weight_mask = np.zeros((x_test.shape[0], self.max_seq_length))
        for i in range(len(test_lengths)):
            weight_mask[i, : test_lengths[i]] = 1.0
        return self.model.evaluate(x=x_test, y=y_test,
          batch_size=batch_size,
          sample_weight=weight_mask)

    def predict(self, x_test, test_lengths, batch_size=50):
        vectorized_predictions = self.model.predict(x_test,
            batch_size=batch_size,
            verbose=1)
        predictions = []
        for i in range(vectorized_predictions.shape[0]):
            predictions.append('')
            labels = np.argmax(vectorized_predictions[i, :, :], axis=1)
            for j in range(test_lengths[i]):
                predictions[-1] += LABEL_SET[labels[j]]
        return predictions

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

