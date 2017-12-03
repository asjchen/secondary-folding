# Neural Network Setup

from keras.models import Sequential, Model
from keras.layers import Bidirectional, Dense, Activation, Masking
from keras.layers import LSTM, TimeDistributed, RepeatVector, Input
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD
import keras.backend as backend
import numpy as np

from global_vars import *

import keras.callbacks as cbks

class CustomMetrics(cbks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        for k in logs:
            if k.endswith('truncated_accuracy'):
                f = open('y_true.txt', 'w')
                f.write(str(logs[k]))
                f.close()

# TODO: write a F1 score function on a batch
def f1_score(y_true, y_predict):
    #return y_true
    return truncated_accuracy(y_true, y_predict)

def truncated_accuracy(y_true, y_predict):
    mask = backend.sum(y_true, axis=2)
    is_same = backend.cast(backend.all(backend.equal(y_true, y_predict), axis=2), 'float32')
    num_same = backend.sum(is_same * mask, axis=1)
    lengths = backend.sum(mask, axis=1)
    return backend.mean(num_same / lengths, axis=0)


# class EncoderDecoder(object):
#     def __init__(self):
#         encoder_inputs = Input(shape=(None, INPUT_DIM))
#         # TODO: change activation function
#         encoder = LSTM(HIDDEN_DIM, return_state=True)
#         _, state_h, state_c = encoder(encoder_inputs)

#         decoder_inputs = Input(shape=(None, OUTPUT_DIM))
#         decoder = LSTM(HIDDEN_DIM, return_sequences=True)
#         decoder_outputs = decoder(decoder_inputs, initial_state=[state_h, state_c])
#         decoder_softmax = Dense(OUTPUT_DIM, activation='softmax')
#         decoder_outputs = decoder_softmax(decoder_outputs)

#         self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#         self.model.compile(optimizer='adagrad',
#             loss='categorical_crossentropy',
#             # metrics=[f1_score])
#             metrics=[f1_score])
#         self.model.summary()

#     def train(self, encoder_input_train, decoder_input_train, decoder_output_train,
#         num_epochs=5, batch_size=10):
#         self.model.fit([encoder_input_train, decoder_input_train], decoder_output_train,
#             batch_size=batch_size,
#             epochs=num_epochs,
#             validation_split=0.25)
#         self.model.save('model.h5')



class EncoderDecoder(object):
    def __init__(self, max_seq_length):
        self.model = Sequential()
        self.max_seq_length = max_seq_length
        self.add_encoder()
        self.add_decoder()
        self.model.compile(optimizer='adagrad',
            loss='categorical_crossentropy',
            sample_weight_mode='temporal',
            #metrics=[truncated_accuracy])
            metrics=[categorical_accuracy])
        self.model.summary()

    def add_encoder(self, max_seq_length):
        pass

    def add_decoder(self, max_seq_length, activation='tanh'):
        pass

    def train(self, x_train, y_train, train_lengths, num_epochs=5, batch_size=10):
        weight_mask = np.zeros((x_train.shape[0], self.max_seq_length))
        for i in range(len(train_lengths)):
            weight_mask[i, : train_lengths[i]] = 1.0
        self.model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          validation_split=0.25,
          shuffle=True,
          sample_weight=weight_mask, callbacks=[CustomMetrics()])

    def predict(self, x_test, test_lengths, batch_size=10):
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

