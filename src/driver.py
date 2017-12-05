# Driver Script

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from global_vars import *
import process_data
from predictors import BidirectionalLSTMPredictor

def proportion_unclassified(row):
    labels = row['Labels']
    count = len([1 for i in range(len(labels)) if labels[i] == UNCLASSIFIED])
    return float(count) / len(labels)

def main():
    parser = argparse.ArgumentParser(
        description=('Neural network architecture to predict '
            'secondary structure in proteins'))
    parser.add_argument('filename', help='Path to file of protein data')
    args = parser.parse_args()

    inputs, outputs, lengths = process_data.npy_to_input_data(args.filename)
    x_train = inputs[TRAINING_RANGE[0]: TRAINING_RANGE[1], :, :]
    y_train = outputs[TRAINING_RANGE[0]: TRAINING_RANGE[1], :, :]
    lengths_train = lengths[TRAINING_RANGE[0]: TRAINING_RANGE[1]]


    x_val = inputs[VALIDATION_RANGE[0]: VALIDATION_RANGE[1], :, :]
    y_val = outputs[VALIDATION_RANGE[0]: VALIDATION_RANGE[1], :, :]
    lengths_val = lengths[VALIDATION_RANGE[0]: VALIDATION_RANGE[1]]

    x_train = x_train[:100, :, :]
    y_train = y_train[:100, :, :]
    lengths_train = lengths_train[:100]

    x_val = x_val[:50, :, :]
    y_val = y_val[:50, :, :]
    lengths_val = lengths_val[:50]

    model = BidirectionalLSTMPredictor(SEQUENCE_LIMIT)
    # x_train, y_train = process_data.dataframe_to_np_data(train_data, SEQUENCE_LIMIT)
    model.train(x_train, y_train, lengths_train, x_val, y_val, lengths_val)
























    # train_data['Proportion Unclassified'] = train_data.apply(proportion_unclassified, axis=1)
    # train_data = train_data[train_data['Proportion Unclassified'] < 0.5]
    

    # # freq = {}
    # # for i in range(len(train_data)):
    # #     for j in range(len(train_data.iloc[i]['Sequence']) - WINDOW_SIZE + 1):
    # #         sub = train_data.iloc[i]['Sequence'][j: j + WINDOW_SIZE]
    # #         if sub not in freq:
    # #             freq[sub] = { l: 0 for l in LABEL_SET }
    # #         freq[sub][train_data.iloc[i]['Labels'][j + (WINDOW_SIZE - 1) / 2]] += 1
    # # for sub in freq:
    # #     print '{} {}'.format(sub, freq[sub])



    # max_seq_length = max(train_data['Sequence'].str.len().max(), test_data['Sequence'].str.len().max())
    
    # model = BidirectionalLSTMPredictor(max_seq_length)
    # x_train, y_train = process_data.dataframe_to_np_data(train_data, max_seq_length)
    # train_lengths = train_data['Sequence'].str.len().values
    # model.train(x_train, y_train, train_lengths)

    # x_test, y_test = process_data.dataframe_to_np_data(test_data, max_seq_length)
    # test_lengths = test_data['Sequence'].str.len().values
    # # print 'Test Loss: {}'.format(model.evaluate_loss(x_test, y_test, test_lengths))
    # predictions = model.predict(x_test, test_lengths)


    #print predictions
    

if __name__ == '__main__':
    main()
