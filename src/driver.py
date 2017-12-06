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


    x_test = inputs[TEST_RANGE[0]: TEST_RANGE[1], :, :]
    y_test = outputs[TEST_RANGE[0]: TEST_RANGE[1], :, :]
    lengths_test = lengths[TEST_RANGE[0]: TEST_RANGE[1]]

    #x_train = x_train[:100, :, :]
    #y_train = y_train[:100, :, :]
    #lengths_train = lengths_train[:100]

    #x_val = x_val[:20, :, :]
    #y_val = y_val[:20, :, :]
    #lengths_val = lengths_val[:20]

    model = BidirectionalLSTMPredictor(SEQUENCE_LIMIT)
    model.train(x_train, y_train, lengths_train, x_val, y_val, lengths_val)

    print 'Test Accuracy: {}'.format(model.evaluate_loss(x_test, y_test, lengths_test)[1])
    predictions = model.predict(x_test, lengths_test)
    

if __name__ == '__main__':
    main()
