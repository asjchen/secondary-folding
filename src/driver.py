# Driver Script

import argparse
import numpy as np
import matplotlib.pyplot as plt

from global_vars import *
import process_data
from predictors import BidirectionalLSTMPredictor

def main():
    parser = argparse.ArgumentParser(
        description=('Neural network architecture to predict '
            'secondary structure in proteins'))
    parser.add_argument('filename', help='Path to file of protein data')
    args = parser.parse_args()
    train_data, test_data = process_data.convert_data_labels(args.filename)
    train_data = train_data[:100]
    test_data = test_data[:100]
    max_seq_length = max(train_data['Sequence'].str.len().max(), test_data['Sequence'].str.len().max())
    model = BidirectionalLSTMPredictor(max_seq_length)

    print (len(train_data), max_seq_length, INPUT_DIM)
    train_x = np.zeros((len(train_data), max_seq_length, INPUT_DIM))
    for i in range(len(train_data)):
        train_x[i] = process_data.sequence_to_vectors(train_data.iloc[i]['Sequence'], max_seq_length)
    train_y = np.zeros((len(train_data), max_seq_length, len(LABEL_SET)))
    for i in range(len(train_data)):
        train_y[i] = process_data. labels_to_vector(train_data.iloc[i]['Labels'], max_seq_length)

    model.train(train_x, train_y, train_data['Sequence'].str.len().values)



if __name__ == '__main__':
    main()
