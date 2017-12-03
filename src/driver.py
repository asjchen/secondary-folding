# Driver Script

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    # import random, re
    # ll = ['A' * random.randint(10, 20) + 'B' * random.randint(10, 20) for _ in range(1000)]

    # d = { 'Sequence': ll, 'Labels': [re.sub('A', 'S', re.sub('B', 'H', s)) for s in ll] }
    
    # train_data = pd.DataFrame(data=d)
    # test_data = pd.DataFrame(data=d)
    max_seq_length = max(train_data['Sequence'].str.len().max(), test_data['Sequence'].str.len().max())
    
    model = BidirectionalLSTMPredictor(max_seq_length)
    x_train, y_train = process_data.dataframe_to_np_data(train_data, max_seq_length)
    train_lengths = train_data['Sequence'].str.len().values
    model.train(x_train, y_train, train_lengths)

    x_test, y_test = process_data.dataframe_to_np_data(test_data, max_seq_length)
    test_lengths = test_data['Sequence'].str.len().values
    print 'Test Accuracy: {}'.format(model.evaluate(x_test, y_test, test_lengths)[0])
    predictions = model.predict(x_test, test_lengths)
    

if __name__ == '__main__':
    main()
