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
    
    train_data = train_data[:10]
    test_data = test_data[:10]

    import random, re
    ll = ['A' * random.randint(10, 20) + 'B' * random.randint(10, 20) for _ in range(1000)]

    d = { 'Sequence': ll, 'Labels': [re.sub('A', 'S', re.sub('B', 'H', s)) for s in ll] }
    
    train_data = pd.DataFrame(data=d)
    test_data = pd.DataFrame(data=d)

    max_seq_length = max(train_data['Sequence'].str.len().max(), test_data['Sequence'].str.len().max())
    

    # TODO: change this model to a simpler one for initial testing
    model = BidirectionalLSTMPredictor(max_seq_length)
    # model = EncoderDecoder()


    encoder_input_train = np.zeros((len(train_data), max_seq_length, INPUT_DIM))
    for i in range(len(train_data)):
        encoder_input_train[i] = process_data.sequence_to_vectors(train_data.iloc[i]['Sequence'], max_seq_length)
    decoder_input_train = np.zeros((len(train_data), max_seq_length, OUTPUT_DIM))
    for i in range(len(train_data)):
        decoder_input_train[i] = process_data.labels_to_vector(train_data.iloc[i]['Labels'], max_seq_length)
    decoder_output_train = np.zeros((len(train_data), max_seq_length, OUTPUT_DIM))
    for i in range(len(train_data) - 1):
        decoder_output_train[i] = decoder_input_train[i + 1]
    #print encoder_input_train 
    #print decoder_input_train
    #print decoder_output_train
    # model.train(encoder_input_train, decoder_input_train, decoder_output_train)
    


    # train_x = np.zeros((len(train_data), max_seq_length, INPUT_DIM))
    # for i in range(len(train_data)):
    #     train_x[i] = process_data.sequence_to_vectors(train_data.iloc[i]['Sequence'], max_seq_length)
    # train_y = np.zeros((len(train_data), max_seq_length, len(LABEL_SET)))
    # for i in range(len(train_data)):
    #     train_y[i] = process_data. labels_to_vector(train_data.iloc[i]['Labels'], max_seq_length)
    
    model.train(encoder_input_train, decoder_input_train, train_data['Sequence'].str.len().values)

    encoder_input_test = np.zeros((len(test_data), max_seq_length, INPUT_DIM))
    for i in range(len(test_data)):
        encoder_input_test[i] = process_data.sequence_to_vectors(test_data.iloc[i]['Sequence'], max_seq_length)
    predictions = model.predict(encoder_input_test, test_data['Sequence'].str.len().values)
    

    # errors = []
    # for i in range(len(test_data)):
    #     num_diff = len([1 for j in range(len(predictions[i])) if predictions[i][j] != train_data.iloc[i]['Labels'][j]])
    #     errors.append(float(num_diff) / len(predictions[i]))
    # print errors

if __name__ == '__main__':
    main()
