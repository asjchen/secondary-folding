# Processing the raw input data

import numpy as np

from global_vars import *

# The filename should map to the location of the .npy file containing
# the raw input data. Generally, the suffix is cullpdb+profile_6133.npy
# See http://www.princeton.edu/~jzthree/datasets/ICML2014/dataset_readme.txt
# for the structure of the dataset
def npy_to_input_data(filename):
    raw_data = np.load(filename)
    assert raw_data.shape[0] == VALIDATION_RANGE[1]

    raw_data = raw_data.reshape(
        (raw_data.shape[0], SEQUENCE_LIMIT, NUM_ORIG_FEATURES))
    
    # Ignore the NoSeq characters
    raw_input_vectors = raw_data[:, :, : NUM_AMINO_ACIDS]

    # We record the amino acid sequence lengths to create a boolean mask
    lengths = np.sum(np.sum(raw_input_vectors, axis=2), axis=1).astype(int)
    input_vectors = np.zeros(
        (raw_input_vectors.shape[0], SEQUENCE_LIMIT, INPUT_DIM))

    # Each row of dimension INPUT_DIM is the concatenation of the one-hot
    # vectors of the amino acids at positions 
    # [p - (WINDOW_SIZE - 1) / 2, ..., p + (WINDOW_SIZE - 1) / 2]
    last_step_idx = INPUT_DIM - INDIV_INPUT_DIM
    for i in range(input_vectors.shape[0]):
        for j in range(lengths[i]):
            if j == 0:
                init_idx = ((WINDOW_SIZE - 1) / 2) * INDIV_INPUT_DIM
                for k in range((WINDOW_SIZE - 1) / 2):
                    start_idx = init_idx + k * INDIV_INPUT_DIM
                    end_idx = start_idx + INDIV_INPUT_DIM
                    last_indiv_vec = raw_input_vectors[i, k, :]
                    input_vectors[i, 0, start_idx: end_idx] = last_indiv_vec
            else:
                input_vectors[i, j, : last_step_idx] = input_vectors[i, j - 1, 
                    INDIV_INPUT_DIM: ]
            next_idx = j + ((WINDOW_SIZE - 1) / 2)        
            if next_idx < lengths[i]:
                input_vectors[i, j, last_step_idx: ] = raw_input_vectors[
                    i, next_idx, :]

    # Plus one because there is a NoSeq character in addition to the amino acids
    out_start = NUM_AMINO_ACIDS + 1
    out_end = NUM_AMINO_ACIDS + 1 + NUM_LABELS
    output_vectors = raw_data[:, :, out_start: out_end]
    return input_vectors, output_vectors, lengths
