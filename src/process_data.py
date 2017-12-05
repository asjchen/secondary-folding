# Processing the data downloaded from UniProt

import pandas as pd
import math
import numpy as np

from global_vars import *

def npy_to_input_data(filename):
    raw_data = np.load(filename)
    assert raw_data.shape[0] == VALIDATION_RANGE[1]

    raw_data = raw_data.reshape((raw_data.shape[0], SEQUENCE_LIMIT, NUM_ORIG_FEATURES))
    
    # If we create a mask and ignore the noseq characters
    raw_input_vectors = raw_data[:, :, : NUM_AMINO_ACIDS]
    lengths = np.sum(np.sum(raw_input_vectors, axis=2), axis=1).astype(int)
    input_vectors = np.zeros((raw_input_vectors.shape[0], SEQUENCE_LIMIT, INPUT_DIM))
    for i in range(input_vectors.shape[0]):
        for j in range(lengths[i]):
            if j == 0:
                init_idx = ((WINDOW_SIZE - 1) / 2) * NUM_AMINO_ACIDS
                for k in range((WINDOW_SIZE - 1) / 2):
                    start_idx = init_idx + k * NUM_AMINO_ACIDS
                    input_vectors[i, 0, start_idx: start_idx + NUM_AMINO_ACIDS] = raw_input_vectors[i, k, :]
            else:
                input_vectors[i, j, : INPUT_DIM - NUM_AMINO_ACIDS] = input_vectors[i, j - 1, NUM_AMINO_ACIDS: ]
            next_idx = j + ((WINDOW_SIZE - 1) / 2)        
            if next_idx < lengths[i]:
                input_vectors[i, j, INPUT_DIM - NUM_AMINO_ACIDS: ] = raw_input_vectors[i, next_idx, :]
    # Plus one because there is a NoSeq character in addition to the amino acids
    output_vectors = raw_data[:, :, NUM_AMINO_ACIDS + 1: NUM_AMINO_ACIDS + 1 + NUM_LABELS]
    return input_vectors, output_vectors, lengths





# def construct_labels(row):
#     labels = [UNCLASSIFIED] * row['Length']
#     name_to_structure = { 'Beta strand': STRAND, 'Turn': TURN, 'Helix': HELIX }
#     for name in name_to_structure:
#         if type(row[name]) == str:
#             structure_data = row[name].split(';')
#             for structure in structure_data:
#                 indices = [int(raw_idx) - 1 for raw_idx in structure.strip().split()[1:3]]
#                 labels[indices[0]: indices[1] + 1] = [name_to_structure[name]] * (indices[1] - indices[0] + 1)
#     return ''.join(labels)

# # data_split = [proportion of training data, proportion of development data]
# def convert_data_labels(filename, train_proportion=0.6):
#     protein_df = pd.read_csv(filename, sep='\t')
#     protein_df['Labels'] = protein_df.apply(construct_labels, axis=1)
#     protein_df = protein_df[protein_df['Labels'].str.replace(UNCLASSIFIED, '').str.len() != 0]
    
#     # Filter out long proteins for computation feasibility
#     protein_df = protein_df[protein_df['Labels'].str.len() <= SEQUENCE_LIMIT]
#     protein_df = protein_df.sample(frac=1)
#     train_boundary = int(train_proportion * len(protein_df))
#     train_data = protein_df.iloc[: train_boundary]
#     test_data = protein_df.iloc[train_boundary: ]
#     return train_data, test_data

# def sequence_to_vectors(sequence, max_seq_length):
#     assert WINDOW_SIZE % 2 == 1
#     new_step_boundary = (WINDOW_SIZE - 1) * NUM_AMINO_ACIDS
#     seq_vectors = np.zeros((max_seq_length, INPUT_DIM))
#     for i in range(len(sequence)):
#         if i > 0:
#             seq_vectors[i, : new_step_boundary] = seq_vectors[i - 1, NUM_AMINO_ACIDS: ]
#         else:
#             for j in range((WINDOW_SIZE - 1) / 2):
#                 amino_idx = AMINO_ACIDS.index(sequence[j])
#                 start_idx = ((WINDOW_SIZE - 1) / 2 + j) * NUM_AMINO_ACIDS
#                 seq_vectors[i, start_idx + amino_idx] += 1
#         end_idx = i + (WINDOW_SIZE - 1) / 2
#         if end_idx < len(sequence):
#             amino_idx = AMINO_ACIDS.index(sequence[end_idx])
#             seq_vectors[i, new_step_boundary + amino_idx] += 1
#     return seq_vectors

# def labels_to_vector(labels, max_seq_length):
#     label_vector = np.zeros((max_seq_length, len(LABEL_SET)))
#     for i in range(len(labels)):
#         label_vector[i, LABEL_SET.index(labels[i])] += 1
#     return label_vector

# def dataframe_to_np_data(dataframe, max_seq_length):
#     x = np.zeros((len(dataframe), max_seq_length, INPUT_DIM))
#     for i in range(len(dataframe)):
#         x[i] = sequence_to_vectors(dataframe.iloc[i]['Sequence'], max_seq_length)
#     y = np.zeros((len(dataframe), max_seq_length, OUTPUT_DIM))
#     for i in range(len(dataframe)):
#         y[i] = labels_to_vector(dataframe.iloc[i]['Labels'], max_seq_length)
#     return x, y

