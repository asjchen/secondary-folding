# Processing the data downloaded from UniProt

import pandas as pd
import math
import numpy as np

from global_vars import *

def construct_labels(row):
    labels = [UNCLASSIFIED] * row['Length']
    name_to_structure = { 'Beta strand': STRAND, 'Turn': TURN, 'Helix': HELIX }
    for name in name_to_structure:
        if type(row[name]) == str:
            structure_data = row[name].split(';')
            for structure in structure_data:
                indices = [int(raw_idx) - 1 for raw_idx in structure.strip().split()[1:3]]
                labels[indices[0]: indices[1] + 1] = [name_to_structure[name]] * (indices[1] - indices[0] + 1)
    return ''.join(labels)

# data_split = [proportion of training data, proportion of development data]
def convert_data_labels(filename, train_proportion=0.6):
    protein_df = pd.read_csv(filename, sep='\t')
    protein_df['Labels'] = protein_df.apply(construct_labels, axis=1)
    protein_df = protein_df[protein_df['Labels'].str.replace(UNCLASSIFIED, '').str.len() != 0]
    
    # Filter out long proteins for computation feasibility
    protein_df = protein_df[protein_df['Labels'].str.len() <= SEQUENCE_LIMIT]
    protein_df = protein_df.sample(frac=1)
    train_boundary = int(train_proportion * len(protein_df))
    train_data = protein_df.iloc[: train_boundary]
    test_data = protein_df.iloc[train_boundary: ]
    return train_data, test_data

def sequence_to_vectors(sequence, max_seq_length):
    assert WINDOW_SIZE % 2 == 1
    new_step_boundary = (WINDOW_SIZE - 1) * NUM_AMINO_ACIDS
    seq_vectors = np.zeros((max_seq_length, INPUT_DIM))
    for i in range(len(sequence)):
        if i > 0:
            seq_vectors[i, : new_step_boundary] = seq_vectors[i - 1, NUM_AMINO_ACIDS: ]
            amino_idx = AMINO_ACIDS.index(sequence[i])
            seq_vectors[i, new_step_boundary + amino_idx] += 1
    return seq_vectors

def labels_to_vector(labels, max_seq_length):
    label_vector = np.zeros((max_seq_length, len(LABEL_SET)))
    for i in range(len(labels)):
        label_vector[i, LABEL_SET.index(labels[i])] += 1
    return label_vector
