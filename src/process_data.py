# Processing the data downloaded from UniProt

import pandas as pd
import math

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
def convert_data_labels(filename, data_split=[0.6, 0.2]):
    protein_df = pd.read_csv(filename, sep='\t')
    protein_df['Labels'] = protein_df.apply(construct_labels, axis=1)
    protein_df = protein_df[protein_df['Labels'].str.replace(UNCLASSIFIED, '').str.len() != 0]
    protein_df = protein_df.sample(frac=1)
    train_boundary = int(data_split[0] * len(protein_df))
    dev_boundary = train_boundary + int(data_split[1] * len(protein_df))
    train_data = protein_df.iloc[: train_boundary]
    dev_data = protein_df.iloc[train_boundary: dev_boundary]
    test_data = protein_df.iloc[dev_boundary: ]
    return train_data, dev_data, test_data

