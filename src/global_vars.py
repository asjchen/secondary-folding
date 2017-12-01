# Global Variables for Neural Network and Data Processing

import math

STRAND = 'S'
HELIX = 'H'
TURN = 'T'
UNCLASSIFIED = 'U'
LABEL_SET = [STRAND, HELIX, TURN, UNCLASSIFIED]

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'B', 'C', 'E', 'Q', 'Z', 'G', 'H', 'I', 'L', 'K', 
    'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'U']

WINDOW_SIZE = 17
NUM_AMINO_ACIDS = len(AMINO_ACIDS)

INPUT_DIM = WINDOW_SIZE * NUM_AMINO_ACIDS
HIDDEN_DIM = int(math.sqrt(len(LABEL_SET) * INPUT_DIM))

SEQUENCE_LIMIT = 2000
