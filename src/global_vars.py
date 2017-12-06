# Global Variables for Neural Network and Data Processing

import math

LABEL_SET = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
AMINO_ACIDS = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']

WINDOW_SIZE = 25
NUM_AMINO_ACIDS = len(AMINO_ACIDS)
NUM_LABELS = len(LABEL_SET)

SEQUENCE_LIMIT = 700
NUM_ORIG_FEATURES = 57

TRAINING_RANGE = (0, 5600)
TEST_RANGE = (5605, 5877)
VALIDATION_RANGE = (5877, 6133)

INDIV_INPUT_DIM = NUM_AMINO_ACIDS 
INPUT_DIM = WINDOW_SIZE * INDIV_INPUT_DIM
HIDDEN_DIM = int(math.sqrt(len(LABEL_SET) * INPUT_DIM)) 
OUTPUT_DIM = len(LABEL_SET)
