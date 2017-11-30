# Global Variables for Neural Network and Data Processing

STRAND = 'S'
HELIX = 'H'
TURN = 'T'
UNCLASSIFIED = 'U'
label_set = [STRAND, HELIX, TURN, UNCLASSIFIED]

AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 
    'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

WINDOW_SIZE = 17
NUM_AMINO_ACIDS = len(AMINO_ACIDS)

