# Protein Secondary Structure Prediction with Neural Networks

This is a brief survey of neural network methods to predict secondary 
structure in proteins. The dataset, originally from the PDB database, 
contains amino acid sequence and structure information for roughly 6100 proteins.

Currently, we use a Bidirection LSTM Encoder-Decoder.

## Python Requirements

The required Python modules are in requirements.txt. You can install them with
```
pip install -r requirements.txt
```

TODO: update requirements.txt with version numbers


## Running the Scripts

To run the current version of the algorithm, you can run the following command:
```
python src/driver.py data/cullpdb+profile_6133.npy
```

