# Protein Secondary Structure Prediction with Neural Networks

This is a brief survey of neural network methods to predict secondary 
structure in proteins. The dataset, originally from the PDB database, 
contains amino acid sequence and structure information for roughly 6100 
proteins. Currently, we use a Bidirectional LSTM RNN architecture.

## Python Requirements

The required Python modules are in requirements.txt. You can install them with
```
pip install -r requirements.txt
```

## Running the Scripts

To run the current version of the algorithm, you can run the following command:
```
python src/driver.py data/cullpdb+profile_6133.npy
```
where the last argument is the location of the dataset. We use the publicly 
available dataset from Zhou, J. & Troyanskaya O. 2014. Currently, this data
is accessible here: <http://www.princeton.edu/~jzthree/datasets/ICML2014/>
