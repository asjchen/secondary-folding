# Protein Secondary Structure Prediction with Deep Learning

This is a deep learning architecture to predict secondary structure in 
proteins. The dataset, originally from the Protein Data Bank (PDB), contains 
amino acid sequence and structure information for roughly 6100 proteins. 

Currently, we use a Bidirectional LSTM RNN architecture to solve the Q8
classification problem; for each amino acid in the protein sequences, we
assign one of eight different labels:

* alpha helix
* beta strand
* loop or irregular
* beta turn
* bend
* 3<sub>10</sub>-helix
* beta bridge
* pi helix

The neural network architecture yields a roughly 56% Q8 accuracy in testing.


## Python Requirements

The required Python modules are in requirements.txt. You can install them with
```
pip install -r requirements.txt
```

## Running the Scripts

To run the current version of the algorithm, you can run the following command:
```
python src/driver.py data/cullpdb+profile_6133.npy [-c]
```
where the last argument is the location of the dataset. We use the publicly 
available dataset from Zhou, J. & Troyanskaya O. 2014. Currently, this data
is accessible here: <http://www.princeton.edu/~jzthree/datasets/ICML2014/>
Use the [-c] to see the 8x8 confusion matrix of the labels on the validation
data after each epoch of training.
