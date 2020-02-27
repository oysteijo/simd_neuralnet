# A working example

This directory contains code that will serve as an example of how to get started
with this software and explain some of the concepts. 

## A mushroom classifier

At the UCI Machine Learning repository there was a mushroom dataset posted.
The dataset was donated to UCI ML in April back in 1987, but will serve as a good
first example for use of this software.

The dataset contains 8124 samples of different mushrooms where each sample is
given 23 different categorical features. One of the categorical features is
whether the mushroom is edible or poisonous, and in this example we will build
a simple classifier from the other features. The data is provided as a `.csv`
file in this directory.

### Mushroom feature engineering

This software, `simd_neuralnet` is a neural network library. It does not to any
feature engineering or other machine learning tricks. We will therefore use python
and numpy to do simple feature engineering (just one hot encoding). We will also
randomize the order of the samples and then split into a train partition and a
test partition.

Let's start python:

    # First we import the libraries we are using
    import csv
    import numpy as np
    from collections import defaultdict
    
    columns = defaultdict(list) # each value in each column is appended to a list
    
    # Read the file into a dictionary where each column is tehe key and the value
    # is a list of all elements in the column.
    with open('mushrooms.csv') as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                columns[k].append(v) # append the value into the appropriate list
                                     # based on column name k
    
    # This is code does one-hot-encoding of all the features. It finds the number of unique
    # elements in each column and makes a numpy array of one hot encoded features for each.
    # At the end of each loop iterations the numpy array is concatenated to the other
    # previous features form previous iterations. 
    # It there is only one unique value in a column, that column will be discarded in the
    # features. If there are only two unique values in a column, it will only produce one
    # feature column since another column will be complementary to the other.
    
    features = None
    for c in list(columns.keys()):
        u, e =  np.unique(columns[c], return_inverse=True)
        if len(u) == 1:
            continue
        if len(u) == 2:
            f = np.array(e, dtype=np.float32).reshape((-1,1))
        else:
            n_values = np.max(e) + 1
            f = np.eye(n_values)[e].astype(np.float32)
        print( "{:25s}".format(c), f.shape )
        if features is None:
            features = f
        else:
            features = np.hstack((features, f ))
    
    # "features" is actually a bit of a misnomer as the first column is actually the target value.
    # We will also use python to split the dataset. First we randomize the order, and then we split
    # into train and test partitions.
    
    np.random.shuffle( features )
    split_ratio = 0.7
    split_idx = int(features.shape[0] * split_ratio)
    
    # We save everything into four numpy arrays in the `.npz` format (which is actually zip).
    # - train_features
    # - train_targets
    # - test_features
    # - test_targets
    
    np.savez("mushroom_train.npz", features[:split_idx,1:], features[:split_idx,0].reshape((-1,1)),
                                   features[split_idx:,1:], features[split_idx:,0].reshape((-1,1)))
    
The code above is also available in `mushroom_to_numpy.py`.

### Building a simd_neuralnet from scratch in ANSI C.

(in progress)

### Compiling and training
(in progress)

### Using optimizers and callbacks.
(in progress)

(Discuss)
## Other examples


(Work in progress)
