import csv
import numpy as np
from collections import defaultdict

columns = defaultdict(list) # each value in each column is appended to a list

with open('mushrooms.csv') as f:
    reader = csv.DictReader(f) # read rows into a dictionary format
    for row in reader: # read a row as {column1: value1, column2: value2,...}
        for (k,v) in row.items(): # go over each column name and value
            columns[k].append(v) # append the value into the appropriate list
                                 # based on column name k

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

# features is actually a bit of a misnomer as the first column is actually the target value
np.random.shuffle( features )
split_ratio = 0.7
split_idx = int(features.shape[0] * split_ratio)
np.savez("mushroom_train.npz", features[:split_idx,1:], features[:split_idx,0].reshape((-1,1)),
                               features[split_idx:,1:], features[split_idx:,0].reshape((-1,1)))

