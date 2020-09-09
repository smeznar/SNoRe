from snore import SNoRe
from scipy.io import loadmat
from sklearn.utils import shuffle
from catboost import CatBoost
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

# Load adjacency matrix and labels
dataset = loadmat("../data/cora.mat")
network_adj = dataset["network"]
labels = dataset["group"]

# Create the model
model = SNoRe(dimension=256, num_walks=1024, max_walk_length=5,
              inclusion=0.005, fixed_dimension=False, metric="cosine",
              num_bins=256)

# Embed the network
embedding = model.embed(network_adj)

# Train the classifier
nodes = shuffle([i for i in range(network_adj.shape[0])])
train_mask = nodes[:int(network_adj.shape[0]*0.8)]
test_mask = nodes[int(network_adj.shape[0]*0.8):]
classifier = CatBoost(params={'loss_function': 'MultiRMSE', 'iterations': 500})
df = pd.DataFrame.sparse.from_spmatrix(embedding)
classifier.fit(df.iloc[train_mask], labels[train_mask])

# Test prediction
predictions = classifier.predict(df.iloc[test_mask])
print("Micro score:",
      f1_score(np.argmax(labels[test_mask], axis=1),
               np.argmax(predictions, axis=1),
               average='micro'))
