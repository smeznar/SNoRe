# SNoRe: Scalable Unsupervised Learning of Symbolic Node Representations
This repository contains the implementation of SNoRe algorithm from SNoRe paper
found here:

```
@misc{meznar2020snore,
    title={SNoRe: Scalable Unsupervised Learning of Symbolic Node Representations},
    author={Sebastian Me\v{z}nar and Nada Lavra\v{c} and Bla\v{z} \v{S}krlj},
    year={2020},
    eprint={2009.04535},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

An overview of the algorithm is presented in the image below:

![algorithm overview](https://github.com/smeznar/SNoRe/blob/master/images/algorithm_overview.png)

# Installing SNoRe
```
python setup.py install
```

or

```
pip install snore-embedding
```

# Using SNoRe
A simple use-case is shown below.
First, we import the necessary libraries and load the dataset and its labels.

```
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
```

We then create the SNoRe model and embed the network. 
In code, the default parameters are shown.

```
# Create the model
model = SNoRe(dimension=256, num_walks=1024, max_walk_length=5,
              inclusion=0.005, fixed_dimension=False, metric="cosine",
              num_bins=256)

# Embed the network
embedding = model.embed(network_adj)
```

Finally, we train the classifier and test on the remaining data.

```
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

```

Further examples of evaluation and embedding explainability can be found in the example folder.