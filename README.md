# SNoRe: Scalable Unsupervised Learning of Symbolic Node Representations
This repository contains the implementation of SNoRe algorithm from SNoRe paper
found [here](https://ieeexplore.ieee.org/document/9265235):

```
@ARTICLE{meznar2020snore,
         author={S. {Me\v{z}nar} and N. {Lavra\v{c}} and B. {\v{S}krlj}},
         journal={IEEE Access}, 
         title={SNoRe: Scalable Unsupervised Learning of Symbolic Node Representations},
         year={2020},
         volume={8},
         number={}, 
         pages={212568-212588},
         doi={10.1109/ACCESS.2020.3039541}}
```

An overview of the algorithm is presented in the image below.

![algorithm overview](https://github.com/smeznar/SNoRe/blob/master/images/algorithm_overview.png)

## Installing SNoRe
```
python setup.py install
```

or

```
pip install snore-embedding
```

## Using SNoRe
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
dataset = loadmat("data/cora.mat")
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

Further examples of evaluation and embedding explainability can be found in the examples folder.

## Hyperparameter explanation

SNoRe uses the following hyperparameters and their default values:

| Hyperparameter  | Description                                                                                                                                                                                                                                | Default Value |
|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| dimension       | The number of features if fixed number of features are used, otherwise the number of features that make up space equivalent to &#124;N&#124;*dimensions                                                                                    | 256           |
| num_walks       | The number of random walks for every node                                                                                                                                                                                                  | 1024          |
| max_walk_length | The length of the longest random walk                                                                                                                                                                                                      | 5             |
| inclusion       | Inclusion threshold. Node needs to be encountered with frequency inclusion to appear in the hash representation                                                                                                                            | 0.005         |
| fixed_dimension | If True, fixed number of features are used, otherwise space equivalent to &#124;N&#124;*dimensions is used                                                                                                                                 | False         |
| metric          | Metric used for similarity calculation. Metrics 'cosine','HPI','HDI','euclidean', 'jaccard', 'seuclidean', and 'canberra' can be used when calculating the embedding of fixed dimensions, otherwise 'cosine', 'HPI', and 'HDI' can be used | 'cosine'      |
| num_bins        | Number of bins used in SNoRe SDF to digitize the embedding and reduce it's size. The values are not digitized if None is chosen.                                                                                                           | 256           |

## Results against other baselines

In the above mentioned paper we test SNoRe and it's extension SNoRe SDF against NetMF (SCD), Deepwalk, node2vec,
LINE, PPRS, VGAE, Label Propagation, and the random baseline. The results can be seen on the image below.

![micro f1 results](https://github.com/smeznar/SNoRe/blob/master/images/micro_plot_baseline.png)

By aggregating this results we get scores presented in the table below.

![micro f1 table](https://github.com/smeznar/SNoRe/blob/master/images/f1_table.png)

## Embedding interpretability with SHAP

An advantage of SNoRe is the ability to interpret why instances were predicted the way they were. We can do such
interpretation for a single instance as show in the image below.

![micro f1 table](https://github.com/smeznar/SNoRe/blob/master/images/waterfall.png)

We can also see which features are the most important with the summary plot shown in the image below.

![micro f1 table](https://github.com/smeznar/SNoRe/blob/master/images/Shap_pubmed.png)

To try the interpretation for yourself use code in the example *examples/explainability_example.py*. 

## Latent clustering with UMAP

We can use tools such as UMAP to cluster the embedding we create with SNoRe and see if nodes with similar labels
cluster together. Such clusterings can be seen in the image below.

![micro f1 results](https://github.com/smeznar/SNoRe/blob/master/images/umap.png)

To create such clustering you can start with code in *examples/umap_example.py*.


