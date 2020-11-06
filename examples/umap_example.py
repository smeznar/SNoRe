import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from snore import from_mat_file, SNoRe


def cluster_embedding(emb, labels):
    # Select nodes with one label for easier vizualization
    emb = emb[np.sum(labels, axis=1) == 1]
    labels = labels[np.sum(labels, axis=1) == 1]
    classes = np.where(labels == 1)[1]

    # Transform embedding to UMAP coordinates
    umaped = umap.UMAP().fit_transform(emb.toarray())
    df = pd.DataFrame(data={"x": umaped[:, 0], "y": umaped[:, 1], "Class": classes})
    df["Class"] = df["Class"].astype("category")

    # Plot points of the transformed nodes
    sp = sns.scatterplot(x="x", y="y", hue="Class", data=df, linewidth=0, legend=False)
    plt.show()


if __name__ == '__main__':
    sns.set(palette=sns.color_palette("Set2"), style='white')

    # Read the network and labels
    network, labels, mlb = from_mat_file("../data/cora.mat")
    labels = labels.toarray()

    # Create the embedding using SNoRe
    emb = SNoRe().embed(network)

    # Cluster the embedding
    cluster_embedding(emb, labels)
