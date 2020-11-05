from numba import jit, prange
import numpy as np
from scipy.io import loadmat
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from collections import Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


@jit(parallel=True, nogil=True, nopython=True)
def numba_walk_kernel(walk_matrix,
                      node_name,
                      sparse_pointers,
                      sparse_neighbors,
                      num_steps=3,
                      num_walks=100):
    """ Create num_walks random walks of length num_steps starting at node node_name.
    Walks are saved to the walk_matrix.

    Args:
        walk_matrix (ndarray): Numpy array where random walks will be saved.
        node_name (int): Starting node.
        sparse_pointers (ndarray): Pointers from a sparse matrix denoting the network.
        sparse_neighbors (ndarray): Indices from a sparse matrix denoting the network.
        num_steps (int): Random walk length.
        num_walks (int): Number of random walks to create.
    """
    length = num_steps + 1
    for walk in prange(num_walks):
        curr = node_name
        offset = walk * length
        walk_matrix[offset] = node_name
        for step in prange(num_steps):
            num_neighs = sparse_pointers[curr + 1] - sparse_pointers[curr]
            if num_neighs > 0:
                curr = sparse_neighbors[sparse_pointers[curr] +
                                        np.random.randint(num_neighs)]
            idx = offset + step + 1
            walk_matrix[idx] = curr


class SNoRe:
    """ Python class to embed a network using SNoRe algorithm.

    Args:
        dimension (int): Dimension of the embedding. Look at fixed_dimension parameter for more detail about the
                         embedding matrix shape
        num_walks (int): Number of random walks from each node.
        max_walk_length (int): The maximum length of random walks
        inclusion (float): At least how frequently a node has to appear in random walks to be included
        fixed_dimension (bool): If fixed_dimension is False the embedding uses at most (|N| * dimension) float values
                                else the shape of embedding matrix is |N| x dimension
        metric (str): If fixed_dimension is True this parameter dictates which distance metric is chosen to calculate
                      similarity between two hashed walks
        num_bins (int): If fixed_dimension is False, how similarity score is ... .
    """
    def __init__(self,
                 dimension=256,
                 num_walks=1024,
                 max_walk_length=5,
                 inclusion=0.005,
                 fixed_dimension=False,
                 metric="cosine",
                 num_bins=256):
        self.dimension = dimension
        self.num_walks = num_walks
        self.max_walk_length = max_walk_length
        self.inclusion = inclusion
        self.num_bins = num_bins
        self.distribution = self.generate_walk_distribution()
        self.fixed_dimension = fixed_dimension
        self.metric = metric
        self.selected_features = []

    def generate_walk_distribution(self):
        """ Generates a vector with number of walks per walk length. We create this in advance so all nodes make same
        number of steps.

        Returns:
            ndarray: Numpy array of walks per walk length.
        """
        samples = np.random.uniform(0, 1, self.num_walks)
        return np.histogram(samples, self.max_walk_length)[
            0]  # We use histogram function to sort samples into
        # bins that represent walk lengths

    def embed(self, network):
        """ Create an embedding of the network

        Args:
            network (scipy sparse matrix): Sparse network adjacency matrix .

        Returns:
            scipy sparse matrix: Symbolic node embedding.
        """
        logging.info("Generating and hashing random walks")
        hashes = self.generate_walk_hashes(network)

        # Rank nodes
        pagerank_scores = nx.pagerank_scipy(
            nx.from_scipy_sparse_matrix(network))
        ranked_features = np.argsort(
            [pagerank_scores[i] for i in range(len(pagerank_scores))])[::-1]

        logging.info("Generating similarity matrix")
        if self.fixed_dimension:
            embedding = self.generate_similarity_fixed(
                hashes,
                ranked_features[:min(self.dimension, network.shape[0])])
        else:
            embedding = self.generate_similarity_matrix(
                hashes, ranked_features).tocsr()

        # Check if embedding size is less then tau
        assert (not sparse.issparse(embedding)) or len(
            embedding.data) <= (self.dimension * network.shape[0])

        logging.info("Embedding done")
        return embedding

    def generate_walk_hashes(self, network):
        """ Generate random walks and hash them.

        Args:
            network (scipy sparse matrix): Sparse network adjacency matrix .

        Returns:
            scipy sparse matrix: Sparse matrix of hashed random walks.
        """
        if network.shape[0] < 2**16:
            dtype = np.uint16
        else:
            dtype = np.uint32
        sparse_pointers = network.indptr
        sparse_neighbors = network.indices
        hashes = []
        for i in range(network.shape[0]):
            generated_walks = []
            # Generate walks
            for j, num in enumerate(self.distribution):
                walk_matrix = -np.ones((num, (j + 2)), dtype=dtype, order='C')
                walk_matrix = np.reshape(walk_matrix, (walk_matrix.size, ),
                                         order='C')
                numba_walk_kernel(walk_matrix,
                                  i,
                                  sparse_pointers,
                                  sparse_neighbors,
                                  num_steps=j + 1,
                                  num_walks=num)
                generated_walks += walk_matrix.tolist()
            # Count occurrences of nodes in random walks
            score_hash = Counter(generated_walks)
            # We only include node that are visited with frequency at least self.inclusion
            thresh = len(generated_walks) * self.inclusion
            rows = []
            cols = []
            vals = []
            # Remove nodes not visited often enough and create the hash value
            for node, occurances in score_hash.most_common():
                if occurances > thresh:
                    rows.append(0)
                    cols.append(node)
                    vals.append(occurances)
            hashes.append(
                sparse.coo_matrix((vals, (rows, cols)),
                                  shape=(1, network.shape[1])))
        # Stack hashes to make the hash matrix and normalize them
        return normalize(sparse.vstack(hashes), "l1")

    def generate_similarity_matrix(self, hashes, ranked_features):
        """ Incrementally generate similarity matrix by adding a column of similarities to i-th most important node.

        Args:
            hashes (scipy sparse matrix): Sparse matrix of random walk hashes.
            ranked_features (ndarray): Numpy array of ranked network nodes.

        Returns:
            scipy sparse matrix: Sparse similarity matrix.
        """
        # Create a sparse matrix with tau floating point values (16 bit). tau = d * |N|
        tau = self.dimension * hashes.shape[0]
        columns = []
        bins = np.linspace(start=0, stop=1, num=(self.num_bins if self.num_bins is not None else 1))
        for i in ranked_features:
            # "Round" similarity values so they need less space allowing more features to be chosen
            if self.metric == "cosine":
                similarity_column = cosine_similarity(hashes, hashes[i, :], dense_output=False)
            elif self.metric == "HPI":
                similarity_column = SNoRe.HPI(hashes, [i])
            else:
                # HDI
                similarity_column = SNoRe.HDI(hashes, [i])
            if self.num_bins is not None:
                feature = (np.digitize(similarity_column.toarray(),
                                       bins=bins) - 1) / (self.num_bins - 1)
                feature = sparse.coo_matrix(feature, dtype=np.half)
            else:
                feature = similarity_column
            # Reduce tau by number of nonzero similarities
            tau -= len(feature.data)
            if tau >= 0:
                columns.append(feature)
                self.selected_features.append(i)
            else:
                break

        return sparse.hstack(columns)

    @staticmethod
    def HPI(hashes, features):
        nonzero_items = Counter(hashes.nonzero()[0])
        y = []
        x = []
        val = []
        for i, f in enumerate(features):
            a = hashes.multiply(hashes[f, :])
            b = Counter(a.nonzero()[0])
            for k in b.keys():
                x.append(i)
                y.append(k)
                val.append(b[k]/min(nonzero_items[f], nonzero_items[k]))
        return sparse.csc_matrix((val, (y, x)), shape=(hashes.shape[0], len(features)))

    @staticmethod
    def HDI(hashes, features):
        nonzero_items = Counter(hashes.nonzero()[0])
        y = []
        x = []
        val = []
        for i, f in enumerate(features):
            a = hashes.multiply(hashes[f, :])
            b = Counter(a.nonzero()[0])
            for k in b.keys():
                x.append(i)
                y.append(k)
                val.append(b[k]/max(nonzero_items[f], nonzero_items[k]))
        return sparse.csc_matrix((val, (y, x)), shape=(hashes.shape[0], len(features)))

    def generate_similarity_fixed(self, hashes, features):
        """ Generates a similarity matrix of fixed dimension

        Args:
            hashes (scipy sparse matrix): Sparse matrix of random walk hashes.
            features (ndarray): Numpy array of ranked network nodes.

        Returns:
            matrix: similarity matrix.
        """
        self.selected_features = features

        # Jaccard, Canberra and Standardized Euclidean don't take sparse matrices as input.
        # Only Cosine Similarity returns a sparse representation
        if self.metric == "jaccard":
            return sparse.csr_matrix(
                1 - pairwise_distances(hashes.toarray(),
                                       hashes[features, :].toarray(),
                                       metric="jaccard"))
        elif self.metric == "euclidean":
            return pairwise_distances(hashes,
                                      hashes[features, :],
                                      metric="euclidean")
        elif self.metric == "seuclidean":
            return pairwise_distances(hashes.toarray(),
                                      hashes[features, :].toarray(),
                                      metric="seuclidean")
        elif self.metric == "canberra":
            return pairwise_distances(hashes.toarray(),
                                      hashes[features, :].toarray(),
                                      metric="canberra")
        elif self.metric == "HPI":
            return SNoRe.HPI(hashes, features)
        elif self.metric == "HDI":
            return SNoRe.HDI(hashes, features)
        else:
            return cosine_similarity(hashes,
                                     hashes[features, :],
                                     dense_output=False)


if __name__ == '__main__':
    network_adj = loadmat("../data/cora.mat")["network"]
    embedder = SNoRe(metric="HPI")
    emb = embedder.embed(network_adj)
    print(emb.shape)
