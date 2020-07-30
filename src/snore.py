from numba import jit, prange
import numpy as np
from scipy.io import loadmat
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from collections import Counter
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import time


@jit(parallel=True, nogil=True, nopython=True)
def numba_walk_kernel(walk_matrix, node_name, sparse_pointers, sparse_neighbors, num_steps=3, num_walks=100):
    length = num_steps + 1
    for walk in prange(num_walks):
        curr = node_name
        offset = walk * length
        walk_matrix[offset] = node_name
        for step in prange(num_steps):
            num_neighs = sparse_pointers[curr+1] - sparse_pointers[curr]
            if num_neighs > 0:
                curr = sparse_neighbors[sparse_pointers[curr] + np.random.randint(num_neighs)]
            idx = offset+step+1
            walk_matrix[idx] = curr


class SNoRe:
    def __init__(self, dense_dimension=128, num_walks=1024, max_walk_length=5, inclusion=0.005):
        self.dense_dimension = dense_dimension
        self.num_walks = num_walks
        self.max_walk_length = max_walk_length
        self.inclusion = inclusion
        self.distribution = self.generate_walk_distribution()
        self.selected_features = []

    def generate_walk_distribution(self):
        samples = np.random.uniform(0, 1, self.num_walks)
        return np.histogram(samples, self.max_walk_length)[0]

    def embed(self, network):
        print("Generating and hashing random walks")
        hashes = self.generate_walk_hashes(network)

        pagerank_scores = nx.pagerank_scipy(nx.from_scipy_sparse_matrix(network))
        ranked_features = np.argsort([pagerank_scores[i] for i in range(len(pagerank_scores))])[::-1]

        print("Generating similarity matrix")
        return self.generate_similarity_matrix(hashes, ranked_features)

    def generate_walk_hashes(self, network):
        sparse_pointers = network.indptr
        sparse_neighbors = network.indices
        hashes = []
        for i in range(network.shape[0]):
            generated_walks = []
            for j, num in enumerate(self.distribution):
                walk_matrix = -np.ones((num, (j + 2)), dtype=np.int32, order='C')
                walk_matrix = np.reshape(walk_matrix, (walk_matrix.size,), order='C')
                numba_walk_kernel(walk_matrix, i, sparse_pointers, sparse_neighbors, num_steps=j+1, num_walks=num)
                generated_walks += walk_matrix.tolist()
            score_hash = Counter(generated_walks)
            thresh = len(generated_walks)*self.inclusion
            rows = []
            cols = []
            vals = []
            for node, occurances in score_hash.most_common():
                if occurances > thresh:
                    rows.append(0)
                    cols.append(node)
                    vals.append(occurances)
            hashes.append(sparse.coo_matrix((vals, (rows, cols)), shape=(1, network.shape[1])))
        return normalize(sparse.vstack(hashes), "l1")

    def generate_similarity_matrix(self, hashes, ranked_features):
        tau = self.dense_dimension * hashes.shape[0]
        columns = []
        for i in ranked_features:
            feature = cosine_similarity(hashes, hashes[i, :], dense_output=False)
            tau -= len(feature.data)
            if tau > 0:
                columns.append(feature)
                self.selected_features.append(i)
            elif tau < 0:
                break
            else:
                columns.append(feature)
                self.selected_features.append(i)
                break
        return sparse.hstack(columns)


if __name__ == '__main__':
    network = loadmat("dataset path")["network"]
    embedder = SNoRe()
    stime = time.time()
    emb = embedder.embed(network)
    print(emb.shape)
    print("done", time.time()-stime)
