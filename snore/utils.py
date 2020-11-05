from scipy.io import loadmat
import scipy.sparse as sparse
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def from_mat_file(path):
    try:
        mat = loadmat(path)
    except Exception:
        print(path, " is not a .mat file")
        return None
    network = sparse.csr_matrix(mat['network'])  # Network
    labels = mat['group']  # Labels
    labels_matrix = sparse.csr_matrix(labels)
    mlb = MultiLabelBinarizer(range(labels_matrix.shape[1]))
    return network, labels_matrix, mlb


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels
