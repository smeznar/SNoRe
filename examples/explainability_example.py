import shap
from catboost import CatBoost
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sparse

from snore import SNoRe, from_mat_file


def show_shap_summary(dataset, labels):
    # Create the embedding
    embedder = SNoRe()
    embedding = embedder.embed(dataset)
    node_indexes = embedder.selected_features

    # Create the classification (regression) model
    # We used CatBoost instead of XGBoost (used in the paper), because is simpler to setup since XGBoost has some bugs
    classifier = CatBoost(params={'loss_function': 'MultiRMSE', "iterations": 250})
    if sparse.issparse(embedding):
        df = pd.DataFrame.sparse.from_spmatrix(
            embedding, columns=["node " + str(node) for node in node_indexes])
    else:
        df = pd.DataFrame(
            data=embedding,
            columns=["node " + str(node) for node in node_indexes])
    classifier.fit(df, labels.toarray())

    # Explain the classification (regression) model
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(df)

    for i in range(labels.shape[1]):
        shap_plot = shap.summary_plot(shap_values[i],
                                      df,
                                      show=False,
                                      plot_size=None)
        plt.title("Class" + str(i))
        plt.show()
    shap.summary_plot(shap_values, df, show=False)
    plt.show()


if __name__ == '__main__':
    network, labels, mlp = from_mat_file("../data/cora.mat")
    show_shap_summary(network, labels)
