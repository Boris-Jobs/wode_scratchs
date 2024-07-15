# -*- coding: utf-8 -*-
"""
Created on 2024-04-13 19:34:42

@author: Boris Jobs, Chairman of FrameX Inc.

Our mission is as same as xAi's, 'Understand the Universe'.

AI that benefits all humanity is all you need.
"""

print('This is the first class in machine learning Hands-On')

import sys

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pathlib import Path
from packaging import version
import sklearn


assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
assert sys.version_info >= (3, 7)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Qt5Agg')

# extra code – just formatting details
from matplotlib.colors import ListedColormap


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


if __name__ == '__main__':

    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    IMAGES_PATH = Path() / "images" / "decision_trees"
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    iris = load_iris(as_frame=True)
    X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y_iris = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X_iris, y_iris)

    export_graphviz(
        tree_clf,
        out_file=str(IMAGES_PATH / "iris_tree.dot"),  # path differs in the book
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

    # tree_clf.tree_

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.figure(figsize=(8, 4))

    lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))
    X_iris_all = np.c_[lengths.ravel(), widths.ravel()]
    y_pred = tree_clf.predict(X_iris_all).reshape(lengths.shape)
    plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)
    for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
        plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],
                 style, label=f"Iris {name}")

    # extra code – this section beautifies and saves Figure 6–2
    tree_clf_deeper = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf_deeper.fit(X_iris, y_iris)
    th0, th1, th2a, th2b = tree_clf_deeper.tree_.threshold[[0, 2, 3, 6]]
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.plot([th0, th0], [0, 3], "k-", linewidth=2)
    plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)
    plt.plot([th2a, th2a], [0, th1], "k:", linewidth=2)
    plt.plot([th2b, th2b], [th1, 3], "k:", linewidth=2)
    plt.text(th0 - 0.05, 1.0, "Depth=0", horizontalalignment="right", fontsize=15)
    plt.text(3.2, th1 + 0.02, "Depth=1", verticalalignment="bottom", fontsize=13)
    plt.text(th2a + 0.05, 0.5, "(Depth=2)", fontsize=11)
    plt.axis([0, 7.2, 0, 3])
    plt.legend()
    save_fig("decision_tree_decision_boundaries_plot")

    plt.show()