import pandas as pd
import numpy as np

#plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

#sklearn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='r', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    for centroid in range(len(centroids)):
        plt.scatter(centroids[centroid, 0], centroids[centroid, 1]+0.1,
                    marker=f'$Group:{centroid}$', s=5000, c="b")
        plt.scatter(centroids[centroid, 0], centroids[centroid, 1], c="r")

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True, \
    colours="PuBuGn"):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12), cmap=colours)
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidths=2, colors='r', linestyles='dashed')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)

    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft=False)

def build_graph(y_scaled):
    df = pd.read_csv('data.csv') #<<<< Data as csv
    df.drop(columns="Competitive gap", inplace=True)

    features = ['Analyst value (0 - 5)', 'Partner value (0 - 5)',
           'Persona value (0 - 5)', 'Growing market',
           'Organic Search Volume', 'SEO Value (0 - 3)']
    df[features] = StandardScaler().fit_transform(df[features])

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(df[features])
    gm_for_pca_2dims = GaussianMixture(n_components=3, n_init=10, \
        random_state=0).fit(reduced)
    gm_for_predicton_6dims = GaussianMixture(n_components=3, n_init=10, \
        random_state=0).fit(df[features])

    plt.figure(figsize=(20, 9))
    plot_gaussian_mixture(gm_for_pca_2dims, reduced, colours="Blues")
    plt.scatter(y_scaled[0][0], y_scaled[0][3], c="r", marker="o", s=1000)
    plt.scatter(y_scaled[0][0], y_scaled[0][3]+0.3, marker='$NEW API$', \
        s=3000, c="r")
    plt.axis("off")
    plt.show()
