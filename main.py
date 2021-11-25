import numpy as np
import pandas as pd
from scipy.stats.stats import mode
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
np.random.seed(0)

proj_dir = os.path.dirname(__file__)
plot_dir = os.path.join(proj_dir, 'plots')
anchor_dir = os.path.join(proj_dir, 'anchors')
csv_path = os.path.join(proj_dir, 'annotations.csv')
num_anchors = [5, 6, 9]


def save_anchors(model: KMeans):
    n_clusters = model.n_clusters
    centroids = model.cluster_centers_

    path = os.path.join(anchor_dir, f'anchors_k_{n_clusters}.txt')
    with open(path, 'w') as fil:
        for anchor in centroids:
            fil.write(str(anchor[0]) + ', ' + str(anchor[1]) + '\n')

# creates a map showing the boundaries for each cluster
# marks the centroids of the clusters as well
def create_voronoi_diagram(model: KMeans):
    h = 0.001
    x_min, x_max = (0,1)
    y_min, y_max = (0,1)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.imshow(
        z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )
    n_clusters = model.n_clusters
    centroids = model.cluster_centers_
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="k",
        zorder=10,
    )
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title(f'Voronoi Diagram for K-Means Model, K={n_clusters}')
    path = f'voronoi_diagram_k_{n_clusters}.png'
    plt.savefig(os.path.join(plot_dir, path))
    plt.clf()


# takes in width and height arrays and assigns them labels
# then plots them accordingly 
def create_clustered_plot(model: KMeans, w, h):
    X = np.stack((w,h)).transpose()
    labels = model.predict(X)

    n_clusters = model.n_clusters
    centroids = model.cluster_centers_

    plt.scatter(w, h, c=labels, cmap='tab10', s=20, alpha=0.9)

    plt.scatter(
        centroids[:, 0], 
        centroids[:, 1], 
        c='yellow',
        marker='*',
        edgecolor='k',
        s=144
    )
    plt.xlabel("width")
    plt.ylabel("height")
    plt.title(f"K-Means Clustering Example, K={n_clusters}")
    path = f'clustered_plot_k_{n_clusters}.png'
    plt.savefig(os.path.join(plot_dir, path))
    plt.clf()


def main():
    df = pd.read_csv(csv_path, index_col=0)
    width = df['xmax'] - df['xmin']
    height = df['ymax'] - df['ymin']
    width_n = width / df['width']
    height_n = height / df['height']
    x = width_n.values
    y = height_n.values
    X = np.stack((x,y)).transpose()

    for k in num_anchors:
        model = KMeans(n_clusters=k, init='k-means++', algorithm="auto")
        model.fit(X)
        save_anchors(model)
        create_voronoi_diagram(model)
        create_clustered_plot(model, x[:1000], y[:1000])


if __name__ == '__main__':
    main()