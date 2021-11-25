import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
np.random.seed(0)

proj_dir = os.path.dirname(__file__)
plot_dir = os.path.join(proj_dir, 'plots')
csv_path = os.path.join(proj_dir, 'annotations.csv')

df = pd.read_csv(csv_path, index_col=0)
width = df['xmax'] - df['xmin']
height = df['ymax'] - df['ymin']
width_n = width / df['width']
height_n = height / df['height']
x = width_n.values
y = height_n.values
X = np.stack((x,y)).transpose()

# may take a while to execute, could lower number of samples
n_clusters = np.arange(2,16)
inertia = np.empty(n_clusters.shape)
for i,k in enumerate(n_clusters):
    model = KMeans(n_clusters=k, init='k-means++')
    model.fit(X)
    inertia[i] = model.inertia_

plt.plot(n_clusters, inertia, marker="X")
plt.title("Inertia vs. K Anchors")
plt.xlabel("Value of K")
plt.ylabel("Sum of Squared Distances (inertia)")
plt.savefig(os.path.join(plot_dir, 'elbow_plot.png'))