import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
np.random.seed(0)

proj_dir = os.path.dirname(__file__)
plot_dir = os.path.join(proj_dir, 'plots')
csv_path = os.path.join(proj_dir, 'annotations.csv')

df = pd.read_csv(csv_path, index_col=0)
width = df['xmax'] - df['xmin']
height = df['ymax'] - df['ymin']
width_n = width / df['width']
height_n = height / df['height']

data = pd.DataFrame({'width': width_n, 'height': height_n, 'class': df['class']})
classes = data['class'].unique()
num_classes = len(classes)
del df

# width-height plot by classes, n = 500 per plot
figure, axes = plt.subplots(nrows=num_classes//2, ncols=num_classes//2 + num_classes%2)
figure.suptitle('width-height ratios by classes, n=500/class', y=0.95, fontsize=12)
for i, ax in enumerate(axes.flatten()):
    if i >= num_classes:
        figure.delaxes(ax)
        continue
    sample = data[data['class'] == classes[i]].sample(n=500, axis=0)
    x, y = sample['width'].values, sample['height'].values
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    ax.scatter(x, y, alpha=1.0, s=4, c=z, cmap='gist_heat')
    ax.set_xlabel('width', fontsize=8)
    ax.set_ylabel('height', fontsize=8)
    ax.set_title(classes[i], fontsize=10)
figure.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.85, wspace=0.45, hspace=0.45)
figure.savefig(os.path.join(plot_dir, 'wh_plot_classes.png'))
plt.clf()

sample = data.sample(n=10000, axis=0)
x, y = sample['width'].values, sample['height'].values
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)
cb = plt.scatter(x, y, alpha=1.0, s=1, c=z, cmap='gist_heat')
plt.colorbar(cb, label='point density')
plt.xlabel('width', fontsize=8)
plt.ylabel('height', fontsize=8)
plt.title('width-height ratios, all classes, n=10,000', fontsize=12, y=1.05)
plt.savefig(os.path.join(plot_dir, 'wh_plot_all.png'))

