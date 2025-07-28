import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hclust
import statsmodels.graphics.mosaicplot as smosaic

# Consistent color palette
_COLORS = ['y', 'r', 'b', 'g', 'c', 'm', 'sienna', 'coral',
           'darkblue', 'lime', 'grey',
           'tomato', 'indigo', 'teal', 'orange', 'darkgreen']

# Plotting clusters based on x, y coordinates and group labels
def plot_clusters(x, y, g, groups, labels=None, title="Plot clusters"):
    g_ = np.array(g)
    f = plt.figure(figsize=(12, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    noOfGroups = len(_COLORS)
    for v in groups:
        x_ = x[g_ == v]
        y_ = y[g_ == v]
        k = int(v[1:])
        if len(x_) == 1:
            ax.scatter(x_, y_, color='k', label=v)
        else:
            ax.scatter(x_, y_, color=_COLORS[k % noOfGroups], label=v)
    ax.legend()
    if labels is not None:
        for i in range(len(labels)):
            ax.text(x[i], y[i], labels[i])

# Plotting histograms for each group
def histograms(x, g, var):
    groups = set(g)
    g_ = np.array(g)
    m = len(groups)
    l = int(np.ceil(np.sqrt(m)))
    c = (m + l - 1) // l
    axes = []
    f = plt.figure(figsize=(12, 7))
    for i in range(1, m + 1):
        ax = f.add_subplot(l, c, i)
        axes.append(ax)
        ax.set_xlabel(var, fontsize=12, color='k')
    for v, ax in zip(groups, axes):
        y = x[g_ == v]
        ax.hist(y, bins=10, label=v, rwidth=0.9, range=(min(x), max(x)))
        ax.legend()

# Plot dendrogram
def dendrogram(h, labels=None, title='Hierarchical classification',
               threshold=None, colors=None):
    f = plt.figure(figsize=(12, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    if colors is None:
        hclust.dendrogram(h, labels=labels, leaf_rotation=30, ax=ax, color_threshold=threshold)
    else:
        hclust.dendrogram(h, labels=labels, leaf_rotation=30, ax=ax,
                          link_color_func=lambda k: colors[k])
    if threshold is not None:
        plt.axhline(y=threshold, color='r')

#Correlogram
def correlogram(matrix=None, dec=2, title='Correlogram', valmin=-1, valmax=1):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), vmin=valmin, vmax=valmax, cmap='bwr', annot=True)

#Link intensity as a heatmap
def linkIntensity(matrix=None, dec=2, title='Link Intensity'):
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')
    sb.heatmap(data=np.round(matrix, dec), cmap='Oranges', annot=True)

#Correlation circle
def correlCircle(matrix=None, V1=0, V2=1, dec=1,
                 XLabel=None, YLabel=None, title='Correlation Circle'):
    plt.figure(title, figsize=(8, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    T = [t for t in np.arange(0, np.pi * 2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y)
    plt.axhline(y=0, color='g')
    plt.axvline(x=0, color='g')
    if XLabel is None or YLabel is None:
        if isinstance(matrix, pd.DataFrame):
            plt.xlabel(matrix.columns[V1], fontsize=14, color='k', verticalalignment='top')
            plt.ylabel(matrix.columns[V2], fontsize=14, color='k', verticalalignment='bottom')
        else:
            plt.xlabel(f'Var {V1+1}', fontsize=14, color='k', verticalalignment='top')
            plt.ylabel(f'Var {V2+1}', fontsize=14, color='k', verticalalignment='bottom')
    else:
        plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
        plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')

    if isinstance(matrix, np.ndarray):
        plt.scatter(x=matrix[:, V1], y=matrix[:, V2], c='r')
        for i in range(matrix.shape[0]):
            plt.text(x=matrix[i, V1], y=matrix[i, V2],
                     s=f'({np.round(matrix[i, V1], dec)}, {np.round(matrix[i, V2], dec)})')

    if isinstance(matrix, pd.DataFrame):
        plt.scatter(x=matrix.iloc[:, V1], y=matrix.iloc[:, V2], c='b')
        for i in range(matrix.shape[0]):
            plt.text(x=matrix.iloc[i, V1], y=matrix.iloc[i, V2],
                     s=matrix.index[i])

#Eigenvalues for principal components
def principalComponents(eigenvalues=None, XLabel='Principal components', YLabel='Eigenvalues (variance)',
                        title='Explained variance by the principal components'):
    plt.figure(title, figsize=(13, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')
    components = [f'C{j+1}' for j in range(len(eigenvalues))]
    plt.plot(components, eigenvalues, 'bo-')
    plt.axhline(y=1, color='r')
def show():
    plt.show()
