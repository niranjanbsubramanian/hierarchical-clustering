import numpy as np
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.datasets import make_blobs

X, Y = make_blobs(n_samples=250, n_features=7, centers=5, cluster_std=1.0, random_state=47)
plt.scatter(X[:,0],X[:,1])
plt.show()

Z = linkage(X, method='ward')

dendrogram(Z, truncate_mode='lastp',p=20)
plt.axhline(y=50, linestyle='--', color='black')
plt.show()

print(Z[-10:, 2])

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
hc.fit_predict(X)
#print(hc.labels_)

plt.scatter(X[:,0], X[:,1],c=hc.labels_)
plt.show()
