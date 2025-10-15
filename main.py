from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from knn import KMeans

centorids = [(-5,-5), (5,5), (1,1), (-5,6)]
cluster_std=[1,1,1,1]

X,y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centorids, n_features=2, random_state=2)

clusters =4
km =KMeans(clusters,200)
y_means = km.fit_predict(X)

colors= ['red', 'green', 'blue', 'yellow', 'pink']

for i in range(clusters):
    plt.scatter(X[y_means == i,0], X[y_means == i,1], color=colors[i])

   
plt.show()