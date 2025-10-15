import random as rand
import numpy as np

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids =None

    def fit_predict(self, X):
        random_index = rand.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]
       
        for i in range(self.max_iter):
            cluster_group = self.assign_cluster(X)
            old_cent = self.centroids
            self.centroids = self.move_centrodis(X, cluster_group)
            if (old_cent == self.centroids).all():
                break
        return cluster_group

    def assign_cluster(self, X):
        cluster_group =[]
        dist = []
        for row in X:
            for centroid in self.centroids:
                dist.append(np.sqrt(np.dot(row-centroid, row-centroid)))
            min_dist = min(dist)
            index_pos = dist.index(min_dist)
            cluster_group.append(index_pos)
            dist.clear()
        return np.array(cluster_group)

    def move_centrodis(self,X, cluster_group):
        new_cent = []
        cluster_type = np.unique(cluster_group)
        for type in cluster_type:
            new_cent.append(X[cluster_group == type].mean(axis=0))

        return np.array(new_cent)