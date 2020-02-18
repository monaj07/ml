#
from .base import Algorithm
import numpy as np
from sklearn.cluster import MiniBatchKMeans, MeanShift as MeanShiftSklearn, SpectralClustering as SpectralClusteringSklearn


class Kmeans(Algorithm):
    def __init__(self, data=None, k=None, n_epochs=1000, method="kmeans", order=1):
        super().__init__(data=data, method=method, order=order)
        self.k = k
        self.n_epochs = n_epochs

    def fit_kmeans(self):
        # Initializing the cluster centres (mu)
        x_mins = np.array([self.x[:, i].min() for i in range(self.x.shape[1])]).reshape(1, -1)
        x_maxs = np.array([self.x[:, i].max() for i in range(self.x.shape[1])]).reshape(1, -1)
        x_ranges = x_maxs - x_mins
        self.mu = np.random.rand(self.k, self.x.shape[1]) * x_ranges + x_mins
        mu_old = self.mu.copy()
        idx_old = np.zeros(self.x.shape[0])
        epoch = 0
        for ii in range(self.n_epochs):            
            # Maximization (Finding cluster assignments)
            self.x_resh = self.x.reshape(-1, 1, self.x.shape[1])
            self.mu_resh = self.mu.reshape(-1, 1, self.x.shape[1])
            distance = np.linalg.norm(self.x_resh - self.mu_resh.transpose(1, 0, 2), axis=-1)
            self.idx = np.argmin(distance, axis=1)
            # Expectation (Updating cluster centroids)
            self.mu = np.vstack([np.mean(self.x[self.idx==c, :], axis=0) for c in range(self.k)])
            epoch += 1
            if np.array_equal(mu_old, self.mu) or np.array_equal(idx_old, self.idx):
                break
        print(f"Kmeans Finished in {epoch} epochs.")
    
    def fit_kmeans_sklearn(self):
        self.model = MiniBatchKMeans(n_clusters=self.k, init="k-means++", max_iter=self.n_epochs, batch_size=100)
        self.model.fit(self.x)
        self.idx = self.model.labels_
                          

class MeanShift(Algorithm):
    def __init__(self, data=None, order=1, method="meanshift", radius=0.1, n_epochs=1000):
        super().__init__(data=data, method=method, order=order)
        self.n_epochs = n_epochs
        self.radius = radius

    def fit_meanshift(self):
        # Initialize centroids to be on every single data point.
        self.mu = self.x.copy()
        self.x_resh = self.x.reshape(-1, 1, self.x.shape[1])
        for ii in range(self.n_epochs):

            self.mu_resh = self.mu.reshape(-1, 1, self.x.shape[1])
            distance = np.linalg.norm(self.x_resh - self.mu_resh.transpose(1, 0, 2), axis=-1)
            assignments = [np.where(distance[:, i] < self.radius)[0] for i in range(self.mu.shape[0])]
            assignments_unique = list({tuple(assignment) for assignment in assignments})
            self.mu = np.vstack([self.x[assignment, :].mean(axis=0) for assignment in assignments_unique])
        self.idx = np.zeros(self.x.shape[0])
        self.mu_resh = self.mu.reshape(-1, 1, self.x.shape[1])
        distance = np.linalg.norm(self.x_resh - self.mu_resh.transpose(1, 0, 2), axis=-1)
        labels = np.argmin(distance, 1)
        labels_unique = np.unique(labels)
        labels_refined = np.arange(labels_unique.size)
        temp = labels.copy()
        for id, label in enumerate(labels_unique):
            temp[labels == label] = labels_refined[id]
        self.idx = temp

    def fit_meanshift_sklearn(self):
        self.model = MeanShiftSklearn(bandwidth=self.radius)
        self.model.fit(self.x)
        labels_ = self.model.labels_
        self.idx = labels_
        

class SpectralClustering(Algorithm):
    def __init__(self, data=None, order=1, method="spectral", k=None):
        super().__init__(data=data, method=method, order=order)
        self.k = k

    def fit_spectral_sklearn(self):
        self.model = SpectralClusteringSklearn(n_clusters=self.k, gamma=10)
        self.model.fit(self.x)
        self.idx = self.model.labels_