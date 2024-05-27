"""PCA and KMeans Clustering"""
import os
from typing import Union, Callable
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


def euclidean(samples, centers, squared=True):
    '''Calculate the pointwise distance.

    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        squared: boolean. 
    Returns:
        pointwise distances (n_sample, n_center).
    '''
    samples_norm = np.sum(samples**2, axis=1, keepdims=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = np.sum(centers**2, axis=1, keepdims=True)
    centers_norm.reshape(1, -1)
    distances = samples@centers.T
    distances *= -2
    distances += samples_norm
    distances += centers_norm
    if not squared:
        distances = np.sqrt(distances)
    return distances


def gaussian(samples, centers, gamma) -> np.ndarray:
    '''Gaussian kernel.
    ---
    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth. 即高斯核的标准差。
        gamma: 1/(2*bandwidth^2)

    Returns:
        kernel matrix of shape (n_sample, n_center).
    '''
    assert gamma > 0
    kernel_mat = np.exp(-gamma*euclidean(samples, centers, squared=True))
    return kernel_mat


class PCA:
    def __init__(self, n_components: int = 2,
                 kernel: str = "linear",
                 **kwargs
                 ) -> None:
        self.n_components = n_components  # d'
        self.kernel = kernel
        self.eignvalues = None  # [n_components, ]
        self.eignvectors = None  # [n_features, n_components]
        # ...

    def get_kernel_mat(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self.kernel is None:
            return
        if self.kernel == "rbf":
            if "gamma" not in kwargs:
                d = X.shape[1]  # n_features
                gamma = 2/d
            else:
                gamma = kwargs["gamma"]
            return gaussian(X.T, X.T, gamma)
        elif self.kernel == "linear":
            return X.T@X
        elif self.kernel == "poly":
            degree = kwargs.get("degree", 3)
            return (1+X@X.T)**degree
        elif self.kernel == "cosine":
            return X.T@X / (np.linalg.norm(X, axis=0).reshape(-1, 1)@np.linalg.norm(X, axis=0).reshape(1, -1))
        else:
            raise ValueError("Invalid kernel function.")

    def fit(self, X: np.ndarray):
        # X: [n_samples, n_features]

        X = (X - X.mean(axis=0)) / X.std(axis=0)
        kernel_matrix = self.get_kernel_mat(X)
        eignvalues, eignvectors = np.linalg.eig(kernel_matrix)
        # most k large eignvalues and corresponding eignvectors
        indices = np.argsort(eignvalues)[::-1]
        self.eignvalues = eignvalues[indices]
        self.eignvectors = eignvectors[:, indices]
        return self

    def transform(self, X: np.ndarray):
        # X: [n_samples, n_features]
        if self.eignvectors is None:
            raise ValueError("Fit the model first.")
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        W = self.eignvectors[:, :self.n_components]
        return X@W


class KMeans:
    def __init__(self, n_clusters: int = 3, max_iter: int = 10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    # Randomly initialize the centers
    def initialize_centers(self, points):
        # points: (n_samples, n_dims,)
        n, d = points.shape

        self.centers = np.zeros((self.n_clusters, d))
        for k in range(self.n_clusters):
            # use more random points to initialize centers, make kmeans more stable
            random_index = np.random.choice(n, size=10, replace=False)
            self.centers[k] = points[random_index].mean(axis=0)

        return self.centers

    # Assign each point to the closest center
    def assign_points(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        n_samples, n_dims = points.shape
        self.labels = np.zeros(n_samples)
        # TODO: Compute the distance between each point and each center
        # and Assign each point to the closest center
        for i in range(n_samples):
            distances = np.linalg.norm(points[i] - self.centers, axis=1)
            self.labels[i] = np.argmin(distances)
        return self.labels

    # Update the centers based on the new assignment of points
    def update_centers(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Update the centers based on the new assignment of points
        clusters = self.assign_points(points)
        new_centers = np.zeros_like(self.centers)
        for k in range(self.n_clusters):
            cluster_points = points[clusters == k]
            if cluster_points.size == 0:  # 如果这个簇没有任何点，那么跳过这次循环
                continue
            new_centers[k] = cluster_points.mean(axis=0)
        return new_centers

    # k-means clustering
    def fit(self, points):
        # points: (n_samples, n_dims,)
        self.initialize_centers(points)
        for _ in range(self.max_iter):
            new_centers = self.update_centers(points)
            if np.allclose(new_centers, self.centers):  # check convergence, early stop
                break
            self.centers = new_centers
        return self

    # Predict the closest cluster each sample in X belongs to
    def predict(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        return self.assign_points(points)


def load_data():
    words = [
        'computer', 'laptop', 'minicomputers', 'PC', 'software', 'Macbook',
        'king', 'queen', 'monarch', 'prince', 'ruler', 'princes', 'kingdom', 'royal',
        'man', 'woman', 'boy', 'teenager', 'girl', 'robber', 'guy', 'person', 'gentleman',
        'banana', 'pineapple', 'mango', 'papaya', 'coconut', 'potato', 'melon',
        'shanghai', 'HongKong', 'chinese', 'Xiamen', 'beijing', 'Guilin',
        'disease', 'infection', 'cancer', 'illness',
        'twitter', 'facebook', 'chat', 'hashtag', 'link', 'internet',
    ]
    if not os.path.exists('./data/vectors.npy'):
        w2v = KeyedVectors.load_word2vec_format(
            './data/GoogleNews-vectors-negative300.bin', binary=True)
        vectors = []
        for w in words:
            vectors.append(w2v[w].reshape(1, 300))
        vectors = np.concatenate(vectors, axis=0)
        # 保存vector
        np.save('./data/vectors.npy', vectors)
        return words, vectors

    vectors = np.load('./data/vectors.npy')
    return words, vectors


if __name__ == '__main__':
    words, data = load_data()
    kernel_str = "cosine"
    pca = PCA(n_components=2, kernel=kernel_str).fit(data)
    data_pca = pca.transform(data)

    kmeans = KMeans(n_clusters=7).fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # plot the data

    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :])
    plt.title("Your student ID")
    plt.savefig(f"PCA_KMeans{kernel_str}.png")
