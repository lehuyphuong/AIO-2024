import numpy as np
import matplotlib.pyplot as plt

data = np.array([2.0, 3.0, 1.5,
                 3.0, 3.5, 2.0,
                 3.5, 3.0, 2.5,
                 8.0, 8.0, 7.5,
                 8.5, 8.5, 8.0,
                 9.0, 8.0, 8.5,
                 1.0, 2.0, 1.0,
                 1.5, 2.5, 1.5])

data = np.resize(data, (8, 3))


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = None

    def initalize_centroids(self, data):
        rng = np.random.default_rng(seed=np.random.seed(42))
        self.centroids = data[rng.choice(
            data.shape[0], self.k, replace=False)]

    def euclidean_distance(self, x1, x2):
        retval = np.sqrt(np.sum(np.power(x1-x2, 2)))
        return retval

    def assign_clusters(self, data):
        distances = np.array([[self.euclidean_distance(x, centroid)
                             for centroid in self.centroids] for x in data])
        return np.argmin(distances, axis=1)

    def update_centroids(self, data):
        return np.array([data[self.clusters == i].mean(axis=0)
                         for i in range(self.k)])

    def plot_clusters(self, data, iteration):
        # Create a new figure for 3D plotting
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=self.clusters,
                   cmap="viridis", marker="o", alpha=.6)
        ax.scatter(
            self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2],
            s=200, c="red", marker="x")
        fig.canvas.draw()

        plt.title(f"Iteration {iteration + 1}")
        ax.set_xlabel("feature1")
        ax.set_ylabel("feature2")
        ax.set_zlabel("feature3")
        plt.show()

    def plot_final_clusters(self, data):
        # Create a new figure for 3D plotting
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=self.clusters,
                   cmap="viridis", marker="o", alpha=.6)
        ax.scatter(
            self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2],
            s=300, c="red", marker="x")
        fig.canvas.draw()
        plt.title("Final clusters and centroid")
        ax.set_xlabel("feature1")
        ax.set_ylabel("feature2")
        ax.set_zlabel("feature3")
        plt.show()
        plt.show()

    def fit(self, data):
        self.initalize_centroids(data)

        for i in range(self.max_iters):
            self.clusters = self.assign_clusters(data)

            self.plot_clusters(data, i)

            new_centroids = self.update_centroids(data)
            print(f"new centroid point would be {new_centroids}")

            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        self.plot_final_clusters(data)


kmeans = KMeans(k=3, max_iters=100)
kmeans.fit(data)
