# Testing
import numpy as np
from src.kmeans import Kmeans
from src.io_utils import load_points

points = load_points("data/sample.csv")
# print(type(points))
# print(type(points[0]), points[0])

K = 2
km = Kmeans(K=2 , max_iters=100)
centroids, assignments, sse, iterations = km.kmeans(points,0.0001,init="random",verbose = True, seed = None)
sizes = km.cluster_sizes()
print("Centroids:", centroids)
print("Assignments:", assignments)
print("SSE:", sse)
print("Iterations:", iterations)
labels, counts = np.unique(km.assignments, return_counts=True)
print("Cluster sizes:", dict(zip(labels, counts)))
print(sizes)

