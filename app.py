# Testing
import numpy as np
from src.kmeans import Kmeans
from src.io_utils import load_points

points = load_points("data/sample.csv")
# print(type(points))
# print(type(points[0]), points[0])

K = 2
km = Kmeans(K=2 , max_iters=100)
km.fit(points)

print("Centroids:", km.centroids)
print("Assignments:", km.assignments)
print("SSE:", km.sse)
print("Iterations:", km.iterations)
labels, counts = np.unique(km.assignments, return_counts=True)
print("Cluster sizes:", dict(zip(labels, counts)))

