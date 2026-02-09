"""K-Means Clustering ALgorithm Implementation
This module contains core K-Means Clustering Algorithm and helper Functions.

Author: Aneeqah Ashraf
Date : Februray 2026
"""

import random
import math

def distance_squared(p1,p2):
    "Calculate squared Euclidean distance between two points"
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 

def distance(p1,p2):
    "Calculate Euclidean distance bewteen two points"
    return math.sqrt(distance_squared(p1,p2))

def init_centroids(points, K,seed = None):
    "Initialize K centroids by randomly selecting k distinct points"
    if K >len(points):
        raise ValueError(f"K ({K} cannot be greater than the number of points {len(points)}")
    
    if seed is not None:
        random.seed(seed)
    
    return random.sample(points, K)

def assign_clusters(points , centroids):
    """Assign each point to the nearst centroid"""

    assignments = []

    for point in points:
        min_distance = float("inf")
        closest_centroid = 0

        for i, centroid in enumerate(centroids):
            dist = distance_squared(point, centroid)

            if dist< min_distance:
                min_distance = dist
                closest_centroid = i

        assignments.append(closest_centroid)

    return assignments

def update_clusters(points, assignments,K):
    new_centroids = []

    for k in range(K):
        cluster_points = [points[i] for i in range(len(points))]

        if len(cluster_points) == 0:
            new_centroid = random.choice(points)
        else:
            mean_x = sum(p[0] for p in cluster_points) / len(cluster_points)
            mean_y = sum(p[1] for p in cluster_points) / len(cluster_points)
            new_centroid = (mean_x, mean_y)

        new_centroids.append(new_centroid)
    return new_centroids

def compute_sse(points,assignment,centroids):
    "Computer Sum of Squared Errors SSE"
    sse = 0.0

    for i, point in enumerate(points):
        clusters_id = assignment[i]
        centroid = centroids[clusters_id]
        sse += distance_squared(points,centroid)

    return sse

def max_centroid_movement(old_centroids, new_centroids):
    """Calculate the maximum distance of any centroid"""
    max_movement = 0.0

    for old, new in zip(old_centroids, new_centroids):
        movement = distance(old, new)
        if movement>max_movement:
            max_movement = movement

        return max_movement
    
def kmeans(points, K, max_iters = 100, tol = 1e-4, seed = None):
    """Perform Kmeans clustering
    Returns : (centroids, assignments,sse, iterations)
    """
    if K<1:
        raise ValueError("K must be at least 1")
    if K>len(points):
        raise ValueError(f"k:{K} cannot be greater than number of points ({len(points)})")
    centroids = init_centroids(points,K, seed)
    previous_assignment = None
    iterations = 0

    for iteration in range(max_iters):
        iterations = iteration + 1

        assignments = assign_clusters(points, centroids)

        new_centroids = update_clusters(points, assignments)

        if previous_assignment is not None and previous_assignment== assignments :
            break

        movement = max_centroid_movement(centroids, new_centroids)
        if movement < tol:
            break

        centroids = new_centroids
        previous_assignment = assignments[:]

    sse = compute_sse(points,assignments,centroids)

    return centroids, assignments, sse, iterations

def get_cluster_sizes(assignments,K):
    """Count Number of points in each cluster"""
    sizes = [0] * K
    for cluster_id in assignments:
        sizes[cluster_id] +=1

    return sizes

