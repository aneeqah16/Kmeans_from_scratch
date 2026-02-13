 
"""  
K-Means Clustering Algorithm Implementation  
  
A complete K-Means clustering solution with smart initialization,  
multiple trial runs, and helpful analysis tools.  
  
Author: Aneeqah Ashraf  
Date: February 2026  
"""  
  
import random  
import numpy as np  
  
  
class KMeans:  
    """  
    K-Means clustering - groups similar data points together.  
      
    Args:  
        K: How many groups (clusters) you want  
        max_iters: Maximum number of tries to find best grouping (default: 100)  
      
    Example:  
        >>> model = KMeans(K=3)  
        >>> model.fit(points, n_init=10, init='kmeans++')  
        >>> print(model)  
        >>> predictions = model.predict(new_points)  
    """  
  
    def __init__(self, K: int, max_iters: int = 100):  
        if K < 1:  
            raise ValueError("K must be at least 1")  
  
        self.K = K  
        self.max_iters = max_iters  
        self.centroids = None  
        self.assignments = None  
        self.sse = None  
        self.iterations = None  
  
    # ============================================================  
    #                   DISTANCE CALCULATIONS  
    # ============================================================  
  
    @staticmethod  
    def distance(p1, p2):  
        """Calculate straight-line distance between two points.""" 
        result =  np.linalg.norm(p1 - p2)
        return result  
  
    @staticmethod  
    def distance_squared(p1, p2):  
        """Calculate squared distance (faster, for comparisons only)."""  
        diff = p1 - p2  
        result = np.dot(diff, diff) 
        return  result
  
    # ============================================================  
    #                   INITIALIZATION METHODS  
    # ============================================================  
  
    def _init_random(self, points, seed=None):  
        """Pick K random points as starting cluster centers."""  
        if seed is not None:  
            random.seed(seed)  
        indices = random.sample(range(len(points)), self.K)  
        return [points[i].astype(float) for i in indices]  
  
    def _init_kmeans_plusplus(self, points, seed=None):  
        """  
        Pick starting centers smartly (K-Means++ method).  
        Chooses centers that are far apart for better results.  
        """  
        if seed is not None:  
            random.seed(seed)  
            np.random.seed(seed)  
  
        centroids = []  
          
        # Pick first center randomly  
        first_idx = random.randint(0, len(points) - 1)  
        centroids.append(points[first_idx])  
  
        # Pick remaining centers (farther points more likely)  
        for _ in range(self.K - 1):  
            distances = np.array([  
                min(self.distance_squared(p, c) for c in centroids)  
                for p in points  
            ])  
  
            total = distances.sum()  
            if total == 0:  
                next_centroid = points[random.randint(0, len(points) - 1)]  
            else:  
                probs = distances / total  
                next_idx = np.random.choice(len(points), p=probs)  
                next_centroid = points[next_idx]  
  
            centroids.append(next_centroid)  
  
        return centroids  
  
    # ============================================================  
    #                   CORE ALGORITHM STEPS  
    # ============================================================  
  
    def _assign_clusters(self, points, centroids):  
        """Assign each point to its nearest cluster center."""  
        assignments = []  
        for point in points:  
            distances = [self.distance_squared(point, c) for c in centroids]  
            assignments.append(np.argmin(distances))  
        return assignments  
  
    def _update_centroids(self, points, assignments):  
        """Move each cluster center to the middle of its points."""  
        new_centroids = []  
        for k in range(self.K):  
            cluster_points = points[np.array(assignments) == k]  
              
            if len(cluster_points) == 0:  
                new_centroids.append(points[random.randint(0, len(points) - 1)])  
            else:  
                new_centroids.append(cluster_points.mean(axis=0))  
                  
        return new_centroids  
  
    def _compute_sse(self, points, assignments, centroids):  
        """  
        Calculate how good the clustering is.  
        Lower number = better clustering (points closer to their centers).  
        """  
        sse = 0.0  
        for i, point in enumerate(points):  
            sse += self.distance_squared(point, centroids[assignments[i]])  
        return sse  
  
    def _max_centroid_movement(self, old, new):  
        """Check how far the centers moved."""  
        for old, new in zip(old, new):
            movement = max(self.distance(o, n) for o, n in zip(old, new))  
  
        return movement
    # ============================================================  
    #                   MAIN ALGORITHM  
    # ============================================================  
  
    def run(self, points, tol=1e-4, init="random", verbose=False, seed=None):  
        """  
        Run the K-Means algorithm to cluster data.  
          
        How it works:  
          1. Pick K starting centers  
          2. Assign each point to nearest center  
          3. Move centers to middle of their groups  
          4. Repeat steps 2-3 until centers stop moving  
          
        Args:  
            points: Your data points  
            tol: Stop when centers move less than this (default: 0.0001)  
            init: How to pick starting centers - 'random' or 'kmeans++' (default: 'random')  
            verbose: Print progress at each step (default: False)  
            seed: Number for reproducible results (default: None)  
          
        Returns:  
            self (for method chaining)  
        """  
        points = np.array(points)  
        if self.K > len(points):  
            raise ValueError(f"K ({self.K}) can't be more than number of points ({len(points)})")  
  
        # Step 1: Pick starting centers  
        if init == "kmeans++":  
            centroids = self._init_kmeans_plusplus(points, seed)  
        elif init == "random":  
            centroids = self._init_random(points, seed)  
        else:  
            raise ValueError("init must be 'random' or 'kmeans++'")  
  
        previous_assignments = None  
  
        # Main loop: keep improving until it stops changing  
        for iteration in range(self.max_iters):  
            assignments = self._assign_clusters(points, centroids)  
            new_centroids = self._update_centroids(points, assignments)  
            movement = self._max_centroid_movement(centroids, new_centroids)  
  
            if verbose:  
                sse = self._compute_sse(points, assignments, centroids)  
                print(f"Step {iteration + 1}: centers moved {movement:.6f}, quality={sse:.4f}")  
  
            # Stop if nothing changed or centers barely moved  
            if previous_assignments == assignments or movement < tol:  
                break  
  
            centroids = new_centroids  
            previous_assignments = assignments  
  
        # Save results  
        self.centroids = centroids  
        self.assignments = assignments  
        self.sse = self._compute_sse(points, assignments, centroids)  
        self.iterations = iteration + 1  
  
        return self  
  
    # ============================================================  
    #                   FIT WITH MULTIPLE RUNS  
    # ============================================================  
  
    def fit(self, points, n_init=10, init="random", tol=1e-4, verbose=False):  
        """  
        Run K-Means multiple times and keep the best result.  
          
        Why? Sometimes you get unlucky with starting points. Running it  
        multiple times and picking the best gives more reliable results.  
          
        Args:  
            points: Your data points  
            n_init: How many times to try (default: 10)  
            init: 'random' or 'kmeans++' (default: 'random')  
            tol: Stop when centers move less than this (default: 0.0001)  
            verbose: Show progress (default: False)  
          
        Returns:  
            self (for method chaining)  
        """  
        points = np.array(points)  
        best_sse = float("inf")  
        best_centroids = None  
        best_assignments = None  
        best_iterations = None  
  
        if verbose:  
            print(f"Trying K-Means {n_init} times (using {init})...")  
  
        for i in range(n_init):  
            # Create temporary model for this run  
            temp_model = KMeans(K=self.K, max_iters=self.max_iters)  
            temp_model.run(points, tol=tol, init=init, seed=i, verbose=False)  
  
            if verbose:  
                print(f"  Try {i+1}/{n_init}: quality={temp_model.sse:.4f}")  
  
            # Keep this result if it's the best so far  
            if temp_model.sse < best_sse:  
                best_sse = temp_model.sse  
                best_centroids = temp_model.centroids  
                best_assignments = temp_model.assignments  
                best_iterations = temp_model.iterations  
  
        # Store best results  
        self.centroids = best_centroids  
        self.assignments = best_assignments  
        self.sse = best_sse  
        self.iterations = best_iterations  
  
        if verbose:  
            print(f"Best quality: {best_sse:.4f}\n")  
  
        return self  
  
    # ============================================================  
    #                   PREDICTION & UTILITIES  
    # ============================================================  
  
    def predict(self, points):  
        """  
        Predict which cluster new points belong to.  
        Model must be fitted first.  
        """  
        if self.centroids is None:  
            raise ValueError("Model not fitted. Call fit() or run() first.")  
        return self._assign_clusters(np.array(points), self.centroids)  
  
    def get_cluster_sizes(self):  
        """Get how many points are in each cluster (as a list)."""  
        if self.assignments is None:  
            raise ValueError("Model not fitted yet.")  
          
        sizes = [0] * self.K  
        for cluster_id in self.assignments:  
            sizes[cluster_id] += 1  
        return sizes  
  
    def export_results(self, points, filepath):  
        """Save clustering results to a CSV file."""  
        import csv  
          
        if self.assignments is None:  
            raise ValueError("Model not fitted yet.")  
          
        points = np.array(points)  
        n_dims = points.shape[1]  
          
        with open(filepath, 'w', newline='') as f:  
            writer = csv.writer(f)  
            headers = [f'feature_{i}' for i in range(n_dims)] + ['cluster_id']  
            writer.writerow(headers)  
            for point, cluster in zip(points, self.assignments):  
                writer.writerow(list(point) + [cluster])  
          
        print(f"Results saved to '{filepath}'")  
  
    # ============================================================  
    #                   STATIC ANALYSIS METHODS  
    # ============================================================  
  
    @staticmethod  
    def elbow_method(points, max_k=10, max_iters=100, init="random",   
                     n_init=5, verbose=True):  
        """  
        Elbow method - helps you find the best number of clusters (K).  
          
        Tries different K values (1, 2, 3, ...) and shows quality for each.  
        Plot the results - look for an "elbow" where quality stops improving much.  
          
        Args:  
            points: Your data  
            max_k: Test up to this many clusters (default: 10)  
            max_iters: Max steps per try (default: 100)  
            init: 'random' or 'kmeans++' (default: 'random')  
            n_init: Tries per K value (default: 5)  
            verbose: Show progress (default: True)  
          
        Returns:  
            Dictionary like {1: 245.3, 2: 123.4, 3: 89.2, ...}  
            Keys are K values, values are quality scores  
        """  
        points = np.array(points)  
        results = {}  
  
        if verbose:  
            print(f"Testing K from 1 to {max_k}:")  
  
        for k in range(1, max_k + 1):  
            model = KMeans(K=k, max_iters=max_iters)  
            model.fit(points, n_init=n_init, init=init, tol=1e-4, verbose=False)  
              
            results[k] = model.sse  
            if verbose:  
                print(f"  K={k}: quality={model.sse:.4f}")  
  
        return results  
  
    @staticmethod  
    def scale_minmax(points):  
        """  
        Scale all features to range [0, 1].  
        Useful when features have different units/scales.  
          
        Returns:  
            (scaled_points, scaling_info)  
        """  
        points = np.array(points)  
        mins = points.min(axis=0)  
        maxs = points.max(axis=0)  
        ranges = maxs - mins  
        ranges[ranges == 0] = 1  
        scaled = (points - mins) / ranges  
        return scaled, {'mins': mins, 'maxs': maxs}  
  
    @staticmethod  
    def scale_zscore(points):  
        """  
        Scale features to mean=0, standard deviation=1.  
        Useful when you want standardized values.  
          
        Returns:  
            (scaled_points, scaling_info)  
        """  
        points = np.array(points)  
        means = points.mean(axis=0)  
        stds = points.std(axis=0)  
        stds[stds == 0] = 1  
        scaled = (points - means) / stds  
        return scaled, {'means': means, 'stds': stds}  
  
    # ============================================================  
    #                   STRING REPRESENTATIONS  
    # ============================================================  
  
    def __repr__(self):  
        return f"KMeans(K={self.K}, max_iters={self.max_iters})"  
  
    def __str__(self):  
        if self.centroids is None:  
            return f"{repr(self)} - Not fitted"  
        sizes = self.get_cluster_sizes()  
        return (  
            f"{repr(self)}\n"  
            f"  Steps taken: {self.iterations}\n"  
            f"  Quality (SSE): {self.sse:.4f}\n"  
            f"  Cluster sizes: {sizes}"  
        )

