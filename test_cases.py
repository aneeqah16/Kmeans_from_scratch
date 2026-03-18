"""
test_cases.py
Test suite for K-Means Clustering CLI project.

Covers:
  - KMeans class (kmeans.py)
  - Data utilities (io_utils.py)
  - Input validation logic (app.py)

Run with:
    python -m pytest test_cases.py -v
    or
    python test_cases.py
"""

import sys
import os
import csv
import math
import random
import tempfile
import unittest
import numpy as np

# Make sure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kmeans import KMeans
from io_utils import (
    generate_sample_data,
    save_points_to_csv,
    load_points,
    scale_minmax,
    scale_zscore,
)


# ══════════════════════════════════════════════════════════════
# SECTION 1 — KMeans Class: Initialisation & Basic Checks
# ══════════════════════════════════════════════════════════════

class TestKMeansInit(unittest.TestCase):
    """Tests for KMeans.__init__() and basic object state."""

    def test_valid_k_and_max_iters(self):
        """KMeans object is created with correct K and max_iters."""
        model = KMeans(K=3, max_iters=50)
        self.assertEqual(model.K, 3)
        self.assertEqual(model.max_iters, 50)

    def test_default_max_iters(self):
        """Default max_iters should be 100."""
        model = KMeans(K=2)
        self.assertEqual(model.max_iters, 100)

    def test_invalid_k_zero(self):
        """K=0 should raise ValueError."""
        with self.assertRaises(ValueError):
            KMeans(K=0)

    def test_invalid_k_negative(self):
        """Negative K should raise ValueError."""
        with self.assertRaises(ValueError):
            KMeans(K=-5)

    def test_unfitted_model_state(self):
        """Unfitted model should have None for all result attributes."""
        model = KMeans(K=2)
        self.assertIsNone(model.centroids)
        self.assertIsNone(model.assignments)
        self.assertIsNone(model.sse)
        self.assertIsNone(model.iterations)

    def test_repr(self):
        """__repr__ should return correct string."""
        model = KMeans(K=3, max_iters=50)
        self.assertEqual(repr(model), "KMeans(K=3, max_iters=50)")

    def test_str_unfitted(self):
        """__str__ on unfitted model should mention 'Not fitted'."""
        model = KMeans(K=2)
        self.assertIn("Not fitted", str(model))


# ══════════════════════════════════════════════════════════════
# SECTION 2 — Distance Methods
# ══════════════════════════════════════════════════════════════

class TestDistanceMethods(unittest.TestCase):
    """Tests for KMeans.distance() and KMeans.distance_squared()."""

    def test_distance_same_point(self):
        """Distance from a point to itself is 0."""
        p = np.array([3.0, 4.0])
        self.assertAlmostEqual(KMeans.distance(p, p), 0.0)

    def test_distance_known_value(self):
        """Distance between (0,0) and (3,4) is 5.0."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        self.assertAlmostEqual(KMeans.distance(p1, p2), 5.0)

    def test_distance_symmetric(self):
        """distance(a, b) == distance(b, a)."""
        a = np.array([1.0, 2.0])
        b = np.array([4.0, 6.0])
        self.assertAlmostEqual(KMeans.distance(a, b), KMeans.distance(b, a))

    def test_distance_squared_same_point(self):
        """Squared distance from a point to itself is 0."""
        p = np.array([5.0, 5.0])
        self.assertAlmostEqual(KMeans.distance_squared(p, p), 0.0)

    def test_distance_squared_known_value(self):
        """Squared distance between (0,0) and (3,4) is 25.0."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        self.assertAlmostEqual(KMeans.distance_squared(p1, p2), 25.0)

    def test_distance_squared_equals_distance_squared(self):
        """distance_squared should equal distance ** 2."""
        a = np.array([2.0, 7.0])
        b = np.array([5.0, 3.0])
        self.assertAlmostEqual(
            KMeans.distance_squared(a, b),
            KMeans.distance(a, b) ** 2
        )

    def test_distance_multidimensional(self):
        """Distance works for points with more than 2 dimensions."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        self.assertAlmostEqual(KMeans.distance(a, b), 1.0)


# ══════════════════════════════════════════════════════════════
# SECTION 3 — KMeans: Fitting and Assignments
# ══════════════════════════════════════════════════════════════

class TestKMeansFitting(unittest.TestCase):
    """Tests for KMeans.fit() and KMeans.kmeans()."""

    def setUp(self):
        """Two clearly separated clusters — easy to cluster perfectly."""
        random.seed(42)
        cluster1 = [(random.uniform(0, 2), random.uniform(0, 2)) for _ in range(20)]
        cluster2 = [(random.uniform(8, 10), random.uniform(8, 10)) for _ in range(20)]
        self.points = np.array(cluster1 + cluster2)

    def test_fit_sets_centroids(self):
        """After fit(), centroids should not be None."""
        model = KMeans(K=2)
        model.fit(self.points, n_init=5)
        self.assertIsNotNone(model.centroids)
        self.assertEqual(len(model.centroids), 2)

    def test_fit_sets_assignments(self):
        """After fit(), assignments length equals number of points."""
        model = KMeans(K=2)
        model.fit(self.points, n_init=5)
        self.assertEqual(len(model.assignments), len(self.points))

    def test_fit_sets_sse(self):
        """After fit(), SSE is a positive float."""
        model = KMeans(K=2)
        model.fit(self.points, n_init=5)
        self.assertIsInstance(model.sse, float)
        self.assertGreater(model.sse, 0)

    def test_fit_sets_iterations(self):
        """After fit(), iterations is a positive integer."""
        model = KMeans(K=2)
        model.fit(self.points, n_init=5)
        self.assertIsNotNone(model.iterations)
        self.assertGreater(model.iterations, 0)

    def test_two_cluster_separation(self):
        """Two clear clusters: all cluster-0 and cluster-1 points should be grouped together."""
        model = KMeans(K=2)
        model.fit(self.points, n_init=10, init='kmeans++')
        assignments = model.assignments
        # First 20 points should share one cluster label
        self.assertEqual(len(set(assignments[:20])), 1)
        # Last 20 points should share one cluster label
        self.assertEqual(len(set(assignments[20:])), 1)
        # The two groups should be in different clusters
        self.assertNotEqual(assignments[0], assignments[20])

    def test_k_exceeds_points_raises(self):
        """K greater than number of points should raise ValueError."""
        model = KMeans(K=50)
        with self.assertRaises(ValueError):
            model.fit(self.points[:5], n_init=1)

    def test_k_equals_points(self):
        """K == number of points is valid — each point is its own cluster."""
        small = self.points[:5]
        model = KMeans(K=5)
        model.fit(small, n_init=1)
        self.assertAlmostEqual(model.sse, 0.0, places=5)

    def test_k1_sse(self):
        """K=1 assigns all points to one cluster."""
        model = KMeans(K=1)
        model.fit(self.points, n_init=3)
        self.assertEqual(len(set(model.assignments)), 1)

    def test_multiple_runs_sse_not_worse(self):
        """Running with n_init=10 should give SSE <= n_init=1."""
        model1 = KMeans(K=2)
        model1.fit(self.points, n_init=1, seed=0) if hasattr(model1.fit, 'seed') else model1.fit(self.points, n_init=1)
        model10 = KMeans(K=2)
        model10.fit(self.points, n_init=10)
        self.assertLessEqual(model10.sse, model1.sse + 1e-6)

    def test_returns_self(self):
        """fit() should return self for method chaining."""
        model = KMeans(K=2)
        result = model.fit(self.points, n_init=3)
        self.assertIs(result, model)

    def test_invalid_init_method(self):
        """Unknown init method should raise ValueError."""
        model = KMeans(K=2)
        with self.assertRaises(ValueError):
            model.kmeans(self.points, init='invalid_method')


# ══════════════════════════════════════════════════════════════
# SECTION 4 — Initialisation Methods
# ══════════════════════════════════════════════════════════════

class TestInitialisationMethods(unittest.TestCase):
    """Tests for random and K-Means++ initialisation."""

    def setUp(self):
        random.seed(0)
        np.random.seed(0)
        self.points = np.array(
            [(random.uniform(0, 2), random.uniform(0, 2)) for _ in range(15)] +
            [(random.uniform(8, 10), random.uniform(8, 10)) for _ in range(15)]
        )

    def test_random_init_returns_k_centroids(self):
        """Random init should return exactly K centroids."""
        model = KMeans(K=3)
        centroids = model._init_random(self.points, seed=42)
        self.assertEqual(len(centroids), 3)

    def test_random_init_centroids_are_from_data(self):
        """Random init centroids should all be data points."""
        model = KMeans(K=3)
        centroids = model._init_random(self.points, seed=42)
        for c in centroids:
            self.assertTrue(any(np.allclose(c, p) for p in self.points))

    def test_kmeans_plusplus_returns_k_centroids(self):
        """K-Means++ init should return exactly K centroids."""
        model = KMeans(K=3)
        centroids = model._init_kmeans_plusplus(self.points, seed=42)
        self.assertEqual(len(centroids), 3)

    def test_kmeans_plusplus_centroids_are_from_data(self):
        """K-Means++ centroids should all be data points."""
        model = KMeans(K=3)
        centroids = model._init_kmeans_plusplus(self.points, seed=42)
        for c in centroids:
            self.assertTrue(any(np.allclose(c, p) for p in self.points))

    def test_kmeans_plusplus_no_duplicate_centroids(self):
        """K-Means++ should not select the same centroid twice (on well-separated data)."""
        model = KMeans(K=2)
        centroids = model._init_kmeans_plusplus(self.points, seed=0)
        self.assertFalse(np.allclose(centroids[0], centroids[1]))

    def test_random_init_reproducible_with_seed(self):
        """Same seed should produce same centroids."""
        model = KMeans(K=2)
        c1 = model._init_random(self.points, seed=7)
        c2 = model._init_random(self.points, seed=7)
        for a, b in zip(c1, c2):
            self.assertTrue(np.allclose(a, b))

    def test_kmeans_plusplus_reproducible_with_seed(self):
        """Same seed for K-Means++ should produce same centroids."""
        model = KMeans(K=2)
        c1 = model._init_kmeans_plusplus(self.points, seed=7)
        c2 = model._init_kmeans_plusplus(self.points, seed=7)
        for a, b in zip(c1, c2):
            self.assertTrue(np.allclose(a, b))


# ══════════════════════════════════════════════════════════════
# SECTION 5 — Predict and Utilities
# ══════════════════════════════════════════════════════════════

class TestPredictAndUtilities(unittest.TestCase):
    """Tests for KMeans.predict(), get_cluster_sizes(), export_results()."""

    def setUp(self):
        random.seed(1)
        cluster1 = [(random.uniform(0, 2), random.uniform(0, 2)) for _ in range(15)]
        cluster2 = [(random.uniform(8, 10), random.uniform(8, 10)) for _ in range(15)]
        self.points = np.array(cluster1 + cluster2)
        self.model = KMeans(K=2)
        self.model.fit(self.points, n_init=5, init='kmeans++')

    def test_predict_returns_correct_length(self):
        """predict() should return one label per input point."""
        new_points = np.array([[1.0, 1.0], [9.0, 9.0]])
        preds = self.model.predict(new_points)
        self.assertEqual(len(preds), 2)

    def test_predict_near_cluster1(self):
        """Point near cluster 1 centre should be assigned to cluster 1."""
        near_c1 = np.array([[1.0, 1.0]])
        near_c2 = np.array([[9.0, 9.0]])
        pred1 = self.model.predict(near_c1)[0]
        pred2 = self.model.predict(near_c2)[0]
        self.assertNotEqual(pred1, pred2)

    def test_predict_unfitted_raises(self):
        """predict() on unfitted model should raise ValueError."""
        model = KMeans(K=2)
        with self.assertRaises(ValueError):
            model.predict(self.points)

    def test_get_cluster_sizes_sum(self):
        """Cluster sizes should sum to total number of points."""
        sizes = self.model.get_cluster_sizes()
        self.assertEqual(sum(sizes), len(self.points))

    def test_get_cluster_sizes_length(self):
        """get_cluster_sizes() should return exactly K values."""
        sizes = self.model.get_cluster_sizes()
        self.assertEqual(len(sizes), 2)

    def test_get_cluster_sizes_all_positive(self):
        """Each cluster should have at least one point."""
        sizes = self.model.get_cluster_sizes()
        for s in sizes:
            self.assertGreater(s, 0)

    def test_get_cluster_sizes_unfitted_raises(self):
        """get_cluster_sizes() on unfitted model should raise ValueError."""
        model = KMeans(K=2)
        with self.assertRaises(ValueError):
            model.get_cluster_sizes()

    def test_export_results_creates_file(self):
        """export_results() should create a CSV file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'results.csv')
            self.model.export_results(self.points, path)
            self.assertTrue(os.path.exists(path))

    def test_export_results_row_count(self):
        """Exported CSV should have one row per point plus a header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'results.csv')
            self.model.export_results(self.points, path)
            with open(path) as f:
                rows = list(csv.reader(f))
            self.assertEqual(len(rows), len(self.points) + 1)

    def test_export_results_unfitted_raises(self):
        """export_results() on unfitted model should raise ValueError."""
        model = KMeans(K=2)
        with self.assertRaises(ValueError):
            model.export_results(self.points, 'dummy.csv')


# ══════════════════════════════════════════════════════════════
# SECTION 6 — SSE and Elbow Method
# ══════════════════════════════════════════════════════════════

class TestSSEAndElbow(unittest.TestCase):
    """Tests for SSE computation and elbow_method()."""

    def setUp(self):
        random.seed(2)
        cluster1 = [(random.uniform(0, 2), random.uniform(0, 2)) for _ in range(15)]
        cluster2 = [(random.uniform(8, 10), random.uniform(8, 10)) for _ in range(15)]
        self.points = np.array(cluster1 + cluster2)

    def test_sse_decreases_with_more_clusters(self):
        """SSE for K=2 should be less than SSE for K=1."""
        m1 = KMeans(K=1)
        m1.fit(self.points, n_init=3)
        m2 = KMeans(K=2)
        m2.fit(self.points, n_init=5, init='kmeans++')
        self.assertLess(m2.sse, m1.sse)

    def test_sse_is_non_negative(self):
        """SSE should always be >= 0."""
        model = KMeans(K=2)
        model.fit(self.points, n_init=5)
        self.assertGreaterEqual(model.sse, 0)

    def test_sse_zero_when_k_equals_n(self):
        """SSE should be 0 when K equals number of points."""
        small = self.points[:5]
        model = KMeans(K=5)
        model.fit(small, n_init=1)
        self.assertAlmostEqual(model.sse, 0.0, places=5)

    def test_elbow_method_returns_dict(self):
        """elbow_method() should return a dictionary."""
        result = KMeans.elbow_method(self.points, max_k=5, n_init=3)
        self.assertIsInstance(result, dict)

    def test_elbow_method_keys(self):
        """elbow_method() keys should be 1 to max_k."""
        result = KMeans.elbow_method(self.points, max_k=5, n_init=3)
        self.assertEqual(sorted(result.keys()), list(range(1, 6)))

    def test_elbow_method_sse_decreasing(self):
        """SSE values from elbow method should be non-increasing."""
        result = KMeans.elbow_method(self.points, max_k=5, n_init=5)
        values = [result[k] for k in sorted(result.keys())]
        for i in range(len(values) - 1):
            self.assertGreaterEqual(values[i], values[i+1] - 1e-6)

    def test_elbow_method_k1_largest_sse(self):
        """K=1 should always have the largest SSE."""
        result = KMeans.elbow_method(self.points, max_k=5, n_init=5)
        self.assertEqual(max(result.values()), result[1])


# ══════════════════════════════════════════════════════════════
# SECTION 7 — io_utils: Data Generation and CSV
# ══════════════════════════════════════════════════════════════

class TestDataGeneration(unittest.TestCase):
    """Tests for generate_sample_data() and CSV save/load."""

    def test_generate_returns_20_points(self):
        """generate_sample_data() should return 20 points."""
        points = generate_sample_data()
        self.assertEqual(len(points), 20)

    def test_generate_returns_tuples(self):
        """Each generated point should be a tuple of 2 floats."""
        points = generate_sample_data()
        for p in points:
            self.assertEqual(len(p), 2)
            self.assertIsInstance(p[0], float)
            self.assertIsInstance(p[1], float)

    def test_generate_two_clusters(self):
        """First 10 points near (2,2) and last 10 near (8,8)."""
        points = generate_sample_data()
        for x, y in points[:10]:
            self.assertGreaterEqual(x, 1.0)
            self.assertLessEqual(x, 3.0)
        for x, y in points[10:]:
            self.assertGreaterEqual(x, 7.0)
            self.assertLessEqual(x, 9.0)

    def test_save_and_load_roundtrip(self):
        """save_points_to_csv then load_points should return same data."""
        points = [(1.5, 2.5), (8.0, 9.0), (0.5, 0.5)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            save_points_to_csv(points, path)
            loaded = load_points(path)
        self.assertEqual(len(loaded), len(points))
        for orig, load in zip(points, loaded):
            self.assertAlmostEqual(orig[0], load[0], places=5)
            self.assertAlmostEqual(orig[1], load[1], places=5)

    def test_load_skips_header(self):
        """load_points() should skip the CSV header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.csv')
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y'])
                writer.writerow([1.0, 2.0])
                writer.writerow([3.0, 4.0])
            points = load_points(path)
        self.assertEqual(len(points), 2)

    def test_load_missing_file_generates_data(self):
        """load_points() with a missing file should generate and return fallback data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'nonexistent.csv')
            points = load_points(path)
        self.assertEqual(len(points), 20)

    def test_load_missing_file_saves_csv(self):
        """load_points() with missing file should also save the generated CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'nonexistent.csv')
            load_points(path)
            self.assertTrue(os.path.exists(path))

    def test_load_skips_invalid_rows(self):
        """load_points() should skip non-numeric rows without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'bad.csv')
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y'])
                writer.writerow([1.0, 2.0])
                writer.writerow(['bad', 'row'])
                writer.writerow([3.0, 4.0])
            points = load_points(path)
        self.assertEqual(len(points), 2)


# ══════════════════════════════════════════════════════════════
# SECTION 8 — io_utils: Feature Scaling
# ══════════════════════════════════════════════════════════════

class TestFeatureScaling(unittest.TestCase):
    """Tests for scale_minmax() and scale_zscore()."""

    def setUp(self):
        self.points = np.array([
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ])

    def test_minmax_range(self):
        """Min-max scaled data should be in [0, 1] for each feature."""
        scaled, _ = scale_minmax(self.points)
        self.assertAlmostEqual(scaled.min(), 0.0)
        self.assertAlmostEqual(scaled.max(), 1.0)

    def test_minmax_min_is_zero(self):
        """Minimum value after min-max scaling should be 0."""
        scaled, _ = scale_minmax(self.points)
        self.assertAlmostEqual(scaled[:, 0].min(), 0.0)
        self.assertAlmostEqual(scaled[:, 1].min(), 0.0)

    def test_minmax_max_is_one(self):
        """Maximum value after min-max scaling should be 1."""
        scaled, _ = scale_minmax(self.points)
        self.assertAlmostEqual(scaled[:, 0].max(), 1.0)
        self.assertAlmostEqual(scaled[:, 1].max(), 1.0)

    def test_minmax_returns_params(self):
        """scale_minmax() should return scaling parameters dict with mins and maxs."""
        _, params = scale_minmax(self.points)
        self.assertIn('mins', params)
        self.assertIn('maxs', params)

    def test_minmax_shape_preserved(self):
        """Min-max scaling should not change the shape of the data."""
        scaled, _ = scale_minmax(self.points)
        self.assertEqual(scaled.shape, self.points.shape)

    def test_minmax_constant_column(self):
        """Min-max scaling on a constant column should not produce NaN."""
        pts = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        scaled, _ = scale_minmax(pts)
        self.assertFalse(np.isnan(scaled).any())

    def test_zscore_mean_near_zero(self):
        """After z-score scaling, each feature mean should be ~0."""
        scaled, _ = scale_zscore(self.points)
        for col in range(scaled.shape[1]):
            self.assertAlmostEqual(scaled[:, col].mean(), 0.0, places=5)

    def test_zscore_std_near_one(self):
        """After z-score scaling, each feature std should be ~1."""
        scaled, _ = scale_zscore(self.points)
        for col in range(scaled.shape[1]):
            self.assertAlmostEqual(scaled[:, col].std(), 1.0, places=5)

    def test_zscore_returns_params(self):
        """scale_zscore() should return scaling params with means and stds."""
        _, params = scale_zscore(self.points)
        self.assertIn('means', params)
        self.assertIn('stds', params)

    def test_zscore_shape_preserved(self):
        """Z-score scaling should not change the shape of the data."""
        scaled, _ = scale_zscore(self.points)
        self.assertEqual(scaled.shape, self.points.shape)

    def test_zscore_constant_column(self):
        """Z-score scaling on a constant column should not produce NaN."""
        pts = np.array([[3.0, 2.0], [3.0, 4.0], [3.0, 6.0]])
        scaled, _ = scale_zscore(pts)
        self.assertFalse(np.isnan(scaled).any())


# ══════════════════════════════════════════════════════════════
# SECTION 9 — Edge Cases
# ══════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Edge case and boundary tests."""

    def test_single_point_k1(self):
        """Single point with K=1 should work with SSE=0."""
        points = np.array([[5.0, 5.0]])
        model = KMeans(K=1)
        model.fit(points, n_init=1)
        self.assertAlmostEqual(model.sse, 0.0)

    def test_all_identical_points(self):
        """All identical points should cluster without error."""
        points = np.array([[3.0, 3.0]] * 10)
        model = KMeans(K=2)
        model.fit(points, n_init=3)
        self.assertIsNotNone(model.assignments)

    def test_high_dimensional_points(self):
        """K-Means should work on data with more than 2 dimensions."""
        np.random.seed(0)
        points = np.random.rand(30, 5)
        model = KMeans(K=3)
        model.fit(points, n_init=3)
        self.assertEqual(len(model.assignments), 30)

    def test_large_k_many_runs(self):
        """K=5 with n_init=10 on a 50-point dataset should complete without error."""
        np.random.seed(0)
        points = np.random.rand(50, 2) * 20
        model = KMeans(K=5)
        model.fit(points, n_init=10, init='kmeans++')
        self.assertIsNotNone(model.sse)

    def test_kmeans_plusplus_k_equals_n_points(self):
        """K-Means++ init with K == N should return N distinct centroids."""
        points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model = KMeans(K=3)
        centroids = model._init_kmeans_plusplus(points, seed=0)
        self.assertEqual(len(centroids), 3)

    def test_fit_with_kmeans_plusplus_init(self):
        """fit() with init='kmeans++' should produce valid results."""
        np.random.seed(5)
        points = np.random.rand(40, 2)
        model = KMeans(K=3)
        model.fit(points, n_init=5, init='kmeans++')
        self.assertIsNotNone(model.centroids)

    def test_convergence_tolerance(self):
        """Tighter tolerance should still converge and produce valid results."""
        np.random.seed(0)
        points = np.random.rand(30, 2)
        model = KMeans(K=3)
        model.fit(points, n_init=5, tol=1e-8)
        self.assertIsNotNone(model.sse)

    def test_str_after_fit(self):
        """__str__ on a fitted model should contain SSE and cluster sizes."""
        np.random.seed(0)
        points = np.random.rand(20, 2)
        model = KMeans(K=2)
        model.fit(points, n_init=3)
        s = str(model)
        self.assertIn('SSE', s)
        self.assertIn('Cluster sizes', s)


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("K-Means Clustering CLI — Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
