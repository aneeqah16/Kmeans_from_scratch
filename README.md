# K-Means Clustering CLI

A from-scratch implementation of the K-Means clustering algorithm in Python — no scikit-learn, no black-box libraries. Just NumPy and clean code.

---

## Features

- K-Means algorithm built from scratch using NumPy
- Two initialisation methods: **random** and **K-Means++**
- Runs multiple times and keeps the best result
- Min-max and z-score feature scaling
- Elbow method to help choose the right K
- Simple command-line interface
- Export results to CSV
- Jupyter notebook with 6 visualisation graphs

---

## Project Structure

```
project/
├── data/
│   └── sample.csv          # Default dataset (150 points, 3 clusters)
├── graphs/                 # Saved visualisation plots
├── src/
│   ├── kmeans.py           # Core KMeans class and algorithm
│   └── io_utils.py         # Data loading, saving, feature scaling
├── app.py                  # CLI entry point
├── requirements.txt
└── visualization.ipynb     # Elbow curve, cluster plots, etc.
```

---

## Installation

```bash
git clone https://github.com/your-username/kmeans-clustering-cli.git
cd kmeans-clustering-cli
pip install -r requirements.txt
```

---

## Usage

**Basic:**
```bash
python app.py --csv data/sample.csv --k 3
```

**With K-Means++ and scaling:**
```bash
python app.py --csv data/sample.csv --k 3 --init kmeans++ --scale minmax
```

**Save results to file:**
```bash
python app.py --csv data/sample.csv --k 3 --output results.csv
```

**Full options:**
```bash
python app.py --csv data/sample.csv --k 3 --init kmeans++ --n_init 20 \
  --scale zscore --max_iters 200 --tol 0.0001 --seed 42 \
  --preview 15 --output results.csv --verbose
```

---

## All Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--csv` | required | Path to input CSV file |
| `--k` | required | Number of clusters |
| `--init` | `random` | Initialisation method: `random` or `kmeans++` |
| `--n_init` | `10` | Number of times to run the algorithm |
| `--max_iters` | `100` | Maximum iterations per run |
| `--tol` | `0.0001` | Convergence tolerance |
| `--scale` | `None` | Feature scaling: `minmax` or `zscore` |
| `--output` | `None` | Path to save results as CSV |
| `--preview` | `10` | Number of assignments to display |
| `--verbose` | `False` | Show progress for each run |
| `--seed` | `None` | Random seed for reproducibility |

---

## Dataset

The default `data/sample.csv` was generated using `sklearn.make_blobs`:

```python
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=150, centers=3, n_features=2,
                  cluster_std=2.5, center_box=(-10, 10), random_state=42)
```

If the file is missing, a simple 20-point fallback dataset is generated automatically.

---

## How It Works

1. Load data from CSV
2. Optionally scale features
3. Initialise K centroids (random or K-Means++)
4. Assign each point to its nearest centroid
5. Move centroids to the mean of their assigned points
6. Repeat steps 4–5 until convergence
7. Run multiple times, keep the result with the lowest SSE

---

## Requirements

```
numpy
matplotlib
scikit-learn
jupyter
```

---

## Authors

| Name | Module |
|------|--------|
| Aneeqah Ashraf | `kmeans.py` — Core algorithm |
| Abrar Ahmad | `io_utils.py` — Data handling |
| Shaista Shafi | `app.py` — CLI interface |

---

## Acknowledgments

- [GeeksforGeeks — K-Means Clustering](https://www.geeksforgeeks.org/k-means-clustering-introduction/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn — K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
