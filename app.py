"""
K-Means Clustering - Command Line Interface

A CLI tool for clustering data using K-Means algorithm.
Supports multiple initialization methods, feature scaling, and result export.

Author: Person 2
Date: February 2026
"""

import argparse
import sys
from src.kmeans import KMeans
from src.io_utils import load_points, scale_minmax, scale_zscore


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='K-Means Clustering CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --csv data/sample.csv --k 3
  python app.py --csv data.csv --k 5 --init kmeans++ --n_init 10
  python app.py --csv data.csv --k 3 --scale minmax --output results.csv
        """
    )
    
    # Required arguments
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to CSV file with data points')
    parser.add_argument('--k', type=int, required=True,
                       help='Number of clusters')
    
    # Optional arguments
    parser.add_argument('--max_iters', type=int, default=100,
                       help='Maximum iterations (default: 100)')
    parser.add_argument('--tol', type=float, default=1e-4,
                       help='Convergence tolerance (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--init', type=str, default='random',
                       choices=['random', 'kmeans++'],
                       help='Initialization method (default: random)')
    parser.add_argument('--n_init', type=int, default=10,
                       help='Number of runs (default: 10)')
    parser.add_argument('--scale', type=str, default=None,
                       choices=['minmax', 'zscore'],
                       help='Feature scaling method (optional)')
    parser.add_argument('--preview', type=int, default=10,
                       help='Number of assignments to display (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                       help='Export results to CSV file (optional)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed progress')
    
    return parser.parse_args()


def validate_inputs(args, num_points):
    """Validate user inputs."""
    # Validate K
    if args.k < 1:
        print("Error: K must be at least 1")
        sys.exit(1)
    
    if args.k > num_points:
        print(f"Error: K ({args.k}) cannot be greater than number of points ({num_points})")
        sys.exit(1)
    
    # Validate max_iters
    if args.max_iters < 1:
        print("Error: max_iters must be at least 1")
        sys.exit(1)
    
    # Validate tol
    if args.tol < 0:
        print("Error: tolerance must be non-negative")
        sys.exit(1)
    
    # Validate n_init
    if args.n_init < 1:
        print("Error: n_init must be at least 1")
        sys.exit(1)
    
    # Validate preview
    if args.preview < 0:
        print("Error: preview must be non-negative")
        sys.exit(1)


def print_header():
    """Print application header."""
    print("=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)


def print_results(model, points, args):
    """Print clustering results."""
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    # Summary
    print(f"\nDataset Information:")
    print(f"  Points: {len(points)}")
    print(f"  K (clusters): {args.k}")
    print(f"  Initialization: {args.init}")
    print(f"  Runs: {args.n_init}")
    
    print(f"\nConvergence:")
    print(f"  Iterations: {model.iterations}")
    print(f"  SSE (quality): {model.sse:.4f}")
    
    # Centroids
    print(f"\nCentroids:")
    sizes = model.get_cluster_sizes()
    for i, (centroid, size) in enumerate(zip(model.centroids, sizes)):
        coord_str = ", ".join(f"{c:.4f}" for c in centroid)
        pct = 100 * size / len(points)
        print(f"  C{i} = ({coord_str}) | size={size} ({pct:.1f}%)")
    
    # Preview assignments
    preview_count = min(args.preview, len(points))
    if preview_count > 0:
        print(f"\nAssignments (first {preview_count}):")
        for i in range(preview_count):
            point = points[i]
            cluster = model.assignments[i]
            coord_str = ", ".join(f"{p:.4f}" for p in point)
            print(f"  ({coord_str}) -> C{cluster}")
    
    print("\n" + "=" * 60)


def main():
    """Main application entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print_header()
    
    # Load data
    print(f"\nLoading data from '{args.csv}'...")
    try:
        points = load_points(args.csv)
        print(f"✓ Loaded {len(points)} points")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Validate inputs
    validate_inputs(args, len(points))
    
    # Scale features if requested
    if args.scale:
        print(f"\nScaling features using {args.scale}...")
        try:
            if args.scale == 'minmax':
                points, _ = scale_minmax(points)
            elif args.scale == 'zscore':
                points, _ = scale_zscore(points)
            print("✓ Features scaled")
        except Exception as e:
            print(f"Error scaling features: {e}")
            sys.exit(1)
    
    # Create model
    print(f"\nInitializing K-Means (K={args.k}, max_iters={args.max_iters})...")
    model = KMeans(K=args.k, max_iters=args.max_iters)
    
    # Run clustering
    print(f"Running K-Means clustering...\n")
    try:
        model.fit(
            points,
            n_init=args.n_init,
            init=args.init,
            tol=args.tol,
            verbose=args.verbose
        )
        print("\nClustering complete")
    except Exception as e:
        print(f"Error during clustering: {e}")
        sys.exit(1)
    
    # Print results
    print_results(model, points, args)
    
    # Export if requested
    if args.output:
        print(f"\nExporting results to '{args.output}'...")
        try:
            model.export_results(points, args.output)
            print("Export complete")
        except Exception as e:
            print(f"Error exporting results: {e}")
            sys.exit(1)
    
    print("\nDone!\n")


if __name__ == "__main__":
    main()