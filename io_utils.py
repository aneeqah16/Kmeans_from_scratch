import csv
import random
import os


def generate_sample_data():
    """Generate random 2D points with 2 clusters."""
    points = []

    # Cluster 1: around (2, 2)
    for _ in range(10):
        x = random.uniform(1.0, 3.0)
        y = random.uniform(1.0, 3.0)
        points.append((x, y))

    # Cluster 2: around (8, 8)
    for _ in range(10):
        x = random.uniform(7.0, 9.0)
        y = random.uniform(7.0, 9.0)
        points.append((x, y))

    return points


def save_points_to_csv(points, csv_path):
    """Save points to CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in points:
            writer.writerow([x, y])


def load_points(csv_path):
    """Load points from CSV file.
    If file not found, generate and save sample data."""
    points = []

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                if row[0].lower() == "x":
                    continue
                try:
                    x = float(row[0])
                    y = float(row[1])
                    points.append((x, y))
                except ValueError:
                    print(f"⚠ Skipping invalid row: {row}")

    except FileNotFoundError:
        print(f"File '{csv_path}' not found. Generating sample data...")
        points = generate_sample_data()
        save_points_to_csv(points, csv_path)

    return points


# Test
if __name__ == "__main__":
    points = load_points("data/sample.csv")
    print(f"Loaded {len(points)} points:")
    for p in points[:5]:
        print(p)