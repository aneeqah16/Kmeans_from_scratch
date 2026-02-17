"""
Data loading and preprocessing utilities for K-means clustering.

Handles CSV loading, sample data generation , feature scaling.

Author : Abrar Ahmad
Date: Feburary 2026
"""
import csv
import random
import os
import numpy as np

#===================================================================
#                        DATA GENERATION
#===================================================================
def generate_sample_data():
    """Generate random 2D points with two clusters."""
    points = []

    # Cluster 1 around (2,2)
    for _ in range(10):
        x = random.uniform(1.0, 3.0)
        y = random.uniform(1.0, 3.0)
        points.append((x, y))

    # Cluster 2 around (8,8)
    for _ in range(10):
        x = random.uniform(7.0, 9.0)
        y = random.uniform(7.0, 9.0)
        points.append((x, y))

    return points
# ==================================================================
#                      FILE HANDLING
#===================================================================

def save_points_to_csv(points, csv_path):
    """Save points to CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in points:
            writer.writerow([x, y])


def load_points(csv_path):
    """
    Load points from CSV file.
    If file not found, generate and save sample data.
    """
    points = []

    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)

            for row in reader:
                if len(row) < 2:
                    continue

                # Skip header
                if row[0].lower() == "x" and row[1].lower() == "y":
                    continue

                try:
                    x = float(row[0])
                    y = float(row[1])
                    points.append((x, y))
                except ValueError:
                    print(f"Warning: Skipping invalid row: {row}")

        return points

    except FileNotFoundError:
        print(f"File '{csv_path}' not found. Generating sample data...")
        points = generate_sample_data()
        save_points_to_csv(points, csv_path)
        return points
#====================================================================
#                         FEATURE SCALING
#====================================================================

  
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
#===================================================================
#                        TEST
#===================================================================
if __name__ == "__main__":
    points = load_points("data/sample.csv")
    print(f"Loaded {len(points)} points:")
    for p in points[:5]:
        print(p)