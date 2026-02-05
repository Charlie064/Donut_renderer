import os
import numpy as np

def load_object_vertices(file_name):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, file_name)

    vertices = []

    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
        
        points = np.asarray(vertices, dtype=float)
        return points
    

def subsample_points(points, max_points=10000):
    if len(points) <= max_points:
        return points
    step = len(points) // max_points
    return points[::step]


def validated_points(file_name, max_points=10000):
    all_points = load_object_vertices(file_name)
    subsampled_points = subsample_points(all_points, max_points)
    return subsampled_points



if __name__ == "__main__":
    print(validated_points(file_name="skull.obj"))
