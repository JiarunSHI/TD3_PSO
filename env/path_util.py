import numpy as np

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def collision(point, obstacles):
    for ox, oy, r in obstacles:
        if distance(point, np.array([ox, oy])) < r:
            return True
    return False

def path_length(path):
    return sum(distance(path[i], path[i+1]) for i in range(len(path)-1))

def min_obstacle_dist(point, obstacles):
    dists = [distance(point, np.array([ox, oy])) - r for ox, oy, r in obstacles]
    return min(dists)

