import numpy as np

def graphier_FPS(pts, K, ori_idx):
    """ returns "greedy" farthest points based on Euclidean Distances
    Code by Graipher from https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

    Args:
        pts ((N, 3), float): Sampled points
        K (int): the number of farthest sampled points
        ori_idx: array of original vertice indices

    Returns:
        farthest_pts ((K, 3), float): farthest sampled points
        idx : array of their indices

    """
    idx = []
    farthest_pts = np.zeros((K, 3))
    random_init = np.random.randint(pts.shape[0])
    farthest_pts[0] = pts[random_init]
    idx.append(ori_idx[random_init])
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        idx.append(ori_idx[np.argmax(distances)])
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, np.array(idx)

def calc_distances(p0, points):
    """calculate Euclidean distance between points
    Code by Graipher from https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python

    Args:
        p0 (float): anchor points
        points (float): neighbor pointspoints

    Returns:
        (float): Euclidean Distance(s)
    """
    """
    if len(p0.shape) == len(points.shape) == 1: # Distance point A <-> point B
        return ((p0 - points)**2).sum(axis = 0)
    elif len(p0.shape) == 1 and len(points.shape) == 2:
        return ((p0 - points)**2).sum(axis=1)
    """
    return np.sqrt(((p0 - points)**2).sum(axis = len(points.shape) - 1))    