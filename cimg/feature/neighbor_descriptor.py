"""
This module and its friend, neighbor_match, were an attempt to capture features as spatially-related points.
It captures nearest neighbor information, and attempts to normalize that data into scale and rotation invariant
information.
"""

import numpy as np


def collect_points(original_points, nearest_n, vectors=True):
    """Look up nearest n points, return array where each pair of columns
    is a collection of neighbor points from original point.

    vectors: if True, returns output as the difference of the
             neighbor point minus the reference point
    """
    # each column is the collection of distances from a point
    #    to its neighbors, minus the sqrt (for speed)
    distances = np.zeros((len(original_points), len(original_points)))
    pts = np.zeros((len(original_points), 2))
    for index, kp in enumerate(original_points):
        pts[index] = kp.pt
    for index, pt in enumerate(pts):
        diffs = pts-pt
        distances[:, index:index+1] = np.expand_dims(diffs[:, 0]**2 + diffs[:, 1]**2, 1)

    out = np.zeros((nearest_n, len(original_points)*2))
    for index, pt in enumerate(pts):
        indices = np.argpartition(distances[:,index], nearest_n+1)[:nearest_n+1]
        indices = indices[np.argsort(distances[indices, index])]
        out[:, 2*index:2*index+2] = pts[indices[1:]]
        if vectors:
            out[:, 2*index:2*index+2] -= pt
    return out


def normalize_vectors(point_distance_vectors):
    """
    normalizing by the magnitude of the longest vector
    to achieve relative scale (and hopefully scale invariance)
    """
    output = point_distance_vectors.copy()
    for point in range(point_distance_vectors.shape[1]/2):
        max_length = np.sqrt((point_distance_vectors[:, 2*point:2*point+2]**2).sum(axis=0)).max()
        output[:, 2*point:2*point+2] /= max_length
    return output


def project_vectors(normalized_vectors):
    """
    Establish new coordinate system for point set, and express vectors in that coordinate system.  This step is meant
    to confer rotation invariance.

    Principal axis is mean of all vectors.


    perpendicular vector is taken as:

    x2 = -y1 * y2 / x1
        with y2 assumed to be 1 (arbitrarily for scaling purposes.)
    thus
    x2 = -y1/x1
    y2 = 1

    Finally, the transformation matrix is normalized to unit length, so that the input vector lengths do not change.
    """
    projected_data = normalized_vectors.copy()
    directions = np.array([np.mean(normalized_vectors[:, 2*pt:2*pt+2], axis=0)
                                 for pt in range(normalized_vectors.shape[1]/2)])
    for index, direction in enumerate(directions):
        projector = np.array([[1, -direction[0]/direction[1]], direction])
        projector[0] /= np.linalg.norm(projector[0])
        projector[1] /= np.linalg.norm(projector[1])
        projected_data[:, 2*index:2*index+2] = np.dot(projected_data[:, 2*index:2*index+2],
                                                      projector)
    return projected_data


