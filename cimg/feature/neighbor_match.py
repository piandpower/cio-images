import numpy as np


def match_subpoints(reference_subpoints, unknown_subpoints, max_distance=0.2):
    """
    Compute score as abs(difference between closest matching vectors)
    If nearest match is beyond some max distance (normalized).  Match score
    is returned as the total possible matches, minus the number of matches,
    plus the mismatch sum between matches.
    """
    matrix_size = min(len(reference_subpoints), len(unknown_subpoints))
    differentials = np.zeros((matrix_size, matrix_size))
    for unknown in range(matrix_size):
        for reference in range(matrix_size):
            # create a matrix of distance differentials
            differentials[reference, unknown] = ((reference_subpoints[reference]-unknown_subpoints[unknown])**2).sum()
    # Loop over them, masking the smallest row/col each time, if smallest is below max_distance
    matches = []
    max_distance **= 2  # because differentials do not do sqrt
    for match in range(matrix_size):
        if np.min(differentials) < max_distance:
            min_index = np.argmin(differentials)
            row, col = np.unravel_index(min_index, differentials.shape)
            matches.append((row, col, differentials[row, col]))
            # mask differentials for this found result, so that points are not
            #    matched multiple
            differentials[row] = 1000
            differentials[:, col] = 1000
    return matrix_size - len(matches) + np.array(matches)[:,2].sum() if matches else matrix_size


def build_match_matrix(reference_points, unknown_points):
    """
    Builds a matrix of the match scores between each point in each image.
    """
    rows = reference_points.shape[1]/2
    cols = unknown_points.shape[1]/2
    match_matrix = np.zeros((rows, cols))
    for row in range(rows):
        for col in range(cols):
            match_matrix[row, col] = match_subpoints(
                reference_points[:, 2*row:2*row+2],
                unknown_points[:, 2*col:2*col+2]
            )
    return match_matrix


def find_matches(match_matrix, reference_kps, unknown_kps):
    """
    identifies matching points based on match matrix.

    Establishes association between points in each image space.

    Returns list of points, with (point coordinates, match score) (in ref, then unknown order)
    """
    matches=[]
    for match in range(min(match_matrix.shape)):
        min_index = np.argmin(match_matrix)
        row, col = np.unravel_index(min_index, match_matrix.shape)
        matches.append((reference_kps[row].pt, unknown_kps[col].pt, match_matrix[row, col]))
        # mask differentials for this found result, so that points are not
        #    matched multiple
        match_matrix[row] = 1000
        match_matrix[:, col] = 1000
    return matches