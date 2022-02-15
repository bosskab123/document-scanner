import numpy as np


def reorder(points):
    '''
    :param points: 4 coordinates of a rectangular
    :return pointsNew: rectangular points rearranged in a way from left to right, top to bottom
    '''
    points = points.reshape((4, 2))
    pointsNew = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew
