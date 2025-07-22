import numpy as np

def is_point_in_box(point, box):
    relative_point = np.dot(point - box.center, box.R)
    max_extents = box.extent / 2.0
    return np.all(np.abs(relative_point) <= max_extents)