import numpy as np
import open3d as o3d
from utils.read_frames import read_length_delimited_frames
from config import FRAME_FILE


def _normal_to_rotation_matrix(vec):
    vec = vec / np.linalg.norm(vec)
    if abs(np.dot(vec, [0, 1, 0])) < 0.999:
        up = np.array([0, 1, 0])
    else:
        up = np.array([1, 0, 0])

    x_axis = np.cross(up, vec)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(vec, x_axis)

    R = np.column_stack((x_axis, y_axis, vec))
    return R


def estimate_plane():
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        if frame_iterator == 0:  # always uses first frame as plane reference
            for scan_iterator, scan in enumerate(frame.lidars):
                if scan_iterator == 0:
                    pcl = scan.pointcloud.points
                    arr = np.array([[p.x, p.y, p.z] for p in pcl], dtype=np.float32)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(arr)
                    plane_model, inliers = pcd.segment_plane(
                        distance_threshold=0.1, ransac_n=3, num_iterations=1000
                    )
                    [a, b, c, d] = plane_model
                    normal_vector = np.array([a, b, c])
                    return _normal_to_rotation_matrix(normal_vector)
        else:
            continue
