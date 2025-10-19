import numpy as np
import open3d as o3d
from utils.read_frames import read_length_delimited_frames
from config import FRAME_FILE_NEW, FRAME_FILE_OLD


def _normal_to_rotation_matrix(vec):  # coordinate transformation
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


def pointcloud_bytes_to_xyz(cartesian_bytes: bytes) -> np.ndarray:
    # Jede Koordinate ist float32 -> 4 Bytes; pro Punkt 3 Koordinaten -> 12 Bytes
    if len(cartesian_bytes) % 12 != 0:
        raise ValueError(f"Byte-Länge ({len(cartesian_bytes)}) ist kein Vielfaches von 12 – Format passt nicht.")
    arr = np.frombuffer(cartesian_bytes, dtype='<f4')  # little-endian float32
    return arr.reshape(-1, 3) 
def estimate_plane():
   # for FRAME_FILE in [FRAME_FILE_OLD, FRAME_FILE_NEW]:
        for frame_iterator, frame in enumerate(
            read_length_delimited_frames(FRAME_FILE_NEW)
        ):  # standard plane estimation from open3d documentation
 
            if frame_iterator == 0:  # always uses first frame as plane reference
                for scan_iterator, scan in enumerate(frame.lidars):
                    if scan_iterator == 0:
                        cart = scan.pointcloud.cartesian  # bytes
                        arr = pointcloud_bytes_to_xyz(cart) 
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
