import os
import sys
import struct
import time
import logging
import numpy as np
import open3d as o3d
from google.protobuf.message import DecodeError
from a42.frame_pb2 import Frame

FRAME_FILE = r"C:/Users/selin/Documents/studium/forschungsprojekt/data/with_objects_new.pb" # hardcoded custom path, change to where the protobuf file is strored

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

vis = o3d.visualization.Visualizer()
vis.create_window(width=1920, height=1080)
view_ctl = vis.get_view_control()
parameters = o3d.io.read_pinhole_camera_parameters("/users/emil/Documents/HS_Esslingen/Studienprojekt/Visualization_Testing/Frame_anim/ScreenCamera_2025-07-16-12-27-20.json") # hardcoded path to the JSON of the camera parameters

def read_length_delimited_frames(path):
    """Liest length-delimited protobuf Frames aus Datei."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                return
            size = struct.unpack("<I", hdr)[0]
            blob = f.read(size)
            if len(blob) < size:
                log.warning(f"{path}: erwartete {size} Bytes, nur {len(blob)} gelesen")
                return
            frame = Frame()
            try:
                frame.ParseFromString(blob)
            except DecodeError as e:
                log.error(f"{path}: Parse-Fehler: {e}")
                return
            yield frame

def create_lane_box():
    n = np.array([0.02212, -0.01383, 0.99966]) # hardcoded rotation values for
    R = normal_to_rotation_matrix(n)
    extent = np.array([3.3,90,3.5]) 
    center = np.array([0,0,1.88]) 
    cropbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    return cropbox

def pcl_to_pcd(pcl):
    arr = np.array([[p.x, p.y, p.z, p.intensity] for p in pcl], dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3])

    if arr.shape[1] > 3:
        intensities = arr[:, 3]
        colors = np.repeat(intensities[:, None], 3, axis=1)
        colors = (colors - colors.min()) / (np.ptp(colors) + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd = pcd.crop(create_lane_box())
    return pcd

def visualize_pointcloud(pcl):
    pcd = pcl_to_pcd(pcl)
    vis.add_geometry(pcd)
    return pcd

def normal_to_rotation_matrix(vec):
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

def is_point_in_box(point, box):
    relative_point = np.dot(point - box.center, box.R)
    max_extents = box.extent / 2.0
    return np.all(np.abs(relative_point) <= max_extents)

def position_to_array(point):
    point_arr = np.array([point.x, point.y, point.z], dtype=np.float32)
    return point_arr

def main():

    if not os.path.exists(FRAME_FILE):
        log.error(f"Datei nicht gefunden: {FRAME_FILE}")
        return
    
    print(f"Animation for {FRAME_FILE}")

    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        print(f"Frame: {frame_iterator + 1}")
        vis.clear_geometries()

        for scan_iterator, scan in enumerate(frame.lidars):
                    
            if len(scan.pointcloud.points) == 0:
                continue

            pcd = visualize_pointcloud(scan.pointcloud.points)   

            if len(pcd.points):

                labels = np.array(pcd.cluster_dbscan(eps=3, min_points=5, print_progress=False))
                max_label = labels.max()

                for cluster_id in range(max_label +1):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    cluster_points = np.asarray(pcd.points)[cluster_indices]
                    if cluster_points.shape[0] == 0:
                        continue
                    cluster_pcd = o3d.geometry.PointCloud()
                    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                    aabb = cluster_pcd.get_axis_aligned_bounding_box()
                    aabb.color = (1, 0, 0)
                    vis.add_geometry(aabb)

        view_ctl.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

    vis.destroy_window()

if __name__ == "__main__":
    main()