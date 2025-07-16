# Creates Open3d window with camera controls, looking at a singular frame of the cleaned protobuf data including the global pointcloud and all detected objects as bounding boxes. Frame indecies start at 1 for the input
import os
import sys
import struct
import logging
import numpy as np
import open3d as o3d
from google.protobuf.message import DecodeError
from a42.frame_pb2 import Frame

FRAME_FILE = r"/users/emil/Documents/HS_Esslingen/Studienprojekt/Data/TestData1/all_frames.pb" # hardcoded custom path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# initialize o3d window
vis = o3d.visualization.Visualizer()
vis.create_window()

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

def create_lane_box():
    n = np.array([0.02212, -0.01383, 0.99966])
    R = normal_to_rotation_matrix(n)
    extent = np.array([3.3,90,3.5]) 
    center = np.array([0,0,1.88]) 
    cropbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    return cropbox

def is_point_in_box(point, box):
    relative_point = np.dot(point - box.center, box.R)
    max_extents = box.extent / 2.0
    return np.all(np.abs(relative_point) <= max_extents)

def pcl_to_pcd(pcl):
    arr = np.array([[p.x, p.y, p.z, p.intensity] for p in pcl], dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3])

    if arr.shape[1] > 3:
        intensities = arr[:, 3]
        colors = np.repeat(intensities[:, None], 3, axis=1)
        colors = (colors - colors.min()) / (np.ptp(colors) + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_pointcloud(pcl):
    pcd = pcl_to_pcd(pcl)
    cropped_pcd = pcd.crop(create_lane_box())
    vis.add_geometry(cropped_pcd) # can be changed to pcd to view the full scene

def visualize_object_bounding_box(pcl):
    pcd = pcl_to_pcd(pcl)
    boundingbox = pcd.get_axis_aligned_bounding_box()
    boundingbox.color = (1, 0, 0)
    vis.add_geometry(boundingbox)

def position_to_array(point):
    point_arr = np.array([point.x, point.y, point.z], dtype=np.float32)
    return point_arr

def main():

    if not os.path.exists(FRAME_FILE):
        log.error(f"Datei nicht gefunden: {FRAME_FILE}")
        return

    print(f"Object length analysis for {FRAME_FILE}")
    print(f"Enter frame to be analyzed")
    viewed_frame = int(input())-1 # index of input starts at 1
    
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        if frame_iterator == viewed_frame:
            print(f"Frame {frame_iterator+1}:")

            for scan_iterator, scan in enumerate(frame.lidars):
                if len(scan.pointcloud.points) == 0:
                    continue

                print(f"Scan {scan_iterator+1}:")
                timestamp_unix = scan.scan_timestamp_ns
                dt64 = np.datetime64(timestamp_unix, 'ns')
                print(f"Timestamp = {scan.scan_timestamp_ns}ns")
                print(f"Datetime = {dt64}")

                visualize_pointcloud(scan.pointcloud.points)

                for object_iterator, obj in enumerate(scan.object_list.objects):
                    if is_point_in_box(position_to_array(obj.position), create_lane_box()):
                        visualize_object_bounding_box(obj.pointcloud.points)
                        print(f"Objekt {object_iterator+1}:")
                        print(f"    x = {obj.dimension.x}m")
                        print(f"    y = {obj.dimension.y}m")
                        print(f"    z = {obj.dimension.z}m")
                    
                    else:
                        print(f"Objekt {object_iterator+1} nicht in der Lane-Box")

                vis.run()
                vis.destroy_window()
                return # only first scan is viewed

if __name__ == "__main__":
    main()




    