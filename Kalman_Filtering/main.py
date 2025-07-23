import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import open3d as o3d
import logging

from sequence_analysis import get_period, get_sequence_length
from utils import FRAME_FILE, read_length_delimited_frames, create_lane_box, is_point_in_box
from wim_correspondence import get_wim_correspondence

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
LANE_BOX_DIMS = np.array([3.3, 90, 3.5])
LANE_BOX_POS = np.array([0, 0, 1.88])
Y_WIM_DEFAULT = 10.67
DISTANCE_THRESH = 3
Q = (1 / 3.6)**2
R = 0.2**2
P_INIT = np.array([[5**2, 0],
                   [0, (10 / 3.6)**2]])

def load_lidar_data():
    """
    Reads frames from LiDAR data and extracts clusters with bounding boxes.
    Returns:
        List of dicts: each frame has a timestamp and cluster list.
    """
    lane_box = create_lane_box(LANE_BOX_DIMS, LANE_BOX_POS)
    lidar_data = []

    for frame_idx, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        frame_data = {
            "timestamp_ns": frame.frame_timestamp_ns,
            "clusters": []
        }
        for scan in frame.lidars:
            raw_points = np.array([[p.x, p.y, p.z] for p in scan.pointcloud.points])
            filtered_points = [p for p in raw_points if is_point_in_box(np.array(p), lane_box)]
            
            if filtered_points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(filtered_points)
                labels = np.array(pcd.cluster_dbscan(eps=3, min_points=5, print_progress=False))
                max_label = labels.max()

                for cluster_id in range(max_label + 1):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    cluster_points = np.asarray(pcd.points)[cluster_indices]
                    if cluster_points.shape[0] == 0:
                        continue
                    cluster_pcd = o3d.geometry.PointCloud()
                    cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                    aabb = cluster_pcd.get_axis_aligned_bounding_box()
                    frame_data["clusters"].append({
                        "front_x": aabb.get_min_bound()[0],
                        "front_y": aabb.get_min_bound()[1],
                        "front_z": aabb.get_min_bound()[2],
                        "extent_x": aabb.get_extent()[0],
                        "extent_y": aabb.get_extent()[1],
                        "extent_z": aabb.get_extent()[2]
                    })

        lidar_data.append(frame_data)
    return lidar_data

def kalman_track(lidar_data, x_init, P_init, start_frame, direction, N, T, Q, R, distance_thresh):
    """
    Kalman tracking with cluster association.
    Returns positions, velocities, and chosen cluster info per frame.
    """
    if direction == 'forward':
        Ad = np.array([[1, T], [0, 1]])
        Gd = np.array([[T], [1]])
        frame_range = range(start_frame, N)
    elif direction == 'backward':
        Ad = np.array([[1, -T], [0, 1]])
        Gd = np.array([[-T], [1]])
        frame_range = range(start_frame, 0, -1)
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")

    C = np.array([[1, 0]])
    x = x_init.copy()
    P = P_init.copy()

    y_filtered, v_filtered, cluster_info = [], [], []

    for k in frame_range:
        clusters = lidar_data[k].get("clusters", [])
        if not clusters:
            logging.info(f"No LiDAR observations at timestep {k} ({direction})")
            break

        cluster_ys = [c["front_y"] for c in clusters]
        diffs = np.abs(np.array(cluster_ys) - x[0, 0])
        closest_idx = np.argmin(diffs)
        min_dist = diffs[closest_idx]

        obs = np.array([[cluster_ys[closest_idx]]]) if min_dist < distance_thresh else None
        chosen_cluster = clusters[closest_idx] if obs is not None else None

        # correction
        if obs is not None:
            S = C @ P @ C.T + R
            K = P @ C.T @ np.linalg.pinv(S)
            x_tilde = x + K @ (obs - C @ x)
            P_tilde = (np.eye(2) - K @ C) @ P
        else:
            x_tilde = x
            P_tilde = P

        # prediction
        x = Ad @ x_tilde
        P = Ad @ P_tilde @ Ad.T + Gd @ Gd.T * Q

        # stop tracking out of range
        if (direction == 'forward' and x[0, 0] < 0.1) or (direction == 'backward' and x[0, 0] > 25):
            logging.info(f"Tracking stopped at timestep {k}, predicted position {x[0, 0]:.3f} ({direction})")
            break

        y_filtered.append(x_tilde[0, 0])
        v_filtered.append(abs(x_tilde[1, 0]))  # absolute velocity
        cluster_info.append((k, chosen_cluster))

    return y_filtered, v_filtered, cluster_info

def export_track_to_rows(kalman_tracks, lidar_data, frame_file, object_id_start=1):
    """
    Convert tracked objects into CSV-friendly rows.
    """
    rows = []
    obj_id = object_id_start

    for (start_frame, y_forward, v_forward, forward_info,
         y_backward, v_backward, backward_info, wim_row) in kalman_tracks:
        full_info = backward_info[::-1] + forward_info
        full_velocities = [v for v in v_backward[::-1]] + [v for v in v_forward]

        max_length = max((c["extent_y"] for _, c in full_info if c), default=None)
        max_width = max((c["extent_x"] for _, c in full_info if c), default=None)

        for (frame_idx, cluster), vel in zip(full_info, full_velocities):
            if cluster is None:
                continue
            timestamp = lidar_data[frame_idx]["timestamp_ns"]
            rows.append({
                'object_id': obj_id,
                'frame': frame_idx,
                'timestamp_ns': timestamp,
                'front_x': cluster["front_x"],
                'front_y': cluster["front_y"],
                'front_z': cluster["front_z"],
                'bbox_extent_x': cluster["extent_x"],
                'bbox_extent_y': cluster["extent_y"],
                'bbox_extent_z': cluster["extent_z"],
                'max_length': max_length,
                'max_width': max_width,
                'axle_spaces_in_cm': wim_row['axle_spaces_in_cm'],
                'axle_weights_in_kg': wim_row['axle_weights_in_kg'],
                'total_weight_in_kg': wim_row['total_weight_in_kg'],
                'initial_velocity': wim_row['speed_in_kmh'],
                'velocity_in_ms': vel,
                'vehicle_class': wim_row['class_id_8p1'],
                'frame_file': frame_file
            })
        obj_id += 1
    return rows

def visualize(lidar_data, N, kalman_tracks):
    """
    Visualizes LiDAR clusters and Kalman filter tracks.
    """
    plt.figure(figsize=(14, 7))

    # Plot raw LiDAR clusters
    for k in range(N):
        for cluster in lidar_data[k].get("clusters", []):
            plt.plot(k, cluster["front_y"], 'bo', alpha=0.3)

    # Plot tracks
    for (start_frame, y_forward, v_forward, forward_info,
         y_backward, v_backward, backward_info, wim_row) in kalman_tracks:
        backward_frames = [fi for fi, c in backward_info[::-1] if c]
        backward_positions = [c["front_y"] for _, c in backward_info[::-1] if c]
        plt.plot(backward_frames, backward_positions, 'gx-')

        forward_frames = [fi for fi, c in forward_info if c]
        forward_positions = [c["front_y"] for _, c in forward_info if c]
        plt.plot(forward_frames, forward_positions, 'rx-')

    # Legend
    blue_dot = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', label='LiDAR datapoints', alpha=0.3)
    red_cross = mlines.Line2D([], [], color='red', marker='x', linestyle='-', label='Forward Kalman track')
    green_cross = mlines.Line2D([], [], color='green', marker='x', linestyle='-', label='Backward Kalman track')
    plt.legend(handles=[blue_dot, red_cross, green_cross])

    plt.grid(True)
    plt.xlabel('Frame')
    plt.ylabel('Position (m)')
    plt.title('LiDAR Observations and Kalman Filter Tracking')
    plt.show()

def main():
    logging.info("Loading LiDAR data...")
    lidar_data = load_lidar_data()

    T = get_period()
    N = get_sequence_length()
    wim_data = get_wim_correspondence(r"/Users/emil/Documents/HS_Esslingen/Studienprojekt/Data/TestData1/data-1752218465752.csv")
    
    logging.info(f"Total frames: {N}")
    logging.info(f"Time between frames: {T:.3f} s")

    kalman_tracks = []

    for _, row in wim_data.iterrows():
        k_WIM = int(row['detection_frame'])
        v_WIM = -row['speed_in_kmh'] / 3.6  # Convert km/h to m/s and negate for direction
        y_WIM = Y_WIM_DEFAULT

        # Correct for missing LiDAR data in starting frame
        if not lidar_data[k_WIM]["clusters"]:
            k_WIM += 1
            if k_WIM >= N:
                logging.warning(f"No LiDAR data available near WIM detection frame {k_WIM}. Skipping...")
                continue

        # Find closest object in detection frame
        if lidar_data[k_WIM]["clusters"]:
            distances = np.abs(np.array([c["front_y"] for c in lidar_data[k_WIM]["clusters"]]) - y_WIM)
            y_WIM = lidar_data[k_WIM]["clusters"][np.argmin(distances)]["front_y"]

        x_init = np.array([[y_WIM], [v_WIM]])

        y_forward, v_forward, forward_info = kalman_track(lidar_data, x_init, P_INIT, k_WIM, 'forward', N, T, Q, R, DISTANCE_THRESH)
        y_backward, v_backward, backward_info = kalman_track(lidar_data, x_init, P_INIT, k_WIM, 'backward', N, T, Q, R, DISTANCE_THRESH)

        kalman_tracks.append((k_WIM, y_forward, v_forward, forward_info,
                              y_backward, v_backward, backward_info, row))

    visualize(lidar_data, N, kalman_tracks)

    rows = export_track_to_rows(kalman_tracks, lidar_data, FRAME_FILE)
    pd.DataFrame(rows).to_csv("tracked_objects.csv", index=False)
    logging.info("Exported tracked objects to tracked_objects.csv")

if __name__ == "__main__":
    main()
