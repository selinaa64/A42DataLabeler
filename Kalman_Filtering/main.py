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

# Constants (should be dynamically changable, hardcoded right now)
LANE_BOX_DIMS = np.array([3.3, 90, 3.5])
LANE_BOX_POS = np.array([0, 0, 1.88])
Y_WIM_DEFAULT = 10.67
DISTANCE_THRESH = 0.5
Q = (5 / 3.6)**2  
R = 0.2**2       
P_INIT = np.array([[5**2, 0],
                   [0, (10 / 3.6)**2]])

def load_lidar_data():
    """
    Reads frames from LiDAR data and extracts front positions (Y) of objects
    filtered to the relevant lane.
    
    Returns:
        List of lists: each sublist contains Y positions for objects detected in that frame.
    """
    lane_box = create_lane_box(LANE_BOX_DIMS, LANE_BOX_POS)
    lidar_data = []

    for frame_idx, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        y_positions = []
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
                    front_y = aabb.get_min_bound()[1]
                    y_positions.append(front_y)

        lidar_data.append(y_positions)
    return lidar_data

def kalman_track(lidar_data, x_init, P_init, start_frame, direction, N, T, Q, R, distance_thresh):
    """
    Generic Kalman tracking function for forward or backward tracking.
    
    Args:
        lidar_data (list): List of observations per frame.
        x_init (np.array): Initial state vector (2x1).
        P_init (np.array): Initial covariance matrix (2x2).
        start_frame (int): Frame index to start tracking.
        direction (str): 'forward' or 'backward'.
        N (int): Total number of frames.
        T (float): Time step size.
        Q (float): Process noise covariance.
        R (float): Measurement noise covariance.
        distance_thresh (float): Maximum acceptable distance for association.
        
    Returns:
        Tuple of lists: filtered positions and velocities.
    """
    if direction == 'forward':
        Ad = np.array([[1, T], [0, 1]])
        Gd = np.array([[T], [1]])
        frame_range = range(start_frame + 1, N)
    elif direction == 'backward':
        Ad = np.array([[1, -T], [0, 1]])
        Gd = np.array([[-T], [1]])
        frame_range = range(start_frame - 1, -1, -1)
    else:
        raise ValueError("Direction must be 'forward' or 'backward'")

    C = np.array([[1, 0]])
    x = x_init.copy()
    P = P_init.copy()

    y_filtered = []
    v_filtered = []

    for k in frame_range:
        observations = lidar_data[k]
        if not observations:
            logging.info(f"No LiDAR observations at timestep {k} ({direction})")
            break

        diffs = np.abs(np.array(observations) - x[0, 0])
        closest_idx = np.argmin(diffs)
        min_dist = diffs[closest_idx]

        obs = np.array([[observations[closest_idx]]]) if min_dist < distance_thresh else None

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

        # stops tracking if prediction past optimal sensor range is made
        if (direction == 'forward' and x[0, 0] < 0.1) or (direction == 'backward' and x[0, 0] > 25):
            logging.info(f"Tracking stopped at timestep {k}, predicted position {x[0, 0]:.3f} ({direction})")
            break

        y_filtered.append(x_tilde[0, 0])
        v_filtered.append(x_tilde[1, 0])

    return y_filtered, v_filtered

def export_to_csv(filename, y_forward, v_forward, y_backward, v_backward, start_frame):
    """
    Exports filtered tracking results to a CSV file.
    """
    frames_forward = list(range(start_frame, start_frame + len(y_forward)))
    frames_backward = list(range(start_frame - len(y_backward) + 1, start_frame + 1))[::-1]

    data = {
        'frame': frames_forward + frames_backward,
        'position': y_forward + y_backward[::-1],
        'velocity': v_forward + v_backward[::-1],
        'direction': ['forward'] * len(y_forward) + ['backward'] * len(y_backward)
    }

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logging.info(f"Exported filtered data to {filename}")

def visualize(lidar_data, N, kalman_tracks):
    """
    Visualizes raw observations and Kalman filter tracking.
    
    Args:
        lidar_data (list): Raw observed y-positions per frame.
        N (int): Total number of frames.
        kalman_tracks (list): List of tuples containing (start_frame, y_forward, y_backward).
    """
    plt.figure(figsize=(14, 7))
    
    # raw lidar data from dbscan
    for k in range(N):
        obs = lidar_data[k]
        for pos in obs:
            plt.plot(k, pos, 'bo', alpha=0.3)

    # backward and forwad Kalman tracking
    for (start_frame, y_forward, y_backward) in kalman_tracks:
        plt.plot(range(start_frame, start_frame + len(y_forward)), y_forward, 'rx-')
        backward_frames = range(start_frame + 1 - len(y_backward), start_frame + 1)
        plt.plot(backward_frames, y_backward[::-1], 'gx-')

    # manual legend
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
        if not lidar_data[k_WIM]:
            k_WIM += 1
            if k_WIM >= N:
                logging.warning(f"No LiDAR data available near WIM detection frame {k_WIM}. Skipping...")
                continue

        # Find closest object in detection frame
        if lidar_data[k_WIM]:
            distances = np.abs(np.array(lidar_data[k_WIM]) - y_WIM)
            y_WIM = lidar_data[k_WIM][np.argmin(distances)]

        x_init = np.array([[y_WIM], [v_WIM]])

        y_forward, v_forward = kalman_track(lidar_data, x_init, P_INIT, k_WIM, 'forward', N, T, Q, R, DISTANCE_THRESH)
        y_backward, v_backward = kalman_track(lidar_data, x_init, P_INIT, k_WIM, 'backward', N, T, Q, R, DISTANCE_THRESH)

        kalman_tracks.append((k_WIM, y_forward, y_backward))

        # csv_filename = f'tracking_output_vehicle_{k_WIM}.csv'
        # export_to_csv(csv_filename, y_forward, v_forward, y_backward, v_backward, k_WIM)

    visualize(lidar_data, N, kalman_tracks)

if __name__ == "__main__":
    main()
