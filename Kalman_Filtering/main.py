from sequence_analysis import get_period
from sequence_analysis import get_sequence_length
from utils import FRAME_FILE
from utils import read_length_delimited_frames
from utils import create_lane_box
from utils import is_point_in_box
from wim_correspondence import get_wim_correspondence

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open3d as o3d

def main():
    '''[x] read lidar frames and extract period between frames'''
    '''[x] lidar/WIM correspondence (object_id + detection frame + speed)'''
    '''[x] kalman (prediction + tracking) with starting values from corresponence + lidar period'''
    '''[] csv as output'''
    '''[] (optional) show raw lidar data with color coded bounding boxes for each object'''

    T = get_period() # for kalman step size
    N = get_sequence_length() # for tracking loop
    wim_data = get_wim_correspondence(r"/Users/emil/Documents/HS_Esslingen/Studienprojekt/Data/TestData1/data-1752218465752.csv") 
    
    print(f"Total frames: {N}")
    print(f"(ideal) Time between frames: {T}")


    # lidar_data (cleaned to only show relevant lane) filled with every measured object (front middle pos)
    lidar_data = []
    
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        y_positions = []
        for scan_iterator, scan in enumerate(frame.lidars):
            for object_iterator, obj in enumerate(scan.object_list.objects):
                # Convert raw points to Open3D point cloud
                raw_points = [[p.x, p.y, p.z] for p in scan.pointcloud.points]
                lane_box = create_lane_box(np.array([3.3, 90, 3.5]), np.array([0, 0, 1.88]))
                            
                    # Filter raw points by box
                filtered_points = [p for p in raw_points if is_point_in_box(np.array(p), lane_box)]
                if len(filtered_points) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(filtered_points)

                    labels = np.array(pcd.cluster_dbscan(eps=3, min_points=5, print_progress=False))
                    max_label = labels.max()

                    for i in range(max_label + 1):
                        cluster_indices = np.where(labels == i)[0]
                        cluster_points = np.asarray(pcd.points)[cluster_indices]

                        if cluster_points.shape[0] == 0:
                            continue

                        cluster_pcd = o3d.geometry.PointCloud()
                        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

                        aabb = cluster_pcd.get_axis_aligned_bounding_box()

                        center = aabb.get_center() 
                        extent = aabb.get_extent()  

                        '''todo: store these'''
                        width_x = extent[0]
                        length_y = extent[1]
                        height_z = extent[2]

                        front_y = aabb.get_min_bound()[1]
                        y_positions.append(front_y)

        lidar_data.append(y_positions)
                

    # visualization of the cleaned datapoints
    plt.figure()
    for k in range(N):
        obs = lidar_data[k]
        for j in range(len(obs)):
            plt.plot(k, obs[j], 'bo')

    # Kalman filter setup
    Ad = np.array([[1, T],
                [0, 1]])
    Ad_rev = np.array([[1, -T], 
                    [0, 1]])

    Gd = np.array([[T],
                [1]])
    Gd_rev = np.array([[-T],
                    [1]])

    Q = (5 / 3.6)**2 # to be changed

    C = np.array([[1, 0]])

    R = 0.2**2 # to be changed

    distance_thresh = 0.5 # to be changed

    # initial state
    P_dach = np.array([[5**2, 0],
                    [0, (10 / 3.6)**2]])

    for row_iterator, row in wim_data.iterrows():
        k_WIM = int(row['detection_frame']) 
        v_WIM = -row['speed_in_kmh'] / 3.6  # negative since objects are moving towards us
        y_WIM = 10.67 # assumed default for object passing the wim is the position of the wim itself

        # maximum correction by one frame due to possible offset in data
        if not lidar_data[k_WIM]:
            k_WIM += 1

        # closest object to the wim as starting position
        if lidar_data[k_WIM]:
            distances_from_wim = np.abs(np.array(lidar_data[k_WIM])-y_WIM)
            y_WIM = lidar_data[k_WIM][np.argmin(distances_from_wim)]


        x_dach = np.array([[y_WIM],
                            [v_WIM]])
                            
        y_filtered = []
        v_filtered = []
        y_filtered_backward = []
        v_filtered_backward = []
        ### ---- Reverse Tracking ---- ###
        x_dach_b = x_dach.copy()
        P_dach_b = P_dach.copy()

        for k in range(k_WIM - 1, -1, -1):
            observations = lidar_data[k]
            if not observations:
                print(f"No LiDAR observations at timestep {k} (backward)")
                obs = None
                break

            diffs = np.abs(np.array(observations) - x_dach_b[0, 0])
            closest_index = np.argmin(diffs)
            min_value = diffs[closest_index]

            obs = np.array([[observations[closest_index]]]) if min_value < distance_thresh else None

            # Correction
            if obs is not None:
                S = C @ P_dach_b @ C.T + R
                K = P_dach_b @ C.T @ np.linalg.pinv(S)
                x_tilde = x_dach_b + K @ (obs - C @ x_dach_b)
                P_tilde = (np.eye(2) - K @ C) @ P_dach_b
            else:
                x_tilde = x_dach_b
                P_tilde = P_dach_b

            # Prediction (backward)
            x_dach_b = Ad_rev @ x_tilde
            P_dach_b = Ad_rev @ P_tilde @ Ad_rev.T + Gd_rev @ Gd_rev.T * Q

            if x_dach_b[0, 0] > 25:
                print(f"Tracking stopped at time step {k}, predicted position {x_dach[0, 0]:.3f} below threshold.")
                break

            y_filtered_backward.append(x_tilde[0, 0])
            v_filtered_backward.append(x_tilde[1, 0])

        for k in range(k_WIM, N):
            # Data association
            observations = lidar_data[k]
            if not observations:
                print(f"No LiDAR observations at time step {k}")
                obs = None  # No objects detected at this timestep
                break
            else:
                diffs = np.abs(np.array(observations) - x_dach[0, 0])
                closest_index = np.argmin(diffs)
                min_value = diffs[closest_index]

            if min_value > distance_thresh:
                obs = None  # too far, discard
            else:
                obs = np.array([[observations[closest_index]]])

            # Correction
            if obs is not None:
                S = C @ P_dach @ C.T + R
                K = P_dach @ C.T @ np.linalg.pinv(S)
                x_tilde = x_dach + K @ (obs - C @ x_dach)
                P_tilde = (np.eye(2) - K @ C) @ P_dach
            else:
                x_tilde = x_dach
                P_tilde = P_dach

            # Prediction
            x_dach = Ad @ x_tilde
            P_dach = Ad @ P_tilde @ Ad.T + Gd @ Gd.T * Q

            if x_dach[0, 0] < 0.1:
                print(f"Tracking stopped at time step {k}, predicted position {x_dach[0, 0]:.3f} below threshold.")
                break
            
            # Logging
            y_filtered.append(x_tilde[0, 0])
            v_filtered.append(x_tilde[1, 0])

    # Plot filtered results
        plt.plot(np.arange(k_WIM, k_WIM + len(y_filtered)), y_filtered, 'rx-')
        backward_timesteps = np.arange(k_WIM - len(y_filtered_backward) + 1, k_WIM+1)
        plt.plot(backward_timesteps, y_filtered_backward[::-1], 'gx-')  # green line for backward tracking   
    plt.grid(True)
    plt.title('Kalman Filter Tracking')
    plt.xlabel('Time step')
    plt.ylabel('Position in m')
    plt.show()

if __name__ == "__main__":
    main()