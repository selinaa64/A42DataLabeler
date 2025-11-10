"""
Script to read Lidar- and WIM-data and track objects over the entire frame sequence of the Lidar-data with the help of the initial WIM-based states and a Kalman filter.
The priority datawise is being placed on the WIM => if there is no detection on the WIM, there will not be any tracking even if there are detected objects/clusters as tracking cannot occur without the initial states from the WIM.
First returns a plot of the frame sequence and then the CSV with the data of the objects.
Values regarding the Kalman filter and other constants can be changed in /config/config.py
!Important!: The csv only exports AFTER the plot, so make sure to close the window of the plot or comment out the visualization.
10 minutes of data take about 5 minutes on a m3 macbook with 24gb of ram, performance may vary.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import open3d as o3d
import logging

from sequence_analysis import get_period, get_sequence_length
from utils import read_length_delimited_frames, create_lane_box, is_point_in_box
from wim_correspondence import get_wim_correspondence
from config import (
    LANE_BOX_DIMS,
    LANE_BOX_POS,
    Y_WIM_DEFAULT,
    DISTANCE_THRESH,
    Q,
    R,
    P_INIT,
    WIM_FILE,
    FRAME_FILE_NEW,
    FRAME_FILE_OLD,
)
from comark.comark import load_comark_data, get_neccessary_comark_infos, cut_comark_data_to_lidar_date, cut_comark_data_to_lidar_time
import bisect
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from bisect import bisect_left
# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_lidar_data():
    """
    Iterates over Lidar frames and either directly extracts object positions+dimensions or uses DBSCAN to extract object positions+dimensions.
    Returns:
        List of dicts: Each frame has a timestamp and no/one/multiple objects/clusters with the associated information.
    """
    lane_box = create_lane_box(LANE_BOX_DIMS, LANE_BOX_POS)
    lidar_data = []
    for frame_idx, frame in enumerate(read_length_delimited_frames(FRAME_FILE_NEW)):
            if frame_idx==50: return lidar_data  # for testing purposes, limit to first 10 frames
            frame_data = {"timestamp_ns": frame.frame_timestamp_ns, "clusters": []}
            for scan in frame.lidars:
                raw_points = np.frombuffer(scan.pointcloud.cartesian, dtype="<f4").reshape(-1, 3)
                filtered_points = [
                    p
                    for p in raw_points
                    if is_point_in_box(
                        np.array(p), lane_box
                    )  # only points inside lanebox considered
                ]

                if len(scan.object_list) > 0:  # predetected objects (no demo possible right now since the only frame sequences with objects are in an older format or broken :( )
                    for object_idx, obj in enumerate(scan.object_list):
                        obj_center = np.array(
                            [obj.position.x, obj.position.y, obj.position.z],
                            dtype=np.float32,
                        )
                        if is_point_in_box(
                            obj_center, lane_box
                        ):  # only objects (center) inside lanebox considered
                            frame_data["clusters"].append(
                                {
                                    "front_x": obj.position.x,
                                    "front_y": obj.position.y - (obj.dimension.y / 2),
                                    "front_z": obj.position.z,
                                    "extent_x": obj.dimension.x,
                                    "extent_y": obj.dimension.y,
                                    "extent_z": obj.dimension.z,
                                }
                            )
                else:  # manual object detection (standard DBSCAN implementation from open3d docs)
                    if filtered_points:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filtered_points)
                        labels = np.array(
                            pcd.cluster_dbscan(eps=3, min_points=5, print_progress=False)
                        )
                        max_label = labels.max()

                        for cluster_id in range(max_label + 1):
                            cluster_indices = np.where(labels == cluster_id)[0]
                            cluster_points = np.asarray(pcd.points)[cluster_indices]
                            if cluster_points.shape[0] == 0:
                                continue
                            cluster_pcd = o3d.geometry.PointCloud()
                            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                            aabb = cluster_pcd.get_axis_aligned_bounding_box()
                            frame_data["clusters"].append(
                                {
                                    "front_x": aabb.get_min_bound()[0],
                                    "front_y": aabb.get_min_bound()[1],
                                    "front_z": aabb.get_min_bound()[2],
                                    "extent_x": aabb.get_extent()[0],
                                    "extent_y": aabb.get_extent()[1],
                                    "extent_z": aabb.get_extent()[2],
                                }
                            )

            lidar_data.append(frame_data)
    return lidar_data

def convert_unix_timestamp_to_ISO(lidar_data):
    """
    Changes the timestamps of the lidar data from nanoseconds to seconds.
    Returns:
        List of dicts: Each frame has a timestamp in seconds and no/one/multiple objects/clusters with the associated information.
    """
    for frame in lidar_data:
        ts_ns = frame["timestamp_ns"]
        dt = datetime.fromtimestamp(ts_ns / 1e9)
        frame["timestamp_s"] = dt.isoformat()
    return lidar_data

def merge_clusters(clusters, expected_length):
    """
    Checks if a cluster might have another cluster within the axle distance given by the WIM and concatenates the two clusters if so. (important for trucks with beds or trailers)
    Returns:
        List of merged axis-aligned bounding boxes.
    """
    if not clusters:
        return []
    clusters = sorted(clusters, key=lambda c: c["front_y"])
    merged = []
    used = set()

    for i, c in enumerate(clusters):
        if i in used:
            continue
        min_x = c["front_x"]
        max_x = c["front_x"] + c["extent_x"]
        min_y = c["front_y"]
        max_y = c["front_y"] + c["extent_y"]
        min_z = c["front_z"]
        max_z = c["front_z"] + c["extent_z"]

        for j, c2 in enumerate(clusters[i + 1 :], start=i + 1):
            if (
                c2["front_y"] - min_y < expected_length
            ):  # only merge if within expected length
                used.add(j)
                min_x = min(min_x, c2["front_x"])
                max_x = max(max_x, c2["front_x"] + c2["extent_x"])
                max_y = max(max_y, c2["front_y"] + c2["extent_y"])
                min_z = min(min_z, c2["front_z"])
                max_z = max(max_z, c2["front_z"] + c2["extent_z"])
            else:
                break

        merged.append(
            {
                "front_x": min_x,
                "front_y": min_y,
                "front_z": min_z,
                "extent_x": max_x - min_x,
                "extent_y": max_y - min_y,
                "extent_z": max_z - min_z,
            }
        )
    return merged


def kalman_track(
    lidar_data, x_init, P_init, start_frame, direction, N, T, Q, R, distance_thresh
):
    """
    Forward and backward tracking for clusters/objects
    Returns:
        Positions, velocities and closest observation at every timestep.
    """
    if direction == "forward":
        Ad = np.array([[1, T], [0, 1]])
        Gd = np.array([[T], [1]])
        frame_range = range(start_frame, N)
    elif direction == "backward":
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

        obs = (
            np.array([[cluster_ys[closest_idx]]])
            if min_dist < distance_thresh
            else None
        )
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

        # stop tracking if out of range
        if (direction == "forward" and x[0, 0] < 0.1) or (
            direction == "backward" and x[0, 0] > 25
        ):
            logging.info(
                f"Tracking stopped at timestep {k}, predicted position {x[0, 0]:.3f} ({direction})"
            )
            break

        y_filtered.append(x_tilde[0, 0])
        v_filtered.append(abs(x_tilde[1, 0]))  # absolute velocity in m/s
        cluster_info.append((k, chosen_cluster))

    return y_filtered, v_filtered, cluster_info


def export_track_to_rows(kalman_tracks, lidar_data, frame_file, object_id_start=1):
    """
    Convert tracked objects into CSV rows.
    """
    rows = []
    obj_id = object_id_start

    for (
        start_frame,
        y_forward,
        v_forward,
        forward_info,
        y_backward,
        v_backward,
        backward_info,
        wim_row,
    ) in kalman_tracks:
        full_info = (
            backward_info[::-1] + forward_info[1:]
        )  # first frame is a duplicate => forward track frame gets dropped
        full_velocities = v_backward[::-1] + v_forward[1:]

        max_length = max((c["extent_y"] for _, c in full_info if c), default=None)
        max_width = max((c["extent_x"] for _, c in full_info if c), default=None)

        for (frame_idx, cluster), vel in zip(full_info, full_velocities):
            if cluster is None:
                continue
            timestamp = lidar_data[frame_idx]["timestamp_ns"]
            rows.append(
                {
                    "object_id": obj_id,
                    "frame": frame_idx,
                    "utc_timestamp_ns": timestamp,
                    "front_x": cluster["front_x"],
                    "front_y": cluster["front_y"],
                    "front_z": cluster["front_z"],
                    "bbox_extent_x": cluster["extent_x"],
                    "bbox_extent_y": cluster["extent_y"],
                    "bbox_extent_z": cluster["extent_z"],
                    "max_length": max_length,
                    "max_width": max_width,
                    "axle_spaces_in_cm": wim_row["axle_spaces_in_cm"],
                    "axle_weights_in_kg": wim_row["axle_weights_in_kg"],
                    "total_weight_in_kg": wim_row["total_weight_in_kg"],
                    "initial_velocity": wim_row["speed_in_kmh"],
                    "velocity_in_ms": vel,
                    "vehicle_class": wim_row["class_id_8p1"],
                    "frame_file": frame_file,
                }
            )
        obj_id += 1
    return rows


def visualize(lidar_data, N, kalman_tracks):
    """
    Visualizes LiDAR clusters and Kalman filter tracks.
    """
    plt.figure(figsize=(14, 7))

    # plot raw Lidar clusters
    for k in range(N):
        for cluster in lidar_data[k].get("clusters", []):
            plt.plot(k, cluster["front_y"], "bo", alpha=0.3)

    # plot tracks
    for (
        start_frame,
        y_forward,
        v_forward,
        forward_info,
        y_backward,
        v_backward,
        backward_info,
        wim_row,
    ) in kalman_tracks:
        backward_frames = [fi for fi, c in backward_info[::-1] if c]
        backward_positions = [c["front_y"] for _, c in backward_info[::-1] if c]
        plt.plot(backward_frames, backward_positions, "gx-")

        forward_frames = [fi for fi, c in forward_info if c]
        forward_positions = [c["front_y"] for _, c in forward_info if c]
        plt.plot(forward_frames, forward_positions, "rx-")

    # manual legend
    blue_dot = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        linestyle="None",
        label="LiDAR datapoints",
        alpha=0.3,
    )
    red_cross = mlines.Line2D(
        [], [], color="red", marker="x", linestyle="-", label="Forward Kalman track"
    )
    green_cross = mlines.Line2D(
        [], [], color="green", marker="x", linestyle="-", label="Backward Kalman track"
    )
    plt.legend(handles=[blue_dot, red_cross, green_cross])

    plt.grid(True)
    plt.xlabel("Frame")
    plt.ylabel("Position (m)")
    plt.title("Lidar Observations and Kalman Filter Tracking")
    plt.show()

    
import datetime
import logging
from bisect import bisect_left
import datetime as dt
import bisect
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
from bisect import bisect_left
def assign_comark_labels(comark_data: dict, lidar_data: list, offset_ms: int = 200) -> list:
    """
    Assigns comark labels to lidar frames by finding the nearest matching timestamp.
    For each lidar frame, finds the comark entry where (comark_timestamp + 200ms) 
    is closest to the lidar timestamp.

    Args:
        comark_data: Dict with comark timestamps as keys and [label, timestamp] as values
        lidar_data: List of dicts with 'timestamp_s' field
        offset_ms: Offset to add to comark timestamps (default 200ms)

    Returns:
        Modified lidar_data with added 'object_label' field
    """
    if not comark_data or not lidar_data:
        return lidar_data

    # Convert comark entries to (timestamp + offset, label) pairs
    offset = timedelta(milliseconds=offset_ms)
    comark_times = []
    for _, (label, ts_str) in comark_data.items():
        try:
            ts = datetime.fromisoformat(ts_str)
            comark_times.append((ts + offset, label))
        except Exception:
            continue

    # Sort by timestamp for better readability of results
    comark_times.sort(key=lambda x: x[0])

    # Process each lidar frame
    for frame in lidar_data:
        if frame['clusters'] is None or len(frame['clusters']) == 0:
            frame["object_label"] = None
            continue
        try:
            lidar_ts = datetime.fromisoformat(frame["timestamp_s"])
            
            # Find closest comark timestamp
            best_diff = timedelta(days=1)  # Large initial value
            best_label = None
            
            for comark_ts, label in comark_times:
                diff = abs(lidar_ts - comark_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_label = label
            
            frame["object_label"] = best_label
            
        except Exception as e:
            logging.warning(f"Could not process lidar frame: {e}")
            frame["object_label"] = None

    return lidar_data

def get_report(lidar_data_with_labels: list, current_comark_data) -> None:
    from collections import Counter

    counts = Counter()

    for frame in lidar_data_with_labels:
        label = frame.get("object_label", None)
        # support single label or iterable of labels
        if isinstance(label, (list, tuple, set)):
            for l in set(label):
                counts[l] += 1
        else:
            counts[label] += 1

    logging.info(f"Unique labels: {len(counts)}")
    for lbl, cnt in counts.items():
        logging.info(f"Label '{lbl}': {cnt} frames")
    # --- CoMark counts ---
    comark_counts = Counter()
    if current_comark_data:
        for key, val in current_comark_data.items():
            #label = None
            # expect value is a list/tuple and label is at index 1 per spec
            # if isinstance(val, (list, tuple)):
            #     if len(val) > 1:
            #         label = val[0]
            #     elif len(val) == 1:
            #         label = val[0]
            # else:
            label = val[0]
            # if label is None:
            #     continue
            comark_counts[label] += 1

    logging.info(f"CoMark: Unique labels: {len(comark_counts)}")
    for lbl, cnt in comark_counts.items():
        logging.info(f"CoMark Label '{lbl}': {cnt} entries")
def save_to_protobuf(lidar_data_with_labels: list, output_file: str) -> None:
    # import lidar_data_pb2

    # lidar_dataset = lidar_data_pb2.LidarDataset()

    # for frame in lidar_data_with_labels:
    #     frame_msg = lidar_dataset.frames.add()
    #     frame_msg.timestamp_s = frame["timestamp_s"]
    #     if "object_label" in frame:
    #         frame_msg.object_label = str(frame["object_label"]) if frame["object_label"] is not None else ""
    #     for cluster in frame.get("clusters", []):
    #         cluster_msg = frame_msg.clusters.add()
    #         cluster_msg.front_x = cluster["front_x"]
    #         cluster_msg.front_y = cluster["front_y"]
    #         cluster_msg.front_z = cluster["front_z"]
    #         cluster_msg.extent_x = cluster["extent_x"]
    #         cluster_msg.extent_y = cluster["extent_y"]
    #         cluster_msg.extent_z = cluster["extent_z"]

    # with open(output_file, "wb") as f:
    #     f.write(lidar_dataset.SerializeToString())

    logging.info(f"Saved LiDAR data with labels to {output_file}")
def main():
    logging.info("Loading LiDAR data...")
    lidar_data = load_lidar_data()
    lidar_data=convert_unix_timestamp_to_ISO(lidar_data)
    T = get_period()
    N = get_sequence_length()
    logging.info("Loading Comark data...")

    comark_data = load_comark_data()  # function to load comark data if needed
    comark_data_dict=get_neccessary_comark_infos(comark_data)
    #print_comark_data_grouped_by_date(comark_data_dict)
    current_comark_data= cut_comark_data_to_lidar_date(comark_data_dict, lidar_data)
    current_comark_data= cut_comark_data_to_lidar_time(current_comark_data, lidar_data)

    #print_comark_data_grouped_by_date(current_comark_data)
        
    lidar_data_with_labels = assign_comark_labels(current_comark_data, lidar_data)




    get_report(lidar_data_with_labels, current_comark_data)
    
    save_to_protobuf(lidar_data_with_labels, "lidar_data.pb")
    #print(matched_data)
    
    
    
    
    
def print_comark_data_grouped_by_date(comark_data_dict):
    date_counts = {}
    
    for timestamp_str in comark_data_dict.keys():
        # Convert timestamp string to datetime object
        dt = datetime.datetime.fromisoformat(timestamp_str)
        # Extract just the date part
        date = dt.date()
        
        # Count occurrences
        if date in date_counts:
            date_counts[date] += 1
        else:
            date_counts[date] = 1
            
    # Print results in a readable format
    for date, count in date_counts.items():
        logging.info(f"Date: {date}, Number of entries: {count}")
    
    
    
    
#object_class
    
    
    
    
### AM ENDE ALS PROTOBUFF SPEICHERN 
    
    
    #print(lidar_data)
    # wim_data = get_wim_correspondence(WIM_FILE)

    # logging.info(f"Total frames: {N}")
    # logging.info(f"Time between frames: {T:.3f} s")

    # kalman_tracks = []

    # for _, row in wim_data.iterrows():
    #     k_WIM = int(row["detection_frame"])
    #     v_WIM = (
    #         -row["speed_in_kmh"] / 3.6
    #     )  # convert km/h to m/s and negate for direction
    #     y_WIM = Y_WIM_DEFAULT

    #     axle_spaces = [
    #         float(a) for a in str(row["axle_spaces_in_cm"]).split(";") if a
    #     ]  # parses ';' separated values in WIM CSV
    #     vehicle_length = sum(axle_spaces) / 100.0  # in meters

    #     # correct for missing Lidar data in starting frame (possible discrepancy between calculated and actual detection frame) (maximum of one frame as "grace" period)
    #     if not lidar_data[k_WIM]["clusters"]:
    #         k_WIM += 1
    #         if k_WIM >= N:
    #             logging.warning(
    #                 f"No LiDAR data available near WIM detection frame {k_WIM}. Skipping..."
    #             )
    #             continue

    #     # find closest object in detection frame
    #     if lidar_data[k_WIM]["clusters"]:
    #         distances = np.abs(
    #             np.array([c["front_y"] for c in lidar_data[k_WIM]["clusters"]]) - y_WIM
    #         )
    #         y_WIM = lidar_data[k_WIM]["clusters"][np.argmin(distances)]["front_y"]

    #     x_init = np.array([[y_WIM], [v_WIM]])

    #     y_forward, v_forward, forward_info = kalman_track(
    #         lidar_data, x_init, P_INIT, k_WIM, "forward", N, T, Q, R, DISTANCE_THRESH
    #     )
    #     y_backward, v_backward, backward_info = kalman_track(
    #         lidar_data, x_init, P_INIT, k_WIM, "backward", N, T, Q, R, DISTANCE_THRESH
    #     )

    #     # merge clusters dynamically for this vehicle for all frames in its track
    #     for frame_idx, _ in backward_info[::-1] + forward_info:
    #         clusters = lidar_data[frame_idx].get("clusters", [])
    #         if clusters:
    #             lidar_data[frame_idx]["clusters"] = merge_clusters(
    #                 clusters, vehicle_length
    #             )

    #     # filter out tracks with only two points as two points (one per tracking direction) indicates an error in the data
    #     total_points = sum(
    #         c is not None for _, c in (backward_info[::-1] + forward_info)
    #     )
    #     if total_points <= 2:
    #         logging.info(
    #             f"Skipping track at frame {k_WIM} (only {total_points} point detected)"
    #         )
    #         continue

    #     kalman_tracks.append(
    #         (
    #             k_WIM,
    #             y_forward,
    #             v_forward,
    #             forward_info,
    #             y_backward,
    #             v_backward,
    #             backward_info,
    #             row,
    #         )
    #     )

    # visualize(
    #     lidar_data, N, kalman_tracks
    # )  # deduplication only relevant for csv and not visualization (connected dots look nicer :))

    # rows = export_track_to_rows(kalman_tracks, lidar_data, FRAME_FILE_NEW)
    # pd.DataFrame(rows).to_csv("tracked_objects_2.csv", index=False)
    # logging.info("Exported tracked objects to tracked_objects_2.csv")


if __name__ == "__main__":
    main()
