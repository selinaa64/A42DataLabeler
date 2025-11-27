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
            #if frame_idx==100: return lidar_data  # for testing purposes, limit to first 10 frames
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
        for scan in frame["scan_data"]:
            pass
            #for cluster in scan["clusters"]:
            #    cluster["timestamp_s"] = dt.isoformat()
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
import bisect
from datetime import datetime, timedelta
from collections import defaultdict
from datetime import timedelta

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
    for _, (label, ts_str, filename) in comark_data.items():
        try:
            ts = datetime.fromisoformat(ts_str)
            comark_times.append((ts + offset, label, filename))
        except Exception:
            continue

    # Sort by timestamp for better readability of results
    comark_times.sort(key=lambda x: x[0])
    obj_groups: dict[Any, List[tuple[dict, datetime | None]]] = defaultdict(list)

    for frame in lidar_data:
        for scan in frame.get("scan_data", []):
            oid = scan.get("object_id")
            if oid is None:
                # Scans ohne object_id bekommen kein Label
                scan["object_label"] = None
                continue

            lidar_ts = None
            ts_str = scan.get("timestamp_date")
            if ts_str is not None:
                try:
                    lidar_ts = datetime.fromisoformat(ts_str)
                except Exception as e:
                    logging.warning(f"Could not parse lidar timestamp '{ts_str}': {e}")

            obj_groups[oid[0]].append((scan, lidar_ts))
    # Process each lidar frame
    # --- 3) Für jede object_id: letzten Scan finden und Label bestimmen ---
    object_label_map: dict[Any, str | None] = {}


    # z.B. oben in der Funktion definieren
    travel_offset = timedelta(seconds=1.2)

    for oid, scan_list in obj_groups.items():
        # nur Scans mit gültigem Timestamp verwenden
        scans_with_ts = [(s, ts) for (s, ts) in scan_list if ts is not None]

        if not scans_with_ts:
            object_label_map[oid] = None
            continue

        # letzten Scan (größter Timestamp) bestimmen
        last_scan, last_ts = max(scans_with_ts, key=lambda st: st[1])

        # nächstgelegenen CoMark-Zeitpunkt zu last_ts finden
        best_diff = timedelta(days=9999)
        best_label = None
        curr_filename = None
        for comark_ts, label, filename in comark_times:
            # CoMark-Zeitstempel um 1,2 s in die Zukunft verschieben,
            # weil das Fahrzeug erst dann beim LiDAR ankommt
            shifted_comark_ts = comark_ts + travel_offset
            diff = abs(last_ts - shifted_comark_ts)

            if diff < best_diff:
                curr_filename = filename
                best_diff = diff
                best_label = label
            # optionaler Abbruch wie gehabt hier möglich

        object_label_map[oid] = [best_label, curr_filename]

    # --- 4) Label auf alle Scans je object_id übertragen ---
    for oid, scan_list in obj_groups.items():
        label = object_label_map.get(oid)[0]
        filename = object_label_map.get(oid)[1]
        for scan, _ in scan_list:
            scan["object_label"] = label
            scan["comark_filename"] = filename


    # Optional: Logging, wie viele Scans/Objekte pro Label existieren
    label_counts: dict[str, int] = defaultdict(int)
    filename_counts: dict[str, int] = defaultdict(int)
    for oid, scan_list in obj_groups.items():
        label = object_label_map.get(oid)[0]
        filename = object_label_map.get(oid)[1]
        if label is None:
            continue
        label_counts[label] += len(scan_list)
        filename_counts[filename] += len(scan_list)

    for label, count in label_counts.items():
        logging.info(f"CoMark Label '{label}': {count} scans")
    for filename, count in filename_counts.items():
        logging.info(f"CoMark filename '{filename}': {count} scans")

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

import os
import numpy as np
import matplotlib.pyplot as plt
## TODO Plot bums weiter machen

def visualize_frames(frames_out: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Visualizes LiDAR clusters from loaded protobuf frames.
    Alle Plots bekommen die gleichen Achsen-Limits.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1) Globale Bounds über alle Frames bestimmen ----------
    xs_all, ys_all, zs_all = [], [], []

    for entry in frames_out:
        pcs = entry.get("pointclouds", [])
        if not pcs:
            continue

        pc = pcs[0]
        if isinstance(pc, list):
            pc = np.array(pc)

        if pc.ndim != 2 or pc.shape[1] != 3:
            continue

        xs_all.append(pc[:, 0])
        ys_all.append(pc[:, 1])
        zs_all.append(pc[:, 2])

        # optional: Cluster-Boxen mit einbeziehen, damit die auch sicher im Bereich liegen
        for c in entry.get("clusters", []):
            fx = c["front_x"]
            fy = c["front_y"]
            fz = c["front_z"]
            ex = c["extent_x"]
            ey = c["extent_y"]
            ez = c["extent_z"]

            xs_all.append(np.array([fx, fx + ex]))
            ys_all.append(np.array([fy, fy + ey]))
            zs_all.append(np.array([fz, fz + ez]))

    if not xs_all:
        # falls gar nichts da ist, irgendeinen Default-Bereich
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        z_min, z_max = -1, 1
    else:
        xs_all = np.concatenate(xs_all)
        ys_all = np.concatenate(ys_all)
        zs_all = np.concatenate(zs_all)

        padding = 0.5  # kleiner Rand drumherum
        x_min, x_max = xs_all.min() - padding, xs_all.max() + padding
        y_min, y_max = ys_all.min() - padding, ys_all.max() + padding
        z_min, z_max = zs_all.min() - padding, zs_all.max() + padding

    # ---------- 2) Frames mit festen Achsen plotten ----------
    for i, entry in enumerate(frames_out):
        pcs = entry.get("pointclouds", [])
        if not pcs:
            print(f"Keine Punktwolken in Eintrag {i}, überspringe.")
            continue

        pc = pcs[0]
        if isinstance(pc, list):
            pc = np.array(pc)

        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"Pointcloud muss Shape (N, 3) haben, hat {pc.shape}")

        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, s=1)

        # --- Cluster als 3D-Boxen einzeichnen --------------------------------
        clusters = entry.get("clusters", [])
        for c in clusters:
            fx = c["front_x"]
            fy = c["front_y"]
            fz = c["front_z"]
            ex = c["extent_x"]
            ey = c["extent_y"]
            ez = c["extent_z"]

            x0, x1 = fx, fx + ex
            y0, y1 = fy, fy + ey
            z0, z1 = fz, fz + ez

            corners = np.array([
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ])

            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]

            for i1, i2 in edges:
                xs = [corners[i1, 0], corners[i2, 0]]
                ys = [corners[i1, 1], corners[i2, 1]]
                zs = [corners[i1, 2], corners[i2, 2]]
                ax.plot(xs, ys, zs)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Feste Achsenlimits für alle Plots
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Aspect: "echte" Geometrie beibehalten
        try:
            ax.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
        except AttributeError:
            pass  # ältere Matplotlib-Version

        ts = entry.get("timestamp_s", "")
        oid = entry.get("object_id", "")
        label = entry.get("object_label", "")

        ax.set_title(f"{ts}\nobject_id: {oid} | label: {label}")
        plt.tight_layout()

        ts_ns = entry.get("timestamp_ns", 0)
        filename = f"{ts_ns}_id{oid}.png"
        filepath = os.path.join(output_dir, filename)

        fig.savefig(filepath, dpi=200)
        plt.close(fig)

        print(f"Gespeichert: {filepath}")

from typing import List, Dict, Any
import logging

import numpy as np

from lidar_frame import LidarDataset, LidarFrame, Cluster, Pointclouds


def save_to_protobuf(lidar_data_with_labels: List[Dict[str, Any]], output_file: str) -> None:
    frames: List[LidarFrame] = []

    for frame_dict in lidar_data_with_labels:
        ts_ns = int(frame_dict.get("timestamp_ns", 0))
        # timestamp_date -> timestamp_s im Proto
        ts_s = str(frame_dict.get("timestamp_date", ""))

        for scan in frame_dict.get("scan_data", []):
            # --------- CLUSTERS ----------------------------------------
            clusters: List[Cluster] = []
            for cluster_dict in scan.get("clusters", []) or []:
                if not cluster_dict:
                    continue
                clusters.append(
                    Cluster(
                        front_x=float(cluster_dict.get("front_x", 0.0)),
                        front_y=float(cluster_dict.get("front_y", 0.0)),
                        front_z=float(cluster_dict.get("front_z", 0.0)),
                        extent_x=float(cluster_dict.get("extent_x", 0.0)),
                        extent_y=float(cluster_dict.get("extent_y", 0.0)),
                        extent_z=float(cluster_dict.get("extent_z", 0.0)),
                    )
                )

            # --------- POINTCLOUDS -------------------------------------
            pointcloud_messages: List[Pointclouds] = []

            pc_raw = scan.get("pointclouds")

            # Fall 1: Liste mit Arrays [array(...), array(...), ...]
            if isinstance(pc_raw, list):
                arrays = [a for a in pc_raw if isinstance(a, np.ndarray)]
            # Fall 2: direkt ein ndarray
            elif isinstance(pc_raw, np.ndarray):
                arrays = [pc_raw]
            else:
                arrays = []

            for arr in arrays:
                if arr.ndim != 2 or arr.shape[1] != 3:
                    raise ValueError(f"Pointcloud array must have shape (N, 3), got {arr.shape}")

                rows, cols = arr.shape
                arr32 = arr.astype(np.float32, copy=False)
                raw_bytes = arr32.tobytes()

                pointcloud_messages.append(
                    Pointclouds(
                        raw=raw_bytes,
                        rows=rows,
                        cols=cols,
                    )
                )

            # --------- LABEL / OBJECT_ID --------------------------------
            obj_label = scan.get("object_label")
            if isinstance(obj_label, (list, tuple, set)):
                obj_label_str = ",".join(str(x) for x in obj_label)
            elif obj_label is None:
                obj_label_str = ""
            else:
                obj_label_str = str(obj_label)

            object_id = scan.get("object_id")
            object_id_int = int(object_id[0]) if object_id is not None else 0
            file_name = scan.get("comark_filename", "")
            # --------- FRAME (ein Scan = ein LidarFrame) ----------------
            frame_msg = LidarFrame(
                timestamp_ns=ts_ns,
                timestamp_s=ts_s,
                clusters=clusters,
                object_label=obj_label_str,
                pointclouds=pointcloud_messages,
                object_id=object_id_int,
                comark_file_name=file_name,
            )
            frames.append(frame_msg)

    dataset = LidarDataset(frames=frames)
    data = bytes(dataset)

    with open(output_file, "wb") as f:
        f.write(data)

    logging.info("Saved %d frames to %s", len(dataset.frames), output_file)


def load_from_protobuf(input_file: str) -> List[Dict[str, Any]]:
    with open(input_file, "rb") as f:
        data = f.read()

    dataset = LidarDataset().parse(data)

    frames_out: List[Dict[str, Any]] = []

    for pf in dataset.frames:
        # --- Clusters zurück in Dicts --------------------------------------
        clusters = [
            {
                "front_x": c.front_x,
                "front_y": c.front_y,
                "front_z": c.front_z,
                "extent_x": c.extent_x,
                "extent_y": c.extent_y,
                "extent_z": c.extent_z,
            }
            for c in pf.clusters
        ]

        # --- Pointclouds rekonstruieren -----------------------------------
        pointclouds_list = []
        for pc in pf.pointclouds:
            # aus Bytes wieder float32-Array mit richtiger Shape
            arr = np.frombuffer(pc.raw, dtype=np.float32)
            arr = arr.reshape(pc.rows, pc.cols)
            pointclouds_list.append(arr)

        frame_dict: Dict[str, Any] = {
            "timestamp_ns": pf.timestamp_ns,
            "timestamp_s": pf.timestamp_s,
            "clusters": clusters,
            "pointclouds": pointclouds_list,  # gleiche Struktur wie beim Speichern
            "object_label": pf.object_label or None,
            "object_id": pf.object_id,
            "comark_file_name": pf.comark_file_name or None,
        }

        frames_out.append(frame_dict)

    return frames_out
def load_lidar_data_NEW(input_file=None):
    lidar_data = []
    if input_file is None:
        input_file = FRAME_FILE_NEW
    for frame in read_length_delimited_frames(input_file):
        ts_ns = int(frame.frame_timestamp_ns)
        frame_data = {
            "timestamp_ns": ts_ns,
            "timestamp_date": datetime.fromtimestamp(ts_ns / 1e9).isoformat(),
            "scan_data": [],
        }

        for scan in frame.lidars:
            
            # Nur bereits detektierte Objekte aus object_list übernehmen
            if len(scan.object_list) == 0:
                continue

            for obj in scan.object_list:
                raw_points = np.frombuffer(obj.pointcloud.cartesian, dtype="<f4").reshape(-1, 3)
                ts_ns_scan=obj.timestamp_ns
                scan_data={  
                    "timestamp_ns": ts_ns_scan,
                    "timestamp_date": datetime.fromtimestamp(ts_ns_scan / 1e9).isoformat(),
                    "object_id": [],
                    "clusters": [],
                    "pointclouds": [],
                } 
                scan_data["clusters"].append(
                    {
                        "front_x": obj.position.x,
                        "front_y": obj.position.y - (obj.dimension.y / 2),
                        "front_z": obj.position.z,
                        "extent_x": obj.dimension.x,
                        "extent_y": obj.dimension.y,
                        "extent_z": obj.dimension.z,
                    })
                scan_data["object_id"].append(obj.id)
                scan_data["pointclouds"].append(raw_points)
                frame_data["scan_data"].append(scan_data)
                

        lidar_data.append(frame_data)


    return lidar_data
def main():
    ### TODO evaluieren von frames und labels
    ### TODO verschiedene Objekte (1 frame mit einem reinfahrendem und einem rausfahrendem) gleicher Comark label 
    # logging.info("Loading LiDAR data...")
    # lidar_data = load_lidar_data_NEW()
    # # visualize_frames(lidar_data, "output_frames")
    # # #lidar_data=convert_unix_timestamp_to_ISO(lidar_data)
    # T = get_period()
    # N = get_sequence_length()
    # logging.info("Loading Comark data...")

    # comark_data = load_comark_data()  # function to load comark data if needed
    # comark_data_dict=get_neccessary_comark_infos(comark_data)
    # #print_comark_data_grouped_by_date(comark_data_dict)
    # current_comark_data= cut_comark_data_to_lidar_date(comark_data_dict, lidar_data)
    # current_comark_data= cut_comark_data_to_lidar_time(current_comark_data, lidar_data)

    # #print_comark_data_grouped_by_date(current_comark_data)
        
    # lidar_data_with_labels = assign_comark_labels(current_comark_data, lidar_data)




    # get_report(lidar_data_with_labels, current_comark_data)
    
    #save_to_protobuf(lidar_data_with_labels, "lidar_data.pb")
    frames_out = load_from_protobuf("lidar_data.pb")
    #visualize_frames(frames_out, "output_frames")
    #load_lidar_data_NEW("lidar_data.pb")
    test_vis(frames_out, "output_frames_test")

import os
from typing import List, Dict, Any

import numpy as np
import open3d as o3d

import os
from typing import List, Dict, Any

import numpy as np
import open3d as o3d


BYTES_PER_POINT = 12  # xyz float32

import os
from typing import List, Dict, Any
from datetime import datetime

import numpy as np
import open3d as o3d

BYTES_PER_POINT = 12  # xyz float32


def unpack_xyz(pc) -> np.ndarray:
    if not pc or not getattr(pc, "cartesian", None):
        return np.zeros((0, 3), dtype=np.float32)
    n = len(pc.cartesian) // BYTES_PER_POINT
    return np.frombuffer(pc.cartesian, dtype="<f4").reshape(n, 3)


def bbox_to_lineset(bbox) -> o3d.geometry.LineSet:
    # Proto-Objekt mit position / dimension
    if hasattr(bbox, "position") and hasattr(bbox, "dimension"):
        center = [bbox.position.x, bbox.position.y, bbox.position.z]
        extent = [bbox.dimension.x, bbox.dimension.y, bbox.dimension.z]
    else:
        # Dict mit front_x/extent_x
        fx = bbox["front_x"]
        fy = bbox["front_y"]
        fz = bbox["front_z"]
        ex = bbox["extent_x"]
        ey = bbox["extent_y"]
        ez = bbox["extent_z"]
        center = [fx + ex / 2.0, fy + ey / 2.0, fz + ez / 2.0]
        extent = [ex, ey, ez]

    obb = o3d.geometry.OrientedBoundingBox(center, np.eye(3), extent)
    ls = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    ls.paint_uniform_color([1, 0, 0])
    return ls


def _get_xyz_from_entry(entry: Dict[str, Any]) -> np.ndarray:
    pcs = entry.get("pointclouds", [])
    if not pcs:
        return np.zeros((0, 3), dtype=np.float32)

    pc0 = pcs[0]

    if hasattr(pc0, "cartesian"):
        return unpack_xyz(pc0)

    if isinstance(pc0, list):
        pc0 = np.array(pc0, dtype=np.float32)
    else:
        pc0 = np.array(pc0, dtype=np.float32)

    if pc0.ndim != 2 or pc0.shape[1] != 3:
        raise ValueError(f"Pointcloud muss Shape (N, 3) haben, hat {pc0.shape}")

    return pc0


def test_vis(frames_out: List[Dict[str, Any]], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1) Sortierreihenfolge (object_id, timestamp) vorbereiten ----------
    meta: list[tuple[Any, datetime, int, Dict[str, Any]]] = []

    for i, entry in enumerate(frames_out):
        oid = entry.get("object_id")
        # falls object_id z.B. eine Liste/Tuple ist
        if isinstance(oid, (list, tuple)):
            oid_key = oid[0]
        else:
            oid_key = oid

        ts_str = entry.get("timestamp_s", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            ts = datetime.min  # zur Not ganz nach vorne

        meta.append((oid_key, ts, i, entry))

    # sortieren nach object_id, dann timestamp, dann ursprünglichem Index
    meta_sorted = sorted(meta, key=lambda x: (x[0], x[1], x[2]))

    # ---------- 2) In dieser Reihenfolge plotten ----------
    for oid_key, ts_dt, i, entry in meta_sorted:
        xyz = _get_xyz_from_entry(entry)
        if xyz.size == 0:
            print(f"Keine Punktwolke in Eintrag {i}, überspringe.")
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.paint_uniform_color([0.6, 0.6, 0.6])

        overlays = [pcd]

        for bbox in entry.get("clusters", []):
            try:
                overlays.append(bbox_to_lineset(bbox))
            except Exception as e:
                print(f"Fehler beim Erzeugen der BBox für Eintrag {i}: {e}")

        ts = entry.get("timestamp_s", "")
        oid = entry.get("object_id", "")
        label = entry.get("object_label", "")
        file_name = entry.get("comark_file_name", "")

        print(f"Frame {i}: ts={ts}, object_id={oid}, label={label}, file={file_name}")

        o3d.visualization.draw_geometries(
            overlays,
            window_name=f"id={oid} | ts={ts} | label={label} | file={file_name}"
        )

    
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
