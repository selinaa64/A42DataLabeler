# !currently not quite working as inteded but midifies the pandas correctly!
# checks lidar data and wim data to find the position of objects crossing the wim in 3d space
# speed and position later used as starting values of the kalman filter
import os
import sys
import time
import struct
import logging
import numpy as np
import pandas as pd
from google.protobuf.message import DecodeError
from a42.frame_pb2 import Frame

FRAME_FILE = r"/users/emil/Documents/HS_Esslingen/Studienprojekt/Data/TestData1/all_frames.pb"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

def read_length_delimited_frames(path):
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

def datetime_to_unix_ns(datetime):
    return pd.to_datetime(datetime).value

# used to crop WIM DataFrame
def get_minmax_lidar_timestamp():
    min_timestamp = 0
    max_timestamp = 0
    all_frames = read_length_delimited_frames(FRAME_FILE)

    for frame_iterator, frame in enumerate(all_frames):
        if frame_iterator == 0:
            min_timestamp = frame.frame_timestamp_ns
        elif frame.frame_timestamp_ns > max_timestamp: # end is not indexable so the entire loop is being run until the last object is found
            max_timestamp = frame.frame_timestamp_ns

    minmax_timestamp = np.array([min_timestamp, max_timestamp])
    return(minmax_timestamp)

# should check for the first frame to have a later timestamp than the WIM data and thus be the first frame after detection
def get_detection_frame(timestamp):
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        if frame.frame_timestamp_ns >= timestamp:
            return frame_iterator

# used to get the postion (front center) of the object with the closest distance to the WIM's y position
def get_closest_object_to_wim(frame_index):
    wim_position_y = 10.67 #!might be incorrect!
    smallest_distance = 100
    closest_object = np.array([-1,-1,-1])
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        if frame_iterator == frame_index:
            for scan_iterator, scan in enumerate(frame.lidars):
                for object_iterator, obj in enumerate(scan.object_list.objects):
                    front_of_object_y = obj.position.y - (obj.dimension.y/2)
                    distance = np.abs(front_of_object_y-wim_position_y)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        closest_object = [obj.position.x, front_of_object_y, obj.position.z]
            return closest_object


def main():
    if not os.path.exists(FRAME_FILE):
        log.error(f"Datei nicht gefunden: {FRAME_FILE}")
        return

    wim_data = pd.read_csv('/Users/emil/Documents/HS_Esslingen/Studienprojekt/Data/TestData1/data-1752218465752.csv')

    wim_data = wim_data[wim_data['data_source_device_id'] == 'A42-FR-GeKi-Spur-Rechts'] #only look at the right lane

    wim_data['merge_unix_timestamp_ns'] = wim_data['merge_timestamp'].apply(datetime_to_unix_ns) # timestamp for easy processing
    wim_data['merge_unix_timestamp_ns'] = wim_data['merge_unix_timestamp_ns'] - 7_200_000_000_000 # utc+2 to utc (7.2 trillion seconds)
    minmax_timestamp = get_minmax_lidar_timestamp() 
    wim_data = wim_data[(wim_data['merge_unix_timestamp_ns'] >= minmax_timestamp[0]) & (wim_data['merge_unix_timestamp_ns'] <= minmax_timestamp[1])] #only look at the time of lidar frames
    
    wim_data = wim_data.sort_values(by='merge_unix_timestamp_ns')

    wim_data['detection_frame'] = wim_data['merge_unix_timestamp_ns'].apply(get_detection_frame) #!there is either an issue here or with position of the bounding box!
    wim_data['closest_object'] = wim_data['detection_frame'].apply(get_closest_object_to_wim) #!issue might also be related to the other lane!
    
    print(wim_data[['detection_frame', 'closest_object']])

        

if __name__ == "__main__":
    main()