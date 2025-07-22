# !not sure if the offset is working as inteded but modifies the pandas correctly!
# checks lidar data and wim data to find the position of objects crossing the wim in 3d space
# speed and position later used as starting values of the kalman filter
from utils import FRAME_FILE
from utils import read_length_delimited_frames
import pandas as pd
import numpy as np

def _datetime_to_unix_ns(datetime):
    return pd.to_datetime(datetime).value

# used to crop WIM DataFrame
def _get_minmax_lidar_timestamp():
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

# checks for the first frame to have a later timestamp than the WIM data and thus be the first frame after detection
def _get_detection_frame(timestamp):
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        if frame.frame_timestamp_ns >= timestamp:
            return frame_iterator

# used to get the postion (front center) of the object with the closest distance to the WIM's y position
def _get_closest_object_to_wim(frame_index):
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


def get_wim_correspondence(PATH):
    wim_data = pd.read_csv(PATH)

    wim_data = wim_data[wim_data['data_source_device_id'] == 'A42-FR-GeKi-Spur-Rechts'] #only look at the right lane

    wim_data['merge_unix_timestamp_ns'] = wim_data['merge_timestamp'].apply(_datetime_to_unix_ns) # timestamp for easy processing
    wim_data['merge_unix_timestamp_ns'] = wim_data['merge_unix_timestamp_ns'] - 7_200_000_000_000 # utc+2 to utc (7.2 trillion seconds)

    '''1752218382402629927 example frame where ther should be a detection
       1752218383819644000 closest WIM detection <- 8 frames too late => ~ 1.41 second offset'''
       
    #!experiment!
    wim_data['merge_unix_timestamp_ns'] = wim_data['merge_unix_timestamp_ns'] - 1_400_000_000  # 1.41 seconds
    #!experiment!

    minmax_timestamp = _get_minmax_lidar_timestamp() 
    wim_data = wim_data[(wim_data['merge_unix_timestamp_ns'] >= minmax_timestamp[0]) & (wim_data['merge_unix_timestamp_ns'] <= minmax_timestamp[1])] #only look at the time of lidar frames
    
    wim_data = wim_data.sort_values(by='merge_unix_timestamp_ns')

    wim_data['detection_frame'] = wim_data['merge_unix_timestamp_ns'].apply(_get_detection_frame) 
    wim_data['closest_object'] = wim_data['detection_frame'].apply(_get_closest_object_to_wim)
    
    return wim_data