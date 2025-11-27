import json
import logging
from config import COMARK_FILE


def load_comark_data():
    """Load co-marking JSON file specified in the project config.

    Returns:
        The parsed JSON object on success, or None if the file could not be read/parsed.
    """
    try:
        with open(COMARK_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data
    except FileNotFoundError:
        logging.error(f"COMARK file not found: {COMARK_FILE}")
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse COMARK JSON ({COMARK_FILE}): {e}")
    except Exception as e:
        logging.error(f"Unexpected error while loading COMARK file {COMARK_FILE}: {e}")
    return None


def get_neccessary_comark_infos(comark_data):
    """Transform comark_data to keep only timestamp and labels, using timestamp as key.
    
    Args:
        comark_data (dict): Original comark data with 'labels' dictionary
        
    Returns:
        dict: Transformed data with timestamps as keys and labels as values
    """
    transformed = {}
    
    # Get the labels dictionary
    labels_dict = comark_data.get('labels', {})
    
    # Extract timestamp and labels from each entry
    for idx, entry_data in labels_dict.items():
        if not isinstance(entry_data, dict):
            continue
            
        date = entry_data.get('file_info', {}).get('capture_date')
        time = entry_data.get('file_info', {}).get('capture_time')
        timestamp_unix=entry_data.get('timestamp_unix', {})
        timestamp=entry_data.get('timestamp', {})
        labels = entry_data.get('labels')
        if labels is None: 
            continue
        labels=labels[0].get("type")
        
        if timestamp and labels is not None:
            transformed[timestamp_unix] = [labels, timestamp, idx]
            
    return transformed


def cut_comark_data_to_lidar_date(comark_data_dict, lidar_data):
    """Cut comark data to only include entries that match the dates in lidar_data.
    
    Args:
        comark_data_dict (dict): Transformed comark data with timestamps as keys
        lidar_data (list): List of lidar data entries with 'date' field
    """
    filtered_comark_data = {}
    curr_date=lidar_data[0].get("timestamp_date").split('T')[0]

    for timestamp, labels in comark_data_dict.items():
        temp_timestamp=labels[1]
        date = temp_timestamp.split('T')[0]
        if date in curr_date:
            filtered_comark_data[timestamp] = labels

    sorted_by_key = dict(sorted(filtered_comark_data.items()))

    return sorted_by_key

def cut_comark_data_to_lidar_time(comark_data_dict, lidar_data):
    """Cut comark data to only include entries that match the timestamps in lidar_data.
    
    Args:
        comark_data_dict (dict): Transformed comark data with timestamps as keys
        lidar_data (list): List of lidar data entries with 'timestamp_s' field
    """
    time_filtered = {}
    lidar_start_time=lidar_data[0].get("timestamp_date")
    lidar_end_time=lidar_data[-1].get("timestamp_date")

    for timestamp, labels in comark_data_dict.items():
        temp_timestamp=labels[1]
        if temp_timestamp>=lidar_start_time and temp_timestamp<=lidar_end_time:
            time_filtered[timestamp] = labels
    return time_filtered