from utils import read_length_delimited_frames
from config import FRAME_FILE
import numpy as np

def get_sequence_length():
    total_frames = 0
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        total_frames += 1
    return total_frames

def get_period():
    first_timestamp = 0
    second_timestamp = 0
    for frame_iterator, frame in enumerate(read_length_delimited_frames(FRAME_FILE)):
        if frame_iterator == 0:
            first_timestamp = frame.frame_timestamp_ns
        elif frame_iterator == 1:
            second_timestamp = frame.frame_timestamp_ns
        else:
            continue
    period = (second_timestamp-first_timestamp) * (10**(-9))
    return period
