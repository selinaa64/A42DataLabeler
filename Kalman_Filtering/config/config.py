import numpy as np

FRAME_FILE_OLD = r"C:/Users/selin/Documents/studium/forschungsprojekt/data/version_0_2_data.pb"
FRAME_FILE_NEW = r"C:/Users/selin/Documents/studium/forschungsprojekt/data/version_0_4_data_lane_2.pb"

WIM_FILE = r"C:/Users/selin/Documents/studium/forschungsprojekt/data/data-1753369613718.csv"

LANE_BOX_DIMS = np.array([3.3, 90, 3.5])
LANE_BOX_POS = np.array([0, 0, 1.88])
Y_WIM_DEFAULT = 10.67
DISTANCE_THRESH = 1.5
Q = (1 / 3.6) ** 2
R = 0.2**2
P_INIT = np.array([[5**2, 0], [0, (10 / 3.6) ** 2]])
