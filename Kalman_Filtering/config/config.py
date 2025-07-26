import numpy as np

FRAME_FILE = r"/users/emil/Documents/HS_Esslingen/Studienprojekt/Data/TestData2/no_objects_new.pb"
WIM_FILE = r"/Users/emil/Documents/HS_Esslingen/Studienprojekt/Data/TestData2/data-1753189594175.csv"

LANE_BOX_DIMS = np.array([3.3, 90, 3.5])
LANE_BOX_POS = np.array([0, 0, 1.88])
Y_WIM_DEFAULT = 10.67
DISTANCE_THRESH = 1.5
Q = (1 / 3.6) ** 2
R = 0.2**2
P_INIT = np.array([[5**2, 0], [0, (10 / 3.6) ** 2]])
