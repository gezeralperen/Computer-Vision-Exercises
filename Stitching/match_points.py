from utils.point_picker import point_matcher
import cv2
import numpy as np

import os
file_path = os.path.dirname(os.path.realpath(__file__))
np.set_printoptions(suppress=True)

image1 = cv2.imread(file_path + "/data/north_campus/left-1.jpg")
image2 = cv2.imread(file_path + "/data/north_campus/left-2.jpg")


point_matcher(image1, image2)