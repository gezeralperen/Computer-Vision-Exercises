import numpy as np
from points_matching import get_points
from utils.homography import computeH
from utils.warping import warp
from utils.merge import merge
import cv2

import os
file_path = os.path.dirname(os.path.realpath(__file__))
np.set_printoptions(suppress=True)



# Left addition
base_image = cv2.imread(file_path + '/data/north_campus/middle.jpg')
addition_image = cv2.imread(file_path + '/data/north_campus/left-1.jpg')
base, addition = get_points('north_campus', 'middle', 'left-1')
h = computeH(addition, base)
image_dist, coord = warp(addition_image, h)
left_image, origin = merge(base_image, image_dist, coord)


# Right addition
base_image = cv2.imread(file_path + '/data/north_campus/middle.jpg')
addition_image = cv2.imread(file_path + '/data/north_campus/right-1.jpg')
addition, base = get_points('north_campus', 'middle', 'right-1')
h = computeH(addition, base)
image_dist, coord = warp(addition_image, h)
merged_image, origin = merge(left_image, image_dist, coord, origin)


cv2.imshow('Right', merged_image)
cv2.waitKey()