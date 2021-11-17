import numpy as np
from points_matching import get_points
from utils.homography import computeH
from utils.warping import warp
from utils.merge import merge
import cv2

import os
file_path = os.path.dirname(os.path.realpath(__file__))
np.set_printoptions(suppress=True)

im2Points, im1Points = get_points('north_campus', 'middle', 'left-1')
h = computeH(im1Points, im2Points)


image = cv2.imread(file_path + '/data/north_campus/middle.jpg')
image2 = cv2.imread(file_path + '/data/north_campus/left-1.jpg')

image_dist, coord = warp(image2, h)

cv2.imshow('Frame', merge(image, image_dist, coord))

cv2.waitKey()