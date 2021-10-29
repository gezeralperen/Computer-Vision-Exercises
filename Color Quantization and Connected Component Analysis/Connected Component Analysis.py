import numpy as np
import cv2
from matplotlib import pyplot as plt


import os

from numpy.lib.arraysetops import union1d
file_path = os.path.dirname(os.path.realpath(__file__))

def label_adj(mask, x, y, segmentation):
    # Check Right
    if x<mask.shape[1] and mask[y,x+1] == 1 and segmentation[y, x+1]==0:
        segmentation[y, x+1] = segmentation[y, x]
        segmentation = label_adj(mask, x+1, y, segmentation)
    # Check Bottom-Right
    if x<mask.shape[1] and y<mask.shape[0] and mask[y+1,x+1] == 1 and segmentation[y+1, x+1]==0:
        segmentation[y+1, x+1] = segmentation[y, x]
        segmentation = label_adj(mask, x+1, y+1, segmentation)
    # Check Top-Right
    if x<mask.shape[1] and y>0 and mask[y-1,x+1] == 1 and segmentation[y-1, x+1]==0:
        segmentation[y-1, x+1] = segmentation[y, x]
        segmentation = label_adj(mask, x+1, y-1, segmentation)
    # Check Bottom
    if y<mask.shape[0] and mask[y+1,x] == 1 and segmentation[y+1, x]==0:
        segmentation[y+1, x] = segmentation[y, x]
        segmentation = label_adj(mask, x, y+1, segmentation)
    # Check Top
    if y>0 and mask[y-1,x] == 1 and segmentation[y-1, x]==0:
        segmentation[y-1, x] = segmentation[y, x]
        segmentation = label_adj(mask, x, y-1, segmentation)
    # Check Bottom-Left
    if x>0 and y<mask.shape[0] and mask[y+1,x-1] == 1 and segmentation[y+1, x-1]==0:
        segmentation[y+1, x-1] = segmentation[y, x]
        segmentation = label_adj(mask, x-1, y+1, segmentation)
    # Check Left
    if x>0 and mask[y,x-1] == 1 and segmentation[y, x-1]==0:
        segmentation[y, x-1] = segmentation[y, x]
        segmentation = label_adj(mask, x-1, y, segmentation)
    # Check Top-Left
    if x>0 and y>0 and mask[y-1,x-1] == 1 and segmentation[y-1, x-1]==0:
        segmentation[y-1, x-1] = segmentation[y, x]
        segmentation = label_adj(mask, x-1, y-1, segmentation)
    return segmentation


def connected_components(img_threshold):
    object_count = 0
    segmentation = np.zeros(img_threshold.shape, dtype=int)

    for y in range(img_threshold.shape[0]):
        for x in range(img_threshold.shape[1]):
            if img_threshold[y,x] == 1 and segmentation[y,x] == 0:
                segmentation[y,x] = object_count + 1
                object_count += 1
                segmentation = label_adj(img_threshold, x, y, segmentation)

    return segmentation, object_count


# Birds
images = [file_path + '/birds1.jpg',
          file_path + '/birds2.jpg',
          file_path + '/birds3.jpg']


image = cv2.imread(images[0], 0)
image_threshold = (image < 100).astype(np.float32)
image_dilated = cv2.dilate(image_threshold, np.ones((5,5)), iterations=1)

result, count = connected_components(image_dilated)

print(f'{count} objects detected!')

plt.imshow(result, cmap='gist_ncar')
plt.show()