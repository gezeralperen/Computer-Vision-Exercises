import numpy as np
import cv2
from matplotlib import pyplot as plt

import os
file_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.setrecursionlimit(999999)

def label_adj(mask, x, y, segmentation):
    # Check Right
    if x<mask.shape[1]-1 and mask[y,x+1] == 1 and segmentation[y, x+1]==0:
        segmentation[y, x+1] = segmentation[y, x]
        segmentation = label_adj(mask, x+1, y, segmentation)
    # Check Bottom-Right
    if x<mask.shape[1]-1 and y<mask.shape[0]-1 and mask[y+1,x+1] == 1 and segmentation[y+1, x+1]==0:
        segmentation[y+1, x+1] = segmentation[y, x]
        segmentation = label_adj(mask, x+1, y+1, segmentation)
    # Check Top-Right
    if x<mask.shape[1]-1 and y>0 and mask[y-1,x+1] == 1 and segmentation[y-1, x+1]==0:
        segmentation[y-1, x+1] = segmentation[y, x]
        segmentation = label_adj(mask, x+1, y-1, segmentation)
    # Check Bottom
    if y<mask.shape[0]-1 and mask[y+1,x] == 1 and segmentation[y+1, x]==0:
        segmentation[y+1, x] = segmentation[y, x]
        segmentation = label_adj(mask, x, y+1, segmentation)
    # Check Top
    if y>0 and mask[y-1,x] == 1 and segmentation[y-1, x]==0:
        segmentation[y-1, x] = segmentation[y, x]
        segmentation = label_adj(mask, x, y-1, segmentation)
    # Check Bottom-Left
    if x>0 and y<mask.shape[0]-1 and mask[y+1,x-1] == 1 and segmentation[y+1, x-1]==0:
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


def connected_components(mask):
    object_count = 0
    segmentation = np.zeros(mask.shape, dtype=int)
    mask = mask.astype(int)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y,x] == 1 and segmentation[y,x] == 0:
                segmentation[y,x] = object_count + 1
                object_count += 1
                segmentation = label_adj(mask, x, y, segmentation)

    return segmentation, object_count

################### Okey Tiles #####################
image = cv2.imread(file_path + '/demo4.PNG')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
sat_channel = image[:,:,1]

numbers_mask = (sat_channel>150).astype(np.float32)
numbers_mask = cv2.erode(numbers_mask, np.ones((3,3)), iterations=1)

masked_image = (image*cv2.cvtColor(numbers_mask, cv2.COLOR_GRAY2BGR)).astype(np.uint8)

hue_channel = masked_image[:,:,0]

blue_tiles = (hue_channel<120).astype(np.float32)*numbers_mask
red_tiles = (hue_channel>120).astype(np.float32)


result, count = connected_components(blue_tiles)
print(f'{count} blue tiles detected in demo4.PNG!')
plt.imsave(file_path + f'/results_{count}_blue_tiles_demo4.PNG',result, cmap='gist_ncar')

result, count = connected_components(red_tiles)
print(f'{count} red tiles detected in demo4.PNG!')
plt.imsave(file_path + f'/results_{count}_red_tiles_demo4.PNG',result, cmap='gist_ncar')



###################### Dices ########################
images = [file_path + '/dice5.PNG',
          file_path + '/dice6.PNG',]

kernels = [np.ones((5,5)),
           np.ones((5,5)),]

for i, image_path in enumerate(images):
    file_name = image_path.split('/')[-1]
    image = cv2.imread(image_path, 0)
    extended_background = cv2.copyMakeBorder(image,20,20,20,20,cv2.BORDER_CONSTANT,value=0)
    image_mask = (extended_background < 150).astype(np.float32)
    mask_eroded = cv2.erode(image_mask, kernels[i], iterations=1)

    result, count = connected_components(mask_eroded)

    print(f'{count-1} objects detected in {file_name}!')

    plt.imsave(file_path + f'/results_{count-1}_objects_'+ file_name,result, cmap='gist_ncar')


####################### Birds ##########################
images = [file_path + '/birds1.jpg',
          file_path + '/birds2.jpg',
          file_path + '/birds3.jpg']

kernels = [np.ones((3,3)),
           np.ones((1,1)),
           np.ones((7,7)),]

for i, image_path in enumerate(images):
    file_name = image_path.split('/')[-1]
    image = cv2.imread(image_path, 0)
    image_mask = (image < 200).astype(np.float32)
    mask_dilated = cv2.dilate(image_mask, kernels[i], iterations=1)

    result, count = connected_components(mask_dilated)

    print(f'{count} objects detected in {file_name}!')

    plt.imsave(file_path + f'/results_{count}_objects_'+ file_name,result, cmap='gist_ncar')