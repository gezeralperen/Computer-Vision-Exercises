import numpy as np
from points_matching import get_points
from utils.homography import computeH
from utils.warping import warp
from utils.merge import merge
import cv2

import os
file_path = os.path.dirname(os.path.realpath(__file__))
np.set_printoptions(suppress=True)


def connect(base_image, addition_image, base, addition, origin=(0,0), noise=False, n_wrong = 0, n_points=None, normalize=True, add_origin=(0,0)):
    base_mean = np.mean(base, axis=0)
    add_mean = np.mean(addition, axis=0)
    orientation = add_mean-base_mean
    if noise:
        base_noise = np.random.normal(0, 5, base.shape).astype(int)
        add_noise = np.random.normal(0, 5, addition.shape).astype(int)
        base += base_noise
        addition += add_noise
    for _ in range(n_wrong):
        base_max = np.max(base, axis=0)
        base_min = np.min(base, axis=0)
        base_rand = [np.random.randint(base_min[0], base_max[0]), np.random.randint(base_min[1], base_max[1])]
        add_rand = [base_rand[0]+orientation[0]+np.random.randint(-30,30), base_rand[1]+orientation[1]+np.random.randint(-30,30)]
        base = np.vstack((base, base_rand))
        addition = np.vstack((addition, add_rand))
    if n_points is None:
        h = computeH(addition, base, normalization=normalize)
    else:
        h = computeH(addition[:n_points], base[:n_points], normalization=normalize)
    image_dist, coord = warp(addition_image, h, add_origin=add_origin)
    res_image, origin = merge(base_image, image_dist, coord, origin=origin)
    return res_image, origin
    


def experiment(output_file, n_points = None, normalize=True, n_wrong = 0, noise=False):
    
    # Right addition
    base_image = cv2.imread(file_path + '/data/north_campus/middle.jpg') 
    addition_image = cv2.imread(file_path + '/data/north_campus/right-1.jpg')
    base, addition = get_points('north_campus', 'middle', 'right-1')
    
    res_image, origin = connect(base_image, addition_image, base, addition, noise=noise, n_wrong=n_wrong, n_points=n_points, normalize=normalize)

    # Left addition
    addition_image = cv2.imread(file_path + '/data/north_campus/left-1.jpg')
    base, addition = get_points('north_campus', 'middle', 'left-1')
    res_image, origin = connect(res_image, addition_image, base, addition, origin=origin, noise=noise, n_wrong=n_wrong, n_points=n_points, normalize=normalize)


    cv2.imwrite(file_path + "/" + output_file, res_image)
    
def panorama():
    
    
    print('Merging right side...')
    base_image = cv2.imread(file_path + '/data/north_campus/right-1.jpg')
    addition_image = cv2.imread(file_path + '/data/north_campus/right-2.jpg')
    base, addition = get_points('north_campus', 'right-1', 'right-2')
    
    right_connected, right_origin = connect(base_image, addition_image, base, addition)
    
    
    print('Merging right side to middle..')
    base_image = cv2.imread(file_path + '/data/north_campus/middle.jpg')
    base, addition = get_points('north_campus', 'middle', 'right-1')
    
    res_image, origin = connect(base_image, right_connected, base, addition, add_origin = right_origin)
    
    
    print('Merging left side...')
    base_image = cv2.imread(file_path + '/data/north_campus/left-1.jpg')
    addition_image = cv2.imread(file_path + '/data/north_campus/left-2.jpg')
    base, addition = get_points('north_campus', 'left-1', 'left-2')
    
    left_connected, left_origin = connect(base_image, addition_image, base, addition)
    
    
    print('Merging all together...')
    base, addition = get_points('north_campus', 'middle', 'left-1')
    
    res_image, origin = connect(res_image, left_connected, base, addition,origin=origin, add_origin = left_origin)
    
    cv2.imwrite(file_path + "/results/panorama.jpeg", res_image)
    print("Done!")


def main():
    print('Starting Experiment 1...')
    experiment('results/Experiment1.jpeg', n_points=5)
    print('Starting Experiment 2...')
    experiment('results/Experiment2.jpeg', n_points=12)
    print('Starting Experiment 3...')
    experiment('results/Experiment3.jpeg', n_wrong=3, normalize=False)
    print('Starting Experiment 4...')
    experiment('results/Experiment4.jpeg', n_wrong=3)
    print('Starting Experiment 5...')
    experiment('results/Experiment5.jpeg', n_wrong=5)
    print('Starting Experiment 6...')
    experiment('results/Experiment6.jpeg', noise=True)
    print('Starting Panorama...')
    panorama()
    
    

if __name__ == '__main__':
    main()