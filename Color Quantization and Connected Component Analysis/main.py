import numpy as np
import cv2
from matplotlib import pyplot as plt

import os
cwd = os.path.dirname(os.path.abspath(__file__))


class center_points:
    def __init__(self, K, dims, range=(0,1)):
        self.centers = np.random.random((K, dims), )* (range[1]-range[0]) +range[0]
        self.centers = self.centers.astype(int)

    def find_solution(self, data):
        i = 1
        while True:
            labeled_data = self.find_nearest(data)
            isChanged = False
            for K in range(self.centers.shape[0]):
                cluster = labeled_data[:,-1]
                cluster = np.where(cluster == K)
                if cluster[0].shape[0]<1:
                    self.centers[K] = data[np.random.randint(0, data.shape[0])]
                    continue
                cluster = labeled_data[cluster]
                new_point = cluster[:,:-1].mean(axis=0).astype(int)
                if not np.allclose(self.centers[K],new_point):
                    self.centers[K] = new_point
                    isChanged = True
            i += 1
            if not isChanged:
                break
        return labeled_data[:,-1]

    def find_nearest(self, data):
        labeled_data = np.empty((0,data.shape[1]+1), dtype=int)
        inertia = 0
        for i in range(data.shape[0]):
            distances = [np.square(data[i]-self.centers[K]).sum() for K in range(self.centers.shape[0])]
            inertia += np.min(distances)/(65025*len(data))
            labeled_row = np.hstack((data[i], [np.argmin(distances)]))
            labeled_data = np.vstack((labeled_data, labeled_row))
        self.inertia = inertia
        return labeled_data
    
    def convert_to_image(self, shape, clusters):
        colors = self.centers[clusters]
        return np.reshape(colors, shape)


image = cv2.imread(cwd + '/cq1.jpeg')
image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4))

K=8
centers = center_points(K, 3, (0,255))
classes = centers.find_solution(image.reshape(-1,3))
quintized_image = centers.convert_to_image(image.shape, classes)

cv2.imshow('frame', quintized_image.astype(np.uint8))
cv2.waitKey()