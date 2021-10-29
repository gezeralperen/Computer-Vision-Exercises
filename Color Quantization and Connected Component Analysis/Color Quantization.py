import numpy as np
import cv2
from sklearn.cluster import KMeans
from picker import color_picker

import os
file_path = os.path.dirname(os.path.realpath(__file__))

############################### HYPERPARAMETERS ##############################

# Uses SKLearn insted of my method. Used for verification.
USE_SKLEARN = False

# Use picker to determine initial cluster centers.
PICKER = False

# Resize images to fit in a rectangle.
IMAGE_SIZE = (400, 200)

# Images to be processed.
IMAGE_QUERY = [file_path + '/cq1.jpeg',
               file_path + '/cq2.jpeg',
               file_path + '/cq3.jpeg']

# Output file
OUTPUT_IMAGE = file_path + '/results.jpeg'

###############################################################################


def convert_to_image(shape, clusters, centers):
    colors = centers[clusters]
    img = np.reshape(colors, shape)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_LAB2BGR)

def find_nearest(cluster_centers, data):
    label = np.empty((data.shape[0],), dtype=int)
    for i in range(data.shape[0]):
        distances = [np.square(data[i]-cluster_centers[K]).sum() for K in range(cluster_centers.shape[0])]
        label[i] = np.argmin(distances)
    return label

class custom_kmeans:
    def __init__(self, K, dims, range=(0,1)):
        self.cluster_centers_ = np.random.random((K, dims), )* (range[1]-range[0]) +range[0]

    def fit(self, data):
        while True:
            label = find_nearest(self.cluster_centers_, data)
            isChanged = False
            for K in range(self.cluster_centers_.shape[0]):
                cluster = np.where(label == K)
                if cluster[0].shape[0]<1:
                    self.cluster_centers_[K] = data[np.random.randint(0, data.shape[0])]
                    continue
                cluster = data[cluster]
                new_point = cluster.mean(axis=0)
                if not np.allclose(self.cluster_centers_[K],new_point, atol=10):
                    self.cluster_centers_[K] = new_point
                    isChanged = True
            if not isChanged:
                break
            self.labels_ = label
        self.cluster_centers_ = self.cluster_centers_.astype(np.uint8)
        return self



if __name__ == '__main__':

    width = IMAGE_SIZE[0]
    heigth = IMAGE_SIZE[1]

    results = np.empty((0,width*5,3), dtype=np.uint8)

    if USE_SKLEARN:
        for file in IMAGE_QUERY:
            print(f'Processing {file}')
            image = cv2.imread(file)
            image = cv2.resize(image, (width, heigth))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            row = np.empty((heigth,0,3), dtype=np.uint8)
            for K in [2, 4, 8, 16, 32]:
                print(f'K={K}')
                cluster = KMeans(K)
                if PICKER:
                    cluster.cluster_centers_ = color_picker(image, K)
                cluster.fit(image.reshape(-1,3))
                custom_quantized_image = convert_to_image(image.shape, cluster.labels_, cluster.cluster_centers_)
                row = np.hstack((row, custom_quantized_image))
            results = np.vstack((results, row))
    else:
        for file in IMAGE_QUERY:
            print(f'Processing {file}')
            image = cv2.imread(file)
            image = cv2.resize(image, (width, heigth))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            row = np.empty((heigth,0,3), dtype=np.uint8)
            for K in [2, 4, 8, 16, 32]:
                print(f'K={K}')
                cluster = custom_kmeans(K, 3, (0,255))
                if PICKER:
                    cluster.cluster_centers_ = color_picker(image, K)
                cluster.fit(image.reshape(-1,3))
                custom_quantized_image = convert_to_image(image.shape, cluster.labels_, cluster.cluster_centers_)
                row = np.hstack((row, custom_quantized_image))
            results = np.vstack((results, row))


    cv2.imwrite(OUTPUT_IMAGE, results)