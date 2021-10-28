import numpy as np
import cv2
from sklearn.cluster import KMeans
from picker import color_picker

import os
cwd = os.path.dirname(os.path.realpath(__file__))

############### HYPERPARAMETERS ###################

USE_SKLEARN = True
PICKER = True
IMAGE_SIZE = (400, 200)
IMAGE_QUERY = ['cq1.jpeg', 'cq2.jpeg', 'cq3.jpeg']

###################################################


def convert_to_image(shape, clusters, centers):
    colors = centers[clusters]
    img = np.reshape(colors, shape)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_LAB2BGR)

def find_nearest(cluster_centers, data, len, dim, Ks):
    labeled_data = np.empty((0,dim+1), dtype=int)
    for i in range(len):
        distances = [np.square(data[i]-cluster_centers[K]).sum() for K in range(Ks)]
        labeled_row = np.hstack((data[i], [np.argmin(distances)]))
        labeled_data = np.vstack((labeled_data, labeled_row))
    return labeled_data

class custom_kmeans:
    def __init__(self, K, dims, range=(0,1)):
        self.cluster_centers_ = np.random.random((K, dims), )* (range[1]-range[0]) +range[0]
        self.cluster_centers_ = self.cluster_centers_.astype(np.uint8)
        self.labels_ = []

    def fit(self, data):
        i = 1
        while True:
            labeled_data = find_nearest(self.cluster_centers_, data, data.shape[0], data.shape[1], self.cluster_centers_.shape[0])
            isChanged = False
            for K in range(self.cluster_centers_.shape[0]):
                cluster = labeled_data[:,-1]
                cluster = np.where(cluster == K)
                if cluster[0].shape[0]<1:
                    self.cluster_centers_[K] = data[np.random.randint(0, data.shape[0])]
                    continue
                cluster = labeled_data[cluster]
                new_point = cluster[:,:-1].mean(axis=0).astype(np.uint8)
                if not np.allclose(self.cluster_centers_[K],new_point, atol=10):
                    self.cluster_centers_[K] = new_point
                    isChanged = True
            i += 1
            if not isChanged:
                break
        self.labels_ = labeled_data[:,-1]
        return self



if __name__ == '__main__':

    width = IMAGE_SIZE[0]
    heigth = IMAGE_SIZE[1]

    results = np.empty((0,width*5,3), dtype=np.uint8)

    if USE_SKLEARN:
        for file in IMAGE_QUERY:
            print(f'Processing {file}')
            image = cv2.imread(cwd + '/' + file)
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
            image = cv2.imread(cwd + '/' + file)
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


    cv2.imwrite('results.jpeg', results)