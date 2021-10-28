from PIL import Image
from matplotlib import pyplot as plt      
import numpy as np
import cv2

def color_picker(im, K):
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_LAB2RGB))
    points = plt.ginput(K,show_clicks=True)
    plt.close()
    colors = []
    for point in points:
        colors.append(im[int(point[1]),int(point[0])])
    return np.array(colors,dtype=np.uint8)