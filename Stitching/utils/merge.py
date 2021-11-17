import numpy as np

def merge(img1, img2, img2_coordinates):
    
    ymax = img1.shape[0]
    xmax = img1.shape[1]
    ymin = 0
    xmin = 0
    
    start = 0
    end = 0
    
    if ymax < img2_coordinates[3]:
        ymax = img2_coordinates[3]
    if xmax < img2_coordinates[1]:
        xmax = img2_coordinates[1]
        start = img2.shape[1]
        end = img2_coordinates[0]
    if ymin > img2_coordinates[2]:
        ymin = img2_coordinates[2]
    if xmin > img2_coordinates[0]:
        xmin = img2_coordinates[0]
        start = 0
        end = img2_coordinates[1]
    
    canvas1 = np.zeros((ymax-ymin, xmax-xmin,3), dtype=img1.dtype)
    canvas2 = np.zeros((ymax-ymin, xmax-xmin,3), dtype=img1.dtype)
    
    canvas1[-ymin:img1.shape[0]-ymin, -xmin:img1.shape[1]-xmin] = img1
    canvas2[img2_coordinates[2]-ymin:img2_coordinates[3]-ymin, img2_coordinates[0]-xmin:img2_coordinates[1]-xmin] = img2
    mask = (canvas2 > 0).astype(float)
    
    gradient = np.tile(np.linspace(1,0,end-start),(img1.shape[0],1))
    gradient = np.stack((gradient,)*3, axis=-1)
    
    mask[-ymin:img1.shape[0]-ymin:,start-xmin:end-xmin] *= gradient
    canvas = (canvas1*(1-mask)+canvas2*mask).astype(np.uint8)
    return canvas