import numpy as np

def merge(img1, img2, img2_coordinates, origin=(0,0)):
    
    ymax = img1.shape[0]
    xmax = img1.shape[1]
    ymin = 0
    xmin = 0
    
    start = 0
    end = 0
    
    # Check for bottom overflow
    if ymax < img2_coordinates[3]+origin[0]:
        ymax = img2_coordinates[3]+origin[0]
    
    # Check for right overflow
    if xmax < img2_coordinates[1]+origin[1]:
        xmax = img2_coordinates[1]+origin[1]
        start = img1.shape[1]+origin[1]
        end = img2_coordinates[0]+origin[1]
    
    # Check for top overflow
    if ymin > img2_coordinates[2]+origin[0]:
        ymin = img2_coordinates[2]+origin[0]
        
    # Check for left overflow
    if xmin > img2_coordinates[0]+origin[1]:
        xmin = img2_coordinates[0]+origin[1]
        start = origin[1]
        end = img2_coordinates[1]+origin[1]
    
    canvas1 = np.zeros((ymax-ymin, xmax-xmin,3), dtype=img1.dtype)
    canvas2 = np.zeros((ymax-ymin, xmax-xmin,3), dtype=img1.dtype)
    
    canvas1[-ymin:img1.shape[0]-ymin, -xmin:img1.shape[1]-xmin] = img1
    canvas2[img2_coordinates[2]-ymin+origin[0]:img2_coordinates[3]-ymin+origin[0], img2_coordinates[0]-xmin+origin[1]:img2_coordinates[1]-xmin+origin[1]] = img2
    mask = (canvas2 > 0).astype(float)
    
    length = end-start
    if length > 0:
        gradient = np.tile(np.linspace(1,0,length),(img1.shape[0],1))
        gradient = np.stack((gradient,)*3, axis=-1)
        mask[-ymin:img1.shape[0]-ymin:,start-xmin:end-xmin] *= gradient
    else:
        gradient = np.tile(np.linspace(0,1,-length),(img1.shape[0],1))
        gradient = np.stack((gradient,)*3, axis=-1)
        mask[-ymin:img1.shape[0]-ymin:,end-img2_coordinates[0]:start-img2_coordinates[0]] *= gradient
    
    canvas = (canvas1*(1-mask)+canvas2*mask).astype(np.uint8)
    return canvas, (-ymin+origin[0], -xmin+origin[1])