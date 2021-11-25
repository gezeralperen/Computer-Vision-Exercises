import numpy as np
from numba import jit

def warp(image, H, add_origin=(0,0)):
    
    @jit(nopython=True)
    def get_points(m, x, y):
        return (m[0,0]*x+m[0,1]*y+m[0,2])/(m[2,0]*x+m[2,1]*y+m[2,2]), (m[1,0]*x+m[1,1]*y+m[1,2])/(m[2,0]*x+m[2,1]*y+m[2,2])
    
    def try_or_zero(x,y,im):
        try:
            return im[y,x]
        except IndexError:
            return np.array([0,0,0])
    
    ymax = image.shape[0]-add_origin[0]
    xmax = image.shape[1]-add_origin[1]
    ymin = -add_origin[0]
    xmin = -add_origin[1]
    
    corners = np.array([
        get_points(H, xmax, ymax),
        get_points(H, xmin, ymax),
        get_points(H, xmax, ymin),
        get_points(H, xmin, ymin),
    ], dtype=int)
    
    ymax_out = np.max(corners[:,1])
    ymin_out = np.min(corners[:,1])
    xmax_out = np.max(corners[:,0])
    xmin_out = np.min(corners[:,0])
    
    
    canvas_height = ymax_out - ymin_out
    canvas_width = xmax_out - xmin_out
    
    canvas = np.zeros((canvas_height, canvas_width,3), dtype=np.uint8)
    
    hinv = np.linalg.inv(H)
    
    for y in range(ymin_out, ymax_out):
        for x in range(xmin_out, xmax_out):
            p_src = get_points(hinv, x, y)
            if p_src[0] < xmax and  p_src[0] > xmin and p_src[1] < ymax and  p_src[1] > ymin:
                
                ## LINEAR INTERPOLATION ##
                # x_left = np.floor(p_src[0]).astype(int) 
                # x_right = np.ceil(p_src[0]).astype(int) 
                # y_top = np.floor(p_src[1]).astype(int) 
                # y_bottom = np.ceil(p_src[1]).astype(int) 
                # a1 = try_or_zero(x_left,y_top, image)
                # a2 = try_or_zero(x_right,y_top, image)
                # a3 = try_or_zero(x_left,y_bottom, image)
                # a4 = try_or_zero(x_right,y_bottom, image)
                # x_weight = p_src[0]-x_left
                # y_weight = p_src[1]-y_top
                # middle = [x_weight*a1 + (1-x_weight)*a2,
                #           x_weight*a3 + (1-x_weight)*a4,]
                # center = y_weight*middle[0] + (1-y_weight)*middle[1]
                # canvas[y-ymin_out,x-xmin_out] = center.astype(np.uint8)
                ###########################

                ### NEAREST NEIGHBOR ####
                nearest = np.round(p_src).astype(int)
                canvas[y-ymin_out,x-xmin_out] = image[nearest[1]-1-ymin, nearest[0]-1-xmin]
                #########################
                
    return canvas, (xmin_out, xmax_out, ymin_out, ymax_out)
            
    
    