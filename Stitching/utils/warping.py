import numpy as np

def warp(image, H):
    
    def get_points(m, x, y):
        return (m[0,0]*x+m[0,1]*y+m[0,2])/(m[2,0]*x+m[2,1]*y+m[2,2]), (m[1,0]*x+m[1,1]*y+m[1,2])/(m[2,0]*x+m[2,1]*y+m[2,2])
    
    def try_or_zero(x,y,im):
        try:
            return im[y,x]
        except IndexError:
            return np.array([0,0,0])
    
    ymax = image.shape[0]
    xmax = image.shape[1]
    
    # corners = np.array([
    #     np.matmul(H,np.array([xmax,ymax,1])),
    #     np.matmul(H,np.array([0,ymax,1])),
    #     np.matmul(H,np.array([xmax,0,1])),
    #     np.matmul(H,np.array([0,0,1])),
    # ], dtype=int)
    
    corners = np.array([
        get_points(H, xmax, ymax),
        get_points(H, 0, ymax),
        get_points(H, xmax, 0),
        get_points(H, 0, 0),
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
            if p_src[0] < xmax and  p_src[0] > 0 and p_src[1] < ymax and  p_src[1] > 0:
                
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
                canvas[y-ymin_out,x-xmin_out] = image[nearest[1]-1, nearest[0]-1]
                #########################
                
    return canvas, (xmin_out, xmax_out, ymin_out, ymax_out)
            
    
    