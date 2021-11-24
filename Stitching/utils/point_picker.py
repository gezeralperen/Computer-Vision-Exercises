import cv2
import numpy as np



point_matrix = np.zeros((12,4),np.int)
 
counter = 0
match = False

colors = [(255,0,0),
          (0,255,0),
          (0,0,255),
          (255,255,0),
          (0,255,255),
          (255,0,255),
          (127,255,0),
          (0,255,127),
          (255,127,0),
          (0,127,255),
          (255,0,127),
          (127,0,255)]

def mousePoints(event,x,y,flags,params):
    global counter
    global match
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        if not match:
            point_matrix[counter, :2] = x,y
            match = not match
        else:
            point_matrix[counter, 2:] = x,y
            counter = counter + 1
            match = not match
    
def point_matcher(image1, image2):
    img = np.hstack((image1, image2))
    global point_matrix
    global colors
    point_matrix = np.zeros((12,4),np.int)
    while True:
        for x in range(counter+1):
            if x != 0:
                cv2.circle(img,(point_matrix[x-1][0],point_matrix[x-1][1]),3,colors[x-1],cv2.FILLED)
                cv2.circle(img,(point_matrix[x-1][2],point_matrix[x-1][3]),3,colors[x-1],cv2.FILLED)
            if match:
                cv2.circle(img,(point_matrix[x][0],point_matrix[x][1]),3,colors[x],cv2.FILLED)
            
        if counter == 12:
            point_matrix[:,2] -= image1.shape[1]
            str = "[" + ",\n".join([f"(({x[0]}, {x[1]}), ({x[2]},{x[3]}))" for x in point_matrix]) + "]" 
            print(str)
            break
        
        cv2.imshow("Original Image ", img)
        cv2.setMouseCallback("Original Image ", mousePoints)
        cv2.waitKey(1)
    return point_matrix