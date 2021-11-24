import numpy as np

# image_set -> pair -> ((x1,y1),(x2,y2)) in pixels

MATCHED_POINTS = {
    'north_campus':
        {
            'middle&left-1' :
                [
                    ((5, 254), (570,267)),
                    ((711, 196), (1274,170)),
                    ((97, 29), (641,61)),
                    ((168, 702), (750,681)),
                    ((47, 224), (604,240)),
                    ((692, 62), (1246,26)),
                    ((186, 34), (718,56)),
                    ((57, 430), (639,423)),
                    ((347, 154), (903,155)),
                    ((564, 415), (1131,418)),
                    ((348, 84), (868,88)),
                    ((651, 262), (1202,248)),
                ],
                
            'middle&right-1' :
                [
                    ((686, 612), (1186,639)),
                    ((149, 258), (630,271)),
                    ((370, 195), (817,205)),
                    ((694, 273), (1166,264)),
                    ((665, 335), (1133,333)),
                    ((369, 67), (813,73)),
                    ((231, 107), (688,125)),
                    ((512, 402), (979,409)),
                    ((238, 414), (714,417)),
                    ((687, 118), (1159,96)),
                    ((501, 331), (947,332)),
                    ((253, 684), (743,675))
                ]
        }
}

def get_points(set, img1, img2):
    pair = '&'.join([img1,img2])
    points = MATCHED_POINTS[set][pair]
    length = len(points)
    im1Points = np.zeros((length,2), dtype=int)
    im2Points = np.zeros((length,2), dtype=int)
    for i in range(length):
        im1Points[i,0] = points[i][0][0]
        im1Points[i,1] = points[i][0][1]
        im2Points[i,0] = points[i][1][0]
        im2Points[i,1] = points[i][1][1]
    return im1Points, im2Points