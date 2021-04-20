import cv2
import math 

img=cv2.imread('/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/test/seq1/riskMaps/041007_017_risk.jpg')
print(img.shape)

def getDistance(img,pt):
    """Finds Normalised Distance between a given point and center of frame.

    Args:
        img (cv2 Frame): image where the point resides
        pt (array): point in the form [x,y]
    """
    dim=img.shape
    furthestDistance=math.hypot(dim[0]/2,dim[1]/2)
    distance=math.dist(pt,[dim[0]/2,dim[1]/2])
    return distance/furthestDistance

print(getDistance(img,[384,575]))