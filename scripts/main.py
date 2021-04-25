HEADLESS=False
SAVE_TO_FILE=False
SIMULATE=True

from cv2 import cv2 as cv
from enum import Enum     # for enum34, or the stdlib version
import numpy
import sys
import glob
import cProfile
from matplotlib import pyplot as plt
from numpy.ma.core import mask_rowcols
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial import distance
import math
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize
from pathlib import Path
if not SIMULATE:
    from seg_util import SegmentationEngine
    from yolo_util import ObjectDetector

class Label(Enum):
    unlabeled=0
    pavedArea=1
    dirt=2
    grass=3
    gravel=4
    water=5
    rocks=6
    pool=7
    vegetation=8
    roof=9
    wall=10
    window=11
    door=12
    fence=13
    fencePole=14
    person=15
    dog=16
    car=17
    bicycle=18
    tree=19
    baldTree=20
    arMarker=21
    obstacle=22
    conflicting=23
class RiskLevel(Enum):
    VERY_HIGH=100
    HIGH=20
    MEDIUM=10
    LOW=5
    ZERO=0

risk_table = dict([
    (Label.unlabeled, RiskLevel.ZERO),
    (Label.pavedArea, RiskLevel.ZERO),
    (Label.dirt, RiskLevel.LOW),
    (Label.grass, RiskLevel.ZERO),
    (Label.gravel, RiskLevel.LOW),
    (Label.water, RiskLevel.HIGH),
    (Label.rocks, RiskLevel.MEDIUM),
    (Label.pool, RiskLevel.HIGH),
    (Label.vegetation, RiskLevel.MEDIUM),
    (Label.roof, RiskLevel.HIGH),
    (Label.wall, RiskLevel.HIGH),
    (Label.window, RiskLevel.HIGH),
    (Label.door, RiskLevel.HIGH),
    (Label.fence, RiskLevel.HIGH),
    (Label.fencePole, RiskLevel.HIGH),
    (Label.person, RiskLevel.VERY_HIGH),
    (Label.dog, RiskLevel.HIGH),
    (Label.car, RiskLevel.VERY_HIGH),
    (Label.bicycle, RiskLevel.VERY_HIGH),
    (Label.tree, RiskLevel.HIGH),
    (Label.baldTree, RiskLevel.HIGH),
    (Label.arMarker, RiskLevel.ZERO),
    (Label.obstacle, RiskLevel.HIGH),
    (Label.conflicting, RiskLevel.MEDIUM)
])

def getDistance(img,pt):
    """Finds Normalised Distance between a given point and center of frame.

    Args:
        img (cv2 Frame): image where the point resides
        pt (array): point in the form [x,y]
    """
    dim=img.shape
    furthestDistance=math.hypot(dim[0]/2,dim[1]/2)
    dist=distance.euclidean(pt,[dim[0]/2,dim[1]/2])
    return 1-abs(dist/furthestDistance)

def circles_intersect(x1,x2,y1,y2,r1,r2):
    """Checks if two circle intersect
    
    Args:
        x: x coordinate of the circle center
        y: y coordinate of the circle center
        r: radius of the circle     

    Returns:
        -3: C2 is in C1
        -2: C1 is in C2
        -1: circles intersect
         0: circles do not intersect
    """

    d = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    if d < r1-r2:
        #'print("C2  is in C1")
        return -3
    elif d < r2-r1:
        return -2
        #print("C1  is in C2")
    elif d > r1+r2:
        return 0
        #print("Circumference of C1  and C2  intersect")
    else:
        return -1
        #print("C1 and C2  do not overlap")


def meets_min_safety_requirement(zone_proposed, obstacles_list):
    """Checks if a proposed safety zone is breaking the min. safe distance of all the high-risk obstacles detected in an image
    
    Args:
        zone_proposed (tuple): coordinates of the proposed zone in the x,y,r_landing format
        obstacles_list (list of tuples): list of coordinates of the high-risk obstacles in the x,y,r_min_safe_dist format 
    
    Returns:
        (bool): True if it meets safety req., False otherwise.
    """
    posLz=zone_proposed.get("position")
    radLz=zone_proposed.get("radius")
    for obstacle in obstacles_list:
        touch=circles_intersect(posLz[0],obstacle[0],posLz[1],obstacle[1],radLz,obstacle[2])
        if touch<0:
            return False
    return True

def get_landing_zones_proposals(high_risk_obstacles, stride, r_landing, image):
    """Returns list of lzs proposal based that meet the min safe distance of all the high risk obstacles

    Args:
        high_risk_obstacles (tuple): tuple in the following format (x,y,min_safe_dist)
        stride (int): how much stride between the proposed regions.
        r_landing (int): min safe landing radius - size of lz in pixels
        image (OpenCv img): image to find lzs on

    Returns:
        list of tuples: list of proposed zones. Each zone is expressed in a tuple (x,y,r_landing)
    """
    zones_proposed=[]

    for y in range(r_landing,image.shape[0]-r_landing, stride):
        for x in range(r_landing,image.shape[1]-r_landing, stride):
            lzProposed = {
                'confidence':math.nan,
                'radius':r_landing,
                'position':[x,y]

            }
            if meets_min_safety_requirement(lzProposed, high_risk_obstacles):
                zones_proposed.append(lzProposed)
    return zones_proposed

def draw_lzs_obs(list_lzs, list_obs,img,thickness=3):
    """Adds annotations on image for obstacles and landing zone proposals to an image for visualisation. 

    Args:
        list_lzs (list of tuples): list of lzs in the standard lz format - (x,y,r_landing)
        list_obs (list of tuples): list of high risk objtacle in format - (x,y,min_safe_dist)
        img (opencv image): image to add annotation on

    Returns:
        image: OpenCv image with added annotation
    """
    for obstacle in list_obs:
        cv.circle(img,(obstacle[0], obstacle[1]), obstacle[2], (0,0,255),thickness=thickness)
    for lz in list_lzs:
        posLz=lz.get("position")
        radLz=lz.get("radius")
        cv.circle(img,(posLz[0], posLz[1]), radLz, (0,255,0),thickness=thickness)
    return img

def get_risk(image_segment):
    """Obtain a risk factor based on the section of an image. 
    
    Args:
        image_segment (Mat): section of an image to be assessed. The assessment is based on the risk level defined in the risk_table.

    Returns:
        float: risk level. The higher, the riskier it is to land.
    """
    num_pix=image_segment.shape[0]*image_segment.shape[1]
    risk_level=0
    for label in Label:
        label_pix=numpy.count_nonzero((image_segment ==label.value).all(axis = 2))
        ratio_label=label_pix/num_pix
        risk_level+=ratio_label*risk_table[label].value
    return risk_level

def get_risk_map_slow(seg_img, windowsize, gaussian_sigma=7):
    risk_r=int(seg_img.shape[0]/windowsize)
    risk_c=int(seg_img.shape[1]/windowsize)
    risk_array=numpy.zeros(shape=(risk_r,risk_c))
    for r in range(0,seg_img.shape[0] - windowsize, windowsize):
        for c in range(0,seg_img.shape[1] - windowsize, windowsize):
            window=seg_img[r:r+windowsize,c:c+windowsize]
            risk_array[int(r/windowsize)][int(c/windowsize)]=get_risk(window)

    risk_array = gaussian_filter(risk_array, sigma=gaussian_sigma)
    risk_array = (risk_array / risk_array.max())*255
    risk_array=np.uint8(risk_array)
    img = cv.resize(risk_array, (seg_img.shape[1],seg_img.shape[0]), interpolation=cv.INTER_CUBIC)
    return img

def get_risk_map(seg_img, gaussian_sigma=25):
    if SIMULATE:
        image,_,_ = cv.split(seg_img)
    else:
        image=seg_img
    risk_array = image.astype('float32')
    for label in Label:
        np.where(risk_array==label.value, risk_table[label].value, risk_array)
    risk_array = gaussian_filter(risk_array, sigma=gaussian_sigma)
    risk_array = (risk_array / risk_array.max())*255
    risk_array=np.uint8(risk_array)
    return risk_array

def _risk_map_eval_basic(img,areaLz):
    """Evaluate normalised risk in a landing zone

    Args:
        img(risk map)): risk map containin pixels between 0 and 255

    Returns:
        float: float between 0.0 and 1.0
    """
    maxRisk=areaLz*255
    totalRisk=np.sum(img)
    return 1-(totalRisk/maxRisk)

def rank_lzs(lzsProposals,riskMap,weightDist=3,weightRisk=15):
    for lz in lzsProposals:
        lzRad=lz.get("radius")
        lzPos=lz.get("position")
        mask = np.zeros_like(riskMap)
        mask = cv.circle(mask, (lzPos[0],lzPos[1]), lzRad, (255,255,255), -1)
        areaLz=math.pi* lzRad*lzRad
        crop = cv.bitwise_and(riskMap, mask)
        riskFactor=_risk_map_eval_basic(crop,areaLz)
        distanceFactor=getDistance(riskMap,[lzPos[0],lzPos[1]])
        lz["confidence"]=(weightRisk*riskFactor+weightDist*distanceFactor)/(weightRisk+weightDist)
#        lz["confidence"]=(weightRisk*riskFactor+weightDist*distanceFactor)/(weightRisk+weightDist)
 #       lz["confidence"]=(weightRisk*riskFactor+weightDist*distanceFactor)/(weightRisk+weightDist)

    lzsSorted = sorted(lzsProposals, key=lambda k: k['confidence']) 
    return lzsSorted



def main():
    rgbImgs=glob.glob("/content/Safe-UAV-Landing/data/test/seq1/images/*.jpg")    
    rgbImgs=glob.glob("/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/test/seq1/images/*.jpg")    
    if not SIMULATE:
        objectDetector=ObjectDetector('/content/Safe-UAV-Landing/models/yolo-v3/visdrone.names',
                                    '/content/Safe-UAV-Landing/models/yolo-v3/yolov3_leaky.weights',
                                    '/content/Safe-UAV-Landing/models/yolo-v3/yolov3_leaky.cfg')
        segEngine = SegmentationEngine('/content/Safe-UAV-Landing/models/seg/Unet-Mobilenet.pt')
    else:
        segImgs=glob.glob("/home/kubitz/Documents/fyp/Safe-UAV-Landing/data/test/seq1/masks/*.jpg")
        segImgs.sort()
        rgbImgs.sort()
        seq_obstacles=[
            [(640,330,100)], 
            [(643,346,100)], 
            [(638,365,100)],  
            [(643,387,100)], 
            [(645,398,100)],
            [(642,414,100)],
            [(640,437,100)],
            [(638,468,100)],
            [(643,488,100)],
            [(640,488,100)]
        ]
    
    i=0
    for i in range(len(rgbImgs)):
        fileName=Path(rgbImgs[i]).stem
        img = cv.imread(rgbImgs[i])
        if not SIMULATE:
            height, width = img.shape[:2]
            img_pil = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_pil)
            segImg=segEngine.inferImage(img_pil)
            segImg = numpy.array(segImg) 
            _, objs=objectDetector.infer_image(height, width, img)
            obstacles=[]
            for obstacle in objs:
                print(obstacle)
                posOb=obstacle.get('box')
                minDist=100
                w, h = posOb[2], posOb[3]
                obstacles.append([int(posOb[0]+w/2),int(posOb[1]+h/2),minDist])
        else:
            segImg = cv.imread(segImgs[i])
            obstacles = seq_obstacles[i]

        image=img.copy()
        lzs=get_landing_zones_proposals(obstacles,75, 120,image)
        risk_map=get_risk_map(segImg)
        lzs_ranked=rank_lzs(lzs,risk_map)
        image=draw_lzs_obs(lzs_ranked[-5:],obstacles,img)

        if SAVE_TO_FILE:
            cv.imwrite('/content/Safe-UAV-Landing/data/results/riskMaps/'+fileName+'_risk.jpg',cv.applyColorMap(risk_map, cv.COLORMAP_JET))
            cv.imwrite('/content/Safe-UAV-Landing/data/results/landingZones/'+fileName+'_lzs.jpg',img)
        
        if not HEADLESS:
            cv.imshow("best landing zones",cv.applyColorMap(risk_map, cv.COLORMAP_JET))
            cv.waitKey(0)
            cv.imshow("best landing zones",img)  
            cv.waitKey(0)
            cv.destroyAllWindows()
#cProfile.run('main()')


main()